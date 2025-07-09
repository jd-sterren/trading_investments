import webbrowser, os, time, shutil, sys
from rauth import OAuth1Service
from datetime import datetime, timedelta
import time, os
import requests
import pandas as pd
import glob, json
import joblib
import numpy as np
from inc.credential_manager import inject_decrypted_env
from inc.indicators import apply_all_indicators

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os
import pickle
from tensorflow import keras

pd.set_option('display.precision', 8)
log_path = "inc/logs/coinbase_data_save_log.txt"
headers = {'Content-Type': 'application/json'}

# === E*TRADE API FUNCTIONS === #
def oauth():
    """Handles OAuth 1.0a authentication flow for E*TRADE API access."""
    CONSUMER_KEY = os.getenv("PROD_API")
    CONSUMER_SECRET = os.getenv("PROD_SEC")
    
    etrade = OAuth1Service(
        name="etrade",
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
        request_token_url="https://api.etrade.com/oauth/request_token",
        access_token_url="https://api.etrade.com/oauth/access_token",
        authorize_url="https://us.etrade.com/e/t/etws/authorize?key={}&token={}",
        base_url="https://api.etrade.com"
    )

    # Step 1: Get OAuth 1 request token and secret
    request_token, request_token_secret = etrade.get_request_token(
        params={"oauth_callback": "oob", "format": "json"}
    )

    # Step 2: Authenticate
    authorize_url = etrade.authorize_url.format(etrade.consumer_key, request_token)
    webbrowser.open(authorize_url)
    text_code = input("Please accept agreement and enter verification code from browser: ").strip()

    # Step 3: Exchange authorized token for session
    session = etrade.get_auth_session(
        request_token,
        request_token_secret,
        params={"oauth_verifier": text_code}
    )
    
    return session, request_token, request_token_secret

def fetch_stock_info(session, symbols, current_time=None, sleep_between_calls=True, sleep_time=0.5, 
                     max_retries=1, retry_wait=2, log_file="inc/logs/stock_fetch_log.txt", max_log_size_mb=5):
    """Fetch, parse, and format stock data for one or multiple symbols.
    
    Example (one Sybmbol):
        stock_info = fn.fetch_stock_info(session, "AAPL")

        print(stock_info["AAPL"]["formatted_data"])
        print(stock_info["AAPL"]["msg"])
    
    Example (multiple Symbols):
        stock_info = fn.fetch_stock_info(session, ["AAPL", "GOOG"])

        print(stock_info["AAPL"]["formatted_data"])
        print(stock_info["GOOG"]["msg"])
    
    Example (With Fake Time):
        fake_time = datetime.strptime("2024-05-01 15:00:00", "%Y-%m-%d %H:%M:%S")
        base_data, last_trade, volume, msg = fetch_stock_info(session, "AAPL", current_time=fake_time)
    
    Exampe (With Real Time):
        base_data, last_trade, volume, msg = fetch_stock_info(session, ["AAPL", "GOOG"])
    """

    if current_time is None:
        current_time = datetime.now()

    # If a single symbol is passed as a string, wrap it in a list
    if isinstance(symbols, str):
        symbols = [symbols]

    # Ensure logs folder exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Rotate log if too big
    if os.path.exists(log_file) and (os.path.getsize(log_file) > max_log_size_mb * 1024 * 1024):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rotated_log = log_file.replace(".txt", f"_{timestamp}.txt")
        shutil.move(log_file, rotated_log)

    results = {}

    # Initialize or append to the log file
    with open(log_file, "a") as log:
        log.write(f"\n=== Fetch Session Started: {datetime.now()} ===\n")

        for idx, symbol in enumerate(symbols):
            url = f"https://api.etrade.com/v1/market/quote/{symbol}.json"

            attempt = 0
            while attempt <= max_retries:
                try:
                    response = session.get(url, params={"detailFlag": "All"})
                    response.encoding = 'utf-8'
                    stock_data = response.json()

                    if 'QuoteResponse' not in stock_data or 'QuoteData' not in stock_data['QuoteResponse']:
                        log.write(f"[{datetime.now()}] Warning: No data found for {symbol}. Skipping.\n")
                        break

                    if not stock_data['QuoteResponse']['QuoteData']:
                        log.write(f"[{datetime.now()}] Warning: Empty data returned for {symbol}. Skipping.\n")
                        break

                    quote_data = stock_data['QuoteResponse']['QuoteData'][0]

                    # Determine if regular trading hours or extended
                    time_str = current_time.strftime("%H:%M:%S")

                    if "09:30:00" <= time_str <= "16:00:00":
                        base_data = quote_data['All']
                        last_trade = base_data['lastTrade']
                        volume = base_data['totalVolume']
                        msg = "We're in business hours."
                    else:
                        base_data = quote_data['All']['ExtendedHourQuoteDetail']
                        last_trade = base_data['lastPrice']
                        volume = base_data['volume']
                        msg = "We're in extended hours."

                    # Format the output dictionary for this symbol
                    formatted_data = {
                        'dt_lastTrade': pd.to_datetime(base_data['timeOfLastTrade'], unit='s') - pd.Timedelta(hours=4),
                        'symbol': symbol,
                        'Close': last_trade,
                        'bid': base_data['bid'],
                        'bid size': base_data['bidSize'],
                        'ask': base_data['ask'],
                        'ask size': base_data['askSize'],
                        'bid_ask_spread': abs(base_data['bid'] - base_data['ask']),
                        'Volume': volume,
                        'market_status': msg
                    }

                    results[symbol] = {
                        "base_data": base_data,
                        "last_trade": last_trade,
                        "volume": volume,
                        "msg": msg,
                        "formatted_data": formatted_data
                    }

                    log.write(f"[{datetime.now()}] Success: Retrieved data for {symbol}.\n")
                    break  # Successful fetch

                except Exception as e:
                    attempt += 1
                    if attempt > max_retries:
                        log.write(f"[{datetime.now()}] ERROR: Failed to fetch {symbol} after {max_retries} retries. Reason: {e}\n")
                        break
                    else:
                        log.write(f"[{datetime.now()}] Retry {attempt}/{max_retries} for {symbol} after error: {e}. Waiting {retry_wait} seconds...\n")
                        time.sleep(retry_wait)

            # Sleep between API calls if needed
            if sleep_between_calls and idx < len(symbols) - 1:
                time.sleep(sleep_time)

        log.write(f"=== Fetch Session Ended: {datetime.now()} ===\n")

    return results

def data_collector(symbols, save_folder="data", interval_seconds=30):
    """Collects stock data every interval and saves to CSV."""

    inject_decrypted_env(environment="prod")
    os.makedirs(save_folder, exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(save_folder, f"stock_data_{today_str}.csv")

    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
        log_message(f"Loaded existing data file with {len(df)} rows.")
    else:
        df = pd.DataFrame()
        log_message(f"Started new collection file: {save_path}")

    session, _, _ = fetch_session()
    log_message(f"Started data collection for symbols: {symbols} every {interval_seconds} seconds.")

    try:
        while True:
            now = datetime.now()
            market_open = now.replace(hour=7, minute=0, second=0, microsecond=0)
            market_close = now.replace(hour=20, minute=0, second=0, microsecond=0)

            if market_open <= now <= market_close:
                stock_info = fetch_stock_info(session, symbols, sleep_between_calls=True)
                timestamp = now
                rows = []

                for symbol, data in stock_info.items():
                    row = {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "Close": data["formatted_data"]["Close"],
                        "Volume": data["formatted_data"]["Volume"],
                        "Bid": data["formatted_data"]["bid"],
                        "BidSize": data["formatted_data"]["bid size"],
                        "Ask": data["formatted_data"]["ask"],
                        "AskSize": data["formatted_data"]["ask size"],
                        "Spread": data["formatted_data"]["bid_ask_spread"],
                        "MarketStatus": data["formatted_data"]["market_status"]
                    }
                    rows.append(row)

                new_df = pd.DataFrame(rows)
                df = pd.concat([df, new_df], ignore_index=True)
                df.to_csv(save_path, index=False)

                log_message(f"Saved {len(new_df)} records at {timestamp.strftime('%H:%M:%S')}. Total: {len(df)}")
            else:
                log_message("Market closed. Skipping data collection.")
                # Need to exit if market is closed.
                sys.exit(0)
            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        log_message("Data collection manually stopped.")
        sys.exit(0)

    except Exception as e:
        log_message(f"ERROR during data collection: {e}")
        sys.exit(1)

def fetch_session():
    from inc.functions import oauth
    return oauth()

# === LOGGING FUNCTION === #
def log_message(message, LOG_PATH="inc/logs/data_save_log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as log:
        log.write(f"[{timestamp}] {message}\n")

def load_crypto_data(symbol="BTC-USD", folder_path="data/crypto"):
    """
    Loads and concatenates CSV files for a given crypto symbol.
    
    Parameters:
        symbol (str): The symbol to load (default "BTC-USD").
        folder_path (str): Path to the folder containing CSV files.
    
    Returns:
        pd.DataFrame: Combined DataFrame sorted by datetime.
    """
    pattern = f"{folder_path}/{symbol.replace('-', '_')}_*.csv"
    files = glob.glob(pattern)

    if not files:
        raise ValueError(f"No files found for {symbol} in {folder_path}")

    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs, ignore_index=True)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values(['Datetime']).reset_index(drop=True)  # <- Keep all rows, just sort

    return df

def generate_signal(row, rsi_buy_threshold=40, rsi_sell_threshold=65):
    if row['Crossover'] == 'Bullish' and row['RSI'] < rsi_buy_threshold:
        return 'BUY'
    elif row['Crossover'] == 'Bearish' and row['RSI'] > rsi_sell_threshold:
        return 'SELL'
    else:
        return 'HOLD'

def backtest_signals(df, initial_balance=1000):
    """
    Simulates trading based on generated signals.

    Parameters:
        df (pd.DataFrame): DataFrame with 'Signal' and 'Close' columns.
        initial_balance (float): Starting cash balance.

    Returns:
        float: Final account balance after simulation.
        list: History of account values for plotting if needed.
    """
    balance = initial_balance
    position = 0  # Number of coins held
    account_history = []

    for i, row in df.iterrows():
        price = row['Close']
        signal = row['Signal']

        if signal == 'BUY' and balance > 0:
            position = balance / price  # Buy as much as possible
            balance = 0
        elif signal == 'SELL' and position > 0:
            balance = position * price  # Sell everything
            position = 0

        # Track account value at each step
        account_value = balance + (position * price)
        account_history.append(account_value)

    # Final value (if still holding position, it will be liquidated at last price)
    final_value = balance + (position * df['Close'].iloc[-1])

    return final_value, account_history

def auto_backtest(symbols, folder_path="data/crypto", initial_balance=1000):
    """
    Automatically backtests multiple crypto symbols.

    Parameters:
        symbols (list): List of symbol strings.
        folder_path (str): Path to the folder with CSV files.
        initial_balance (float): Starting cash for each symbol.

    Returns:
        pd.DataFrame: Summary of final balances for each symbol.
    """
    results = []

    for symbol in symbols:
        try:
            df = load_crypto_data(symbol=symbol, folder_path=folder_path)
            df = apply_all_indicators(df)
            df['Signal'] = df.apply(generate_signal, axis=1)
            final_balance, history = backtest_signals(df, initial_balance=initial_balance)

            results.append({
                'Symbol': symbol,
                'Final_Balance': final_balance,
                'Profit': final_balance - initial_balance,
                'Profit_%': ((final_balance - initial_balance) / initial_balance) * 100
            })
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    summary_df = pd.DataFrame(results)
    return summary_df

def live_audit(symbols, rsi_settings=None, folder_path="data/crypto"):
    """
    Provides the latest signal for each symbol for real-time auditing,
    using custom RSI thresholds if available.
    
    Parameters:
        symbols (list): List of symbols to audit.
        rsi_settings (dict, optional): Preloaded RSI settings. If missing, use default thresholds.
        folder_path (str): Path to candle data.
    
    Returns:
        pd.DataFrame: Latest audit snapshot.
    """
    audit_results = []

    for symbol in symbols:
        try:
            df = load_crypto_data(symbol=symbol, folder_path=folder_path)
            df = apply_all_indicators(df)

            # Use custom thresholds if available, else default to 40/65
            buy_threshold = 40
            sell_threshold = 65

            if rsi_settings and symbol in rsi_settings:
                buy_threshold = rsi_settings[symbol].get('buy_threshold', 40)
                sell_threshold = rsi_settings[symbol].get('sell_threshold', 65)

            df['Signal'] = df.apply(lambda row: generate_signal(row, rsi_buy_threshold=buy_threshold, rsi_sell_threshold=sell_threshold), axis=1)

            latest = df.iloc[-1]
            audit_results.append({
                'Symbol': symbol,
                'Datetime': latest['Datetime'],
                'Close': latest['Close'],
                'Signal': latest['Signal'],
                'RSI': latest['RSI'],
                'MACD': latest['MACD'],
                'Signal_Line': latest['Signal_Line'],
                'MACD_Diff': latest['MACD_Diff']
            })

        except Exception as e:
            print(f"Error auditing {symbol}: {e}")

    audit_df = pd.DataFrame(audit_results)
    return audit_df

def save_rsi_settings(settings, output_path="data/rsi_settings.json"):
    """
    Saves optimized RSI thresholds to a JSON file.

    Parameters:
        settings (dict): Dictionary with symbol as key and buy/sell thresholds.
        output_path (str): Where to save the JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(settings, f, indent=4)
    print(f"RSI settings saved to {output_path}")

def load_rsi_settings(input_path="data/rsi_settings.json"):
    """
    Loads saved RSI thresholds from a JSON file.

    Returns:
        dict: Symbol-based RSI thresholds.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"RSI settings file not found at {input_path}")

    with open(input_path, 'r') as f:
        settings = json.load(f)

    return settings

def optimize_rsi_thresholds(df, buy_range=(30, 50, 5), sell_range=(60, 80, 5), initial_balance=1000):
    """
    Finds the best RSI buy/sell thresholds by backtesting different combinations.

    Parameters:
        df (pd.DataFrame): DataFrame with indicators already applied.
        buy_range (tuple): (start, end, step) for buy RSI threshold testing.
        sell_range (tuple): (start, end, step) for sell RSI threshold testing.
        initial_balance (float): Starting balance for backtest.

    Returns:
        dict: Best parameters and corresponding performance.
    """
    best_result = {
        'buy_threshold': None,
        'sell_threshold': None,
        'final_balance': -float('inf')
    }

    for buy_threshold in range(*buy_range):
        for sell_threshold in range(*sell_range):
            temp_df = df.copy()
            temp_df['Signal'] = temp_df.apply(
                lambda row: generate_signal(row, rsi_buy_threshold=buy_threshold, rsi_sell_threshold=sell_threshold),
                axis=1
            )
            final_balance, _ = backtest_signals(temp_df, initial_balance=initial_balance)

            if final_balance > best_result['final_balance']:
                best_result = {
                    'buy_threshold': buy_threshold,
                    'sell_threshold': sell_threshold,
                    'final_balance': final_balance
                }

    return best_result

class PaperTrader:
    def __init__(self, symbols, initial_cash=1000):
        self.cash = {symbol: initial_cash for symbol in symbols}
        self.positions = {symbol: 0 for symbol in symbols}
        self.history = []  # Store dicts instead of just strings
        self.snapshots = []  # New list to store snapshots


    def act(self, audit_row):
        symbol = audit_row.Symbol
        signal = audit_row.Signal
        price = audit_row.Close
        timestamp = audit_row.Datetime

        action = None
        amount = 0
        proceeds = 0

        if signal == "BUY" and self.positions[symbol] == 0 and self.cash[symbol] > 0:
            amount = self.cash[symbol] / price
            self.positions[symbol] = amount
            self.cash[symbol] = 0
            action = "BUY"

        elif signal == "SELL" and self.positions[symbol] > 0:
            proceeds = self.positions[symbol] * price
            self.cash[symbol] = proceeds
            self.positions[symbol] = 0
            action = "SELL"

        if action:
            self.history.append({
                'Datetime': timestamp,
                'Symbol': symbol,
                'Action': action,
                'Price': price,
                'Amount': amount if action == "BUY" else 0,
                'Proceeds': proceeds if action == "SELL" else 0,
                'Portfolio_Value': self.portfolio_value({symbol: price})
            })

    def portfolio_value(self, prices):
        total = 0
        for symbol, cash_balance in self.cash.items():
            total += cash_balance
            total += self.positions[symbol] * prices.get(symbol, 0)
        return total

    def save_history(self, output_dir="data/paper_trader"):
        """Saves the trade history to a CSV file, only if there are trades."""
        if not self.history:
            print("No trades to save.")
            return

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        history_df = pd.DataFrame(self.history)
        output_path = os.path.join(output_dir, f"paper_trades_{timestamp}.csv")
        history_df.to_csv(output_path, index=False)
        log_message(f"Paper trading history saved to {output_path}", "inc/logs/coinbase_data_save_log.txt")


    def snapshot_portfolio(self, current_prices, timestamp):
        """
        Takes a snapshot of total portfolio value at the current time.

        Parameters:
            current_prices (dict): Latest prices {symbol: price}.
            timestamp (datetime): Current timestamp.
        """
        snapshot = {
            'Datetime': timestamp,
            'Total_Portfolio_Value': self.portfolio_value(current_prices)
        }
        self.snapshots.append(snapshot)

    def save_snapshots(self, output_dir="data/paper_trader_snapshots"):
        """Saves portfolio snapshots to a CSV file, only if there are snapshots."""
        if not self.snapshots:
            print("No snapshots to save.")
            return

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        snapshots_df = pd.DataFrame(self.snapshots)
        output_path = os.path.join(output_dir, f"portfolio_snapshots_{timestamp}.csv")
        snapshots_df.to_csv(output_path, index=False)
        print(f"Portfolio snapshots saved to {output_path}")

def preload_models_and_scalers(symbols, model_dir="inc/models"):
    models = {}
    scalers = {}

    for symbol in symbols:
        try:
            model_path = os.path.join(model_dir, f"{symbol.replace('-', '_')}_rf_model.pkl")
            scaler_path = os.path.join(model_dir, f"{symbol.replace('-', '_')}_scaler.pkl")

            models[symbol] = joblib.load(model_path)
            scalers[symbol] = joblib.load(scaler_path)

        except FileNotFoundError:
            print(f"Missing model or scaler for {symbol}")
            continue

    return models, scalers

# def create_profit_labels(df, profit_threshold=0.005, loss_threshold=-0.005, future_window=5):
#     """
#     Labels data based on future returns, handled separately for each symbol.
    
#     Parameters:
#         df (pd.DataFrame): Must contain 'Symbol' and 'Close'.
#         profit_threshold (float): Percent gain to trigger BUY label.
#         loss_threshold (float): Percent loss to trigger SELL label.
#         future_window (int): Number of rows (minutes) to look ahead.

#     Returns:
#         pd.DataFrame: DataFrame with added 'Label' column.
#     """
#     df = df.copy()
#     df['Label'] = 0  # Default HOLD

#     symbols = df['Symbol'].unique()

#     for symbol in symbols:
#         df_symbol = df[df['Symbol'] == symbol]

#         future_returns = (df_symbol['Close'].shift(-future_window) - df_symbol['Close']) / df_symbol['Close']

#         buy_condition = future_returns > profit_threshold
#         sell_condition = future_returns < loss_threshold

#         df.loc[df_symbol.index[buy_condition], 'Label'] = 1   # BUY
#         df.loc[df_symbol.index[sell_condition], 'Label'] = -1  # SELL

#     return df

# def create_profit_labels(df, profit_threshold=0.005, loss_threshold=-0.005, future_window=5, rsi_settings_path='data/rsi_settings.json'):
#     """
#     Labels data based on future returns and adds RSI threshold flags.
    
#     Parameters:
#         df (pd.DataFrame): Must contain 'Symbol', 'Close', and 'RSI'.
#         profit_threshold (float): Percent gain to trigger BUY label.
#         loss_threshold (float): Percent loss to trigger SELL label.
#         future_window (int): Number of rows (minutes) to look ahead.
#         rsi_settings_path (str): Path to RSI threshold JSON.

#     Returns:
#         pd.DataFrame: DataFrame with added 'Label', 'RSI_Buy_Zone', 'RSI_Sell_Zone'.
#     """
#     # Load RSI settings
#     with open(rsi_settings_path, 'r') as f:
#         rsi_settings = json.load(f)

#     df = df.copy()
#     df['Label'] = 0
#     df['RSI_Buy_Zone'] = 0
#     df['RSI_Sell_Zone'] = 0

#     symbols = df['Symbol'].unique()

#     for symbol in symbols:
#         if symbol not in rsi_settings:
#             continue  # Skip symbols without thresholds

#         # Get thresholds
#         rsi_buy = rsi_settings[symbol]['buy_threshold']
#         rsi_sell = rsi_settings[symbol]['sell_threshold']

#         df_symbol = df[df['Symbol'] == symbol]

#         # Future return calculation
#         future_returns = (df_symbol['Close'].shift(-future_window) - df_symbol['Close']) / df_symbol['Close']

#         buy_condition = future_returns > profit_threshold
#         sell_condition = future_returns < loss_threshold

#         df.loc[df_symbol.index[buy_condition], 'Label'] = 1
#         df.loc[df_symbol.index[sell_condition], 'Label'] = -1

#         # Add RSI zone flags
#         df.loc[df_symbol.index, 'RSI_Buy_Zone'] = (df_symbol['RSI'] < rsi_buy).astype(int)
#         df.loc[df_symbol.index, 'RSI_Sell_Zone'] = (df_symbol['RSI'] > rsi_sell).astype(int)

#     return df

def create_profit_labels(df, profit_multiplier=0.5, loss_multiplier=0.5, future_window=5):
    df = df.copy()
    df['Label'] = 0  # Default: Hold
    df['RSI_Buy_Zone'] = 0
    df['RSI_Sell_Zone'] = 0

    symbols = df['Symbol'].unique() if 'Symbol' in df.columns else [None]
    for symbol in symbols:
        df_symbol = df if symbol is None else df[df['Symbol'] == symbol]
        future_returns = (df_symbol['Close'].shift(-future_window) - df_symbol['Close']) / df_symbol['Close']
        
        atr = df_symbol['ATR']
        profit_threshold = profit_multiplier * atr
        loss_threshold = -loss_multiplier * atr

        buy_condition = future_returns > profit_threshold
        sell_condition = future_returns < loss_threshold

        df.loc[df_symbol.index[buy_condition], 'Label'] = 1   # BUY
        df.loc[df_symbol.index[sell_condition], 'Label'] = -1  # SELL

    return df


# def backtest_labels_per_symbol(df, initial_balance=1000, fee_rate=0.0035):
#     """
#     Backtests label-based trading per symbol with fee adjustment and trade count.

#     Parameters:
#         df (pd.DataFrame): Must include 'Symbol', 'Close', and 'Label' columns.
#         initial_balance (float): Starting balance per symbol.
#         fee_rate (float): Total round-trip fee (e.g., 0.007 = 0.35% buy + 0.35% sell).

#     Returns:
#         pd.DataFrame: Summary with balance, profit, percent return, and trade count.
#     """
#     results = []
#     symbols = df['Symbol'].unique()
#     fee_per_trade = fee_rate / 2  # split between buy/sell

#     for symbol in symbols:
#         df_symbol = df[df['Symbol'] == symbol]
#         balance = initial_balance
#         position = 0
#         trades = 0  # count of buy + sell
#         history = []

#         for _, row in df_symbol.iterrows():
#             price = row['Close']
#             label = row['Label']

#             if label == 1 and balance > 0:  # BUY
#                 position = (balance / price) * (1 - fee_per_trade)
#                 balance = 0
#                 trades += 1

#             elif label == -1 and position > 0:  # SELL
#                 balance = (position * price) * (1 - fee_per_trade)
#                 position = 0
#                 trades += 1

#             account_value = balance + (position * price)
#             history.append(account_value)

#         # Final liquidation if still holding
#         if position > 0:
#             balance = (position * df_symbol['Close'].iloc[-1]) * (1 - fee_per_trade)

#         final_balance = balance


#         results.append({
#             'Symbol': symbol,
#             'Final_Balance': final_balance,
#             'Profit': final_balance - initial_balance,
#             'Profit_%': ((final_balance - initial_balance) / initial_balance) * 100,
#             'Trades': trades
#         })

#     return pd.DataFrame(results)

# def backtest_labels_per_symbol(df, initial_balance=1000, fee_rate=0.0035):
#     """
#     Backtests label-based trading per symbol with fee adjustment, trade count, and average trade profit.

#     Returns:
#         pd.DataFrame: Summary with balance, profit, percent return, trade count, and avg profit per trade.
#     """
#     results = []
#     symbols = df['Symbol'].unique()
#     fee_per_trade = fee_rate / 2  # split between buy/sell

#     for symbol in symbols:
#         df_symbol = df[df['Symbol'] == symbol]
#         balance = initial_balance
#         position = 0
#         trades = 0  # count of buy + sell
#         trade_profits = []  # to store each trade % return

#         buy_price = None

#         for _, row in df_symbol.iterrows():
#             price = row['Close']
#             label = row['Label']

#             if label == 1 and balance > 0:  # BUY
#                 buy_price = price
#                 position = (balance / price) * (1 - fee_per_trade)
#                 balance = 0
#                 trades += 1

#             elif label == -1 and position > 0 and buy_price is not None:  # SELL
#                 sell_price = price
#                 balance = (position * price) * (1 - fee_per_trade)
#                 position = 0
#                 trades += 1

#                 # Calculate individual trade profit %
#                 net_return = ((sell_price - buy_price) / buy_price) * 100
#                 net_return -= fee_rate * 100  # account for round-trip fee
#                 trade_profits.append(net_return)

#         # Final liquidation if still holding
#         if position > 0 and buy_price is not None:
#             final_price = df_symbol['Close'].iloc[-1]
#             balance = (position * final_price) * (1 - fee_per_trade)

#             # Estimate profit % if liquidated
#             net_return = ((final_price - buy_price) / buy_price) * 100
#             net_return -= fee_rate * 100
#             trade_profits.append(net_return)
#             trades += 1  # consider it a final sell

#         final_balance = balance
#         avg_trade_profit = sum(trade_profits) / len(trade_profits) if trade_profits else 0

#         results.append({
#             'Symbol': symbol,
#             'Final_Balance': final_balance,
#             'Profit': final_balance - initial_balance,
#             'Profit_%': ((final_balance - initial_balance) / initial_balance) * 100,
#             'Trades': trades,
#             'Avg_Trade_Profit_%': avg_trade_profit
#         })

#     return pd.DataFrame(results)
def backtest_labels_with_transactions(df, initial_balance=1000, fee_rate=0.0035):
    df = df.copy()
    df['Transaction'] = ""  # initialize empty column

    results = []
    symbols = df['Symbol'].unique()
    fee_per_trade = fee_rate / 2

    for symbol in symbols:
        df_symbol = df[df['Symbol'] == symbol]
        balance = initial_balance
        position = 0
        trades = 0
        trade_profits = []
        buy_price = None

        for idx, row in df_symbol.iterrows():
            price = row['Close']
            label = row['Label']

            if label == 1 and balance > 0:
                buy_price = price
                position = (balance / price) * (1 - fee_per_trade)
                balance = 0
                trades += 1
                df.loc[idx, 'Transaction'] = "BUY"

            elif label == -1 and position > 0 and buy_price is not None:
                sell_price = price
                balance = (position * price) * (1 - fee_per_trade)
                position = 0
                trades += 1
                df.loc[idx, 'Transaction'] = "SELL"

                net_return = ((sell_price - buy_price) / buy_price) * 100
                net_return -= fee_rate * 100
                trade_profits.append(net_return)

        # Final liquidation if needed
        if position > 0 and buy_price is not None:
            final_price = df_symbol['Close'].iloc[-1]
            balance = (position * final_price) * (1 - fee_per_trade)
            net_return = ((final_price - buy_price) / buy_price) * 100
            net_return -= fee_rate * 100
            trade_profits.append(net_return)
            trades += 1

            last_idx = df_symbol.index[-1]
            df.loc[last_idx, 'Transaction'] = "SELL"

        final_balance = balance
        avg_trade_profit = sum(trade_profits) / len(trade_profits) if trade_profits else 0

        results.append({
            'Symbol': symbol,
            'Final_Balance': final_balance,
            'Profit': final_balance - initial_balance,
            'Profit_%': ((final_balance - initial_balance) / initial_balance) * 100,
            'Trades': trades,
            'Avg_Trade_Profit_%': avg_trade_profit
        })

    return pd.DataFrame(results), df


### === Machine Learning === ###
def save_model_assets(model, scaler, feature_columns, folder="inc/models", name="crypto_features"):
    """
    Save the model, scaler, and feature columns to the specified folder with a shared name prefix.
    """
    os.makedirs(folder, exist_ok=True)

    # Paths
    model_path = os.path.join(folder, f"{name}.keras")
    scaler_path = os.path.join(folder, f"{name}_scaler.pkl")
    features_path = os.path.join(folder, f"{name}_features.pkl")

    # Save model
    model.save(model_path)

    # Save scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save feature columns
    with open(features_path, "wb") as f:
        pickle.dump(feature_columns, f)

    print(f"Model, scaler, and features saved to {folder} as {name}.*")

def load_model_assets(folder="inc/models", name="crypto_features"):
    """
    Load the model, scaler, and feature columns from the specified folder and shared name prefix.
    
    Returns:
        model: Keras model
        scaler: StandardScaler
        feature_columns: List[str]
    """
    # Paths
    model_path = os.path.join(folder, f"{name}.keras")
    scaler_path = os.path.join(folder, f"{name}_scaler.pkl")
    features_path = os.path.join(folder, f"{name}_features.pkl")

    # Load assets
    model = keras.models.load_model(model_path)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    with open(features_path, "rb") as f:
        feature_columns = pickle.load(f)

    print(f"Model, scaler, and features loaded from {folder} as {name}.*")
    return model, scaler, feature_columns

# def drop_unused_columns(df):
#     drop_cols = [
#         'Symbol', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
#         'Price_Position', 'Crossover', 'Pre_x_Warning', 'Pre_x_angle',
#         'Best_Bid_Size', 'Best_Ask_Size'  # Dropped in favor of ratio versions
#     ]
#     df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
#     return df

# def normalize_columns(df):
#     # Normalize by Close price
#     normalize_cols = [
#         'SMA_9', 'SMA_21', 'SMA_50', 'SMA_200',
#         'EMA_5', 'EMA_9', 'EMA_12', 'EMA_13', 'EMA_26',
#         'MACD', 'Signal_Line', 'MACD_Diff',
#         'VWAP_1m', 'VWAP_15m', 'VWAP_1h', 'VWAP_Day',
#         'Middle_Band', 'Upper_Band', 'Lower_Band', 'Band_Width',
#         'Peak', 'Trough',
#         'Fib_23.6%', 'Fib_38.2%', 'Fib_50.0%', 'Fib_61.8%', 'Fib_100.0%',
#         'Fib_127.2%', 'Fib_161.8%', 'Fib_200.0%', 'Fib_261.8%', 'Fib_423.6%',
#         'Fib_Bearish_127.2%', 'Fib_Bearish_161.8%', 'Fib_Bearish_200.0%',
#         'Fib_Bearish_261.8%', 'Fib_Bearish_423.6%',
#         'TR', 'ATR'
#     ]
#     for col in normalize_cols:
#         if col in df.columns and 'Close' in df.columns:
#             df[col] = df[col] / df['Close']
#     return df

def drop_unused_columns(df):
    drop_cols = [
        'Symbol', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Price_Position', 'Crossover', 'Pre_x_Warning', 'Pre_x_angle',
        'Best_Bid_Size', 'Best_Ask_Size','Transaction', 'VWAP_Day', 'RSI_Buy_Zone', 'RSI_Sell_Zone',  # Dropped in favor of ratio versions
        'Fib_50.0%', 'Fib_61.8%', 'Fib_100.0%', 'Fib_127.2%', 'Fib_161.8%',
        'Fib_200.0%', 'Fib_261.8%', 'Fib_423.6%', 'Fib_Bearish_127.2%',
        'Fib_Bearish_161.8%', 'Fib_Bearish_200.0%', 'Fib_Bearish_261.8%',
        'Fib_Bearish_423.6%','VWAP_15m', 'VWAP_1h', 'VWAP_Day', 'Price_Position',
        'Crossover', 'Pre_x_Warning', 'Pre_x_angle'
    ]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')
    return df

def normalize_columns(df):
    # Normalize by Close price
    # VWAP_Day was taken out after VWAP_1h
    normalize_cols = [
        'SMA_9', 'SMA_21', 'SMA_50', 'SMA_200',
        'EMA_5', 'EMA_9', 'EMA_12', 'EMA_13', 'EMA_26',
        'MACD', 'Signal_Line', 'MACD_Diff',
        'VWAP_1m', 'VWAP_15m', 'VWAP_1h',
        'Middle_Band', 'Upper_Band', 'Lower_Band', 'Band_Width',
        'Peak', 'Trough',
        'Fib_23.6%', 'Fib_38.2%', 'Fib_50.0%', 'Fib_61.8%', 'Fib_100.0%',
        'Fib_127.2%', 'Fib_161.8%', 'Fib_200.0%', 'Fib_261.8%', 'Fib_423.6%',
        'Fib_Bearish_127.2%', 'Fib_Bearish_161.8%', 'Fib_Bearish_200.0%',
        'Fib_Bearish_261.8%', 'Fib_Bearish_423.6%',
        'TR', 'ATR'
    ]
    for col in normalize_cols:
        if col in df.columns and 'Close' in df.columns:
            df[col] = df[col] / df['Close']
    return df

def clean_features(df):
    df = df.copy()
    
    # Generate order book ratios before dropping
    if {'Best_Bid_Size', 'Total_Bid_Depth'}.issubset(df.columns):
        df['Best_Bid_Ratio'] = df['Best_Bid_Size'] / df['Total_Bid_Depth']
    if {'Best_Ask_Size', 'Total_Ask_Depth'}.issubset(df.columns):
        df['Best_Ask_Ratio'] = df['Best_Ask_Size'] / df['Total_Ask_Depth']

    # Apply log scaling to total depth values
    for col in ['Total_Bid_Depth', 'Total_Ask_Depth']:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Normalize OBV and Volume_SMA per symbol using z-score
    if 'Symbol' in df.columns:
        if 'OBV' in df.columns:
            df['OBV'] = df.groupby('Symbol')['OBV'].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
        if 'Volume_SMA' in df.columns:
            df['Volume_SMA'] = df.groupby('Symbol')['Volume_SMA'].transform(lambda x: (x - x.mean()) / x.std(ddof=0))

    df = normalize_columns(df)
    df = drop_unused_columns(df)
    return df


def combine_symbol_datasets(symbols, load_func, indicator_func, label_func):
    """
    Loads, processes, and combines multiple symbol DataFrames into one training dataset.
    
    Parameters:
        symbols (list): List of symbol strings (e.g., ["BTC-USD", "ETH-USD"]).
        load_func (function): Function to load raw data for a symbol.
        indicator_func (function): Function to apply indicators to a DataFrame.
        label_func (function): Function to generate profit labels for a DataFrame.
    
    Returns:
        pd.DataFrame: Combined and shuffled DataFrame ready for feature cleaning.
    """
    combined = []

    for symbol in symbols:
        try:
            df = load_func(symbol)
            df = indicator_func(df)
            df = label_func(df)
            df["Symbol"] = symbol  # Track origin if needed later
            combined.append(df)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")

    if not combined:
        raise ValueError("No symbol data could be combined.")

    merged_df = pd.concat(combined, ignore_index=True)
    merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    return merged_df

def crypto_ml_training(df):
    training_df = clean_features(df)
    training_df = training_df.replace([np.inf, -np.inf], np.nan)
    training_df = training_df.dropna(subset=['Label'])  # Make sure Label is present

    X = training_df.drop(columns=['Label'], errors='ignore')
    y = training_df['Label']

    # print("Columns with most NaNs:")
    # print(X.isna().sum().sort_values(ascending=False).head(10))


    # Drop feature columns with too many NaNs
    X = X.dropna(axis=1, thresh=0.95 * len(X))  # Keep columns with >=95% valid rows

    # Replace infs and drop any rows with remaining NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()
    y = y.loc[X.index]  # Align

    # Recode labels
    y = y.replace({-1: 0, 0: 1, 1: 2})

    # print(f"Final usable rows: {len(X)}")

    # Split and scale (unchanged)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=78)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))  # 3-class output

    # Compile
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train
    history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Accuracy: {test_accuracy}')
    save_model_assets(model, scaler, X.columns.tolist(), folder="inc/models", name="crypto_features_all")
    
# Redefine the function after kernel reset
def predict_trade_signal(live_df, model, scaler, feature_columns, thresholds={-1: 0.55, 0: 0.7, 1: 0.55}):
    """
    Predicts the trade signal (Buy=1, Hold=0, Sell=-1) for a given row of market data.
    
    Parameters:
        live_df (pd.DataFrame): A DataFrame containing a single latest row of raw data.
        model (tf.keras.Model): Trained Keras model.
        scaler (StandardScaler): Fitted scaler used during training.
        feature_columns (list): Ordered list of feature column names used during training.
        thresholds (dict): Confidence thresholds for each class {-1, 0, 1}.
    
    Returns:
        decision (int): One of {-1, 0, 1}
        confidence (float): Confidence of the chosen class
        probs (dict): Full probability breakdown
    """
    # Step 1: Clean and prepare
    processed = clean_features(live_df.copy())
    processed = processed.replace([np.inf, -np.inf], np.nan)
    processed = processed[feature_columns]
    processed = processed.dropna()

    if processed.empty:
        return 0, 0.0, {"Sell (-1)": 0.0, "Hold (0)": 1.0, "Buy (1)": 0.0}

    # Step 2: Scale the data
    scaled = scaler.transform(processed)

    # Step 3: Predict probabilities
    probs = model.predict(scaled, verbose=0)[0]
    pred_index = int(np.argmax(probs))
    class_map = {0: -1, 1: 0, 2: 1}
    pred_class = class_map[pred_index]
    pred_conf = probs[pred_index]

    # Step 4: Enforce threshold
    if pred_conf >= thresholds[pred_class]:
        decision = pred_class
    else:
        decision = 0  # Default to Hold

    # Step 5: Format probabilities
    label_map = {0: "Sell (-1)", 1: "Hold (0)", 2: "Buy (1)"}
    #prob_dict = {label_map[i]: float(p) for i, p in enumerate(probs)}
    prob_dict = {label_map[i]: round(p * 100, 5) for i, p in enumerate(probs)}

    return decision, float(pred_conf), prob_dict

def save_prediction_to_excel(symbol, prediction, confidence, probabilities, close_price, rsi_reading, output_folder="data/audit"):
    # Construct output path
    filename = f"{symbol}_Predicts.xlsx"
    filepath = os.path.join(output_folder, filename)
    
    # Prepare row as a dictionary
    row = {
        "Datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Symbol": symbol,
        "Prediction": prediction,
        "Confidence": round(confidence * 100, 2),
        "Close": close_price,
        "RSI": rsi_reading,
        "Prob_Sell": probabilities.get("SELL", None),
        "Prob_Hold": probabilities.get("HOLD", None),
        "Prob_Buy": probabilities.get("BUY", None)
    }

    # Load existing file or create new DataFrame
    if os.path.exists(filepath):
        df_existing = pd.read_excel(filepath)
        df_new = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([row])
    
    # Save to Excel
    os.makedirs(output_folder, exist_ok=True)
    df_new.to_excel(filepath, index=False)
    print(f"Prediction saved to: {filepath}")
    
# def check_current_prediction(symbol, model, scaler, feature_columns):
#     df = load_crypto_data(symbol)
#     df = apply_all_indicators(df)
#     df = clean_features(df)

#     # Load or fetch latest row of data
#     latest_row = df.iloc[[-1]]  # Wrap in double brackets to keep it as DataFrame

#     # Predict trade action
#     action, confidence, prob_matrix = predict_trade_signal(latest_row, model, scaler, feature_columns)
#     return action, confidence, prob_matrix
def check_current_prediction(symbol, model, scaler, feature_columns):
    df = load_crypto_data(symbol)
    df = apply_all_indicators(df)
    
    # Get latest row for prediction
    latest_row = df.iloc[[-1]]
    latest_datetime = latest_row['Datetime'].values[0] if 'Datetime' in latest_row.columns else df.index[-1]
    latest_close = latest_row['Close'].values[0] if 'Close' in latest_row.columns else df.index[-1]

    # Drop unused columns and clean features
    df = clean_features(df)
    # Get latest row for prediction
    latest_row = df.iloc[[-1]]    

    # Predict trade action
    action, confidence, prob_matrix = predict_trade_signal(latest_row, model, scaler, feature_columns)

    # Prepare audit row
    audit_data = {
        "Datetime": latest_datetime,
        "Close": latest_close,
        "Logged_At": datetime.now(),
        "Symbol": symbol,
        "Action": action,
        "Confidence": confidence,
        "Prob_Buy": prob_matrix.get("Buy (1)", None),
        "Prob_Hold": prob_matrix.get("Hold (0)", None),
        "Prob_Sell": prob_matrix.get("Sell (-1)", None)
    }
    audit_df = pd.DataFrame([audit_data])

    # Create audit directory if needed
    audit_path = f"data/audit_ml_predictions/{symbol}_ml_predictions.csv"
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)

    # Append to file or create new one
    if os.path.exists(audit_path):
        audit_df.to_csv(audit_path, mode='a', header=False, index=False)
    else:
        audit_df.to_csv(audit_path, index=False)

    return action, confidence, prob_matrix
## === COINBASE API FUNCTIONS === ##
def get_coinbase_order_book(symbol, level=2):
    url = f"https://api.exchange.coinbase.com/products/{symbol}/book"
    params = {"level": level}
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'YourAppNameHere'
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        bids = data.get('bids', [])
        asks = data.get('asks', [])

        total_bid_depth = sum(float(bid[1]) for bid in bids)
        total_ask_depth = sum(float(ask[1]) for ask in asks)

        best_bid_price = float(bids[0][0]) if bids else None
        best_bid_size = float(bids[0][1]) if bids else None
        best_ask_price = float(asks[0][0]) if asks else None
        best_ask_size = float(asks[0][1]) if asks else None

        return {
            "best_bid_price": best_bid_price,
            "best_bid_size": best_bid_size,
            "best_ask_price": best_ask_price,
            "best_ask_size": best_ask_size,
            "total_bid_depth": total_bid_depth,
            "total_ask_depth": total_ask_depth
        }

    except requests.RequestException as e:
        log_message(f"Error fetching order book for {symbol}: {e}", log_path)
        return None

def coinbase_candles(symbol, end_date=None, interval='FIVE_MINUTE', convert=True, timezone='America/New_York'):
    interval_hours = {
        "ONE_MINUTE": 5, "FIVE_MINUTE": 24,
        "FIFTEEN_MINUTE": 48, "THIRTY_MINUTE": 72,
        "ONE_HOUR": 168, "TWO_HOUR": 336, "SIX_HOUR": 720, "ONE_DAY": 2000
    }
    hour_limit = interval_hours.get(interval, 24)

    now = datetime.now()
    if end_date is None:
        end = now
    else:
        end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    start = end - timedelta(hours=hour_limit)

    start_unix = int(time.mktime(start.timetuple()))
    end_unix = int(time.mktime(end.timetuple()))

    url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{symbol}/candles"
    params = {"start": start_unix, "end": end_unix, "granularity": interval, "limit": 350}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if "candles" not in data or not data["candles"]:
            log_message(f"No candle data returned for {symbol}.", log_path)
            return None

    except requests.RequestException as e:
        log_message(f"Error fetching data for {symbol}: {e}", log_path)
        return None

    df = pd.DataFrame(data["candles"])

    if df.empty or not {'low', 'high', 'open', 'close', 'volume', 'start'}.issubset(df.columns):
        log_message(f"Invalid structure in Coinbase response for {symbol}.", log_path)
        return None

    df['Datetime'] = pd.to_datetime(df['start'].astype(int), unit='s', utc=True).dt.tz_convert(timezone)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df.drop(columns=['start'], inplace=True)
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    df.set_index('Datetime', inplace=True)
    df.sort_index(inplace=True)

    return df

def live_candle_book_logger(
    symbols=["BTC-USD", "ETH-USD"],
    interval="FIVE_MINUTE",
    output_dir="data/crypto",
    retries=3
):
    os.makedirs(output_dir, exist_ok=True)

    for symbol in symbols:
        attempt = 0
        success = False

        while attempt < 3 and not success:
            try:
                candle = coinbase_candles(symbol, interval=interval, convert=False)
                book = get_coinbase_order_book(symbol, level=2)

                if candle is not None and book is not None:
                    latest = candle.iloc[-1]

                    log_row = {
                        "Symbol": symbol,
                        "Datetime": latest.name,
                        "Open": latest['Open'],
                        "High": latest['High'],
                        "Low": latest['Low'],
                        "Close": latest['Close'],
                        "Volume": latest['Volume'],
                        "Best_Bid_Size": book['best_bid_size'],
                        "Best_Ask_Size": book['best_ask_size'],
                        "Total_Bid_Depth": book['total_bid_depth'],
                        "Total_Ask_Depth": book['total_ask_depth'],
                        "Spread": (book['best_ask_price'] - book['best_bid_price']) if (book['best_ask_price'] and book['best_bid_price']) else None
                    }

                    today = datetime.now().strftime("%Y-%m-%d")
                    symbol_filename = os.path.join(output_dir, f"{symbol.replace('-', '_')}_{today}.csv")

                    if os.path.exists(symbol_filename):
                        df_existing = pd.read_csv(symbol_filename)
                        df_existing = pd.concat([df_existing, pd.DataFrame([log_row])], ignore_index=True)
                    else:
                        df_existing = pd.DataFrame([log_row])

                    df_existing.to_csv(symbol_filename, index=False)
                    log_message(f"[{datetime.now()}] Logged {symbol}", log_path)

                    success = True

                else:
                    log_message(f"[{datetime.now()}] No data for {symbol}, skipping this cycle.", log_path)

            except Exception as e:
                attempt += 1
                log_message(f"[{datetime.now()}] Error fetching data for {symbol} (Attempt {attempt}/{retries}): {e}", log_path)
                time.sleep(5)

        if not success and attempt == 3:
            log_message(f"[{datetime.now()}] Failed to fetch {symbol} after {attempt} attempts. Skipping.", log_path)