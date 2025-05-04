import webbrowser, os, time, shutil, sys
from rauth import OAuth1Service
from datetime import datetime
import pandas as pd
import glob, json
import joblib
from inc.credential_manager import inject_decrypted_env
from inc.indicators import apply_all_indicators

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

def create_profit_labels(df, profit_threshold=0.005, loss_threshold=-0.005, future_window=5):
    """
    Labels data based on future returns, handled separately for each symbol.
    
    Parameters:
        df (pd.DataFrame): Must contain 'Symbol' and 'Close'.
        profit_threshold (float): Percent gain to trigger BUY label.
        loss_threshold (float): Percent loss to trigger SELL label.
        future_window (int): Number of rows (minutes) to look ahead.

    Returns:
        pd.DataFrame: DataFrame with added 'Label' column.
    """
    df = df.copy()
    df['Label'] = 0  # Default HOLD

    symbols = df['Symbol'].unique()

    for symbol in symbols:
        df_symbol = df[df['Symbol'] == symbol]

        future_returns = (df_symbol['Close'].shift(-future_window) - df_symbol['Close']) / df_symbol['Close']

        buy_condition = future_returns > profit_threshold
        sell_condition = future_returns < loss_threshold

        df.loc[df_symbol.index[buy_condition], 'Label'] = 1   # BUY
        df.loc[df_symbol.index[sell_condition], 'Label'] = -1  # SELL

    return df

def backtest_labels_per_symbol(df, initial_balance=1000, fee_rate=0.007):
    """
    Backtests label-based trading per symbol with fee adjustment and trade count.

    Parameters:
        df (pd.DataFrame): Must include 'Symbol', 'Close', and 'Label' columns.
        initial_balance (float): Starting balance per symbol.
        fee_rate (float): Total round-trip fee (e.g., 0.007 = 0.35% buy + 0.35% sell).

    Returns:
        pd.DataFrame: Summary with balance, profit, percent return, and trade count.
    """
    results = []
    symbols = df['Symbol'].unique()
    fee_per_trade = fee_rate / 2  # split between buy/sell

    for symbol in symbols:
        df_symbol = df[df['Symbol'] == symbol]
        balance = initial_balance
        position = 0
        trades = 0  # count of buy + sell
        history = []

        for _, row in df_symbol.iterrows():
            price = row['Close']
            label = row['Label']

            if label == 1 and balance > 0:  # BUY
                position = (balance / price) * (1 - fee_per_trade)
                balance = 0
                trades += 1

            elif label == -1 and position > 0:  # SELL
                balance = (position * price) * (1 - fee_per_trade)
                position = 0
                trades += 1

            account_value = balance + (position * price)
            history.append(account_value)

        # Final liquidation if still holding
        if position > 0:
            balance = (position * df_symbol['Close'].iloc[-1]) * (1 - fee_per_trade)

        final_balance = balance
        results.append({
            'Symbol': symbol,
            'Final_Balance': final_balance,
            'Profit': final_balance - initial_balance,
            'Profit_%': ((final_balance - initial_balance) / initial_balance) * 100,
            'Trades': trades
        })

    return pd.DataFrame(results)
