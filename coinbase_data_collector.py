import pandas as pd
import time, os
import requests
import nest_asyncio
from datetime import datetime, timedelta
import inc.functions as fn

nest_asyncio.apply()

pd.set_option('display.precision', 8)
log_path = "inc/logs/coinbase_data_save_log.txt"
headers = {'Content-Type': 'application/json'}

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
        fn.log_message(f"Error fetching order book for {symbol}: {e}", log_path)
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
            fn.log_message(f"No candle data returned for {symbol}.", log_path)
            return None

    except requests.RequestException as e:
        fn.log_message(f"Error fetching data for {symbol}: {e}", log_path)
        return None

    df = pd.DataFrame(data["candles"])

    if df.empty or not {'low', 'high', 'open', 'close', 'volume', 'start'}.issubset(df.columns):
        fn.log_message(f"Invalid structure in Coinbase response for {symbol}.", log_path)
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
    output_dir="data/crypto"
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
                    fn.log_message(f"[{datetime.now()}] Logged {symbol}", log_path)

                    success = True

                else:
                    fn.log_message(f"[{datetime.now()}] No data for {symbol}, skipping this cycle.", log_path)

            except Exception as e:
                attempt += 1
                fn.log_message(f"[{datetime.now()}] Error fetching data for {symbol} (Attempt {attempt}/{retries}): {e}", log_path)
                time.sleep(5)

        if not success and attempt == 3:
            fn.log_message(f"[{datetime.now()}] Failed to fetch {symbol} after {attempt} attempts. Skipping.", log_path)

if __name__ == "__main__":
    symbols = [
        "BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD",
        "ADA-USD", "AVAX-USD", "DOGE-USD", "MATIC-USD",
        "XRP-USD", "PEPE-USD", "XLM-USD"
    ]

    # 1. Collect candles
    live_candle_book_logger(symbols=symbols, interval="FIVE_MINUTE")

    # 2. Load optimized thresholds
    rsi_settings = fn.load_rsi_settings()

    # 3. Run live audit using tuned RSI settings
    audit = fn.live_audit(symbols, rsi_settings=rsi_settings)

    # 4. Save audit snapshot
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    audit_output_dir = "data/audit"
    os.makedirs(audit_output_dir, exist_ok=True)
    audit.to_csv(f"{audit_output_dir}/audit_snapshot_{timestamp}.csv", index=False)

    # 5. Initialize PaperTrader
    paper_trader = fn.PaperTrader(symbols)

    # 6. Act on each audit signal
    for row in audit.itertuples():
        paper_trader.act(row)

    # 7. Take portfolio snapshot
    current_prices = {row.Symbol: row.Close for row in audit.itertuples()}
    current_timestamp = datetime.now()
    paper_trader.snapshot_portfolio(current_prices, current_timestamp)

    # 8. SAVE Trade History and Snapshots
    paper_trader.save_history()     # NEW: Save buy/sell trade actions
    paper_trader.save_snapshots()   # NEW: Save portfolio value timeline
