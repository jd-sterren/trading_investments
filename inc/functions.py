import webbrowser, os, time, shutil, sys
from rauth import OAuth1Service
from datetime import datetime
import pandas as pd
from inc.credential_manager import inject_decrypted_env

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