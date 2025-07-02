from coinbase.rest import RESTClient
import requests
from inc.credential_manager import inject_decrypted_env, get_passphrase
import jwt
from cryptography.hazmat.primitives import serialization
import time
import secrets
import os

# Load env vars
passphrase = get_passphrase()
inject_decrypted_env(environment="prod", passphrase=passphrase)

# Env settings
client_id = os.getenv('cb_org_id')
api_key_id = os.getenv('cb_api_key')
key_secret = os.getenv('cb_secret_key').replace("\\n", "\n")
request_host = os.getenv('REQUEST_HOST', 'api.coinbase.com')

# Build correct key name
key_name = f"organizations/{client_id}/apiKeys/{api_key_id}"

def build_jwt(method, path):
    uri = f"{method.upper()} {request_host}{path}"
    private_key = serialization.load_pem_private_key(key_secret.encode(), password=None)
    payload = {
        'sub': key_name,
        'iss': "cdp",
        'nbf': int(time.time()),
        'exp': int(time.time()) + 120,
        'uri': uri
    }
    return jwt.encode(payload, private_key, algorithm='ES256', headers={'kid': key_name, 'nonce': secrets.token_hex()})

def coinbase_get(path):
    token = build_jwt("GET", path)
    url = f"https://{request_host}{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"GET {path} failed: {response.status_code} - {response.text}")
        return None

## GET BALANCES ##
def check_account_balance():
    accounts = coinbase_get("/api/v3/brokerage/accounts")
    if not accounts:
        return

    print("Non-zero account balances:")
    for acct in accounts.get("accounts", []):
        value = float(acct['available_balance']['value'])
        if value > 0:
            print(f"- {acct['name']}: {value} {acct['available_balance']['currency']}")

def get_usdc_balance():
    accounts = coinbase_get("/api/v3/brokerage/accounts")
    if not accounts:
        return 0.0

    for acct in accounts.get("accounts", []):
        if acct["name"] == "USDC Wallet":
            return float(acct["available_balance"]["value"])

    return 0.0

def get_base_balance(symbol):
    base_currency = symbol.split("-")[0]
    accounts = coinbase_get("/api/v3/brokerage/accounts")
    if not accounts:
        return 0.0
    for acct in accounts.get("accounts", []):
        if acct["available_balance"]["currency"] == base_currency:
            return float(acct["available_balance"]["value"])
    return 0.0

def get_available_balance(currency="USD"):
    accounts = coinbase_get("/api/v3/brokerage/accounts")
    if not accounts:
        return 0.0

    for acct in accounts.get("accounts", []):
        if acct["available_balance"]["currency"] == currency:
            return float(acct["available_balance"]["value"])

    return 0.0


## ORDER FUNCTIONS ##

# def place_limit_buy(symbol, quote_size, limit_price):
#     path = "/api/v3/brokerage/orders"
#     token = build_jwt("POST", path)
#     url = f"https://{request_host}{path}"
    
#     order_payload = {
#         "client_order_id": f"buy-{symbol}-{int(time.time())}",
#         "product_id": symbol,
#         "side": "BUY",
#         "order_configuration": {
#             "limit_limit_gtc": {
#                 "quote_size": f"{quote_size:.2f}",
#                 "limit_price": f"{limit_price:.8f}",
#                 "post_only": True
#             }
#         }
#     }

#     headers = {
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json",
#         "Accept": "application/json"
#     }

#     response = requests.post(url, headers=headers, json=order_payload)
#     if response.status_code == 200:
#         print(f"Limit BUY placed: {symbol} for ${quote_size:.2f} @ ${limit_price:.8f}")
#         print(response.json())
#     else:
#         print(f"Failed to place limit buy: {response.status_code} - {response.text}")

# def place_limit_sell(symbol, limit_price):
#     base_size = get_base_balance(symbol)
#     if base_size == 0:
#         print(f"No holdings to sell for {symbol}.")
#         return

#     path = "/api/v3/brokerage/orders"
#     token = build_jwt("POST", path)
#     url = f"https://{request_host}{path}"

#     order_payload = {
#         "client_order_id": f"sell-{symbol}-{int(time.time())}",
#         "product_id": symbol,
#         "side": "SELL",
#         "order_configuration": {
#             "limit_limit_gtc": {
#                 "base_size": f"{base_size:.8f}",
#                 "limit_price": f"{limit_price:.8f}",
#                 "post_only": True
#             }
#         }
#     }

#     headers = {
#         "Authorization": f"Bearer {token}",
#         "Content-Type": "application/json",
#         "Accept": "application/json"
#     }

#     response = requests.post(url, headers=headers, json=order_payload)
#     if response.status_code == 200:
#         print(f"Limit SELL placed: {symbol} for {base_size} @ ${limit_price:.8f}")
#         print(response.json())
#     else:
#         print(f"Failed to place limit sell: {response.status_code} - {response.text}")

# def has_open_order(symbol):
#     orders = coinbase_get("/api/v3/brokerage/orders/historical?limit=100&order_status=OPEN")
#     if not orders:
#         return False
#     return any(order['product_id'] == symbol for order in orders.get("orders", []))
def has_open_order(symbol):
    path = f"/api/v3/brokerage/orders/historical/batch"
    token = build_jwt("GET", path)
    url = f"https://{request_host}{path}?order_status=OPEN&product_id={symbol}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"GET {path} failed: {response.status_code} - {response.text}")
        return False

    orders = response.json().get("orders", [])
    return len(orders) > 0


def place_limit_buy(symbol, quote_size, limit_price):
    if has_open_order(symbol):
        print(f"Skipping buy: open order already exists for {symbol}")
        return

    path = "/api/v3/brokerage/orders"
    token = build_jwt("POST", path)
    url = f"https://{request_host}{path}"

    order_payload = {
        "client_order_id": f"buy-{symbol}-{int(time.time())}",
        "product_id": symbol,
        "side": "BUY",
        "order_configuration": {
            "limit_limit_gtc": {
                "quote_size": f"{quote_size:.2f}",
                "limit_price": f"{limit_price:.8f}",
                "post_only": True
            }
        }
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.post(url, headers=headers, json=order_payload)
    if response.status_code == 200:
        print(f"Limit BUY placed: {symbol} for ${quote_size:.2f} @ ${limit_price:.8f}")
        print(response.json())
    else:
        print(f"Failed to place limit buy: {response.status_code} - {response.text}")

def place_limit_sell(symbol, limit_price):
    if has_open_order(symbol):
        print(f"Skipping sell: open order already exists for {symbol}")
        return

    base_size = get_base_balance(symbol)
    if base_size == 0:
        print(f"No holdings to sell for {symbol}.")
        return

    path = "/api/v3/brokerage/orders"
    token = build_jwt("POST", path)
    url = f"https://{request_host}{path}"

    order_payload = {
        "client_order_id": f"sell-{symbol}-{int(time.time())}",
        "product_id": symbol,
        "side": "SELL",
        "order_configuration": {
            "limit_limit_gtc": {
                "base_size": f"{base_size:.8f}",
                "limit_price": f"{limit_price:.8f}",
                "post_only": True
            }
        }
    }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    response = requests.post(url, headers=headers, json=order_payload)
    if response.status_code == 200:
        print(f"Limit SELL placed: {symbol} for {base_size} @ ${limit_price:.8f}")
        print(response.json())
    else:
        print(f"Failed to place limit sell: {response.status_code} - {response.text}")
