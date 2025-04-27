""" Run once a week to optimize RSI thresholds for all symbols. """
import inc.functions as fn

symbols = [
    "BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD",
    "ADA-USD", "AVAX-USD", "DOGE-USD", "MATIC-USD",
    "XRP-USD", "PEPE-USD", "XLM-USD"
]

optimized_settings = {}

for symbol in symbols:
    df = fn.load_crypto_data(symbol)
    df = fn.apply_all_indicators(df)
    best = fn.optimize_rsi_thresholds(df)
    optimized_settings[symbol] = {
        'buy_threshold': best['buy_threshold'],
        'sell_threshold': best['sell_threshold']
    }

fn.save_rsi_settings(optimized_settings)
