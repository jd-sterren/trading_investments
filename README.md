# Project Overview
Trading Investments is a live data collection and feature engineering framework for real-time and predictive modeling.
It is designed to pull, process, and store minute-level market data for stocks (currently via E*TRADE API) and future integration with cryptocurrency markets (Coinbase).
The purpose of this project is not limited to data collection but analysis using stock indicators such as RSI, MACD, OBV and new sentiment values to train a machine learning
model for possible predictive measures in real time.

## Project Structure
trading_investments/</br>
│</br>
├── data/                           # Collected live data saved here</br>
│   └── stock_data_YYYY-MM-DD.csv</br>
│</br>
├── inc/                            # Core code</br>
│   ├── functions.py                # Fetch, collect, log stock data</br>
│   ├── credential_manager.py       # AES-GCM encryption for API keys</br>
│   ├── credentials/                # Encrypted environment files (prod/dev/staging)</br>
│   └── logs/                       # Logs for collection events and errors</br>
│</br>
├── etrade_data_collector.py        # Standalone launcher for stock data collection</br>
├── .gitignore                      # (Ignore credentials, logs, compiled files)</br>
└── README.md                       # (This file)</br>

## Project Steps
<h3>🔵 Step 1: Real-Time Data Collection</h3>
Fetch live stock data via E*TRADE API (and crypto exchange APIs soon).
Collect price, volume, bid/ask sizes, bid/ask spread every 30 seconds.
Save data immediately to disk (/data/) to prevent loss during crashes.
Collect sentiment values (planned) for symbols from news, social media, or analyst ratings during each data pull window.

<h3>🔵 Step 2: Feature Engineering</h3>
Using the collected real-time data, compute:
Technical indicators:
RSI (Relative Strength Index)
MACD (Moving Average Convergence Divergence)
OBV (On-Balance Volume)
Bid/Ask Imbalance, Spread Percentage, etc.
Sentiment indicators:
Aggregated Sentiment Score per symbol per 30-second window (planned)

<h3>🔵 Step 3: Machine Learning Model Training</h3>
Training Phase:
Load historical collected data.
Train simple supervised models (starting with Decision Trees, Random Forests, Logistic Regression).
Input features:
Price action features (Close, Spread, Volume Delta)
Technical indicators (RSI, MACD, OBV)
Sentiment score
Time-based features (Time of Day, Day of Week)
Labels:
Future price movement (up, down, stable)
Volatility change (high, low)

<h3>🔵 Step 4: Real-Time Prediction and Strategy</h3>
Live Phase:
As new data is collected every 30 seconds:
Update rolling features in memory
Feed into the trained model
Predict next move:

0 Hold

1 Buy

-1 Sell

Log predictions and suggested actions for analysis

Files used currently: coinbase_data_collector.py, optimize_rsi.py