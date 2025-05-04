import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import argrelextrema
import statistics
from langchain_openai import ChatOpenAI

def calculate_macd(data, signal_window=9):
    """
    Calculate MACD and related SMAs/EMAs.
    """
    # Compute SMAs
    sma_windows = [9, 21, 50, 200]
    for window in sma_windows:
        data[f'SMA_{window}'] = data['Close'].rolling(window=window).mean()

    # Compute EMAs
    ema_windows = [5, 9, 12, 13, 26]
    for window in ema_windows:
        data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()

    # MACD Calculation
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()

    return data

def calculate_rsi(data, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    return data

def calculate_obv(data):
    """
    Calculate On-Balance Volume (OBV).
    """
    data['OBV'] = (data['Volume'] * data['Close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
    return data

def calculate_true_range(data):
    """
    Calculate True Range (TR) for ATR calculation.
    """
    prev_close = data['Close'].shift(1)
    data['TR'] = pd.concat([
        (data['High'] - data['Low']), 
        (data['High'] - prev_close).abs(), 
        (data['Low'] - prev_close).abs()
    ], axis=1).max(axis=1)

    return data

def calculate_atr(data, period=14):
    """
    Calculate Average True Range (ATR).
    """
    data['ATR'] = data['TR'].rolling(window=period).mean()
    return data

""" FROM OPEN AI CHATGPT """
def calculate_stochastic_oscillator(df, period=14, smooth_k=3):
    """
    Calculates the Stochastic Oscillator (%K and %D) and adds them to the DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' prices.
    period (int): Lookback period for calculating the highest high and lowest low. Default is 14.
    smooth_k (int): Smoothing period for %K (SMA). Default is 3.
    
    Returns:
    pd.DataFrame: DataFrame with additional columns '%K' and '%D'.
    """
    # Calculate %K
    df['Lowest_Low'] = df['Low'].rolling(window=period).min()
    df['Highest_High'] = df['High'].rolling(window=period).max()
    df['%K'] = ((df['Close'] - df['Lowest_Low']) / (df['Highest_High'] - df['Lowest_Low'])) * 100

    # Calculate %D (SMA of %K)
    df['%D'] = df['%K'].rolling(window=smooth_k).mean()

    # Drop intermediate columns
    df.drop(columns=['Lowest_Low', 'Highest_High'], inplace=True)

    return df

def calculate_volume_sma(df, period=14):
    """
    Calculates the Simple Moving Average (SMA) of Volume.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'Volume' column.
    period (int): Lookback period for SMA calculation. Default is 14.
    
    Returns:
    pd.DataFrame: Updated DataFrame with 'Volume_SMA' column.
    """
    df['Volume_SMA'] = df['Volume'].rolling(window=period).mean()
    return df

def calculate_macd_roc(df, period=5):
    """
    Calculates the Rate of Change (ROC) of MACD.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'MACD' column.
    period (int): Lookback period for ROC calculation. Default is 5.
    
    Returns:
    pd.DataFrame: Updated DataFrame with 'MACD_ROC' column.
    """
    df['MACD_ROC'] = df['MACD'].pct_change(periods=period) * 100
    return df

def calculate_vwap(df):
    """
    Calculates VWAP for multiple timeframes (1-minute, 15-minute, 1-hour, and daily VWAP).
    
    Parameters:
    df (pd.DataFrame): DataFrame containing 'Datetime', 'Close', and 'Volume' columns.
    
    Returns:
    pd.DataFrame: Updated DataFrame with 'VWAP_1m', 'VWAP_15m', 'VWAP_1h', and 'VWAP_Day' columns.
    """
    df = df.copy()

    # Ensure datetime is in proper format
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    # Function to calculate VWAP for a given rolling window
    def rolling_vwap(df, window):
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
        return vwap

    # Compute VWAP for different timeframes
    df['VWAP_1m'] = rolling_vwap(df, window=1)     # 1-minute VWAP (most reactive)
    df['VWAP_15m'] = rolling_vwap(df, window=15)   # 15-minute VWAP (short-term trend)
    df['VWAP_1h'] = rolling_vwap(df, window=60)    # 1-hour VWAP (mid-term trend)
    df['VWAP_Day'] = rolling_vwap(df, window=len(df))  # Daily VWAP (overall trend)
    df['Price_Position'] = df['Close'] - df['VWAP_1m']

    return df

def calculate_bollinger_bands(data, period=15):
    """
    Calculates Bollinger Bands (Upper, Middle, Lower Bands).
    
    Parameters:
        data (pd.DataFrame): DataFrame with 'Close' prices.
        period (int): Rolling window for SMA and standard deviation (default is 15).
    
    Returns:
        pd.DataFrame: Updated DataFrame with Bollinger Bands.
    """
    # Calculate Middle Band (SMA)
    data['Middle_Band'] = data['Close'].rolling(window=period).mean()

    # Calculate Standard Deviation
    std_dev = data['Close'].rolling(window=period).std()

    # Compute Upper & Lower Bands
    data['Upper_Band'] = data['Middle_Band'] + (std_dev * 2)
    data['Lower_Band'] = data['Middle_Band'] - (std_dev * 2)
    data['Band_Width'] = data['Upper_Band'] - data['Lower_Band']
    data['Price_vs_Band'] = (data['Close'] - data['Middle_Band']) / data['Band_Width']

    return data

def detect_dynamic_peaks_troughs(data, order=10):
    """
    Identifies dynamic peaks and troughs based on local extrema in price action.

    Parameters:
        data (pd.DataFrame): DataFrame with 'Close' prices.
        order (int): Number of candles to compare for local extrema.

    Returns:
        pd.DataFrame: DataFrame with new 'Peak' and 'Trough' columns.
    """
    data = data.copy()  # Prevents modifying the original DataFrame

    # Identify local peaks (high points)
    peak_idx = argrelextrema(data['High'].values, np.greater, order=order)[0]
    data.loc[:, 'Peak'] = np.nan  # Proper assignment using .loc
    data.loc[peak_idx, 'Peak'] = data.loc[peak_idx, 'High']

    # Identify local troughs (low points)
    trough_idx = argrelextrema(data['Low'].values, np.less, order=order)[0]
    data.loc[:, 'Trough'] = np.nan
    data.loc[trough_idx, 'Trough'] = data.loc[trough_idx, 'Low']

    # Fix the fillna() warning
    data['Peak'] = data['Peak'].ffill()  # Correct method in Pandas 3.0+
    data['Trough'] = data['Trough'].ffill()

    return data

def calculate_fibonacci_levels(data, order=10):
    """
    Calculates Fibonacci retracement and extension levels using dynamically detected peaks and troughs.

    Parameters:
        data (pd.DataFrame): DataFrame with 'High' and 'Low' columns.
        order (int): Number of periods for dynamic peak/trough detection.

    Returns:
        pd.DataFrame: Updated DataFrame with Fibonacci retracement and extension levels.
    """
    # Detect dynamic peaks & troughs
    data = detect_dynamic_peaks_troughs(data, order=order)

    # Forward-fill peaks and troughs to maintain recent detected values
    data['Peak'] = data['Peak'].ffill()
    data['Trough'] = data['Trough'].ffill()

    # Calculate price range difference
    diff = data['Peak'] - data['Trough']

    # Fibonacci Retracement Levels (Support Zones)
    data['Fib_23.6%'] = data['Peak'] - (0.236 * diff)
    data['Fib_38.2%'] = data['Peak'] - (0.382 * diff)
    data['Fib_50.0%'] = data['Peak'] - (0.500 * diff)
    data['Fib_61.8%'] = data['Peak'] - (0.618 * diff)
    data['Fib_100.0%'] = data['Trough']

    # Bullish Fibonacci Extensions (Above Peak)
    data['Fib_127.2%'] = data['Peak'] + (0.272 * diff)
    data['Fib_161.8%'] = data['Peak'] + (0.618 * diff)
    data['Fib_200.0%'] = data['Peak'] + (1.000 * diff)
    data['Fib_261.8%'] = data['Peak'] + (1.618 * diff)
    data['Fib_423.6%'] = data['Peak'] + (2.236 * diff)

    # Bearish Fibonacci Extensions (Below Trough)
    data['Fib_Bearish_127.2%'] = data['Trough'] - (0.272 * diff)
    data['Fib_Bearish_161.8%'] = data['Trough'] - (0.618 * diff)
    data['Fib_Bearish_200.0%'] = data['Trough'] - (1.000 * diff)
    data['Fib_Bearish_261.8%'] = data['Trough'] - (1.618 * diff)
    data['Fib_Bearish_423.6%'] = data['Trough'] - (2.236 * diff)

    return data

def apply_all_indicators(df):
    """
    Applies all technical indicators to the DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing OHLCV data.

    Returns:
        pd.DataFrame: Updated DataFrame with all indicators.
    """
    df = df.reset_index()

    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_true_range(df)
    df = calculate_atr(df)
    df = calculate_bollinger_bands(df)
    df = detect_dynamic_peaks_troughs(df)
    df = calculate_fibonacci_levels(df)  # Includes dynamic peak/trough detection
    df = calculate_obv(df)
    df = calculate_stochastic_oscillator(df)
    df = calculate_volume_sma(df)
    df = calculate_macd_roc(df)
    df = calculate_vwap(df)
    df = df.drop(columns=['level_0', 'index'], errors='ignore')

    # Ensure MACD_Diff is present
    df['MACD_Diff'] = df['MACD'] - df['Signal_Line']

    # Detect MACD Crossovers
    df['Crossover'] = np.where(
        (df['MACD_Diff'].shift(1) < 0) & (df['MACD_Diff'] > 0), 'Bullish',
        np.where((df['MACD_Diff'].shift(1) > 0) & (df['MACD_Diff'] < 0), 'Bearish', None)
    )

    # Pre-Crossover Warning: Detecting imminent crossovers
    df['Pre_x_Warning'] = np.where(
        (df['MACD_Diff'] < 0) & (df['MACD_Diff'].shift(-1) > 0), 'Bullish',
        np.where((df['MACD_Diff'] > 0) & (df['MACD_Diff'].shift(-1) < 0), 'Bearish', None)
    )

    # Pre-Crossover Angle: Capturing slope of MACD before crossover
    df['Pre_x_angle'] = np.where(
        df['Pre_x_Warning'].notnull(),
        abs(df['MACD_Diff'].diff()),  # Slope based on MACD_Diff changes
        None
    )


    return df

def aggregate_sentiment(feed):
    """Aggregate the sentiment score for a specific crypto."""
    sentiment_scores = [item['sentiment_score'] for item in feed]
    avg_sentiment = statistics.mean(sentiment_scores)
    return avg_sentiment

# def determine_trading_signal(sentiment_score, article_text, OPENAI_API_KEY):
#     """Call GPT API to determine Buy/Sell/Hold based on sentiment score."""    
#     OPENAI_MODEL = "gpt-3.5-turbo"
#     llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.3)

#     prompt = f"""
#         Based on the following average sentiment score: {sentiment_score:.2f}, and recent article text {article_text}, determine if it is a good time to:
#         - Buy if the score is significantly positive (e.g., above 0.2).
#         - Sell if it is significantly negative (e.g., below -0.2).
#         - Hold if it is neutral (e.g., between -0.2 and 0.2).
        
#         Please provide your reasoning and a final recommendation: Buy, Sell, or Hold with a percentage from 0 to 100 on what you'd recommend if buying or selling.
#         """
#     result = llm.invoke(prompt)
#     return result.content

def summarize_technical_indicators(df, num_rows=3):
    """Summarize key technical indicators from the DataFrame."""
    summary = df.tail(num_rows)[['Datetime', 'Low', 'High', 'Close', 'Volume', 'RSI', 'MACD', 'Signal_Line', 'SMA_50', 'SMA_200', 'ATR', 'VWAP_Day']]
    
    # Convert to a plain-language description
    recent_prices = summary[['Datetime', 'Close']].to_dict(orient='records')
    avg_rsi = summary['RSI'].mean()
    avg_macd = summary['MACD'].mean()
    avg_atr = summary['ATR'].mean()
    
    description = f"""
    Recent Price Trend (Last {num_rows} Entries):
    {recent_prices}
    
    Average RSI: {avg_rsi:.2f} (Relative Strength Index)
    Average MACD: {avg_macd:.2f} (Momentum Indicator)
    Average ATR: {avg_atr:.2f} (Volatility Indicator)
    VWAP (Day): {summary['VWAP_Day'].iloc[-1]:.2f}
    """
    return description

def determine_trading_signal(sentiment_score, article_text, df, symbol, OPENAI_API_KEY):
    """Call GPT API to determine Buy/Sell/Hold based on sentiment score and summarized technical indicators."""
    OPENAI_MODEL = "gpt-3.5-turbo"
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.3)
    
    summarized_data = summarize_technical_indicators(df)
    
    prompt = f"""
    Based on the following information, determine if it is a good time to Buy, Sell, or Hold:
    
    Average Sentiment Score For all Articles: {sentiment_score:.2f}
    Symbol most in question from all the articles is: {symbol}
    Recent Article Text: {article_text}
    Key Technical Indicator Summary:
    {summarized_data}
    
    Please provide final recommendation only of Buy, Sell, or Hold, along with a confidence percentage (0-100%).
    """
    #Please provide your reasoning and a final recommendation: Buy, Sell, or Hold, along with a confidence percentage (0-100%).
    
    result = llm.invoke(prompt)
    return result.content