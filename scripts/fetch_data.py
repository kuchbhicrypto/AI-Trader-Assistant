# scripts/fetch_data.py
''' first one ... by chatgpt
import yfinance as yf
import pandas as pd
import ta

def download_data(ticker='AAPL', period='2y', interval='1d'):
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)

    # Add indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd_diff()
    df.dropna(inplace=True)

    # Label: 1 = Buy, -1 = Sell, 0 = Hold
    df['Target'] = df['Close'].shift(-1) - df['Close']
    df['Target'] = df['Target'].apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))

    df.to_csv('data/trend_data.csv')
    print("[✓] Data saved to 'data/trend_data.csv'")

if __name__ == "__main__":
    download_data()
'''

# scripts/fetch_data.py

import yfinance as yf
import pandas as pd
import ta
import os

def download_data(ticker='AAPL', period='2y', interval='1d'):
    # Download historical data
    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)

    # Ensure 'Close' is 1D Series (fixes the ValueError)
    close_series = df[['Close']].squeeze()  # Converts (502,1) -> (502,)

    # Add technical indicators
    df['RSI'] = ta.momentum.RSIIndicator(close=close_series).rsi()
    df['MACD'] = ta.trend.MACD(close=close_series).macd_diff()
    df.dropna(inplace=True)

    # Add label: 1 = Buy, -1 = Sell, 0 = Hold
    df['Target'] = df['Close'].shift(-1) - df['Close']
    df['Target'] = df['Target'].apply(lambda x: 2 if x > 0.5 else (0 if x < -0.5 else 1))

    # Ensure output folder exists
    os.makedirs('data', exist_ok=True)

    # Save to CSV
    df.to_csv('data/trend_data.csv')
    print("[✓] Data saved to 'data/trend_data.csv'")

if __name__ == "__main__":
    download_data()


