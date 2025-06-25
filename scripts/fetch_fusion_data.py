'''
# scripts/fetch_fusion_data.py

import yfinance as yf
import pandas as pd
import ta
import os

def fetch_and_create_features(ticker='AAPL', period='2y', interval='1d'):
    print(f"[~] Downloading data for {ticker}...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        print("❌ Error: No data downloaded.")
        return

    print(f"[✓] Data downloaded: {len(df)} rows")

    try:
        # Use 1D pandas Series
        close = pd.Series(df['Close'].values.flatten(), index=df.index, name='Close')
        volume = df['Volume']

        # Indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
        print(f"▶️ Close type: {type(close)}, shape: {getattr(close, 'shape', 'No shape')}")
        print(close.head())
        df['MACD'] = ta.trend.MACD(close=close).macd_diff()
        df['MA20'] = close.rolling(20).mean()
        df['MA50'] = close.rolling(50).mean()
        bb = ta.volatility.BollingerBands(close=close)
        df['Bollinger_High'] = bb.bollinger_hband()
        df['Bollinger_Low'] = bb.bollinger_lband()
        df['Volatility'] = close.rolling(10).std()

        # Label
        df['Target'] = (close.shift(-1) - close).apply(lambda x: 2 if x > 0.5 else (0 if x < -0.5 else 1))

        df.dropna(inplace=True)

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/fusion_data.csv", index=False)
        print("✅ Features saved to data/fusion_data.csv")

    except Exception as e:
        print("❌ Feature creation error:", e)

if __name__ == "__main__":
    fetch_and_create_features()
'''


# scripts/fetch_fusion_data.py

import yfinance as yf
import pandas as pd
import ta
import os

def fetch_and_create_features(ticker='AAPL', period='2y', interval='1d'):
    print(f"[~] Downloading data for {ticker}...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        print("❌ Error: No data downloaded.")
        return

    print(f"[✓] Data downloaded: {len(df)} rows")

    try:
        # Force 1D Series for indicators
        df['Close'] = df['Close'].astype(float)
        close = pd.Series(df['Close'].values.flatten(), index=df.index)

        # Technical indicators
        df['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
        df['MACD'] = ta.trend.MACD(close=close).macd_diff()
        df['MA20'] = close.rolling(20).mean()
        df['MA50'] = close.rolling(50).mean()

        bb = ta.volatility.BollingerBands(close=close)
        df['Bollinger_High'] = bb.bollinger_hband()
        df['Bollinger_Low'] = bb.bollinger_lband()

        df['Volatility'] = close.rolling(10).std()

        # Target labels
        df['Target'] = (close.shift(-1) - close).apply(lambda x: 2 if x > 0.5 else (0 if x < -0.5 else 1))

        df.dropna(inplace=True)

        os.makedirs("data", exist_ok=True)
        df.to_csv("data/fusion_data.csv", index=False)
        print("✅ Features saved to data/fusion_data.csv")

    except Exception as e:
        print("❌ Feature creation error:", e)

if __name__ == "__main__":
    fetch_and_create_features()



