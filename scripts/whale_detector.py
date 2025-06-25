# scripts/whale_detector.py

import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def detect_whale_activity(ticker='AAPL', period='6mo', interval='1d', threshold=2.5):
    print(f"[~] Fetching {ticker} data for whale activity detection...")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)

    if df.empty:
        print("❌ No data found.")
        return

    # Ensure Volume is a Series (not DataFrame)
    volume = df['Volume'].squeeze()  # ensures 1D Series

    # Calculate average volume
    avg_volume = volume.rolling(window=20).mean()

    # Calculate volume spike ratio
    vol_spike = volume / avg_volume

    # Combine into DataFrame
    df['Avg_Volume'] = avg_volume
    df['Vol_Spike'] = vol_spike

    # Detect whale spikes
    whales = df[df['Vol_Spike'] > threshold]

    print(f"\n🔍 Whale Activity Alerts for {ticker} (Volume Spike > {threshold}x):\n")
    for date, row in whales.iterrows():
        print(f"📅 {date.date()} | Volume: {int(row['Volume'].item()):,} | Spike: {row['Vol_Spike'].item():.2f}x")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Volume'], label='Volume', alpha=0.6)
    plt.plot(df.index, df['Avg_Volume'], label='20-Day Avg Volume', linestyle='--')
    plt.scatter(whales.index, whales['Volume'], color='red', label='Whale Spike', zorder=5)
    plt.title(f"{ticker} - Whale Detection (Volume Spikes)")
    plt.xlabel("Date")
    plt.ylabel("Volume")
    plt.legend()
    os.makedirs("plots", exist_ok=True)
    path = f"plots/{ticker}_whale_activity.png"
    plt.savefig(path, bbox_inches='tight', dpi=150)
    print(f"\n📈 Chart saved to {path}")

if __name__ == "__main__":
    detect_whale_activity('AAPL')
