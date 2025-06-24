'''
# scripts/generate_chart_images.py

import yfinance as yf
import mplfinance as mpf
import os
import pandas as pd

def generate_chart_images(ticker='AAPL', period='6mo', interval='1d', step=30):
    os.makedirs('data/pattern_images/No_Pattern', exist_ok=True)

    df = yf.download(ticker, period=period, interval=interval)
    df.dropna(inplace=True)

    total = 0
    for i in range(0, len(df) - step, step):
        chunk = df.iloc[i:i+step]
        if len(chunk) < step:
            continue

        file_path = f"data/pattern_images/No_Pattern/{ticker}_{i}.png"
        mpf.plot(chunk, type='candle', style='charles', savefig=file_path)
        total += 1

    print(f"[✓] Generated {total} images for {ticker}.")

if __name__ == "__main__":
    generate_chart_images()
'''


# scripts/generate_chart_images.py

import yfinance as yf
import mplfinance as mpf
import os
import pandas as pd
from datetime import datetime

def generate_chart_images(ticker='AAPL', period='6mo', interval='1d', step=30):
    output_dir = 'data/pattern_images/No_Pattern'
    os.makedirs(output_dir, exist_ok=True)

    print(f"[~] Downloading data for {ticker}...")
    df = yf.download(ticker, period=period, interval=interval)
    
    if df.empty or len(df) < step:
        print(f"❌ Not enough data to generate charts. Data length: {len(df)}")
        return

    df.dropna(inplace=True)
    df.index.name = 'Date'  # Required for mplfinance

    # 🧪 Show raw column format
    print("\n🧪 Column Names and Types:")
    for col in df.columns:
        print(f"{repr(col)} - {type(col)}")

    # 🔧 Fix MultiIndex columns if any
    df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]

    # 🔧 Ensure financial columns are numeric
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            print(f"❌ Column missing: {col}")
            return
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"⚠️ Could not convert column '{col}' to numeric: {e}")
            return

    df.dropna(inplace=True)  # Final cleanup

    print(f"[✓] Data cleaned. Total rows: {len(df)}\n")

    total = 0
    for i in range(0, len(df) - step + 1, step):
        chunk = df.iloc[i:i + step]
        if len(chunk) < step:
            continue

        start_date = chunk.index[0].strftime('%Y-%m-%d')
        end_date = chunk.index[-1].strftime('%Y-%m-%d')
        file_path = os.path.join(output_dir, f"{ticker}_{start_date}_to_{end_date}.png")

        try:
            mpf.plot(
                chunk,
                type='candle',
                style='charles',
                savefig=dict(fname=file_path, dpi=150, bbox_inches='tight'),
                title=f"{ticker} Chart [{start_date} → {end_date}]",
                ylabel='Price ($)',
                volume=True
            )
            print(f"[+] Saved: {file_path}")
            total += 1
        except Exception as e:
            print(f"⚠️ Failed to save image for chunk {i}-{i+step}: {e}")

    if total == 0:
        print("❌ No images generated. Check if plotting failed.")
    else:
        print(f"\n✅ Successfully generated {total} chart images for {ticker}.")

if __name__ == "__main__":
    generate_chart_images()




