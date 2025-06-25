# scripts/analyze_event_impact.py

import pandas as pd
import yfinance as yf
import os

def label_event_reactions(ticker='AAPL', econ_file='data/econ_calendar.csv', output_file='data/labeled_event_data.csv'):
    if not os.path.exists(econ_file):
        print(f"❌ Economic event file not found: {econ_file}")
        return

    # Load economic events
    df = pd.read_csv(econ_file)
    df['Date'] = pd.to_datetime(df['Date'])

    # Download stock price data
    print(f"[~] Downloading price data for {ticker}...")
    price_df = yf.download(ticker, period='2y', interval='1d', auto_adjust=True)
    if price_df.empty:
        print("❌ Failed to fetch price data.")
        return

    price_df.index = pd.to_datetime(price_df.index).date  # Normalize index to date only

    market_days = list(price_df.index)
    labeled_data = []

    for _, row in df.iterrows():
        event_date = row['Date'].date()

        # Find the event and next trading day
        valid_dates = [d for d in market_days if d >= event_date]
        if len(valid_dates) < 2:
            continue

        event_trading_day = valid_dates[0]
        next_trading_day = valid_dates[1]

        try:
            price_today = float(price_df.loc[event_trading_day]['Close'])
            price_next = float(price_df.loc[next_trading_day]['Close'])
        except Exception as e:
            print(f"⚠️ Skipping {event_trading_day} due to data error: {e}")
            continue

        # Calculate percentage change
        pct_change = ((price_next - price_today) / price_today) * 100

        # Classify the market reaction
        if pct_change > 1.0:
            label = 'Positive'
        elif pct_change < -1.0:
            label = 'Negative'
        else:
            label = 'Neutral'

        labeled_data.append({
            'Date': event_trading_day,
            'Event': row['Event'],
            'Actual': row.get('Actual', ''),
            'Forecast': row.get('Forecast', ''),
            'Previous': row.get('Previous', ''),
            'Price_Change(%)': round(pct_change, 2),
            'Label': label
        })

    # Save the labeled data
    if not labeled_data:
        print("⚠️ No valid labeled data found.")
        return

    result_df = pd.DataFrame(labeled_data)
    os.makedirs('data', exist_ok=True)
    result_df.to_csv(output_file, index=False)
    print(f"✅ Labeled event data saved to {output_file}")

if __name__ == "__main__":
    label_event_reactions('AAPL')
