# scripts/fetch_macro_events.py

import tradingeconomics as te
import pandas as pd
import os
from datetime import datetime, timedelta

# === CONFIGURATION ===
te.login('guest:guest')  # Guest key provides up to 90 days historical data
COUNTRY = 'united states'  # country filter, lowercase
DAYS_BACK = 90
DAYS_FORWARD = 7
SAVE_PATH = 'data/econ_calendar.csv'

def fetch_macro_events():
    print("[~] Fetching economic events...")

    start = (datetime.today() - timedelta(days=DAYS_BACK)).strftime('%Y-%m-%d')
    end = (datetime.today() + timedelta(days=DAYS_FORWARD)).strftime('%Y-%m-%d')

    # Use calendar endpoint directly
    df = te.getCalendarData(country=COUNTRY, initDate=start, endDate=end, output_type='df')

    if df.empty:
        print("❌ No economic events returned! Please try increasing date range or check API access.")
        return

    # Select relevant columns
    df = df[['Date', 'Country', 'Category', 'Event', 'Actual', 'Forecast', 'Previous', 'Importance']]
    df = df.sort_values(by='Date', ascending=False)

    os.makedirs('data', exist_ok=True)
    df.to_csv(SAVE_PATH, index=False)
    print(f"✅ Saved {len(df)} events to {SAVE_PATH}")
    print(df.head(5))

if __name__ == "__main__":
    fetch_macro_events()
