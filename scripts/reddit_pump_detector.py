# scripts/reddit_pump_detector.py

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

PUMP_KEYWORDS = ['pump', 'moon', '100x', 'buy now', 'don’t miss', 'whale', 'gem', 'exploding']
SUBREDDITS = ['CryptoMoonShots', 'cryptocurrency', 'AltcoinDiscussion']

def fetch_reddit_posts(query='pepe', limit=100, hours=3):
    end_time = int(time.time())
    start_time = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())

    all_posts = []

    for sub in SUBREDDITS:
        url = (
            f"https://api.pushshift.io/reddit/search/submission/"
            f"?q={query}&subreddit={sub}&after={start_time}&before={end_time}&size={limit}&sort=desc"
        )
        print(f"[~] Fetching from r/{sub}...")
        res = requests.get(url)
        if res.status_code != 200:
            print(f"❌ Failed to fetch from {sub}")
            continue
        data = res.json().get('data', [])
        for post in data:
            all_posts.append({
                'title': post.get('title', ''),
                'subreddit': sub,
                'score': post.get('score', 0),
                'created_utc': datetime.utcfromtimestamp(post.get('created_utc', 0)),
                'hype_score': sum([1 for word in PUMP_KEYWORDS if word in post.get('title', '').lower()])
            })

    return pd.DataFrame(all_posts)

def analyze_pump(df):
    if df.empty:
        print("📉 No Reddit pump signals found.")
        return

    hype_posts = df[df['hype_score'] > 0]
    total = len(df)
    hype_count = len(hype_posts)
    avg_hype = hype_posts['hype_score'].mean()

    print(f"\n📊 Reddit Pump Report:")
    print(f"🧵 Total Posts: {total}")
    print(f"🚀 Hype Posts: {hype_count}")
    print(f"🔥 Avg Hype Score: {avg_hype:.2f}")

    if hype_count > 5 and avg_hype >= 1:
        print("✅ PUMP DETECTED based on Reddit sentiment!")
        print(hype_posts[['title', 'subreddit', 'hype_score']].head(5).to_string(index=False))
    else:
        print("⚠️ No strong pump signal yet.")

if __name__ == "__main__":
    coin = input("🔍 Enter coin name to search on Reddit: ")
    df = fetch_reddit_posts(coin)
    analyze_pump(df)
