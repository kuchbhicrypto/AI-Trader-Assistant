# scripts/google_pump_detector.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime

HYPE_KEYWORDS = ['pump', 'moon', 'buy now', '100x', 'don’t miss', 'exploding', 'whale', 'going to blow', 'altseason']

def fetch_google_news(ticker='pepe', limit=20):
    url = f"https://news.google.com/rss/search?q={ticker}+crypto&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"❌ Failed to fetch news. Status: {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, features='xml')
    items = soup.find_all('item')

    articles = []
    for item in items[:limit]:
        title = item.title.text
        link = item.link.text
        published = item.pubDate.text
        articles.append({
            'title': title,
            'link': link,
            'published': published,
            'hype_score': sum([1 for word in HYPE_KEYWORDS if word.lower() in title.lower()])
        })

    return articles

def detect_pump(articles):
    df = pd.DataFrame(articles)
    df['published'] = pd.to_datetime(df['published'])
    recent = df[df['published'] > (datetime.utcnow() - pd.Timedelta(hours=3))]

    total_mentions = len(recent)
    avg_hype = recent['hype_score'].mean() if total_mentions > 0 else 0

    if total_mentions > 5 and avg_hype >= 1:
        print(f"\n🚨 PUMP SIGNAL DETECTED for keyword!")
        print(f"📰 {total_mentions} hype news articles in the last 3 hours")
        print(f"🔥 Avg Hype Score: {avg_hype:.2f}")
        print(f"Top Titles:")
        print(recent[['title', 'hype_score']].head(5).to_string(index=False))
    else:
        print("\n📉 No pump signal yet...")

if __name__ == "__main__":
    keyword = input("🔍 Enter coin/keyword to monitor (e.g., pepe, dogecoin, shiba): ")
    articles = fetch_google_news(keyword)
    detect_pump(articles)
