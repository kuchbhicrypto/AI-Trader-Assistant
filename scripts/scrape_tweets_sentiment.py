# scripts/scrape_tweets_sentiment.py

import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import snscrape.modules.twitter as sntwitter
import pandas as pd
from transformers import pipeline
from datetime import datetime, timedelta

def scrape_and_analyze(keyword="pepe", limit=50):
    print(f"\n🔍 Searching tweets for: {keyword}")
    
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    tweets = []
    since_date = (datetime.utcnow() - timedelta(hours=2)).strftime('%Y-%m-%d')
    query = f"{keyword} since:{since_date} lang:en"

    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit:
            break

        result = sentiment_model(tweet.content)[0]
        label = result['label']
        score = result['score']

        tweets.append({
            'time': tweet.date.strftime("%Y-%m-%d %H:%M:%S"),
            'user': tweet.user.username,
            'text': tweet.content,
            'sentiment': label,
            'confidence': round(score, 2)
        })

        emoji = "🟢" if label == "POSITIVE" else "🔴" if label == "NEGATIVE" else "⚪"
        print(f"\n💬 {tweet.content[:80]}...")
        print(f"🧠 {emoji} Sentiment: {label} ({score:.2f})")

    df = pd.DataFrame(tweets)
    df.to_csv(f"data/{keyword}_tweets_sentiment.csv", index=False)
    print(f"\n💾 Saved to data/{keyword}_tweets_sentiment.csv")

if __name__ == "__main__":
    scrape_and_analyze("pepe")

