# scripts/fetch_news.py

from newspaper import Article
from datetime import datetime
import pandas as pd
import requests

def fetch_google_news(ticker='AAPL'):
    url = f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en"
    response = requests.get(url)

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.content, features='xml')
    items = soup.findAll('item')

    headlines = []
    for item in items:
        title = item.title.text
        link = item.link.text
        pub_date = item.pubDate.text
        headlines.append({
            'title': title,
            'link': link,
            'published': pub_date
        })

    df = pd.DataFrame(headlines)
    df.to_csv(f"data/news_{ticker}.csv", index=False)
    print(f"[✓] Fetched and saved news for {ticker}")

if __name__ == "__main__":
    fetch_google_news('AAPL')  # or TSLA, BTC, etc.
