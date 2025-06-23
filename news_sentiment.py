'''1st chatgpt 
# scripts/news_sentiment.py

from transformers import pipeline
import pandas as pd

def analyze_sentiment(file='data/news_AAPL.csv'):
    df = pd.read_csv(file)
    headlines = df['title'].tolist()

    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    sentiments = []
    for title in headlines:
        result = sentiment_model(title)[0]
        label = result['label']
        score = result['score']

        if label == 'POSITIVE':
            sentiments.append(1)
        elif label == 'NEGATIVE':
            sentiments.append(-1)
        else:
            sentiments.append(0)

    df['Sentiment'] = sentiments
    avg_sentiment = df['Sentiment'].mean()
    print(df[['title', 'Sentiment']])
    print(f"\n🧠 Average News Sentiment Score: {avg_sentiment:.2f}")

    df.to_csv(file.replace('.csv', '_scored.csv'), index=False)

if __name__ == "__main__":
    analyze_sentiment()
'''

# updated version 
# scripts/news_sentiment.py

import pandas as pd
from transformers import pipeline
from transformers.pipelines import PipelineException
import os

def analyze_sentiment(file='data/news_AAPL.csv'):
    if not os.path.exists(file):
        print(f"❌ File not found: {file}")
        return

    # Load news titles
    df = pd.read_csv(file)
    if 'title' not in df.columns:
        print("❌ 'title' column not found in the CSV.")
        return

    df = df.dropna(subset=['title'])
    headlines = df['title'].tolist()

    if len(headlines) == 0:
        print("❌ No headlines found in the file.")
        return

    try:
        # Load sentiment model using PyTorch
        sentiment_model = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            framework="pt"
        )
    except PipelineException as e:
        print("❌ Failed to load sentiment model:", str(e))
        return

    sentiments = []
    for title in headlines:
        try:
            result = sentiment_model(title)[0]
            label = result['label']
            score = result['score']

            if label == 'POSITIVE':
                sentiments.append(1)
                sentiment_str = "🟢 POSITIVE"
            elif label == 'NEGATIVE':
                sentiments.append(-1)
                sentiment_str = "🔴 NEGATIVE"
            else:
                sentiments.append(0)
                sentiment_str = "⚪ NEUTRAL"

            print(f"📰 {title.strip()[:80]}... → {sentiment_str} ({score:.2f})")

        except Exception as e:
            print(f"⚠️ Failed to analyze: {title}\n   Reason: {str(e)}")
            sentiments.append(0)

    df['Sentiment'] = sentiments
    avg_sentiment = df['Sentiment'].mean()

    print(f"\n📊 Average News Sentiment Score: {avg_sentiment:.2f}")
    
    output_file = file.replace('.csv', '_scored.csv')
    df.to_csv(output_file, index=False)
    print(f"💾 Scored data saved to: {output_file}")

if __name__ == "__main__":
    analyze_sentiment()

