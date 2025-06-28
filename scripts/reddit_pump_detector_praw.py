'''
# scripts/reddit_pump_detector_praw.py

import praw
import os
from datetime import datetime, timedelta

PUMP_WORDS = ['pump', 'moon', 'buy now', 'next 100x', 'gem', 'whale', 'rocket']
SUBREDDITS = ['CryptoMoonShots', 'cryptocurrency', 'AltcoinDiscussion']

reddit = praw.Reddit(
    client_id="tum7MW1J0UHmVzRuzmP2GA",
    client_secret="cyGjpljDpOeWFkLlVIDPsMfn3hqnzA",
    user_agent="Ai Trader Assistent",
    username="Low_Gazelle_3071",
    password="7383003804"
)

def detect_pump(keyword='pepe', hours=3):
    now = datetime.utcnow()
    window = timedelta(hours=hours)
    hype_posts = []

    for sub in SUBREDDITS:
        print(f"🔍 Searching r/{sub} for '{keyword}'...")
        try:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.search(keyword, sort='new', limit=50):
                post_time = datetime.utcfromtimestamp(post.created_utc)
                if now - post_time > window:
                    continue
                hype_score = sum(word in post.title.lower() for word in PUMP_WORDS)
                if hype_score > 0:
                    hype_posts.append((post.title, sub, hype_score))
        except Exception as e:
            print(f"❌ Failed on r/{sub}: {e}")

    if not hype_posts:
        print("📉 No Reddit pump signals found.")
    else:
        print(f"\n🚀 {len(hype_posts)} HYPE POSTS FOUND!")
        sorted_posts = sorted(hype_posts, key=lambda x: -x[2])
        for title, sub, score in sorted_posts[:5]:
            print(f"🧠 [{sub}] {title} → Score: {score}")
        print("\n✅ PUMP SIGNAL DETECTED based on Reddit trends!")

if __name__ == "__main__":
    keyword = input("🔍 Enter coin name (e.g. pepe): ")
    detect_pump(keyword)
'''

'''
import praw
from datetime import datetime, timedelta

PUMP_WORDS = ['pump', 'moon', 'buy now', 'next 100x', 'gem', 'whale', 'rocket']
SUBREDDITS = ['CryptoMoonShots', 'cryptocurrency', 'AltcoinDiscussion']

reddit = praw.Reddit(
    client_id="kZvx_S_DRe_NrHfQLYHRsg",
    client_secret="2V7mjTtFxezyNlfaBmnMSOk4GiBMgg",
    user_agent="AI",
    username="cryptokuchbhi",
    password="7383003804"
)

def detect_pump(keyword='pepe', hours=3):
    now = datetime.utcnow()
    window = timedelta(hours=hours)
    hype_posts = []

    for sub in SUBREDDITS:
        print(f"🔍 Searching r/{sub} for '{keyword}'...")
        try:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.search(keyword, sort='new', limit=50):
                post_time = datetime.utcfromtimestamp(post.created_utc)
                if now - post_time > window:
                    continue
                hype_score = sum(word in post.title.lower() for word in PUMP_WORDS)
                if hype_score > 0:
                    hype_posts.append((post.title, sub, hype_score))
        except Exception as e:
            print(f"❌ Failed on r/{sub}: {e}")

    if not hype_posts:
        print("📉 No Reddit pump signals found.")
    else:
        print(f"\n🚀 {len(hype_posts)} HYPE POSTS FOUND!")
        sorted_posts = sorted(hype_posts, key=lambda x: -x[2])
        for title, sub, score in sorted_posts[:5]:
            print(f"🧠 [{sub}] {title} → Score: {score}")
        print("\n✅ PUMP SIGNAL DETECTED based on Reddit trends!")

if __name__ == "__main__":
    keyword = input("🔍 Enter coin name (e.g. pepe): ")
    detect_pump(keyword)
'''



# scripts/reddit_pump_detector.py

import praw
import os
import pandas as pd
from datetime import datetime, timedelta

# 🚨 Reddit API credentials (use environment variables in production!)
reddit = praw.Reddit(
    client_id="kZvx_S_DRe_NrHfQLYHRsg",
    client_secret="2V7mjTtFxezyNlfaBmnMSOk4GiBMgg",
    user_agent="AI",
    username="cryptokuchbhi",
    password="7383003804"
)

# Subreddits & keywords
PUMP_WORDS = ['pump', 'moon', 'buy now', 'next 100x', 'gem', 'whale', 'rocket']
SUBREDDITS = ['CryptoMoonShots', 'cryptocurrency', 'AltcoinDiscussion']

def detect_pump(keyword='pepe', hours=6, debug=False):
    now = datetime.utcnow()
    window = timedelta(hours=hours)
    hype_posts = []

    for sub in SUBREDDITS:
        print(f"🔍 Searching r/{sub} for '{keyword}'...")
        try:
            subreddit = reddit.subreddit(sub)
            for post in subreddit.search(keyword, sort='new', limit=100):
                post_time = datetime.utcfromtimestamp(post.created_utc)
                if now - post_time > window:
                    continue

                hype_score = sum(word in post.title.lower() for word in PUMP_WORDS)

                if debug:
                    print(f"→ [{sub}] {post.title} | Hype Score: {hype_score}")

                if hype_score > 0:
                    hype_posts.append({
                        'title': post.title,
                        'subreddit': sub,
                        'score': hype_score,
                        'url': post.url,
                        'created': post_time.strftime("%Y-%m-%d %H:%M:%S")
                    })

        except Exception as e:
            print(f"❌ Failed on r/{sub}: {e}")

    if not hype_posts:
        print("📉 No Reddit pump signals found.")
    else:
        print(f"\n🚀 {len(hype_posts)} HYPE POSTS FOUND!")
        sorted_posts = sorted(hype_posts, key=lambda x: -x['score'])

        for post in sorted_posts[:5]:
            print(f"🧠 [{post['subreddit']}] {post['title']} → Score: {post['score']}")

        # Save to CSV
        df = pd.DataFrame(sorted_posts)
        os.makedirs("data/reddit", exist_ok=True)
        filename = f"data/reddit/pump_posts_{keyword.lower()}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ Saved hype posts to {filename}")
        print("✅ PUMP SIGNAL DETECTED based on Reddit trends!")

if __name__ == "__main__":
    user_keyword = input("🔍 Enter coin name (e.g. pepe): ").strip()
    detect_pump(keyword=user_keyword, hours=6, debug=True)
