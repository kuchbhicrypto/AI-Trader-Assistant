# app.py

import streamlit as st
import os

st.set_page_config(page_title="AI Trader Assistant", layout="wide")
st.title("📈 AI-Trader Assistant Dashboard")

st.markdown("Choose a module to explore:")

options = [
    "1️⃣ Trend Prediction (LSTM)",
    "2️⃣ News Sentiment Analyzer",
    "3️⃣ Chart Pattern Recognition",
    "4️⃣ Indicator Fusion",
    "5️⃣ Whale Detector",
    "6️⃣ Economic Event Impact",
    "🧠 All-in-One Signal View"
]

choice = st.selectbox("📂 Select Module", options)

# 🚀 Module Runners
if choice.startswith("1"):
    os.system("python scripts/predict_trend.py")
    st.success("✅ Trend Prediction Executed")

elif choice.startswith("2"):
    st.info("🔍 Run: `python scripts/fetch_news.py` and then `news_sentiment.py`")
    st.code("python scripts/fetch_news.py\npython scripts/news_sentiment.py")

elif choice.startswith("3"):
    st.info("🧠 CNN Pattern Predictor Loaded")
    os.system("python scripts/predict_pattern.py")

elif choice.startswith("4"):
    os.system("python scripts/predict_fusion.py")
    st.success("✅ Fusion Model Prediction Done")

elif choice.startswith("5"):
    os.system("python scripts/whale_detector.py")
    st.success("✅ Whale Detection Done")

elif choice.startswith("6"):
    os.system("python scripts/predict_event_impact.py")
    st.success("✅ Event Impact Prediction Done")

elif choice.startswith("🧠"):
    st.markdown("This will combine output from all models in future version.")

st.markdown("---")
st.caption("Built with ❤️ by kuchbhicrypto (GitHub)")
