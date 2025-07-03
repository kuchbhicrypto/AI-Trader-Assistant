# app.py

import streamlit as st
import os
import subprocess
import pandas as pd
from PIL import Image

st.set_page_config(page_title="🧠 AI Trader Assistant", layout="wide")

st.title("📊 AI Trader Assistant 🚀")
st.markdown("Welcome to your smart trading assistant powered by AI. Choose a feature below to run different modules.")

# Utility functions
def run_script(script_path):
    with st.spinner(f"Running `{script_path}`..."):
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        st.code(result.stdout)
        if result.stderr:
            st.error(result.stderr)

def display_csv(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        st.dataframe(df)
    else:
        st.warning(f"File not found: {file_path}")

def show_image_from_folder(folder):
    if not os.path.exists(folder):
        st.warning("Image folder does not exist.")
        return
    images = [f for f in os.listdir(folder) if f.endswith(".png")]
    for img_file in images[:3]:  # Show only 3
        img_path = os.path.join(folder, img_file)
        st.image(Image.open(img_path), caption=img_file, use_column_width=True)


# Sidebar Navigation
option = st.sidebar.selectbox("Choose Module", (
    "📈 Market Trend Prediction",
    "📰 News Sentiment Analyzer",
    "📉 Chart Pattern Detector",
    "📊 Fusion Indicator Builder",
    "⚠️ Event Impact Classifier",
    "🔎 Pattern Recognition Predictor",
    "🧠 Model & Predictions",
    "🛠 About"
))

# 1 - Trend Prediction
if option == "📈 Market Trend Prediction":
    st.header("📈 Market Trend Prediction")
    if st.button("Download + Preprocess Data"):
        run_script("scripts/fetch_data.py")
        display_csv("data/trend_data.csv")

    if st.button("Train LSTM Model"):
        run_script("scripts/train_trend_model.py")

    if st.button("Predict Latest Trend"):
        run_script("scripts/predict_trend.py")

# 2 - News Sentiment
elif option == "📰 News Sentiment Analyzer":
    st.header("📰 Real-Time News Sentiment")
    ticker = st.text_input("Enter Ticker Symbol", "AAPL")
    if st.button("Fetch News"):
        run_script(f"scripts/fetch_news.py")
        display_csv(f"data/news_{ticker}.csv")

    if st.button("Analyze Sentiment"):
        run_script("scripts/news_sentiment.py")
        display_csv(f"data/news_{ticker}_scored.csv")

# 3 - Chart Pattern Detector
elif option == "📉 Chart Pattern Detector":
    st.header("📉 Technical Pattern Detection (via Chart Images)")
    if st.button("Generate Chart Images"):
        run_script("scripts/generate_chart_images.py")
        show_image_from_folder("data/pattern_images/No_Pattern")

    if st.button("Train CNN (Transfer Learning)"):
        run_script("scripts/train_pattern_transfer.py")

    if st.button("Predict Pattern from Sample Image"):
        run_script("scripts/predict_pattern.py")

# 4 - Fusion Indicator Builder
elif option == "📊 Fusion Indicator Builder":
    st.header("📊 Build Multi-Indicator Dataset (RSI + MACD + MA + BB)")
    if st.button("Generate Fusion Features"):
        run_script("scripts/fetch_fusion_data.py")
        display_csv("data/fusion_data.csv")

# 5 - Event Impact Classifier
elif option == "⚠️ Event Impact Classifier":
    st.header("⚠️ Predict Market Reaction to Economic Events")
    if st.button("Train Event Reaction Classifier"):
        run_script("scripts/train_event_reaction.py")
        display_csv("data/labeled_event_data.csv")

    if st.button("Predict Reaction to Sample Event"):
        run_script("scripts/predict_event_impact.py")

# 6 - Pattern Prediction
elif option == "🔎 Pattern Recognition Predictor":
    st.header("🔎 Predict Pattern from Random Image")
    run_script("scripts/predict_pattern.py")

# 7 - Combined Prediction Panel
elif option == "🧠 Model & Predictions":
    st.header("🧠 Full Prediction Panel Coming Soon")
    st.info("🚧 This section will combine trend, sentiment, chart, and event models to give a unified prediction.")
    st.write("Stay tuned!")

# About Section
elif option == "🛠 About":
    st.header("🛠 About the Project")
    st.markdown("""
    **AI Trader Assistant** is a modular system that combines:
    - LSTM-based Trend Prediction 📈
    - Real-time News Sentiment Analysis 📰
    - Chart Pattern Detection using CNNs 📉
    - Technical Indicator Fusion 📊
    - Economic Event Impact Prediction ⚠️
    
    Created by: [@kuchbhicrypto](https://github.com/kuchbhicrypto)
    """)

