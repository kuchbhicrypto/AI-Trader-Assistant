# 💹 AI-Trader-Assistant

An intelligent trading assistant powered by AI, built to assist traders in making smarter and faster decisions using real-time market data, news, chart patterns, and technical indicators.

---

## 🚀 Project Overview

This multi-phase AI model aims to simulate a smart trader's toolkit:

- 📈 Predict price trends (LSTM)
- 📰 Analyze market sentiment from news/Twitter
- 📊 Detect candlestick chart patterns using CNN
- 🔁 Fuse multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
- 🐋 Detect unusual activity (whale moves, insider trades)
- 📅 Predict impact of economic events
- 🧠 Real-time pump detection from social media
- 💼 Optimize portfolios based on predictions

---

## ✅ Phases Completed

| Phase | Feature                              | Status     |
| ----- | ------------------------------------ | ---------- |
| 1     | Trend Prediction (LSTM)              | ✅ Done    |
| 2     | News Sentiment Analysis (BERT)       | ✅ Done    |
| 3     | Pattern Recognition (CNN)            | ✅ Done    |
| 4     | Indicator Fusion (RandomForest+LSTM) | 🛠️ Ongoing |
| 5     | Whale/Big Order Detection            | ⏳ Coming  |
| 6     | Economic Event Predictor             | ⏳ Coming  |
| 7     | Twitter Pump Detection Bot           | ⏳ Coming  |
| 8     | AI Portfolio Optimizer               | ⏳ Coming  |

---

## 📁 Folder Structure

AI-Trader-Assistant/
│
├── data/ # Contains all generated datasets & images
├── models/ # Trained ML/DL models saved here
├── scripts/ # All feature scripts (fetch, train, predict)
├── README.md # Project overview

---

## 🧠 How to Use

1. **Clone this repo**  
   `git clone https://github.com/kuchbhicrypto/AI-Trader-Assistant.git`

2. **Install dependencies**  
   Create a `requirements.txt` like:

yfinance
pandas
ta
transformers
tensorflow
scikit-learn
mplfinance
newspaper3k
opencv-python

Then run:  
`pip install -r requirements.txt`

3. **Run any phase step-by-step**  
   Example:

python scripts/fetch_data.py
python scripts/train_trend_model.py
python scripts/predict_trend.py

---

## 📸 Preview (Optional)

> Add screenshots or sample predictions here for Phase 1–3 output.

---

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- HuggingFace Transformers
- Scikit-learn
- YFinance
- TA-Lib
- OpenCV
- MobileNetV2 (Transfer Learning)

---

## 📜 License

MIT License — feel free to use, modify, and contribute!

---

## 🙌 Contribution

Found a bug? Have ideas? Feel free to [open an issue](https://github.com/kuchbhicrypto/AI-Trader-Assistant/issues) or submit a PR.

---

## 🔗 Connect

Created with 💻 by [@kuchbhicrypto](https://github.com/kuchbhicrypto)
