# 🤖 AI-Trader-Assistant

An AI-powered assistant for traders that predicts market moves using LSTM, Transformers, CNNs, and Ensemble Models.

> 🔬 This project is being built in structured **phases**, each one adding a key intelligent module for smarter trading decisions.

---

## 📈 Features by Phase

### ✅ Phase 1: Trend Prediction (LSTM)

- Predicts whether to **Buy**, **Sell**, or **Hold** a stock.
- Uses historical price, RSI, MACD, and Volume.
- Deep learning model: **LSTM (Long Short-Term Memory)**.

### ✅ Phase 2: News Sentiment Analyzer (BERT)

- Fetches latest news headlines using Google News RSS.
- Uses **BERT** model to analyze sentiment (positive/negative).
- Provides average sentiment score for each asset.

### ✅ Phase 3: Chart Pattern Recognition (CNN)

- Generates candlestick chart images.
- Trains a **CNN (Convolutional Neural Network)** using **MobileNetV2** (transfer learning).
- Detects classic chart patterns from image data.

### ✅ Phase 4: Smart Indicator Fusion

- Fuses multiple indicators: **RSI**, **MACD**, **Moving Averages**, **Bollinger Bands**, **Volatility**.
- Ensemble model (LSTM + Random Forest – coming up next).
- Provides market sentiment prediction with stronger reliability.

---

## 🗂️ Folder Structure

````bash
AI-Trader-Assistant/
├── data/                   # Market data, news, pattern images
│   ├── trend_data.csv
│   ├── news_AAPL.csv
│   └── pattern_images/
├── models/                 # Saved models (LSTM, CNN, Scalers, etc.)
├── scripts/                # All feature scripts (fetching, training, predicting)
├── requirements.txt        # All Python dependencies
└── README.md               # Project overview (this file)


---


🚀 Installation
Clone the repo:

bash
Copy
Edit
git clone https://github.com/kuchbhicrypto/AI-Trader-Assistant.git
cd AI-Trader-Assistant
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt


---


📦 Requirements
All Python packages used:

makefile
Copy
Edit
yfinance
pandas
numpy
ta==0.11.0
tensorflow
scikit-learn
matplotlib
mplfinance
transformers
torch
newspaper3k
beautifulsoup4
requests
opencv-python


---


📩 Author
Made with ❤️ by @kuchbhicrypto
For questions or contributions, feel free to open issues or submit a PR.

---

### ✅ What to do next:

- Save this content as `README.md` in your repo root.
- Add your `requirements.txt` if not already done.
- Commit and push:

```bash
git add README.md requirements.txt
git commit -m "Added README and requirements"
git push origin main
````
