'''
# scripts/predict_trend.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('data/trend_data.csv')
df = df[['Close', 'Volume', 'RSI', 'MACD', 'Target']]

# Preprocess
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('Target', axis=1))
n_steps = 10
X_lstm = [X[i-n_steps:i] for i in range(n_steps, len(X))]

X_lstm = np.array(X_lstm)
model = load_model('models/trend_lstm_model.h5')

predictions = model.predict(X_lstm)
labels = np.argmax(predictions, axis=1)

last_pred = labels[-1]
label_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}

print(f"\n📈 Latest Market Signal: **{label_map.get(last_pred)}**")
'''

# scripts/predict_trend.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

# Load data
df = pd.read_csv('data/trend_data.csv')
df = df[['Close', 'Volume', 'RSI', 'MACD', 'Target']]
df.dropna(inplace=True)

# Load scaler and model
from sklearn.preprocessing import StandardScaler
scaler = pd.read_pickle('models/scaler.pkl')
model = load_model('models/trend_lstm_model.h5')

# Preprocess latest data
X_raw = df.drop('Target', axis=1).values
X_scaled = scaler.transform(X_raw)

# LSTM input shape
n_steps = 10
if len(X_scaled) < n_steps:
    raise ValueError("Not enough data points for prediction.")

X_input = np.array([X_scaled[-n_steps:]])  # Shape: (1, n_steps, n_features)

# Predict
pred = model.predict(X_input, verbose=0)
pred_label = np.argmax(pred)

label_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
print(f"\n📈 Latest Market Signal: **{label_map[pred_label]}**")
