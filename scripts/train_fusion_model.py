# scripts/train_fusion_model.py

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load and clean data
df = pd.read_csv('data/fusion_data.csv')
df.dropna(inplace=True)

# Keep only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Ensure Target is present
if 'Target' not in numeric_cols:
    raise ValueError("❌ 'Target' column missing or non-numeric in fusion_data.csv.")

# Feature selection
features = [col for col in numeric_cols if col != 'Target']
X = df[features].values
y = df['Target'].values

# Ensure no NaN values
assert not np.isnan(X).any(), "❌ X contains NaN"
assert not np.isnan(y).any(), "❌ y contains NaN"

# Split for Random Forest
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

print("\n🌲 Random Forest Results:")
print(classification_report(y_test_rf, rf_model.predict(X_test_rf)))

# Save Random Forest model
os.makedirs('models', exist_ok=True)
joblib.dump(rf_model, 'models/fusion_rf_model.pkl')
print("✅ Random Forest model saved.")

# Normalize for LSTM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences
n_steps = 10
X_seq, y_seq = [], []
for i in range(n_steps, len(X_scaled)):
    X_seq.append(X_scaled[i - n_steps:i])
    y_seq.append(y[i])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Convert labels for categorical crossentropy
y_seq_cat = to_categorical(y_seq, num_classes=3)

# Train/Test Split for LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_seq, y_seq_cat, test_size=0.2, random_state=42)

# Build LSTM Model
lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(3, activation='softmax')
])

lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=15, batch_size=32, validation_split=0.1)

# Save LSTM Model & Scaler
lstm_model.save('models/fusion_lstm_model.h5')
joblib.dump(scaler, 'models/fusion_scaler.pkl')
print("✅ LSTM model and scaler saved.")
