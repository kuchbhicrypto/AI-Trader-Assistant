''' first code by chatgpt

# scripts/train_trend_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load and preprocess
df = pd.read_csv('data/trend_data.csv')
df = df[['Close', 'Volume', 'RSI', 'MACD', 'Target']]

scaler = StandardScaler()
X = scaler.fit_transform(df.drop('Target', axis=1))
y = df['Target'].values

# Convert to LSTM shape [samples, time_steps, features]
X_lstm = []
y_lstm = []
n_steps = 10  # past 10 days
for i in range(n_steps, len(X)):
    X_lstm.append(X[i-n_steps:i])
    y_lstm.append(y[i])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(3, activation='softmax'))  # 3 classes: Buy, Hold, Sell

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save('models/trend_lstm_model.h5')
print("[✓] LSTM Model Trained and Saved.")

''' 
'''
# scripts/train_trend_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load and preprocess
df = pd.read_csv('data/trend_data.csv')

# Drop any unnamed columns like 'Unnamed: 0' or non-numeric accidentally included
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df[['Close', 'Volume', 'RSI', 'MACD', 'Target']]  # Ensure only numeric features

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(df.drop('Target', axis=1))
y = df['Target'].values

# Convert to LSTM shape [samples, time_steps, features]
X_lstm = []
y_lstm = []
n_steps = 10  # past 10 days
for i in range(n_steps, len(X)):
    X_lstm.append(X[i-n_steps:i])
    y_lstm.append(y[i])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(3, activation='softmax'))  # 3 classes: Buy, Hold, Sell

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model
import os
os.makedirs('models', exist_ok=True)
model.save('models/trend_lstm_model.h5')
print("[✓] LSTM Model Trained and Saved.")
'''



# scripts/train_trend_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Load and clean data
df = pd.read_csv('data/trend_data.csv')
df = df[['Close', 'Volume', 'RSI', 'MACD', 'Target']]
df.dropna(inplace=True)

# Features & Labels
X_raw = df.drop('Target', axis=1).values
y = df['Target'].values

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Create LSTM sequences
n_steps = 10
X_seq, y_seq = [], []
for i in range(n_steps, len(X_scaled)):
    X_seq.append(X_scaled[i-n_steps:i])
    y_seq.append(y[i])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model and scaler
os.makedirs('models', exist_ok=True)
model.save('models/trend_lstm_model.h5')
pd.to_pickle(scaler, 'models/scaler.pkl')
print("[✓] Model and scaler saved.")
