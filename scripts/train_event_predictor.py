# scripts/train_event_predictor.py

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import joblib

def train_event_model(input_file='data/labeled_event_data.csv', model_path='models/event_reaction_model.pkl'):
    if not os.path.exists(input_file):
        print("❌ Labeled event data not found.")
        return

    df = pd.read_csv(input_file)
    df.dropna(subset=['Actual', 'Previous'], inplace=True)

    # Convert K to float (e.g., "1910K" -> 1910000)
    def parse_number(x):
        try:
            return float(x.replace('K','')) * 1000 if 'K' in x else float(x)
        except:
            return np.nan

    df['Actual'] = df['Actual'].apply(parse_number)
    df['Previous'] = df['Previous'].apply(parse_number)
    df['Forecast'] = df['Forecast'].apply(parse_number)

    # Calculate Surprise (Actual - Forecast)
    df['Surprise'] = df['Actual'] - df['Forecast']
    df['Delta_vs_Prev'] = df['Actual'] - df['Previous']

    df.dropna(subset=['Surprise', 'Delta_vs_Prev'], inplace=True)

    # Encode event type
    # Encode event type
    encoder = OneHotEncoder(sparse_output=False)
    event_encoded = encoder.fit_transform(df[['Event']])
    event_labels = encoder.get_feature_names_out(['Event'])

    features = pd.DataFrame(event_encoded, columns=event_labels)
    features['Surprise'] = df['Surprise']
    features['Delta_vs_Prev'] = df['Delta_vs_Prev']

    y = df['Label']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

    # Train Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n📊 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, model_path)
    joblib.dump(encoder, 'models/event_encoder.pkl')

    print(f"\n✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_event_model()
