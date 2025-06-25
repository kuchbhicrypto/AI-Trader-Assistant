# scripts/predict_event_impact.py

import pandas as pd
import numpy as np
import joblib
import os

def parse_number(x):
    """Convert strings like '219K' or '1.2M' to float."""
    try:
        x = str(x).upper().replace(',', '')
        if 'K' in x:
            return float(x.replace('K', '')) * 1000
        elif 'M' in x:
            return float(x.replace('M', '')) * 1_000_000
        return float(x)
    except Exception as e:
        print(f"❌ Failed to parse number: {x} → {e}")
        return np.nan

def predict_event_impact(event, actual, forecast, previous):
    model_path = 'models/event_reaction_model.pkl'
    encoder_path = 'models/event_encoder.pkl'

    # === Load model and encoder ===
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        print("❌ Model or encoder not found. Make sure Phase 6 training is complete.")
        return

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # === Validate and parse input ===
    actual_num = parse_number(actual)
    forecast_num = parse_number(forecast)
    previous_num = parse_number(previous)

    if np.isnan(actual_num) or np.isnan(forecast_num) or np.isnan(previous_num):
        print("❌ Invalid input: one or more numbers could not be parsed.")
        return

    surprise = actual_num - forecast_num
    delta = actual_num - previous_num

    # === Encode event ===
    try:
        event_df = pd.DataFrame([[event]], columns=['Event'])
        encoded_event = encoder.transform(event_df)
    except Exception as e:
        print(f"❌ Event encoding failed: {e}")
        return

    # === Prepare full input with feature names ===
    encoded_df = pd.DataFrame(encoded_event, columns=encoder.get_feature_names_out(['Event']))
    encoded_df['Surprise'] = surprise
    encoded_df['Delta_vs_Prev'] = delta

    # === Predict ===
    try:
        pred = model.predict(encoded_df)[0]
        prob = model.predict_proba(encoded_df)[0]
        confidence = np.max(prob)

        print("\n🧠 Economic Event AI Prediction")
        print(f"📅 Event: {event}")
        print(f"📊 Actual: {actual_num} | Forecast: {forecast_num} | Previous: {previous_num}")
        print(f"🧮 Surprise: {surprise:.2f} | Δ vs Prev: {delta:.2f}")
        print(f"📈 Predicted Market Reaction: **{pred}** ({confidence:.2f} confidence)")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

if __name__ == "__main__":
    # 🔁 Test with any event
    predict_event_impact(
        event="Initial Jobless Claims",
        actual="219K",
        forecast="224K",
        previous="220K"
    )
