import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta, timezone
from keras.models import load_model
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# Constants
HOURS_TO_PREDICT = 72
FUTURE_FILE = "predictions/aqi_predictions_3days.csv"
os.makedirs("predictions", exist_ok=True)

# Load historical data
df = pd.read_csv("data/aqi_features.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
last_row = df.iloc[-1]
last_timestamp = last_row["datetime"]

# Required features
feature_cols = ['hour', 'day', 'month', 'pm25', 'pm10', 'co', 'no2', 'so2', 'o3', 'nh3', 'aqi_change_rate']

# Generate future timestamps
future_timestamps = [last_timestamp + timedelta(hours=i + 1) for i in range(HOURS_TO_PREDICT)]

# Fill future data using last known pollutant values
base_features = {
    "pm25": last_row["pm25"],
    "pm10": last_row["pm10"],
    "co": last_row["co"],
    "no2": last_row["no2"],
    "so2": last_row["so2"],
    "o3": last_row["o3"],
    "nh3": last_row["nh3"],
    "aqi_change_rate": 0  # assumed stable
}

# Build future feature DataFrame
future_data = []
for ts in future_timestamps:
    row = {
        "datetime": ts,
        "hour": ts.hour,
        "day": ts.day,
        "month": ts.month,
        **base_features
    }
    future_data.append(row)

future_df = pd.DataFrame(future_data)
X_future = future_df[feature_cols]

# Load models
xgb_model = joblib.load("models/xgboost_model.joblib")
lgb_model = joblib.load("models/lightgbm_model.joblib")
lstm_model = load_model("models/lstm_model.keras")
scaler = joblib.load("models/scaler_lstm.pkl")

# Predict using LightGBM and XGBoost
future_df["aqi_lgb"] = lgb_model.predict(X_future)
future_df["aqi_xgb"] = xgb_model.predict(X_future)

# Predict using LSTM (requires scaled input and reshaping)
X_scaled = scaler.transform(X_future)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
future_df["aqi_lstm"] = lstm_model.predict(X_lstm).flatten()

# Average prediction
future_df["aqi_avg"] = future_df[["aqi_lgb", "aqi_xgb", "aqi_lstm"]].mean(axis=1)

# Save final output
future_df[["datetime", "aqi_xgb", "aqi_lgb", "aqi_lstm", "aqi_avg"]].to_csv(FUTURE_FILE, index=False)
print(f"✅ Future AQI predictions saved to: {FUTURE_FILE}")
