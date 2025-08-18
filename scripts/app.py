import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import numpy as np

# ---------------------------
# Load Models & Scaler
# ---------------------------
xgb_model = joblib.load("models/xgboost_model.joblib")
lgb_model = joblib.load("models/lightgbm_model.joblib")
lstm_model = load_model("models/lstm_model.keras")
scaler = joblib.load("models/scaler_lstm.pkl")

# ---------------------------
# Load Features / Predictions
# ---------------------------
pred_file = "predictions/aqi_predictions_3days.csv"
df = pd.read_csv(pred_file, parse_dates=["datetime"])

# ---------------------------
# Streamlit App
# ---------------------------
st.set_page_config(page_title="🌍 AQI Forecast Dashboard", layout="wide")
st.title("🌍 Air Quality Index (AQI) Forecast")
st.markdown("This dashboard shows **hourly AQI predictions for the next 3 days** using Machine Learning and Deep Learning models.")

# Show data preview
st.subheader("📄 Forecast Data")
st.dataframe(df.head(20))

# Line chart
st.subheader("📈 AQI Predictions Over Time")
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(df["datetime"], df["aqi_xgb"], label="XGBoost", color="blue")
ax.plot(df["datetime"], df["aqi_lgb"], label="LightGBM", color="green")
ax.plot(df["datetime"], df["aqi_lstm"], label="LSTM", color="orange")
ax.plot(df["datetime"], df["aqi_avg"], label="Average", color="red", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Predicted AQI")
ax.set_title("AQI Forecast (Next 72 Hours)")
ax.legend()
st.pyplot(fig)

# Metrics
st.subheader("📊 Model Comparison (Example Last Point)")
last_row = df.iloc[-1]
st.metric("XGBoost", f"{last_row['aqi_xgb']:.2f}")
st.metric("LightGBM", f"{last_row['aqi_lgb']:.2f}")
st.metric("LSTM", f"{last_row['aqi_lstm']:.2f}")
st.metric("Average", f"{last_row['aqi_avg']:.2f}")
