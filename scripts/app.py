import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import timedelta

# ---------------- Page Config ----------------
st.set_page_config(page_title="AQI Dashboard", page_icon="🌍", layout="wide")
st.title("🌍 Air Quality Index (AQI) Dashboard")
st.markdown("Real-time Monitoring, Forecasting, and Model Evaluation")

# ---------------- Load Models ----------------
xgb_model = joblib.load("models/xgboost_model.joblib")
lgb_model = joblib.load("models/lightgbm_model.joblib")
lstm_model = load_model("models/lstm_model.keras")
scaler = joblib.load("models/scaler_lstm.pkl")

# ---------------- Load Data ----------------
df = pd.read_csv("data/aqi_features.csv")
df["datetime"] = pd.to_datetime(df["datetime"])
df["datetime"] = df["datetime"].dt.tz_localize(None)  # remove timezone

# ---------------- Sidebar Navigation ----------------
section = st.sidebar.radio("Go to", ["Historical Data", "Forecast", "Evaluation"])

# ==================================================================
# 📜 Historical Data
# ==================================================================
if section == "Historical Data":
    st.subheader("📜 Historical AQI Data")

    min_date, max_date = df["datetime"].min(), df["datetime"].max()
    date_range = st.date_input(
        "Select Date Range",
        [min_date.date(), max_date.date()],
        min_value=min_date.date(),
        max_value=max_date.date()
    )

    if len(date_range) == 2:
        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered_df = df[(df["datetime"] >= start_date) & (df["datetime"] <= end_date)]
    else:
        filtered_df = df.copy()

    st.line_chart(filtered_df.set_index("datetime")["aqi"])
    st.dataframe(filtered_df)

# ==================================================================
# 📈 Forecast
# ==================================================================
elif section == "Forecast":
    st.subheader("📈 Forecasted AQI (Next 3 Days)")

    last_timestamp = df["datetime"].iloc[-1]
    HOURS_TO_PREDICT = 72
    future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(HOURS_TO_PREDICT)]

    # Take last row for pollution features
    last_row = df.iloc[-1]
    future_df = pd.DataFrame({
        "datetime": future_timestamps,
        "hour": [dt.hour for dt in future_timestamps],
        "day": [dt.day for dt in future_timestamps],
        "month": [dt.month for dt in future_timestamps],
        "pm25": [last_row["pm25"]]*HOURS_TO_PREDICT,
        "pm10": [last_row["pm10"]]*HOURS_TO_PREDICT,
        "co": [last_row["co"]]*HOURS_TO_PREDICT,
        "no2": [last_row["no2"]]*HOURS_TO_PREDICT,
        "so2": [last_row["so2"]]*HOURS_TO_PREDICT,
        "o3": [last_row["o3"]]*HOURS_TO_PREDICT,
        "nh3": [last_row["nh3"]]*HOURS_TO_PREDICT,
        "aqi_change_rate": [0]*HOURS_TO_PREDICT
    })

    X_future = future_df.drop(columns=["datetime"])

    # Predictions
    future_df["aqi_xgb"] = xgb_model.predict(X_future)
    future_df["aqi_lgb"] = lgb_model.predict(X_future)
    X_lstm_scaled = scaler.transform(X_future).reshape((X_future.shape[0], 1, X_future.shape[1]))
    future_df["aqi_lstm"] = lstm_model.predict(X_lstm_scaled).flatten()
    future_df["aqi_avg"] = future_df[["aqi_xgb", "aqi_lgb", "aqi_lstm"]].mean(axis=1)

    # Model Selection
    model_choice = st.radio("Select Model", ["XGBoost", "LightGBM", "LSTM", "Ensemble Average"])
    pred_column = {
        "XGBoost": "aqi_xgb",
        "LightGBM": "aqi_lgb",
        "LSTM": "aqi_lstm",
        "Ensemble Average": "aqi_avg"
    }[model_choice]

    # Plot forecast
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["datetime"].tail(72), df["aqi"].tail(72), label="Historical AQI", color="blue")
    ax.plot(future_df["datetime"], future_df[pred_column], label=f"Forecasted AQI ({model_choice})", color="red")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("AQI")
    ax.legend()
    st.pyplot(fig)

    # Show forecasted table
    st.dataframe(future_df[["datetime", pred_column]].rename(columns={pred_column: "Predicted AQI"}))

# ==================================================================
# 📊 Evaluation
# ==================================================================
elif section == "Evaluation":
    st.subheader("📊 Model Evaluation Results")

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # Train-test split (last 20% for testing)
    split_idx = int(0.8 * len(df))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df.drop(columns=["datetime", "aqi"]), train_df["aqi"]
    X_test, y_test = test_df.drop(columns=["datetime", "aqi"]), test_df["aqi"]

    # Evaluation function
    def evaluate(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        st.markdown(f"**{model_name}**")
        st.write(f"📌 MAE:  {mae:.4f}")
        st.write(f"📌 RMSE: {rmse:.4f}")
        st.write(f"📌 R²:   {r2:.4f}")
        st.write("---")

    # Evaluate models
    evaluate(y_test, xgb_model.predict(X_test), "XGBoost")
    evaluate(y_test, lgb_model.predict(X_test), "LightGBM")

    # LSTM evaluation
    X_test_scaled = scaler.transform(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_pred_lstm = lstm_model.predict(X_test_scaled).flatten()
    evaluate(y_test, y_pred_lstm, "LSTM")
