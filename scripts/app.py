import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from datetime import timedelta, datetime, timezone
from streamlit_autorefresh import st_autorefresh
import requests
import os
from dotenv import load_dotenv

# ---------------- Page Config ----------------
st.set_page_config(page_title="AQI Dashboard", page_icon="🌍", layout="wide")
st_autorefresh(interval=3600000, limit=None, key="aqi_refresh")  # auto-refresh every hour
st.title("🌍 Air Quality Index (AQI) Dashboard By Quntara Ashraf")
st.markdown("Real-time Monitoring, Forecasting, Model Evaluation and Exploratory Data Analysis of AQI")

# ---------------- Load Env ----------------
load_dotenv()
API_KEY = os.getenv("OWM_API_KEY")
LAT, LON = 24.8607, 67.0011
CSV_FILE = "data/aqi_features.csv"

# ---------------- Helper Functions ----------------
def compute_aqi_pm25(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for Clow, Chigh, Ilow, Ihigh in breakpoints:
        if Clow <= pm25 <= Chigh:
            return round(((Ihigh - Ilow) / (Chigh - Clow)) * (pm25 - Clow) + Ilow, 2)
    return 0.0

def fetch_live_aqi():
    """Fetch latest AQI from OWM API and return new row as DataFrame"""
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": LAT, "lon": LON, "appid": API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    item = response.json()["list"][0]

    ts = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
    c = item["components"]
    pm25 = c.get("pm2_5", 0)

    new_row = {
        "datetime": ts.isoformat(),
        "hour": ts.hour,
        "day": ts.day,
        "month": ts.month,
        "aqi": compute_aqi_pm25(pm25),
        "pm25": pm25,
        "pm10": c.get("pm10"),
        "co": c.get("co"),
        "no2": c.get("no2"),
        "so2": c.get("so2"),
        "o3": c.get("o3"),
        "nh3": c.get("nh3"),
    }
    return pd.DataFrame([new_row])

# ---------------- Load Models ----------------
xgb_model = joblib.load("models/xgboost_model.joblib")
lgb_model = joblib.load("models/lightgbm_model.joblib")
lstm_model = load_model("models/lstm_model.keras")
scaler = joblib.load("models/scaler_lstm.pkl")

# ---------------- Load & Update Data ----------------
df = pd.read_csv(CSV_FILE)
df["datetime"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")

# fetch & append new live data
try:
    new_data = fetch_live_aqi()
    new_data["datetime"] = pd.to_datetime(new_data["datetime"], utc=True)
    df = pd.concat([df, new_data], ignore_index=True).drop_duplicates(subset="datetime", keep="last")
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    df.to_csv(CSV_FILE, index=False)
except Exception as e:
    st.warning(f"⚠️ Could not fetch live data: {e}")

# ---------------- Sidebar Navigation ----------------
section = st.sidebar.radio("Go to", ["Historical Data", "Forecast", "Evaluation", "EDA"])

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
        start_date = pd.to_datetime(date_range[0], utc=True)
        end_date = pd.to_datetime(date_range[1], utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
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
    future_df["aqi_xgb"] = xgb_model.predict(X_future)
    future_df["aqi_lgb"] = lgb_model.predict(X_future)
    X_lstm_scaled = scaler.transform(X_future).reshape((X_future.shape[0], 1, X_future.shape[1]))
    future_df["aqi_lstm"] = lstm_model.predict(X_lstm_scaled).flatten()
    future_df["aqi_avg"] = future_df[["aqi_xgb", "aqi_lgb", "aqi_lstm"]].mean(axis=1)

    model_choice = st.radio("Select Model", ["XGBoost", "LightGBM", "LSTM", "Ensemble Average"])
    pred_column = {
        "XGBoost": "aqi_xgb",
        "LightGBM": "aqi_lgb",
        "LSTM": "aqi_lstm",
        "Ensemble Average": "aqi_avg"
    }[model_choice]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["datetime"].tail(72), df["aqi"].tail(72), label="Historical AQI", color="blue")
    ax.plot(future_df["datetime"], future_df[pred_column], label=f"Forecasted AQI ({model_choice})", color="red")
    ax.set_xlabel("Datetime (UTC)")
    ax.set_ylabel("AQI")
    ax.legend()
    st.pyplot(fig)

    st.dataframe(future_df[["datetime", pred_column]].rename(columns={pred_column: "Predicted AQI"}))

# ==================================================================
# 📊 Evaluation
# ==================================================================
elif section == "Evaluation":
    st.subheader("📊 Model Evaluation Results")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    split_idx = int(0.8 * len(df))
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df.drop(columns=["datetime", "aqi"]), train_df["aqi"]
    X_test, y_test = test_df.drop(columns=["datetime", "aqi"]), test_df["aqi"]

    def evaluate(y_true, y_pred, model_name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        st.markdown(f"**{model_name}**")
        st.write(f"📌 MAE:  {mae:.4f}")
        st.write(f"📌 RMSE: {rmse:.4f}")
        st.write(f"📌 R²:   {r2:.4f}")
        st.write("---")

    evaluate(y_test, xgb_model.predict(X_test), "XGBoost")
    evaluate(y_test, lgb_model.predict(X_test), "LightGBM")

    X_test_scaled = scaler.transform(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_pred_lstm = lstm_model.predict(X_test_scaled).flatten()
    evaluate(y_test, y_pred_lstm, "LSTM")

# ==================================================================
# 🔬 Exploratory Data Analysis (EDA)
# ==================================================================
elif section == "EDA":
    st.subheader("🔬 Exploratory Data Analysis (EDA)")

    # Dataset overview
    st.markdown("### 📌 Dataset Overview")
    st.write("Shape:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Missing values:", df.isnull().sum().to_dict())
    st.dataframe(df.head())

    # Statistical summary
    st.markdown("### 📊 Statistical Summary")
    st.dataframe(df.describe())

    # Distribution of AQI
    st.markdown("### 🌍 Distribution of AQI")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(df["aqi"], kde=True, bins=30, color="skyblue", ax=ax)
    ax.set_title("Distribution of AQI")
    st.pyplot(fig)

    # AQI over time
    st.markdown("### ⏳ AQI Over Time")
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df["datetime"], df["aqi"], label="AQI", color="red")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("AQI")
    ax.legend()
    st.pyplot(fig)

    # Interactive pollutant trend
    pollutants = ["pm25", "pm10", "co", "no2", "so2", "o3", "nh3"]
    pollutant_choice = st.selectbox("Select a pollutant to visualize", pollutants)

    st.markdown(f"### 💨 {pollutant_choice.upper()} Trend Over Time")
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(df["datetime"], df[pollutant_choice], label=pollutant_choice, color="green")
    ax.legend()
    st.pyplot(fig)

    # Correlation heatmap
    st.markdown("### 🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Time-based analysis
    st.markdown("### 🕒 AQI Trend by Time")
    time_option = st.radio("Select time granularity", ["Hour of Day", "Day of Week", "Month"])

    if time_option == "Hour of Day":
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(x="hour", y="aqi", data=df, ci=None, marker="o", ax=ax)
        ax.set_title("Average AQI by Hour of Day")
        st.pyplot(fig)

    elif time_option == "Day of Week":
        df["dayofweek"] = df["datetime"].dt.day_name()
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x="dayofweek", y="aqi", data=df, ax=ax, order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        ax.set_title("Average AQI by Day of Week")
        st.pyplot(fig)

    elif time_option == "Month":
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x="month", y="aqi", data=df, ax=ax)
        ax.set_title("Average AQI by Month")
        st.pyplot(fig)

    # Outliers detection toggle
    if st.checkbox("🚨 Show Outlier Analysis"):
        st.markdown("#### Boxplot of AQI")
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(x=df["aqi"], ax=ax)
        st.pyplot(fig)

        st.markdown("#### Scatterplot of AQI Over Time")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.scatter(df.index, df["aqi"], alpha=0.5)
        ax.set_title("Scatterplot of AQI Over Time")
        st.pyplot(fig)
