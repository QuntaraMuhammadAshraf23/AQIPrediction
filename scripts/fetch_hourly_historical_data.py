# scripts/fetch_hourly_historical_data.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("OWM_API_KEY")
LAT, LON = 24.8607, 67.0011
CSV_FILE = "data/aqi_features.csv"

def get_unix_timestamp(dt):
    return int(dt.timestamp())

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

def fetch_data(start_dt, end_dt):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    params = {
        "lat": LAT,
        "lon": LON,
        "start": get_unix_timestamp(start_dt),
        "end": get_unix_timestamp(end_dt),
        "appid": API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json().get("list", [])

def fetch_hourly_data(from_date, to_date):
    all_rows = []
    current = from_date

    print(f"📦 Fetching hourly data from {from_date.date()} to {to_date.date()}...")

    while current < to_date:
        batch_end = min(current + timedelta(days=5), to_date)
        print(f"🔄 Fetching: {current} → {batch_end}")

        try:
            data_points = fetch_data(current, batch_end)
        except Exception as e:
            print(f"❌ Error fetching batch: {e}")
            break

        for item in data_points:
            ts = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
            c = item["components"]
            pm25 = c.get("pm2_5", 0)

            row = {
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
                "nh3": c.get("nh3")
            }
            all_rows.append(row)

        current = batch_end

    return all_rows

def save_to_csv(data, filename=CSV_FILE):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = pd.DataFrame(data)
    df = df.sort_values("datetime")  # ensure ascending time order

    # ✅ compute aqi_change_rate
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)

    df.to_csv(filename, index=False)
    print(f"✅ Saved {len(df)} rows to {filename}")


if __name__ == "__main__":
    start_date = datetime(2025, 6, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)

    result = fetch_hourly_data(start_date, end_date)
    if result:
        save_to_csv(result)
    else:
        print("⚠️ No data collected.")
