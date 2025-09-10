import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from keras.callbacks import EarlyStopping
import os

# Load data
df = pd.read_csv("data/aqi_features.csv")

# Add change rate
df["aqi_change_rate"] = df["aqi"].diff().fillna(0)

# Drop datetime
df = df.drop(columns=["datetime"])
df = df.dropna()

# Feature-target split
X = df.drop(columns=["aqi"])
y = df["aqi"]

# Scaling for LSTM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save for future prediction
pd.DataFrame(X_scaled, columns=X.columns).to_csv("data/X_scaled_all.csv", index=False)
scaler_filename = "models/scaler_lstm.pkl"
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, scaler_filename)

# ---------- Model 1: LightGBM ----------
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X, y)
joblib.dump(lgb_model, "models/lightgbm_model.joblib")

# ---------- Model 2: XGBoost ----------
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X, y)
joblib.dump(xgb_model, "models/xgboost_model.joblib")

# ---------- Model 3: LSTM ----------
# Reshape for LSTM: (samples, timesteps, features)
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
model = Sequential()
model.add(Input(shape=(1, X_scaled.shape[1])))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(patience=5, restore_best_weights=True)
model.fit(X_lstm, y, epochs=50, batch_size=16, validation_split=0.2, callbacks=[early_stop])

model.save("models/lstm_model.keras")

print("✅ All models trained and saved.")
