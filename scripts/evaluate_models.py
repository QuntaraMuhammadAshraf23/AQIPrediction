# scripts/evaluate_models.py

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import load_model

# ----------- Load test data -----------
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()  # ensures it's a Series, not DataFrame

# ----------- Load models -----------
xgb_model = joblib.load("models/xgboost_model.joblib")
lgb_model = joblib.load("models/lightgbm_model.joblib")
lstm_model = load_model("models/lstm_model.keras")
scaler = joblib.load("models/scaler_lstm.pkl")

# ----------- Evaluation function -----------
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 {model_name} Performance:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")

# ----------- Evaluate XGBoost -----------
y_pred_xgb = xgb_model.predict(X_test)
evaluate(y_test, y_pred_xgb, "XGBoost")

# ----------- Evaluate LightGBM -----------
y_pred_lgb = lgb_model.predict(X_test)
evaluate(y_test, y_pred_lgb, "LightGBM")

# ----------- Evaluate LSTM -----------
X_test_scaled = scaler.transform(X_test).reshape((X_test.shape[0], 1, X_test.shape[1]))
y_pred_lstm = lstm_model.predict(X_test_scaled).flatten()
evaluate(y_test, y_pred_lstm, "LSTM")
