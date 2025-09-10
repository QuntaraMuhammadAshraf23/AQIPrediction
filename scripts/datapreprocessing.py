import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load the dataset
df = pd.read_csv("data/aqi_features.csv")

# Add 'aqi_change_rate' if not present
if "aqi_change_rate" not in df.columns:
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)

# Drop 'datetime' column
if "datetime" in df.columns:
    df = df.drop(columns=["datetime"])

# Drop any rows with missing values
df = df.dropna()

# Separate features and target
X = df.drop(columns=["aqi"])
y = df["aqi"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save raw (unscaled) features for LightGBM & XGBoost
os.makedirs("data", exist_ok=True)
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)

# Save targets
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

# Standardize features for LSTM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaled features for LSTM
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("data/X_test_scaled.csv", index=False)

print("✅ Preprocessing complete with `aqi_change_rate`, raw and scaled features saved.")
