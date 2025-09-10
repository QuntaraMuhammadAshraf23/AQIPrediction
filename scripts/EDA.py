import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load dataset
df = pd.read_csv("data/aqi_features.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

df.head()

# Shape and column info
print("Shape:", df.shape)
print("Columns:", df.columns)

# Data types and missing values
print(df.info())
print(df.isnull().sum())

# Statistical summary
print(df.describe())

plt.figure(figsize=(8,5))
sns.histplot(df["aqi"], kde=True, bins=30, color="skyblue")
plt.title("Distribution of AQI")
plt.show()

plt.figure(figsize=(15,6))
plt.plot(df["datetime"], df["aqi"], label="AQI", color="red")
plt.xlabel("Datetime")
plt.ylabel("AQI")
plt.title("AQI Over Time")
plt.legend()
plt.show()

pollutants = ["pm25", "pm10", "co", "no2", "so2", "o3", "nh3"]

plt.figure(figsize=(15,8))
for pollutant in pollutants:
    plt.plot(df.index, df[pollutant], label=pollutant)

plt.legend()
plt.title("Pollutant Trends Over Time")
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Average AQI by hour of day
plt.figure(figsize=(8,5))
sns.lineplot(x="hour", y="aqi", data=df, ci=None, marker="o")
plt.title("Average AQI by Hour of Day")
plt.show()

# Monthly average
plt.figure(figsize=(8,5))
sns.barplot(x="month", y="aqi", data=df)
plt.title("Average AQI by Month")
plt.show()

# Outliers detection
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df["aqi"])
plt.title("Boxplot of AQI (Outliers Detection)")
plt.show()
plt.scatter(df.index, df["aqi"], alpha=0.5)
plt.title("Scatterplot of AQI Over Time")
plt.show()

