
# 🌍 AQI Predictor Dashboard (Karachi)


Real-time monitoring and **3-day forecasting of Air Quality Index (AQI)** using **Machine Learning** and **Deep Learning** models.  
The dashboard visualizes **historical trends**, performs **EDA**, and provides **real-time predictions** for Karachi, Pakistan.  

👉 **Live Dashboard:** [View on Streamlit](https://aqiprediction-byquntaraashraf.streamlit.app/)  

---

## 📌 Overview  

Air pollution is a critical health concern worldwide. This project focuses on **Karachi, Pakistan** and provides:  
- 📡 Real-time AQI data collection via **OpenWeatherMap API**  
- 🔍 Data preprocessing & feature engineering  
- 🤖 Predictions using **XGBoost**, **LightGBM**, and **LSTM**  
- 📊 Model evaluation (MAE, RMSE, R²)  
- 🌍 Interactive **Streamlit Dashboard**  
- 🔄 **Auto data refresh & prediction update every hour** via **GitHub Actions**  

---

## 🛠️ Tech Stack  

- **Frontend**: Streamlit  
- **Data**: OpenWeatherMap API  
- **Models**: XGBoost, LightGBM, LSTM (Keras/TensorFlow)  
- **Data Processing**: Pandas, NumPy, Scikit-learn  
- **Visualization**: Matplotlib, Seaborn  
- **CI/CD**: GitHub Actions  
- **Deployment**: Streamlit Cloud  

---

## 📊 Features  

✅ Fetch **real-time AQI** data (updates hourly)  
✅ Store data in `aqi_features.csv`  
✅ Train & evaluate **3 predictive models**  
✅ Forecast **next 72 hours (3 days)** AQI  
✅ Dashboard Sections:  
- **Historical Data** – Explore past AQI trends  
- **EDA** – Visualizations & outlier detection  
- **Forecast** – Future AQI predictions  
- **Evaluation** – Compare ML/DL model performance  

---

## ⚙️ How AQI is Calculated  

We use **US-EPA PM2.5 breakpoints** to compute AQI:  

```python
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
````

---

## 📂 Project Structure

```
AQIPrediction/
│── data/                        # CSV files (historical + processed data)
│   ├── aqi_features.csv
│── models/                      # Trained models
│   ├── xgboost_model.joblib
│   ├── lightgbm_model.joblib
│   ├── lstm_model.keras
│── predictions/                 # Forecast results
│── scripts/                     # Project scripts
│   ├── app.py                   # Streamlit Dashboard
│   ├── fetch_hourly_historical_data.py
│   ├── datapreprocessing.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── predict_aqi.py
│   └── EDA.py
│── .github/workflows/           # CI/CD workflows
│   └── update.yml
│── requirements.txt             # Dependencies
│── README.md                    # Documentation
│── .env                         # API key (ignored in repo)
```

---

## 🚀 Running Locally

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/QuntaraMuhammadAshraf23/AQIPrediction.git
cd AQIPrediction
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Add API Key

Create `.env` in root:

```
OWM_API_KEY=your_openweathermap_api_key
```

### 4️⃣ Run the Dashboard

```bash
streamlit run scripts/app.py
```

Now visit 👉 `http://localhost:8501`

---

## 📈 Model Performance

| Model    | MAE    | RMSE   | R²     |
| -------- | ------ | ------ | ------ |
| XGBoost  | 0.0459 | 0.0622 | 1.0000 |
| LightGBM | 0.3150 | 0.8331 | 0.9989 |
| LSTM     | 5.9038 | 8.8290 | 0.8727 |

✅ **XGBoost performed the best**, followed by LightGBM.
⚠️ LSTM needs larger dataset & tuning.

---

## 🔄 CI/CD Automation

Using **GitHub Actions** (`.github/workflows/update.yml`):

* Fetch live AQI data **hourly**
* Update dataset
* Generate new forecasts
* Commit changes automatically

---

## 📍 Future Improvements

* Multi-city AQI comparison
* Add weather features (temp, humidity, wind) for better accuracy
* Deploy on **Docker + AWS/GCP/Azure**
* Add **alerts for dangerous AQI levels**

---
## 📜 License  

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---
## 👩‍💻 Author

**Quntara Muhammad Ashraf**
🌐 [GitHub](https://github.com/QuntaraMuhammadAshraf23)

---

