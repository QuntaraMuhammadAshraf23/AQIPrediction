# 🌍 AQI Prediction Project

This project predicts the **Air Quality Index (AQI)** for the **next 3 days (hourly)** using **Machine Learning and Deep Learning models**.
It also includes a **Streamlit-based web application** for interactive AQI forecasting and visualization.

---

## 📌 Features

* 📊 **Data Preprocessing** of AQI dataset (pollutants, timestamps, AQI rate change).
* 🌫️ **AQI Calculation** using **EPA PM2.5 breakpoints**.
* 🤖 **Model Training & Evaluation** with:

  * **XGBoost** – Best performance with R² = 1.0.
  * **LightGBM** – Fast and efficient boosting model.
  * **LSTM** – Time-series deep learning model.
* ⏳ **Future Predictions**: Forecast AQI for the next **72 hours (3 days)**.
* 💻 **Streamlit App**: Upload data, run predictions, and view interactive visualizations.

---

## 📂 Project Structure

```
AQI-Predictor/
│── data/                     # Raw and processed AQI data
│── models/                   # Trained models (XGBoost, LightGBM, LSTM, Scaler)
│── predictions/              # Output CSV with 3-day AQI forecast
│── scripts/
│   ├── train_models.py       # Train and save models
│   ├── predict_future.py     # Predict AQI for next 3 days
│   ├── evaluate_models.py    # Evaluate models (MAE, RMSE, R²)
│   └── app.py                # Streamlit web app
│── README.md                 # Project documentation
│── requirements.txt          # Python dependencies
```

---

## 🧮 AQI Calculation

The AQI is calculated using **EPA standards**.
For **PM2.5**:

$$
I_p = \frac{(I_{HI} - I_{LO})}{(C_{HI} - C_{LO})} \times (C_p - C_{LO}) + I_{LO}
$$

Example:
If **PM2.5 = 40 µg/m³**, AQI ≈ **108** (*Unhealthy for Sensitive Groups*).

---

## 🚀 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/AQI-Predictor.git
cd AQI-Predictor
```

### 2️⃣ Create Virtual Environment & Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate # On Linux/Mac

pip install -r requirements.txt
```

### 3️⃣ Train Models

```bash
python scripts/train_models.py
```

### 4️⃣ Predict Future AQI (Next 3 Days)

```bash
python scripts/predict_future.py
```

Output is saved in:

```
predictions/aqi_predictions_3days.csv
```

### 5️⃣ Evaluate Models

```bash
python scripts/evaluate_models.py
```

### 6️⃣ Run the Streamlit App

```bash
streamlit run scripts/app.py
```

---

## 📊 Model Performance

| Model       | MAE    | RMSE   | R²         |
| ----------- | ------ | ------ | ---------- |
| **XGBoost** | 0.0459 | 0.0622 | **1.0000** |
| LightGBM    | 0.3150 | 0.8331 | 0.9989     |
| LSTM        | 5.9038 | 8.8290 | 0.8727     |

✅ **XGBoost performed the best** with the highest accuracy.

---

## 📸 Streamlit App Preview

* Upload new data.
* Get **72-hour AQI predictions**.
* View **interactive line charts** for AQI trends.

---

## 🏆 Conclusion

* AQI is calculated using **EPA PM2.5 breakpoints**.
* **XGBoost** gives the most accurate predictions.
* The **Streamlit App** makes predictions accessible and user-friendly.

---

## 👨‍💻 Author

**Quntara Muhammad Ashraf**

---

