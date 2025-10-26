# 🌧️ Rain Prediction App using XGBoost (Streamlit Cloud)

This project predicts **rainfall** using a trained **XGBoost Regressor (XGBRegressor)** model built on historical weather data.  
It includes a **Streamlit web application** that allows users to manually input weather features — both numeric and categorical — and get instant rainfall predictions.

---

## 📊 Dataset Information

The dataset used for model training contains **weather observations** such as temperature, humidity, visibility, pressure, and categorical summaries.

| Column Name | Description | Data Type |
|--------------|-------------|------------|
| **Formatted Date** | Combined date & time (e.g. `"2012-05-01 00:00:00"`) | datetime |
| **Summary** | General weather condition summary | categorical |
| **Precip Type** | Type of precipitation (rain/snow/none) | categorical |
| **Temperature (C)** | Recorded temperature in Celsius | float |
| **Apparent Temperature (C)** | Perceived temperature in Celsius | float |
| **Humidity** | Relative humidity (0–1 scale) | float |
| **Wind Speed (km/h)** | Wind speed in km/h | float |
| **Wind Bearing (degrees)** | Wind direction (0–360°) | int |
| **Visibility (km)** | Visibility distance in kilometers | float |
| **Cloud Cover** | Cloud cover percentage (0–100%) | int |
| **Pressure (millibars)** | Atmospheric pressure | float |
| **Daily Summary** | Summary of the day’s weather | categorical |

### 🧾 Data Preprocessing
During model training:
- Missing values were handled (e.g., filling categorical NaNs with mode)
- `Formatted Date` was converted or encoded as needed
- Categorical features were encoded using an **encoder**
- Continuous features were normalized/scaled
- Feature transformation (if any) was handled using a **ColumnTransformer**

All preprocessing components were serialized along with the model.

---

## 🤖 Model Details

The model used is an **XGBoost Regressor (XGBRegressor)** trained to predict rainfall intensity or related continuous target values.

The following objects were saved together in a single pickle file:

```python
with open('rain_prediction_model.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)     # Trained XGBRegressor model
    pickle.dump(encoder, f)      # Encoder for categorical variables
    pickle.dump(scaling, f)      # Scaler for normalization
    pickle.dump(transform, f)    # Transformer (if any)
