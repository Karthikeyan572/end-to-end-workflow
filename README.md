app link : https://end-to-end-workflow-7u9h38cnxfwwhwcd5dfxpk.streamlit.app/
# 🌧️ Rain Prediction App using XGBoost + Streamlit

A fully interactive web app that predicts **Precipitation Type (Rain or Snow)** using weather features.  
It reproduces the preprocessing pipeline and trained model from the original `rain_pred.ipynb` notebook.

---

## 📊 Dataset Overview

The dataset contains historical weather data with the following features:

| Column | Description |
|---------|-------------|
| Formatted Date | Date and time of observation |
| Summary | Short description of weather (e.g., Overcast, Foggy) |
| Temperature (C) | Recorded temperature |
| Apparent Temperature (C) | Perceived temperature |
| Humidity | Relative humidity (0–1) |
| Wind Speed (km/h) | Wind velocity |
| Wind Bearing (degrees) | Wind direction in degrees |
| Visibility (km) | Visibility distance |
| Pressure (millibars) | Atmospheric pressure |
| Daily Summary | Daily textual summary of weather |
| Precip Type | Target variable — predicted by model |

---

## ⚙️ Preprocessing Pipeline (from `rain_pred.ipynb`)

1. **Datetime Engineering**  
   - Convert “Formatted Date” into cyclic sine/cosine features for seasonal representation.
   - Extract Day of Year and Hour of Day.
2. **Feature Drop**  
   - Dropped “Cloud Cover”.
3. **Rare Category Handling**  
   - “Summary” values with frequency <1000 replaced with `"Others"`.
4. **Scaling and Encoding**  
   - Continuous features scaled with StandardScaler.
   - Categorical features encoded.
5. **Model**  
   - Trained using **XGBRegressor** to predict `Precip Type`.

---

## 🧠 Model Storage (`rain_prediction_model.pkl`)

The pickle file stores:
1. `best_xgb` → trained model  
2. `encoder` → for categorical features  
3. `scaling` → for numerical features  
4. `transform` → preprocessing transformer

---

## 💻 Streamlit App Features

- Manual numeric + categorical inputs  
- Calendar + Clock input for datetime  
- Automatic date-to-cyclic transformation  
- Predicts **Rain / Snow** without requiring Precip Type input  
- Restricts date to year **≥ 2000**

---

## 🗂️ Folder Structure

