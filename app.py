import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# -------------------------------
# Load artifacts
# -------------------------------
with open('artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
scaler = artifacts['scaler']
transformer = artifacts['transformer']

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Rain or Snow Predictor", layout="centered")
st.title("🌦️ Precipitation Type Prediction App")
st.write("Predict whether it will **Rain** or **Snow** based on given weather conditions.")
st.markdown("---")

# -------------------------------
# User Inputs
# -------------------------------
st.sidebar.header("Input Weather Features")

temperature = st.sidebar.number_input("Temperature (°C)", value=10.0)
apparent_temp = st.sidebar.number_input("Apparent Temperature (°C)", value=9.0)
humidity = st.sidebar.number_input("Humidity (0–1)", value=0.75, min_value=0.0, max_value=1.0, step=0.01)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", value=10.0)
wind_bearing = st.sidebar.number_input("Wind Bearing (°)", value=180, min_value=0, max_value=359, step=1)
visibility = st.sidebar.number_input("Visibility (km)", value=10.0)
pressure = st.sidebar.number_input("Pressure (millibars)", value=1015.0)
datetime_input = st.sidebar.text_input("Datetime (YYYY-MM-DD HH:MM:SS)", value="2006-04-01 00:00:00")

# -------------------------------
# Feature Engineering
# -------------------------------
try:
    dt = pd.to_datetime(datetime_input, utc=True)
except Exception:
    st.error("❌ Invalid datetime format! Please use YYYY-MM-DD HH:MM:SS")
    st.stop()

day_of_year = dt.timetuple().tm_yday
hour = dt.hour

# Cyclical Encoding
day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365.25)
day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365.25)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# -------------------------------
# Create DataFrame
# -------------------------------
input_df = pd.DataFrame({
    'Temperature (C)': [temperature],
    'Apparent Temperature (C)': [apparent_temp],
    'Humidity': [humidity],
    'Wind Speed (km/h)': [wind_speed],
    'Wind Bearing (degrees)': [wind_bearing],
    'Visibility (km)': [visibility],
    'Pressure (millibars)': [pressure],
    'day_of_year_sin': [day_of_year_sin],
    'day_of_year_cos': [day_of_year_cos],
    'hour_sin': [hour_sin],
    'hour_cos': [hour_cos]
})

# -------------------------------
# Apply Preprocessing (Final Fix)
# -------------------------------
num_col = ['Humidity', 'Wind Speed (km/h)', 'Visibility (km)']
scale_col = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
             'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)']

try:
    # Apply PowerTransformer column by column
    for col in num_col:
        input_df[col] = transformer.transform(input_df[[col]])

    # Apply StandardScaler column by column
    for col in scale_col:
        input_df[col] = scaler.transform(input_df[[col]])

except ValueError as e:
    st.error(f"⚠️ Preprocessing error: {e}")
    st.stop()

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Precipitation Type"):
    try:
        pred_value = model.predict(input_df)[0]
        label = "🌧️ Rain" if pred_value < 0.5 else "❄️ Snow"

        st.subheader("✅ Prediction Result:")
        st.success(f"Predicted Precipitation Type: **{label}**")

        st.markdown("---")
        st.write("### Input Summary (After Preprocessing):")
        st.dataframe(input_df)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
