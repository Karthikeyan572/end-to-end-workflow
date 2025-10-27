import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===== Load saved artifacts =====
model = pickle.load(open('xgb_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
transformer = pickle.load(open('transform.pkl', 'rb'))
feature_names = pickle.load(open('feature_names.pkl', 'rb'))

st.set_page_config(page_title="Rain or Snow Prediction", page_icon="🌧️", layout="wide")
st.title("🌧️ Rain or Snow Prediction App")

st.write("This app predicts whether it will rain or snow based on weather parameters.")

# ===== Input Section =====
with st.form("weather_form"):
    st.subheader("Enter Weather Conditions")
    col1, col2, col3 = st.columns(3)

    with col1:
        temperature = st.number_input("Temperature (C)", value=10.0)
        apparent_temp = st.number_input("Apparent Temperature (C)", value=10.0)
        humidity = st.number_input("Humidity (0–1)", value=0.8)

    with col2:
        wind_speed = st.number_input("Wind Speed (km/h)", value=5.0)
        wind_bearing = st.number_input("Wind Bearing (°)", value=180.0)
        visibility = st.number_input("Visibility (km)", value=10.0)

    with col3:
        pressure = st.number_input("Pressure (millibars)", value=1012.0)
        day_of_year = st.number_input("Day of Year (1–365)", min_value=1, max_value=365, value=150)
        hour = st.number_input("Hour (0–23)", min_value=0, max_value=23, value=12)

    submitted = st.form_submit_button("Predict")

# ===== Prediction Section =====
if submitted:
    # Create sinusoidal encodings for cyclical features
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Build DataFrame in same order as training
    input_data = pd.DataFrame([[
        temperature, apparent_temp, humidity,
        wind_speed, wind_bearing, visibility,
        pressure, day_of_year_sin, day_of_year_cos,
        hour_sin, hour_cos
    ]], columns=feature_names)

    # Match training: use NumPy arrays (no column names)
    transformed = transformer.transform(input_data.values)
    scaled = scaler.transform(transformed)

    prediction = model.predict(scaled)[0]
    result = "🌧️ Rain" if prediction in [1, "rain", "Rain"] else "❄️ Snow"

    st.success(f"### Predicted Precipitation Type: {result}")
    st.write("#### Input Summary")
    st.dataframe(input_data)

