import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

# Load the saved XGB model
with open('rain_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Precipitation Type Prediction App")

# Manual input for continuous variables via sliders or number inputs
temp = st.number_input('Temperature (C)', value=10.0, format="%.2f")
app_temp = st.number_input('Apparent Temperature (C)', value=9.0, format="%.2f")
humidity = st.slider('Humidity (0 to 1)', min_value=0.0, max_value=1.0, value=0.7, step=0.01)
wind_speed = st.number_input('Wind Speed (km/h)', value=10.0, format="%.2f")
wind_bearing = st.slider('Wind Bearing (degrees)', 0, 359, 180)
visibility = st.number_input('Visibility (km)', value=10.0, format="%.2f")
pressure = st.number_input('Pressure (millibars)', value=1015.0, format="%.2f")

# Date and Time input widgets
date_input = st.date_input("Select date")
time_input = st.time_input("Select time")

# Convert datetime input into cyclical features
dt = datetime.combine(date_input, time_input)
day_of_year = dt.timetuple().tm_yday
hour = dt.hour

day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365.25)
day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365.25)
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# Prepare feature vector as DataFrame (matching training order)
features = pd.DataFrame([[
    temp,
    app_temp,
    humidity,
    wind_speed,
    wind_bearing,
    visibility,
    pressure,
    day_of_year_sin,
    day_of_year_cos,
    hour_sin,
    hour_cos
]], columns=[
    'Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
    'Wind Speed (km/h)', 'Wind Bearing (degrees)', 'Visibility (km)',
    'Pressure (millibars)', 'day_of_year_sin', 'day_of_year_cos',
    'hour_sin', 'hour_cos'
])

# Prediction button
if st.button('Predict Precipitation Type'):
    pred = model.predict(features)[0]
    precip_type = "Snow" if pred >= 0.5 else "Rain"
    st.success(f"Predicted Precipitation Type is: {precip_type}")

