import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load saved model and transformer
with open('rain_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('pt_transformer.pkl', 'rb') as f:
    pt = pickle.load(f)

st.title("Precipitation Type Prediction App with Yeo-Johnson Transformation")

# User inputs
temp = st.number_input('Temperature (C)', value=10.0, format="%.2f")
app_temp = st.number_input('Apparent Temperature (C)', value=9.0, format="%.2f")

# We will transform these two inputs using Yeo-Johnson
humidity_raw = st.slider('Humidity (0 to 1)', min_value=0.0, max_value=1.0, value=0.7, step=0.01)
wind_speed_raw = st.number_input('Wind Speed (km/h)', value=10.0, format="%.2f")

wind_bearing = st.slider('Wind Bearing (degrees)', 0, 359, 180)
visibility = st.number_input('Visibility (km)', value=10.0, format="%.2f")
pressure = st.number_input('Pressure (millibars)', value=1015.0, format="%.2f")

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

# Apply yeo-johnson transform on Humidity and Wind Speed
transformed = pt.transform(np.array([[humidity_raw, wind_speed_raw]]))
humidity = transformed[0, 0]
wind_speed = transformed[0, 1]

# Assemble features in the same order as training
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

if st.button('Predict Precipitation Type'):
    pred = model.predict(features)[0]
    precip_type = "Snow" if pred >= 0.5 else "Rain"
    st.success(f"Predicted Precipitation Type is: {precip_type}")
