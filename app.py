import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ----- 1. Load Pickled Objects in Correct Order -----
with open("rain_prediction_model.pkl", "rb") as f:
    scaling = pickle.load(f)       # StandardScaler
    transform = pickle.load(f)     # PowerTransformer
    encoder = pickle.load(f)       # OneHotEncoder
    best_xgb = pickle.load(f)      # XGBRegressor or XGBClassifier

# ----- 2. Define Features List -----
feature_names = [
    'Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
    'Wind Bearing (degrees)', 'Visibility (km)', 'Pressure (millibars)',
    'day_of_year_sin', 'day_of_year_cos', 'hour_sin', 'hour_cos',
    'Summary_Clear', 'Summary_Foggy', 'Summary_Mostly Cloudy', 'Summary_Others', 'Summary_Overcast', 'Summary_Partly Cloudy',
    'Daily Summary_Foggy in the morning.', 'Daily Summary_Foggy overnight.', 'Daily Summary_Foggy starting in the evening.',
    'Daily Summary_Foggy starting overnight continuing until morning.', 'Daily Summary_Foggy until morning.',
    'Daily Summary_Mostly cloudy starting in the morning.', 'Daily Summary_Mostly cloudy starting overnight.',
    'Daily Summary_Mostly cloudy throughout the day.', 'Daily Summary_Mostly cloudy until night.',
    'Daily Summary_Others', 'Daily Summary_Overcast throughout the day.',
    'Daily Summary_Partly cloudy starting in the afternoon continuing until evening.',
    'Daily Summary_Partly cloudy starting in the afternoon.',
    'Daily Summary_Partly cloudy starting in the morning continuing until evening.',
    'Daily Summary_Partly cloudy starting in the morning continuing until night.',
    'Daily Summary_Partly cloudy starting in the morning.',
    'Daily Summary_Partly cloudy starting overnight.',
    'Daily Summary_Partly cloudy throughout the day.',
    'Daily Summary_Partly cloudy until evening.',
    'Daily Summary_Partly cloudy until night.'
]

summary_choices = ["Clear", "Foggy", "Mostly Cloudy", "Others", "Overcast", "Partly Cloudy"]
daily_summary_choices = [
    "Foggy in the morning.", "Foggy overnight.", "Foggy starting in the evening.",
    "Foggy starting overnight continuing until morning.", "Foggy until morning.",
    "Mostly cloudy starting in the morning.", "Mostly cloudy starting overnight.",
    "Mostly cloudy throughout the day.", "Mostly cloudy until night.", "Others",
    "Overcast throughout the day.", "Partly cloudy starting in the afternoon continuing until evening.",
    "Partly cloudy starting in the afternoon.",
    "Partly cloudy starting in the morning continuing until evening.",
    "Partly cloudy starting in the morning continuing until night.",
    "Partly cloudy starting in the morning.", "Partly cloudy starting overnight.",
    "Partly cloudy throughout the day.", "Partly cloudy until evening.", "Partly cloudy until night."
]

# ----- 3. Helper for Cyclical Features -----
def encode_cyclical(val, max_val):
    radians = 2 * np.pi * val / max_val
    return np.sin(radians), np.cos(radians)

# ----- 4. Streamlit UI for Inputs -----
st.title("Rain Prediction App")

temperature = st.number_input("Temperature (°C)", value=25.0)
apparent_temp = st.number_input("Apparent Temperature (°C)", value=26.0)
humidity = st.slider("Humidity (0-1)", 0.0, 1.0, 0.5)
windspeed = st.number_input("Wind Speed (km/h)", value=10.0)
wind_bearing = st.slider("Wind Bearing (degrees)", 0, 360, 180)
visibility = st.number_input("Visibility (km)", value=10.0)
pressure = st.number_input("Pressure (millibars)", value=1010.0)
date = st.date_input("Date")
hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
summary = st.selectbox("Summary", summary_choices)
daily_summary = st.selectbox("Daily Summary", daily_summary_choices)

# ----- 5. Cyclical Time Feature Engineering -----
day_of_year = pd.to_datetime(date).dayofyear
day_sin, day_cos = encode_cyclical(day_of_year, 365)
hour_sin, hour_cos = encode_cyclical(hour, 24)

# ----- 6. Build Main Input DataFrame -----
input_dict = {
    'Temperature (C)': [temperature],
    'Apparent Temperature (C)': [apparent_temp],
    'Humidity': [humidity],
    'Wind Speed (km/h)': [windspeed],
    'Wind Bearing (degrees)': [wind_bearing],
    'Visibility (km)': [visibility],
    'Pressure (millibars)': [pressure],
    'day_of_year_sin': [day_sin],
    'day_of_year_cos': [day_cos],
    'hour_sin': [hour_sin],
    'hour_cos': [hour_cos],
    'Summary': [summary],
    'Daily Summary': [daily_summary]
}
input_df = pd.DataFrame(input_dict)

# ----- 7. OneHot Encoding for Categoricals -----
cat_col = ['Summary', 'Daily Summary']
encoded_array = encoder.transform(input_df[cat_col])
encoded_cols = encoder.get_feature_names_out(cat_col)
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=input_df.index)
input_df = pd.concat([input_df.drop(columns=cat_col), encoded_df], axis=1)

# ----- 8. Fill Missing Features, Reorder to Model Spec -----
for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[feature_names]

# ----- 9. Power Transform (Numerical Columns) -----
num_col = ['Humidity', 'Wind Speed (km/h)', 'Visibility (km)']
input_df[num_col] = transform.transform(input_df[num_col])

# ----- 10. Standard Scaling (Numerical Columns) -----
scale_col = [
    'Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
    'Wind Speed (km/h)', 'Wind Bearing (degrees)',
    'Visibility (km)', 'Pressure (millibars)'
]
input_df[scale_col] = scaling.transform(input_df[scale_col])

# ----- 11. Predict -----
if st.button("Predict Rain?"):
    pred = best_xgb.predict(input_df)[0]
    label = "Snow" if pred == 1 else "Rain"
    st.success(f"Prediction: **{label}**")

