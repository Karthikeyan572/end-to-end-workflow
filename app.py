import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ======================================================
# 1. LOAD ALL ARTIFACTS (model, encoder, scaler, etc.)
# ======================================================
@st.cache_resource
def load_artifacts():
    with open('artifacts.pkl', 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    encoder = data['encoder']
    scaler = data['scaler']
    transformer = data['transformer']
    return model, encoder, scaler, transformer


model, encoder, scaler, transformer = load_artifacts()

# ======================================================
# 2. STREAMLIT APP CONFIGURATION
# ======================================================
st.title("🌦️ Rain Prediction App (XGBoost)")
st.write("Predicts the type of precipitation using weather data.")

# ======================================================
# 3. USER INPUT FIELDS
# ======================================================
st.sidebar.header("Input Weather Features")

temperature = st.sidebar.number_input("Temperature (°C)", -20.0, 50.0, 20.0)
apparent_temp = st.sidebar.number_input("Apparent Temperature (°C)", -20.0, 50.0, 18.0)
humidity = st.sidebar.slider("Humidity", 0.0, 1.0, 0.5)
wind_speed = st.sidebar.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0)
visibility = st.sidebar.number_input("Visibility (km)", 0.0, 20.0, 10.0)
pressure = st.sidebar.number_input("Pressure (millibars)", 900.0, 1100.0, 1012.0)
cloud_cover = st.sidebar.slider("Cloud Cover (0–1)", 0.0, 1.0, 0.3)

summary = st.sidebar.selectbox(
    "Summary",
    ['Partly Cloudy', 'Mostly Cloudy', 'Overcast', 'Clear', 'Foggy', 'Others']
)

daily_summary = st.sidebar.selectbox(
    "Daily Summary",
    [
        'Mostly cloudy throughout the day.',
        'Partly cloudy throughout the day.',
        'Foggy in the morning.',
        'Overcast throughout the day.',
        'Partly cloudy until night.',
        'Foggy until morning.',
        'Partly cloudy starting in the morning.',
        'Mostly cloudy until night.',
        'Foggy overnight.',
        'Others'
    ]
)

# ======================================================
# 4. CREATE INPUT DATAFRAME
# ======================================================
input_df = pd.DataFrame({
    'Temperature (C)': [temperature],
    'Apparent Temperature (C)': [apparent_temp],
    'Humidity': [humidity],
    'Wind Speed (km/h)': [wind_speed],
    'Visibility (km)': [visibility],
    'Pressure (millibars)': [pressure],
    'Cloud Cover': [cloud_cover],
    'Summary': [summary],
    'Daily Summary': [daily_summary]
})

st.write("### 🔍 Input Data")
st.dataframe(input_df)

# ======================================================
# 5. APPLY TRANSFORMATIONS
# ======================================================
# Apply the same transformations used in the notebook
try:
    # Apply transformations in the same order as training
    transformed_data = transformer.transform(input_df)
    transformed_df = pd.DataFrame(
        transformed_data,
        columns=transformer.get_feature_names_out()
    )

    # Apply scaling
    scaled_data = scaler.transform(transformed_df)

except Exception as e:
    st.error(f"Error during preprocessing: {e}")
    st.stop()

# ======================================================
# 6. MAKE PREDICTION
# ======================================================
if st.button("Predict Precipitation Type"):
    try:
        prediction = model.predict(scaled_data)
        predicted_label = encoder.inverse_transform(prediction)[0]

        st.success(f"🌧️ Predicted Precipitation Type: **{predicted_label}**")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
