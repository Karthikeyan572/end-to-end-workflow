import streamlit as st
import pandas as pd
import pickle
import numpy as np
from xgboost import XGBClassifier

# --------------------------
# Load trained model & encoder
# --------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open('xgb_model.pkl', 'rb'))
    encoder = pickle.load(open('encoder.pkl', 'rb'))
    feature_names = pickle.load(open('feature_names.pkl', 'rb'))  # saved during training
    return model, encoder, feature_names

model, encoder, feature_names = load_artifacts()

st.title("🌦️ Rain Type Prediction App")
st.markdown("Predict the precipitation type using weather parameters.")

# --------------------------
# Get user inputs
# --------------------------
st.header("Enter Weather Details")

col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", value=20.0)
    apparent_temp = st.number_input("Apparent Temperature (°C)", value=19.0)
    humidity = st.slider("Humidity", 0.0, 1.0, 0.7)
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
    wind_bearing = st.number_input("Wind Bearing (°)", value=180.0)

with col2:
    visibility = st.number_input("Visibility (km)", value=10.0)
    pressure = st.number_input("Pressure (millibars)", value=1015.0)
    summary = st.selectbox(
        "Summary",
        ['Partly Cloudy', 'Mostly Cloudy', 'Overcast', 'Clear', 'Foggy', 'Others']
    )
    daily_summary = st.selectbox(
        "Daily Summary",
        [
            'Mostly cloudy throughout the day.', 'Partly cloudy throughout the day.',
            'Overcast throughout the day.', 'Foggy in the morning.',
            'Partly cloudy until night.', 'Others'
        ]
    )

# --------------------------
# Create DataFrame
# --------------------------
input_dict = {
    'Temperature (C)': [temperature],
    'Apparent Temperature (C)': [apparent_temp],
    'Humidity': [humidity],
    'Wind Speed (km/h)': [wind_speed],
    'Wind Bearing (degrees)': [wind_bearing],
    'Visibility (km)': [visibility],
    'Pressure (millibars)': [pressure],
    'Summary': [summary],
    'Daily Summary': [daily_summary]
}

input_df = pd.DataFrame(input_dict)

# --------------------------
# Encode categorical columns
# --------------------------
cat_col = ['Summary', 'Daily Summary']

try:
    encoded_array = encoder.transform(input_df[cat_col])
    encoded_cols = encoder.get_feature_names_out(cat_col)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=input_df.index)

    # Replace original categorical columns with encoded ones
    input_df = pd.concat([input_df.drop(columns=cat_col), encoded_df], axis=1)

    # Handle missing columns (those present in training but not in current input)
    missing_cols = [col for col in feature_names if col not in input_df.columns]
    for col in missing_cols:
        input_df[col] = 0

    # Reorder columns to match model training
    input_df = input_df[feature_names]

except Exception as e:
    st.error(f"Encoding Error: {e}")
    st.stop()

# --------------------------
# Predict
# --------------------------
if st.button("Predict Precipitation Type"):
    try:
        prediction = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]

        label_map = {0: 'rain', 1: 'snow'}  # Adjust based on your encoding
        st.success(f"Predicted Precipitation Type: **{label_map.get(prediction, 'Unknown')}**")

        st.write("### Prediction Probabilities:")
        st.write({label_map[i]: round(prob, 3) for i, prob in enumerate(pred_proba)})

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --------------------------
# Debug / Developer Info
# --------------------------
with st.expander("🔍 Debug Info"):
    st.write("Input DataFrame after Encoding:")
    st.dataframe(input_df)
    st.write("Columns count:", len(input_df.columns))
    st.write("Expected count:", len(feature_names))
