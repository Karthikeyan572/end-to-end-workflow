import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, date, time

st.set_page_config(page_title="Rain Prediction - XGBRegressor", layout="centered")

st.title("🌧️ Rain Prediction App (XGBRegressor)")
st.write("This app predicts **Precipitation Type (Rain or Snow)** using weather data and your trained XGBoost model.")

# -------------------------------
# Load Model and Transformers
# -------------------------------
@st.cache_resource
def load_pickle_objects(pickle_path="rain_prediction_model.pkl"):
    try:
        with open(pickle_path, "rb") as f:
            best_xgb = pickle.load(f)
            encoder = pickle.load(f)
            scaling = pickle.load(f)
            transform = pickle.load(f)
        return {"model": best_xgb, "encoder": encoder, "scaler": scaling, "transform": transform}
    except Exception as e:
        st.error(f"❌ Error loading pickle file: {e}")
        return None

objs = load_pickle_objects()

# -------------------------------
# User Input Form
# -------------------------------
st.header("🧾 Input Weather Data")

with st.form("input_form"):
    st.subheader("📅 Date & Time Input")
    date_val = st.date_input(
        "Select Date",
        value=datetime.today().date(),
        min_value=date(2000, 1, 1),
        max_value=date.today()
    )
    time_val = st.time_input("Select Time", value=datetime.now().time().replace(microsecond=0))
    formatted_date = datetime.combine(date_val, time_val)

    st.subheader("🌤️ Weather Summary (Categorical Inputs)")
    summary = st.text_input("Summary", value="Overcast")
    daily_summary = st.text_input("Daily Summary", value="Mostly cloudy throughout the day")

    st.subheader("🌡️ Numerical Inputs")
    temp = st.number_input("Temperature (C)", value=20.0, step=0.1)
    app_temp = st.number_input("Apparent Temperature (C)", value=19.5, step=0.1)
    humidity = st.number_input("Humidity (0–1)", value=0.7, step=0.01)
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0, step=0.1)
    wind_bearing = st.slider("Wind Bearing (degrees)", 0, 360, 180)
    visibility = st.number_input("Visibility (km)", value=10.0, step=0.1)
    pressure = st.number_input("Pressure (millibars)", value=1013.25, step=0.1)
    # Cloud Cover dropped in training, not included

    submitted = st.form_submit_button("🔮 Predict Precip Type")

# -------------------------------
# Feature Engineering Function
# -------------------------------
def preprocess_input(df):
    """Replicates transformations from rain_pred.ipynb"""
    df = df.copy()

    # Convert Formatted Date to cyclic features
    df["datetime"] = pd.to_datetime(df["Formatted Date"], utc=True)
    df["DayOfYear"] = df["datetime"].dt.dayofyear
    df["Hour"] = df["datetime"].dt.hour

    df["day_of_year_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    # Drop original date columns
    df.drop(["Formatted Date", "datetime", "DayOfYear", "Hour"], axis=1, inplace=True)

    # Drop Cloud Cover if present
    if "Cloud Cover" in df.columns:
        df.drop("Cloud Cover", axis=1, inplace=True)

    # Handle rare categories in Summary (as done in training)
    if "Summary" in df.columns:
        df["Summary"] = df["Summary"].replace("", "Others")

    return df

# -------------------------------
# Prediction Logic
# -------------------------------
if submitted:
    if objs is None:
        st.error("⚠️ Model not loaded. Ensure 'rain_prediction_model.pkl' is in the same folder.")
    else:
        try:
            # Build input DataFrame
            row = {
                "Formatted Date": formatted_date.strftime("%Y-%m-%d %H:%M:%S"),
                "Summary": summary,
                "Temperature (C)": float(temp),
                "Apparent Temperature (C)": float(app_temp),
                "Humidity": float(humidity),
                "Wind Speed (km/h)": float(wind_speed),
                "Wind Bearing (degrees)": int(wind_bearing),
                "Visibility (km)": float(visibility),
                "Pressure (millibars)": float(pressure),
                "Daily Summary": daily_summary
            }
            input_df = pd.DataFrame([row])

            st.write("### 🧩 Raw Input Data")
            st.dataframe(input_df)

            # Apply feature engineering
            processed_df = preprocess_input(input_df)
            st.write("### 🔧 Processed Data for Model")
            st.dataframe(processed_df)

            # Apply transformation
            transform = objs["transform"]
            model = objs["model"]

            X_transformed = transform.transform(processed_df)

            # Predict
            pred = model.predict(X_transformed)
            prediction = pred[0]

            # Map prediction output (if binary)
            if isinstance(prediction, (int, float)) and prediction in [0, 1]:
                label = "🌧️ Rain" if prediction == 0 else "❄️ Snow"
            else:
                label = f"{prediction:.3f}"

            st.success(f"### Predicted Precipitation Type: {label}")

        except Exception as e:
            st.error(f"🚨 Error during prediction: {e}")
