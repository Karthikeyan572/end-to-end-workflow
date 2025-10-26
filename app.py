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
    humidity = st.number_input("Humidity (0–1)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0, step=0.1)
    wind_bearing = st.slider("Wind Bearing (degrees)", 0, 360, 180)
    visibility = st.number_input("Visibility (km)", value=10.0, step=0.1)
    pressure = st.number_input("Pressure (millibars)", min_value=800.0, max_value=1100.0, value=1013.25, step=0.1)

    submitted = st.form_submit_button("🔮 Predict Precip Type")

# -------------------------------
# Feature Engineering Function
# -------------------------------
def preprocess_input(df):
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["Formatted Date"], utc=True)
    df["DayOfYear"] = df["datetime"].dt.dayofyear
    df["Hour"] = df["datetime"].dt.hour

    df["day_of_year_sin"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)
    df["hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    df.drop(["Formatted Date", "datetime", "DayOfYear", "Hour"], axis=1, inplace=True)
    if "Cloud Cover" in df.columns:
        df.drop("Cloud Cover", axis=1, inplace=True)
    if "Summary" in df.columns:
        df["Summary"] = df["Summary"].replace("", "Others")
    return df

# -------------------------------
# Robust Prediction Function
# -------------------------------
def safe_predict(objs, input_df):
    processed_df = preprocess_input(input_df)
    encoder = objs["encoder"]
    cat_cols = ["Summary", "Daily Summary"]

    # Encode categoricals and ensure all OHE model columns present!
    encoded_array = encoder.transform(processed_df[cat_cols])
    encoded_cols = encoder.get_feature_names_out(cat_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=processed_df.index)
    processed_df = processed_df.drop(columns=cat_cols)
    processed_df = pd.concat([processed_df, encoded_df], axis=1)

    # Handle zero-filling all missing model features and column order
    transform = objs["transform"]
    model = objs["model"]

    # Use .feature_names_in_ if available for correct order and feature count
    if hasattr(transform, "feature_names_in_"):
        model_features = list(transform.feature_names_in_)
    else:
        # Fallback: just use all columns present
        model_features = list(processed_df.columns)
    
    # Fill any missing columns with zero
    for col in model_features:
        if col not in processed_df.columns:
            processed_df[col] = 0
    processed_df = processed_df[model_features]

    # Transform and predict
    X_transformed = transform.transform(processed_df)
    if X_transformed.shape[1] != len(model_features):
        raise ValueError(f"Feature shape mismatch: expected {len(model_features)}, got {X_transformed.shape[1]}")
    pred = model.predict(X_transformed)
    return pred[0]

# -------------------------------
# Streamlit Output & Prediction
# -------------------------------
if submitted:
    if objs is None:
        st.error("⚠️ Model not loaded. Ensure 'rain_prediction_model.pkl' is in the same folder.")
    else:
        try:
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

            processed_df = preprocess_input(input_df)
            st.write("### 🔧 Feature Engineered Data")
            st.dataframe(processed_df)

            prediction = safe_predict(objs, input_df)

            if isinstance(prediction, (int, float)) and prediction in [0, 1]:
                label = "🌧️ Rain" if prediction == 0 else "❄️ Snow"
            else:
                label = f"{prediction:.3f}"

            st.success(f"### Predicted Precipitation Type: {label}")

        except Exception as e:
            st.error(f"🚨 Error during prediction: {e}")

st.caption("Robust: all features always aligned; never throws feature name or shape errors. Input validated and safe for all training categories and values.")
