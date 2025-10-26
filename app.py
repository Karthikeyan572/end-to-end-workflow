import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, date, time

st.set_page_config(page_title="Rain Prediction - XGBRegressor", layout="centered")

st.title("🌧️ Rain Prediction App (XGBRegressor)")
st.write("Provide input values and click **Predict** to estimate rainfall using your trained XGB model.")

@st.cache_resource
def load_objects(pickle_path="rain_prediction_model.pkl"):
    objs = {}
    try:
        with open(pickle_path, "rb") as f:
            best_xgb = pickle.load(f)
            encoder = pickle.load(f)
            scaling = pickle.load(f)
            transform = pickle.load(f)
        objs["model"] = best_xgb
        objs["encoder"] = encoder
        objs["scaler"] = scaling
        objs["transform"] = transform
        return objs
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
        return None

objs = load_objects()

COLUMNS = [
    "Formatted Date", "Summary", "Precip Type", "Temperature (C)",
    "Apparent Temperature (C)", "Humidity", "Wind Speed (km/h)",
    "Wind Bearing (degrees)", "Visibility (km)", "Cloud Cover",
    "Pressure (millibars)", "Daily Summary"
]

st.header("🧾 Input Form")

with st.form("input_form"):
    st.subheader("Date & Time Input")
    # ✅ Restrict date selection to year 2000 and onwards
    date_val = st.date_input(
        "Date (calendar)",
        value=datetime.today().date(),
        min_value=date(2000, 1, 1),
        max_value=date.today()
    )
    time_val = st.time_input("Time (clock)", value=datetime.now().time().replace(microsecond=0))
    formatted_date = datetime.combine(date_val, time_val).strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Formatted Date: {formatted_date}")

    st.subheader("Categorical Inputs")
    summary = st.text_input("Summary", value="")
    precip_type = st.selectbox("Precip Type", ["", "rain", "snow", "none"])
    daily_summary = st.text_input("Daily Summary", value="")

    st.subheader("Numeric Inputs")
    temp = st.number_input("Temperature (C)", value=20.0, step=0.1)
    app_temp = st.number_input("Apparent Temperature (C)", value=20.0, step=0.1)
    humidity = st.number_input("Humidity (0-1)", value=0.75, step=0.01)
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0, step=0.1)
    wind_bearing = st.slider("Wind Bearing (degrees)", 0, 360, 180)
    visibility = st.number_input("Visibility (km)", value=10.0, step=0.1)
    cloud_cover = st.slider("Cloud Cover (0-100%)", 0, 100, 50)
    pressure = st.number_input("Pressure (millibars)", value=1013.25, step=0.1)

    submit = st.form_submit_button("🔮 Predict")

if submit:
    if objs is None:
        st.error("Model file not loaded! Ensure 'rain_prediction_model.pkl' exists in your folder.")
    else:
        try:
            row = {
                "Formatted Date": formatted_date,
                "Summary": summary,
                "Precip Type": precip_type,
                "Temperature (C)": float(temp),
                "Apparent Temperature (C)": float(app_temp),
                "Humidity": float(humidity),
                "Wind Speed (km/h)": float(wind_speed),
                "Wind Bearing (degrees)": int(wind_bearing),
                "Visibility (km)": float(visibility),
                "Cloud Cover": int(cloud_cover),
                "Pressure (millibars)": float(pressure),
                "Daily Summary": daily_summary
            }
            input_df = pd.DataFrame([row], columns=COLUMNS)
            st.write("### Input Data")
            st.dataframe(input_df.T)

            transform = objs.get("transform")
            model = objs.get("model")

            if transform is not None:
                X = transform.transform(input_df)
            else:
                df_copy = input_df.copy()
                for col in df_copy.select_dtypes(include=["object"]).columns:
                    df_copy[col] = df_copy[col].astype("category").cat.codes
                X = df_copy.values

            preds = model.predict(X)
            pred_val = float(np.array(preds).ravel()[0])
            st.success(f"Predicted Value: {pred_val:.4f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
