# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

st.set_page_config(page_title="Rain prediction - XGBRegressor", layout="centered")

st.title("Rain prediction (XGBRegressor)")
st.write("Simple input form for single-row manual prediction. Provide values and click Predict.")

# --- load model and preprocessing objects ---
@st.cache_resource
def load_objects(pickle_path="rain_prediction_model.pkl"):
    objs = {}
    try:
        with open(pickle_path, "rb") as f:
            # You told me you dumped objects in this order:
            # best_xgb, encoder, scaling, transform
            best_xgb = pickle.load(f)
            try:
                encoder = pickle.load(f)
            except Exception:
                encoder = None
            try:
                scaling = pickle.load(f)
            except Exception:
                scaling = None
            try:
                transform = pickle.load(f)
            except Exception:
                transform = None

        objs["model"] = best_xgb
        objs["encoder"] = encoder
        objs["scaler"] = scaling
        objs["transform"] = transform
        return objs
    except FileNotFoundError:
        st.error(f"Pickle file not found: {pickle_path}. Upload it to the app folder.")
        return None
    except Exception as e:
        st.error(f"Error loading pickle file: {e}")
        return None

objs = load_objects()

# Provided dataset columns and types from your message
COLUMNS = [
    "Formatted Date",    # object
    "Summary",           # object (categorical/text)
    "Precip Type",       # object (categorical)
    "Temperature (C)",   # float
    "Apparent Temperature (C)", # float
    "Humidity",          # float
    "Wind Speed (km/h)", # float
    "Wind Bearing (degrees)", # int
    "Visibility (km)",   # float
    "Cloud Cover",       # int
    "Pressure (millibars)", # float
    "Daily Summary"      # object (categorical/text)
]

# --- Helper to extract categories from encoder if possible ---
def get_categories_from_encoder(encoder, feature_name):
    """
    Try to extract categories for a column name from a fitted OneHotEncoder or similar.
    This is heuristic: many encoders expose .feature_names_in_ or .categories_.
    """
    if encoder is None:
        return None
    try:
        # If encoder is a sklearn OneHotEncoder
        if hasattr(encoder, "categories_"):
            # categories_ is a list aligned with categorical feature order.
            # We have no reliable mapping of which entry belongs to which column name,
            # so return the flattened unique values across all categories as a fallback.
            all_vals = []
            for cat in encoder.categories_:
                all_vals.extend(list(cat))
            uniq = sorted(set(all_vals))
            return uniq
        # If encoder is a dict-like mapping feature -> categories
        if isinstance(encoder, dict):
            return encoder.get(feature_name, None)
        # If encoder is a pipeline with named step 'encoder'
        if hasattr(encoder, "named_steps"):
            for name, step in encoder.named_steps.items():
                if hasattr(step, "categories_"):
                    all_vals = []
                    for cat in step.categories_:
                        all_vals.extend(list(cat))
                    return sorted(set(all_vals))
    except Exception:
        return None
    return None

# --- Build input widgets ---
st.header("Input features")

with st.form("input_form"):
    # Formatted Date: date + time
    st.subheader("Date & time")
    date_val = st.date_input("Date (calendar)", value=datetime.today().date())
    time_val = st.time_input("Time (clock)", value=datetime.now().time().replace(microsecond=0))
    # Combine into a string representation similar to many CSV exports
    datetime_val = datetime.combine(date_val, time_val)
    # Format as ISO-like string (you can adjust to match training format if needed)
    formatted_date_str = datetime_val.strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Formatted Date: {formatted_date_str}")

    st.subheader("Categorical fields")

    # Try to populate picklists for the categorical columns using encoder if possible
    encoder = objs["encoder"] if objs else None
    precip_cats = get_categories_from_encoder(encoder, "Precip Type")
    summary_cats = get_categories_from_encoder(encoder, "Summary")
    daily_cats = get_categories_from_encoder(encoder, "Daily Summary")

    if precip_cats:
        precip_default = precip_cats[0] if len(precip_cats) else ""
        precip_type = st.selectbox("Precip Type", options=[""] + precip_cats, index=0, help="Select precipitation type")
    else:
        precip_type = st.selectbox("Precip Type", options=["rain", "snow", "none", ""] , index=0)

    if summary_cats:
        summary = st.selectbox("Summary", options=[""] + summary_cats, index=0)
    else:
        summary = st.text_input("Summary (short categorical text)", value="")

    if daily_cats:
        daily_summary = st.selectbox("Daily Summary", options=[""] + daily_cats, index=0)
    else:
        daily_summary = st.text_input("Daily Summary (short text)", value="")

    st.subheader("Numeric fields")

    temp = st.number_input("Temperature (C)", value=20.0, format="%.2f", step=0.1)
    app_temp = st.number_input("Apparent Temperature (C)", value=20.0, format="%.2f", step=0.1)
    humidity = st.number_input("Humidity (0-1 or 0-100?)", value=0.75, format="%.3f", step=0.01)
    wind_speed = st.number_input("Wind Speed (km/h)", value=10.0, format="%.2f", step=0.1)
    wind_bearing = st.slider("Wind Bearing (degrees)", min_value=0, max_value=360, value=180)
    visibility = st.number_input("Visibility (km)", value=10.0, format="%.2f", step=0.1)
    cloud_cover = st.slider("Cloud Cover (0-1 or percent?)", min_value=0, max_value=100, value=50)
    pressure = st.number_input("Pressure (millibars)", value=1013.25, format="%.2f", step=0.1)

    submit = st.form_submit_button("Predict")

# When user hits predict
if submit:
    if objs is None:
        st.error("Model objects not loaded. Please upload 'rain_prediction_model.pkl' to the app directory.")
    else:
        try:
            # Create DataFrame with same column order you provided
            row = {
                "Formatted Date": formatted_date_str,
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

            st.write("### Input snapshot")
            st.dataframe(input_df.T, width=700)

            # Preprocessing & transform
            X = None
            # If user saved a ColumnTransformer / pipeline as 'transform', use that
            transform = objs.get("transform")
            scaler = objs.get("scaler")
            encoder = objs.get("encoder")
            model = objs.get("model")

            if transform is not None:
                # If transform expects a DataFrame, pass DataFrame; else convert to numpy
                try:
                    X = transform.transform(input_df)
                except Exception as e:
                    # Try passing values only
                    X = transform.transform(input_df.values)
            else:
                # No transform: try to use encoder + scaler heuristically
                # We'll attempt to apply encoder to categorical columns and scaler to numeric ones.
                # This is best-effort — if your actual pipeline is different, prefer saving & loading a single pipeline.
                df_proc = input_df.copy()

                # Convert formatted date into features if your transformer originally did that.
                # Many pipelines extract hour/day/month from 'Formatted Date'. If your pipeline did that,
                # and you only saved separate encoder/scaler, prediction may fail. We will not attempt complex
                # datetime feature engineering here beyond storing the original string.
                # If the transform you used earlier did datetime operations, please save and load it as `transform`.
                try:
                    # Apply encoder if it has transform
                    if encoder is not None and hasattr(encoder, "transform"):
                        # Attempt to find categorical columns by dtype object
                        cat_cols = [c for c in df_proc.columns if df_proc[c].dtype == "object"]
                        if len(cat_cols) > 0:
                            enc_res = encoder.transform(df_proc[cat_cols])
                            # If it returns array, combine with numeric columns
                            num_cols = [c for c in df_proc.columns if c not in cat_cols]
                            num_res = df_proc[num_cols].values.astype(float)
                            X = np.hstack([num_res, enc_res])
                        else:
                            X = df_proc.values
                    else:
                        # No encoder: convert to numeric where possible and use raw values
                        # Replace missing categorical columns with simple label-encoding via pandas (fallback)
                        df_fallback = df_proc.copy()
                        for col in df_fallback.select_dtypes(include=["object"]).columns:
                            df_fallback[col] = df_fallback[col].astype("category").cat.codes
                        if scaler is not None and hasattr(scaler, "transform"):
                            X = scaler.transform(df_fallback.values)
                        else:
                            X = df_fallback.values
                except Exception as e:
                    st.warning("Heuristic preprocessing failed; trying raw DataFrame values. Error: " + str(e))
                    X = input_df.values

            # Ensure X is 2D numeric array for model
            if hasattr(X, "toarray"):  # sparse
                try:
                    X = X.toarray()
                except Exception:
                    pass

            # Predict
            preds = model.predict(X)
            # If prediction returns a 1-element array
            if isinstance(preds, (list, np.ndarray, pd.Series)):
                pred_val = float(np.array(preds).ravel()[0])
            else:
                pred_val = float(preds)

            st.success("Prediction complete!")
            st.metric(label="Predicted value", value=f"{pred_val:.4f}")

            # Extra: show probability or additional outputs if model supports it
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)
                    st.write("Model predict_proba output (first row):")
                    st.write(proba[0])
                except Exception:
                    pass

        except Exception as e:
            st.error("Error during prediction: " + str(e))
            import traceback
            st.text(traceback.format_exc())
