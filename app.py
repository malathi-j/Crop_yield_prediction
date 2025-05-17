
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

# Title
st.title("🌾 Crop Yield Prediction App")
st.markdown("This app predicts **crop yield (tons/hectare)** using a trained machine learning model and label encoders.")

# Check for required files
if not os.path.exists("crop_yield_model.pkl") or not os.path.exists("label_encoders.pkl"):
    st.error("❌ Required files not found. Please upload `crop_yield_model.pkl` and `label_encoders.pkl` to the same folder.")
    st.stop()

# Load model and encoders
model = joblib.load("crop_yield_model.pkl")
encoders = joblib.load("label_encoders.pkl")

le_crop = encoders["Crop"]
le_season = encoders["Season"]
le_state = encoders["State"]

# Input form
with st.form("prediction_form"):
    st.subheader("🔢 Enter Input Features:")
    crop = st.selectbox("Crop", sorted(le_crop.classes_))
    crop_year = st.number_input("Crop Year", min_value=1990, max_value=2100, value=2022)
    season = st.selectbox("Season", sorted(le_season.classes_))
    state = st.selectbox("State", sorted(le_state.classes_))
    area = st.number_input("Area (hectares)", min_value=0.0, value=1.0)
    production = st.number_input("Production (tons)", min_value=0.0, value=1.0)
    annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=800.0)
    fertilizer = st.number_input("Fertilizer Used (kg/ha)", min_value=0.0, value=50.0)
    pesticide = st.number_input("Pesticide Used (kg/ha)", min_value=0.0, value=5.0)

    submitted = st.form_submit_button("🔍 Predict Yield")

if submitted:
    try:
        # Encode categorical values
        encoded_crop = le_crop.transform([crop])[0]
        encoded_season = le_season.transform([season])[0]
        encoded_state = le_state.transform([state])[0]

        # Prepare input for prediction
        input_df = pd.DataFrame([[
            encoded_crop, crop_year, encoded_season, encoded_state,
            area, production, annual_rainfall, fertilizer, pesticide
        ]], columns=[
            'Crop', 'Crop_Year', 'Season', 'State',
            'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'
        ])

        # Predict
        prediction = model.predict(input_df)[0]
        st.success(f"🌾 **Predicted Yield: {prediction:.2f} tons/hectare**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
