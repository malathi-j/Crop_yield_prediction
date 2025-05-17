import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Try loading model and encoders
try:
    model = joblib.load("crop_yield_model.pkl")
    le_crop = joblib.load("le_crop.pkl")
    le_season = joblib.load("le_season.pkl")
    le_state = joblib.load("le_state.pkl")
except FileNotFoundError as e:
    st.error(f"❌ Required file not found: {e.filename}. Please make sure all .pkl files are in the same folder as this app.")
    st.stop()

# Title
st.title("🌾 Crop Yield Prediction App")
st.write("This app predicts **crop yield (tons/hectare)** using a trained ML model.")

# Input form
with st.form("prediction_form"):
    crop = st.selectbox("Crop", le_crop.classes_)
    crop_year = st.number_input("Crop Year", min_value=1990, max_value=2100, value=2022)
    season = st.selectbox("Season", le_season.classes_)
    state = st.selectbox("State", le_state.classes_)
    area = st.number_input("Area (hectares)", min_value=0.0, value=1.0)
    production = st.number_input("Production (tons)", min_value=0.0, value=1.0)
    annual_rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=800.0)
    fertilizer = st.number_input("Fertilizer Used (kg/ha)", min_value=0.0, value=50.0)
    pesticide = st.number_input("Pesticide Used (kg/ha)", min_value=0.0, value=5.0)

    submitted = st.form_submit_button("Predict Yield")

if submitted:
    try:
        # Encode categorical values
        crop_encoded = le_crop.transform([crop])[0]
        season_encoded = le_season.transform([season])[0]
        state_encoded = le_state.transform([state])[0]

        # Prepare input
        input_data = pd.DataFrame([[
            crop_encoded, crop_year, season_encoded, state_encoded,
            area, production, annual_rainfall, fertilizer, pesticide
        ]], columns=['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production',
                     'Annual_Rainfall', 'Fertilizer', 'Pesticide'])

        # Prediction
        prediction = model.predict(input_data)[0]
        st.success(f"🌾 Predicted Yield: {prediction:.2f} tons/hectare")

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {str(e)}")
