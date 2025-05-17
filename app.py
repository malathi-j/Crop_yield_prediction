import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Title
st.title("🌾 Crop Yield Prediction App")
st.write("This app predicts **crop yield (tons/hectare)** using a trained machine learning model.")

# Check if all necessary model files exist
required_files = [
    "crop_yield_model.pkl"
]

missing_files = [file for file in required_files if not os.path.exists(file)]
if missing_files:
    st.error(f"❌ Missing file(s): {', '.join(missing_files)}. Please upload all required `.pkl` files.")
    st.stop()

# Load model and encoders
model = joblib.load("crop_yield_model.pkl")


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

# Prediction
if submitted:
    try:
        input_data = pd.DataFrame([[
            le_crop.transform([crop])[0],
            crop_year,
            le_season.transform([season])[0],
            le_state.transform([state])[0],
            area,
            production,
            annual_rainfall,
            fertilizer,
            pesticide
        ]], columns=['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production',
                     'Annual_Rainfall', 'Fertilizer', 'Pesticide'])

        prediction = model.predict(input_data)[0]
        st.success(f"🌾 Predicted Yield: {prediction:.2f} tons/hectare")
    except Exception as e:
        st.error(f"⚠️ Prediction error: {str(e)}")
