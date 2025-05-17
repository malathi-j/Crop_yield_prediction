import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and label encoders
model = joblib.load("crop_yield_model.pkl")
le_crop = joblib.load("le_crop.pkl")
le_season = joblib.load("le_season.pkl")
le_state = joblib.load("le_state.pkl")

# Streamlit app
st.title("🌾 Crop Yield Prediction App")
st.write("This app predicts **crop yield (tons/hectare)** using trained machine learning model.")

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
        # Encode categorical features
        encoded_crop = le_crop.transform([crop])[0]
        encoded_season = le_season.transform([season])[0]
        encoded_state = le_state.transform([state])[0]

        input_data = pd.DataFrame([[
            encoded_crop, crop_year, encoded_season, encoded_state,
            area, production, annual_rainfall, fertilizer, pesticide
        ]], columns=['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Production',
                     'Annual_Rainfall', 'Fertilizer', 'Pesticide'])

        prediction = model.predict(input_data)[0]
        st.success(f"🌾 Predicted Yield: {prediction:.2f} tons/hectare")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
