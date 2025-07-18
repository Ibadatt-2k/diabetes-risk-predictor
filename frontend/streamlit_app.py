import streamlit as st
import requests
import shap
import joblib
import numpy as np
import os
# Load model and scaler
# Get the path to the script’s parent directory
base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "models", "rf_model.pkl")
model = joblib.load(model_path)
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
scaler = joblib.load(scaler_path)

explainer = shap.TreeExplainer(model)

st.title("Diabetes Risk Prediction")

with st.form(key='diabetes_form'):
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)

    submit_button = st.form_submit_button(label='Predict')

if submit_button:
    input_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    response = requests.post("https://diabetes-risk-predictor-x2ri.onrender.com/predict", json=input_data)
    # Input as DataFrame for SHAP
    import pandas as pd

    input_dict = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)

    # Compute SHAP values
    shap_values = explainer.shap_values(input_scaled)

    if response.status_code == 200:
        prediction = response.json()
        st.success(f"Prediction: {'Diabetic' if prediction['prediction'] == 1 else 'Non-Diabetic'}")
        st.info(f"Probability: {prediction['probability']*100:.2f}%")

        st.subheader("Feature Contribution (SHAP Explanation)")

        shap_row = shap_values[0, :, 1]  # first prediction row, all features, class 1

        shap_df = pd.DataFrame({
            "Feature": input_df.columns,
            "SHAP Value": shap_row
        }).sort_values(by="SHAP Value", key=abs, ascending=False)

        st.bar_chart(data=shap_df.set_index("Feature"))

    else:
        st.error("Error occurred during prediction.")
