from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Load model and scaler
base_dir = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_dir, "models", "rf_model.pkl")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
                            data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    probability = model.predict_proba(scaled_data)[0][1]
    return {
        "prediction": int(prediction),
        "probability": float(round(probability, 2))
    }
