import joblib
import os

def save_model(model, filename="rf_model.pkl"):
    path = os.path.join("../models", filename)
    joblib.dump(model, path)
    print(f"✅ Model is being saved to: {path}")
