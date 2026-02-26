from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="Market Sentinel API")

MODEL_PATH = "models/production_model.pkl"

# -----------------------------
# Request Schema
# -----------------------------
class PredictionRequest(BaseModel):
    Quantity: float
    UnitPrice: float
    year: int
    month: int
    day: int
    day_of_week: int
    hour: int
    Country_freq: float


# -----------------------------
# Utility: Load Model
# -----------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise Exception("Production model not found.")
    return joblib.load(MODEL_PATH)


# -----------------------------
# Health Endpoint
# -----------------------------
@app.get("/health")
def health():
    return {"status": "API is running"}


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(request: PredictionRequest):
    model = load_model()

    input_data = pd.DataFrame([request.model_dump()])
    prediction = model.predict(input_data)[0]

    return {"predicted_price": float(prediction)}


# -----------------------------
# Drift Status Endpoint
# -----------------------------
@app.get("/drift-status")
def drift_status():
    if os.path.exists("drift_report.json"):
        return {"drift_report_available": True}
    else:
        return {"drift_report_available": False}