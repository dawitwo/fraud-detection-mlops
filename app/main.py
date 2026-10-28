# app/main.py
import mlflow.xgboost
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from app.schemas import Transaction, PredictionResponse
from src.config import MLFLOW_TRACKING_URI, SCALER_PATH

app = FastAPI(title="Fraud Detection API", version="1.0.0")

model = None
scaler = None
THRESHOLD = 0.5

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.xgboost.load_model("models:/fraud-detector/latest")
    scaler = joblib.load(SCALER_PATH)
    print("Model and scaler loaded successfully.")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = np.array(transaction.features).reshape(1, -1)
        proba = model.predict_proba(features)[0, 1]
        return PredictionResponse(
            fraud_probability=float(proba),
            is_fraud=bool(proba >= THRESHOLD),
            threshold=THRESHOLD,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))