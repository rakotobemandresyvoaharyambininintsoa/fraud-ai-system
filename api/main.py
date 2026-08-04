from fastapi import FastAPI
import numpy as np
import joblib
import os
import pandas as pd


app = FastAPI(
    title="Fraud AI Detection API",
    version="2.0"
)


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(
    BASE_DIR,
    "model",
    "fraud_model.pkl"
)

model = joblib.load(model_path)


@app.get("/")
def home():
    return {
        "message": "Fraud AI Monitoring System Running"
    }


@app.post("/predict")
def predict(
    amount: float,
    country: str = "UNKNOWN",
    time: str = "unknown",
    device: str = "unknown",
    merchant: str = "unknown"
):

    transaction = pd.DataFrame(
        [
            {
                "amount": amount,
                "country": country,
                "time": time,
                "device": device,
                "merchant": merchant
            }
        ]
    )
        # Prediction
    prediction = model.predict(transaction)

    if prediction[0] == 1:
        risk_score = 85
        status = "FRAUD"
    else:
        risk_score = 20
        status = "SAFE"

    return {
        "transaction": {
            "amount": amount,
            "country": country,
            "time": time,
            "device": device,
            "merchant": merchant
        },
        "risk_score": risk_score,
        "status": status
    }