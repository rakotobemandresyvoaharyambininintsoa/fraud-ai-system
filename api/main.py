from pydantic import BaseModel
from utils.explanation import generate_explanation
from fastapi import FastAPI
import joblib
import os
import pandas as pd


app = FastAPI(
    title="Fraud AI Detection API",
    version="3.0"
)


# ==========================
# REQUEST MODEL
# ==========================

class TransactionRequest(BaseModel):

    amount: float

    country: str = "UNKNOWN"

    time: str = "unknown"

    device: str = "unknown"

    merchant: str = "unknown"



# ==========================
# LOAD MODEL
# ==========================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))


model_path = os.path.join(
    BASE_DIR,
    "model",
    "fraud_model.pkl"
)


model = joblib.load(model_path)



# ==========================
# HOME
# ==========================

@app.get("/")
def home():

    return {
        "message": "Fraud AI Monitoring System Running",
        "model": "RandomForest",
        "version": "3.0"
    }




# ==========================
# PREDICTION
# ==========================

@app.post("/predict")
def predict(transaction_data: TransactionRequest):


    amount = transaction_data.amount

    country = transaction_data.country

    time = transaction_data.time

    device = transaction_data.device

    merchant = transaction_data.merchant



    # Création dataframe compatible modèle

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



    # ==========================
    # MODEL PREDICTION
    # ==========================

    prediction = model.predict(transaction)


    probability = model.predict_proba(transaction)[0][1]


    fraud_probability = round(
        float(probability),
        2
    )


    risk_score = int(
        fraud_probability * 100
    )



    # ==========================
    # RISK ANALYSIS
    # ==========================

    if risk_score >= 80:

        risk_level = "HIGH"

        decision = "BLOCK"


    elif risk_score >= 50:

        risk_level = "MEDIUM"

        decision = "REVIEW"


    else:

        risk_level = "LOW"

        decision = "ALLOW"



    status = (

        "FRAUD"

        if prediction[0] == 1

        else

        "SAFE"

    )



    # ==========================
    # AI EXPLANATION
    # ==========================

    explanation = generate_explanation(

        amount,

        country,

        time,

        device,

        merchant,

        risk_score

    )



    # ==========================
    # RESPONSE
    # ==========================

    return {


        "transaction": {

            "amount": amount,

            "country": country,

            "time": time,

            "device": device,

            "merchant": merchant

        },


        "analysis": {

            "fraud_probability": fraud_probability,

            "risk_score": risk_score,

            "risk_level": risk_level,

            "status": status,

            "decision": decision,

            "ai_explanation": explanation

        }

    }