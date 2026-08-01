from fastapi import FastAPI
import numpy as np
import joblib
import os

# Création de l'application API
app = FastAPI()

# Chargement du modèle entraîné
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, "model", "fraud_model.pkl")
model = joblib.load(model_path)

@app.get("/")
def home():
    # Route de vérification si l'API fonctionne
    return {"message": "Banking Monitoring System Running"}

@app.post("/transaction/")
def transaction(amount: float, country: str = "UNKNOWN", device: str = "UNKNOWN"):
    """
    Analyse une transaction bancaire et retourne un score de risque.
    """

    # Prédiction du modèle (anomalie ou non)
    prediction = model.predict(np.array([[amount]]))

    # Transformation du résultat du modèle en score de risque
    if prediction[0] == -1:
        risk_score = 85  # transaction suspecte
    else:
        risk_score = 20  # transaction normale

    # Détermination du statut bancaire
    if risk_score < 30:
        status = "SAFE"
    elif risk_score < 70:
        status = "SUSPICIOUS"
    else:
        status = "FRAUD"

    # Résultat retourné par l'API
    return {
        "amount": amount,
        "country": country,
        "device": device,
        "risk_score": risk_score,
        "status": status
    }