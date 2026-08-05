"""
API de scoring de fraude bancaire.

Corrections apportées par rapport à la version précédente :
  - L'explication utilise maintenant les vraies valeurs SHAP du modèle
    (utils/shap_explanation.py), plus des règles fixes déconnectées du
    modèle (dont une qui codait en dur un nom de pays réel comme "à risque").
  - Authentification par clé API restaurée (elle avait disparu).
  - Validation des entrées (montant strictement positif).
  - Chargement du modèle protégé par un message d'erreur clair au démarrage.
  - Chaque transaction scorée est journalisée, pour la traçabilité.
"""

import json
import logging
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from utils.shap_explanation import generer_explication

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fraud_pipeline.pkl")
LOG_PATH = os.path.join(BASE_DIR, "logs", "transactions_scored.log")

# Ne jamais coder une vraie clé en dur : à surcharger via la variable
# d'environnement en production. Cette valeur par défaut n'est valable
# qu'en développement local.
API_KEY = os.environ.get("FRAUD_API_KEY", "dev-only-key-change-me")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

app = FastAPI(
    title="Fraud AI Detection API",
    description="Scoring de risque de fraude sur transactions bancaires (projet de démonstration).",
    version="5.0",
)


class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0, description="Montant de la transaction (doit être positif)")
    country: str = Field(default="UNKNOWN", description="Pays de la transaction")
    time: str = Field(default="unknown", description="Moment de la transaction (morning/afternoon/evening/night)")
    device: str = Field(default="unknown", description="Appareil utilisé")
    merchant: str = Field(default="unknown", description="Marchand")


def verifier_cle_api(x_api_key: str = Header(...)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Clé API invalide.")


def charger_modele():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Entraînez-le d'abord avec : python data/generate_data.py && python model/train_model.py"
        )
    return joblib.load(MODEL_PATH)


try:
    MODEL = charger_modele()
except RuntimeError as e:
    logger.error(str(e))
    raise


@app.get("/")
def home():
    return {
        "message": "Fraud AI Monitoring System Running",
        "model": "RandomForest Pipeline",
        "version": "5.0",
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", dependencies=[Depends(verifier_cle_api)])
def predict(transaction_data: TransactionRequest):
    """Analyse une transaction bancaire et retourne un score de risque avec explication SHAP réelle."""

    transaction = pd.DataFrame([transaction_data.model_dump()])

    prediction = MODEL.predict(transaction)
    probability = MODEL.predict_proba(transaction)[0][1]

    fraud_probability = round(float(probability), 4)
    risk_score = int(fraud_probability * 100)

    if risk_score >= 80:
        risk_level, decision = "HIGH", "BLOCK"
    elif risk_score >= 50:
        risk_level, decision = "MEDIUM", "REVIEW"
    else:
        risk_level, decision = "LOW", "ALLOW"

    status = "FRAUD" if prediction[0] == 1 else "SAFE"

    # Explication réelle basée sur les valeurs SHAP du modèle pour CETTE
    # transaction précise — pas une règle fixe déconnectée du modèle.
    explanation = generer_explication(MODEL, transaction, risk_score)

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **transaction_data.model_dump(),
                    "risk_score": risk_score,
                    "status": status,
                }
            )
            + "\n"
        )

    return {
        "transaction": transaction_data.model_dump(),
        "analysis": {
            "fraud_probability": fraud_probability,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "status": status,
            "decision": decision,
            "ai_explanation": explanation,
        },
    }
