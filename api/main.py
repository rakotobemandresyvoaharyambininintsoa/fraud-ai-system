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
BOUNDS_PATH = os.path.join(BASE_DIR, "model", "validity_bounds.json")
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


def charger_bornes():
    if not os.path.exists(BOUNDS_PATH):
        logger.warning(
            "Bornes de validité introuvables (%s) — la détection d'entrées hors "
            "distribution est désactivée. Ré-entraînez le modèle pour les générer.",
            BOUNDS_PATH,
        )
        return None
    with open(BOUNDS_PATH, encoding="utf-8") as f:
        return json.load(f)


try:
    MODEL = charger_modele()
except RuntimeError as e:
    logger.error(str(e))
    raise

BORNES = charger_bornes()


def detecter_anomalies_hors_distribution(transaction_data: TransactionRequest) -> tuple[bool, list[str]]:
    """
    Détecte les entrées que le modèle ne peut pas correctement évaluer :
    valeurs catégorielles jamais vues à l'entraînement (que le OneHotEncoder
    ignore silencieusement au lieu de les traiter comme suspectes), et
    montants très au-delà de tout ce que le modèle a appris (au-delà duquel
    un RandomForest n'extrapole pas de façon fiable).

    Toutes les anomalies ne se valent pas :
      - un montant extrême est à lui seul un signal suffisamment grave pour
        forcer l'escalade (un montant ~40x le maximum observé n'est jamais
        anodin, quel que soit le contexte) ;
      - une SEULE valeur catégorielle inconnue isolée (un nouveau marchand
        jamais vu, par exemple) est un signal faible pris seul — beaucoup de
        marchands légitimes n'apparaissent simplement pas dans un jeu de
        données d'entraînement forcément limité. Elle ne force l'escalade
        que combinée à au moins un autre signal (montant OU une deuxième
        valeur inconnue).

    Retourne (faut_escalader, raisons).
    """
    if BORNES is None:
        return False, []

    raisons = []
    montant_extreme = transaction_data.amount > BORNES["amount_max_observed"] * 5

    if montant_extreme:
        raisons.append(
            f"Montant ({transaction_data.amount:,.0f}) très supérieur à tout ce que le "
            f"modèle a vu à l'entraînement (max observé : {BORNES['amount_max_observed']:,.0f})"
        )

    champs = {
        "country": transaction_data.country,
        "time": transaction_data.time,
        "device": transaction_data.device,
        "merchant": transaction_data.merchant,
    }
    valeurs_inconnues = [
        f"Valeur inconnue du modèle pour {champ} : « {valeur} »"
        for champ, valeur in champs.items()
        if valeur not in BORNES["known_values"].get(champ, [])
    ]
    raisons.extend(valeurs_inconnues)

    nb_signaux = int(montant_extreme) + len(valeurs_inconnues)
    faut_escalader = montant_extreme or len(valeurs_inconnues) >= 2

    return faut_escalader, raisons


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

    # Le modèle seul ne suffit pas : un OneHotEncoder(handle_unknown="ignore")
    # traite une valeur inconnue comme "aucune information" (pas "suspect"),
    # et un RandomForest n'extrapole pas au-delà des montants qu'il a vus à
    # l'entraînement. Un montant extrême force toujours l'alerte ; une seule
    # valeur catégorielle inconnue isolée ne le fait que combinée à un autre
    # signal (voir detecter_anomalies_hors_distribution).
    faut_escalader, raisons_ood = detecter_anomalies_hors_distribution(transaction_data)
    if faut_escalader:
        risk_score = max(risk_score, 90)

    if risk_score >= 80:
        risk_level, decision = "HIGH", "BLOCK"
    elif risk_score >= 50:
        risk_level, decision = "MEDIUM", "REVIEW"
    else:
        risk_level, decision = "LOW", "ALLOW"

    status = "FRAUD" if (prediction[0] == 1 or faut_escalader) else "SAFE"

    # Explication réelle basée sur les valeurs SHAP du modèle pour CETTE
    # transaction précise — pas une règle fixe déconnectée du modèle.
    explanation = generer_explication(MODEL, transaction, risk_score)
    if faut_escalader:
        explanation["risk_factors"] = raisons_ood + explanation["risk_factors"]
        explanation["summary"] = (
            "Transaction non évaluable de façon fiable par le modèle (données "
            "inhabituelles) — traitée par prudence comme à haut risque."
        )
        explanation["recommendation"] = "Bloquer et vérifier manuellement : données hors normes."

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
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         