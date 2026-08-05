"""
Tests de l'API de scoring de fraude.

Prérequis : le modèle doit être entraîné avant de lancer ces tests
(python data/generate_data.py && python model/train_model.py).
"""

import os

os.environ.setdefault("FRAUD_API_KEY", "dev-only-key-change-me")

from fastapi.testclient import TestClient

from api.main import API_KEY, app

client = TestClient(app)

TRANSACTION_NORMALE = {
    "amount": 1200,
    "country": "Madagascar",
    "time": "afternoon",
    "device": "mobile",
    "merchant": "amazon",
}

TRANSACTION_SUSPECTE = {
    "amount": 9800,
    "country": "XX",
    "time": "night",
    "device": "desktop",
    "merchant": "unknown",
}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_requires_api_key():
    response = client.post("/predict", json=TRANSACTION_NORMALE)
    assert response.status_code == 422


def test_predict_rejects_wrong_api_key():
    response = client.post(
        "/predict", json=TRANSACTION_NORMALE, headers={"X-API-Key": "mauvaise-cle"}
    )
    assert response.status_code == 401


def test_predict_rejects_negative_amount():
    payload = {**TRANSACTION_NORMALE, "amount": -100}
    response = client.post("/predict", json=payload, headers={"X-API-Key": API_KEY})
    assert response.status_code == 422


def test_predict_valid_request_structure():
    response = client.post(
        "/predict", json=TRANSACTION_NORMALE, headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200
    body = response.json()
    analysis = body["analysis"]

    assert 0 <= analysis["risk_score"] <= 100
    assert analysis["status"] in {"SAFE", "FRAUD"}
    assert analysis["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert "ai_explanation" in analysis
    assert "risk_factors" in analysis["ai_explanation"]


def test_suspicious_transaction_scores_higher_than_normal_one():
    normal = client.post(
        "/predict", json=TRANSACTION_NORMALE, headers={"X-API-Key": API_KEY}
    ).json()
    suspect = client.post(
        "/predict", json=TRANSACTION_SUSPECTE, headers={"X-API-Key": API_KEY}
    ).json()

    assert suspect["analysis"]["risk_score"] > normal["analysis"]["risk_score"]


def test_explanation_never_names_a_real_country_as_inherently_risky():
    """
    Garde-fou explicite contre la régression corrigée dans ce projet : l'explication
    ne doit jamais coder en dur un nom de pays réel comme facteur de risque. Elle ne
    doit citer un pays que si le modèle l'a réellement identifié comme facteur pour
    CETTE transaction (via SHAP), jamais par une règle fixe.
    """
    payload = {**TRANSACTION_NORMALE, "country": "France"}
    response = client.post("/predict", json=payload, headers={"X-API-Key": API_KEY})
    explication = response.json()["analysis"]["ai_explanation"]

    for facteur in explication["risk_factors"]:
        assert "nigeria" not in facteur.lower()


def test_unknown_values_do_not_crash_the_api():
    payload = {"amount": 500, "country": "Unknown", "time": "unknown", "device": "unknown", "merchant": "unknown"}
    response = client.post("/predict", json=payload, headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
