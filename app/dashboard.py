"""
Dashboard Streamlit pour tester l'API de scoring de fraude.

Corrections par rapport à la version précédente :
  - Appelait /transaction/, qui n'existe plus depuis le passage au pipeline
    RandomForest — corrigé pour appeler /predict.
  - Envoyait des paramètres de requête bruts (params=) au lieu d'un corps
    JSON, et n'envoyait pas la clé API requise, ni les champs time/merchant.
  - Aucune gestion d'erreur : plantait si l'API était indisponible ou
    renvoyait une erreur.
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("FRAUD_API_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("FRAUD_API_KEY", "dev-only-key-change-me")

st.set_page_config(page_title="Fraud AI Dashboard", page_icon="🔍")
st.title("Banking Fraud Monitoring Dashboard")
st.caption(f"API cible : {API_URL}")

with st.form("transaction_form"):
    amount = st.number_input("Montant de la transaction", min_value=0.0, value=1000.0)
    country = st.text_input("Pays", value="Madagascar")
    time = st.selectbox("Moment de la transaction", ["morning", "afternoon", "evening", "night"])
    device = st.selectbox("Appareil", ["mobile", "desktop", "tablet"])
    merchant = st.text_input("Marchand", value="amazon")
    submitted = st.form_submit_button("Analyser la transaction")

if submitted:
    try:
        response = requests.post(
            f"{API_URL}/predict",
            headers={"X-API-Key": API_KEY},
            json={
                "amount": float(amount),
                "country": country or "UNKNOWN",
                "time": time,
                "device": device,
                "merchant": merchant or "unknown",
            },
            timeout=10,
        )
    except requests.exceptions.RequestException as e:
        st.error(f"Impossible de joindre l'API ({API_URL}) : {e}")
        st.stop()

    if response.status_code != 200:
        st.error(f"Erreur API ({response.status_code}) : {response.text}")
        st.stop()

    result = response.json()
    analysis = result["analysis"]

    st.metric("Score de risque", f"{analysis['risk_score']} / 100")

    if analysis["status"] == "FRAUD":
        st.error(f"🚨 Fraude détectée — décision : {analysis['decision']}")
    else:
        st.success(f"✅ Transaction saine — décision : {analysis['decision']}")

    explication = analysis["ai_explanation"]
    st.write("**", explication["summary"], "**")

    st.write("**Facteurs de risque (basés sur les valeurs SHAP réelles du modèle) :**")
    for facteur in explication["risk_factors"]:
        st.write(f"- {facteur}")

    st.caption(explication["recommendation"])
