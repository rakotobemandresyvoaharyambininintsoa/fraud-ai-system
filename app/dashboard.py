import streamlit as st
import requests

# Titre de l'application
st.title("Banking Monitoring Dashboard")

# Saisie utilisateur
amount = st.number_input("Transaction Amount")
country = st.text_input("Country")
device = st.text_input("Device")

# Bouton d'analyse
if st.button("Analyze Transaction"):

    # Appel de l'API FastAPI
    response = requests.post(
        "http://127.0.0.1:8000/transaction/",
        params={
            "amount": amount,
            "country": country,
            "device": device
        }
    )

    result = response.json()

    # Affichage des résultats
    st.write("Amount:", result["amount"])
    st.write("Country:", result["country"])
    st.write("Device:", result["device"])
    st.write("Risk Score:", result["risk_score"])
    st.write("Status:", result["status"])

    # Affichage visuel du statut
    if result["status"] == "FRAUD":
        st.error("Fraud detected")
    elif result["status"] == "SUSPICIOUS":
        st.warning("Suspicious transaction")
    else:
        st.success("Safe transaction")