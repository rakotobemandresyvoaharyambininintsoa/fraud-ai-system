"""
Explication SHAP réelle des prédictions du modèle — remplace l'ancien
utils/explanation.py, qui générait une explication à base de règles fixes
(if amount > 5000, if country == "nigeria", ...) totalement déconnectée de
ce que le modèle avait réellement appris. Ici, l'explication reflète
l'importance SHAP réelle calculée sur CETTE transaction précise, pour CE
modèle entraîné — pas une supposition codée en dur.
"""

import numpy as np
import shap


def calculer_shap_values(pipeline, transaction):
    """Retourne les 5 facteurs qui ont le plus influencé la prédiction, par ordre d'impact."""

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(transaction)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)

    # Compatibilité entre versions de SHAP : certaines renvoient une liste
    # (une entrée par classe), d'autres un tableau (features, classes).
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[1][0])
    else:
        values = np.asarray(shap_values[0])
        if values.ndim > 1:
            values = values[:, 1]

    feature_names = preprocessor.get_feature_names_out()

    resultats = [
        {"feature": nom, "impact": round(float(valeur), 4)}
        for nom, valeur in zip(feature_names, values)
    ]

    return sorted(resultats, key=lambda x: abs(x["impact"]), reverse=True)[:5]


def _nom_lisible(feature_technique: str) -> str:
    """Convertit un nom de feature encodé (ex: 'categorical__merchant_unknown')
    en description lisible pour un utilisateur non technique."""

    nom = feature_technique.split("__")[-1]

    if nom == "amount":
        return "le montant de la transaction"

    prefixes_lisibles = {
        "country_": "le pays de la transaction ({})",
        "device_": "l'appareil utilisé ({})",
        "merchant_": "le marchand ({})",
        "time_": "le moment de la transaction ({})",
    }

    for prefixe, gabarit in prefixes_lisibles.items():
        if nom.startswith(prefixe):
            return gabarit.format(nom[len(prefixe):])

    return "le facteur " + nom.replace("_", " ")


def generer_explication(pipeline, transaction, risk_score: float) -> dict:
    """Construit une explication humainement lisible à partir des vraies valeurs SHAP,
    au lieu de règles fixes déconnectées du modèle."""

    facteurs_shap = calculer_shap_values(pipeline, transaction)

    # On ne retient que les facteurs qui ont poussé le score VERS la fraude
    # (impact positif) — un facteur qui réduit le risque n'est pas une "raison
    # d'être suspicieux".
    facteurs_risque = [f for f in facteurs_shap if f["impact"] > 0]

    risk_factors = [_nom_lisible(f["feature"]) for f in facteurs_risque[:3]]

    if risk_score >= 80:
        summary = "Cette transaction présente un risque élevé de fraude."
        recommendation = "Bloquer temporairement la transaction et demander une vérification."
    elif risk_score >= 50:
        summary = "Cette transaction présente des éléments suspects."
        recommendation = "Envoyer la transaction en revue manuelle."
    else:
        summary = "Cette transaction semble normale."
        recommendation = "Transaction autorisée."

    return {
        "summary": summary,
        "risk_factors": risk_factors if risk_factors else ["Aucun facteur dominant identifié"],
        "recommendation": recommendation,
        "shap_details": facteurs_shap,
    }
