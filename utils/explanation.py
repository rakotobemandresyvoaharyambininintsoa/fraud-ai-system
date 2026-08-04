def generate_explanation(
    amount,
    country,
    time,
    device,
    merchant,
    risk_score
):

    reasons = []


    if amount > 5000:
        reasons.append(
            "Montant de transaction inhabituellement élevé"
        )


    if country.lower() in [
        "unknown",
        "nigeria"
    ]:
        reasons.append(
            "Origine géographique présentant un risque élevé"
        )


    if device.lower() in [
        "unknown",
        "tablet"
    ]:
        reasons.append(
            "Appareil considéré comme inhabituel"
        )


    if merchant.lower() == "unknown":
        reasons.append(
            "Marchand non identifié"
        )


    if time.lower() == "night":
        reasons.append(
            "Transaction effectuée durant une période inhabituelle"
        )


    if risk_score >= 80:

        recommendation = (
            "Bloquer temporairement la transaction "
            "et demander une vérification."
        )

    elif risk_score >= 50:

        recommendation = (
            "Envoyer la transaction en revue manuelle."
        )

    else:

        recommendation = (
            "Transaction autorisée."
        )


    return {
        "summary": (
            "Cette transaction présente un risque "
            "élevé de fraude."
            if risk_score >= 80
            else
            "Cette transaction semble normale."
        ),

        "risk_factors": reasons,

        "recommendation": recommendation
    }