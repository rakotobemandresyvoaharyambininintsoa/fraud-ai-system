def format_shap_factors(shap_results):

    formatted = []


    for item in shap_results:


        feature = item["feature"]

        impact = item["impact"]



        if "amount" in feature:

            factor = "Montant de transaction"


        elif "merchant_unknown" in feature:

            factor = "Marchand inconnu"


        elif "device" in feature:

            factor = "Appareil inhabituel"


        elif "country" in feature:

            factor = "Pays à risque"


        elif "time" in feature:

            factor = "Horaire inhabituel"


        else:

            factor = feature



        formatted.append(
            {
                "factor": factor,
                "impact": f"{impact*100:.2f}%"
            }
        )


    return formatted