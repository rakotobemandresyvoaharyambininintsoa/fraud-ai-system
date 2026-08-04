import shap
import pandas as pd


def explain_prediction(pipeline, transaction):

    # Récupération des composants du pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]


    # Transformation des données comme pendant l'entraînement
    transformed_data = preprocessor.transform(transaction)


    # Explicateur SHAP pour RandomForest
    explainer = shap.TreeExplainer(model)


    shap_values = explainer.shap_values(
        transformed_data
    )


    # Cas classification binaire
    if isinstance(shap_values, list):
        values = shap_values[1][0]
    else:
        values = shap_values[0]


    # Noms des features après OneHotEncoding
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )


    results = []


    for feature, value in zip(
        feature_names,
        values
    ):

        results.append(
            {
                "feature": feature,
                "impact": round(
                    float(abs(value)),
                    4
                )
            }
        )


    # Trier les facteurs les plus importants

    results = sorted(
        results,
        key=lambda x: x["impact"],
        reverse=True
    )


    return results[:5]