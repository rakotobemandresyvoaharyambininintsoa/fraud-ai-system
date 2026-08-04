import shap
import numpy as np


def explain_prediction(pipeline, transaction):

    # Récupération du pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]


    # Transformation des données
    transformed_data = preprocessor.transform(transaction)


    # Conversion sparse -> dense
    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()


    transformed_data = np.asarray(
        transformed_data,
        dtype=float
    )


    # Création explicateur SHAP
    explainer = shap.TreeExplainer(model)


    shap_values = explainer.shap_values(
        transformed_data
    )


    # Compatibilité plusieurs versions SHAP
    if isinstance(shap_values, list):

        values = shap_values[1][0]

    else:

        values = shap_values[0]


    values = np.asarray(values)


    # SHAP récent retourne parfois (features, classes)
    if len(values.shape) > 1:

        values = values[:, 1]


    # Récupération noms features
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


    # Trier par importance
    results = sorted(
        results,
        key=lambda x: x["impact"],
        reverse=True
    )


    return results[:5]