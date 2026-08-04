import joblib
import pandas as pd

from utils.shap_explanation import explain_prediction


model = joblib.load(
    "model/fraud_pipeline.pkl"
)


transaction = pd.DataFrame(
    [
        {
            "amount":999999,
            "country":"Unknown",
            "time":"night",
            "device":"unknown",
            "merchant":"unknown"
        }
    ]
)


result = explain_prediction(
    model,
    transaction
)


print(result)