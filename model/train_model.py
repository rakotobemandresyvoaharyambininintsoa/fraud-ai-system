import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==========================
# PATHS
# ==========================

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "transactions.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "fraud_model.pkl"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "metrics.json"
)

IMPORTANCE_PATH = os.path.join(
    BASE_DIR,
    "feature_importance.csv"
)


# ==========================
# LOAD DATA
# ==========================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)


print("\nDataset:")
print(df.head())

print("\nClasses:")
print(df["is_fraud"].value_counts())


# ==========================
# FEATURES
# ==========================

X = df[
    [
        "amount",
        "country",
        "time",
        "device",
        "merchant"
    ]
]


y = df["is_fraud"]


# ==========================
# PREPROCESSING
# ==========================


numeric_features = [
    "amount"
]


categorical_features = [
    "country",
    "time",
    "device",
    "merchant"
]


preprocessor = ColumnTransformer(
    transformers=[

        (
            "num",
            "passthrough",
            numeric_features
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ==========================
# MODEL
# ==========================


classifier = RandomForestClassifier(

    n_estimators=300,

    max_depth=15,

    min_samples_split=5,

    class_weight="balanced",

    random_state=42,

    n_jobs=-1
)



model = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            classifier
        )

    ]

)



# ==========================
# TRAIN TEST SPLIT
# ==========================


X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)



print("\nTraining model...")


model.fit(

    X_train,

    y_train

)



# ==========================
# EVALUATION
# ==========================


prediction = model.predict(X_test)


probability = model.predict_proba(X_test)[:,1]


accuracy = accuracy_score(
    y_test,
    prediction
)


precision = precision_score(
    y_test,
    prediction
)


recall = recall_score(
    y_test,
    prediction
)


f1 = f1_score(
    y_test,
    prediction
)


roc_auc = roc_auc_score(
    y_test,
    probability
)



print("\n===== MODEL PERFORMANCE =====")


print(
    classification_report(
        y_test,
        prediction
    )
)


print(
    "ROC-AUC:",
    roc_auc
)



print(
    "\nConfusion Matrix:"
)


print(
    confusion_matrix(
        y_test,
        prediction
    )
)



# ==========================
# CROSS VALIDATION
# ==========================


cv = cross_val_score(

    model,

    X,

    y,

    cv=5,

    scoring="f1"

)


print(
    "\nCross validation F1:",
    cv.mean()
)



# ==========================
# SAVE MODEL
# ==========================


joblib.dump(

    model,

    MODEL_PATH

)


print(
    "\nModel saved:",
    MODEL_PATH
)



# ==========================
# SAVE METRICS
# ==========================


metrics = {


    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "roc_auc": roc_auc,

    "cross_validation_f1": cv.mean()

}



with open(
    METRICS_PATH,
    "w"
) as file:

    json.dump(

        metrics,

        file,

        indent=4

    )



print(
    "Metrics saved"
)



print(
    "\nTraining completed successfully"
)