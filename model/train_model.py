import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ==========================
# PATH
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
    "fraud_pipeline.pkl"
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

categorical_features = [
    "country",
    "time",
    "device",
    "merchant"
]


preprocessor = ColumnTransformer(

    transformers=[

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )

    ],

    remainder="passthrough"
)


# ==========================
# MODEL
# ==========================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=None,

    random_state=42,

    class_weight="balanced"

)


pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            model
        )

    ]

)


# ==========================
# TRAIN TEST
# ==========================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


print("\nTraining model...")


pipeline.fit(
    X_train,
    y_train
)


# ==========================
# EVALUATION
# ==========================

pred = pipeline.predict(X_test)

proba = pipeline.predict_proba(X_test)[:,1]


print("\n===== MODEL PERFORMANCE =====")

print(
    classification_report(
        y_test,
        pred
    )
)


print(
    "ROC-AUC:",
    roc_auc_score(
        y_test,
        proba
    )
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y_test,
        pred
    )
)


cv = cross_val_score(

    pipeline,

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
# SAVE METRICS
# ==========================

import json

METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

report_dict = classification_report(y_test, pred, output_dict=True)

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "accuracy": report_dict["accuracy"],
            "precision_fraud": report_dict["1"]["precision"],
            "recall_fraud": report_dict["1"]["recall"],
            "f1_fraud": report_dict["1"]["f1-score"],
            "roc_auc": roc_auc_score(y_test, proba),
            "cross_validation_f1": cv.mean(),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "fraud_rate_dataset": float(y.mean()),
        },
        f,
        indent=2,
    )

print("\nMétriques sauvegardées :", METRICS_PATH)
print(
    "\nNOTE : les métriques ci-dessus sont volontairement imparfaites sur la "
    "classe minoritaire (fraude) — un modèle affichant 1.0 partout sur un "
    "problème de fraude est un signal de fuite de données ou de jeu de "
    "données trop simple, pas une performance à mettre en avant."
)


# ==========================
# SAVE
# ==========================

joblib.dump(

    pipeline,

    MODEL_PATH

)


print(
    "\nModel saved:",
    MODEL_PATH
)


print(
    "\nTraining completed successfully"
)