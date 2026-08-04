import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os


BASE_DIR = os.path.dirname(__file__)

data_path = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "transactions.csv"
)

df = pd.read_csv(data_path)


# Features
X = df[
    [
        "amount",
        "country",
        "time",
        "device",
        "merchant"
    ]
]

# Target
y = df["is_fraud"]


# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


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


model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)


# Entraînement
model.fit(
    X_train,
    y_train
)


# Evaluation
prediction = model.predict(X_test)

print(
    classification_report(
        y_test,
        prediction
    )
)


# Sauvegarde
model_path = os.path.join(
    BASE_DIR,
    "fraud_model.pkl"
)

joblib.dump(
    model,
    model_path
)


print(
    "Random Forest fraud model saved"
)