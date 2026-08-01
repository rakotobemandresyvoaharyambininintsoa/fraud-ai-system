import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

# chemin propre (niveau entreprise)
BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "..", "data", "transactions.csv")

# charger data
df = pd.read_csv(data_path)

# features
X = df[["amount"]]

# model IA
model = IsolationForest(contamination=0.2, random_state=42)
model.fit(X)

# sauvegarde modèle
model_path = os.path.join(BASE_DIR, "fraud_model.pkl")
joblib.dump(model, model_path)

print("Model trained successfully + saved")