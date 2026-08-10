# Fraud AI Detection System

Système de détection de fraude bancaire combinant un modèle de machine learning supervisé (RandomForest), une explication de chaque décision par valeurs SHAP réelles, une API REST (FastAPI) et un dashboard de test (Streamlit).

## ⚠️ Sur les données et les métriques

Ce projet utilise des **données synthétiques générées volontairement** (`data/generate_data.py`) — aucune donnée bancaire réelle n'est utilisée, pour des raisons évidentes de confidentialité. Le label de fraude est généré de façon **probabiliste** (facteurs de risque combinés + bruit), pas par un simple seuil déterministe — un modèle qui atteint 100% de précision/rappel sur un problème de fraude est un signal de fuite de données ou de jeu de données trop simple, pas une performance à revendiquer. Les métriques actuelles (`model/metrics.json`, régénérées à chaque entraînement) sont volontairement imparfaites sur la classe minoritaire (la fraude), ce qui est plus honnête et plus réaliste.

## Architecture

```
data/          → génération de données synthétiques (transactions bancaires)
model/         → pipeline scikit-learn (OneHotEncoder + RandomForest), évaluation, métriques
api/           → API FastAPI de scoring en temps réel (authentifiée)
utils/         → explication des prédictions par valeurs SHAP réelles
app/           → dashboard Streamlit pour tester l'API manuellement
tests/         → tests automatisés de l'API (pytest)
```

## Ce que fait réellement le modèle

Le pipeline (`model/train_model.py`) encode les variables catégorielles (pays, appareil, marchand, moment de la journée) via `OneHotEncoder`, puis entraîne un `RandomForestClassifier` supervisé sur un jeu d'entraînement/test séparé, avec validation croisée.

**Chaque prédiction est expliquée par les vraies valeurs SHAP** calculées pour cette transaction précise (`utils/shap_explanation.py`) — pas par des règles fixes. Une version précédente de ce projet expliquait les décisions via des seuils codés en dur, dont un qui désignait un pays réel comme intrinsèquement "à risque" : c'était à la fois factuellement incorrect (le modèle n'apprend aucun lien réel avec ce pays) et problématique dans la forme. Ça n'existe plus : l'explication ne cite un pays, un appareil ou un marchand que si le modèle l'a réellement identifié comme facteur déterminant pour cette transaction précise.

### Protection contre les entrées hors distribution

Un modèle de ML ne sait pas dire "je n'ai jamais vu ça" — testé manuellement avec un montant de 5 000 000 (~400x le maximum vu à l'entraînement) combiné à des valeurs jamais rencontrées (`country="UNKNOWN"`, `time="03AM"`, `merchant="black_market"`), l'API ressortait `SAFE` à 12% : le `OneHotEncoder(handle_unknown="ignore")` traite une valeur inconnue comme "aucune information", pas comme "suspect", et un `RandomForest` n'extrapole pas au-delà des montants qu'il a appris.

L'API ajoute donc une couche de règles explicites (`detecter_anomalies_hors_distribution` dans `api/main.py`), en complément du modèle : toute transaction avec un montant très supérieur au maximum observé à l'entraînement, ou avec une valeur catégorielle jamais vue, est automatiquement classée à haut risque — indépendamment de ce que dit le modèle seul. C'est une pratique standard en production : le ML ne doit jamais être la seule ligne de défense.

## Démarrage rapide

```bash
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env
# éditez .env et changez FRAUD_API_KEY

# 1. Générer les données synthétiques
python data/generate_data.py

# 2. Entraîner le modèle (affiche precision/recall/AUC, sauvegarde les métriques)
python model/train_model.py

# 3. Lancer l'API
export FRAUD_API_KEY=<la valeur de votre .env>
uvicorn api.main:app --reload

# 4. Dans un autre terminal, lancer le dashboard de test
export FRAUD_API_KEY=<la même valeur>
streamlit run app/dashboard.py
```

L'API est documentée automatiquement sur `http://127.0.0.1:8000/docs` (Swagger UI).

### Avec Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 -e FRAUD_API_KEY=<votre clé> fraud-api
```

Le modèle est entraîné automatiquement au moment du build de l'image.

## Tester l'API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <votre clé>" \
  -d '{"amount": 9800, "country": "XX", "time": "night", "device": "desktop", "merchant": "unknown"}'
```

Réponse :
```json
{
  "transaction": { "amount": 9800.0, "country": "XX", "time": "night", "device": "desktop", "merchant": "unknown" },
  "analysis": {
    "fraud_probability": 0.7033,
    "risk_score": 70,
    "risk_level": "MEDIUM",
    "status": "FRAUD",
    "decision": "REVIEW",
    "ai_explanation": {
      "summary": "Cette transaction présente des éléments suspects.",
      "risk_factors": [
        "le pays de la transaction (XX)",
        "le marchand (unknown)",
        "le moment de la transaction (night)"
      ],
      "recommendation": "Envoyer la transaction en revue manuelle.",
      "shap_details": [ /* impact SHAP brut de chaque feature, pour audit */ ]
    }
  }
}
```

## Lancer les tests

```bash
pytest tests/ -v
```

8 tests couvrent : l'authentification (clé API requise/invalide), la validation des entrées (montant négatif rejeté), la cohérence du scoring (une transaction suspecte doit scorer plus haut qu'une transaction normale), la robustesse face à des valeurs inconnues, **et un garde-fou explicite qui vérifie que l'explication ne désigne jamais un pays réel comme intrinsèquement risqué** — pour empêcher la régression corrigée dans ce projet de revenir sans qu'on s'en rende compte.

## Sécurité

- Authentification par clé API (header `X-API-Key`), à surcharger via `FRAUD_API_KEY` — jamais de valeur en dur en production.
- Validation stricte des entrées (Pydantic) : montant strictement positif.
- Chaque transaction scorée est journalisée (`logs/transactions_scored.log`) pour la traçabilité.

## Limites connues et pistes d'amélioration

- Données synthétiques : les performances ne préjugent pas du comportement sur des transactions réelles, où le signal est plus faible et plus subtil.
- Un seul modèle testé (RandomForest) ; comparer avec XGBoost/LightGBM/régression logistique serait une extension naturelle.
- Pas de base de données relationnelle (les transactions scorées sont journalisées en fichier plat, pas en base) ; pour un vrai système de production, une base type PostgreSQL avec tables `transactions`/`alerts`/`users` serait l'étape suivante logique.
- Authentification par clé API unique — un système multi-clients voudrait une clé par consommateur, avec rotation et révocation.

## License

MIT
