# Fraud Detection API

Système de détection d'anomalies sur transactions bancaires, combinant un modèle de machine learning non-supervisé (Isolation Forest), une API REST (FastAPI) et un dashboard de test (Streamlit).

## ⚠️ Sur les données et les résultats

Ce projet utilise des **données synthétiques générées volontairement** (`data/generate_data.py`) — aucune donnée bancaire réelle n'est utilisée ni publiée, pour des raisons évidentes de confidentialité. Le générateur crée une fraude clairement séparable du comportement normal (montant très supérieur à l'habitude de l'utilisateur, pays inhabituel, nouvel appareil), ce qui donne des métriques d'évaluation excellentes (AUC proche de 1.0) sur ces données de test. **Ces métriques ne préjugent pas de la performance sur des données réelles**, où les patterns de fraude sont plus subtils et le signal plus faible. Ce projet est une démonstration d'architecture (pipeline ML → API → dashboard, avec de bonnes pratiques d'ingénierie), pas un modèle validé pour la production.

## Architecture

```
data/          → génération de données synthétiques (transactions + profils utilisateurs)
model/         → entraînement du modèle Isolation Forest, avec évaluation
api/           → API FastAPI de scoring en temps réel (authentifiée)
app/           → dashboard Streamlit pour tester l'API manuellement
tests/         → tests automatisés de l'API (pytest)
```

## Ce que le modèle utilise réellement

Contrairement à une version initiale qui ne se basait que sur le montant brut, le modèle actuel utilise :
- le montant de la transaction,
- l'heure de la transaction,
- si l'appareil utilisé diffère de l'appareil habituel de l'utilisateur,
- l'écart entre le montant et le comportement habituel de l'utilisateur.

Le pays et l'appareil déclarés influencent également le score final via des règles explicites (pays à risque, changement d'appareil), en plus d'alimenter le modèle — l'API renvoie la liste des facteurs de risque détectés (`reasons`) pour rendre chaque score explicable.

## Démarrage rapide

```bash
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements-dev.txt

cp .env.example .env
# éditez .env et changez FRAUD_API_KEY

# 1. Générer les données synthétiques
python data/generate_data.py

# 2. Entraîner le modèle (affiche precision/recall/AUC)
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
curl -X POST http://127.0.0.1:8000/transaction/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <votre clé>" \
  -d '{"user_id": 1, "amount": 900000, "country": "XX", "device": "tablet"}'
```

Réponse :
```json
{
  "user_id": 1,
  "amount": 900000.0,
  "country": "XX",
  "device": "tablet",
  "risk_score": 98.6,
  "status": "FRAUD",
  "reasons": [
    "Pays inhabituel ou à risque (XX)",
    "Appareil différent de l'appareil habituel",
    "Montant très supérieur au comportement habituel de l'utilisateur"
  ]
}
```

## Lancer les tests

```bash
pytest tests/ -v
```

7 tests couvrent : l'authentification (clé API requise/invalide), la validation des entrées (montant négatif rejeté), la cohérence du scoring (une transaction clairement frauduleuse doit scorer plus haut qu'une transaction normale — le bug initial que ce test aurait immédiatement détecté), et la robustesse face à un utilisateur inconnu.

## Sécurité

- Authentification par clé API (header `X-API-Key`), à surcharger via la variable d'environnement `FRAUD_API_KEY` — jamais de valeur en dur en production.
- Validation stricte des entrées (Pydantic) : montant positif obligatoire, heure entre 0 et 23.
- Chaque transaction scorée est journalisée (`logs/transactions_scored.log`) pour la traçabilité — indispensable dans un contexte de conformité bancaire.

## Limites connues et pistes d'amélioration

- Modèle non-supervisé simple (Isolation Forest) : un vrai système de production combinerait plusieurs modèles, des règles métier plus riches, et un ré-entraînement périodique sur des données réelles labellisées a posteriori.
- Les profils utilisateurs sont statiques (générés une fois) ; un système réel mettrait à jour le comportement habituel en continu (fenêtre glissante).
- Pas de gestion de fuseau horaire réel pour le champ `hour`.
- Authentification par clé API unique — un système multi-clients voudrait une clé par consommateur, avec rotation et révocation.

## License

MIT
