FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Génère les données synthétiques et entraîne le modèle au moment du build,
# pour que l'image produise une API directement utilisable — sans cette
# étape, le conteneur démarrerait mais planterait au premier appel (le
# problème exact que ce projet avait avant correction).
RUN python data/generate_data.py && python model/train_model.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
