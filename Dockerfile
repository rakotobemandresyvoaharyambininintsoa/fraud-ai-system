FROM python:3.11

# Dossier de travail dans le conteneur
WORKDIR /app

# Copier tous les fichiers du projet
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Port exposé pour l'API
EXPOSE 8000

# Lancer l'API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]