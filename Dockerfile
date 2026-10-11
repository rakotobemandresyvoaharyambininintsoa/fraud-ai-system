FROM python@sha256:1eb6b7d4b76454b1de8317863ac3213b678c337b27e604a4e3fb70bddbb2bad7

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --only-binary :all: --require-hashes -r requirements.txt \
    && groupadd -r appuser && useradd -r -g appuser appuser

COPY api/ ./api/
COPY app/ ./app/
COPY data/ ./data/
COPY model/ ./model/
COPY utils/ ./utils/

RUN python data/generate_data.py && python model/train_model.py \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
