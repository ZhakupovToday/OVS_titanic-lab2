# ---------- 1. Общий слой: Python + зависимости сервиса ----------
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# requirements копируется отдельно, чтобы слой с зависимостями кэшировался
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- 2. Обучение модели внутри сборки ----------
# Пайплайн обучается теми же версиями библиотек, что и в рабочем образе,
# поэтому несовместимости версий scikit-learn при загрузке не будет.
FROM base AS trainer
COPY data ./data
COPY src ./src
RUN PYTHONPATH=./src python src/train.py \
        --data data/titanic.csv --out models/pipeline.pkl

# ---------- 3. Рабочий образ ----------
FROM base AS runtime
RUN groupadd --gid 1000 service && \
    useradd --uid 1000 --gid service --shell /bin/bash --create-home service
COPY src ./src
COPY --from=trainer /app/models/pipeline.pkl ./models/pipeline.pkl
ENV MODEL_PATH=/app/models/pipeline.pkl
WORKDIR /app/src
USER service
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthcheck')"
ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
