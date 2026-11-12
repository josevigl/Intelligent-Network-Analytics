FROM node:22-bookworm-slim AS frontend-build

WORKDIR /workspace/web
COPY web/package*.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.11-slim

WORKDIR /app
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml config.yaml model_config.yaml ./
COPY app ./app
COPY src ./src
COPY pipelines ./pipelines
COPY results/models ./results/models
RUN python -m pip install --no-cache-dir --upgrade \
    'pip>=26.2.0' \
    'setuptools>=84.1.0' \
    'wheel>=0.46.2' \
    'jaraco.context>=6.1.0' \
    && python -m pip install --no-cache-dir '.[app,data,cpu-serving]'
COPY --from=frontend-build /workspace/web/dist ./web/dist

ENV HF_HOME=/tmp/huggingface \
    INA_HOST=0.0.0.0 \
    INA_PUBLIC_DEMO=true \
    PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app \
    && chown -R app:app /app /home/app

USER app:app
EXPOSE 10000
CMD ["sh", "-c", "exec ina-app --host \"$INA_HOST\" --port \"${PORT:-10000}\""]
