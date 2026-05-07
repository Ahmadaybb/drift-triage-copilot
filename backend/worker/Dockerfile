# Dockerfile — worker
# OWNER: Person 2
# STATUS: 🔲 TODO
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv pip install --system -r pyproject.toml

COPY agent_service ./agent_service
COPY model_service ./model_service
COPY worker ./worker
COPY data ./data
COPY artifacts ./artifacts
COPY mlflow_registry ./mlflow_registry

CMD ["python", "-m", "worker.worker"]