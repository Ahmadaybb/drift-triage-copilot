# Dockerfile — agent_service
# OWNER: Person 2
# STATUS: 🔲 TODO

FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv && uv pip install --system -r pyproject.toml

COPY agent_service ./agent_service
COPY model_service ./model_service
COPY worker ./worker

EXPOSE 8001

CMD ["uvicorn", "agent_service.webhook_server:app", "--host", "0.0.0.0", "--port", "8001"]