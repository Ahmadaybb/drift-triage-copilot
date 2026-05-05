# Dockerfile — model_service
# OWNER: Person 1
# STATUS: Done
# Dockerfile — model_service
# OWNER: Person 1
#
# Builds the FastAPI model service container.
# Serves predictions, computes drift, gates promotions.

# ── Base image ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install uv ────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir uv

# ── Copy dependency files first (better Docker cache) ─────────────────────────
COPY pyproject.toml .

# ── Install dependencies ───────────────────────────────────────────────────────
RUN uv pip install --system --no-cache \
    fastapi \
    uvicorn[standard] \
    pydantic \
    mlflow \
    scikit-learn \
    pandas \
    numpy \
    scipy \
    joblib \
    httpx \
    python-dotenv

# ── Copy model service source code ────────────────────────────────────────────
COPY model_service/ ./model_service/

# ── Copy shared artifacts (reference stats, schema, model card) ───────────────
COPY artifacts/ ./artifacts/

# ── Copy MLflow registry (model weights + metadata) ───────────────────────────
COPY mlflow_registry/ ./mlflow_registry/

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Run the FastAPI app ────────────────────────────────────────────────────────
CMD ["uvicorn", "model_service.app:app", "--host", "0.0.0.0", "--port", "8000"]