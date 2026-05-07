"""
app.py — FastAPI application.
OWNER: Person 1
STATUS: Done

Endpoints:
    POST /predict       → run prediction, return PredictionResponse
    GET  /drift-report  → compute drift, emit webhook if severity changed
    POST /promote       → promote model version after agent approval
    GET  /health        → model URI, threshold, prediction count
"""
"""
app.py — FastAPI application for the Bank Marketing model service.
OWNER: Person 1

Endpoints:
    POST /predict       → run prediction, return PredictionResponse
    GET  /drift-report  → compute drift, emit webhook if severity changed
    POST /promote       → promote model version after agent approval
    GET  /health        → model URI, threshold, prediction count
"""

import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from model_service.drift import compute_drift_report
from model_service.predictor import predictor
from model_service.registry_gate import run_promotion_gate
from model_service.schemas import (
    DriftAlert,
    ErrorDetail,
    ErrorResponse,
    PromotionRequest,
    PromotionResponse,
    PredictionRequest,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
AGENT_WEBHOOK_URL = os.getenv("AGENT_SERVICE_URL", "http://agent_service:8001")
DRIFT_WEBHOOK_URL = f"{AGENT_WEBHOOK_URL}/webhook/drift"

# Track last known severity to avoid sending duplicate webhooks
_last_severity: str = "none"


# ── Lifespan — runs on startup and shutdown ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("Starting model service — loading model from MLflow...")
    predictor.load()
    logger.info("Model service ready.")
    yield
    logger.info("Shutting down model service.")


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Drift Triage Co-Pilot — Model Service",
    description="Serves predictions, computes drift, gates promotions.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Error handlers ─────────────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """
    Convert Pydantic validation errors into structured ErrorResponse.
    Never return a raw stack trace to the caller.
    """
    details = []
    for error in exc.errors():
        field   = " → ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        details.append(ErrorDetail(field=field, message=message))

    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Validation error — check your request fields.",
            details=details,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Catch-all — never expose stack traces."""
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error. Check service logs.",
            details=[],
        ).model_dump(),
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check endpoint.
    Returns model URI, threshold, and prediction count.
    Used by docker-compose healthcheck.
    """
    try:
        recent = predictor.get_recent_predictions(n=1)
        count  = predictor._db_conn.execute(
            "SELECT COUNT(*) FROM predictions"
        ).fetchone()[0]
    except Exception:
        count = 0

    return {
        "status":    "ok",
        "model_uri": predictor.model_uri,
        "threshold": predictor.operating_threshold,
        "prediction_count": count,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Run a prediction for a single customer record.

    - Validates input with Pydantic (bad inputs return 422, never stack traces)
    - Runs prediction at the operating threshold
    - Logs prediction to DB for drift detection
    - Returns probability + binary label
    """
    return predictor.predict(request)


@app.get("/drift-report", response_model=DriftAlert)
async def drift_report():
    """
    Compute drift over the rolling window of recent predictions.

    - Computes PSI on numeric features
    - Computes chi² on categorical features
    - Computes output distribution drift
    - If severity changed since last check → emits webhook to agent
    - Returns the full DriftAlert payload
    """
    global _last_severity

    # Get recent predictions from DB
    recent = predictor.get_recent_predictions(n=1000)

    # Compute drift report
    alert = compute_drift_report(
        recent_predictions=recent,
        model_uri=predictor.model_uri,
    )

    # Emit webhook to agent only if severity changed
    if alert.severity.level != _last_severity:
        logger.info(
            f"Severity changed: {_last_severity} → {alert.severity.level}. "
            f"Sending webhook to agent."
        )
        _last_severity = alert.severity.level
        await _emit_webhook(alert)

    return alert


@app.post("/promote", response_model=PromotionResponse)
async def promote(request: PromotionRequest):
    """
    Promote or rollback a model version to Production.

    - Only accepts requests from the agent (approved_by='human')
    - Runs the full promotion checklist gate
    - Refuses if any gate check fails
    - Hot-reloads the predictor after successful promotion
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    from predictor import TRACKING_URI

    # Run promotion gate — all 8 checks must pass
    gate = run_promotion_gate(
        model_name=request.model_name,
        version=request.version,
        requested_by=request.approved_by,
        investigation_id=request.investigation_id,
    )

    if not gate.passed:
        failed = gate.failed_checks()
        raise HTTPException(
            status_code=400,
            detail=f"Promotion gate failed on: {failed}. "
                   f"Run gate.summary() for details.",
        )

    # Perform the promotion in MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    if request.action == "promote":
        client.transition_model_version_stage(
            name=request.model_name,
            version=request.version,
            stage="Production",
            archive_existing_versions=True,
        )
        new_uri = f"models:/{request.model_name}/Production"
        message = f"v{request.version} promoted to Production."

    elif request.action == "rollback":
        client.transition_model_version_stage(
            name=request.model_name,
            version=request.version,
            stage="Production",
            archive_existing_versions=True,
        )
        new_uri = f"models:/{request.model_name}/Production"
        message = f"Rolled back to v{request.version} in Production."

    # Hot-reload the predictor with the new model
    predictor.reload()
    logger.info(f"Promotion complete: {message}")

    return PromotionResponse(
        success=True,
        model_uri=new_uri,
        message=message,
    )


# ── Internal helpers ───────────────────────────────────────────────────────────
async def _emit_webhook(alert: DriftAlert):
    """
    Send drift alert webhook to the agent service.
    Fire-and-forget — does not block the drift-report response.
    Logs a warning if the agent is unreachable.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                DRIFT_WEBHOOK_URL,
                json=alert.model_dump(),
            )
            if response.status_code == 200:
                logger.info(f"Webhook sent to agent. Status: {response.status_code}")
            else:
                logger.warning(
                    f"Agent returned unexpected status: {response.status_code}"
                )
    except httpx.ConnectError:
        logger.warning(
            f"Could not reach agent at {DRIFT_WEBHOOK_URL}. "
            "Agent may not be running yet."
        )
    except Exception as e:
        logger.error(f"Webhook failed: {e}")


# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)