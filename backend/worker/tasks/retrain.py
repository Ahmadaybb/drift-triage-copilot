from __future__ import annotations
"""
tasks/retrain.py — Run Person 1 training pipeline (new MLflow version → Staging only).
OWNER: Person 2

Invokes model_service.train.main() with cwd set to backend/ so train.py relative paths work.
Does not promote to Production. Worker-level idempotency keys prevent duplicate queue execution.
"""


import logging
import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_DIR = _BACKEND_ROOT / "mlflow_registry"
TRACKING_URI = f"sqlite:///{(REGISTRY_DIR / 'mlflow.db').as_posix()}"

MODEL_NAME = "bank-marketing-classifier"


def run(payload: dict[str, Any]) -> dict[str, Any]:
    investigation_id = str(payload.get("investigation_id") or "").strip()
    if not investigation_id:
        raise RuntimeError("retrain payload missing investigation_id")

    logger.info("retrain.run start investigation_id=%s", investigation_id)

    prev_cwd = os.getcwd()
    try:
        os.chdir(_BACKEND_ROOT)

        from model_service.train import main as train_main

        train_main()
    except Exception as exc:
        logger.exception("retrain.run training failed investigation_id=%s", investigation_id)
        raise RuntimeError(f"train.main() failed: {exc}") from exc
    finally:
        os.chdir(prev_cwd)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError("Training finished but MLflow has no registered versions.")

    latest = sorted(versions, key=lambda v: int(v.version))[-1]

    out = {
        "ok": True,
        "investigation_id": investigation_id,
        "model_name": MODEL_NAME,
        "model_version": latest.version,
        "current_stage": latest.current_stage,
        "run_id": latest.run_id,
    }
    logger.info(
        "retrain.run done investigation_id=%s version=%s stage=%s",
        investigation_id,
        latest.version,
        latest.current_stage,
    )
    return out
