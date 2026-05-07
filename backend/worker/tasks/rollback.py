from __future__ import annotations

"""
tasks/rollback.py — Roll back Production via model_service POST /promote.
OWNER: Person 2

Resolves the latest registered MLflow version strictly below current Production,
then calls Person 1's /promote with action=rollback (approved_by=human).
"""


import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
REGISTRY_DIR = _BACKEND_ROOT / "mlflow_registry"
TRACKING_URI = f"sqlite:///{(REGISTRY_DIR / 'mlflow.db').as_posix()}"

MODEL_NAME = "bank-marketing-classifier"
MODEL_SERVICE_URL = os.environ.get("MODEL_SERVICE_URL", "http://localhost:8000").rstrip("/")


def _rollback_target_version(client: MlflowClient) -> str:
    prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not prod:
        raise RuntimeError(f"No Production version registered for {MODEL_NAME!r}.")

    current = int(prod[0].version)

    all_mv = client.search_model_versions(f"name='{MODEL_NAME}'")
    older = sorted({int(m.version) for m in all_mv if int(m.version) < current})

    if not older:
        raise RuntimeError(
            f"No registered version older than Production v{current} — nothing to roll back to."
        )

    return str(max(older))


def run(payload: dict[str, Any]) -> dict[str, Any]:
    investigation_id = payload.get("investigation_id") or ""
    if not str(investigation_id).strip():
        raise RuntimeError("rollback payload missing investigation_id")

    logger.info("rollback.run start investigation_id=%s", investigation_id)

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    target_ver = _rollback_target_version(client)

    body = {
        "action": "rollback",
        "model_name": MODEL_NAME,
        "version": target_ver,
        "approved_by": "human",
        "investigation_id": investigation_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    url = f"{MODEL_SERVICE_URL}/promote"

    try:
        with httpx.Client(timeout=120.0) as http:
            resp = http.post(url, json=body)
    except httpx.HTTPError as exc:
        logger.exception("rollback.run HTTP failure investigation_id=%s", investigation_id)
        raise RuntimeError(f"POST {url} failed: {exc}") from exc

    if resp.status_code != 200:
        logger.error(
            "rollback.run promote HTTP %s body=%s investigation_id=%s",
            resp.status_code,
            resp.text[:500],
            investigation_id,
        )
        raise RuntimeError(
            f"POST /promote returned {resp.status_code}: {resp.text[:200]}"
        )

    data = resp.json()
    if not data.get("success"):
        logger.error(
            "rollback.run promote success=false data=%s investigation_id=%s",
            data,
            investigation_id,
        )
        raise RuntimeError(f"Promotion refused: {data}")

    out = {
        "ok": True,
        "investigation_id": investigation_id,
        "rollback_to_version": target_ver,
        "model_uri": data.get("model_uri", ""),
        "message": data.get("message", ""),
    }
    logger.info(
        "rollback.run done investigation_id=%s rolled_back_to=v%s",
        investigation_id,
        target_ver,
    )
    return out
