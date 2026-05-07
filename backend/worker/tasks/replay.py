from __future__ import annotations

"""
tasks/replay.py — Replay test-set inference against the registry model.
OWNER: Person 2

Loads Production (fallback Staging) sklearn pipeline from MLflow, rebuilds the
train.py held-out test split, scores accuracy at the operating threshold — no train/promote.
"""


import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# backend/
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent

REGISTRY_DIR = _BACKEND_ROOT / "mlflow_registry"
TRACKING_URI = f"sqlite:///{(REGISTRY_DIR / 'mlflow.db').as_posix()}"

MODEL_NAME = "bank-marketing-classifier"
MODEL_URI_PRODUCTION = f"models:/{MODEL_NAME}/Production"
MODEL_URI_STAGING = f"models:/{MODEL_NAME}/Staging"

DATA_PATH = _BACKEND_ROOT / "data" / "bank-additional-full.csv"

SEED = 42
DEFAULT_THRESHOLD = 0.3784


def _load_operating_threshold() -> float:
    """Mirror predictor.Predictor._load_threshold_from_registry logic."""

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        return DEFAULT_THRESHOLD

    latest = sorted(versions, key=lambda v: int(v.version))[-1]
    run = client.get_run(latest.run_id)
    threshold = float(run.data.params.get("operating_threshold", DEFAULT_THRESHOLD))
    if threshold == DEFAULT_THRESHOLD:
        threshold = float(run.data.metrics.get("operating_threshold", DEFAULT_THRESHOLD))
    return threshold


def _load_production_pipeline() -> tuple[object, str]:
    mlflow.set_tracking_uri(TRACKING_URI)

    for uri in (MODEL_URI_PRODUCTION, MODEL_URI_STAGING):
        try:
            pipe = mlflow.sklearn.load_model(uri)
            logger.info("Replay loaded model from %s", uri)
            return pipe, uri
        except Exception as exc:
            logger.warning("Replay could not load %s: %s", uri, exc)

    raise RuntimeError(
        "No Production or Staging model for bank-marketing-classifier. Train and register first."
    )


def _build_test_split() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset missing at {DATA_PATH}. Place bank-additional-full.csv under backend/data/"
        )

    df = pd.read_csv(DATA_PATH, sep=";")
    df = df.drop(columns=["duration"])
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    df["pdays_never_contacted"] = (df["pdays"] == 999).astype(int)

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=SEED
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


def run(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Queue payload keys (investigation_id, model_uri, model_name, ...) are logged only;

    replay always evaluates the live Production→Staging pipeline from MLflow against the
    fixed held-out test split (same recipe as train.py).
    """

    inv = payload.get("investigation_id", "")
    logger.info("replay.run start investigation_id=%s", inv)

    try:
        pipeline, loaded_uri = _load_production_pipeline()
        threshold = _load_operating_threshold()
        X_test, y_test = _build_test_split()

        proba = pipeline.predict_proba(X_test)[:, 1]
        y_hat = (proba >= threshold).astype(int)

        acc = float(accuracy_score(y_test, y_hat))

        out = {
            "ok": True,
            "investigation_id": inv,
            "model_uri_loaded": loaded_uri,
            "operating_threshold": threshold,
            "n_test_samples": int(len(y_test)),
            "accuracy": acc,
            "positive_rate_true": float(y_test.mean()),
            "positive_rate_pred": float(y_hat.mean()),
        }
        logger.info(
            "replay.run done investigation_id=%s accuracy=%.5f n=%s",
            inv,
            acc,
            len(y_test),
        )
        return out

    except Exception as exc:
        logger.exception("replay.run failed investigation_id=%s", inv)
        raise RuntimeError(f"replay failed: {exc}") from exc
