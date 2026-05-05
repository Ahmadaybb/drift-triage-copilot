"""
predictor.py — Loads model from MLflow, runs predictions, logs to DB.
OWNER: Person 1
STATUS: ✅ DONE
"""
"""
predictor.py — Model loader and prediction runner.

Loads the registered pipeline from MLflow at startup.
Runs predictions at the operating threshold.
Logs every prediction to SQLite for drift detection.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
REGISTRY_DIR   = Path(__file__).parent.parent / "mlflow_registry"
TRACKING_URI   = f"sqlite:///{(REGISTRY_DIR / 'mlflow.db').as_posix()}"
MODEL_URI      = "models:/bank-marketing-classifier/Production"
MODEL_URI_FALLBACK = "models:/bank-marketing-classifier/Staging"

PREDICTIONS_DB = Path(__file__).parent.parent / "artifacts" / "predictions.db"

# Operating threshold — must match train.py output
# Loaded dynamically from MLflow run params at startup
DEFAULT_THRESHOLD = 0.3784


# ── Predictor ──────────────────────────────────────────────────────────────────
class Predictor:
    """
    Loads the model from MLflow and serves predictions.
    Singleton — instantiated once at app startup.
    """

    def __init__(self):
        self.pipeline          = None
        self.model_uri         = None
        self.operating_threshold = DEFAULT_THRESHOLD
        self._db_conn          = None

    def load(self):
        """Load model from MLflow registry and set up predictions DB."""
        import mlflow
        mlflow.set_tracking_uri(TRACKING_URI)

        # Try Production first, fall back to Staging
        for uri in [MODEL_URI, MODEL_URI_FALLBACK]:
            try:
                self.pipeline  = mlflow.sklearn.load_model(uri)
                self.model_uri = uri
                logger.info(f"Model loaded from: {uri}")
                break
            except Exception as e:
                logger.warning(f"Could not load from {uri}: {e}")

        if self.pipeline is None:
            raise RuntimeError(
                "No model found in Production or Staging. Run train.py first."
            )

        # Load operating threshold from the latest MLflow run
        self.operating_threshold = self._load_threshold_from_registry()
        logger.info(f"Operating threshold: {self.operating_threshold}")

        # Set up predictions database
        self._setup_db()
        logger.info("Predictor ready.")

    def _load_threshold_from_registry(self) -> float:
        """Load the operating threshold logged during training."""
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(
                f"name='bank-marketing-classifier'"
            )
            if not versions:
                return DEFAULT_THRESHOLD

            latest = sorted(versions, key=lambda v: int(v.version))[-1]
            run    = client.get_run(latest.run_id)
            threshold = float(
                run.data.params.get("operating_threshold", DEFAULT_THRESHOLD)
            )
            # Also try metrics (we logged it both ways)
            if threshold == DEFAULT_THRESHOLD:
                threshold = float(
                    run.data.metrics.get("operating_threshold", DEFAULT_THRESHOLD)
                )
            return threshold
        except Exception as e:
            logger.warning(f"Could not load threshold from registry: {e}. Using default.")
            return DEFAULT_THRESHOLD

    def _setup_db(self):
        """Create predictions table if it doesn't exist."""
        PREDICTIONS_DB.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(PREDICTIONS_DB), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT    NOT NULL,
                model_uri   TEXT    NOT NULL,
                threshold   REAL    NOT NULL,
                probability REAL    NOT NULL,
                label       INTEGER NOT NULL,
                features    TEXT    NOT NULL
            )
        """)
        conn.commit()
        self._db_conn = conn
        logger.info(f"Predictions DB ready at: {PREDICTIONS_DB}")

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """
        Validate input, run prediction, log to DB, return response.
        Never raises — errors are handled at the FastAPI layer.
        """
        if self.pipeline is None:
            raise RuntimeError("Predictor not loaded. Call load() first.")

        # Convert request to DataFrame row
        row_dict = request.to_dataframe_row()
        X = pd.DataFrame([row_dict])

        # Run prediction
        proba = float(self.pipeline.predict_proba(X)[0, 1])
        label = int(proba >= self.operating_threshold)

        # Log prediction to DB
        self._log_prediction(
            probability=proba,
            label=label,
            features=row_dict,
        )

        return PredictionResponse(
            model_uri=self.model_uri,
            threshold=self.operating_threshold,
            probability=round(proba, 6),
            label=label,
        )

    def _log_prediction(
        self,
        probability: float,
        label: int,
        features: dict,
    ):
        """Persist prediction to SQLite for drift detection."""
        try:
            self._db_conn.execute(
                """
                INSERT INTO predictions
                    (timestamp, model_uri, threshold, probability, label, features)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    self.model_uri,
                    self.operating_threshold,
                    probability,
                    label,
                    json.dumps(features),
                ),
            )
            self._db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

    def get_recent_predictions(self, n: int = 1000) -> pd.DataFrame:
        """
        Return the n most recent predictions as a DataFrame.
        Used by drift.py to compute drift over a rolling window.
        """
        query = """
            SELECT timestamp, probability, label, features
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """
        rows = self._db_conn.execute(query, (n,)).fetchall()
        if not rows:
            return pd.DataFrame()

        records = []
        for ts, prob, label, feat_json in rows:
            row = json.loads(feat_json)
            row["__timestamp__"]   = ts
            row["__probability__"] = prob
            row["__label__"]       = label
            records.append(row)

        return pd.DataFrame(records)

    def reload(self):
        """Hot-reload the model from MLflow — called after a promotion."""
        logger.info("Reloading model from MLflow...")
        self.pipeline            = None
        self.model_uri           = None
        self.operating_threshold = DEFAULT_THRESHOLD
        self.load()
        logger.info("Model reloaded successfully.")


# ── Singleton ──────────────────────────────────────────────────────────────────
predictor = Predictor()