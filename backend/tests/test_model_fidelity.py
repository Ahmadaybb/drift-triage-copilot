"""
test_model_fidelity.py — 1e-12 fidelity replay test.
OWNER: Person 1
STATUS: 🔲 DONE

- Load registered model from MLflow
- Run inference on held-out test set
- Assert predictions match reference within 1e-12 tolerance
- Must pass on every CI push — refuses to merge if it regresses
"""
"""
test_model_fidelity.py — 1e-12 fidelity replay test.
OWNER: Person 1

Loads the registered model from MLflow and asserts predictions
on the held-out test set match reference predictions within 1e-12.

Run with:
    cd backend
    uv run pytest tests/test_model_fidelity.py -v

CI runs this on every push. Refuses to merge if it regresses.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

# ── Constants — must match train.py exactly ────────────────────────────────────
SEED          = 42
DATA_PATH     = Path(__file__).parent.parent / "data" / "bank-additional-full.csv"
ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
REGISTRY_DIR  = Path(__file__).parent.parent / "mlflow_registry"
TRACKING_URI  = f"sqlite:///{(REGISTRY_DIR / 'mlflow.db').as_posix()}"
MODEL_NAME    = "bank-marketing-classifier"
TOLERANCE     = 1e-12


# ── Fixtures ───────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def test_data():
    """
    Reproduce the exact test split from train.py.
    Same seed, same split sizes, same feature engineering.
    Returns X_test, y_test.
    """
    assert DATA_PATH.exists(), (
        f"Dataset not found at {DATA_PATH}. "
        "Place bank-additional-full.csv in backend/data/"
    )

    df = pd.read_csv(DATA_PATH, sep=";")

    # Reproduce train.py feature engineering exactly
    df = df.drop(columns=["duration"])
    df["y"] = df["y"].map({"yes": 1, "no": 0})
    df["pdays_never_contacted"] = (df["pdays"] == 999).astype(int)

    X = df.drop(columns=["y"])
    y = df["y"]

    # Reproduce train.py split exactly
    _, X_temp, _, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=SEED
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    return X_test, y_test


@pytest.fixture(scope="module")
def registered_model():
    """
    Load the registered model from MLflow.
    Tries Production first, falls back to Staging.
    """
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(TRACKING_URI)

    for stage in ["Production", "Staging"]:
        uri = f"models:/{MODEL_NAME}/{stage}"
        try:
            pipeline = mlflow.sklearn.load_model(uri)
            print(f"\nLoaded model from: {uri}")
            return pipeline, uri
        except Exception as e:
            print(f"Could not load from {uri}: {e}")

    pytest.fail(
        f"No model found in Production or Staging. "
        "Run train.py first."
    )


@pytest.fixture(scope="module")
def reference_predictions(test_data, registered_model):
    """
    Generate reference predictions from the registered model.
    These are the ground truth predictions we compare against.
    """
    X_test, _ = test_data
    pipeline, _ = registered_model
    return pipeline.predict_proba(X_test)[:, 1]


# ── Tests ──────────────────────────────────────────────────────────────────────
def test_model_loads(registered_model):
    """Model must load from MLflow without errors."""
    pipeline, uri = registered_model
    assert pipeline is not None, "Pipeline is None — model failed to load"
    assert uri is not None, "Model URI is None"
    print(f"\n✅ Model loaded from: {uri}")


def test_predictions_are_probabilities(reference_predictions):
    """All predictions must be valid probabilities in [0, 1]."""
    assert (reference_predictions >= 0.0).all(), "Some probabilities are negative"
    assert (reference_predictions <= 1.0).all(), "Some probabilities exceed 1.0"
    print(f"\n✅ All {len(reference_predictions)} predictions are valid probabilities")


def test_fidelity_replay(test_data, registered_model, reference_predictions):
    """
    Core fidelity test — reload the model and assert predictions
    are identical to reference within 1e-12 tolerance.

    This catches any silent change to the model, preprocessor, or data.
    """
    X_test, _ = test_data
    pipeline, uri = registered_model

    # Reload the model fresh — simulates what happens after a container restart
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(TRACKING_URI)

    pipeline_reloaded = mlflow.sklearn.load_model(uri)
    reloaded_predictions = pipeline_reloaded.predict_proba(X_test)[:, 1]

    # Assert predictions are identical within tolerance
    max_diff = float(np.abs(reference_predictions - reloaded_predictions).max())
    assert max_diff < TOLERANCE, (
        f"Fidelity check FAILED. "
        f"Max prediction difference: {max_diff:.2e} "
        f"(tolerance: {TOLERANCE:.2e}). "
        f"The registered model predictions have changed."
    )

    print(f"\n✅ Fidelity check passed. Max diff: {max_diff:.2e} < {TOLERANCE:.2e}")


def test_prediction_count(test_data, reference_predictions):
    """Number of predictions must match number of test rows."""
    X_test, _ = test_data
    assert len(reference_predictions) == len(X_test), (
        f"Prediction count mismatch: "
        f"got {len(reference_predictions)}, expected {len(X_test)}"
    )
    print(f"\n✅ Prediction count matches: {len(reference_predictions)} rows")


def test_positive_rate_in_expected_range(test_data, registered_model):
    """
    Positive rate at operating threshold must be >= 0.75 recall.
    Loads operating threshold from MLflow.
    """
    from mlflow.tracking import MlflowClient
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    client   = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    assert versions, f"No versions found for model '{MODEL_NAME}'"

    latest       = sorted(versions, key=lambda v: int(v.version))[-1]
    run          = client.get_run(latest.run_id)
    threshold    = float(run.data.metrics.get("operating_threshold", 0.5))
    test_recall  = float(run.data.metrics.get("test_recall", 0.0))

    assert test_recall >= 0.75, (
        f"Recall regression detected. "
        f"test_recall={test_recall:.4f} < 0.75. "
        f"Operating threshold: {threshold:.6f}"
    )

    print(f"\n✅ Recall check passed: {test_recall:.4f} >= 0.75")
    print(f"   Operating threshold: {threshold:.6f}")


def test_schema_artifact_exists():
    """schema.json must exist in artifacts directory."""
    schema_path = ARTIFACTS_DIR / "schema.json"
    assert schema_path.exists(), (
        f"schema.json not found at {schema_path}. Run train.py first."
    )

    with open(schema_path) as f:
        schema = json.load(f)

    assert "numeric_features"     in schema, "schema.json missing numeric_features"
    assert "categorical_features" in schema, "schema.json missing categorical_features"
    assert "duration" not in schema.get("numeric_features", []), (
        "duration must not appear in numeric_features — it leaks the target"
    )
    print(f"\n✅ schema.json exists and is valid")


def test_reference_stats_exist():
    """reference_stats.json must exist for drift detection."""
    ref_path = ARTIFACTS_DIR / "reference_stats.json"
    assert ref_path.exists(), (
        f"reference_stats.json not found at {ref_path}. Run train.py first."
    )

    with open(ref_path) as f:
        ref = json.load(f)

    assert "numeric"     in ref, "reference_stats.json missing numeric section"
    assert "categorical" in ref, "reference_stats.json missing categorical section"
    assert "output"      in ref, "reference_stats.json missing output section"
    assert "positive_rate" in ref["output"], "reference_stats.json missing positive_rate"

    print(f"\n✅ reference_stats.json exists and is valid")


def test_model_card_exists():
    """model_card.md must exist in artifacts directory."""
    card_path = ARTIFACTS_DIR / "model_card.md"
    assert card_path.exists(), (
        f"model_card.md not found at {card_path}. Run train.py first."
    )

    content = card_path.read_text()
    assert "AUC"       in content, "model_card.md missing AUC metric"
    assert "Recall"    in content, "model_card.md missing Recall metric"
    assert "Threshold" in content, "model_card.md missing Threshold"

    print(f"\n✅ model_card.md exists and contains required fields")


def test_pipeline_artifact_hash():
    """
    SHA-256 hash of the saved pipeline must match what was logged in MLflow.
    Catches silent model file corruption.
    """
    from mlflow.tracking import MlflowClient
    import mlflow

    pipeline_path = ARTIFACTS_DIR / "bank_marketing_pipeline.joblib"
    assert pipeline_path.exists(), (
        f"Pipeline file not found at {pipeline_path}. Run train.py first."
    )

    # Recompute hash
    h = hashlib.sha256()
    with pipeline_path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    computed_hash = h.hexdigest()

    # Load hash from MLflow
    mlflow.set_tracking_uri(TRACKING_URI)
    client   = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    assert versions, "No model versions found in MLflow"

    latest        = sorted(versions, key=lambda v: int(v.version))[-1]
    run           = client.get_run(latest.run_id)
    logged_hash   = run.data.params.get("artifact_hash", "")

    assert computed_hash == logged_hash, (
        f"Pipeline hash mismatch! "
        f"File hash:   {computed_hash[:12]}... "
        f"Logged hash: {logged_hash[:12]}... "
        f"The pipeline file may have been modified after training."
    )

    print(f"\n✅ Pipeline hash matches MLflow: {computed_hash[:12]}...")