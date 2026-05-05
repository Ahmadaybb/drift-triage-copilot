"""
train.py — Train & register the bank marketing classifier.
OWNER: Person 1
STATUS: ✅ DONE
"""
"""
train.py — Train & Register: UCI Bank Marketing
================================================
Trains a binary classifier on the UCI Bank Marketing dataset,
tunes the operating threshold by the recall >= 0.75 rule,
and registers the fitted pipeline in MLflow.

Usage:
    python backend/model_service/train.py
"""

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
SEED          = 42
MODEL_NAME    = "bank-marketing-classifier"
MODEL_VERSION = "v0.1.0"
DATA_PATH     = Path("data/bank-additional-full.csv")
ARTIFACTS_DIR = Path("artifacts")
REGISTRY_DIR  = Path("mlflow_registry")

np.random.seed(SEED)


# ── Helpers ────────────────────────────────────────────────────────────────────
def sha256_of(path: Path) -> str:
    """Stream-hash a file in 64KB chunks — O(1) memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(name: str, pipeline, X, y, threshold: float = 0.5) -> dict:
    """Evaluate pipeline on a split and return metrics dict."""
    proba = pipeline.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)
    metrics = {
        "split":     name,
        "auc":       roc_auc_score(y, proba),
        "f1":        f1_score(y, pred),
        "precision": precision_score(y, pred),
        "recall":    recall_score(y, pred),
    }
    print(
        f"{name:5s} | AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}  "
        f"P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
    )
    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Drift Triage Co-Pilot — Training Pipeline")
    print("=" * 55)

    # ── 1) Load data ───────────────────────────────────────────────────────────
    print("\n[1/8] Loading data...")
    assert DATA_PATH.exists(), (
        f"Dataset not found at {DATA_PATH}. "
        "Place bank-additional-full.csv in backend/data/"
    )
    df = pd.read_csv(DATA_PATH, sep=";")
    print(f"      Raw shape: {df.shape}")
    print(f"      Target distribution:\n{df['y'].value_counts().to_string()}")

    # ── 2) Clean & engineer features ──────────────────────────────────────────
    print("\n[2/8] Cleaning and engineering features...")

    # Drop leakage column — recorded after call ends, leaks target
    df = df.drop(columns=["duration"])

    # Encode target
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    # Flag pdays sentinel — 999 means never previously contacted
    df["pdays_never_contacted"] = (df["pdays"] == 999).astype(int)

    # 'unknown' is kept as a real category — it is informative, not missing

    # Dataset fingerprint
    df_hash = hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()

    print(f"      Clean shape   : {df.shape}")
    print(f"      Positive rate : {df['y'].mean():.2%}")
    print(f"      Dataset hash  : {df_hash[:12]}...")

    # ── 3) Split ───────────────────────────────────────────────────────────────
    print("\n[3/8] Splitting data 60/20/20 stratified...")
    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    print(f"      Train : {X_train.shape}  positive_rate={y_train.mean():.2%}")
    print(f"      Val   : {X_val.shape}    positive_rate={y_val.mean():.2%}")
    print(f"      Test  : {X_test.shape}   positive_rate={y_test.mean():.2%}")

    # ── 4) Build preprocessor ─────────────────────────────────────────────────
    print("\n[4/8] Building preprocessor...")
    numeric_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols     = X_train.select_dtypes(include=["object"]).columns.tolist()

    print(f"      Numeric ({len(numeric_cols)}):     {numeric_cols}")
    print(f"      Categorical ({len(cat_cols)}): {cat_cols}")

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])

    # ── 5) Train pipeline ─────────────────────────────────────────────────────
    print("\n[5/8] Training pipeline...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=4,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])
    pipeline.fit(X_train, y_train)
    print("      Pipeline trained successfully.")

    # ── 6) Evaluate & tune threshold ──────────────────────────────────────────
    print("\n[6/8] Evaluating and tuning threshold...")
    print("      --- Default threshold (0.5) ---")
    evaluate("train", pipeline, X_train, y_train)
    evaluate("val",   pipeline, X_val,   y_val)
    evaluate("test",  pipeline, X_test,  y_test)

    # Tune on validation set only
    probs_val = pipeline.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs_val)
    valid_pairs = [
        (t, r) for t, r in zip(thresholds, recalls[:-1])
        if r >= 0.75
    ]
    assert valid_pairs, "No threshold achieves recall >= 0.75 on validation set."
    OPERATING_THRESHOLD = max(valid_pairs, key=lambda x: x[0])[0]

    print(f"\n      Operating threshold : {OPERATING_THRESHOLD:.6f}")
    print(f"      (rule: highest threshold where recall >= 0.75 on val set)")

    print("\n      --- Operating threshold on test set ---")
    final_m = evaluate("test", pipeline, X_test, y_test, threshold=OPERATING_THRESHOLD)

    # ── 7) Save artifacts ─────────────────────────────────────────────────────
    print("\n[7/8] Saving artifacts...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save pipeline binary
    pipeline_path = ARTIFACTS_DIR / "bank_marketing_pipeline.joblib"
    joblib.dump(pipeline, pipeline_path)
    artifact_hash = sha256_of(pipeline_path)
    print(f"      Pipeline saved : {pipeline_path}")
    print(f"      SHA-256        : {artifact_hash[:12]}...")

    # Capture environment metadata
    env_meta = {
        "python":      platform.python_version(),
        "platform":    platform.platform(),
        "sklearn":     sklearn.__version__,
        "numpy":       np.__version__,
        "pandas":      pd.__version__,
        "mlflow":      mlflow.__version__,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Write schema.json
    schema = {
        "numeric_features":     numeric_cols,
        "categorical_features": cat_cols,
        "target":               "y",
        "dropped_columns":      ["duration"],
        "engineered_features":  ["pdays_never_contacted"],
        "special_handling": {
            "duration": "dropped — recorded after call ends, leaks target",
            "pdays":    "pdays==999 flagged as pdays_never_contacted binary feature",
            "unknown":  "kept as real category — informative, not missing data",
        },
        "dataset_hash": df_hash,
    }
    schema_path = ARTIFACTS_DIR / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"      Schema saved   : {schema_path}")

    # Write model_card.md
    card = f"""# Model Card — {MODEL_NAME} {MODEL_VERSION}

**Status:** Staging candidate
**Artifact hash:** `{artifact_hash}`
**Dataset hash:** `{df_hash}`

## Intended Use

Predict whether a bank customer will subscribe to a term deposit,
given their demographic, economic, and campaign contact attributes.
Output is a probability in [0, 1]. Operating threshold is set by
the rule: highest threshold where recall >= 0.75 on the validation set.

## Training Data

UCI Bank Marketing — `bank-additional-full.csv`
~41,188 rows x 20 features (19 after dropping `duration`).
Class balance: ~89% no / ~11% yes.
Split: 60/20/20 stratified (random_state=42).

## Architecture

sklearn Pipeline:
1. ColumnTransformer:
   - Numeric: SimpleImputer(median) -> StandardScaler
   - Categorical: SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
2. HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4, class_weight='balanced')

## Feature Notes

- duration dropped — leaks target (recorded after call ends)
- pdays==999 flagged as pdays_never_contacted binary feature
- unknown values kept as real category — they are informative

## Measured Metrics (Test Set)

| Metric    | Value  |
|-----------|--------|
| AUC       | {final_m['auc']:.4f} |
| F1        | {final_m['f1']:.4f} |
| Recall    | {final_m['recall']:.4f} |
| Precision | {final_m['precision']:.4f} |
| Threshold | {OPERATING_THRESHOLD:.6f} |

## Environment

```json
{json.dumps(env_meta, indent=2)}
```
"""
    card_path = ARTIFACTS_DIR / "model_card.md"
    with open(card_path, "w") as f:
        f.write(card)
    print(f"      Model card saved : {card_path}")

    # ── 8) MLflow — log, register, promote ────────────────────────────────────
    print("\n[8/8] Logging to MLflow and registering model...")
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    (REGISTRY_DIR / "artifacts").mkdir(exist_ok=True)

    tracking_uri  = f"sqlite:///{(REGISTRY_DIR / 'mlflow.db').as_posix()}"
    artifact_root = (REGISTRY_DIR / "artifacts").resolve().as_uri()

    mlflow.set_tracking_uri(tracking_uri)

    experiment_name = "bank-marketing-drift-triage"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_root)
    mlflow.set_experiment(experiment_name)

    client    = MlflowClient()
    signature = infer_signature(
        X_train.head(100),
        pipeline.predict_proba(X_train.head(100))
    )

    with mlflow.start_run(run_name="hgb-balanced-baseline") as run:
        mlflow.log_param("model_type",      "HistGradientBoostingClassifier")
        mlflow.log_param("max_iter",        300)
        mlflow.log_param("learning_rate",   0.05)
        mlflow.log_param("max_depth",       4)
        mlflow.log_param("class_weight",    "balanced")
        mlflow.log_param("dropped_column",  "duration")
        mlflow.log_param("threshold_rule",  "recall >= 0.75")
        mlflow.log_param("artifact_hash",   artifact_hash)
        mlflow.log_param("dataset_hash",    df_hash)

        mlflow.log_metric("test_auc",            final_m["auc"])
        mlflow.log_metric("test_f1",             final_m["f1"])
        mlflow.log_metric("test_recall",         final_m["recall"])
        mlflow.log_metric("test_precision",      final_m["precision"])
        mlflow.log_metric("operating_threshold", OPERATING_THRESHOLD)

        mlflow.log_artifact(str(schema_path))
        mlflow.log_artifact(str(card_path))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(2),
            registered_model_name=MODEL_NAME,
        )

        run_id = run.info.run_id

    # Promote latest version to Staging
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_v = sorted(versions, key=lambda mv: int(mv.version))[-1]
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest_v.version,
        stage="Staging",
        archive_existing_versions=True,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  DONE — SUBMISSION SUMMARY")
    print("=" * 55)
    print(f"  Model          : {MODEL_NAME} v{latest_v.version}")
    print(f"  Stage          : Staging")
    print(f"  Load URI       : models:/{MODEL_NAME}/Staging")
    print(f"  Test AUC       : {final_m['auc']:.4f}")
    print(f"  Test F1        : {final_m['f1']:.4f}")
    print(f"  Test Recall    : {final_m['recall']:.4f}")
    print(f"  Test Precision : {final_m['precision']:.4f}")
    print(f"  Threshold      : {OPERATING_THRESHOLD:.6f}  (rule: recall >= 0.75)")
    print(f"  Artifact hash  : {artifact_hash[:12]}...")
    print(f"  MLflow run ID  : {run_id}")
    print("=" * 55)


if __name__ == "__main__":
    main()