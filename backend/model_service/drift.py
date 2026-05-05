"""
drift.py — Drift detection logic.
OWNER: Person 1
STATUS: done

Responsibilities:
- Compute PSI on numeric features
- Compute chi² on categorical features
- Compute output distribution drift
- Classify severity: none / warning / high
- Return DriftAlert payload ready to send as webhook

PSI thresholds:
    < 0.1   → none
    0.1–0.2 → warning
    > 0.2   → high

chi² threshold:
    p-value < 0.05 → drift detected
"""
"""
drift.py — Drift detection logic.
OWNER: Person 1

Computes PSI on numeric features, chi² on categoricals,
and output distribution drift. Returns a DriftAlert payload.

PSI thresholds:
    < 0.1   → none
    0.1–0.2 → warning
    > 0.2   → high

chi² threshold:
    p-value < 0.05 → drift detected
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from schemas import DriftAlert, DriftFeatureReport, DriftSeverity

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
REFERENCE_STATS_PATH = Path(__file__).parent.parent / "artifacts" / "reference_stats.json"

PSI_WARNING  = 0.1
PSI_HIGH     = 0.2
CHI2_PVALUE  = 0.05   # p-value below this = drift detected
N_BINS       = 10     # bins for PSI numeric computation


# ── PSI ────────────────────────────────────────────────────────────────────────
def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = N_BINS) -> float:
    """
    Compute Population Stability Index between reference and current distributions.

    PSI = sum((current% - reference%) * ln(current% / reference%))

    Returns 0.0 if current has fewer than 30 samples (not enough data).
    """
    if len(current) < 30:
        return 0.0

    # Build bins from reference distribution
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    # Count observations in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    cur_counts = np.histogram(current,   bins=breakpoints)[0]

    # Convert to proportions — add small epsilon to avoid division by zero
    eps = 1e-8
    ref_pct = (ref_counts + eps) / (len(reference) + eps * n_bins)
    cur_pct = (cur_counts + eps) / (len(current)   + eps * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi, 6)


# ── Chi² ───────────────────────────────────────────────────────────────────────
def compute_chi2_pvalue(reference_dist: dict, current_series: pd.Series) -> float:
    """
    Compute chi² test p-value between reference category distribution and current.

    reference_dist: {category: proportion} from training data
    current_series: raw current values (not proportions)

    Returns p-value. Low p-value (< 0.05) means drift detected.
    Returns 1.0 if current has fewer than 30 samples.
    """
    if len(current_series) < 30:
        return 1.0

    all_categories = set(reference_dist.keys()) | set(current_series.unique())

    ref_counts = np.array([
        reference_dist.get(cat, 0.0) * len(current_series)
        for cat in all_categories
    ])
    cur_counts = np.array([
        (current_series == cat).sum()
        for cat in all_categories
    ])

    # chi2_contingency needs a 2D table: [reference, current]
    table = np.array([ref_counts, cur_counts])

    # Avoid test if any expected count is zero
    if (ref_counts == 0).any():
        return 1.0

    _, p_value, _, _ = chi2_contingency(table)
    return round(float(p_value), 6)


# ── Output drift ───────────────────────────────────────────────────────────────
def compute_output_drift(reference_positive_rate: float, current_labels: pd.Series) -> float:
    """
    Compute drift in output distribution — difference in positive rate.
    Returns absolute difference between reference and current positive rates.
    """
    if len(current_labels) < 30:
        return 0.0
    current_positive_rate = float(current_labels.mean())
    return round(abs(current_positive_rate - reference_positive_rate), 6)


# ── Severity classifier ────────────────────────────────────────────────────────
def classify_severity(
    psi_scores: dict[str, float],
    chi2_pvals: dict[str, float],
    output_drift: float,
) -> DriftSeverity:
    """
    Classify overall drift severity based on PSI, chi², and output drift.

    Rules (in order of priority):
    1. Any PSI > 0.2          → high
    2. Any chi² p-val < 0.05  → high
    3. Output drift > 0.05    → high
    4. Any PSI 0.1–0.2        → warning
    5. Otherwise              → none
    """
    max_psi     = max(psi_scores.values(), default=0.0)
    min_chi2    = min(chi2_pvals.values(), default=1.0)
    drifted_psi = [f for f, v in psi_scores.items() if v > PSI_HIGH]
    drifted_chi = [f for f, v in chi2_pvals.items() if v < CHI2_PVALUE]

    if max_psi > PSI_HIGH:
        return DriftSeverity(
            level="high",
            reason=f"PSI > {PSI_HIGH} on features: {drifted_psi}",
        )
    if min_chi2 < CHI2_PVALUE:
        return DriftSeverity(
            level="high",
            reason=f"chi² drift detected on features: {drifted_chi}",
        )
    if output_drift > 0.05:
        return DriftSeverity(
            level="high",
            reason=f"Output distribution shifted by {output_drift:.2%}",
        )
    if max_psi > PSI_WARNING:
        warning_features = [f for f, v in psi_scores.items() if v > PSI_WARNING]
        return DriftSeverity(
            level="warning",
            reason=f"PSI warning on features: {warning_features}",
        )

    return DriftSeverity(level="none", reason="No drift detected.")


# ── Main entry point ───────────────────────────────────────────────────────────
def compute_drift_report(
    recent_predictions: pd.DataFrame,
    model_uri: str,
) -> DriftAlert:
    """
    Compute full drift report from recent predictions vs reference stats.

    recent_predictions: DataFrame returned by predictor.get_recent_predictions()
    model_uri: current model URI (included in alert payload)

    Returns a DriftAlert ready to be sent as a webhook to the agent.
    """
    # Load reference statistics saved during training
    if not REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Reference stats not found at {REFERENCE_STATS_PATH}. "
            "Run train.py first."
        )

    with open(REFERENCE_STATS_PATH) as f:
        ref = json.load(f)

    numeric_ref    = ref["numeric"]
    categorical_ref = ref["categorical"]
    output_ref     = ref["output"]

    if recent_predictions.empty:
        logger.warning("No recent predictions available for drift computation.")
        return DriftAlert(
            model_uri=model_uri,
            severity=DriftSeverity(level="none", reason="No predictions yet."),
            drift_report=DriftFeatureReport(
                psi_scores={},
                chi2_pvals={},
                output_drift=0.0,
            ),
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )

    # ── Compute PSI for numeric features ──────────────────────────────────────
    psi_scores = {}
    for col, stats in numeric_ref.items():
        if col not in recent_predictions.columns:
            continue
        # Reconstruct reference distribution from mean/std using normal approximation
        rng = np.random.default_rng(42)
        reference_samples = rng.normal(
            loc=stats["mean"],
            scale=max(stats["std"], 1e-6),
            size=1000,
        )
        reference_samples = np.clip(reference_samples, stats["min"], stats["max"])
        current_samples   = recent_predictions[col].dropna().values
        psi_scores[col]   = compute_psi(reference_samples, current_samples)

    # ── Compute chi² for categorical features ─────────────────────────────────
    chi2_pvals = {}
    for col, ref_dist in categorical_ref.items():
        if col not in recent_predictions.columns:
            continue
        current_series  = recent_predictions[col].dropna()
        chi2_pvals[col] = compute_chi2_pvalue(ref_dist, current_series)

    # ── Compute output drift ──────────────────────────────────────────────────
    output_drift = 0.0
    if "__label__" in recent_predictions.columns:
        output_drift = compute_output_drift(
            output_ref["positive_rate"],
            recent_predictions["__label__"],
        )

    # ── Classify severity ─────────────────────────────────────────────────────
    severity = classify_severity(psi_scores, chi2_pvals, output_drift)

    logger.info(
        f"Drift report computed | severity={severity.level} | "
        f"max_psi={max(psi_scores.values(), default=0):.4f} | "
        f"output_drift={output_drift:.4f}"
    )

    return DriftAlert(
        model_uri=model_uri,
        severity=severity,
        drift_report=DriftFeatureReport(
            psi_scores=psi_scores,
            chi2_pvals=chi2_pvals,
            output_drift=output_drift,
        ),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )