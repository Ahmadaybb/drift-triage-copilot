"""
registry_gate.py — Promotion checklist gate.
OWNER: Person 1
STATUS: Done

Responsibilities:
- Assert all checklist items before promoting any version to Production:
    1. Model has schema artifact
    2. Model has model card artifact
    3. Model has SHA-256 hash
    4. Test AUC >= 0.75
    5. Test Recall >= 0.75
    6. Operating threshold is set
    7. Request came from the agent (not called directly)
- Raise structured error if any check fails
"""
"""
registry_gate.py — Promotion checklist gate.
OWNER: Person 1

Refuses to promote any model version to Production unless
all checklist items pass. Called by app.py /promote endpoint.

No model reaches Production without going through this gate.
"""

import logging
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
MIN_AUC    = 0.75
MIN_RECALL = 0.75


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class GateResult:
    """
    Result of running the promotion checklist.

    passed  → True if all checks passed, False if any failed
    checks  → dict of {check_name: (passed, reason)}
    """
    passed: bool
    checks: dict[str, tuple[bool, str]]

    def failed_checks(self) -> list[str]:
        """Return list of check names that failed."""
        return [name for name, (ok, _) in self.checks.items() if not ok]

    def summary(self) -> str:
        """Human-readable summary of all checks."""
        lines = ["PROMOTION GATE RESULTS", "=" * 40]
        for name, (ok, reason) in self.checks.items():
            status = "✅ PASS" if ok else "❌ FAIL"
            lines.append(f"  {status} | {name}: {reason}")
        lines.append("=" * 40)
        lines.append(f"  OVERALL: {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


# ── Gate ───────────────────────────────────────────────────────────────────────
def run_promotion_gate(
    model_name: str,
    version: str,
    requested_by: str,
    investigation_id: str,
) -> GateResult:
    """
    Run all promotion checklist items for a given model version.

    model_name      : registered model name in MLflow
    version         : version number to promote (e.g. "2")
    requested_by    : must be "human" — direct calls are rejected
    investigation_id: agent investigation ID that triggered this

    Returns GateResult with passed=True only if ALL checks pass.
    """
    checks = {}

    # ── Check 1: Request must come from agent with human approval ──────────────
    checks["approved_by_human"] = (
        requested_by == "human",
        f"requested_by='{requested_by}' — must be 'human'",
    )

    # ── Check 2: Investigation ID must be present ──────────────────────────────
    checks["has_investigation_id"] = (
        bool(investigation_id and investigation_id.strip()),
        f"investigation_id='{investigation_id}'",
    )

    # ── Load MLflow run for remaining checks ───────────────────────────────────
    client = MlflowClient()

    try:
        model_version = client.get_model_version(model_name, version)
        run           = client.get_run(model_version.run_id)
        params        = run.data.params
        metrics       = run.data.metrics
        artifacts     = [
            a.path for a in
            client.list_artifacts(model_version.run_id)
        ]
    except Exception as e:
        # If we can't load the run, fail all remaining checks
        logger.error(f"Could not load MLflow run for {model_name} v{version}: {e}")
        for check in [
            "has_schema_artifact",
            "has_model_card",
            "has_artifact_hash",
            "auc_above_threshold",
            "recall_above_threshold",
            "has_operating_threshold",
        ]:
            checks[check] = (False, f"Could not load MLflow run: {e}")
        passed = all(ok for ok, _ in checks.values())
        return GateResult(passed=passed, checks=checks)

    # ── Check 3: Schema artifact exists ───────────────────────────────────────
    has_schema = any("schema.json" in a for a in artifacts)
    checks["has_schema_artifact"] = (
        has_schema,
        "schema.json found in artifacts" if has_schema else "schema.json MISSING from artifacts",
    )

    # ── Check 4: Model card exists ─────────────────────────────────────────────
    has_card = any("model_card.md" in a for a in artifacts)
    checks["has_model_card"] = (
        has_card,
        "model_card.md found in artifacts" if has_card else "model_card.md MISSING from artifacts",
    )

    # ── Check 5: Artifact hash exists ─────────────────────────────────────────
    artifact_hash = params.get("artifact_hash", "")
    has_hash      = bool(artifact_hash and len(artifact_hash) > 10)
    checks["has_artifact_hash"] = (
        has_hash,
        f"hash={artifact_hash[:12]}..." if has_hash else "artifact_hash MISSING from params",
    )

    # ── Check 6: AUC above threshold ──────────────────────────────────────────
    test_auc = metrics.get("test_auc", 0.0)
    auc_ok   = test_auc >= MIN_AUC
    checks["auc_above_threshold"] = (
        auc_ok,
        f"test_auc={test_auc:.4f} >= {MIN_AUC}" if auc_ok
        else f"test_auc={test_auc:.4f} < {MIN_AUC} — BELOW MINIMUM",
    )

    # ── Check 7: Recall above threshold ───────────────────────────────────────
    test_recall = metrics.get("test_recall", 0.0)
    recall_ok   = test_recall >= MIN_RECALL
    checks["recall_above_threshold"] = (
        recall_ok,
        f"test_recall={test_recall:.4f} >= {MIN_RECALL}" if recall_ok
        else f"test_recall={test_recall:.4f} < {MIN_RECALL} — BELOW MINIMUM",
    )

    # ── Check 8: Operating threshold is set ───────────────────────────────────
    op_threshold = metrics.get("operating_threshold", 0.0)
    has_threshold = op_threshold > 0.0
    checks["has_operating_threshold"] = (
        has_threshold,
        f"operating_threshold={op_threshold:.6f}" if has_threshold
        else "operating_threshold MISSING or zero",
    )

    # ── Final result ───────────────────────────────────────────────────────────
    passed = all(ok for ok, _ in checks.values())

    result = GateResult(passed=passed, checks=checks)
    logger.info(result.summary())

    return result