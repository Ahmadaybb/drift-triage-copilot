from __future__ import annotations
"""
nodes/triage.py — Triage sub-agent.
OWNER: Person 2
STATUS: 🔲 TODO

Input:  DriftAlert payload
Output: severity classification + recommended next action
"""

"""Triage node — deterministic routing from drift severity (no LLM)."""


from pathlib import Path
from typing import Any, Literal, cast

from agent_service.schemas import InvestigationState, TriageDecision, split_prompt_sections

_AGENT_ROOT = Path(__file__).resolve().parents[1]
_TRIAGE_PROMPT_PATH = _AGENT_ROOT / "prompts" / "triage.txt"

_Severity = Literal["none", "warning", "high"]
_ROUTE = Literal["monitor", "replay_test", "retrain", "rollback", "request_promotion_review"]


def _safe_format(template: str, **kwargs: str) -> str:
    class _Missing(dict[str, str]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_Missing(kwargs))


def triage_node(state: InvestigationState) -> dict[str, Any]:
    drift_alert = state.get("drift_alert", {})
    sev = drift_alert.get("severity")
    level_raw = sev.get("level") if isinstance(sev, dict) else None

    route_by_severity: dict[_Severity, _ROUTE] = {
        "none": "monitor",
        "warning": "replay_test",
        "high": "retrain",
    }

    if level_raw in route_by_severity:
        severity_level = cast(_Severity, level_raw)
        recommended_route = route_by_severity[severity_level]
        rationale = (
            "Severity none — monitor without dispatching slow tools."
            if severity_level == "none"
            else (
                "Severity warning — run replay_test before heavier remediation."
                if severity_level == "warning"
                else "Severity high — queue retrain path."
            )
        )
    else:
        severity_level = "high"
        recommended_route = "retrain"
        rationale = (
            f"Missing or unknown severity level ({level_raw!r}); fail-safe to high → retrain."
        )

    decision = TriageDecision(
        severity_level=severity_level,
        recommended_route=recommended_route,
        rationale=rationale,
    )

    out: dict[str, Any] = {"triage": decision.model_dump()}

    if _TRIAGE_PROMPT_PATH.is_file():
        raw_prompt = _TRIAGE_PROMPT_PATH.read_text(encoding="utf-8")
        sections = split_prompt_sections(raw_prompt)
        body = sections.get("RENDER") or sections.get("_preamble") or ""
        model_uri = str(drift_alert.get("model_uri", ""))
        reason = ""
        if isinstance(sev, dict):
            reason = str(sev.get("reason", ""))
        out["triage_prompt_rendered"] = _safe_format(
            body,
            investigation_id=state["investigation_id"],
            model_uri=model_uri,
            severity_level=str(level_raw if level_raw is not None else ""),
            recommended_route=recommended_route,
            severity_reason=reason,
        )

    return out