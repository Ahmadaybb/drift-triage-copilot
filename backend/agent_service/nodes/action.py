from __future__ import annotations
"""
nodes/action.py — Action sub-agent.
OWNER: Person 2
STATUS: 🔲 TODO

Input:  triage result
Output: job dispatched to Redis queue (replay / retrain / rollback)
        graph pauses here for human approval
"""

"""Action node — turn triage routing into prepared queue jobs (no execution)."""

from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from agent_service.schemas import (
    ActionDecision,
    InvestigationState,
    QueueJobSpec,
    TriageDecision,
    split_prompt_sections,
)

_AGENT_ROOT = Path(__file__).resolve().parents[1]
_ACTION_PROMPT_PATH = _AGENT_ROOT / "prompts" / "action.txt"


def _safe_format(template: str, **kwargs: str) -> str:
    class _Missing(dict[str, str]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_Missing(kwargs))


def _extract_model_name(model_uri: str) -> str:
    """Best-effort label for worker payloads; falls back to full URI."""
    uri = model_uri.rstrip("/")
    if not uri:
        return ""
    if "/" in uri:
        parts = uri.split("/")
        return parts[-2] if len(parts) >= 2 else parts[-1]
    return uri


def action_node(state: InvestigationState) -> dict[str, Any]:
    investigation_id = state["investigation_id"]
    drift_alert = state["drift_alert"]
    model_uri = str(drift_alert.get("model_uri", ""))

    sections: dict[str, str] = {}
    render_body = ""
    if _ACTION_PROMPT_PATH.is_file():
        raw = _ACTION_PROMPT_PATH.read_text(encoding="utf-8")
        sections = split_prompt_sections(raw)
        render_body = sections.get("RENDER") or sections.get("_preamble") or ""

    try:
        triage = TypeAdapter(TriageDecision).validate_python(state.get("triage"))
    except ValidationError as exc:
        snippet = sections.get("APPROVAL_TRIAGE_INVALID", "").strip()
        approval = _safe_format(snippet, error_detail=repr(exc.errors())) if snippet else None
        decision = ActionDecision(
            route_taken="halt_for_human_promotion",
            jobs=[],
            requires_human_approval=True,
            approval_prompt=approval,
        )
        out: dict[str, Any] = {"action": decision.model_dump()}
        if render_body:
            out["action_prompt_rendered"] = _safe_format(
                render_body,
                investigation_id=investigation_id,
                model_uri=model_uri,
                recommended_route="",
                rationale="invalid triage payload",
            )
        return out

    route = triage.recommended_route

    if render_body:
        action_prompt_rendered = _safe_format(
            render_body,
            investigation_id=investigation_id,
            model_uri=model_uri,
            recommended_route=route,
            rationale=triage.rationale,
        )
    else:
        action_prompt_rendered = ""

    def _payload() -> dict[str, Any]:
        return {
            "investigation_id": investigation_id,
            "model_uri": model_uri,
            "model_name": _extract_model_name(model_uri),
            "trigger_severity": triage.severity_level,
        }

    if route == "monitor":
        decision = ActionDecision(
            route_taken="noop_monitor",
            jobs=[],
            requires_human_approval=False,
            approval_prompt=None,
        )
        out = {"action": decision.model_dump()}
        if action_prompt_rendered:
            out["action_prompt_rendered"] = action_prompt_rendered
        return out

    if route == "request_promotion_review":
        snippet = sections.get("APPROVAL_PROMOTION", "").strip()
        approval = _safe_format(snippet, investigation_id=investigation_id, model_uri=model_uri) if snippet else None
        decision = ActionDecision(
            route_taken="halt_for_human_promotion",
            jobs=[],
            requires_human_approval=True,
            approval_prompt=approval,
        )
        out = {"action": decision.model_dump()}
        if action_prompt_rendered:
            out["action_prompt_rendered"] = action_prompt_rendered
        return out

    task_map = {
        "replay_test": "replay",
        "retrain": "retrain",
        "rollback": "rollback",
    }
    task = task_map.get(route)
    if task is None:
        snippet = sections.get("APPROVAL_UNKNOWN_ROUTE", "").strip()
        approval = (
            _safe_format(snippet, investigation_id=investigation_id, route=route)
            if snippet
            else None
        )
        decision = ActionDecision(
            route_taken="halt_for_human_promotion",
            jobs=[],
            requires_human_approval=True,
            approval_prompt=approval,
        )
        out = {"action": decision.model_dump()}
        if action_prompt_rendered:
            out["action_prompt_rendered"] = action_prompt_rendered
        return out

    idempotency_key = f"{investigation_id}:{task}"
    job = QueueJobSpec(
        task=task,
        idempotency_key=idempotency_key,
        payload=_payload(),
    )

    production_impacting = route in {"retrain", "rollback"}
    if production_impacting:
        key = "APPROVAL_RETRAIN" if route == "retrain" else "APPROVAL_ROLLBACK"
        snippet = sections.get(key, "").strip()
        approval = (
            _safe_format(snippet, investigation_id=investigation_id, model_uri=model_uri)
            if snippet
            else None
        )
    else:
        approval = None

    decision = ActionDecision(
        route_taken="queued_tools",
        jobs=[job],
        requires_human_approval=production_impacting,
        approval_prompt=approval,
    )
    out = {"action": decision.model_dump()}
    if action_prompt_rendered:
        out["action_prompt_rendered"] = action_prompt_rendered
    return out