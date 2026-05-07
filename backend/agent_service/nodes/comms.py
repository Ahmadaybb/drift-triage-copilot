from __future__ import annotations
"""
nodes/comms.py — Comms sub-agent.
OWNER: Person 2
STATUS: 🔲 TODO

Input:  action result
Output: human-readable summary, investigation status updated
"""
"""Comms node — operator-facing summary from ActionDecision (deterministic)."""


from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from agent_service.schemas import ActionDecision, CommsSummary, InvestigationState, split_prompt_sections

_AGENT_ROOT = Path(__file__).resolve().parents[1]
_COMMS_PROMPT_PATH = _AGENT_ROOT / "prompts" / "comms.txt"


def _safe_format(template: str, **kwargs: str) -> str:
    class _Missing(dict[str, str]):
        def __missing__(self, key: str) -> str:
            return "{" + key + "}"

    return template.format_map(_Missing(kwargs))


def _status(action: ActionDecision, state: InvestigationState) -> str:
    if action.route_taken == "noop_monitor":
        return "monitoring"

    if action.route_taken == "halt_for_human_promotion":
        return "awaiting_approval"

    if action.route_taken == "queued_tools":
        if action.requires_human_approval:
            hil = state.get("hil_approval")

            if isinstance(hil, dict) and hil.get("approved"):
                return "queued"

            return "awaiting_approval"

        return "queued"

    return "open"


def comms_node(state: InvestigationState) -> dict[str, Any]:
    investigation_id = state["investigation_id"]

    sections: dict[str, str] = {}
    if _COMMS_PROMPT_PATH.is_file():
        sections = split_prompt_sections(_COMMS_PROMPT_PATH.read_text(encoding="utf-8"))

    try:
        action = TypeAdapter(ActionDecision).validate_python(state.get("action"))
    except ValidationError as exc:
        tmpl = sections.get("SUMMARY_INVALID_ACTION", "").strip()
        msg = (
            _safe_format(tmpl, investigation_id=investigation_id, error_detail=repr(exc.errors()))
            if tmpl
            else f"[{investigation_id}] Invalid action payload."
        )
        return {"comms": CommsSummary(message=msg, investigation_status="open").model_dump()}

    status = _status(action, state)
    tasks = ", ".join(j.task for j in action.jobs)
    approval = action.approval_prompt or ""

    if status == "monitoring":
        tmpl_key = "SUMMARY_MONITORING"
    elif status == "awaiting_approval":
        tmpl_key = "SUMMARY_AWAITING_APPROVAL"
    elif status == "queued":
        tmpl_key = "SUMMARY_QUEUED"
    else:
        tmpl_key = "SUMMARY_OPEN"

    tmpl = sections.get(tmpl_key, "").strip()
    msg = (
        _safe_format(
            tmpl,
            investigation_id=investigation_id,
            route_taken=action.route_taken,
            tasks=tasks,
            approval_prompt=approval,
            requires_human_approval=str(action.requires_human_approval),
        )
        if tmpl
        else (
            f"[{investigation_id}] {action.route_taken} tasks=[{tasks}] "
            f"approval={approval!r} status={status}"
        )
    )

    return {"comms": CommsSummary(message=msg, investigation_status=status).model_dump()}