from __future__ import annotations
"""Agent-side state and structured outputs for the LangGraph supervisor."""

from typing import Any, Literal, NotRequired, Required, TypedDict

from pydantic import BaseModel, Field

from model_service.schemas import DriftAlert


class TriageDecision(BaseModel):
    """Structured output from the triage sub-agent."""

    severity_level: Literal["none", "warning", "high"] = Field(
        ...,
        description="Normalized drift severity taken from the alert.",
    )
    recommended_route: Literal[
        "monitor",
        "replay_test",
        "retrain",
        "rollback",
        "request_promotion_review",
    ] = Field(
        ...,
        description="Deterministic routing hint for the action node.",
    )
    rationale: str = Field(
        ...,
        description="Short explanation suitable for dashboards and regression fixtures.",
    )


class QueueJobSpec(BaseModel):
    """Description of work for the Redis-backed worker (enqueue happens outside this node)."""

    task: Literal["replay", "retrain", "rollback"] = Field(...)
    idempotency_key: str = Field(
        ...,
        description="Stable key so retries do not duplicate side effects.",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON-serializable payload consumed by the worker.",
    )


class ActionDecision(BaseModel):
    """Structured output from the action sub-agent."""

    route_taken: Literal[
        "noop_monitor",
        "queued_tools",
        "halt_for_human_promotion",
    ] = Field(...)
    jobs: list[QueueJobSpec] = Field(default_factory=list)
    requires_human_approval: bool = Field(
        default=False,
        description="When True, the supervisor should pause for dashboard HIL.",
    )
    approval_prompt: str | None = Field(
        default=None,
        description="Text loaded from prompts/ — surfaced on the approval inbox.",
    )


class CommsSummary(BaseModel):
    """Operator-facing summary for the dashboard / audit trail."""

    message: str = Field(...)
    investigation_status: Literal[
        "monitoring",
        "open",
        "awaiting_approval",
        "queued",
        "resolved",
    ] = Field(...)


class InvestigationState(TypedDict, total=False):
    """LangGraph checkpoint state — prefer JSON-serializable values."""

    investigation_id: Required[str]
    drift_alert: Required[dict[str, Any]]
    triage: NotRequired[dict[str, Any]]
    action: NotRequired[dict[str, Any]]
    comms: NotRequired[dict[str, Any]]
    triage_prompt_rendered: NotRequired[str]
    action_prompt_rendered: NotRequired[str]
    comms_prompt_rendered: NotRequired[str]
    hil_approval: NotRequired[dict[str, Any]]


def split_prompt_sections(text: str) -> dict[str, str]:
    """
    Split a prompt file into sections headed by lines like '### SOME_NAME'.

    Text before the first heading is stored under '_preamble'.
    """

    sections: dict[str, str] = {}
    current = "_preamble"
    buf: list[str] = []

    for raw_line in text.splitlines():
        if raw_line.startswith("### ") and len(raw_line) > 4:
            sections[current] = "\n".join(buf).strip()
            current = raw_line[4:].strip()
            buf = []
        else:
            buf.append(raw_line)

    sections[current] = "\n".join(buf).strip()
    return sections


__all__ = [
    "DriftAlert",
    "InvestigationState",
    "TriageDecision",
    "QueueJobSpec",
    "ActionDecision",
    "CommsSummary",
    "split_prompt_sections",
]