from __future__ import annotations
"""
graph.py — LangGraph supervisor graph.
OWNER: Person 2
STATUS: 🔲 TODO

Topology:
    supervisor → triage → action → comms → supervisor
    supervisor pauses for human approval before Production actions
    every node checkpoints to Postgres after completion
"""

"""LangGraph pipeline: triage → action → [human approval] → comms, with Postgres checkpoints."""

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent_service.nodes.action import action_node
from agent_service.nodes.comms import comms_node
from agent_service.nodes.triage import triage_node
from agent_service.schemas import InvestigationState, QueueJobSpec
from pydantic import TypeAdapter

from agent_service.queue_client import enqueue_queue_jobs
from agent_service.checkpoints import (
    postgres_checkpointer,
    postgres_conn_string,
)


def route_after_action(state: InvestigationState) -> Literal["human_approval", "enqueue_jobs"]:
    """Production-impacting paths pause for HIL before comms."""

    action = state.get("action")
    if isinstance(action, dict) and action.get("requires_human_approval"):
        return "human_approval"
    return "enqueue_jobs"


def human_approval_node(state: InvestigationState) -> dict[str, Any]:
    action = state["action"]

    resume_payload = interrupt(
        {
            "investigation_id": state["investigation_id"],
            "approval_prompt": action.get("approval_prompt"),
            "jobs": action.get("jobs", []),
            "route_taken": action.get("route_taken"),
        }
    )

    if isinstance(resume_payload, dict):
        return {"hil_approval": resume_payload}

    return {"hil_approval": {"approved": bool(resume_payload)}}


def enqueue_jobs_node(state: InvestigationState) -> dict[str, Any]:
    raw = state.get("action")

    if not isinstance(raw, dict):
        return {}

    if raw.get("route_taken") != "queued_tools":
        return {}

    jobs_raw = raw.get("jobs") or []

    if not jobs_raw:
        return {}

    if raw.get("requires_human_approval"):
        hil = state.get("hil_approval")

        if not isinstance(hil, dict):
            return {}

        if not hil.get("approved"):
            return {}

    specs = TypeAdapter(list[QueueJobSpec]).validate_python(jobs_raw)

    enqueue_queue_jobs(specs)

    return {}


def build_investigation_graph(checkpointer: Any):
    builder: StateGraph[InvestigationState] = StateGraph(InvestigationState)

    builder.add_node("triage", triage_node)
    builder.add_node("action", action_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("enqueue_jobs", enqueue_jobs_node)
    builder.add_node("comms", comms_node)

    builder.add_edge(START, "triage")
    builder.add_edge("triage", "action")
    builder.add_conditional_edges(
        "action",
        route_after_action,
        {"human_approval": "human_approval", 
         "enqueue_jobs": "enqueue_jobs",
        },
    )
    builder.add_edge("human_approval", "enqueue_jobs")
    builder.add_edge("enqueue_jobs", "comms")
    builder.add_edge("comms", END)

    return builder.compile(checkpointer=checkpointer)


def investigation_thread_config(thread_id: str) -> dict[str, Any]:
    """Use a stable thread_id per investigation so resumes hit the same checkpoint."""

    return {"configurable": {"thread_id": thread_id}}

__all__ = [
    "build_investigation_graph",
    "human_approval_node",
    "enqueue_jobs_node",
    "investigation_thread_config",
    "postgres_checkpointer",
    "postgres_conn_string",
    "route_after_action",
]
