from __future__ import annotations

"""
test_agent_trajectories.py — Snapshot trajectory tests.
OWNER: Person 2
STATUS: 🔲 TODO

- Run LangGraph agent on a recorded drift event
- Mock the LLM — runs without an API key
- Assert agent trajectory matches recorded fixture exactly
- Must pass on every CI push — refuses to merge if it regresses
"""
"""Deterministic LangGraph trajectory checks — no LLM, no Redis."""

from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent_service.graph import (
    build_investigation_graph,
    investigation_thread_config,
)


def _drift_alert_payload(severity_level: str) -> dict:
    return {
        "event": "drift_alert",
        "model_uri": "models:/bank-marketing-classifier/Production",
        "severity": {"level": severity_level, "reason": "fixture"},
        "drift_report": {
            "psi_scores": {"age": 0.05},
            "chi2_pvals": {"job": 1.0},
            "output_drift": 0.01,
        },
        "timestamp": "2026-01-01T00:00:00",
    }


@patch("agent_service.graph.enqueue_queue_jobs")
def test_trajectory_monitor_no_queue(mock_enqueue):
    graph = build_investigation_graph(MemorySaver())
    tid = "traj-monitor"
    cfg = investigation_thread_config(tid)
    result = graph.invoke(
        {"investigation_id": tid, "drift_alert": _drift_alert_payload("none")},
        config=cfg,
    )
    assert result["triage"]["recommended_route"] == "monitor"
    assert result["action"]["route_taken"] == "noop_monitor"
    assert result["comms"]["investigation_status"] == "monitoring"
    mock_enqueue.assert_not_called()


@patch("agent_service.graph.enqueue_queue_jobs")
def test_trajectory_replay_enqueues_without_hil(mock_enqueue):
    graph = build_investigation_graph(MemorySaver())
    tid = "traj-replay"
    cfg = investigation_thread_config(tid)
    result = graph.invoke(
        {"investigation_id": tid, "drift_alert": _drift_alert_payload("warning")},
        config=cfg,
    )
    assert result["triage"]["recommended_route"] == "replay_test"
    assert result["action"]["route_taken"] == "queued_tools"
    assert result["action"]["requires_human_approval"] is False
    assert result["comms"]["investigation_status"] == "queued"
    mock_enqueue.assert_called_once()


@patch("agent_service.graph.enqueue_queue_jobs")
def test_trajectory_retrain_interrupt_then_resume_enqueues(mock_enqueue):
    graph = build_investigation_graph(MemorySaver())
    tid = "traj-retrain"
    cfg = investigation_thread_config(tid)
    first = graph.invoke(
        {"investigation_id": tid, "drift_alert": _drift_alert_payload("high")},
        config=cfg,
    )
    assert first.get("__interrupt__")
    assert first["action"]["requires_human_approval"] is True

    second = graph.invoke(Command(resume={"approved": True}), config=cfg)
    assert not second.get("__interrupt__")
    assert second["comms"]["investigation_status"] == "queued"
    mock_enqueue.assert_called_once()