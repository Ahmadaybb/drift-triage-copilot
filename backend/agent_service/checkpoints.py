from __future__ import annotations

"""
checkpoints.py — Postgres checkpoint setup for LangGraph.
OWNER: Person 2
STATUS: 🔲 TODO

Ensures agent state survives crashes and restarts.
Killing the agent mid-investigation must resume from last checkpoint.
"""
"""Postgres-backed LangGraph checkpoint store for the agent service."""


import os
from contextlib import contextmanager
from typing import Any, Iterator


def postgres_conn_string() -> str:
    """Prefer DATABASE_URL; otherwise POSTGRES_* (.env.example)."""

    explicit = os.environ.get("DATABASE_URL")
    if explicit:
        return explicit
    user = os.environ.get("POSTGRES_USER", "drift")
    password = os.environ.get("POSTGRES_PASSWORD", "drift")
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "drift_triage")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


@contextmanager
def postgres_checkpointer() -> Iterator[Any]:
    """Sync PostgresSaver context manager; calls setup() when available."""

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise ImportError(
            "Postgres checkpoints require: pip install langgraph-checkpoint-postgres"
        ) from exc

    uri = postgres_conn_string()
    with PostgresSaver.from_conn_string(uri) as saver:
        setup = getattr(saver, "setup", None)
        if callable(setup):
            setup()
        yield saver


__all__ = ["postgres_conn_string", "postgres_checkpointer"]