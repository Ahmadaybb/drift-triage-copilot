from __future__ import annotations
"""
webhook_server.py — Receives drift webhooks from model service.
OWNER: Person 2
STATUS: 🔲 TODO
"""

"""

Endpoint:
    POST /webhook/drift → validate DriftAlert → open new investigation (LangGraph)
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from langgraph.types import Command
from pydantic import BaseModel, Field

from agent_service.graph import (
    build_investigation_graph,
    investigation_thread_config,
    postgres_checkpointer,
)
from model_service.schemas import DriftAlert

load_dotenv()

PENDING_INVESTIGATIONS: dict[str, dict[str, Any]] = {}

DEFAULT_ALERT: dict[str, Any] = {
    "timestamp": None,
    "model_uri": "",
    "severity": {
        "level": "none",
        "reason": "No drift detected.",
    },
    "drift_report": {
        "psi_scores": {},
        "chi2_pvals": {},
        "output_drift": 0.0,
    },
}


class DriftWebhookResponse(BaseModel):
    investigation_id: str = Field(..., description="Stable id for checkpoints / dashboard")
    status: str = Field(..., description="Pipeline outcome hint for the caller")


class ResumeInvestigationRequest(BaseModel):
    investigation_id: str
    approved: bool


@asynccontextmanager
async def _lifespan(app: FastAPI):
    with postgres_checkpointer() as saver:
        app.state.graph = build_investigation_graph(saver)
        app.state.latest_alert = DEFAULT_ALERT.copy()
        yield


app = FastAPI(title="Drift Triage Agent", lifespan=_lifespan)


def _severity_level(alert: DriftAlert) -> str:
    sev = getattr(alert, "severity", None)

    if isinstance(sev, dict):
        return str(sev.get("level", "unknown"))

    return str(getattr(sev, "level", "unknown"))


def _severity_reason(alert: DriftAlert) -> str:
    sev = getattr(alert, "severity", None)

    if isinstance(sev, dict):
        return str(sev.get("reason", "Awaiting human approval"))

    return str(getattr(sev, "reason", "Awaiting human approval"))


def _model_uri(alert: DriftAlert) -> str:
    return str(getattr(alert, "model_uri", "unknown"))


@app.post("/webhook/drift", response_model=DriftWebhookResponse)
async def post_drift_webhook(
    alert: DriftAlert,
    request: Request,
) -> DriftWebhookResponse:
    investigation_id = str(uuid.uuid4())

    alert_payload = alert.model_dump(mode="json")
    request.app.state.latest_alert = alert_payload

    initial_state: dict[str, Any] = {
        "investigation_id": investigation_id,
        "drift_alert": alert_payload,
    }

    config = investigation_thread_config(investigation_id)
    graph = request.app.state.graph

    def _invoke() -> dict[str, Any]:
        return graph.invoke(initial_state, config=config)

    result = await asyncio.to_thread(_invoke)

    pending = result.get("__interrupt__")

    if pending:
        status = "awaiting_approval"

        action = result.get("action") or {}
        jobs = action.get("jobs") or []

        proposed_action = "approval"
        if jobs and isinstance(jobs, list):
            proposed_action = str(jobs[0].get("task", "approval"))

        PENDING_INVESTIGATIONS[investigation_id] = {
            "investigation_id": investigation_id,
            "model_uri": _model_uri(alert),
            "proposed_action": proposed_action,
            "rationale": _severity_reason(alert),
            "severity": _severity_level(alert),
        }

    else:
        comms = result.get("comms") or {}
        status = str(comms.get("investigation_status") or "completed")

    return DriftWebhookResponse(
        investigation_id=investigation_id,
        status=status,
    )


@app.get("/latest-alert")
async def latest_alert(request: Request) -> dict[str, Any]:
    return request.app.state.latest_alert


@app.get("/investigations/pending")
async def get_pending_investigations() -> list[dict[str, Any]]:
    return list(PENDING_INVESTIGATIONS.values())


@app.post("/investigations/resume", response_model=DriftWebhookResponse)
async def resume_investigation(
    body: ResumeInvestigationRequest,
    request: Request,
) -> DriftWebhookResponse:
    config = investigation_thread_config(body.investigation_id)
    graph = request.app.state.graph

    def _invoke() -> dict[str, Any]:
        return graph.invoke(
            Command(resume={"approved": body.approved}),
            config=config,
        )

    result = await asyncio.to_thread(_invoke)

    pending = result.get("__interrupt__")

    if pending:
        status = "awaiting_human_approval"
    else:
        comms = result.get("comms") or {}
        status = str(comms.get("investigation_status") or "completed")

    PENDING_INVESTIGATIONS.pop(body.investigation_id, None)

    return DriftWebhookResponse(
        investigation_id=body.investigation_id,
        status=status,
    )


def get_app() -> FastAPI:
    return app