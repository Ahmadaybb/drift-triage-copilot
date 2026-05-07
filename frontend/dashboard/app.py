"""
Drift Triage Co-Pilot — Streamlit operations console.
Week 5 dashboard: registry, drift, investigations, HIL inbox, queue/DLQ.
"""

from __future__ import annotations
import time
import json
import os
import socket
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
import plotly.graph_objects as go
import redis
import streamlit as st

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drift Triage Co-Pilot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Env ───────────────────────────────────────────────────────────────────────
MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://localhost:8000").rstrip("/")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://localhost:8001").rstrip("/")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://localhost:5000").rstrip("/")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_QUEUE_KEY = os.getenv("REDIS_QUEUE_KEY", "drift_triage:jobs")
REDIS_DLQ_KEY = os.getenv("REDIS_DLQ_KEY", "drift_triage:jobs:dlq")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
MODEL_NAME = "bank-marketing-classifier"
REFRESH_SECONDS = int(os.getenv("DASHBOARD_REFRESH_SECONDS", "6"))

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown(
    f"""
<style>
    .stApp {{
        background: linear-gradient(165deg, #0d1117 0%, #161b22 45%, #0d1117 100%);
        color: #e6edf3;
    }}
    .main .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }}
    div[data-testid="stMetric"] {{
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 12px 14px;
    }}
    div[data-testid="stMetric"] label {{
        color: #8b949e !important;
    }}
    .hil-card {{
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }}
    .pill {{
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }}
    .pill-ok {{ background: #238636; color: #fff; }}
    .pill-warn {{ background: #9e6a03; color: #fff; }}
    .pill-bad {{ background: #da3633; color: #fff; }}
    .pill-muted {{ background: #30363d; color: #8b949e; }}
    .panel-title {{
        font-size: 1.05rem;
        font-weight: 600;
        color: #f0f6fc;
        margin-bottom: 0.35rem;
    }}
    .subtle {{ color: #8b949e; font-size: 0.85rem; }}
    div[data-testid="stExpander"] {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
    }}
</style>
""",
    unsafe_allow_html=True,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _tcp_ok(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def worker_ok() -> bool:
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
        )

        ts = r.get("worker:heartbeat")

        if not ts:
            return False

        age = time.time() - float(ts)

        return age < 15

    except Exception:
        return False


def _http_ok(url: str, timeout: float = 2.5) -> tuple[bool, int | None]:
    try:
        r = httpx.get(url, timeout=timeout)
        return r.status_code < 500, r.status_code
    except httpx.HTTPError:
        return False, None


def _badge(ok: bool, label: str) -> None:
    cls = "pill-ok" if ok else "pill-bad"
    st.markdown(f'<span class="pill {cls}">{label}</span>', unsafe_allow_html=True)


def _severity_color(level: str) -> str:
    return {"none": "#3fb950", "warning": "#d29922", "high": "#f85149"}.get(
        level.lower(), "#8b949e"
    )


def _fetch_json(url: str, timeout: float = 5.0) -> tuple[bool, Any]:
    try:
        r = httpx.get(url, timeout=timeout)
        if r.status_code != 200:
            return False, None
        return True, r.json()
    except Exception:
        return False, None


def _post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> tuple[bool, Any]:
    try:
        r = httpx.post(url, json=payload, timeout=timeout)
        try:
            body = r.json()
        except json.JSONDecodeError:
            body = {"raw": r.text[:500]}
        return r.status_code == 200, body
    except Exception as exc:
        return False, {"error": str(exc)}


def fetch_model_health() -> dict[str, Any] | None:
    ok, data = _fetch_json(f"{MODEL_SERVICE_URL}/health")
    return data if ok else None

def fetch_drift_report() -> dict[str, Any] | None:
    ok, data = _fetch_json(f"{AGENT_SERVICE_URL}/latest-alert")
    return data if ok else None


def fetch_mlflow_versions() -> list[dict[str, Any]]:
    """Registered model versions via MLflow REST."""
    try:
        # MLflow 2.x search API
        r = httpx.get(
            f"{MLFLOW_URL}/api/2.0/mlflow/model-versions/search",
            params={"filter": f"name='{MODEL_NAME}'"},
            timeout=5.0,
        )
        if r.status_code != 200:
            return []
        body = r.json()
        return body.get("model_versions") or []
    except Exception:
        return []


def redis_metrics() -> dict[str, Any]:
    out: dict[str, Any] = {
        "queue_depth": None,
        "dlq_depth": None,
        "queue_samples": [],
        "dlq_samples": [],
        "connected": False,
    }
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        out["connected"] = True
        out["queue_depth"] = int(r.llen(REDIS_QUEUE_KEY))
        out["dlq_depth"] = int(r.llen(REDIS_DLQ_KEY))
        raw_q = r.lrange(REDIS_QUEUE_KEY, 0, 12)
        raw_d = r.lrange(REDIS_DLQ_KEY, 0, 12)
        for s in raw_q:
            try:
                out["queue_samples"].append(json.loads(s))
            except json.JSONDecodeError:
                out["queue_samples"].append({"raw": s[:120]})
        for s in raw_d:
            try:
                out["dlq_samples"].append(json.loads(s))
            except json.JSONDecodeError:
                out["dlq_samples"].append({"raw": s[:120]})
    except redis.RedisError:
        pass
    return out


def ensure_session_lists() -> None:
    if "drift_history" not in st.session_state:
        st.session_state.drift_history = []
    if "investigations_open" not in st.session_state:
        st.session_state.investigations_open = []
    if "investigations_resolved" not in st.session_state:
        st.session_state.investigations_resolved = []
    if "pending_hil" not in st.session_state:
        st.session_state.pending_hil = []
    if "events_log" not in st.session_state:
        st.session_state.events_log = []


def seed_demo_if_empty() -> None:
    """Fallback demo content when backends have no investigation APIs."""
    ensure_session_lists()
    if st.session_state.get("demo_seed_disabled"):
        return
    if st.session_state.pending_hil:
        return
    st.session_state.pending_hil = [
        {
            "investigation_id": "demo-hil-retrain",
            "model_uri": f"models:/{MODEL_NAME}/Production",
            "proposed_action": "retrain",
            "rationale": "High drift severity — queued retrain awaiting operator approval.",
            "severity": "high",
        },
        {
            "investigation_id": "demo-hil-rollback",
            "model_uri": f"models:/{MODEL_NAME}/Production",
            "proposed_action": "rollback",
            "rationale": "Output distribution shift — rollback path staged behind HIL.",
            "severity": "high",
        },
        {
            "investigation_id": "demo-hil-promotion",
            "model_uri": f"models:/{MODEL_NAME}/Staging",
            "proposed_action": "promotion_review",
            "rationale": "Human gate before Production registry transition.",
            "severity": "warning",
        },
    ]
    if not st.session_state.investigations_open:
        st.session_state.investigations_open = [
            {
                "id": "demo-hil-retrain",
                "status": "awaiting_approval",
                "route": "retrain",
                "reason": "Drift severity high (demo)",
                "updated": _now_iso(),
            },
            {
                "id": "demo-open-replay",
                "status": "queued",
                "route": "replay_test",
                "reason": "PSI warning tier (demo)",
                "updated": _now_iso(),
            },
        ]
    if not st.session_state.investigations_resolved:
        st.session_state.investigations_resolved = [
            {
                "id": "demo-resolved-001",
                "status": "monitoring",
                "route": "monitor",
                "reason": "No actionable drift",
                "updated": _now_iso(),
            },
        ]


def try_fetch_agent_pending() -> list[dict[str, Any]]:
    """Optional future endpoint — keeps UI compatible."""
    for path in ("/investigations/pending", "/hil/pending"):
        ok, data = _fetch_json(f"{AGENT_SERVICE_URL}{path}", timeout=2.0)
        if ok and isinstance(data, list):
            return data
        if ok and isinstance(data, dict) and "items" in data:
            return list(data["items"])
    return []


def append_event(kind: str, detail: str) -> None:
    ensure_session_lists()
    st.session_state.events_log.insert(
        0, {"ts": _now_iso(), "kind": kind, "detail": detail}
    )
    st.session_state.events_log = st.session_state.events_log[:80]


def merge_drift_history(report: dict[str, Any] | None) -> None:
    ensure_session_lists()
    if not report:
        return
    row = {
        "ts": _now_iso(),
        "severity": (report.get("severity") or {}).get("level", "unknown"),
        "output_drift": (report.get("drift_report") or {}).get("output_drift"),
        "psi_max": max(
            ((report.get("drift_report") or {}).get("psi_scores") or {}).values(),
            default=None,
        ),
        "model_uri": report.get("model_uri"),
    }
    prev = st.session_state.drift_history[-1] if st.session_state.drift_history else None
    cur_sig = (row["severity"], row["output_drift"], row["psi_max"], row["model_uri"])
    prev_sig = (
        (prev["severity"], prev["output_drift"], prev["psi_max"], prev["model_uri"])
        if prev
        else None
    )
    if cur_sig != prev_sig:
        st.session_state.drift_history.append(row)
        st.session_state.drift_history = st.session_state.drift_history[-60:]


def render_header() -> None:
    st.markdown(
        '<p style="font-size:2rem;font-weight:700;color:#f0f6fc;margin:0;">Drift Triage Co-Pilot</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtle">Supervisor agent • drift-aware registry • HIL gates • Redis remediation queue</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    svc = [
        ("PostgreSQL", _tcp_ok(POSTGRES_HOST, POSTGRES_PORT)),
        ("Redis", _tcp_ok(REDIS_HOST, REDIS_PORT)),
        ("MLflow", _http_ok(f"{MLFLOW_URL}/health")[0]),
        ("Model svc", _http_ok(f"{MODEL_SERVICE_URL}/health")[0]),
        (
            "Agent svc",
            _http_ok(f"{AGENT_SERVICE_URL}/openapi.json")[0]
            or _http_ok(f"{AGENT_SERVICE_URL}/docs")[0],
        ),
    ]
    with cols[0]:
        _badge(svc[0][1], svc[0][0])
    with cols[1]:
        _badge(svc[1][1], svc[1][0])
    with cols[2]:
        _badge(svc[2][1], svc[2][0])
    with cols[3]:
        _badge(svc[3][1], svc[3][0])
    with cols[4]:
        _badge(svc[4][1], svc[4][0])
    with cols[5]:
        _badge(worker_ok(), "Worker")
        
    



def render_registry(health: dict[str, Any] | None, versions: list[dict[str, Any]]) -> None:
    st.markdown('<p class="panel-title">Registry state</p>', unsafe_allow_html=True)
    prod_v = staging_v = "—"
    for mv in versions:
        stage = (mv.get("current_stage") or "").lower()
        if stage == "production":
            prod_v = str(mv.get("version", prod_v))
        elif stage == "staging":
            staging_v = str(mv.get("version", staging_v))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Production version", prod_v)
    with c2:
        st.metric("Staging version", staging_v)
    with c3:
        th = health.get("threshold") if health else None
        st.metric("Operating threshold", f"{th:.4f}" if isinstance(th, (int, float)) else "—")
    with c4:
        pc = health.get("prediction_count") if health else None
        st.metric("Predictions logged", f"{pc:,}" if isinstance(pc, int) else "—")

    st.caption(f"Current serving URI (health): `{health.get('model_uri', '—') if health else '—'}`")

    if versions:
        df = pd.DataFrame(
            [
                {
                    "version": v.get("version"),
                    "stage": v.get("current_stage"),
                    "last_updated": v.get("last_updated_timestamp"),
                }
                for v in sorted(versions, key=lambda x: int(x.get("version") or 0), reverse=True)[
                    :8
                ]
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("MLflow version API unreachable — showing health URI only.")


def render_drift(report: dict[str, Any] | None, hist: list[dict[str, Any]]) -> None:
    st.markdown('<p class="panel-title">Drift monitoring</p>', unsafe_allow_html=True)
    if not report:
        st.warning("Drift report unavailable — model service may be down.")
        return

    sev = (report.get("severity") or {}).get("level", "unknown")
    reason = (report.get("severity") or {}).get("reason", "")
    color = _severity_color(sev)
    st.markdown(
        f'<span class="pill" style="background:{color};color:#0d1117;">SEVERITY: {sev.upper()}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="subtle">{reason}</p>', unsafe_allow_html=True)
    st.caption(f"Alert timestamp: `{report.get('timestamp', '—')}`")

    dr = report.get("drift_report") or {}
    psi = dr.get("psi_scores") or {}
    chi = dr.get("chi2_pvals") or {}
    od = dr.get("output_drift")

    m1, m2 = st.columns(2)
    with m1:
        if psi:
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=list(psi.keys()),
                        y=list(psi.values()),
                        marker_color="#58a6ff",
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark",
                title="PSI (numeric features)",
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                margin=dict(l=40, r=20, t=50, b=60),
                height=320,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No PSI scores in current window.")
    with m2:
        if chi:
            fig2 = go.Figure(
                data=[
                    go.Bar(
                        x=list(chi.keys()),
                        y=list(chi.values()),
                        marker_color="#a371f7",
                    )
                ]
            )
            fig2.update_layout(
                template="plotly_dark",
                title="Chi² p-values (categoricals)",
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                margin=dict(l=40, r=20, t=50, b=60),
                height=320,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.caption("No chi² p-values in current window.")

    if isinstance(od, (int, float)):
        st.metric("Output drift", f"{od:.4f}")

    if len(hist) >= 2:
        dfh = pd.DataFrame(hist)
        sev_map = {"none": 0, "warning": 1, "high": 2}
        dfh["sev_rank"] = dfh["severity"].map(lambda s: sev_map.get(str(s).lower(), 0))
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=dfh["ts"],
                y=dfh["sev_rank"],
                mode="lines+markers",
                line=dict(color="#58a6ff", width=2),
                name="Severity rank",
            )
        )
        fig3.update_yaxes(tickvals=[0, 1, 2], ticktext=["none", "warning", "high"])
        fig3.update_layout(
            template="plotly_dark",
            title="Severity timeline (recent polls)",
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            height=280,
            margin=dict(l=40, r=20, t=50, b=80),
        )
        st.plotly_chart(fig3, use_container_width=True)

def fetch_investigation_status(investigation_id: str) -> dict[str, Any] | None:
    try:
        r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
        )

        raw = r.get(f"investigation_status:{investigation_id}")

        if not raw:
            return None

        return json.loads(raw)

    except Exception:
        return None


def render_investigations(pending_items: list[dict[str, Any]]) -> None:
    ensure_session_lists()
    st.markdown('<p class="panel-title">Investigations</p>', unsafe_allow_html=True)

    open_items = [
        {
            "id": item.get("investigation_id", "—"),
            "status": "awaiting_approval",
            "route": item.get("proposed_action", "—"),
            "reason": item.get("rationale", "—"),
            "updated": _now_iso(),
        }
        for item in pending_items
    ]

    t1, t2 = st.tabs(["Open", "Resolved"])

    with t1:
        if open_items:
            for inv in open_items:
                live = fetch_investigation_status(inv["id"])
                status = live.get("status", inv["status"]) if live else inv["status"]

                with st.expander(f"**{inv['id']}** · `{status}`"):
                    st.write(f"**Route:** {inv.get('route','—')}")
                    st.write(f"**Trigger:** {inv.get('reason','—')}")
                    st.caption(inv.get("updated", ""))

                    if live:
                        st.write("**Worker execution status:**")
                        st.json(live)
        else:
            st.caption("No open investigations from agent service.")

    with t2:
        if st.session_state.investigations_resolved:
            rows = []

            for inv in st.session_state.investigations_resolved:
                live = fetch_investigation_status(inv["id"])
                row = dict(inv)

                if live:
                    row["worker_status"] = live.get("status")
                    row["task"] = live.get("task")
                    row["worker_updated_at"] = live.get("updated_at")

                rows.append(row)

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            for inv in rows:
                live = fetch_investigation_status(inv["id"])
                if live:
                    with st.expander(f"Worker result · {inv['id']}"):
                        st.json(live)
        else:
            st.caption("No resolved investigations yet.")


def render_hil_inbox(pending_items: list[dict[str, Any]]) -> None:
    ensure_session_lists()

    st.markdown('<p class="panel-title">Human approval inbox</p>', unsafe_allow_html=True)
    st.caption(f"POST `{AGENT_SERVICE_URL}/investigations/resume`")

    if not pending_items:
        st.caption("No pending human approvals from agent service.")
        return

    for item in list(pending_items):
        iid = item.get("investigation_id", "")

        st.markdown(
            f'<div class="hil-card"><b>{item.get("proposed_action","").upper()}</b> · '
            f'<code>{iid}</code></div>',
            unsafe_allow_html=True,
        )

        st.write(f"**Model URI:** `{item.get('model_uri','—')}`")
        st.write(f"**Rationale:** {item.get('rationale','—')}")

        b1, b2 = st.columns(2)

        with b1:
            if st.button("Approve", key=f"ap{iid}", type="primary"):
                ok, body = _post_json(
                    f"{AGENT_SERVICE_URL}/investigations/resume",
                    {"investigation_id": iid, "approved": True},
                )

                if ok:
                    st.session_state.investigations_resolved.insert(
                        0,
                        {
                            "id": iid,
                            "status": "approved",
                            "route": item.get("proposed_action"),
                            "reason": item.get("rationale"),
                            "updated": _now_iso(),
                        },
                    )

                    append_event("HIL", f"Approved {iid}")
                    st.success("Approved — graph resumed and queued job if applicable.")
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error(f"Approve failed: {body}")

        with b2:
            if st.button("Reject", key=f"rj{iid}"):
                ok, body = _post_json(
                    f"{AGENT_SERVICE_URL}/investigations/resume",
                    {"investigation_id": iid, "approved": False},
                )

                if ok:
                    st.session_state.investigations_resolved.insert(
                        0,
                        {
                            "id": iid,
                            "status": "rejected",
                            "route": item.get("proposed_action"),
                            "reason": item.get("rationale"),
                            "updated": _now_iso(),
                        },
                    )

                    append_event("HIL", f"Rejected {iid}")
                    st.warning("Rejected — graph resumed.")
                    time.sleep(1.0)
                    st.rerun()
                else:
                    st.error(f"Reject failed: {body}")

        st.divider()

def render_queue(rq: dict[str, Any]) -> None:
    st.markdown('<p class="panel-title">Queue & DLQ</p>', unsafe_allow_html=True)
    qd = rq.get("queue_depth")
    dd = rq.get("dlq_depth")
    c1, c2 = st.columns(2)
    with c1:
        if qd is not None:
            st.metric("Main queue depth", qd)
            st.progress(min(1.0, qd / 50.0) if qd else 0.0)
        else:
            st.metric("Main queue depth", "—")
    with c2:
        if dd is not None:
            st.metric("DLQ depth", dd)
            st.progress(min(1.0, dd / 20.0) if dd else 0.0)
        else:
            st.metric("DLQ depth", "—")

    if not rq.get("connected"):
        st.warning("Redis unreachable — queue metrics unavailable.")

    if rq.get("queue_samples"):
        st.caption("Recent queue payloads (head)")
        st.json(rq["queue_samples"][:5])
    if rq.get("dlq_samples"):
        st.caption("Recent DLQ payloads (head)")
        st.json(rq["dlq_samples"][:5])


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Console")
        st.caption(f"Refresh ≈ {REFRESH_SECONDS}s via fragment")
        st.text_input("MODEL_SERVICE_URL", MODEL_SERVICE_URL, disabled=True)
        st.text_input("AGENT_SERVICE_URL", AGENT_SERVICE_URL, disabled=True)
        st.text_input("MLFLOW_URL", MLFLOW_URL, disabled=True)
        if st.button("Clear demo seed"):
            st.session_state.demo_seed_disabled = True
            st.session_state.pending_hil = []
            st.session_state.investigations_open = []
            st.session_state.investigations_resolved = []
            st.session_state.drift_history = []
            append_event("UI", "Session demo lists cleared")
            st.rerun()
        if st.button("Restore demo seed"):
            st.session_state.demo_seed_disabled = False
            append_event("UI", "Demo seed re-enabled")
            st.rerun()

@st.fragment(run_every=timedelta(seconds=max(3, REFRESH_SECONDS)))
def dashboard_fragment() -> None:
    ensure_session_lists()

    health = fetch_model_health()
    report = fetch_drift_report()
    versions = fetch_mlflow_versions()
    rq = redis_metrics()

    merge_drift_history(report)

    render_header()
    st.divider()

    tab_a, tab_b, tab_c, tab_d = st.tabs(
        ["Overview", "Drift", "Investigations & HIL", "Queues"]
    )

    with tab_a:
        render_registry(health, versions)

    with tab_b:
        render_drift(report, st.session_state.drift_history)

    with tab_c:
        pending_items = try_fetch_agent_pending()

        if pending_items:
            st.session_state.pending_hil = pending_items

        c1, c2 = st.columns((1, 1))

        with c1:
            render_investigations(st.session_state.pending_hil)

        with c2:
            render_hil_inbox(st.session_state.pending_hil)

    with tab_d:
        render_queue(rq)

    with st.expander("Live event log"):
        for ev in st.session_state.events_log[:25]:
            st.write(
                f"`{ev['ts']}` **{ev['kind']}** — {ev['detail']}"
            )


def main() -> None:
    render_sidebar()
    dashboard_fragment()


if __name__ == "__main__":
    main()
