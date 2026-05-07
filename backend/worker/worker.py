from __future__ import annotations

"""
worker.py — Redis queue worker.

- Blocking poll on Redis queue
- Idempotency protection
- Retry with exponential backoff
- DLQ support
- Worker heartbeat for dashboard monitoring
- Redis investigation execution status for dashboard visibility
"""

import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import UTC, datetime
from typing import Any, Callable

import redis

from worker.tasks import replay as replay_task
from worker.tasks import retrain as retrain_task
from worker.tasks import rollback as rollback_task

logger = logging.getLogger(__name__)

QUEUE_KEY = os.environ.get("REDIS_QUEUE_KEY", "drift_triage:jobs")
DLQ_KEY = os.environ.get("REDIS_DLQ_KEY", "drift_triage:jobs:dlq")

IDEM_DONE_PREFIX = os.environ.get("REDIS_IDEM_DONE_PREFIX", "drift_triage:idem:done:")
IDEM_LOCK_PREFIX = os.environ.get("REDIS_IDEM_LOCK_PREFIX", "drift_triage:idem:lock:")

HEARTBEAT_KEY = "worker:heartbeat"
INVESTIGATION_STATUS_PREFIX = "investigation_status:"

MAX_RETRIES = int(os.environ.get("WORKER_MAX_RETRIES", "5"))
BACKOFF_BASE_S = float(os.environ.get("WORKER_BACKOFF_BASE_S", "1.0"))
BACKOFF_CAP_S = float(os.environ.get("WORKER_BACKOFF_CAP_S", "300.0"))
LOCK_TTL_S = int(os.environ.get("WORKER_LOCK_TTL_S", "900"))
DONE_TTL_S = int(os.environ.get("WORKER_DONE_TTL_S", str(86400 * 30)))
BRPOP_TIMEOUT_S = int(os.environ.get("WORKER_BRPOP_TIMEOUT_S", "5"))

_stop = False


def _install_signal_handlers() -> None:
    def _handle(_sig: int, _frame: Any) -> None:
        global _stop
        _stop = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _redis_client() -> redis.Redis:
    url = os.environ.get("REDIS_URL")

    if url:
        return redis.Redis.from_url(url, decode_responses=True)

    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    db = int(os.environ.get("REDIS_DB", "0"))

    return redis.Redis(host=host, port=port, db=db, decode_responses=True)


def _set_investigation_status(
    r: redis.Redis,
    investigation_id: str,
    status: str,
    task: str | None = None,
    result: Any | None = None,
    error: str | None = None,
) -> None:
    if not investigation_id:
        investigation_id = "unknown"

    body: dict[str, Any] = {
        "investigation_id": investigation_id,
        "status": status,
        "task": task,
        "updated_at": datetime.now(UTC).isoformat(),
    }

    if result is not None:
        body["result"] = result

    if error is not None:
        body["error"] = error

    r.set(
        f"{INVESTIGATION_STATUS_PREFIX}{investigation_id}",
        json.dumps(body, separators=(",", ":")),
    )


def heartbeat_loop(r: redis.Redis) -> None:
    while not _stop:
        try:
            r.set(HEARTBEAT_KEY, str(time.time()))
        except Exception:
            pass

        time.sleep(5)


def _stub_run(payload: dict[str, Any]) -> dict[str, Any]:
    logger.warning("Task handler not implemented payload_keys=%s", list(payload))
    return {
        "ok": True,
        "stub": True,
        "payload_keys": list(payload),
    }


TaskRunner = Callable[[dict[str, Any]], dict[str, Any] | None]


def _handlers() -> dict[str, TaskRunner]:
    return {
        "replay": getattr(replay_task, "run", _stub_run),
        "retrain": getattr(retrain_task, "run", _stub_run),
        "rollback": getattr(rollback_task, "run", _stub_run),
    }


def _backoff_seconds(attempt: int) -> float:
    return min(BACKOFF_BASE_S * (2**attempt), BACKOFF_CAP_S)


def _dispatch(job: dict[str, Any], handlers: dict[str, TaskRunner]) -> dict[str, Any] | None:
    task = job.get("task")

    if task not in handlers:
        raise ValueError(f"unknown task {task!r}")

    payload = job.get("payload") or {}

    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")

    return handlers[task](payload)


def _send_to_dlq(r: redis.Redis, job: dict[str, Any], err: str) -> None:
    envelope = {
        **job,
        "last_error": err,
        "failed_at": datetime.now(UTC).isoformat(),
    }

    r.lpush(DLQ_KEY, json.dumps(envelope, separators=(",", ":")))

def process_one(
    r: redis.Redis,
    raw: str,
    handlers: dict[str, TaskRunner],
) -> None:
    job = json.loads(raw)

    payload = job.get("payload") or {}
    investigation_id = str(payload.get("investigation_id") or "unknown")
    task = str(job.get("task") or "unknown")

    _set_investigation_status(
        r=r,
        investigation_id=investigation_id,
        status="running",
        task=task,
    )

    idem = job.get("idempotency_key")

    if not idem:
        raise ValueError("job missing idempotency_key")

    done_key = f"{IDEM_DONE_PREFIX}{idem}"
    lock_key = f"{IDEM_LOCK_PREFIX}{idem}"

    if r.exists(done_key):
        logger.info(
            "Skip duplicate idempotency_key=%s",
            idem,
        )

        _set_investigation_status(
            r=r,
            investigation_id=investigation_id,
            status="already_done",
            task=task,
        )

        return

    got_lock = r.set(
        lock_key,
        "1",
        nx=True,
        ex=LOCK_TTL_S,
    )

    if not got_lock:
        logger.info(
            "Lock held idempotency_key=%s",
            idem,
        )

        _set_investigation_status(
            r=r,
            investigation_id=investigation_id,
            status="deferred_lock_held",
            task=task,
        )

        time.sleep(
            _backoff_seconds(
                int(job.get("_attempts", 0))
            )
        )

        r.rpush(QUEUE_KEY, raw)

        return

    try:
        result = _dispatch(job, handlers)

        # demo visibility delay
        time.sleep(30)

        r.set(
            done_key,
            "1",
            ex=DONE_TTL_S,
        )

        _set_investigation_status(
            r=r,
            investigation_id=investigation_id,
            status="completed",
            task=task,
            result=result,
        )

        logger.info(
            "Job OK idempotency_key=%s task=%s investigation_id=%s",
            idem,
            task,
            investigation_id,
        )

    except Exception as exc:
        attempt = int(job.get("_attempts", 0))

        err = f"{type(exc).__name__}: {exc}"

        logger.exception(
            "Job failed idempotency_key=%s attempt=%s",
            idem,
            attempt,
        )

        if attempt >= MAX_RETRIES:
            _send_to_dlq(r, job, err)

            _set_investigation_status(
                r=r,
                investigation_id=investigation_id,
                status="dlq",
                task=task,
                error=err,
            )

            logger.error(
                "Sent to DLQ idempotency_key=%s",
                idem,
            )

            return

        job["_attempts"] = attempt + 1

        _set_investigation_status(
            r=r,
            investigation_id=investigation_id,
            status="retry_scheduled",
            task=task,
            error=err,
        )

        delay = _backoff_seconds(attempt)

        logger.info(
            "Retry idempotency_key=%s in %.1fs",
            idem,
            delay,
        )

        time.sleep(delay)

        r.rpush(
            QUEUE_KEY,
            json.dumps(job, separators=(",", ":")),
        )

    finally:
        r.delete(lock_key)


def run_worker() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    _install_signal_handlers()

    r = _redis_client()
    handlers = _handlers()

    threading.Thread(
        target=heartbeat_loop,
        args=(r,),
        daemon=True,
    ).start()

    logger.info("Worker listening queue=%s dlq=%s", QUEUE_KEY, DLQ_KEY)

    while not _stop:
        item = r.brpop(QUEUE_KEY, timeout=BRPOP_TIMEOUT_S)

        if item is None:
            continue

        _, raw = item

        try:
            process_one(r, raw, handlers)

        except Exception:
            logger.exception("Unrecoverable job failure raw=%s", raw[:500])


if __name__ == "__main__":
    run_worker()
    sys.exit(0)