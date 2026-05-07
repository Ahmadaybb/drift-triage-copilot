from __future__ import annotations

"""Push QueueJobSpec payloads to Redis."""

import json
import logging
import os

import redis

from agent_service.schemas import QueueJobSpec

logger = logging.getLogger(__name__)

QUEUE_KEY = os.environ.get("REDIS_QUEUE_KEY", "drift_triage:jobs")


def _redis_client() -> redis.Redis:
    url = os.environ.get("REDIS_URL")

    if url:
        return redis.Redis.from_url(url, decode_responses=True)

    host = os.environ.get("REDIS_HOST", "localhost")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    db = int(os.environ.get("REDIS_DB", "0"))

    return redis.Redis(host=host, port=port, db=db, decode_responses=True)


def enqueue_queue_jobs(jobs: list[QueueJobSpec]) -> None:
    if not jobs:
        logger.info("No queue jobs to enqueue.")
        return

    r = _redis_client()

    for job in jobs:
        payload = json.dumps(job.model_dump(mode="json"), separators=(",", ":"))
        r.lpush(QUEUE_KEY, payload)

        logger.info(
            "Enqueued job task=%s idempotency_key=%s queue=%s",
            job.task,
            job.idempotency_key,
            QUEUE_KEY,
        )