"""
worker.py — Redis queue worker.
OWNER: Person 2
STATUS: 🔲 TODO

- Polls Redis queue for jobs
- Idempotency: same job twice runs only once
- Exponential backoff retries on failure
- Dead-letter queue (DLQ) after max retries
"""
