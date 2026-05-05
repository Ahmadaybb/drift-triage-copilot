"""
tasks/rollback.py — Rollback model task.
OWNER: Person 2 (calls /promote endpoint owned by Person 1)
STATUS: 🔲 TODO

- Find last known good Production version in MLflow
- Call model service POST /promote with rollback action
- Verify model service reloaded the rolled-back version
"""
