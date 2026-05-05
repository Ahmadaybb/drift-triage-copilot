"""
tasks/retrain.py — Retrain model task.
OWNER: Person 2 (calls train.py owned by Person 1)
STATUS: 🔲 TODO

- Trigger train.py with updated data
- New version registered in MLflow automatically
- Does NOT promote to Production — waits for human approval
- Idempotency key prevents double training
"""
