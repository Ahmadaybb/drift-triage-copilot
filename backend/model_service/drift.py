"""
drift.py — Drift detection logic.
OWNER: Person 1
STATUS: 🔲 TODO

Responsibilities:
- Compute PSI on numeric features
- Compute chi² on categorical features
- Compute output distribution drift
- Classify severity: none / warning / high
- Return DriftAlert payload ready to send as webhook

PSI thresholds:
    < 0.1   → none
    0.1–0.2 → warning
    > 0.2   → high

chi² threshold:
    p-value < 0.05 → drift detected
"""
