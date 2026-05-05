"""
registry_gate.py — Promotion checklist gate.
OWNER: Person 1
STATUS: 🔲 TODO

Responsibilities:
- Assert all checklist items before promoting any version to Production:
    1. Model has schema artifact
    2. Model has model card artifact
    3. Model has SHA-256 hash
    4. Test AUC >= 0.75
    5. Test Recall >= 0.75
    6. Operating threshold is set
    7. Request came from the agent (not called directly)
- Raise structured error if any check fails
"""
