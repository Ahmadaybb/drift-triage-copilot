"""
graph.py — LangGraph supervisor graph.
OWNER: Person 2
STATUS: 🔲 TODO

Topology:
    supervisor → triage → action → comms → supervisor
    supervisor pauses for human approval before Production actions
    every node checkpoints to Postgres after completion
"""
