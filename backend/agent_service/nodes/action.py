"""
nodes/action.py — Action sub-agent.
OWNER: Person 2
STATUS: 🔲 TODO

Input:  triage result
Output: job dispatched to Redis queue (replay / retrain / rollback)
        graph pauses here for human approval
"""
