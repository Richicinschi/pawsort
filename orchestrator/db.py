"""Jobs table and its state machine. The orchestrator is the only owner of this data."""
import sqlite3
import uuid
from datetime import datetime, timezone

TRANSITIONS = {"new": {"processing"}, "processing": {"done", "failed"}, "done": set(), "failed": set()}


class InvalidTransition(Exception):
    pass


def check_transition(old: str, new: str) -> None:
    if new not in TRANSITIONS.get(old, set()):
        raise InvalidTransition(f"{old} -> {new} is not allowed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, image_path TEXT NOT NULL, status TEXT NOT NULL, "
        "label TEXT, probability REAL, needs_review INTEGER, error TEXT, created_at TEXT NOT NULL, "
        "updated_at TEXT NOT NULL)")
    return conn

