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


def create_job(conn: sqlite3.Connection, image_path: str) -> str:
    job_id, now = str(uuid.uuid4()), _now()
    conn.execute("INSERT INTO jobs (id, image_path, status, created_at, updated_at) VALUES (?, ?, 'new', ?, ?)",
                 (job_id, image_path, now, now))
    return job_id


def get_job(conn: sqlite3.Connection, job_id: str) -> dict | None:
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def claim_job(conn: sqlite3.Connection) -> dict | None:
    """Move the oldest 'new' job to 'processing' in one transaction, so two workers never get the same job."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute("SELECT id FROM jobs WHERE status = 'new' ORDER BY created_at LIMIT 1").fetchone()
        if row is not None:
            conn.execute("UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ?", (_now(), row["id"]))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return get_job(conn, row["id"]) if row else None

