"""Jobs table and its state machine. The orchestrator is the only owner of this data."""
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

TRANSITIONS = {"new": {"processing"}, "processing": {"done", "failed"}, "done": set(), "failed": set()}
STUCK_SECONDS = 30
PERSON = "person"


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
        "label TEXT, probability REAL, needs_review INTEGER, error TEXT, site_code TEXT, model_version TEXT, "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "model_version" not in columns:  # a database from before this column
        conn.execute("ALTER TABLE jobs ADD COLUMN model_version TEXT")
    return conn


def create_job(conn: sqlite3.Connection, image_path: str, site_code: str | None = None) -> str:
    job_id, now = str(uuid.uuid4()), _now()
    conn.execute("INSERT INTO jobs (id, image_path, site_code, status, created_at, updated_at) "
                 "VALUES (?, ?, ?, 'new', ?, ?)", (job_id, image_path, site_code, now, now))
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


def _move(conn: sqlite3.Connection, job_id: str, new_status: str, fields: dict) -> None:
    job = get_job(conn, job_id)
    if job is None:
        raise KeyError(job_id)
    check_transition(job["status"], new_status)
    sets = ", ".join(f"{k} = ?" for k in fields)
    moved = conn.execute(f"UPDATE jobs SET status = ?, {sets}, updated_at = ? WHERE id = ? AND status = ?",
                         (new_status, *fields.values(), _now(), job_id, job["status"]))
    if moved.rowcount == 0:  # somebody else moved this job a moment ago
        raise InvalidTransition(f"{job['status']} -> {new_status} came too late")


def set_result(conn: sqlite3.Connection, job_id: str, label: str, probability: float, needs_review: bool,
               model_version: str | None = None) -> None:
    fields = {"label": label, "probability": probability, "needs_review": int(needs_review),
              "model_version": model_version}
    if label == PERSON:  # FR-14: never waits for a review, and the row keeps nothing of the photo
        fields.update(needs_review=0, image_path="")
    _move(conn, job_id, "done", fields)


def set_failed(conn: sqlite3.Connection, job_id: str, error: str, model_version: str | None = None) -> None:
    _move(conn, job_id, "failed", {"error": error, "model_version": model_version})


def fail_stuck_jobs(conn: sqlite3.Connection, seconds: float = STUCK_SECONDS) -> list[str]:
    """Fail every job that stayed in 'processing' too long. Returns the photos of these jobs."""
    limit = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    stuck = conn.execute("SELECT id, image_path FROM jobs WHERE status = 'processing' AND updated_at < ?",
                         (limit,)).fetchall()
    photos = []
    for job in stuck:
        try:
            set_failed(conn, job["id"], f"no answer within {seconds} seconds")
            photos.append(job["image_path"])
        except InvalidTransition:
            pass  # the AI server answered just in time
    return photos
