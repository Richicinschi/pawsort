"""Numbers per camera site and week, and the weekly sample of sure tags that a person checks (AI-14, AI-15)."""
import random
import sqlite3
from datetime import datetime

from orchestrator import db

WARN_SHARE = 0.40  # a site and week with a larger share of needs review gets a warning
SAMPLE_SIZE = 50


def week_of(moment: str) -> str:
    year, week, _ = datetime.fromisoformat(moment).isocalendar()
    return f"{year}-W{week:02d}"


def this_week() -> str:
    return week_of(db.utc_now())


def site_numbers(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute("SELECT site_code, created_at, needs_review, old_label, new_label, decision FROM jobs "
                        "LEFT JOIN reviews ON job_id = id WHERE status = 'done'").fetchall()
    counts = {}
    for row in rows:
        count = counts.setdefault((week_of(row["created_at"]), row["site_code"]),
                                  {"tags": 0, "marked": 0, "checked": 0, "corrected": 0})
        count["tags"] += 1
        count["marked"] += row["needs_review"]
        count["checked"] += row["new_label"] is not None
        count["corrected"] += row["new_label"] not in (None, row["old_label"]) and row["decision"] != "rejected"
    numbers = []
    by_site = sorted(counts.items(), key=lambda item: item[0][1] or "")
    for (week, site), count in sorted(by_site, key=lambda item: item[0][0], reverse=True):
        share = round(count["marked"] / count["tags"], 4)
        numbers.append({"site_code": site, "week": week, "tags": count["tags"], "needs_review": share,
                        "checked": count["checked"], "warning": share > WARN_SHARE,
                        "corrected": round(count["corrected"] / count["checked"], 4) if count["checked"] else None})
    return numbers


def new_sites(conn: sqlite3.Connection, trained_on: str | None) -> list[str]:
    if not trained_on:
        return []
    rows = conn.execute("SELECT site_code, MIN(created_at) AS first FROM jobs WHERE site_code IS NOT NULL "
                        "GROUP BY site_code ORDER BY site_code").fetchall()
    return [row["site_code"] for row in rows if row["first"][:10] > trained_on]


def pick_sample(conn: sqlite3.Connection, week: str | None = None) -> None:
    """Make the pick of a week if it has none yet: sure tags that came in after the pick before, at random."""
    week = week or this_week()
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM picks WHERE week = ?", (week,)).fetchone() is None:
            last = conn.execute("SELECT MAX(picked_at) FROM picks").fetchone()[0] or ""
            sure = conn.execute("SELECT id FROM jobs WHERE status = 'done' AND needs_review = 0 AND label != ? "
                                "AND photo_name != '' AND updated_at > ? AND id NOT IN (SELECT job_id FROM reviews) "
                                "AND id NOT IN (SELECT job_id FROM samples)", (db.PERSON, last)).fetchall()
            jobs = random.sample([row["id"] for row in sure], min(SAMPLE_SIZE, len(sure)))
            conn.execute("INSERT INTO picks (week, picked_at) VALUES (?, ?)", (week, db.utc_now()))
            conn.executemany("INSERT INTO samples (job_id, week) VALUES (?, ?)", [(job_id, week) for job_id in jobs])
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def sample_jobs(conn: sqlite3.Connection, week: str) -> list[dict]:
    rows = conn.execute("SELECT id, site_code, label, probability, model_version FROM jobs JOIN samples ON job_id = id "
                        "WHERE week = ? AND id NOT IN (SELECT job_id FROM reviews) ORDER BY created_at",
                        (week,)).fetchall()
    return [dict(row) for row in rows]


def sample_numbers(conn: sqlite3.Connection, week: str) -> dict:
    row = conn.execute("SELECT COUNT(*) AS picked, COUNT(reviews.job_id) AS checked, "
                       "COALESCE(SUM(new_label != old_label), 0) AS changed FROM samples "
                       "LEFT JOIN reviews ON reviews.job_id = samples.job_id WHERE week = ?", (week,)).fetchone()
    return dict(row)


def in_sample(conn: sqlite3.Connection, job: dict) -> bool:
    picked = conn.execute("SELECT 1 FROM samples WHERE job_id = ? AND week = ?", (job["id"], this_week())).fetchone()
    return picked is not None and db.get_review(conn, job["id"]) is None


def save_answer(conn: sqlite3.Connection, job_id: str, new_label: str) -> None:
    db.save_review(conn, job_id, new_label, in_sample, "sample")
