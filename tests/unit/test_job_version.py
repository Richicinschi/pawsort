import sqlite3

import pytest

from orchestrator import db

OLD_TABLE = ("CREATE TABLE jobs (id TEXT PRIMARY KEY, image_path TEXT NOT NULL, status TEXT NOT NULL, label TEXT, "
             "probability REAL, needs_review INTEGER, error TEXT, site_code TEXT, created_at TEXT NOT NULL, "
             "updated_at TEXT NOT NULL)")


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def claimed_job(conn):
    job_id = db.create_job(conn, "cat.jpg", "A12")
    db.claim_job(conn)
    return job_id


def test_new_job_has_no_version_yet(conn):
    job_id = db.create_job(conn, "cat.jpg")
    assert db.get_job(conn, job_id)["model_version"] is None


def test_result_keeps_the_version(conn):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, "cat", 0.9, False, "v7")
    assert db.get_job(conn, job_id)["model_version"] == "v7"


def test_failed_job_keeps_the_version(conn):
    job_id = claimed_job(conn)
    db.set_failed(conn, job_id, "broken photo", "v7")
    assert db.get_job(conn, job_id)["model_version"] == "v7"


def test_person_row_keeps_the_version(conn):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, "person", 1.0, False, "v7")
    assert db.get_job(conn, job_id)["model_version"] == "v7"


def test_job_row_has_everything_fr_23_asks_for(conn):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, "cat", 0.9, False, "v7")
    job = db.get_job(conn, job_id)
    for field in ("status", "label", "probability", "site_code", "model_version", "created_at", "updated_at"):
        assert job[field] is not None


def test_old_database_gets_the_column_and_keeps_its_jobs(tmp_path):
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute(OLD_TABLE)
    old.execute("INSERT INTO jobs (id, image_path, status, label, created_at, updated_at) "
                "VALUES ('old-job', '/x/cat.jpg', 'done', 'cat', 'then', 'then')")
    old.commit()
    old.close()

    conn = db.connect(path)

    job = db.get_job(conn, "old-job")
    assert (job["label"], job["model_version"]) == ("cat", None)


def test_opening_the_database_twice_is_fine(tmp_path):
    path = str(tmp_path / "jobs.db")
    db.connect(path).close()
    db.connect(path).close()
