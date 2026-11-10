import sqlite3

import pytest

from orchestrator import db

OLD_TABLE = ("CREATE TABLE jobs (id TEXT PRIMARY KEY, image_path TEXT NOT NULL, status TEXT NOT NULL, label TEXT, "
             "probability REAL, needs_review INTEGER, error TEXT, site_code TEXT, model_version TEXT, "
             "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def claimed_job(conn):
    job_id = db.create_job(conn, "/x/cat.jpg", "A12")
    db.claim_job(conn)
    return job_id


def second_guess(conn, job_id):
    job = db.get_job(conn, job_id)
    return job["second_label"], job["second_probability"]


def test_new_job_has_no_second_guess_yet(conn):
    assert second_guess(conn, db.create_job(conn, "/x/cat.jpg")) == (None, None)


def test_result_keeps_the_second_guess(conn):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, "raccoon", 0.57, True, "v7", "skunk", 0.21)
    assert second_guess(conn, job_id) == ("skunk", 0.21)


def test_result_without_a_second_guess_keeps_none(conn):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, "empty", 1.0, False, "v7")
    assert second_guess(conn, job_id) == (None, None)


def test_review_list_has_the_second_guess_and_the_model_version(conn):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, "raccoon", 0.57, True, "v7", "skunk", 0.21)
    row = db.review_list(conn)[0]
    assert (row["second_label"], row["second_probability"], row["model_version"]) == ("skunk", 0.21, "v7")


def test_correction_keeps_the_second_guess_of_the_model(conn):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, "raccoon", 0.57, True, "v7", "skunk", 0.21)
    db.save_review(conn, job_id, "skunk")
    assert second_guess(conn, job_id) == ("skunk", 0.21)


def test_old_database_gets_the_columns_and_keeps_its_jobs(tmp_path):
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute(OLD_TABLE)
    old.execute("INSERT INTO jobs (id, image_path, status, label, model_version, created_at, updated_at) "
                "VALUES ('old-job', '/x/cat.jpg', 'done', 'cat', 'v6', 'then', 'then')")
    old.commit()
    old.close()

    conn = db.connect(path)

    job = db.get_job(conn, "old-job")
    assert (job["label"], job["model_version"], job["second_label"], job["second_probability"]) == (
        "cat", "v6", None, None)
