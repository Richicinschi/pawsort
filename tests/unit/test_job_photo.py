import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from orchestrator import db

OLD_TABLE = ("CREATE TABLE jobs (id TEXT PRIMARY KEY, image_path TEXT NOT NULL, status TEXT NOT NULL, label TEXT, "
             "probability REAL, needs_review INTEGER, error TEXT, site_code TEXT, model_version TEXT, "
             "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def claimed_job(conn):
    job_id = db.create_job(conn, "abc.jpg", "A12")
    db.claim_job(conn)
    return job_id


def finished_job(conn, label="bobcat", marked=True):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, label, 0.6, marked)
    return job_id


def old_database(tmp_path, status):
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute(OLD_TABLE)
    old.execute("INSERT INTO jobs (id, image_path, status, label, site_code, created_at, updated_at) "
                "VALUES ('old-job', 'C:/pawsort/data/uploads/old.jpg', ?, 'cat', 'A12', 'then', 'then')", (status,))
    old.commit()
    old.close()
    return path


def test_job_row_keeps_the_name_of_the_photo(conn):
    job_id = db.create_job(conn, "abc.jpg")
    assert db.get_job(conn, job_id)["photo_name"] == "abc.jpg"


def test_job_row_has_no_path_any_more(conn):
    assert "image_path" not in db.get_job(conn, db.create_job(conn, "abc.jpg"))


def test_claimed_job_still_knows_its_photo(conn):
    db.create_job(conn, "abc.jpg")
    assert db.claim_job(conn)["photo_name"] == "abc.jpg"


def test_waiting_working_and_tagged_jobs_keep_their_photo(conn):
    finished_job(conn)
    claimed_job(conn)
    db.create_job(conn, "waiting.jpg")
    assert db.unwanted_photos(conn) == []


def test_photo_of_a_failed_job_has_to_go(conn):
    job_id = claimed_job(conn)
    db.set_failed(conn, job_id, "broken photo")
    assert db.unwanted_photos(conn) == [{"id": job_id, "photo_name": "abc.jpg"}]


def test_stuck_job_fails_and_gives_back_its_photo(conn):
    job_id = claimed_job(conn)
    then = (datetime.now(timezone.utc) - timedelta(seconds=31)).isoformat()
    conn.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (then, job_id))

    assert db.fail_stuck_jobs(conn) == [{"id": job_id, "photo_name": "abc.jpg"}]
    assert db.get_job(conn, job_id)["status"] == "failed"


def test_job_keeps_the_name_until_somebody_says_the_photo_is_gone(conn):
    job_id = claimed_job(conn)
    db.set_failed(conn, job_id, "broken photo")
    assert db.get_job(conn, job_id)["photo_name"] == "abc.jpg"

    db.forget_photo(conn, job_id)

    assert db.get_job(conn, job_id)["photo_name"] == ""
    assert db.unwanted_photos(conn) == []


def test_job_that_waits_for_a_person_is_on_the_review_list(conn):
    job = db.get_job(conn, finished_job(conn))
    assert db.on_review_list(conn, job) is True


def test_sure_job_is_not_on_the_review_list(conn):
    job = db.get_job(conn, finished_job(conn, marked=False))
    assert db.on_review_list(conn, job) is False


def test_checked_job_is_not_on_the_review_list_any_more(conn):
    job_id = finished_job(conn)
    db.save_review(conn, job_id, "coyote")
    assert db.on_review_list(conn, db.get_job(conn, job_id)) is False


def test_job_in_progress_is_not_on_the_review_list(conn):
    assert db.on_review_list(conn, db.get_job(conn, claimed_job(conn))) is False


def test_old_database_loses_its_paths_and_keeps_its_jobs(tmp_path):
    conn = db.connect(old_database(tmp_path, "done"))

    job = db.get_job(conn, "old-job")

    assert (job["label"], job["site_code"], job["photo_name"]) == ("cat", "A12", "")
    assert "uploads" not in str(job)


def test_old_database_takes_new_jobs(tmp_path):
    conn = db.connect(old_database(tmp_path, "done"))
    job_id = db.create_job(conn, "abc.jpg")
    assert db.get_job(conn, job_id)["photo_name"] == "abc.jpg"


def test_opening_an_old_database_twice_keeps_the_new_photo_names(tmp_path):
    path = old_database(tmp_path, "done")
    job_id = db.create_job(db.connect(path), "abc.jpg")
    assert db.get_job(db.connect(path), job_id)["photo_name"] == "abc.jpg"


def test_change_of_an_old_database_that_stops_half_way_leaves_no_path_behind(tmp_path, monkeypatch):
    path = old_database(tmp_path, "done")
    steps = []

    class Stops(sqlite3.Connection):
        def execute(self, sql, *args):
            if sql.startswith(("UPDATE jobs SET", "ALTER TABLE jobs RENAME")):
                steps.append(sql)
                if len(steps) == 2:
                    raise sqlite3.OperationalError("the power went off")
            return super().execute(sql, *args)

    real = sqlite3.connect
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **more: real(*args, factory=Stops, **more))
    with pytest.raises(sqlite3.OperationalError):
        db.connect(path)
    monkeypatch.undo()

    job = db.get_job(db.connect(path), "old-job")
    assert (job["label"], job["photo_name"]) == ("cat", "")
    assert "uploads" not in str(job)
