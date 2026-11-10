from datetime import datetime, timedelta, timezone

import pytest

from orchestrator import db
from orchestrator.app import delete_photo


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def make_old(conn, job_id, seconds):
    then = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
    conn.execute("UPDATE jobs SET updated_at = ? WHERE id = ?", (then, job_id))


def claimed_job(conn, seconds, path="/x/cat.jpg"):
    job_id = db.create_job(conn, path)
    db.claim_job(conn)
    make_old(conn, job_id, seconds)
    return job_id


@pytest.mark.parametrize("seconds,status", [(29, "processing"), (31, "failed")])
def test_the_limit_is_30_seconds(conn, seconds, status):
    job_id = claimed_job(conn, seconds)
    db.fail_stuck_jobs(conn)
    assert db.get_job(conn, job_id)["status"] == status


def test_stuck_job_gets_a_reason_and_gives_back_its_photo(conn):
    job_id = claimed_job(conn, 31, "/x/stuck.jpg")

    assert db.fail_stuck_jobs(conn) == ["/x/stuck.jpg"]
    assert "30 seconds" in db.get_job(conn, job_id)["error"]
    assert db.fail_stuck_jobs(conn) == []


def test_every_stuck_job_goes_in_one_round(conn):
    jobs = [claimed_job(conn, 31, f"/x/{n}.jpg") for n in range(3)]

    assert len(db.fail_stuck_jobs(conn)) == 3
    assert [db.get_job(conn, job_id)["status"] for job_id in jobs] == ["failed"] * 3


def test_old_jobs_in_other_states_stay_as_they_are(conn):
    done = claimed_job(conn, 0)
    db.set_result(conn, done, "cat", 0.9, False)
    make_old(conn, done, 60)
    waiting = db.create_job(conn, "/x/waiting.jpg")
    make_old(conn, waiting, 60)

    assert db.fail_stuck_jobs(conn) == []
    assert db.get_job(conn, waiting)["status"] == "new"
    assert db.get_job(conn, done)["status"] == "done"


def test_answer_that_comes_after_the_timeout_is_refused(conn):
    job_id = claimed_job(conn, 31)
    db.fail_stuck_jobs(conn)

    with pytest.raises(db.InvalidTransition):
        db.set_result(conn, job_id, "cat", 0.9, False)
    assert db.get_job(conn, job_id)["label"] is None


def test_two_moves_at_the_same_moment_cannot_both_win(conn, monkeypatch):
    job_id = claimed_job(conn, 0)
    seen_before = db.get_job(conn, job_id)
    db.set_failed(conn, job_id, "first")
    monkeypatch.setattr(db, "get_job", lambda c, i: seen_before)

    with pytest.raises(db.InvalidTransition):
        db.set_result(conn, job_id, "cat", 0.9, False)


def test_deleting_a_photo_that_is_gone_is_fine(tmp_path):
    delete_photo(str(tmp_path / "gone.jpg"))


def test_a_photo_that_cannot_be_deleted_does_not_crash_the_check(tmp_path):
    delete_photo(str(tmp_path))  # a folder, so the delete fails
    assert tmp_path.exists()
