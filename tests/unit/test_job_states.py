import pytest

from orchestrator import db


@pytest.mark.parametrize("old,new", [("new", "processing"), ("processing", "done"), ("processing", "failed")])
def test_allowed_transitions(old, new):
    db.check_transition(old, new)


@pytest.mark.parametrize("old,new", [("new", "done"), ("new", "failed"), ("done", "processing"),
                                     ("failed", "done"), ("done", "done")])
def test_forbidden_transitions(old, new):
    with pytest.raises(db.InvalidTransition):
        db.check_transition(old, new)


def test_job_lifecycle(tmp_path):
    conn = db.connect(str(tmp_path / "jobs.db"))
    job_id = db.create_job(conn, "cat.jpg")
    assert db.get_job(conn, job_id)["status"] == "new"
    claimed = db.claim_job(conn)
    assert claimed["id"] == job_id and claimed["status"] == "processing"
    assert db.claim_job(conn) is None
    db.set_result(conn, job_id, "cat", 0.97, False)
    job = db.get_job(conn, job_id)
    assert (job["status"], job["label"], job["needs_review"]) == ("done", "cat", 0)
    with pytest.raises(db.InvalidTransition):
        db.set_failed(conn, job_id, "late error")
