import pytest

from orchestrator import db


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def claimed_job(conn, path="/x/photo.jpg"):
    job_id = db.create_job(conn, path, "A12")
    db.claim_job(conn)
    return job_id


@pytest.mark.parametrize("confidence,marked", [(1.0, False), (0.8, False), (0.79, True), (0.3, True)])
def test_person_row_at_any_confidence_keeps_no_photo_and_no_mark(conn, confidence, marked):
    job_id = claimed_job(conn)

    db.set_result(conn, job_id, "person", confidence, marked)

    job = db.get_job(conn, job_id)
    assert (job["status"], job["label"], job["probability"]) == ("done", "person", confidence)
    assert job["needs_review"] == 0
    assert job["image_path"] == ""
    assert "photo.jpg" not in str(job)


def test_person_row_keeps_the_site_code(conn):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, "person", 1.0, False)
    assert db.get_job(conn, job_id)["site_code"] == "A12"


@pytest.mark.parametrize("label", ["bobcat", "empty"])
def test_other_labels_stay_as_they_are(conn, label):
    job_id = claimed_job(conn)

    db.set_result(conn, job_id, label, 0.5, True)

    job = db.get_job(conn, job_id)
    assert (job["needs_review"], job["image_path"]) == (1, "/x/photo.jpg")


def test_refused_person_result_changes_nothing(conn):
    job_id = db.create_job(conn, "/x/photo.jpg")

    with pytest.raises(db.InvalidTransition):
        db.set_result(conn, job_id, "person", 1.0, False)

    assert db.get_job(conn, job_id)["image_path"] == "/x/photo.jpg"
