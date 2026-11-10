import pytest

from orchestrator import db


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def claimed_job(conn):
    job_id = db.create_job(conn, "photo.jpg", "A12")
    db.claim_job(conn)
    return job_id


@pytest.mark.parametrize("confidence,marked", [(1.0, False), (0.8, False), (0.79, True), (0.3, True)])
def test_person_row_at_any_confidence_gets_no_mark_and_its_photo_has_to_go(conn, confidence, marked):
    job_id = claimed_job(conn)

    db.set_result(conn, job_id, "person", confidence, marked)

    job = db.get_job(conn, job_id)
    assert (job["status"], job["label"], job["probability"]) == ("done", "person", confidence)
    assert job["needs_review"] == 0
    assert db.unwanted_photos(conn) == [{"id": job_id, "photo_name": "photo.jpg"}]


def test_person_row_keeps_nothing_of_the_photo_once_it_is_gone(conn):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, "person", 1.0, False)

    db.forget_photo(conn, job_id)

    job = db.get_job(conn, job_id)
    assert job["photo_name"] == ""
    assert "photo.jpg" not in str(job)
    assert db.unwanted_photos(conn) == []


def test_person_row_keeps_the_site_code(conn):
    job_id = claimed_job(conn)
    db.set_result(conn, job_id, "person", 1.0, False)
    assert db.get_job(conn, job_id)["site_code"] == "A12"


@pytest.mark.parametrize("label", ["bobcat", "empty"])
def test_other_labels_stay_as_they_are(conn, label):
    job_id = claimed_job(conn)

    db.set_result(conn, job_id, label, 0.5, True)

    job = db.get_job(conn, job_id)
    assert (job["needs_review"], job["photo_name"]) == (1, "photo.jpg")
    assert db.unwanted_photos(conn) == []


def test_refused_person_result_changes_nothing(conn):
    job_id = db.create_job(conn, "photo.jpg")

    with pytest.raises(db.InvalidTransition):
        db.set_result(conn, job_id, "person", 1.0, False)

    assert db.get_job(conn, job_id)["photo_name"] == "photo.jpg"
    assert db.unwanted_photos(conn) == []
