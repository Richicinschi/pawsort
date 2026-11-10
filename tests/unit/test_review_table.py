from datetime import datetime

import pytest

from orchestrator import db


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def finished_job(conn, label="bobcat", probability=0.6, marked=True, version="v7", path="/x/cat.jpg"):
    job_id = db.create_job(conn, path, "A12")
    db.claim_job(conn)
    db.set_result(conn, job_id, label, probability, marked, version)
    return job_id


def on_the_list(conn):
    return [job["id"] for job in db.review_list(conn)]


def test_list_has_only_the_marked_jobs(conn):
    marked = finished_job(conn)
    finished_job(conn, marked=False, probability=0.9)
    assert on_the_list(conn) == [marked]


def test_list_shows_no_path_to_a_photo(conn):
    finished_job(conn)
    assert set(db.review_list(conn)[0]) == {"id", "site_code", "label", "probability", "second_label",
                                            "second_probability", "model_version"}


def test_oldest_job_comes_first_whatever_finished_first(conn):
    first = db.create_job(conn, "/x/1.jpg")
    second = db.create_job(conn, "/x/2.jpg")
    db.claim_job(conn)
    db.claim_job(conn)
    db.set_result(conn, second, "bobcat", 0.6, True)
    db.set_result(conn, first, "bobcat", 0.6, True)
    assert on_the_list(conn) == [first, second]


def test_list_goes_by_upload_time_and_not_by_the_order_in_the_table(conn):
    newer = finished_job(conn)
    older = finished_job(conn)
    conn.execute("UPDATE jobs SET created_at = '2020-01-01' WHERE id = ?", (older,))
    assert on_the_list(conn) == [older, newer]


def test_failed_job_and_job_in_progress_are_not_on_the_list(conn):
    failed = db.create_job(conn, "/x/1.jpg")
    db.claim_job(conn)
    db.set_failed(conn, failed, "broken photo")
    db.create_job(conn, "/x/2.jpg")
    db.claim_job(conn)
    assert on_the_list(conn) == []


def test_unsure_person_is_not_on_the_list(conn):
    finished_job(conn, label="person", probability=0.4)
    assert on_the_list(conn) == []


def test_confirmed_job_leaves_the_list_and_the_other_stays(conn):
    first, second = finished_job(conn), finished_job(conn)

    db.save_review(conn, first, "bobcat")

    assert on_the_list(conn) == [second]
    review = db.get_review(conn, first)
    assert (review["old_label"], review["new_label"]) == ("bobcat", "bobcat")


def test_correction_stores_old_label_new_label_version_and_time(conn):
    job_id = finished_job(conn)

    db.save_review(conn, job_id, "coyote")

    review = db.get_review(conn, job_id)
    assert (review["old_label"], review["new_label"], review["model_version"]) == ("bobcat", "coyote", "v7")
    assert datetime.fromisoformat(review["reviewed_at"]).tzinfo is not None


def test_correction_changes_the_tag_and_not_the_job_state(conn):
    job_id = finished_job(conn)

    db.save_review(conn, job_id, "coyote")

    job = db.get_job(conn, job_id)
    assert (job["status"], job["label"], job["needs_review"]) == ("done", "coyote", 1)


def test_second_review_is_refused_and_the_first_one_stays(conn):
    job_id = finished_job(conn)
    db.save_review(conn, job_id, "coyote")

    with pytest.raises(db.NotOnReviewList):
        db.save_review(conn, job_id, "deer")

    assert db.get_review(conn, job_id)["new_label"] == "coyote"
    assert db.get_job(conn, job_id)["label"] == "coyote"


def test_job_that_is_not_on_the_list_is_refused(conn):
    sure = finished_job(conn, marked=False, probability=0.9)
    with pytest.raises(db.NotOnReviewList):
        db.save_review(conn, sure, "coyote")
    assert db.get_review(conn, sure) is None


def test_unknown_job(conn):
    with pytest.raises(KeyError):
        db.save_review(conn, "nope", "coyote")


def test_after_a_refusal_the_connection_still_works(conn):
    sure = finished_job(conn, marked=False, probability=0.9)
    with pytest.raises(db.NotOnReviewList):
        db.save_review(conn, sure, "coyote")
    marked = finished_job(conn)
    db.save_review(conn, marked, "coyote")
    assert db.get_review(conn, marked) is not None


def test_changing_to_person_clears_the_path(conn):
    job_id = finished_job(conn, path="/x/secret.jpg")

    db.save_review(conn, job_id, "person")

    job = db.get_job(conn, job_id)
    assert (job["label"], job["image_path"]) == ("person", "")
    assert "secret.jpg" not in str(job) and "secret.jpg" not in str(db.get_review(conn, job_id))


def test_other_corrections_keep_the_path(conn):
    job_id = finished_job(conn, path="/x/cat.jpg")
    db.save_review(conn, job_id, "coyote")
    assert db.get_job(conn, job_id)["image_path"] == "/x/cat.jpg"
