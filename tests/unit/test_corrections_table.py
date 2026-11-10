import sqlite3
from datetime import datetime

import pytest

from orchestrator import db


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def reviewed_job(conn, new_label, old_label="bobcat", photo="cat.jpg"):
    job_id = db.create_job(conn, photo, "A12")
    db.claim_job(conn)
    db.set_result(conn, job_id, old_label, 0.6, True, "v7")
    db.save_review(conn, job_id, new_label)
    return job_id


def waiting(conn):
    return [job["id"] for job in db.corrections(conn)]


def for_training(conn):
    return [job["id"] for job in db.for_training(conn)]


def test_changed_tag_waits_for_a_decision(conn):
    job_id = reviewed_job(conn, "coyote")
    assert db.corrections(conn) == [{"id": job_id, "site_code": "A12", "old_label": "bobcat", "new_label": "coyote",
                                     "model_version": "v7"}]


def test_confirmed_tag_needs_no_decision(conn):
    reviewed_job(conn, "bobcat")
    assert waiting(conn) == []


def test_change_to_person_is_final_and_never_waits(conn):
    job_id = reviewed_job(conn, "person")

    assert waiting(conn) == []
    with pytest.raises(db.NotWaiting):
        db.decide_correction(conn, job_id, "rejected")
    assert db.get_job(conn, job_id)["label"] == "person"


def test_oldest_correction_comes_first(conn):
    first, second = reviewed_job(conn, "coyote"), reviewed_job(conn, "deer")
    conn.execute("UPDATE reviews SET reviewed_at = '2020-01-01' WHERE job_id = ?", (second,))
    assert waiting(conn) == [second, first]


def test_approved_correction_leaves_the_list_and_keeps_the_new_label(conn):
    job_id, other = reviewed_job(conn, "coyote"), reviewed_job(conn, "deer")

    db.decide_correction(conn, job_id, "approved")

    assert waiting(conn) == [other]
    assert db.get_job(conn, job_id)["label"] == "coyote"
    review = db.get_review(conn, job_id)
    assert review["decision"] == "approved" and datetime.fromisoformat(review["decided_at"]).tzinfo is not None


def test_rejected_correction_leaves_the_list_and_puts_the_old_label_back(conn):
    job_id = reviewed_job(conn, "coyote")

    db.decide_correction(conn, job_id, "rejected")

    assert waiting(conn) == []
    assert db.get_job(conn, job_id)["label"] == "bobcat"
    review = db.get_review(conn, job_id)
    assert (review["decision"], review["old_label"], review["new_label"]) == ("rejected", "bobcat", "coyote")


def test_rejected_job_does_not_come_back_on_the_review_list(conn):
    job_id = reviewed_job(conn, "coyote")
    db.decide_correction(conn, job_id, "rejected")
    assert db.review_list(conn) == []


@pytest.mark.parametrize("first,second", [("approved", "rejected"), ("rejected", "approved"), ("approved", "approved")])
def test_second_decision_is_refused_and_the_first_one_stays(conn, first, second):
    job_id = reviewed_job(conn, "coyote")
    db.decide_correction(conn, job_id, first)

    with pytest.raises(db.NotWaiting):
        db.decide_correction(conn, job_id, second)

    assert db.get_review(conn, job_id)["decision"] == first
    assert db.get_job(conn, job_id)["label"] == ("coyote" if first == "approved" else "bobcat")


def test_confirmation_cannot_get_a_decision(conn):
    job_id = reviewed_job(conn, "bobcat")
    with pytest.raises(db.NotWaiting):
        db.decide_correction(conn, job_id, "approved")
    assert db.get_review(conn, job_id)["decision"] is None


def test_job_that_nobody_reviewed_cannot_get_a_decision(conn):
    job_id = db.create_job(conn, "cat.jpg")
    with pytest.raises(db.NotWaiting):
        db.decide_correction(conn, job_id, "approved")


def test_unknown_job(conn):
    with pytest.raises(KeyError):
        db.decide_correction(conn, "nope", "approved")


def test_after_a_refusal_the_connection_still_works(conn):
    with pytest.raises(KeyError):
        db.decide_correction(conn, "nope", "approved")
    job_id = reviewed_job(conn, "coyote")
    db.decide_correction(conn, job_id, "approved")
    assert db.get_review(conn, job_id)["decision"] == "approved"


def test_only_an_approved_correction_is_for_training(conn):
    approved, rejected = reviewed_job(conn, "coyote"), reviewed_job(conn, "deer")
    reviewed_job(conn, "skunk")
    reviewed_job(conn, "bobcat")
    db.decide_correction(conn, approved, "approved")
    db.decide_correction(conn, rejected, "rejected")

    assert db.for_training(conn) == [{"id": approved, "label": "coyote", "model_version": "v7"}]


def test_approved_correction_without_a_photo_is_not_for_training(conn):
    job_id = reviewed_job(conn, "coyote")
    db.decide_correction(conn, job_id, "approved")
    db.forget_photo(conn, job_id)
    assert for_training(conn) == []


def test_correction_a_training_took_is_not_given_out_again(conn):
    taken, later = reviewed_job(conn, "coyote"), reviewed_job(conn, "deer")
    db.decide_correction(conn, taken, "approved")
    db.decide_correction(conn, later, "approved")

    assert db.mark_used(conn, [taken]) == 1

    assert for_training(conn) == [later]
    assert datetime.fromisoformat(db.get_review(conn, taken)["used_at"]).tzinfo is not None


def test_only_an_approved_correction_can_be_marked_as_taken(conn):
    waits, rejected, approved = reviewed_job(conn, "coyote"), reviewed_job(conn, "deer"), reviewed_job(conn, "skunk")
    db.decide_correction(conn, rejected, "rejected")
    db.decide_correction(conn, approved, "approved")

    assert db.mark_used(conn, [waits, rejected, approved, approved, "nope"]) == 1

    assert [db.get_review(conn, job_id)["used_at"] is None for job_id in (waits, rejected, approved)] == [
        True, True, False]


def test_a_training_round_is_due_at_500_corrections():
    assert db.TRAINING_ROUND == 500


def test_old_database_gets_the_columns_and_its_corrections_wait(tmp_path):
    path = str(tmp_path / "old.db")
    db.connect(path).close()
    old = sqlite3.connect(path)
    old.execute("DROP TABLE reviews")
    old.execute("CREATE TABLE reviews (job_id TEXT PRIMARY KEY REFERENCES jobs (id), old_label TEXT NOT NULL, "
                "new_label TEXT NOT NULL, model_version TEXT, reviewed_at TEXT NOT NULL)")
    old.execute("INSERT INTO jobs (id, photo_name, status, label, needs_review, created_at, updated_at) "
                "VALUES ('old-job', 'cat.jpg', 'done', 'coyote', 1, 'then', 'then')")
    old.execute("INSERT INTO reviews VALUES ('old-job', 'bobcat', 'coyote', 'v6', '2026-11-01')")
    old.commit()
    old.close()

    conn = db.connect(path)

    assert waiting(conn) == ["old-job"]
    db.decide_correction(conn, "old-job", "approved")
    assert for_training(conn) == ["old-job"]
