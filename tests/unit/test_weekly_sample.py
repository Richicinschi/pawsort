import sqlite3

import pytest

from orchestrator import db, numbers


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def tagged(conn, label="bobcat", probability=0.9, marked=False, photo="cat.jpg"):
    job_id = db.create_job(conn, photo, "A12")
    db.claim_job(conn)
    db.set_result(conn, job_id, label, probability, marked, "v7")
    return job_id


def picked(conn, week):
    numbers.pick_sample(conn, week)
    return sorted(job["id"] for job in numbers.sample_jobs(conn, week))


def test_sample_has_only_sure_tags_that_nobody_checked_and_that_still_have_a_photo(conn):
    sure, empty = tagged(conn), tagged(conn, "empty", 1.0)
    tagged(conn, probability=0.6, marked=True)
    tagged(conn, "person", 1.0)
    db.forget_photo(conn, tagged(conn))
    failed = db.create_job(conn, "broken.jpg")
    db.claim_job(conn)
    db.set_failed(conn, failed, "broken photo")
    db.create_job(conn, "waiting.jpg")

    assert picked(conn, "2026-W46") == sorted([sure, empty])


def test_sample_has_at_most_50_tags(conn):
    jobs = {tagged(conn) for _ in range(60)}

    sample = picked(conn, "2026-W46")

    assert len(sample) == 50 and set(sample) < jobs
    assert numbers.SAMPLE_SIZE == 50


def test_sample_is_picked_at_random(conn):
    jobs = [tagged(conn) for _ in range(60)]
    sample = set(picked(conn, "2026-W46"))
    assert sample != set(jobs[:50]) and sample != set(jobs[10:])


def test_pick_stays_the_same_for_the_week(conn):
    for _ in range(60):
        tagged(conn)
    first = picked(conn, "2026-W46")

    later = tagged(conn)

    assert picked(conn, "2026-W46") == first
    assert later not in first


def test_week_with_nothing_to_pick_keeps_its_empty_pick(conn):
    assert picked(conn, "2026-W46") == []
    tagged(conn)
    assert picked(conn, "2026-W46") == []


def test_next_week_picks_from_the_tags_that_came_after_the_pick_before(conn):
    old = [tagged(conn) for _ in range(60)]
    first = picked(conn, "2026-W46")
    conn.execute("UPDATE picks SET picked_at = '2026-11-09T08:00:00+00:00'")
    conn.execute("UPDATE jobs SET updated_at = '2026-11-08T08:00:00+00:00'")
    new = [tagged(conn) for _ in range(3)]

    assert picked(conn, "2026-W47") == sorted(new)
    assert picked(conn, "2026-W46") == first
    assert not set(picked(conn, "2026-W47")) & set(old)


def test_week_in_which_nobody_looked_loses_no_tag(conn):
    tagged(conn)
    picked(conn, "2026-W45")
    conn.execute("UPDATE picks SET picked_at = '2026-11-02T08:00:00+00:00'")
    conn.execute("UPDATE jobs SET updated_at = '2026-11-01T08:00:00+00:00'")
    from_the_skipped_week = tagged(conn)
    conn.execute("UPDATE jobs SET updated_at = '2026-11-11T08:00:00+00:00' WHERE id = ?", (from_the_skipped_week,))

    assert picked(conn, "2026-W47") == [from_the_skipped_week]


def test_sample_row_has_what_a_person_needs_to_check_the_tag(conn):
    job_id = tagged(conn, "mountain_lion", 0.93)
    numbers.pick_sample(conn, "2026-W46")
    assert numbers.sample_jobs(conn, "2026-W46") == [
        {"id": job_id, "site_code": "A12", "label": "mountain_lion", "probability": 0.93, "model_version": "v7"}]


def test_answer_from_the_sample_is_stored_as_a_review_that_came_from_the_sample(conn, monkeypatch):
    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W46")
    job_id = tagged(conn)
    numbers.pick_sample(conn)

    numbers.save_answer(conn, job_id, "coyote")

    review = db.get_review(conn, job_id)
    assert (review["old_label"], review["new_label"], review["model_version"], review["source"]) == (
        "bobcat", "coyote", "v7", "sample")
    assert db.get_job(conn, job_id)["label"] == "coyote"


def test_answer_from_the_review_list_is_not_marked_as_sample(conn):
    job_id = tagged(conn, probability=0.6, marked=True)
    db.save_review(conn, job_id, "coyote")
    assert db.get_review(conn, job_id)["source"] == "review"


def test_changed_tag_from_the_sample_waits_for_a_decision_like_any_other(conn, monkeypatch):
    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W46")
    changed, confirmed = tagged(conn), tagged(conn)
    numbers.pick_sample(conn)

    numbers.save_answer(conn, changed, "coyote")
    numbers.save_answer(conn, confirmed, "bobcat")

    assert [job["id"] for job in db.corrections(conn)] == [changed]


def test_checked_tag_leaves_the_sample_and_cannot_be_answered_twice(conn, monkeypatch):
    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W46")
    job_id, other = tagged(conn), tagged(conn)
    numbers.pick_sample(conn)
    numbers.save_answer(conn, job_id, "bobcat")

    assert [job["id"] for job in numbers.sample_jobs(conn, "2026-W46")] == [other]
    with pytest.raises(db.NotOnReviewList):
        numbers.save_answer(conn, job_id, "coyote")


def test_sure_tag_that_is_not_in_the_sample_cannot_be_answered(conn, monkeypatch):
    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W46")
    numbers.pick_sample(conn)
    job_id = tagged(conn)

    with pytest.raises(db.NotOnReviewList):
        numbers.save_answer(conn, job_id, "coyote")
    with pytest.raises(db.NotOnReviewList):
        db.save_review(conn, job_id, "coyote")
    assert db.get_review(conn, job_id) is None


def test_tag_from_the_sample_of_an_earlier_week_cannot_be_answered_any_more(conn, monkeypatch):
    job_id = tagged(conn)
    numbers.pick_sample(conn, "2026-W46")
    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W47")

    assert numbers.in_sample(conn, db.get_job(conn, job_id)) is False
    with pytest.raises(db.NotOnReviewList):
        numbers.save_answer(conn, job_id, "coyote")


def test_numbers_of_the_sample_count_the_picked_the_checked_and_the_changed(conn, monkeypatch):
    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W46")
    jobs = [tagged(conn) for _ in range(5)]
    numbers.pick_sample(conn)
    numbers.save_answer(conn, jobs[0], "bobcat")
    numbers.save_answer(conn, jobs[1], "coyote")
    numbers.save_answer(conn, jobs[2], "person")

    assert numbers.sample_numbers(conn, "2026-W46") == {"picked": 5, "checked": 3, "changed": 2}
    assert numbers.sample_numbers(conn, "2026-W47") == {"picked": 0, "checked": 0, "changed": 0}


def test_old_database_gets_the_new_column_and_its_reviews_count_as_from_the_review_list(tmp_path):
    path = str(tmp_path / "old.db")
    db.connect(path).close()
    old = sqlite3.connect(path)
    old.execute("DROP TABLE reviews")
    old.execute("CREATE TABLE reviews (job_id TEXT PRIMARY KEY REFERENCES jobs (id), old_label TEXT NOT NULL, "
                "new_label TEXT NOT NULL, model_version TEXT, reviewed_at TEXT NOT NULL)")
    old.execute("INSERT INTO reviews VALUES ('old-job', 'bobcat', 'coyote', 'v6', '2026-11-01')")
    old.commit()
    old.close()

    assert db.get_review(db.connect(path), "old-job")["source"] == "review"
