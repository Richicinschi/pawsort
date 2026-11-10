import pytest

from orchestrator import db


@pytest.fixture
def conn(tmp_path):
    return db.connect(str(tmp_path / "jobs.db"))


def tagged(conn, label="bobcat", site="A12", marked=False):
    job_id = db.create_job(conn, "cat.jpg", site)
    db.claim_job(conn)
    db.set_result(conn, job_id, label, 0.6 if marked else 0.9, marked, "v7")
    return job_id


def tag(conn, job_id):
    return next(row for row in db.tags(conn) if row["id"] == job_id)


def test_list_of_rare_animals_starts_empty(conn):
    assert db.rare_animals(conn) == []


def test_animal_can_be_added_and_removed(conn):
    db.set_rare(conn, "mountain_lion", True)
    db.set_rare(conn, "badger", True)
    assert db.rare_animals(conn) == ["badger", "mountain_lion"]

    db.set_rare(conn, "badger", False)
    assert db.rare_animals(conn) == ["mountain_lion"]


def test_adding_twice_and_removing_what_is_not_there_is_fine(conn):
    db.set_rare(conn, "badger", True)
    db.set_rare(conn, "badger", True)
    db.set_rare(conn, "fox", False)
    assert db.rare_animals(conn) == ["badger"]


def test_tag_row_has_everything_the_download_needs(conn):
    job_id = tagged(conn)
    row = tag(conn, job_id)
    assert {key: row[key] for key in row if key != "created_at"} == {
        "id": job_id, "site_code": "A12", "label": "bobcat", "probability": 0.9, "needs_review": False,
        "checked": False, "model_version": "v7"}
    assert row["created_at"] == db.get_job(conn, job_id)["created_at"]


def test_only_jobs_that_are_done_have_a_tag(conn):
    done = tagged(conn)
    failed = db.create_job(conn, "broken.jpg")
    db.claim_job(conn)
    db.set_failed(conn, failed, "broken photo")
    db.create_job(conn, "waiting.jpg")

    assert [row["id"] for row in db.tags(conn)] == [done]


def test_tags_come_in_the_order_of_the_job_numbers_and_not_of_the_uploads(conn):
    jobs = sorted(tagged(conn) for _ in range(4))
    for day, job_id in enumerate(reversed(jobs), start=1):
        conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (f"2026-11-0{day}", job_id))

    assert [row["id"] for row in db.tags(conn)] == jobs
    assert [row["created_at"] for row in db.tags(conn)] == ["2026-11-04", "2026-11-03", "2026-11-02", "2026-11-01"]


def test_rare_animal_has_no_time_of_upload_either(conn):
    rare, common = tagged(conn, "mountain_lion"), tagged(conn, "bobcat")
    db.set_rare(conn, "mountain_lion", True)

    assert tag(conn, rare)["created_at"] is None
    assert tag(conn, common)["created_at"] == db.get_job(conn, common)["created_at"]


def test_camera_site_of_a_rare_animal_is_left_out(conn):
    rare, common = tagged(conn, "mountain_lion", "B7"), tagged(conn, "bobcat", "B7")
    db.set_rare(conn, "mountain_lion", True)

    assert (tag(conn, rare)["site_code"], tag(conn, common)["site_code"]) == (None, "B7")


def test_camera_site_comes_back_when_the_animal_is_no_longer_rare(conn):
    job_id = tagged(conn, "mountain_lion", "B7")
    db.set_rare(conn, "mountain_lion", True)
    db.set_rare(conn, "mountain_lion", False)
    assert tag(conn, job_id)["site_code"] == "B7"


def test_tag_that_waits_for_a_person_says_so(conn):
    row = tag(conn, tagged(conn, marked=True))
    assert (row["needs_review"], row["checked"]) == (True, False)


def test_checked_tag_has_the_label_the_person_picked_and_no_longer_needs_review(conn):
    job_id = tagged(conn, marked=True)
    db.save_review(conn, job_id, "coyote")

    row = tag(conn, job_id)

    assert (row["label"], row["needs_review"], row["checked"]) == ("coyote", False, True)


def test_rejected_correction_has_the_old_label_again(conn):
    job_id = tagged(conn, marked=True)
    db.save_review(conn, job_id, "coyote")
    db.decide_correction(conn, job_id, "rejected")
    assert tag(conn, job_id)["label"] == "bobcat"


def test_site_is_hidden_by_the_label_a_person_picked_and_not_by_the_first_tag_of_the_model(conn):
    made_rare, made_common = tagged(conn, "bobcat", "B7", marked=True), tagged(conn, "mountain_lion", "C3", marked=True)
    db.set_rare(conn, "mountain_lion", True)

    db.save_review(conn, made_rare, "mountain_lion")
    db.save_review(conn, made_common, "bobcat")

    assert (tag(conn, made_rare)["site_code"], tag(conn, made_common)["site_code"]) == (None, "C3")
