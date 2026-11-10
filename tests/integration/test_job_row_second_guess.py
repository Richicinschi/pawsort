import sqlite3

import pytest
from fastapi.testclient import TestClient

from ai_server.worker import process_one
from orchestrator.app import create_app


@pytest.fixture
def claimed_job(orch_client, send_photo):
    job_id = send_photo(orch_client)
    orch_client.post("/jobs/claim")
    return job_id


def report(orch, job_id, **second):
    return orch.post(f"/jobs/{job_id}/result",
                     json={"label": "raccoon", "probability": 0.57, "needs_review": True, **second})


def second_guess(orch, job_id):
    job = orch.get(f"/jobs/{job_id}").json()
    return job["second_label"], job["second_probability"]


def test_job_row_keeps_the_second_guess_of_the_worker(orch_client, send_photo, work):
    job_id = send_photo(orch_client)

    process_one(orch_client, lambda image: {"raccoon": 0.57, "cat": 0.12, "skunk": 0.21}, work)

    assert second_guess(orch_client, job_id) == ("skunk", 0.21)


def test_job_row_of_an_empty_photo_has_no_second_guess(orch_client, send_photo, work):
    job_id = send_photo(orch_client, "empty_1.jpg")

    process_one(orch_client, lambda image: {"empty": 1.0}, work)

    assert second_guess(orch_client, job_id) == (None, None)


def test_failed_job_has_no_second_guess(orch_client, send_photo, work):
    job_id = send_photo(orch_client, "not_an_image.jpg")

    process_one(orch_client, lambda image: {"raccoon": 0.57, "skunk": 0.21}, work)

    assert second_guess(orch_client, job_id) == (None, None)


def test_answer_without_a_second_guess_is_still_accepted(orch_client, claimed_job):
    assert report(orch_client, claimed_job).status_code == 200
    assert second_guess(orch_client, claimed_job) == (None, None)


def test_review_list_of_the_middle_server_has_the_second_guess_and_the_version(orch_client, claimed_job):
    report(orch_client, claimed_job, second_label="skunk", second_probability=0.21, model_version="wildlife-9")

    (row,) = orch_client.get("/review").json()["jobs"]

    assert (row["id"], row["second_label"], row["second_probability"], row["model_version"]) == (
        claimed_job, "skunk", 0.21, "wildlife-9")


def test_second_guess_that_is_not_a_known_label_is_refused(orch_client, claimed_job):
    r = report(orch_client, claimed_job, second_label="unicorn", second_probability=0.21)

    assert r.status_code == 422
    assert orch_client.get(f"/jobs/{claimed_job}").json()["status"] == "processing"


def test_second_guess_without_its_score_is_refused(orch_client, claimed_job):
    assert report(orch_client, claimed_job, second_label="skunk").status_code == 422
    assert orch_client.get(f"/jobs/{claimed_job}").json()["status"] == "processing"


def test_middle_server_works_on_a_database_from_before_these_columns(tmp_path, send_photo):
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, image_path TEXT NOT NULL, status TEXT NOT NULL, label TEXT, "
                "probability REAL, needs_review INTEGER, error TEXT, site_code TEXT, model_version TEXT, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    old.execute("INSERT INTO jobs (id, image_path, status, label, probability, needs_review, created_at, updated_at) "
                "VALUES ('old-job', '/x/old.jpg', 'done', 'cat', 0.6, 1, '2026-11-01', '2026-11-01')")
    old.commit()
    old.close()

    orch = TestClient(create_app(path, str(tmp_path / "photos")))

    assert second_guess(orch, "old-job") == (None, None)
    assert orch.get("/review").json()["jobs"][0]["second_label"] is None
    job_id = send_photo(orch)
    orch.post("/jobs/claim")
    report(orch, job_id, second_label="skunk", second_probability=0.21)
    assert second_guess(orch, job_id) == ("skunk", 0.21)
