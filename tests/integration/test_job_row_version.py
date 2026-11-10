import sqlite3

from fastapi.testclient import TestClient

from ai_server.worker import poll, process_one
from orchestrator.app import create_app


def stand_in(image):
    return {"bobcat": 0.9, "coyote": 0.1}


def test_job_row_keeps_the_version_after_a_good_photo(orch_client, copy_photo):
    job_id = orch_client.post("/jobs", json={"image_path": copy_photo("bobcat_1.jpg"), "site_code": "A12"}).json()["id"]

    process_one(orch_client, stand_in, "wildlife-9")

    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["status"], job["label"], job["site_code"], job["model_version"]) == (
        "done", "bobcat", "A12", "wildlife-9")


def test_job_row_keeps_the_version_after_a_broken_photo(orch_client, copy_photo):
    job_id = orch_client.post("/jobs", json={"image_path": copy_photo("not_an_image.jpg")}).json()["id"]

    process_one(orch_client, stand_in, "wildlife-9")

    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["status"], job["model_version"]) == ("failed", "wildlife-9")
    assert job["error"]


def test_worker_loop_passes_the_version_on(orch_client, copy_photo):
    job_id = orch_client.post("/jobs", json={"image_path": copy_photo("bobcat_1.jpg")}).json()["id"]

    poll(orch_client, stand_in, "wildlife-9", wait=lambda seconds: None)

    assert orch_client.get(f"/jobs/{job_id}").json()["model_version"] == "wildlife-9"


def test_person_job_keeps_the_version_and_no_photo(orch_client, copy_photo):
    job_id = orch_client.post("/jobs", json={"image_path": copy_photo("bobcat_1.jpg")}).json()["id"]

    process_one(orch_client, lambda image: {"person": 1.0}, "wildlife-9")

    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["label"], job["model_version"], job["image_path"]) == ("person", "wildlife-9", "")


def test_answer_without_a_version_is_still_accepted(orch_client):
    job_id = orch_client.post("/jobs", json={"image_path": "/x/cat.jpg"}).json()["id"]
    orch_client.post("/jobs/claim")

    r = orch_client.post(f"/jobs/{job_id}/result", json={"label": "cat", "probability": 0.9, "needs_review": False})

    assert r.status_code == 200 and r.json()["model_version"] is None


def test_middle_server_works_on_a_database_from_before_this_column(tmp_path):
    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, image_path TEXT NOT NULL, status TEXT NOT NULL, label TEXT, "
                "probability REAL, needs_review INTEGER, error TEXT, site_code TEXT, created_at TEXT NOT NULL, "
                "updated_at TEXT NOT NULL)")
    old.execute("INSERT INTO jobs (id, image_path, status, label, created_at, updated_at) "
                "VALUES ('old-job', '/x/old.jpg', 'done', 'cat', '2026-11-01', '2026-11-01')")
    old.commit()
    old.close()

    orch = TestClient(create_app(path))

    assert orch.get("/jobs/old-job").json()["label"] == "cat"
    job_id = orch.post("/jobs", json={"image_path": "/x/new.jpg"}).json()["id"]
    orch.post("/jobs/claim")
    r = orch.post(f"/jobs/{job_id}/result",
                  json={"label": "cat", "probability": 0.9, "needs_review": False, "model_version": "wildlife-9"})
    assert r.json()["model_version"] == "wildlife-9"
