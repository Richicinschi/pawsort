import io
import os
import sqlite3
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_server.worker import poll, process_one
from interface.app import create_app as create_website
from orchestrator import db
from orchestrator.app import create_app

FRIENDLY = "We could not check this photo. Please try another one."


def stand_in(image):
    return {"coyote": 0.9, "bobcat": 0.1}


def photo() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def wait_for(orch, job_id, status):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = orch.get(f"/jobs/{job_id}").json()
        if job["status"] == status:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job never reached {status}, it is {job['status']}")


@pytest.fixture
def quick_orch(tmp_path, photos):
    """A real orchestrator with its check running, that gives up on a job after half a second."""
    app = create_app(str(tmp_path / "jobs.db"), str(photos), stuck_seconds=0.5, check_every=0.02)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def ui(quick_orch):
    return TestClient(create_website(quick_orch))


def upload(ui):
    r = ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("cat.jpg", photo()))])
    return r.json()["photos"][0]["job_id"]


def test_worker_dies_and_the_page_shows_the_friendly_message(ui, quick_orch):
    job_id = upload(ui)
    quick_orch.post("/jobs/claim")  # a worker takes the job and is never heard of again
    assert ui.get(f"/result/{job_id}").json() == {"status": "processing"}

    wait_for(quick_orch, job_id, "failed")

    assert ui.get(f"/result/{job_id}").json() == {"status": "failed", "error": FRIENDLY}


def wait_until_forgotten(orch, job_id):
    deadline = time.monotonic() + 5
    while orch.get(f"/jobs/{job_id}").json()["photo_name"] and time.monotonic() < deadline:
        time.sleep(0.02)


def test_photo_of_a_stuck_job_is_deleted(ui, quick_orch, photos):
    job_id = upload(ui)
    quick_orch.post("/jobs/claim")
    assert len(os.listdir(photos)) == 1

    wait_for(quick_orch, job_id, "failed")
    wait_until_forgotten(quick_orch, job_id)

    assert os.listdir(photos) == []
    assert quick_orch.get(f"/jobs/{job_id}").json()["photo_name"] == ""


def test_late_person_answer_for_a_stuck_job_is_refused_and_no_photo_is_left(ui, quick_orch, photos):
    job_id = upload(ui)
    quick_orch.post("/jobs/claim")
    wait_for(quick_orch, job_id, "failed")

    r = quick_orch.post(f"/jobs/{job_id}/result", json={"label": "person", "probability": 1.0, "needs_review": False})

    assert r.status_code == 409
    wait_until_forgotten(quick_orch, job_id)
    job = quick_orch.get(f"/jobs/{job_id}").json()
    assert (job["status"], job["label"], job["photo_name"]) == ("failed", None, "")
    assert os.listdir(photos) == []


def test_stuck_job_never_blocks_the_next_one(quick_orch, send_photo, work):
    stuck = send_photo(quick_orch, "bobcat_1.jpg")
    assert quick_orch.post("/jobs/claim").json()["id"] == stuck

    next_job = send_photo(quick_orch, "coyote_1.jpg")
    assert process_one(quick_orch, stand_in, work) is True
    assert quick_orch.get(f"/jobs/{next_job}").json()["label"] == "coyote"

    wait_for(quick_orch, stuck, "failed")
    last_job = send_photo(quick_orch, "empty_1.jpg")
    assert process_one(quick_orch, stand_in, work) is True
    assert quick_orch.get(f"/jobs/{last_job}").json()["status"] == "done"


def test_job_answered_in_time_stays_done(quick_orch, send_photo, work):
    job_id = send_photo(quick_orch)
    process_one(quick_orch, stand_in, work)

    time.sleep(0.7)

    assert quick_orch.get(f"/jobs/{job_id}").json()["status"] == "done"


def test_waiting_job_is_not_stuck(quick_orch, send_photo):
    job_id = send_photo(quick_orch)

    time.sleep(0.7)

    assert quick_orch.get(f"/jobs/{job_id}").json()["status"] == "new"


def test_slow_worker_gets_refused_and_carries_on_with_the_next_job(quick_orch, send_photo, work):
    slow = send_photo(quick_orch, "bobcat_1.jpg")
    next_job = send_photo(quick_orch, "coyote_1.jpg")

    def too_slow(image):
        wait_for(quick_orch, slow, "failed")
        return stand_in(image)

    poll(quick_orch, too_slow, work, wait=lambda seconds: None)
    job = quick_orch.get(f"/jobs/{slow}").json()
    assert (job["status"], job["label"]) == ("failed", None)
    assert os.listdir(work) == []

    poll(quick_orch, stand_in, work, wait=lambda seconds: None)
    assert quick_orch.get(f"/jobs/{next_job}").json()["status"] == "done"


def test_late_answer_gets_a_409(quick_orch, send_photo):
    job_id = send_photo(quick_orch)
    quick_orch.post("/jobs/claim")
    wait_for(quick_orch, job_id, "failed")

    r = quick_orch.post(f"/jobs/{job_id}/result", json={"label": "cat", "probability": 0.9, "needs_review": False})

    assert r.status_code == 409


def test_job_stuck_before_a_restart_still_fails(tmp_path, send_photo):
    path, photos = str(tmp_path / "jobs.db"), str(tmp_path / "photos")
    before = TestClient(create_app(path, photos))
    job_id = send_photo(before)
    before.post("/jobs/claim")

    with TestClient(create_app(path, photos, stuck_seconds=0, check_every=0.02)) as after:
        assert wait_for(after, job_id, "failed")["error"]


def test_busy_database_does_not_stop_the_check(tmp_path, send_photo, monkeypatch):
    real_check = db.fail_stuck_jobs
    calls = []

    def busy_the_first_time(conn, seconds):
        calls.append(seconds)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_check(conn, seconds)

    monkeypatch.setattr(db, "fail_stuck_jobs", busy_the_first_time)
    app = create_app(str(tmp_path / "jobs.db"), str(tmp_path / "photos"), stuck_seconds=0.1, check_every=0.02)
    with TestClient(app) as orch:
        job_id = send_photo(orch)
        orch.post("/jobs/claim")
        wait_for(orch, job_id, "failed")

    assert len(calls) > 1


def test_real_500_from_the_middle_server_does_not_stop_the_worker(tmp_path, send_photo, work, monkeypatch):
    app = create_app(str(tmp_path / "jobs.db"), str(tmp_path / "photos"))
    orch = TestClient(app, raise_server_exceptions=False)
    job_id = send_photo(orch)
    real_claim = db.claim_job

    def locked(conn):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "claim_job", locked)
    waits = []
    poll(orch, stand_in, work, wait=waits.append)
    assert waits == [1.0]

    monkeypatch.setattr(db, "claim_job", real_claim)
    poll(orch, stand_in, work, wait=waits.append)
    assert orch.get(f"/jobs/{job_id}").json()["status"] == "done"
