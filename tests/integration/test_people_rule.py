import io
import os
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_server.worker import process_one
from interface.app import PERSON_NOTE, create_app as create_website
from orchestrator.app import create_app


def photo() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


def upload(ui):
    r = ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("trail.jpg", photo()))])
    return r.json()["photos"][0]["job_id"]


def report(orch, job_id, label, probability, needs_review):
    return orch.post(f"/jobs/{job_id}/result",
                     json={"label": label, "probability": probability, "needs_review": needs_review})


@pytest.fixture
def uploads(tmp_path):
    return tmp_path / "uploads"


@pytest.fixture
def ui(orch_client, uploads):
    return TestClient(create_website(orch_client, str(uploads)))


@pytest.mark.parametrize("confidence,marked", [(1.0, False), (0.8, False), (0.79, True), (0.2, True)])
def test_middle_server_deletes_a_person_photo_at_any_confidence(orch_client, copy_photo, confidence, marked):
    path = copy_photo("bobcat_1.jpg")
    job_id = orch_client.post("/jobs", json={"image_path": path, "site_code": "A12"}).json()["id"]
    orch_client.post("/jobs/claim")

    r = report(orch_client, job_id, "person", confidence, marked)  # no worker here, so nobody else deletes it

    assert r.status_code == 200
    assert not os.path.exists(path)
    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["label"], job["needs_review"], job["image_path"]) == ("person", 0, "")


def test_person_photo_that_the_worker_already_deleted_is_fine(orch_client, copy_photo):
    job_id = orch_client.post("/jobs", json={"image_path": copy_photo("bobcat_1.jpg")}).json()["id"]

    process_one(orch_client, predict=lambda image: {"person": 1.0})

    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["status"], job["label"], job["image_path"]) == ("done", "person", "")


def test_middle_server_only_deletes_for_the_label_person(orch_client, copy_photo):
    path = copy_photo("bobcat_1.jpg")
    job_id = orch_client.post("/jobs", json={"image_path": path}).json()["id"]
    orch_client.post("/jobs/claim")

    report(orch_client, job_id, "bobcat", 0.9, False)

    assert os.path.exists(path)


def test_refused_answer_deletes_nothing(orch_client, copy_photo):
    path = copy_photo("bobcat_1.jpg")
    job_id = orch_client.post("/jobs", json={"image_path": path}).json()["id"]

    assert report(orch_client, job_id, "person", 1.0, False).status_code == 409  # nobody claimed it yet

    assert os.path.exists(path)
    assert orch_client.get(f"/jobs/{job_id}").json()["image_path"] == path


def test_unknown_job_gets_a_404(orch_client):
    assert report(orch_client, "nope", "person", 1.0, False).status_code == 404


def test_page_says_the_photo_of_a_person_is_deleted(ui, orch_client):
    job_id = upload(ui)

    process_one(orch_client, predict=lambda image: {"person": 1.0})

    assert ui.get(f"/result/{job_id}").json() == {
        "status": "done", "label": "person", "percent": 100, "needs_review": False, "note": PERSON_NOTE}


def test_unsure_person_gets_no_needs_review_on_the_page(ui, orch_client):
    job_id = upload(ui)

    process_one(orch_client, predict=lambda image: {"person": 0.55, "bobcat": 0.45})

    answer = ui.get(f"/result/{job_id}").json()
    assert (answer["needs_review"], answer["percent"], answer["note"]) == (False, 55, PERSON_NOTE)


def test_notice_next_to_the_button_says_the_rule(ui):
    page = ui.get("/").text
    notice = page[page.index('id="notice"'):page.index("</form>")]
    assert "We delete every photo" in notice
    assert "photo of a person is always deleted" in notice
    assert "if (job.note) show(answer" in page


def test_notice_is_true_for_every_kind_of_job(tmp_path):
    """The notice says we delete every photo, so after each way a job can end no photo may be left."""
    uploads = tmp_path / "uploads"
    with TestClient(create_app(str(tmp_path / "jobs.db"), stuck_seconds=0.5, check_every=0.02)) as orch:
        ui = TestClient(create_website(orch, str(uploads)))
        answers = [{"bobcat": 0.9}, {"bobcat": 0.5}, {"empty": 1.0}, {"person": 1.0}]
        jobs = [upload(ui) for _ in answers]
        for scores in answers:
            process_one(orch, predict=lambda image: scores)

        broken = upload(ui)
        process_one(orch, predict=lambda image: 1 / 0)

        stuck = upload(ui)
        orch.post("/jobs/claim")  # this worker never comes back
        deadline = time.monotonic() + 5
        while ui.get(f"/result/{stuck}").json()["status"] != "failed" and time.monotonic() < deadline:
            time.sleep(0.02)

        assert [ui.get(f"/result/{job}").json()["status"] for job in jobs] == ["done"] * 4
        assert ui.get(f"/result/{broken}").json()["status"] == "failed"
        assert ui.get(f"/result/{stuck}").json()["status"] == "failed"
    assert os.listdir(uploads) == []
