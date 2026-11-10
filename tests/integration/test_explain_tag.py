import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_server.logic import decide
from ai_server.worker import process_one
from interface.app import create_app

RACCOON = "The model is less than 80% sure this is a raccoon."
WEAK_BOX = "The model saw something in this photo, but not clearly enough to say what it is."


def photo() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def ui(orch_client, tmp_path):
    return TestClient(create_app(orch_client, str(tmp_path / "uploads")))


def tagged(ui, orch, predict):
    r = ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("trail.jpg", photo()))])
    process_one(orch, predict, "wildlife-9")
    return r.json()["photos"][0]["job_id"]


def unsure_raccoon(image):
    return {"raccoon": 0.57, "cat": 0.12, "skunk": 0.2149}


def weak_box(image):
    return decide({"category": "1", "conf": 0.1, "bbox": [0, 0, 1, 1]}, unsure_raccoon)


def test_answer_says_why_the_tag_needs_review_and_shows_the_second_guess(ui, orch_client):
    job_id = tagged(ui, orch_client, unsure_raccoon)

    assert ui.get(f"/result/{job_id}").json() == {
        "status": "done", "label": "raccoon", "percent": 57, "needs_review": True,
        "reason": RACCOON, "second_guess": "Its second guess is a skunk, 21% sure."}


def test_answer_for_a_weak_box_has_its_own_reason(ui, orch_client):
    job_id = tagged(ui, orch_client, weak_box)

    assert ui.get(f"/result/{job_id}").json() == {
        "status": "done", "label": "empty", "percent": 50, "needs_review": True,
        "reason": WEAK_BOX, "second_guess": None}


def test_answer_for_a_sure_tag_has_no_reason(ui, orch_client):
    job_id = tagged(ui, orch_client, lambda image: {"bobcat": 0.93, "coyote": 0.07})

    assert ui.get(f"/result/{job_id}").json() == {
        "status": "done", "label": "bobcat", "percent": 93, "needs_review": False}


def test_review_list_shows_the_reason_the_second_guess_and_the_model_version(ui, orch_client):
    tagged(ui, orch_client, unsure_raccoon)
    tagged(ui, orch_client, weak_box)

    raccoon, empty = ui.get("/review/jobs").json()["jobs"]

    assert (raccoon["reason"], raccoon["second_guess"], raccoon["model_version"]) == (
        RACCOON, "Its second guess is a skunk, 21% sure.", "wildlife-9")
    assert (empty["reason"], empty["second_guess"], empty["model_version"]) == (WEAK_BOX, None, "wildlife-9")


def test_upload_page_puts_the_reason_and_the_second_guess_next_to_the_mark(ui):
    page = ui.get("/").text
    assert "Needs review. ${why(job)} A person should check this photo." in page
    assert "[job.reason, job.second_guess]" in page


def test_review_page_puts_the_reason_the_second_guess_and_the_version_under_every_job(ui):
    page = ui.get("/review").text
    assert "[job.reason, job.second_guess, `Tagged by model version ${job.model_version}.`]" in page
