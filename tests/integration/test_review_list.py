import io
import os
from datetime import datetime

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_server.worker import process_one
from interface.app import PERSON_NOTE, create_app


def photo() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def ui(orch_client, tmp_path):
    return TestClient(create_app(orch_client, str(tmp_path / "uploads")))


def upload(ui, site_code="A12"):
    r = ui.post("/upload", data={"site_code": site_code}, files=[("files", ("trail.jpg", photo()))])
    return r.json()["photos"][0]["job_id"]


def finish(orch, job_id, label="bobcat", probability=0.62, needs_review=True):
    return orch.post(f"/jobs/{job_id}/result", json={"label": label, "probability": probability,
                                                     "needs_review": needs_review, "model_version": "wildlife-9"})


def marked_job(ui, orch, label="bobcat", probability=0.62, site_code="A12"):
    job_id = upload(ui, site_code)
    orch.post("/jobs/claim")
    finish(orch, job_id, label, probability)
    return job_id


def listed(ui):
    return [job["job_id"] for job in ui.get("/review/jobs").json()["jobs"]]


def decide(ui, job_id, label):
    return ui.post(f"/review/{job_id}", json={"label": label})


def test_list_shows_the_job_site_label_and_percentage(ui, orch_client):
    job_id = marked_job(ui, orch_client, "mountain_lion", 0.7996, "B7")

    assert ui.get("/review/jobs").json()["jobs"] == [{
        "job_id": job_id, "site_code": "B7", "label": "mountain_lion", "label_text": "mountain lion", "percent": 79}]


def test_list_never_gives_away_where_the_photo_was(ui, orch_client):
    marked_job(ui, orch_client)
    assert "uploads" not in ui.get("/review/jobs").text


def test_oldest_upload_comes_first(ui, orch_client):
    jobs = [upload(ui) for _ in range(3)]
    for _ in jobs:
        orch_client.post("/jobs/claim")
    for job_id in (jobs[2], jobs[0], jobs[1]):
        finish(orch_client, job_id)

    assert listed(ui) == jobs


def test_two_marked_jobs_and_after_one_confirmation_only_the_other_is_left(ui, orch_client):
    first, second = marked_job(ui, orch_client), marked_job(ui, orch_client)
    assert listed(ui) == [first, second]

    r = decide(ui, first, "bobcat")

    assert r.status_code == 200
    assert listed(ui) == [second]


def test_confirmation_is_stored_with_the_same_old_and_new_label(ui, orch_client):
    job_id = marked_job(ui, orch_client)

    decide(ui, job_id, "bobcat")

    review = orch_client.get(f"/jobs/{job_id}").json()["review"]
    assert (review["old_label"], review["new_label"], review["model_version"]) == ("bobcat", "bobcat", "wildlife-9")


def test_correction_stores_old_label_new_label_version_and_time(ui, orch_client):
    job_id = marked_job(ui, orch_client)

    decide(ui, job_id, "coyote")

    job = orch_client.get(f"/jobs/{job_id}").json()
    review = job["review"]
    assert (job["status"], job["label"]) == ("done", "coyote")
    assert (review["old_label"], review["new_label"], review["model_version"]) == ("bobcat", "coyote", "wildlife-9")
    assert datetime.fromisoformat(review["reviewed_at"])


def test_result_page_shows_the_new_label_without_the_old_percentage(ui, orch_client):
    job_id = marked_job(ui, orch_client)

    assert decide(ui, job_id, "mountain_lion").json() == {
        "status": "done", "label": "mountain lion", "needs_review": False, "reviewed": True}
    assert ui.get(f"/result/{job_id}").json() == {
        "status": "done", "label": "mountain lion", "needs_review": False, "reviewed": True}


def test_every_label_on_the_list_is_accepted(ui, orch_client):
    labels = ui.get("/review/jobs").json()["labels"]
    assert {"value": "mountain_lion", "text": "mountain lion"} in labels
    assert {"value": "person", "text": "person"} in labels and {"value": "empty", "text": "empty"} in labels

    for label in labels:
        assert decide(ui, marked_job(ui, orch_client), label["value"]).status_code == 200


def test_label_that_is_not_on_the_list_is_refused(ui, orch_client):
    job_id = marked_job(ui, orch_client)

    r = decide(ui, job_id, "unicorn")

    assert r.status_code == 422 and "pick a label" in r.json()["detail"]
    assert listed(ui) == [job_id]


def test_second_click_is_refused_and_the_first_answer_stays(ui, orch_client):
    job_id = marked_job(ui, orch_client)
    decide(ui, job_id, "coyote")

    r = decide(ui, job_id, "deer")

    assert r.status_code == 409 and "already checked" in r.json()["detail"]
    assert orch_client.get(f"/jobs/{job_id}").json()["review"]["new_label"] == "coyote"


def test_unknown_job(ui):
    r = decide(ui, "nope", "coyote")
    assert r.status_code == 404 and "find this job" in r.json()["detail"]


def test_job_still_in_progress_cannot_be_reviewed(ui, orch_client):
    job_id = upload(ui)
    orch_client.post("/jobs/claim")
    assert decide(ui, job_id, "coyote").status_code == 409


def test_sure_tag_failed_job_and_person_are_not_on_the_list(ui, orch_client):
    sure = upload(ui)
    orch_client.post("/jobs/claim")
    finish(orch_client, sure, probability=0.95, needs_review=False)
    person = upload(ui)
    orch_client.post("/jobs/claim")
    finish(orch_client, person, "person", 0.4)
    failed = upload(ui)
    orch_client.post("/jobs/claim")
    orch_client.post(f"/jobs/{failed}/result", json={"error": "broken photo"})

    assert listed(ui) == []


def test_changing_the_label_to_person_deletes_what_is_left(ui, orch_client, copy_photo):
    path = copy_photo("bobcat_1.jpg")  # a photo the AI server did not delete
    job_id = orch_client.post("/jobs", json={"image_path": path, "site_code": "A12"}).json()["id"]
    orch_client.post("/jobs/claim")
    finish(orch_client, job_id)

    answer = decide(ui, job_id, "person").json()

    assert answer["note"] == PERSON_NOTE
    assert not os.path.exists(path)
    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["label"], job["image_path"]) == ("person", "")
    assert job["review"]["old_label"] == "bobcat"
    assert listed(ui) == []


def test_a_refused_person_label_deletes_nothing(ui, orch_client, copy_photo):
    path = copy_photo("bobcat_1.jpg")
    job_id = orch_client.post("/jobs", json={"image_path": path}).json()["id"]
    orch_client.post("/jobs/claim")
    finish(orch_client, job_id, needs_review=False)  # a sure tag, so not on the list

    assert decide(ui, job_id, "person").status_code == 409
    assert os.path.exists(path)


def test_whole_loop_from_upload_to_the_stored_correction(ui, orch_client):
    job_id = upload(ui, "C3")

    process_one(orch_client, lambda image: {"bobcat": 0.55, "coyote": 0.45}, "wildlife-9")

    assert ui.get(f"/result/{job_id}").json()["needs_review"] is True
    assert ui.get("/review/jobs").json()["jobs"][0]["site_code"] == "C3"
    decide(ui, job_id, "coyote")
    assert listed(ui) == []
    assert orch_client.get(f"/jobs/{job_id}").json()["review"]["new_label"] == "coyote"


def test_page_says_the_limit_in_plain_words(ui):
    r = ui.get("/review")
    assert r.status_code == 200
    page = r.text
    limit = page[page.index('id="limit"'):page.index('id="message"')]
    assert "we cannot show you the photo" in limit
    assert "the job, the camera site, the label and the percentage" in limit
    assert "<img" not in page


def test_upload_page_links_to_the_list(ui):
    assert 'href="/review"' in ui.get("/").text


def test_empty_list(ui):
    assert ui.get("/review/jobs").json()["jobs"] == []


def test_middle_server_down_gets_a_friendly_message(tmp_path):
    down = httpx.Client(base_url="http://127.0.0.1:9", timeout=1)
    ui = TestClient(create_app(down, str(tmp_path / "uploads")))

    for r in (ui.get("/review/jobs"), decide(ui, "job-1", "coyote")):
        assert r.status_code == 503 and "try again" in r.json()["detail"]


def test_middle_server_refuses_a_review_of_an_unknown_label_on_its_own(orch_client):
    job_id = orch_client.post("/jobs", json={"image_path": "/x/cat.jpg"}).json()["id"]
    orch_client.post("/jobs/claim")
    finish(orch_client, job_id)

    assert orch_client.post(f"/jobs/{job_id}/review", json={"label": "unicorn"}).status_code == 422
    assert orch_client.get(f"/jobs/{job_id}").json()["review"] is None


def test_job_row_says_review_is_empty_until_somebody_checks(orch_client):
    job_id = orch_client.post("/jobs", json={"image_path": "/x/cat.jpg"}).json()["id"]
    assert orch_client.get(f"/jobs/{job_id}").json()["review"] is None
