import io
import os
import uuid
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
def ui(orch_client):
    return TestClient(create_app(orch_client))


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
        "job_id": job_id, "site_code": "B7", "label": "mountain_lion", "label_text": "mountain lion", "percent": 79,
        "reason": "The model is less than 80% sure this is a mountain lion.", "second_guess": None,
        "model_version": "wildlife-9"}]


def test_list_never_gives_away_the_name_of_the_photo(ui, orch_client, photos):
    marked_job(ui, orch_client)
    (name,) = os.listdir(photos)
    assert name not in ui.get("/review/jobs").text


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
    r = decide(ui, str(uuid.uuid4()), "coyote")
    assert r.status_code == 404 and "find this job" in r.json()["detail"]


@pytest.mark.parametrize("address", ["claim%3F", "nope"])
def test_address_that_is_no_job_number_never_reaches_the_middle_server(ui, orch_client, address):
    waiting = upload(ui)

    r = ui.post(f"/review/{address}", json={"label": "coyote"})

    assert r.status_code == 422
    assert orch_client.get(f"/jobs/{waiting}").json()["status"] == "new"


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


def test_changing_the_label_to_person_deletes_the_photo(ui, orch_client, photos):
    job_id = marked_job(ui, orch_client)
    assert len(os.listdir(photos)) == 1

    answer = decide(ui, job_id, "person").json()

    assert answer["note"] == PERSON_NOTE
    assert os.listdir(photos) == []
    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["label"], job["photo_name"]) == ("person", "")
    assert job["review"]["old_label"] == "bobcat"
    assert listed(ui) == []


def test_a_refused_person_label_deletes_nothing(ui, orch_client, photos):
    job_id = upload(ui)
    orch_client.post("/jobs/claim")
    finish(orch_client, job_id, needs_review=False)  # a sure tag, so not on the list

    assert decide(ui, job_id, "person").status_code == 409
    assert os.listdir(photos) == [orch_client.get(f"/jobs/{job_id}").json()["photo_name"]]


def test_confirmed_and_corrected_jobs_keep_their_photo(ui, orch_client, photos):
    confirmed, corrected = marked_job(ui, orch_client), marked_job(ui, orch_client)

    decide(ui, confirmed, "bobcat")
    decide(ui, corrected, "coyote")

    kept = [orch_client.get(f"/jobs/{job_id}").json()["photo_name"] for job_id in (confirmed, corrected)]
    assert sorted(os.listdir(photos)) == sorted(kept)


def test_whole_loop_from_upload_to_the_stored_correction(ui, orch_client, work):
    job_id = upload(ui, "C3")

    process_one(orch_client, lambda image: {"bobcat": 0.55, "coyote": 0.45}, work, "wildlife-9")

    assert ui.get(f"/result/{job_id}").json()["needs_review"] is True
    assert ui.get("/review/jobs").json()["jobs"][0]["site_code"] == "C3"
    decide(ui, job_id, "coyote")
    assert listed(ui) == []
    assert orch_client.get(f"/jobs/{job_id}").json()["review"]["new_label"] == "coyote"


def test_upload_page_links_to_the_list(ui):
    assert 'href="/review"' in ui.get("/").text


def test_empty_list(ui):
    assert ui.get("/review/jobs").json()["jobs"] == []


def test_middle_server_down_gets_a_friendly_message():
    down = httpx.Client(base_url="http://127.0.0.1:9", timeout=1)
    ui = TestClient(create_app(down))

    for r in (ui.get("/review/jobs"), decide(ui, str(uuid.uuid4()), "coyote")):
        assert r.status_code == 503 and "try again" in r.json()["detail"]


def test_middle_server_refuses_a_review_of_an_unknown_label_on_its_own(orch_client, send_photo):
    job_id = send_photo(orch_client)
    orch_client.post("/jobs/claim")
    finish(orch_client, job_id)

    assert orch_client.post(f"/jobs/{job_id}/review", json={"label": "unicorn"}).status_code == 422
    assert orch_client.get(f"/jobs/{job_id}").json()["review"] is None


def test_job_row_says_review_is_empty_until_somebody_checks(orch_client, send_photo):
    job_id = send_photo(orch_client)
    assert orch_client.get(f"/jobs/{job_id}").json()["review"] is None
