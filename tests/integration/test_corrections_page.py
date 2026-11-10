import io
import os
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_server.worker import process_one
from interface.app import create_app as create_website
from orchestrator.app import create_app


def photo() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def ui(orch_client):
    return TestClient(create_website(orch_client))


def corrected(ui, orch, work, new_label="coyote"):
    """Upload a photo, let a stand-in model call it a bobcat at 60%, and let a user change the label."""
    r = ui.post("/upload", data={"site_code": "B7"}, files=[("files", ("trail.jpg", photo()))])
    process_one(orch, lambda image: {"bobcat": 0.6, "coyote": 0.4}, work, "wildlife-9")
    job_id = r.json()["photos"][0]["job_id"]
    ui.post(f"/review/{job_id}", json={"label": new_label})
    return job_id


def decide(ui, job_id, decision):
    return ui.post(f"/corrections/{job_id}", json={"decision": decision})


def listed(ui):
    return [job["job_id"] for job in ui.get("/corrections/jobs").json()["jobs"]]


def test_page_lists_every_correction_that_waits(ui, orch_client, work):
    job_id = corrected(ui, orch_client, work)
    corrected(ui, orch_client, work, "bobcat")
    corrected(ui, orch_client, work, "person")

    assert ui.get("/corrections/jobs").json() == {
        "jobs": [{"job_id": job_id, "site_code": "B7", "old_label": "bobcat", "new_label": "coyote",
                  "model_version": "wildlife-9"}],
        "note": "0 approved corrections wait for the next training."}


def test_approve_takes_it_off_the_page_and_counts_it_for_the_training(ui, orch_client, work):
    job_id, other = corrected(ui, orch_client, work), corrected(ui, orch_client, work, "deer")

    r = decide(ui, job_id, "approved")

    assert (r.status_code, r.json()) == (200, {"note": "1 approved correction waits for the next training."})
    assert listed(ui) == [other]
    assert ui.get(f"/result/{job_id}").json()["label"] == "coyote"


def test_reject_takes_it_off_the_page_and_shows_the_old_label_again(ui, orch_client, work):
    job_id = corrected(ui, orch_client, work)

    r = decide(ui, job_id, "rejected")

    assert (r.status_code, r.json()) == (200, {"note": "0 approved corrections wait for the next training."})
    assert listed(ui) == []
    assert ui.get(f"/result/{job_id}").json() == {
        "status": "done", "label": "bobcat", "needs_review": False, "reviewed": True}


def test_second_click_is_refused_in_plain_words(ui, orch_client, work):
    job_id = corrected(ui, orch_client, work)
    decide(ui, job_id, "approved")

    r = decide(ui, job_id, "rejected")

    assert r.status_code == 409 and "already has a decision" in r.json()["detail"]
    assert ui.get(f"/result/{job_id}").json()["label"] == "coyote"


def test_unknown_job_and_unknown_decision_are_refused_in_plain_words(ui, orch_client, work):
    job_id = corrected(ui, orch_client, work)

    r = decide(ui, str(uuid.uuid4()), "approved")
    assert r.status_code == 404 and "find this correction" in r.json()["detail"]
    r = decide(ui, job_id, "maybe")
    assert r.status_code == 422 and "Approve or Reject" in r.json()["detail"]
    assert listed(ui) == [job_id]


@pytest.mark.parametrize("address", ["claim%3F", "nope"])
def test_address_that_is_no_job_number_never_reaches_the_middle_server(ui, orch_client, work, address):
    job_id = corrected(ui, orch_client, work)

    assert ui.post(f"/corrections/{address}", json={"decision": "approved"}).status_code == 422
    assert ui.get(f"/corrections/{address}/photo").status_code == 422
    assert listed(ui) == [job_id]


def test_website_passes_on_the_photo_only_while_the_correction_waits(ui, orch_client, work, photos):
    job_id = corrected(ui, orch_client, work)
    (name,) = os.listdir(photos)

    r = ui.get(f"/corrections/{job_id}/photo")
    assert (r.status_code, r.headers["content-type"], r.headers["cache-control"]) == (200, "image/jpeg", "no-store")
    assert r.content == (photos / name).read_bytes()

    decide(ui, job_id, "approved")
    assert ui.get(f"/corrections/{job_id}/photo").status_code == 404


def test_no_photo_for_a_confirmation_a_waiting_job_or_an_unknown_job(ui, orch_client, work):
    confirmed = corrected(ui, orch_client, work, "bobcat")
    waiting = ui.post("/upload", data={"site_code": "B7"}, files=[("files", ("trail.jpg", photo()))])

    for job_id in (confirmed, waiting.json()["photos"][0]["job_id"], str(uuid.uuid4())):
        assert ui.get(f"/corrections/{job_id}/photo").status_code == 404


def test_page_says_when_it_is_time_to_plan_a_training_round(tmp_path, work):
    orch = TestClient(create_app(str(tmp_path / "jobs.db"), str(tmp_path / "photos"), training_round=2))
    ui = TestClient(create_website(orch))
    first, second = corrected(ui, orch, work), corrected(ui, orch, work, "deer")

    assert decide(ui, first, "approved").json() == {"note": "1 approved correction waits for the next training."}
    assert decide(ui, second, "approved").json() == {
        "note": "2 approved corrections wait for the next training. It is time to plan a training round."}
    assert "It is time to plan a training round." in ui.get("/corrections/jobs").json()["note"]


def test_page_loads_with_a_link_back_and_says_who_it_is_for(ui):
    r = ui.get("/corrections")
    assert r.status_code == 200
    assert 'href="/"' in r.text
    assert "This page is meant for the coordinator. There is no login yet, so everybody can use it." in r.text
    for outside in ("http://", "https://", "<link", "<script src"):
        assert outside not in r.text


def test_upload_page_links_to_the_corrections(ui):
    assert 'href="/corrections"' in ui.get("/").text


def test_middle_server_down_gets_a_friendly_message():
    down = httpx.Client(base_url="http://127.0.0.1:9", timeout=1)
    ui = TestClient(create_website(down))

    for r in (ui.get("/corrections/jobs"), decide(ui, str(uuid.uuid4()), "approved"),
              ui.get(f"/corrections/{uuid.uuid4()}/photo")):
        assert r.status_code == 503 and "try again" in r.json()["detail"]


def test_website_has_no_address_for_the_training(ui, orch_client, work):
    job_id = corrected(ui, orch_client, work)
    decide(ui, job_id, "approved")

    assert ui.get("/training/corrections").status_code == 404
    assert ui.get(f"/training/corrections/{job_id}/photo").status_code == 404
    assert ui.post("/training/corrections/used", json={"jobs": [job_id]}).status_code == 404
    assert [job["id"] for job in orch_client.get("/training/corrections").json()["jobs"]] == [job_id]
