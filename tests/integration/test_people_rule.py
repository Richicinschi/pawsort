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
def ui(orch_client):
    return TestClient(create_website(orch_client))


@pytest.mark.parametrize("confidence,marked", [(1.0, False), (0.8, False), (0.79, True), (0.2, True)])
def test_middle_server_deletes_a_person_photo_at_any_confidence(orch_client, send_photo, photos, confidence, marked):
    job_id = send_photo(orch_client, site_code="A12")
    orch_client.post("/jobs/claim")
    assert len(os.listdir(photos)) == 1

    r = report(orch_client, job_id, "person", confidence, marked)

    assert r.status_code == 200
    assert os.listdir(photos) == []
    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["label"], job["needs_review"], job["photo_name"]) == ("person", 0, "")


def test_person_photo_is_gone_from_both_servers_after_the_check(orch_client, send_photo, photos, work):
    job_id = send_photo(orch_client)

    process_one(orch_client, lambda image: {"person": 1.0}, work)

    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["status"], job["label"], job["photo_name"]) == ("done", "person", "")
    assert os.listdir(photos) == [] and os.listdir(work) == []


def test_person_photo_that_is_already_gone_is_fine(orch_client, send_photo, photos):
    job_id = send_photo(orch_client)
    orch_client.post("/jobs/claim")
    os.remove(photos / os.listdir(photos)[0])

    assert report(orch_client, job_id, "person", 1.0, False).status_code == 200


def test_middle_server_only_deletes_for_the_label_person(orch_client, send_photo, photos):
    job_id = send_photo(orch_client)
    orch_client.post("/jobs/claim")

    report(orch_client, job_id, "bobcat", 0.9, False)

    assert os.listdir(photos) == [orch_client.get(f"/jobs/{job_id}").json()["photo_name"]]


def test_deleting_one_person_photo_leaves_the_other_photos(orch_client, send_photo, photos):
    animal, person = send_photo(orch_client), send_photo(orch_client)
    orch_client.post("/jobs/claim")
    orch_client.post("/jobs/claim")

    report(orch_client, animal, "bobcat", 0.9, False)
    report(orch_client, person, "person", 1.0, False)

    assert os.listdir(photos) == [orch_client.get(f"/jobs/{animal}").json()["photo_name"]]


def test_refused_answer_deletes_nothing(orch_client, send_photo, photos):
    job_id = send_photo(orch_client)

    assert report(orch_client, job_id, "person", 1.0, False).status_code == 409  # nobody claimed it yet

    assert os.listdir(photos) == [orch_client.get(f"/jobs/{job_id}").json()["photo_name"]]


def test_unknown_job_gets_a_404(orch_client):
    assert report(orch_client, "nope", "person", 1.0, False).status_code == 404


def test_page_says_the_photo_of_a_person_is_deleted(ui, orch_client, work):
    job_id = upload(ui)

    process_one(orch_client, lambda image: {"person": 1.0}, work)

    assert ui.get(f"/result/{job_id}").json() == {
        "status": "done", "label": "person", "percent": 100, "needs_review": False, "note": PERSON_NOTE}


def test_unsure_person_gets_no_needs_review_on_the_page(ui, orch_client, work):
    job_id = upload(ui)

    process_one(orch_client, lambda image: {"person": 0.55, "bobcat": 0.45}, work)

    answer = ui.get(f"/result/{job_id}").json()
    assert (answer["needs_review"], answer["percent"], answer["note"]) == (False, 55, PERSON_NOTE)


def test_only_a_tagged_photo_without_a_person_stays_whatever_way_a_job_ends(tmp_path, work):
    """What the notice promises: a photo stays when we could tag it and it shows no person, and no copy is left."""
    photos = tmp_path / "photos"
    with TestClient(create_app(str(tmp_path / "jobs.db"), str(photos), stuck_seconds=0.5, check_every=0.02)) as orch:
        ui = TestClient(create_website(orch))
        answers = [{"bobcat": 0.9}, {"bobcat": 0.5}, {"empty": 1.0}, {"person": 1.0}]
        jobs = [upload(ui) for _ in answers]
        for scores in answers:
            process_one(orch, lambda image: scores, work)

        broken = upload(ui)
        process_one(orch, lambda image: 1 / 0, work)

        stuck = upload(ui)
        orch.post("/jobs/claim")  # this worker never comes back
        deadline = time.monotonic() + 5
        while orch.get(f"/jobs/{stuck}").json()["photo_name"] and time.monotonic() < deadline:
            time.sleep(0.02)

        assert [ui.get(f"/result/{job}").json()["status"] for job in jobs] == ["done"] * 4
        assert ui.get(f"/result/{broken}").json()["status"] == "failed"
        assert ui.get(f"/result/{stuck}").json()["status"] == "failed"
        kept = [orch.get(f"/jobs/{job}").json()["photo_name"] for job in jobs + [broken, stuck]]
    assert kept[3:] == ["", "", ""]
    assert sorted(os.listdir(photos)) == sorted(kept[:3])
    assert len(os.listdir(photos)) == 3
    assert os.listdir(work) == []
