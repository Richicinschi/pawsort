import io
import json
import os
import sqlite3
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_server.worker import process_one
from interface.app import PERSON_NOTE, create_app as create_website
from orchestrator import numbers
from orchestrator.app import create_app


def photo() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def orch_client(tmp_path, photos):
    """A middle server that knows no training date, so no camera site counts as new."""
    return TestClient(create_app(str(tmp_path / "jobs.db"), str(photos), versions_path=str(tmp_path / "no.json")))


@pytest.fixture
def ui(orch_client):
    return TestClient(create_website(orch_client))


@pytest.fixture(autouse=True)
def week_46(monkeypatch):
    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W46")


@pytest.fixture
def tagged(ui, orch_client, work, tmp_path):
    """Upload a photo on the website and let a stand-in model tag it. The upload counts for the given day."""
    def _tagged(site="A12", day="2026-11-10", scores=None, ui=ui, orch=orch_client):
        r = ui.post("/upload", data={"site_code": site}, files=[("files", ("trail.jpg", photo()))])
        process_one(orch, lambda image: scores or {"bobcat": 0.9, "coyote": 0.1}, work, "wildlife-9")
        job_id = r.json()["photos"][0]["job_id"]
        conn = sqlite3.connect(str(tmp_path / "jobs.db"))
        conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (f"{day}T12:00:00+00:00", job_id))
        conn.commit()
        conn.close()
        return job_id
    return _tagged


UNSURE = {"bobcat": 0.6, "coyote": 0.4}
NOTHING_CHECKED = "Sample of week 2026-W46: 1 sure tag picked, 0 of them checked, 0 of those changed."


def test_numbers_page_gets_a_row_per_camera_site_and_week(ui, tagged):
    tagged()
    tagged(scores=UNSURE)
    checked = tagged("B7", "2026-11-17", UNSURE)
    ui.post(f"/review/{checked}", json={"label": "coyote"})

    assert ui.get("/numbers/rows").json()["rows"] == [
        {"site_code": "B7", "week": "2026-W47", "tags": 1, "needs_review": "100%", "corrected": "100% (of 1 checked)",
         "warning": True},
        {"site_code": "A12", "week": "2026-W46", "tags": 2, "needs_review": "50%", "corrected": "no tag checked yet",
         "warning": True}]


def test_numbers_page_puts_a_warning_on_top_for_a_site_above_40_percent(ui, tagged):
    for scores in (UNSURE, UNSURE, None, None, None):
        tagged("A12", scores=scores)
    for scores in (UNSURE, UNSURE, UNSURE, None, None):
        tagged("B7", scores=scores)

    answer = ui.get("/numbers/rows").json()

    assert answer["warnings"] == [
        "Camera site B7, week 2026-W46: 60% of the tags needed review. That is more than 40%."]
    assert answer["reasons"] == ["Camera site B7 had more than 40% needs review in week 2026-W46."]


def test_numbers_page_without_a_reason_says_nothing_about_a_training_round(ui, tagged):
    tagged()
    answer = ui.get("/numbers/rows").json()
    assert (answer["warnings"], answer["reasons"]) == ([], [])


def site_with(tmp_path, trained_on, training_round=500):
    versions = tmp_path / "versions.json"
    versions.write_text(json.dumps({"current": "v2", "versions": {"v2": {"trained_on": trained_on}}}))
    orch = TestClient(create_app(str(tmp_path / "jobs.db"), str(tmp_path / "photos"), training_round=training_round,
                                 versions_path=str(versions)))
    return TestClient(create_website(orch)), orch


def test_numbers_page_lists_every_reason_to_plan_a_training_round(tmp_path, tagged):
    ui, orch = site_with(tmp_path, "2026-11-10", training_round=1)
    tagged("OLD1", "2026-11-09", ui=ui, orch=orch)
    corrected = tagged("NEW1", "2026-11-11", UNSURE, ui=ui, orch=orch)
    ui.post(f"/review/{corrected}", json={"label": "coyote"})
    ui.post(f"/corrections/{corrected}", json={"decision": "approved"})

    answer = ui.get("/numbers/rows").json()

    assert answer["reasons"] == [
        "1 approved correction waits for the next training.",
        "Camera site NEW1 is new: its first photo came after the current model was trained on 2026-11-10.",
        "Camera site NEW1 had more than 40% needs review in week 2026-W46."]
    assert answer["trained"] == "The current model was trained on 2026-11-10."


def test_numbers_page_says_so_when_the_training_date_is_not_known(tmp_path, tagged):
    ui, orch = site_with(tmp_path, None)
    tagged("NEW1", "2026-11-11", ui=ui, orch=orch)

    answer = ui.get("/numbers/rows").json()

    assert answer["reasons"] == []
    assert answer["trained"] == ("We do not know when the current model was trained, so this page cannot tell "
                                 "which camera sites are new.")


def test_sample_page_lists_the_sure_tags_of_this_weeks_sample(ui, tagged):
    sure = tagged("B7", scores={"mountain_lion": 0.9349, "bobcat": 0.05})
    tagged(scores=UNSURE)

    answer = ui.get("/sample/jobs").json()

    assert answer["jobs"] == [{"job_id": sure, "site_code": "B7", "label": "mountain_lion",
                               "label_text": "mountain lion", "percent": 93, "model_version": "wildlife-9"}]
    assert answer["note"] == NOTHING_CHECKED
    assert {"value": "mountain_lion", "text": "mountain lion"} in answer["labels"]


def test_sample_page_shows_the_photo_only_for_a_tag_of_the_sample_that_nobody_checked(ui, tagged, photos):
    picked = tagged()
    ui.get("/sample/jobs")
    later, unsure = tagged(), tagged(scores=UNSURE)

    r = ui.get(f"/sample/{picked}/photo")
    assert (r.status_code, r.headers["content-type"], r.headers["cache-control"]) == (200, "image/jpeg", "no-store")
    assert r.content in [(photos / name).read_bytes() for name in os.listdir(photos)]
    for job_id in (later, unsure, str(uuid.uuid4())):
        assert ui.get(f"/sample/{job_id}/photo").status_code == 404

    ui.post(f"/sample/{picked}", json={"label": "bobcat"})
    assert ui.get(f"/sample/{picked}/photo").status_code == 404


def test_confirm_on_the_sample_page_counts_as_checked(ui, tagged):
    job_id, other = tagged(), tagged()
    ui.get("/sample/jobs")

    r = ui.post(f"/sample/{job_id}", json={"label": "bobcat"})

    assert (r.status_code, r.json()) == (200, {"status": "done", "label": "bobcat", "needs_review": False,
                                              "reviewed": True})
    answer = ui.get("/sample/jobs").json()
    assert [job["job_id"] for job in answer["jobs"]] == [other]
    assert answer["note"] == "Sample of week 2026-W46: 2 sure tags picked, 1 of them checked, 0 of those changed."
    assert ui.get("/corrections/jobs").json()["jobs"] == []


def test_another_label_on_the_sample_page_becomes_a_correction_that_waits(ui, tagged):
    job_id = tagged("B7")
    ui.get("/sample/jobs")

    ui.post(f"/sample/{job_id}", json={"label": "coyote"})

    assert ui.get("/sample/jobs").json()["note"] == (
        "Sample of week 2026-W46: 1 sure tag picked, 1 of them checked, 1 of those changed.")
    assert ui.get("/corrections/jobs").json()["jobs"] == [
        {"job_id": job_id, "site_code": "B7", "old_label": "bobcat", "new_label": "coyote",
         "model_version": "wildlife-9"}]
    assert ui.get(f"/result/{job_id}").json()["label"] == "coyote"


def test_person_on_the_sample_page_deletes_the_photo(ui, tagged, photos):
    job_id = tagged()
    ui.get("/sample/jobs")

    r = ui.post(f"/sample/{job_id}", json={"label": "person"})

    assert r.json()["note"] == PERSON_NOTE
    assert os.listdir(photos) == []


def test_sample_page_refuses_in_plain_words(ui, tagged):
    picked = tagged()
    ui.get("/sample/jobs")
    not_picked = tagged()

    r = ui.post(f"/sample/{not_picked}", json={"label": "coyote"})
    assert r.status_code == 409 and "already checked" in r.json()["detail"]
    r = ui.post(f"/sample/{picked}", json={"label": "unicorn"})
    assert r.status_code == 422 and "pick a label" in r.json()["detail"]
    r = ui.post(f"/sample/{uuid.uuid4()}", json={"label": "coyote"})
    assert r.status_code == 404 and "find this job" in r.json()["detail"]
    assert ui.get("/sample/jobs").json()["note"] == NOTHING_CHECKED


@pytest.mark.parametrize("address", ["claim%3F", "nope"])
def test_address_that_is_no_job_number_never_reaches_the_middle_server(ui, tagged, address):
    tagged()
    ui.get("/sample/jobs")

    assert ui.post(f"/sample/{address}", json={"label": "coyote"}).status_code == 422
    assert ui.get(f"/sample/{address}/photo").status_code == 422
    assert ui.get("/sample/jobs").json()["note"] == NOTHING_CHECKED


@pytest.mark.parametrize("address", ["/numbers", "/sample"])
def test_page_loads_with_a_link_back_and_says_who_it_is_for(ui, address):
    r = ui.get(address)
    assert r.status_code == 200
    assert 'href="/"' in r.text
    assert "There is no login yet, so everybody can use it." in r.text
    for outside in ("http://", "https://", "<link", "<script src"):
        assert outside not in r.text


def test_upload_page_links_to_the_numbers_and_to_the_sample(ui):
    page = ui.get("/").text
    assert 'href="/numbers"' in page and 'href="/sample"' in page


def test_middle_server_down_gets_a_friendly_message():
    down = httpx.Client(base_url="http://127.0.0.1:9", timeout=1)
    ui = TestClient(create_website(down))

    for r in (ui.get("/numbers/rows"), ui.get("/sample/jobs"), ui.get(f"/sample/{uuid.uuid4()}/photo"),
              ui.post(f"/sample/{uuid.uuid4()}", json={"label": "coyote"})):
        assert r.status_code == 503 and "try again" in r.json()["detail"]


def test_tag_from_the_sample_of_an_older_week_gets_its_own_sentence(ui, tagged, monkeypatch):
    job_id = tagged()
    ui.get("/sample/jobs")
    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W47")

    r = ui.post(f"/sample/{job_id}", json={"label": "coyote"})

    assert r.status_code == 409
    assert r.json()["detail"] == ("This tag does not wait in the sample of this week. Somebody already checked it, "
                                  "or it belongs to the sample of another week.")
    assert ui.get(f"/result/{job_id}").json()["label"] == "bobcat"
