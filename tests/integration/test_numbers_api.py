import json
import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from orchestrator import numbers
from orchestrator.app import create_app


@pytest.fixture
def tagged(orch_client, send_photo, tmp_path):
    """A done job on the middle server, uploaded on the given day."""
    def _tagged(site="A12", day="2026-11-10", label="bobcat", probability=0.9, orch=orch_client):
        job_id = send_photo(orch, site_code=site)
        orch.post("/jobs/claim")
        orch.post(f"/jobs/{job_id}/result", json={"label": label, "probability": probability,
                                                  "needs_review": probability < 0.8, "model_version": "wildlife-9"})
        conn = sqlite3.connect(str(tmp_path / "jobs.db"))
        conn.execute("UPDATE jobs SET created_at = ? WHERE id = ?", (f"{day}T12:00:00+00:00", job_id))
        conn.commit()
        conn.close()
        return job_id
    return _tagged


def test_numbers_per_camera_site_and_week(orch_client, tagged):
    for probability in (0.9, 0.9, 0.9, 0.6):
        tagged(probability=probability)
    checked = tagged("B7", "2026-11-17", probability=0.6)
    orch_client.post(f"/jobs/{checked}/review", json={"label": "coyote"})

    assert orch_client.get("/numbers").json()["rows"] == [
        {"site_code": "B7", "week": "2026-W47", "tags": 1, "needs_review": 1.0, "checked": 1, "warning": True,
         "corrected": 1.0},
        {"site_code": "A12", "week": "2026-W46", "tags": 4, "needs_review": 0.25, "checked": 0, "warning": False,
         "corrected": None}]


def test_answer_says_which_share_makes_a_warning(orch_client):
    assert orch_client.get("/numbers").json()["warn_share"] == 0.40


def test_numbers_count_the_approved_corrections_that_wait_for_a_training(orch_client, tagged):
    job_id = tagged(probability=0.6)
    orch_client.post(f"/jobs/{job_id}/review", json={"label": "coyote"})
    orch_client.post(f"/corrections/{job_id}", json={"decision": "approved"})

    answer = orch_client.get("/numbers").json()

    assert (answer["for_training"], answer["plan_training"]) == (1, False)


def versions_file(tmp_path, trained_on):
    path = tmp_path / "versions.json"
    path.write_text(json.dumps({"current": "v2", "previous": "v1", "versions": {
        "v1": {"trained_on": "2020-01-01"}, "v2": {"trained_on": trained_on}}}))
    return str(path)


def test_site_whose_first_photo_came_after_the_training_of_the_current_model_is_new(tmp_path, tagged):
    orch = TestClient(create_app(str(tmp_path / "jobs.db"), str(tmp_path / "photos"),
                                 versions_path=versions_file(tmp_path, "2026-11-10")))
    tagged("OLD1", "2026-11-09", orch=orch)
    tagged("OLD1", "2026-11-12", orch=orch)
    tagged("NEW1", "2026-11-11", orch=orch)

    answer = orch.get("/numbers").json()

    assert (answer["trained_on"], answer["new_sites"]) == ("2026-11-10", ["NEW1"])


@pytest.mark.parametrize("content", [None, "not json", json.dumps({"current": "v9", "versions": {}}),
                                     json.dumps({"current": "v2", "versions": {"v2": {"trained_on": None}}})])
def test_without_a_training_date_no_site_is_new_and_the_answer_says_the_date_is_unknown(tmp_path, tagged, content):
    path = tmp_path / "versions.json"
    if content is not None:
        path.write_text(content)
    orch = TestClient(create_app(str(tmp_path / "jobs.db"), str(tmp_path / "photos"), versions_path=str(path)))
    tagged("NEW1", "2026-11-11", orch=orch)

    answer = orch.get("/numbers").json()

    assert (answer["trained_on"], answer["new_sites"]) == (None, [])


def test_middle_server_reads_the_versions_file_in_the_models_folder_when_nobody_names_another(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "versions.json").write_text(json.dumps(
        {"current": "v2", "versions": {"v2": {"trained_on": "2026-11-06"}}}))

    orch = TestClient(create_app(str(tmp_path / "jobs.db"), str(tmp_path / "photos")))

    assert orch.get("/numbers").json()["trained_on"] == "2026-11-06"


@pytest.fixture
def week_46(monkeypatch):
    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W46")


def sample_ids(orch):
    return sorted(job["id"] for job in orch.get("/sample").json()["jobs"])


def test_sample_has_the_sure_tags_with_what_a_person_needs(orch_client, tagged, week_46):
    sure = tagged("B7", label="mountain_lion", probability=0.93)
    tagged(probability=0.6)
    tagged(label="person", probability=1.0)

    answer = orch_client.get("/sample").json()

    assert answer["jobs"] == [{"id": sure, "site_code": "B7", "label": "mountain_lion", "probability": 0.93,
                               "model_version": "wildlife-9"}]
    assert (answer["week"], answer["picked"], answer["checked"], answer["changed"]) == ("2026-W46", 1, 0, 0)
    assert "person" in answer["labels"] and "bobcat" in answer["labels"]


def test_sample_stays_the_same_for_the_week_and_changes_with_the_next_one(orch_client, tagged, monkeypatch, week_46):
    first = [tagged() for _ in range(3)]
    assert sample_ids(orch_client) == sorted(first)

    later = tagged()
    assert sample_ids(orch_client) == sorted(first)

    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W47")
    assert sample_ids(orch_client) == [later]


def test_sample_photo_goes_out_only_for_a_tag_of_this_weeks_sample_that_nobody_checked(orch_client, tagged, monkeypatch,
                                                                                         week_46):
    picked = tagged()
    orch_client.get("/sample")
    not_picked = tagged()

    r = orch_client.get(f"/sample/{picked}/photo")
    assert (r.status_code, r.headers["content-type"]) == (200, "image/jpeg")
    assert orch_client.get(f"/sample/{not_picked}/photo").status_code == 409
    assert orch_client.get("/sample/nope/photo").status_code == 404
    assert orch_client.get(f"/review/{picked}/photo").status_code == 409

    monkeypatch.setattr(numbers, "this_week", lambda: "2026-W47")
    assert orch_client.get(f"/sample/{picked}/photo").status_code == 409


def test_confirmed_tag_from_the_sample_is_stored_and_leaves_the_sample(orch_client, tagged, week_46):
    job_id, other = tagged(), tagged()
    orch_client.get("/sample")

    r = orch_client.post(f"/sample/{job_id}", json={"label": "bobcat"})

    assert r.status_code == 200
    review = r.json()["review"]
    assert (review["old_label"], review["new_label"], review["source"]) == ("bobcat", "bobcat", "sample")
    answer = orch_client.get("/sample").json()
    assert ([job["id"] for job in answer["jobs"]], answer["picked"], answer["checked"], answer["changed"]) == (
        [other], 2, 1, 0)
    assert orch_client.get(f"/sample/{job_id}/photo").status_code == 409


def test_changed_tag_from_the_sample_goes_to_the_corrections_like_any_other(orch_client, tagged, week_46):
    job_id = tagged()
    orch_client.get("/sample")

    orch_client.post(f"/sample/{job_id}", json={"label": "coyote"})

    assert [job["id"] for job in orch_client.get("/corrections").json()["jobs"]] == [job_id]
    assert orch_client.get("/sample").json()["changed"] == 1
    assert orch_client.get(f"/jobs/{job_id}").json()["label"] == "coyote"
    assert orch_client.get(f"/corrections/{job_id}/photo").status_code == 200


def test_person_picked_in_the_sample_deletes_the_photo(orch_client, tagged, photos, week_46):
    job_id = tagged()
    orch_client.get("/sample")

    orch_client.post(f"/sample/{job_id}", json={"label": "person"})

    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["label"], job["photo_name"], os.listdir(photos)) == ("person", "", [])
    assert orch_client.get("/corrections").json()["jobs"] == []


def test_answers_that_the_sample_refuses(orch_client, tagged, week_46):
    picked = tagged()
    orch_client.get("/sample")
    not_picked, on_the_list = tagged(), tagged(probability=0.6)

    assert orch_client.post(f"/sample/{picked}", json={"label": "unicorn"}).status_code == 422
    assert orch_client.post(f"/sample/{not_picked}", json={"label": "coyote"}).status_code == 409
    assert orch_client.post(f"/sample/{on_the_list}", json={"label": "coyote"}).status_code == 409
    assert orch_client.post("/sample/nope", json={"label": "coyote"}).status_code == 404
    orch_client.post(f"/sample/{picked}", json={"label": "bobcat"})
    assert orch_client.post(f"/sample/{picked}", json={"label": "coyote"}).status_code == 409
    assert orch_client.get(f"/jobs/{picked}").json()["label"] == "bobcat"


def test_tag_of_the_sample_cannot_be_answered_through_the_review_list(orch_client, tagged, week_46):
    job_id = tagged()
    orch_client.get("/sample")

    assert orch_client.post(f"/jobs/{job_id}/review", json={"label": "coyote"}).status_code == 409
    assert orch_client.get(f"/jobs/{job_id}").json()["review"] is None
