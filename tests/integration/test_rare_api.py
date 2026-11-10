import pytest


def set_rare(orch, label, rare=True):
    return orch.post("/rare", json={"label": label, "rare": rare})


def tagged(orch, send_photo, label, site="B7"):
    job_id = send_photo(orch, site_code=site)
    orch.post("/jobs/claim")
    orch.post(f"/jobs/{job_id}/result", json={"label": label, "probability": 0.9, "needs_review": False,
                                              "model_version": "wildlife-9"})
    return job_id


def test_list_starts_empty_and_names_every_animal_that_can_be_rare(orch_client):
    answer = orch_client.get("/rare").json()

    assert answer["rare"] == []
    assert len(answer["animals"]) == 18 and "mountain_lion" in answer["animals"]
    assert "person" not in answer["animals"] and "empty" not in answer["animals"]


def test_animal_can_be_added_and_removed(orch_client):
    set_rare(orch_client, "mountain_lion")
    r = set_rare(orch_client, "badger")
    assert (r.status_code, r.json()["rare"]) == (200, ["badger", "mountain_lion"])

    assert set_rare(orch_client, "badger", False).json()["rare"] == ["mountain_lion"]
    assert orch_client.get("/rare").json()["rare"] == ["mountain_lion"]


@pytest.mark.parametrize("label", ["unicorn", "person", "empty", ""])
def test_only_an_animal_we_know_can_be_rare(orch_client, label):
    assert set_rare(orch_client, label).status_code == 422
    assert orch_client.get("/rare").json()["rare"] == []


def test_tags_have_no_camera_site_for_a_rare_animal(orch_client, send_photo):
    rare, common = tagged(orch_client, send_photo, "mountain_lion"), tagged(orch_client, send_photo, "bobcat")
    set_rare(orch_client, "mountain_lion")

    tags = {tag["id"]: tag for tag in orch_client.get("/tags").json()["tags"]}

    assert (tags[rare]["site_code"], tags[common]["site_code"]) == (None, "B7")
    assert (tags[rare]["created_at"], tags[common]["created_at"] is not None) == (None, True)
    assert "B7" not in str(tags[rare])


def test_tags_have_everything_the_download_needs(orch_client, send_photo):
    job_id = tagged(orch_client, send_photo, "bobcat")
    waiting = send_photo(orch_client)

    (tag,) = orch_client.get("/tags").json()["tags"]

    assert tag == {"id": job_id, "site_code": "B7", "label": "bobcat", "probability": 0.9, "needs_review": False,
                   "checked": False, "model_version": "wildlife-9",
                   "created_at": orch_client.get(f"/jobs/{job_id}").json()["created_at"]}
    assert waiting not in str(tag)
