import os

from ai_server.worker import process_one


def test_queued_job_is_classified_and_photo_deleted(orch_client, copy_photo):
    path = copy_photo("bobcat_1.jpg")
    job_id = orch_client.post("/jobs", json={"image_path": path}).json()["id"]

    handled = process_one(orch_client, predict=lambda image: {"bobcat": 0.9, "coyote": 0.1})  # stand-in model

    assert handled is True
    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["status"], job["label"], job["probability"], job["needs_review"]) == ("done", "bobcat", 0.9, 0)
    assert not os.path.exists(path)                                        # FR-09
    assert process_one(orch_client, predict=lambda image: {"bobcat": 0.9}) is False    # queue empty


def test_unreadable_photo_fails_without_blocking(orch_client, copy_photo):
    bad = orch_client.post("/jobs", json={"image_path": copy_photo("not_an_image.jpg")}).json()["id"]
    good = orch_client.post("/jobs", json={"image_path": copy_photo("coyote_1.jpg")}).json()["id"]
    process_one(orch_client, predict=lambda image: {"coyote": 0.9, "bobcat": 0.1})
    process_one(orch_client, predict=lambda image: {"coyote": 0.9, "bobcat": 0.1})
    assert orch_client.get(f"/jobs/{bad}").json()["status"] == "failed"     # FR-21
    assert orch_client.get(f"/jobs/{good}").json()["label"] == "coyote"     # NFR-09
