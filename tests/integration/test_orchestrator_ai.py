import os

from ai_server.worker import process_one


def test_queued_job_is_classified_and_the_working_copy_deleted(orch_client, send_photo, work):
    job_id = send_photo(orch_client)

    handled = process_one(orch_client, lambda image: {"bobcat": 0.9, "coyote": 0.1}, work)  # stand-in model

    assert handled is True
    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["status"], job["label"], job["probability"], job["needs_review"]) == ("done", "bobcat", 0.9, 0)
    assert os.listdir(work) == []                                          # FR-09
    assert process_one(orch_client, lambda image: {"bobcat": 0.9}, work) is False     # queue empty


def test_unreadable_photo_fails_without_blocking(orch_client, send_photo, work):
    bad = send_photo(orch_client, "not_an_image.jpg")
    good = send_photo(orch_client, "coyote_1.jpg")
    process_one(orch_client, lambda image: {"coyote": 0.9, "bobcat": 0.1}, work)
    process_one(orch_client, lambda image: {"coyote": 0.9, "bobcat": 0.1}, work)
    assert orch_client.get(f"/jobs/{bad}").json()["status"] == "failed"     # FR-21
    assert orch_client.get(f"/jobs/{good}").json()["label"] == "coyote"     # NFR-09
    assert os.listdir(work) == []
