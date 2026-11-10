import pytest


@pytest.fixture
def claimed_job(orch_client, send_photo):
    job_id = send_photo(orch_client)
    orch_client.post("/jobs/claim")
    return job_id


def report(orch, job_id, probability, **second):
    return orch.post(f"/jobs/{job_id}/result",
                     json={"label": "cat", "probability": probability, "needs_review": False, **second})


@pytest.mark.parametrize("confidence", [5.0, 1.0001, -0.1])
def test_confidence_outside_0_to_1_is_refused(orch_client, claimed_job, confidence):
    r = report(orch_client, claimed_job, confidence)

    assert r.status_code == 422
    job = orch_client.get(f"/jobs/{claimed_job}").json()
    assert (job["status"], job["probability"]) == ("processing", None)


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_confidence_from_0_to_1_is_accepted(orch_client, claimed_job, confidence):
    assert report(orch_client, claimed_job, confidence).status_code == 200
    assert orch_client.get(f"/jobs/{claimed_job}").json()["probability"] == confidence


@pytest.mark.parametrize("confidence", [5.0, -0.1])
def test_score_of_the_second_guess_outside_0_to_1_is_refused(orch_client, claimed_job, confidence):
    r = report(orch_client, claimed_job, 0.6, second_label="dog", second_probability=confidence)

    assert r.status_code == 422
    assert orch_client.get(f"/jobs/{claimed_job}").json()["status"] == "processing"


def test_refused_job_can_still_get_its_real_answer(orch_client, claimed_job):
    report(orch_client, claimed_job, 5.0)

    assert report(orch_client, claimed_job, 0.9).status_code == 200
    assert orch_client.get(f"/jobs/{claimed_job}").json()["status"] == "done"
