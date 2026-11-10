import os
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from orchestrator.app import create_app

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@pytest.fixture
def corrected(orch_client, send_photo):
    """A job on the middle server whose tag a person changed, from bobcat to the given label."""
    def _corrected(new_label="coyote", name="bobcat_1.jpg"):
        job_id = send_photo(orch_client, name, "A12")
        orch_client.post("/jobs/claim")
        orch_client.post(f"/jobs/{job_id}/result", json={"label": "bobcat", "probability": 0.6, "needs_review": True,
                                                         "model_version": "wildlife-9"})
        orch_client.post(f"/jobs/{job_id}/review", json={"label": new_label})
        return job_id
    return _corrected


def decide(orch, job_id, decision):
    return orch.post(f"/corrections/{job_id}", json={"decision": decision})


def waiting(orch):
    return [job["id"] for job in orch.get("/corrections").json()["jobs"]]


def for_training(orch):
    return [job["id"] for job in orch.get("/training/corrections").json()["jobs"]]


def photo_file(name):
    with open(os.path.join(DATA, name), "rb") as f:
        return f.read()


def test_list_has_every_correction_that_waits_with_both_labels_and_the_version(orch_client, corrected):
    job_id = corrected()
    corrected("bobcat")
    corrected("person")

    assert orch_client.get("/corrections").json()["jobs"] == [
        {"id": job_id, "site_code": "A12", "old_label": "bobcat", "new_label": "coyote", "model_version": "wildlife-9"}]


def test_approve_takes_the_correction_off_the_list_and_keeps_the_new_label(orch_client, corrected):
    job_id, other = corrected(), corrected("deer")

    r = decide(orch_client, job_id, "approved")

    assert r.status_code == 200
    assert waiting(orch_client) == [other]
    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["label"], job["review"]["decision"]) == ("coyote", "approved")


def test_reject_puts_the_old_label_back_on_the_job(orch_client, corrected):
    job_id = corrected()

    r = decide(orch_client, job_id, "rejected")

    assert r.status_code == 200
    assert waiting(orch_client) == []
    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["label"], job["review"]["decision"], job["review"]["new_label"]) == ("bobcat", "rejected", "coyote")


def test_second_decision_gets_a_409_and_the_first_one_stays(orch_client, corrected):
    job_id = corrected()
    decide(orch_client, job_id, "approved")

    assert decide(orch_client, job_id, "rejected").status_code == 409
    assert orch_client.get(f"/jobs/{job_id}").json()["label"] == "coyote"


def test_confirmation_and_change_to_person_cannot_get_a_decision(orch_client, corrected):
    for job_id in (corrected("bobcat"), corrected("person")):
        assert decide(orch_client, job_id, "approved").status_code == 409


def test_unknown_job_and_unknown_decision(orch_client, corrected):
    job_id = corrected()
    assert decide(orch_client, "nope", "approved").status_code == 404
    assert decide(orch_client, job_id, "maybe").status_code == 422
    assert waiting(orch_client) == [job_id]


def test_photo_of_a_correction_goes_out_only_while_it_waits(orch_client, corrected, send_photo):
    job_id = corrected()
    r = orch_client.get(f"/corrections/{job_id}/photo")
    assert (r.status_code, r.headers["content-type"], r.content) == (200, "image/jpeg", photo_file("bobcat_1.jpg"))

    decide(orch_client, job_id, "approved")
    assert orch_client.get(f"/corrections/{job_id}/photo").status_code == 409

    assert orch_client.get(f"/corrections/{corrected('bobcat')}/photo").status_code == 409
    assert orch_client.get(f"/corrections/{send_photo(orch_client)}/photo").status_code == 409
    assert orch_client.get("/corrections/nope/photo").status_code == 404


def test_rejected_correction_gives_out_no_photo_anywhere(orch_client, corrected):
    job_id = corrected()
    decide(orch_client, job_id, "rejected")

    for address in ("corrections", "training/corrections", "review"):
        assert orch_client.get(f"/{address}/{job_id}/photo").status_code == 409


def test_training_gets_only_the_approved_corrections_with_the_right_label(orch_client, corrected):
    approved, rejected = corrected(), corrected("deer")
    corrected("skunk")
    corrected("bobcat")
    decide(orch_client, approved, "approved")
    decide(orch_client, rejected, "rejected")

    assert orch_client.get("/training/corrections").json()["jobs"] == [
        {"id": approved, "label": "coyote", "model_version": "wildlife-9"}]


def test_training_photo_goes_out_only_for_an_approved_correction_that_is_not_taken_yet(orch_client, corrected):
    job_id, waits = corrected(name="coyote_1.jpg"), corrected("deer")
    assert orch_client.get(f"/training/corrections/{job_id}/photo").status_code == 409
    decide(orch_client, job_id, "approved")

    r = orch_client.get(f"/training/corrections/{job_id}/photo")
    assert (r.status_code, r.content) == (200, photo_file("coyote_1.jpg"))
    assert orch_client.get(f"/training/corrections/{waits}/photo").status_code == 409
    assert orch_client.get("/training/corrections/nope/photo").status_code == 404

    orch_client.post("/training/corrections/used", json={"jobs": [job_id]})
    assert orch_client.get(f"/training/corrections/{job_id}/photo").status_code == 409


def test_correction_a_training_took_is_not_given_out_again(orch_client, corrected):
    taken, later = corrected(), corrected("deer")
    decide(orch_client, taken, "approved")
    decide(orch_client, later, "approved")

    r = orch_client.post("/training/corrections/used", json={"jobs": [taken, "nope"]})

    assert r.json() == {"used": 1}
    assert for_training(orch_client) == [later]


def test_correction_that_waits_cannot_be_marked_as_taken(orch_client, corrected):
    job_id = corrected()

    assert orch_client.post("/training/corrections/used", json={"jobs": [job_id]}).json() == {"used": 0}

    decide(orch_client, job_id, "approved")
    assert for_training(orch_client) == [job_id]


def test_photo_of_an_approved_correction_stays_on_the_middle_server(orch_client, corrected, photos):
    job_id = corrected()
    decide(orch_client, job_id, "approved")
    orch_client.post("/training/corrections/used", json={"jobs": [job_id]})

    assert os.listdir(photos) == [orch_client.get(f"/jobs/{job_id}").json()["photo_name"]]


def test_list_counts_the_approved_corrections_that_wait_for_a_training(orch_client, corrected):
    first, second = corrected(), corrected("deer")
    corrected("skunk")
    assert orch_client.get("/corrections").json()["for_training"] == 0

    decide(orch_client, first, "approved")
    decide(orch_client, second, "approved")
    assert orch_client.get("/corrections").json()["for_training"] == 2

    orch_client.post("/training/corrections/used", json={"jobs": [first]})
    assert orch_client.get("/corrections").json()["for_training"] == 1


def approved_rows(path, count):
    """Writes approved corrections straight into the database, because 500 of them through the addresses take long."""
    conn = sqlite3.connect(path)
    for number in range(count):
        conn.execute("INSERT INTO jobs (id, photo_name, status, label, needs_review, created_at, updated_at) "
                     "VALUES (?, 'cat.jpg', 'done', 'coyote', 1, '2026-11-01', '2026-11-01')", (f"job-{number}",))
        conn.execute("INSERT INTO reviews (job_id, old_label, new_label, reviewed_at, decision, decided_at) "
                     "VALUES (?, 'bobcat', 'coyote', '2026-11-02', 'approved', '2026-11-03')", (f"job-{number}",))
    conn.commit()
    conn.close()


@pytest.mark.parametrize("count,plan", [(499, False), (500, True)])
def test_a_training_round_is_due_when_500_approved_corrections_wait(tmp_path, count, plan):
    path = str(tmp_path / "jobs.db")
    orch = TestClient(create_app(path, str(tmp_path / "photos")))
    approved_rows(path, count)

    answer = orch.get("/corrections").json()

    assert (answer["for_training"], answer["plan_training"]) == (count, plan)


def test_a_training_round_is_no_longer_due_once_a_training_took_the_corrections(tmp_path):
    path = str(tmp_path / "jobs.db")
    orch = TestClient(create_app(path, str(tmp_path / "photos")))
    approved_rows(path, 500)

    orch.post("/training/corrections/used", json={"jobs": ["job-0"]})

    assert orch.get("/corrections").json()["plan_training"] is False


def test_two_decisions_on_one_correction_at_the_same_moment_give_one_decision(orch_client, corrected):
    job_id = corrected()
    start = threading.Barrier(2)
    answers = {}

    def decide_now(decision):
        client = TestClient(orch_client.app)
        start.wait()
        answers[decision] = client.post(f"/corrections/{job_id}", json={"decision": decision}).status_code

    people = [threading.Thread(target=decide_now, args=(decision,)) for decision in ("approved", "rejected")]
    for person in people:
        person.start()
    for person in people:
        person.join(timeout=30)

    assert sorted(answers.values()) == [200, 409]
    winner = next(decision for decision, status in answers.items() if status == 200)
    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["review"]["decision"], job["label"]) == (winner, "coyote" if winner == "approved" else "bobcat")
