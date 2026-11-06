from fastapi.testclient import TestClient

from orchestrator.app import create_app


def make_client(tmp_path):
    return TestClient(create_app(str(tmp_path / "jobs.db")))


def test_create_claim_and_finish(tmp_path):
    c = make_client(tmp_path)
    job_id = c.post("/jobs", json={"image_path": "/x/cat.jpg"}).json()["id"]
    assert c.get(f"/jobs/{job_id}").json()["status"] == "new"
    assert c.post("/jobs/claim").json()["id"] == job_id
    assert c.post("/jobs/claim").status_code == 204
    r = c.post(f"/jobs/{job_id}/result", json={"label": "cat", "probability": 0.9, "needs_review": False})
    assert r.status_code == 200 and r.json()["status"] == "done"


def test_errors(tmp_path):
    c = make_client(tmp_path)
    assert c.get("/jobs/nope").status_code == 404
    job_id = c.post("/jobs", json={"image_path": "/x/dog.jpg"}).json()["id"]
    r = c.post(f"/jobs/{job_id}/result", json={"label": "dog", "probability": 0.9, "needs_review": False})
    assert r.status_code == 409  # still "new": cannot jump to done
    c.post("/jobs/claim")
    assert c.post(f"/jobs/{job_id}/result", json={"label": "horse", "probability": 0.9,
                                                  "needs_review": False}).status_code == 422
