import os
import threading
from collections import Counter

from fastapi.testclient import TestClient

from ai_server.worker import process_one
from orchestrator.app import create_app

JOBS = 40


def stand_in(image):
    return {"bobcat": 0.9, "coyote": 0.1}


def test_two_workers_at_once_do_every_job_exactly_once(tmp_path, send_photo):
    app = create_app(str(tmp_path / "jobs.db"), str(tmp_path / "photos"))
    orch = TestClient(app)
    jobs = [send_photo(orch) for _ in range(JOBS)]
    start = threading.Barrier(2)
    handled, errors = Counter(), []

    def work(name):
        client = TestClient(app)
        folder = tmp_path / name
        folder.mkdir()
        start.wait()
        try:
            while process_one(client, stand_in, str(folder), name):
                handled[name] += 1
        except Exception as exc:
            errors.append(exc)

    workers = [threading.Thread(target=work, args=(name,)) for name in ("worker-1", "worker-2")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)

    rows = [orch.get(f"/jobs/{job_id}").json() for job_id in jobs]
    assert errors == []
    assert [row["status"] for row in rows] == ["done"] * JOBS
    assert handled["worker-1"] + handled["worker-2"] == JOBS  # as many claims as jobs, so no job went out twice
    assert Counter(row["model_version"] for row in rows) == handled
    assert handled["worker-1"] and handled["worker-2"]
    assert os.listdir(tmp_path / "worker-1") == os.listdir(tmp_path / "worker-2") == []
    assert len(os.listdir(tmp_path / "photos")) == JOBS
