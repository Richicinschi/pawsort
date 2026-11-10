from PIL import Image

from ai_server import detector
from ai_server.logic import decide
from ai_server.worker import process_one


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = boxes

    def generate_detections_one_image(self, image, name, detection_threshold):
        return {"detections": [box for box in self.boxes if box["conf"] >= detection_threshold]}


def box(conf):
    return {"category": "1", "conf": conf, "bbox": [0, 0, 1, 1]}


def run_job(orch, tmp_path, boxes):
    path = tmp_path / "trail.jpg"
    Image.new("RGB", (200, 200)).save(path)
    job_id = orch.post("/jobs", json={"image_path": str(path), "site_code": "A12"}).json()["id"]
    finder = FakeDetector(boxes)
    process_one(orch, lambda image: decide(detector.find(finder, image), lambda found: {"bobcat": 0.95}))
    return job_id


def on_the_list(orch):
    return [job["id"] for job in orch.get("/review").json()["jobs"]]


def test_weak_box_ends_on_the_review_list(orch_client, tmp_path):
    job_id = run_job(orch_client, tmp_path, [box(0.1)])
    assert orch_client.get(f"/jobs/{job_id}").json()["label"] == "empty"
    assert on_the_list(orch_client) == [job_id]


def test_box_under_the_weak_limit_counts_as_no_box(orch_client, tmp_path):
    job_id = run_job(orch_client, tmp_path, [box(0.01)])
    assert orch_client.get(f"/jobs/{job_id}").json()["probability"] == 1.0
    assert on_the_list(orch_client) == []


def test_strong_box_goes_to_the_model_and_stays_off_the_list(orch_client, tmp_path):
    job_id = run_job(orch_client, tmp_path, [box(0.1), box(0.9)])
    assert orch_client.get(f"/jobs/{job_id}").json()["label"] == "bobcat"
    assert on_the_list(orch_client) == []
