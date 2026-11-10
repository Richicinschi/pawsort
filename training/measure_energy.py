"""Measure the energy of the AI part and of the rest with CodeCarbon, for the Dutch grid.

CodeCarbon estimates the whole PC, so these numbers are an upper bound.
Run from the main folder: python training/measure_energy.py --photos tests/data --count 30 --idle 60
"""
import argparse
import os
import random
import shutil
import sys
import tempfile
import time

from codecarbon import OfflineEmissionsTracker
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_server import detector
from ai_server.logic import decide, format_result
from ai_server.model import crop_to_box, load_model, predict_scores, preprocess
from ai_server.worker import process_one
from orchestrator.app import create_app

SEED = 9


def measure(name, work):
    tracker = OfflineEmissionsTracker(country_iso_code="NLD", project_name=name, log_level="error",
                                      save_to_file=False, tracking_mode="machine")
    started = time.time()
    tracker.start()
    try:
        work()
    finally:
        grams = tracker.stop() * 1000
    wh = tracker.final_emissions_data.energy_consumed * 1000
    print(f"{name}: {time.time() - started:.0f} s, {wh:.2f} Wh, {grams:.2f} g CO2")
    return grams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", default="tests/data")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--idle", type=int, default=60)
    args = parser.parse_args()

    finder = detector.load()
    model, labels = load_model("models/pawsort_wildlife.pt", "models/labels.json")

    def predict(image: Image.Image) -> dict[str, float]:
        def classify(box: list[float]) -> dict[str, float]:
            return predict_scores(model, labels, preprocess(crop_to_box(image, box)))
        return decide(detector.find(finder, image), classify)

    paths = []
    for folder, _, files in os.walk(args.photos):
        paths += [os.path.join(folder, name) for name in files if name.endswith(".jpg")]
    paths = [path for path in paths if "not_an_image" not in path]
    random.seed(SEED)
    work = tempfile.mkdtemp()
    copies = os.path.join(work, "copies")
    os.makedirs(copies)

    def send(orch, path):
        with open(path, "rb") as f:
            return orch.post("/jobs", files={"photo": ("photo", f.read(), "image/jpeg")}).json()["id"]

    client = TestClient(create_app(os.path.join(work, "jobs.db"), os.path.join(work, "photos")))
    for path in random.choices(paths, k=args.count):
        send(client, path)

    def ai_part():
        while process_one(client, predict, copies):
            pass

    def rest():
        fresh = TestClient(create_app(os.path.join(work, "rest.db"), os.path.join(work, "rest_photos")))
        for _ in range(1000):
            job_id = send(fresh, paths[0])
            fresh.post("/jobs/claim")
            fresh.post(f"/jobs/{job_id}/result", json=format_result({"bobcat": 0.9, "coyote": 0.1}))
            fresh.get(f"/jobs/{job_id}")

    grams = measure(f"{args.count} tags", ai_part)
    print(f"so 1,000 tags: {grams / args.count * 1000:.1f} g CO2")
    measure("1,000 jobs without the AI", rest)
    measure(f"PC waiting for {args.idle} s", lambda: time.sleep(args.idle))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
