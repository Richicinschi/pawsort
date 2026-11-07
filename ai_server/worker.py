"""AI server: takes jobs from the orchestrator, classifies the photo, reports back, deletes the photo."""
import os
import time
from typing import Callable

import httpx
from PIL import Image

from ai_server.logic import decide, format_result

Predict = Callable[[Image.Image], dict[str, float]]


def process_one(client: httpx.Client, predict: Predict) -> bool:
    """Handle one job if there is one. Returns False when the queue is empty."""
    r = client.post("/jobs/claim")
    if r.status_code == 204:
        return False
    r.raise_for_status()
    job = r.json()
    try:
        with Image.open(job["image_path"]) as image:
            body = format_result(predict(image))
    except Exception as exc:  # any unreadable or broken photo becomes a failed job, never a crash
        body = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            os.remove(job["image_path"])
        except OSError:
            pass
    client.post(f"/jobs/{job['id']}/result", json=body).raise_for_status()
    return True


def poll(client: httpx.Client, predict: Predict, wait: Callable[[float], None] = time.sleep) -> None:
    """One turn of the worker loop. Trouble with the middle server only makes the worker wait."""
    try:
        if not process_one(client, predict):
            wait(0.5)
    except httpx.TransportError:
        wait(1.0)  # orchestrator not up yet or restarting
    except httpx.HTTPStatusError as exc:
        print(f"middle server answered {exc.response.status_code}, trying again in a moment")
        wait(1.0)


def main() -> None:
    from ai_server import detector
    from ai_server.model import crop_to_box, load_model, predict_scores, preprocess

    finder = detector.load()
    model, labels = load_model(os.environ.get("PAWSORT_MODEL", "models/pawsort_wildlife.pt"), "models/labels.json")

    def predict(image: Image.Image) -> dict[str, float]:
        def classify(box: list[float]) -> dict[str, float]:
            return predict_scores(model, labels, preprocess(crop_to_box(image, box)))
        return decide(detector.find(finder, image), classify)

    base = os.environ.get("PAWSORT_ORCH", "http://127.0.0.1:8001")
    with httpx.Client(base_url=base, timeout=10) as client:
        print(f"AI server polling {base}")
        while True:
            poll(client, predict)


if __name__ == "__main__":
    main()
