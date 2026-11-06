"""AI server: takes jobs from the orchestrator, classifies the photo, reports back, deletes the photo."""
import os
import time
from typing import Callable

import httpx
from PIL import Image

from ai_server.logic import format_result


def process_one(client: httpx.Client, predict: Callable[[Image.Image], float]) -> bool:
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


def main() -> None:
    from ai_server.model import load_model, predict_dog_probability, preprocess

    model = load_model(os.environ.get("PAWSORT_MODEL", "models/pawsort_head.pt"))

    def predict(image: Image.Image) -> float:
        return predict_dog_probability(model, preprocess(image))

    base = os.environ.get("PAWSORT_ORCH", "http://127.0.0.1:8001")
    with httpx.Client(base_url=base, timeout=10) as client:
        print(f"AI server polling {base}")
        while True:
            try:
                if not process_one(client, predict):
                    time.sleep(0.5)
            except httpx.TransportError:
                time.sleep(1.0)  # orchestrator not up yet or restarting


if __name__ == "__main__":
    main()
