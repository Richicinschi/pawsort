"""AI server: takes jobs from the orchestrator, classifies a copy of the photo, deletes the copy, reports back."""
import os
import time
import uuid
from typing import Callable

import httpx
from PIL import Image

from ai_server.logic import decide, format_result

Predict = Callable[[Image.Image], dict[str, float]]
COPY = ".copy"  # every working copy ends like this, and the start deletes nothing else


def delete_copies(folder: str) -> None:
    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(folder):
        if name.endswith(COPY):
            os.remove(os.path.join(folder, name))


def process_one(client: httpx.Client, predict: Predict, work_dir: str, model_version: str | None = None) -> bool:
    """Handle one job if there is one. Returns False when the queue is empty."""
    r = client.post("/jobs/claim")
    if r.status_code == 204:
        return False
    r.raise_for_status()
    job = r.json()
    copy = os.path.join(work_dir, f"{uuid.uuid4()}{COPY}")  # our own name, so no job can point us at another file
    try:
        photo = client.get(f"/jobs/{job['id']}/photo")
        photo.raise_for_status()
        with open(copy, "wb") as f:
            f.write(photo.content)
        with Image.open(copy) as image:
            body = format_result(predict(image))
    except Exception as exc:  # any unreadable or broken photo becomes a failed job, never a crash
        body = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            os.remove(copy)
        except OSError:
            pass
    body["model_version"] = model_version
    client.post(f"/jobs/{job['id']}/result", json=body).raise_for_status()
    return True


def poll(client: httpx.Client, predict: Predict, work_dir: str, model_version: str | None = None,
         wait: Callable[[float], None] = time.sleep) -> None:
    """One turn of the worker loop. Trouble with the middle server only makes the worker wait."""
    try:
        if not process_one(client, predict, work_dir, model_version):
            wait(0.5)
    except httpx.TransportError:
        wait(1.0)  # orchestrator not up yet or restarting
    except httpx.HTTPStatusError as exc:
        print(f"middle server answered {exc.response.status_code}, trying again in a moment")
        wait(1.0)


def main() -> None:
    from ai_server import detector
    from ai_server.model import crop_to_box, load_model, predict_scores, preprocess, read_version

    weights = os.environ.get("PAWSORT_MODEL", "models/pawsort_wildlife.pt")
    model_version = read_version(weights)
    work_dir = os.environ.get("PAWSORT_WORK", "data/work")
    delete_copies(work_dir)  # the ones a crash left behind: FR-09
    finder = detector.load()
    model, labels = load_model(weights, "models/labels.json")

    def predict(image: Image.Image) -> dict[str, float]:
        def classify(box: list[float]) -> dict[str, float]:
            return predict_scores(model, labels, preprocess(crop_to_box(image, box)))
        return decide(detector.find(finder, image), classify)

    base = os.environ.get("PAWSORT_ORCH", "http://127.0.0.1:8001")
    with httpx.Client(base_url=base, timeout=10) as client:
        print(f"AI server polling {base} with model {model_version}")
        while True:
            poll(client, predict, work_dir, model_version)


if __name__ == "__main__":
    main()
