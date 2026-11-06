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

