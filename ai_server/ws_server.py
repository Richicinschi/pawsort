"""WebSocket mode: the orchestrator pushes job doorbells, we claim/fetch/report over HTTP."""
import json
import os
import random
import time
import uuid
from typing import Callable

import httpx
from PIL import Image

from ai_server.logic import format_result

COPY = ".copy"

# Reconnect backoff bounds (seconds).
BACKOFF_MIN = 0.5
BACKOFF_MAX = 30.0


def _backoff(attempt: int) -> float:
    base = min(BACKOFF_MAX, BACKOFF_MIN * (2 ** attempt))
    return base * (0.5 + random.random())  # jitter, so N workers don't stampede


def _handle_job(
    client: httpx.Client,
    predict: Callable[[Image.Image], dict[str, float]],
    work_dir: str,
    model_version: str | None,
    job_id: str,
    cancelled: Callable[[], bool],
) -> None:
    """Claim, fetch, classify, report — the WS equivalent of one process_one turn."""
    claim = client.post(f"/jobs/{job_id}/claim")
    if claim.status_code == 409:      # someone else got it first; nothing to do
        return
    if claim.status_code == 204:
        return
    claim.raise_for_status()

    copy = os.path.join(work_dir, f"{uuid.uuid4()}{COPY}")
    try:
        photo = client.get(f"/jobs/{job_id}/photo")
        photo.raise_for_status()
        with open(copy, "wb") as f:
            f.write(photo.content)
        with Image.open(copy) as image:
            body = format_result(predict(image))
    except Exception as exc:          # unreadable photo -> failed job, not a crash
        body = {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            os.remove(copy)
        except OSError:
            pass

    if cancelled():
        return                        # orchestrator told us to drop it
    body["model_version"] = model_version
    client.post(f"/jobs/{job_id}/result", json=body).raise_for_status()


def run_ws(
    predict: Callable[[Image.Image], dict[str, float]],
    work_dir: str,
    model_version: str | None,
    url: str,
    stop: Callable[[], bool] = lambda: False,
) -> None:
    """Main WS loop: connect, consume doorbells, reconnect forever with backoff."""
    from websockets.sync.client import connect  # pip install websockets>=12

    attempt = 0
    with httpx.Client(base_url=_http_base(url), timeout=10) as client:
        while not stop():
            try:
                with connect(url, open_timeout=10) as ws:
                    attempt = 0            # healthy connection resets backoff
                    cancelled: set[str] = set()
                    for raw in ws:
                        msg = json.loads(raw)
                        kind = msg.get("type")
                        if kind == "job":
                            _handle_job(client, predict, work_dir, model_version,
                                        msg["id"], lambda jid=msg["id"]: jid in cancelled)
                            cancelled.discard(msg["id"])
                        elif kind == "cancel":
                            cancelled.add(msg["id"])
                        elif kind == "ping":
                            ws.send(json.dumps({"type": "pong"}))
                        # unknown types are ignored forward-compatibly
            except Exception as exc:       # transport / protocol / JSON: back off and retry
                print(f"ws loop: {type(exc).__name__}: {exc}; reconnecting")
                time.sleep(_backoff(attempt))
                attempt += 1


def _http_base(ws_url: str) -> str:
    """Derive the HTTP base for photo/result calls from the WS URL."""
    base = ws_url.replace("wss://", "https://", 1).replace("ws://", "http://", 1)
    # Strip the WS path; the orchestrator serves HTTP at the same host/port.
    return base.rsplit("/", 1)[0] if "/" in base.split("://", 1)[1] else base
