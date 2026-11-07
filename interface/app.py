"""Interface server: talks to the user and to the orchestrator. Never to the AI server."""
import io
import os
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from interface.validation import MAX_BYTES, UploadError, check_site_code, validate_upload

PAGE = Path(__file__).parent / "static" / "index.html"
PERSON_NOTE = "This photo shows a person, so we deleted it."


def show_job(job: dict) -> dict:
    out = {"status": job["status"]}
    if job["status"] == "done":
        out.update(label=job["label"].replace("_", " "), percent=int(round(job["probability"] * 100, 2)),
                   needs_review=bool(job["needs_review"]))
        if job["label"] == "person":
            out["note"] = PERSON_NOTE
    elif job["status"] == "failed":
        out["error"] = "We could not check this photo. Please try another one."
    return out


def create_app(orch: httpx.Client, upload_dir: str) -> FastAPI:
    os.makedirs(upload_dir, exist_ok=True)
    app = FastAPI(title="PawSort")

    def save_photo(data: bytes, site_code: str) -> str:
        fmt = validate_upload(data)
        name = f"{uuid.uuid4()}.{'jpg' if fmt == 'JPEG' else 'png'}"  # random name, never the user's
        path = os.path.abspath(os.path.join(upload_dir, name))
        Image.open(io.BytesIO(data)).save(path, format=fmt, quality="keep", comment=b"")  # drops all metadata: FR-04
        try:
            r = orch.post("/jobs", json={"image_path": path, "site_code": site_code})
            r.raise_for_status()
        except httpx.HTTPError:
            os.remove(path)
            raise UploadError("We could not save this photo right now. Please try again.")
        return r.json()["id"]

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE.read_text(encoding="utf-8")

    @app.post("/upload")
    def upload(site_code: str = Form(""), files: list[UploadFile] = File(...)):
        try:
            check_site_code(site_code)
        except UploadError as exc:
            raise HTTPException(400, str(exc))
        photos = []
        for file in files:
            try:
                photos.append({"file": file.filename, "job_id": save_photo(file.file.read(MAX_BYTES + 1), site_code)})
            except UploadError as exc:
                photos.append({"file": file.filename, "error": str(exc)})
        return {"photos": photos}

    @app.get("/result/{job_id}")
    def result(job_id: str):
        r = orch.get(f"/jobs/{job_id}")
        if r.status_code == 404:
            raise HTTPException(404, "Unknown job.")
        r.raise_for_status()
        return show_job(r.json())

    return app


def build_app() -> FastAPI:
    orch = httpx.Client(base_url=os.environ.get("PAWSORT_ORCH", "http://127.0.0.1:8001"), timeout=10)
    return create_app(orch, os.environ.get("PAWSORT_UPLOADS", "data/uploads"))
