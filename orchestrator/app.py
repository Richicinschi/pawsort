"""Orchestrator: the single meeting point between the interface and the AI server. It also keeps the photos."""
import os
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from orchestrator import db


LABELS = {"empty", "person", "badger", "bird", "bobcat", "car", "cat", "cow", "coyote", "deer", "dog", "fox", "lizard",
          "mountain_lion", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"}
CHECK_EVERY = 1.0  # seconds between two looks for stuck jobs
ENDINGS = {"image/jpeg": "jpg", "image/png": "png"}
TRIES = 3  # for a photo that somebody still reads


class JobResult(BaseModel):
    label: str | None = None
    probability: float | None = Field(None, ge=0, le=1)
    needs_review: bool | None = None
    second_label: str | None = None
    second_probability: float | None = Field(None, ge=0, le=1)
    error: str | None = None
    model_version: str | None = None


class Review(BaseModel):
    label: str


def delete_photo(path: str, wait: float = 0.1) -> bool:
    """True when the photo is gone, False when it is still there after a few tries."""
    for _ in range(TRIES):
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return True
        except OSError as exc:
            problem = exc
            time.sleep(wait)
    print(f"could not delete a photo, the next start tries again: {problem}")
    return False


def create_app(db_path: str, photo_dir: str, stuck_seconds: float = db.STUCK_SECONDS,
               check_every: float = CHECK_EVERY) -> FastAPI:
    os.makedirs(photo_dir, exist_ok=True)

    def open_db():
        return db.connect(db_path)

    def photo_path(name: str) -> str:
        return os.path.join(photo_dir, name)

    def drop_photo(conn: sqlite3.Connection, job: dict) -> None:
        if job["photo_name"] and delete_photo(photo_path(job["photo_name"])):
            db.forget_photo(conn, job["id"])  # only now, so a photo that is still there keeps its row

    conn = open_db()  # makes or upgrades the table once, before anything else runs
    for job in db.unwanted_photos(conn):  # a delete that failed before the last stop
        drop_photo(conn, job)
    conn.close()

    def send_photo(job: dict) -> Response:
        name = job["photo_name"]
        try:
            with open(photo_path(name), "rb") as f:
                return Response(f.read(), media_type=next(kind for kind in ENDINGS if name.endswith(ENDINGS[kind])))
        except OSError:
            raise HTTPException(404, "This job has no photo")

    def fail_stuck_jobs() -> None:
        conn = open_db()
        try:
            for job in db.fail_stuck_jobs(conn, stuck_seconds):
                drop_photo(conn, job)
        finally:
            conn.close()

    def keep_checking(stop: threading.Event) -> None:
        while not stop.wait(check_every):
            try:
                fail_stuck_jobs()
            except sqlite3.Error as exc:  # database busy, the next look tries again
                print(f"check for stuck jobs failed: {exc}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = threading.Event()
        threading.Thread(target=keep_checking, args=(stop,), daemon=True).start()
        yield
        stop.set()

    app = FastAPI(title="PawSort orchestrator", lifespan=lifespan)

    @app.post("/jobs", status_code=201)
    def new_job(photo: UploadFile, site_code: str | None = Form(None)):
        if photo.content_type not in ENDINGS:
            raise HTTPException(422, "The photo must be sent as a JPEG or a PNG")
        name = f"{uuid.uuid4()}.{ENDINGS[photo.content_type]}"
        conn = open_db()
        try:
            with open(photo_path(name), "wb") as f:
                f.write(photo.file.read())
            return {"id": db.create_job(conn, name, site_code)}
        except Exception:
            delete_photo(photo_path(name))
            raise
        finally:
            conn.close()

    @app.post("/jobs/claim")
    def claim():
        conn = open_db()
        try:
            job = db.claim_job(conn)
        finally:
            conn.close()
        return job if job else Response(status_code=204)

    @app.get("/jobs/{job_id}")
    def read(job_id: str):
        conn = open_db()
        try:
            job = db.get_job(conn, job_id)
            if job is None:
                raise HTTPException(404, "Unknown job")
            return {**job, "review": db.get_review(conn, job_id)}
        finally:
            conn.close()

    @app.get("/jobs/{job_id}/photo")
    def working_copy(job_id: str):
        conn = open_db()
        try:
            job = db.get_job(conn, job_id)
        finally:
            conn.close()
        if job is None:
            raise HTTPException(404, "Unknown job")
        if job["status"] != "processing":
            raise HTTPException(409, "This job is not in processing")
        return send_photo(job)

    @app.post("/jobs/{job_id}/result")
    def result(job_id: str, body: JobResult):
        if body.error is None and (body.label not in LABELS or body.probability is None
                                   or body.needs_review is None):
            raise HTTPException(422, "label must be a known label, with probability and needs_review")
        if body.second_label is not None and (body.second_label not in LABELS or body.second_probability is None):
            raise HTTPException(422, "second_label must be a known label, with second_probability")
        conn = open_db()
        try:
            job = db.get_job(conn, job_id)
            if body.error is not None:
                db.set_failed(conn, job_id, body.error, body.model_version)
            else:
                db.set_result(conn, job_id, body.label, body.probability, body.needs_review, body.model_version,
                              body.second_label, body.second_probability)
            if body.error is not None or body.label == db.PERSON:  # a failed job keeps no photo, nor a person: FR-14
                drop_photo(conn, job)
            return db.get_job(conn, job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job")
        except db.InvalidTransition as exc:
            raise HTTPException(409, str(exc))
        finally:
            conn.close()

    @app.get("/review")
    def review_list():
        conn = open_db()
        try:
            return {"jobs": db.review_list(conn), "labels": sorted(LABELS)}
        finally:
            conn.close()

    @app.get("/review/{job_id}/photo")
    def review_photo(job_id: str):
        conn = open_db()
        try:
            job = db.get_job(conn, job_id)
            if job is None:
                raise HTTPException(404, "Unknown job")
            if not db.on_review_list(conn, job):
                raise HTTPException(409, "This job is not on the review list")
        finally:
            conn.close()
        return send_photo(job)

    @app.post("/jobs/{job_id}/review")
    def review(job_id: str, body: Review):
        if body.label not in LABELS:
            raise HTTPException(422, "label must be a known label")
        conn = open_db()
        try:
            job = db.get_job(conn, job_id)
            db.save_review(conn, job_id, body.label)
            if body.label == db.PERSON:  # FR-14: also when a user changes the label
                drop_photo(conn, job)
            return {**db.get_job(conn, job_id), "review": db.get_review(conn, job_id)}
        except KeyError:
            raise HTTPException(404, "Unknown job")
        except db.NotOnReviewList:
            raise HTTPException(409, "This job is not on the review list")
        finally:
            conn.close()

    return app


def build_app() -> FastAPI:
    path = os.environ.get("PAWSORT_DB", "data/pawsort.db")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return create_app(path, os.environ.get("PAWSORT_PHOTOS", "data/photos"))
