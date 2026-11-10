"""Orchestrator: the single meeting point between the interface and the AI server."""
import os
import sqlite3
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from orchestrator import db


LABELS = {"empty", "person", "badger", "bird", "bobcat", "car", "cat", "cow", "coyote", "deer", "dog", "fox", "lizard",
          "mountain_lion", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"}
CHECK_EVERY = 1.0  # seconds between two looks for stuck jobs


class NewJob(BaseModel):
    image_path: str
    site_code: str | None = None


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


def delete_photo(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass  # the AI server was first
    except OSError as exc:  # the AI server still has it open and deletes it when it is done
        print(f"could not delete a photo yet: {exc}")


def create_app(db_path: str, stuck_seconds: float = db.STUCK_SECONDS, check_every: float = CHECK_EVERY) -> FastAPI:
    def open_db():
        return db.connect(db_path)

    open_db().close()  # makes or upgrades the table once, before anything else runs

    def fail_stuck_jobs() -> None:
        conn = open_db()
        try:
            for path in db.fail_stuck_jobs(conn, stuck_seconds):
                delete_photo(path)
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
    def new_job(body: NewJob):
        conn = open_db()
        try:
            return {"id": db.create_job(conn, body.image_path, body.site_code)}
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
                if body.label == db.PERSON:  # FR-14: at any confidence, also if the AI server kept it
                    delete_photo(job["image_path"])
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

    @app.post("/jobs/{job_id}/review")
    def review(job_id: str, body: Review):
        if body.label not in LABELS:
            raise HTTPException(422, "label must be a known label")
        conn = open_db()
        try:
            job = db.get_job(conn, job_id)
            db.save_review(conn, job_id, body.label)
            if body.label == db.PERSON:  # FR-14: also when a user changes the label
                delete_photo(job["image_path"])
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
    return create_app(path)
