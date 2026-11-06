"""Orchestrator: the single meeting point between the interface and the AI server."""
import os

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from orchestrator import db


LABELS = {"empty", "person", "badger", "bird", "bobcat", "car", "cat", "cow", "coyote", "deer", "dog", "fox", "lizard",
          "mountain_lion", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"}


class NewJob(BaseModel):
    image_path: str


class JobResult(BaseModel):
    label: str | None = None
    probability: float | None = None
    needs_review: bool | None = None
    error: str | None = None


def create_app(db_path: str) -> FastAPI:
    app = FastAPI(title="PawSort orchestrator")

    def open_db():
        return db.connect(db_path)

    @app.post("/jobs", status_code=201)
    def new_job(body: NewJob):
        conn = open_db()
        try:
            return {"id": db.create_job(conn, body.image_path)}
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
        finally:
            conn.close()
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job

    @app.post("/jobs/{job_id}/result")
    def result(job_id: str, body: JobResult):
        if body.error is None and (body.label not in LABELS or body.probability is None
                                   or body.needs_review is None):
            raise HTTPException(422, "label must be a known label, with probability and needs_review")
        conn = open_db()
        try:
            if body.error is not None:
                db.set_failed(conn, job_id, body.error)
            else:
                db.set_result(conn, job_id, body.label, body.probability, body.needs_review)
            return db.get_job(conn, job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job")
        except db.InvalidTransition as exc:
            raise HTTPException(409, str(exc))
        finally:
            conn.close()

    return app


def build_app() -> FastAPI:
    path = os.environ.get("PAWSORT_DB", "data/pawsort.db")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return create_app(path)
