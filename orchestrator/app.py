"""Orchestrator: the single meeting point between the interface and the AI server."""
import os

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from orchestrator import db


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

