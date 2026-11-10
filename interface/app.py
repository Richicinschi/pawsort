"""Interface server: talks to the user and to the orchestrator. Never to the AI server. It keeps no photo."""
import csv
import io
import os
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

from interface.validation import MAX_BYTES, UploadError, check_site_code, validate_upload

PAGE = Path(__file__).parent / "static" / "index.html"
REVIEW_PAGE = Path(__file__).parent / "static" / "review.html"
CORRECTIONS_PAGE = Path(__file__).parent / "static" / "corrections.html"
NUMBERS_PAGE = Path(__file__).parent / "static" / "numbers.html"
SAMPLE_PAGE = Path(__file__).parent / "static" / "sample.html"
RARE_PAGE = Path(__file__).parent / "static" / "rare.html"
PERSON_NOTE = "This photo shows a person, so we deleted it."
WEAK_BOX = "The model saw something in this photo, but not clearly enough to say what it is."
NO_SERVER = "We could not reach the server. Please try again."
NO_PHOTO = "There is no photo to check for this job."
REVIEW_ERRORS = {404: "We cannot find this job.", 409: "This job is already checked.",
                 422: "Please pick a label from the list."}
DECISION_ERRORS = {404: "We cannot find this correction.", 409: "This correction already has a decision.",
                   422: "Please press Approve or Reject."}
SAMPLE_ERRORS = {**REVIEW_ERRORS, 409: "This tag does not wait in the sample of this week. Somebody already checked "
                                      "it, or it belongs to the sample of another week."}
RARE_ERRORS = {422: "Please pick an animal from the list."}
COLUMNS = ["job number", "camera site", "label", "percentage", "needs review", "checked by a person",
           "model version", "time of the upload"]


class Review(BaseModel):
    label: str


class Decision(BaseModel):
    decision: str


class Rare(BaseModel):
    label: str
    rare: bool


def show_label(label: str) -> str:
    return label.replace("_", " ")


def percent(probability: float) -> int:
    return int(round(probability * 100, 2))


def with_article(label: str) -> str:
    return f"{'an' if label[0] in 'aeiou' else 'a'} {show_label(label)}"


def why_review(label: str) -> str:
    if label == "empty":  # the model is only unsure about empty after a weak box
        return WEAK_BOX
    return f"The model is less than 80% sure this is {with_article(label)}."


def second_guess(job: dict) -> str | None:
    if not job.get("second_label"):
        return None
    return f"Its second guess is {with_article(job['second_label'])}, {percent(job['second_probability'])}% sure."


def show_job(job: dict) -> dict:
    out = {"status": job["status"]}
    if job["status"] == "done":
        out["label"] = show_label(job["label"])
        if job.get("review"):  # a person decided, so the percentage of the model no longer counts
            out.update(needs_review=False, reviewed=True)
        else:
            out.update(percent=percent(job["probability"]), needs_review=bool(job["needs_review"]))
            if job["needs_review"]:
                out.update(reason=why_review(job["label"]), second_guess=second_guess(job))
        if job["label"] == "person":
            out["note"] = PERSON_NOTE
    elif job["status"] == "failed":
        out["error"] = "We could not check this photo. Please try another one."
    return out


def show_review_job(job: dict) -> dict:
    return {"job_id": job["id"], "site_code": job["site_code"] or "unknown", "label": job["label"],
            "label_text": show_label(job["label"]), "percent": percent(job["probability"]),
            "reason": why_review(job["label"]), "second_guess": second_guess(job),
            "model_version": job["model_version"] or "unknown"}


def show_correction(job: dict) -> dict:
    return {"job_id": job["id"], "site_code": job["site_code"] or "unknown", "old_label": show_label(job["old_label"]),
            "new_label": show_label(job["new_label"]), "model_version": job["model_version"] or "unknown"}


def waiting_corrections(numbers: dict) -> str:
    waiting = numbers["for_training"]
    note = "1 approved correction waits" if waiting == 1 else f"{waiting} approved corrections wait"
    return note + " for the next training."


def training_note(numbers: dict) -> str:
    note = waiting_corrections(numbers)
    return note + " It is time to plan a training round." if numbers["plan_training"] else note


def share(number: float) -> str:
    """A share with all its decimals, so 40.48% never reads as 40% next to the words more than 40%."""
    return f"{number * 100:.2f}".rstrip("0").rstrip(".") + "%"


def show_numbers(row: dict) -> dict:
    corrected = "no tag checked yet"
    if row["corrected"] is not None:
        corrected = f"{percent(row['corrected'])}% (of {row['checked']} checked)"
    return {"site_code": row["site_code"] or "unknown", "week": row["week"], "tags": row["tags"],
            "needs_review": share(row["needs_review"]), "corrected": corrected, "warning": row["warning"]}


def site_warnings(numbers: dict) -> list[str]:
    return [f"Camera site {row['site_code'] or 'unknown'}, week {row['week']}: {share(row['needs_review'])} of "
            f"the tags needed review. That is more than {share(numbers['warn_share'])}."
            for row in numbers["rows"] if row["warning"]]


def training_reasons(numbers: dict) -> list[str]:
    """Every reason that holds right now to plan a training round: AI-09."""
    reasons = [waiting_corrections(numbers)] if numbers["plan_training"] else []
    reasons += [f"Camera site {site} is new: its first photo came after the current model was trained on "
                f"{numbers['trained_on']}." for site in numbers["new_sites"]]
    reasons += [f"Camera site {row['site_code'] or 'unknown'} had more than {share(numbers['warn_share'])} "
                f"needs review in week {row['week']}." for row in numbers["rows"] if row["warning"]]
    return reasons


def trained_note(numbers: dict) -> str:
    if numbers["trained_on"]:
        return f"The current model was trained on {numbers['trained_on']}."
    return "We do not know when the current model was trained, so this page cannot tell which camera sites are new."


def show_sample_job(job: dict) -> dict:
    return {"job_id": job["id"], "site_code": job["site_code"] or "unknown", "label": job["label"],
            "label_text": show_label(job["label"]), "percent": percent(job["probability"]),
            "model_version": job["model_version"] or "unknown"}


def sample_note(sample: dict) -> str:
    if not sample["picked"]:
        return (f"Sample of week {sample['week']}: no sure tag came in since the sample before, "
                "so there is nothing to check.")
    tags = "1 sure tag" if sample["picked"] == 1 else f"{sample['picked']} sure tags"
    return (f"Sample of week {sample['week']}: {tags} picked, {sample['checked']} of them checked, "
            f"{sample['changed']} of those changed.")


def options(labels: list[str]) -> list[dict]:
    return [{"value": label, "text": show_label(label)} for label in labels]


def show_rare(body: dict) -> dict:
    return {"rare": options(body["rare"]),
            "animals": options([label for label in body["animals"] if label not in body["rare"]])}


def text_cell(text: str | None) -> str:
    """A spreadsheet reads a cell that starts with one of these signs as a formula, so it gets a ' in front."""
    text = text or ""
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def csv_row(tag: dict) -> list:
    yes_no = {True: "yes", False: "no"}
    return [tag["id"], text_cell(tag["site_code"]), text_cell(show_label(tag["label"])), percent(tag["probability"]),
            yes_no[tag["needs_review"]], yes_no[tag["checked"]], text_cell(tag["model_version"]),
            tag["created_at"] or ""]


def create_app(orch: httpx.Client) -> FastAPI:
    app = FastAPI(title="PawSort")

    def ask(method: str, address: str, errors: dict | None = None, **more) -> dict:
        """Ask the middle server. Its refusals become our own sentences, anything else is a friendly 503."""
        try:
            r = orch.request(method, address, **more)
        except httpx.HTTPError:
            raise HTTPException(503, NO_SERVER)
        if r.status_code in (errors or {}):
            raise HTTPException(r.status_code, errors[r.status_code])
        if r.is_error:
            raise HTTPException(503, NO_SERVER)
        return r.json()

    def pass_photo(address: str) -> Response:
        try:
            r = orch.get(address)
        except httpx.HTTPError:
            raise HTTPException(503, NO_SERVER)
        if r.is_error:  # the middle server alone decides who gets a photo
            raise HTTPException(404, NO_PHOTO)
        return Response(r.content, media_type=r.headers["content-type"], headers={"Cache-Control": "no-store"})

    def send_photo(data: bytes, site_code: str) -> str:
        fmt = validate_upload(data)
        clean = io.BytesIO()
        Image.open(io.BytesIO(data)).save(clean, format=fmt, quality="keep", comment=b"")  # drops all metadata: FR-04
        try:
            r = orch.post("/jobs", data={"site_code": site_code},
                          files={"photo": ("photo", clean.getvalue(), f"image/{fmt.lower()}")})
            r.raise_for_status()
        except httpx.HTTPError:
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
                photos.append({"file": file.filename, "job_id": send_photo(file.file.read(MAX_BYTES + 1), site_code)})
            except UploadError as exc:
                photos.append({"file": file.filename, "error": str(exc)})
        return {"photos": photos}

    @app.get("/result/{job_id}")
    def result(job_id: uuid.UUID):  # only a real job number may go into an address of the middle server
        return show_job(ask("GET", f"/jobs/{job_id}", {404: "Unknown job."}))

    @app.get("/review", response_class=HTMLResponse)
    def review_page():
        return REVIEW_PAGE.read_text(encoding="utf-8")

    @app.get("/review/jobs")
    def review_jobs():
        body = ask("GET", "/review")
        return {"jobs": [show_review_job(job) for job in body["jobs"]], "labels": options(body["labels"])}

    @app.get("/review/{job_id}/photo")
    def review_photo(job_id: uuid.UUID):
        return pass_photo(f"/review/{job_id}/photo")

    @app.post("/review/{job_id}")
    def review(job_id: uuid.UUID, body: Review):
        return show_job(ask("POST", f"/jobs/{job_id}/review", REVIEW_ERRORS, json={"label": body.label}))

    @app.get("/corrections", response_class=HTMLResponse)
    def corrections_page():
        return CORRECTIONS_PAGE.read_text(encoding="utf-8")

    @app.get("/corrections/jobs")
    def correction_jobs():
        body = ask("GET", "/corrections")
        return {"jobs": [show_correction(job) for job in body["jobs"]], "note": training_note(body)}

    @app.get("/corrections/{job_id}/photo")
    def correction_photo(job_id: uuid.UUID):
        return pass_photo(f"/corrections/{job_id}/photo")

    @app.post("/corrections/{job_id}")
    def decide(job_id: uuid.UUID, body: Decision):
        numbers = ask("POST", f"/corrections/{job_id}", DECISION_ERRORS, json={"decision": body.decision})
        return {"note": training_note(numbers)}

    @app.get("/numbers", response_class=HTMLResponse)
    def numbers_page():
        return NUMBERS_PAGE.read_text(encoding="utf-8")

    @app.get("/numbers/rows")
    def numbers_rows():
        numbers = ask("GET", "/numbers")
        return {"rows": [show_numbers(row) for row in numbers["rows"]], "warnings": site_warnings(numbers),
                "reasons": training_reasons(numbers), "trained": trained_note(numbers)}

    @app.get("/sample", response_class=HTMLResponse)
    def sample_page():
        return SAMPLE_PAGE.read_text(encoding="utf-8")

    @app.get("/sample/jobs")
    def sample_jobs():
        sample = ask("GET", "/sample")
        return {"jobs": [show_sample_job(job) for job in sample["jobs"]], "note": sample_note(sample),
                "labels": options(sample["labels"])}

    @app.get("/sample/{job_id}/photo")
    def sample_photo(job_id: uuid.UUID):
        return pass_photo(f"/sample/{job_id}/photo")

    @app.post("/sample/{job_id}")
    def sample_answer(job_id: uuid.UUID, body: Review):
        return show_job(ask("POST", f"/sample/{job_id}", SAMPLE_ERRORS, json={"label": body.label}))

    @app.get("/rare", response_class=HTMLResponse)
    def rare_page():
        return RARE_PAGE.read_text(encoding="utf-8")

    @app.get("/rare/list")
    def rare_list():
        return show_rare(ask("GET", "/rare"))

    @app.post("/rare")
    def set_rare(body: Rare):
        return show_rare(ask("POST", "/rare", RARE_ERRORS, json={"label": body.label, "rare": body.rare}))

    @app.get("/download")
    def download():
        out = io.StringIO()
        rows = csv.writer(out, lineterminator="\n")
        rows.writerow(COLUMNS)
        rows.writerows(csv_row(tag) for tag in ask("GET", "/tags")["tags"])
        return Response(out.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="pawsort-tags.csv"'})

    return app


def build_app() -> FastAPI:
    orch = httpx.Client(base_url=os.environ.get("PAWSORT_ORCH", "http://127.0.0.1:8001"), timeout=10)
    return create_app(orch)
