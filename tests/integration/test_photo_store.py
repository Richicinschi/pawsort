import io
import os
import sqlite3
import threading
import uuid

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_server.worker import process_one
from orchestrator import app as orch_app
from orchestrator import db
from orchestrator.app import create_app

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def data_file(name: str) -> bytes:
    with open(os.path.join(DATA, name), "rb") as f:
        return f.read()


def finish(orch, job_id, label="bobcat", probability=0.62, needs_review=True):
    return orch.post(f"/jobs/{job_id}/result",
                     json={"label": label, "probability": probability, "needs_review": needs_review})


def marked_job(orch, send_photo, label="bobcat"):
    job_id = send_photo(orch)
    orch.post("/jobs/claim")
    finish(orch, job_id, label)
    return job_id


def test_photo_is_saved_under_a_random_name_in_the_photo_folder(orch_client, send_photo, photos):
    send_photo(orch_client)

    (name,) = os.listdir(photos)

    assert uuid.UUID(name.removesuffix(".jpg")).version == 4
    assert (photos / name).read_bytes() == data_file("bobcat_1.jpg")


def test_png_keeps_its_own_ending(orch_client, photos):
    orch_client.post("/jobs", files={"photo": ("photo", b"png bytes", "image/png")})
    assert os.listdir(photos)[0].endswith(".png")


def test_job_row_keeps_only_the_name_and_the_site_code(orch_client, send_photo, photos):
    job_id = send_photo(orch_client, site_code="A12")

    job = orch_client.get(f"/jobs/{job_id}").json()

    assert (job["status"], job["site_code"], [job["photo_name"]]) == ("new", "A12", os.listdir(photos))
    assert str(photos) not in str(job)


def test_name_the_sender_gave_is_never_used(orch_client, photos, tmp_path):
    orch_client.post("/jobs", files={"photo": ("../../trail cam.jpg", data_file("bobcat_1.jpg"), "image/jpeg")})

    (name,) = os.listdir(photos)

    assert "trail" not in name
    assert not list(tmp_path.parent.rglob("trail cam.jpg"))


def test_no_request_can_make_a_job_for_a_path(orch_client, photos, tmp_path):
    victim = tmp_path / "notes.txt"
    victim.write_text("not a photo")

    r = orch_client.post("/jobs", json={"image_path": str(victim)})

    assert r.status_code == 422
    assert orch_client.post("/jobs/claim").status_code == 204
    assert os.listdir(photos) == []


def test_path_next_to_a_photo_is_not_kept(orch_client, photos, tmp_path, work):
    victim = tmp_path / "notes.txt"
    victim.write_text("not a photo")
    r = orch_client.post("/jobs", data={"image_path": str(victim), "photo_name": str(victim)},
                         files={"photo": ("photo", data_file("bobcat_1.jpg"), "image/jpeg")})

    job = orch_client.get(f"/jobs/{r.json()['id']}").json()
    assert [job["photo_name"]] == os.listdir(photos) and "notes.txt" not in str(job)

    process_one(orch_client, lambda image: {"person": 1.0}, work)

    assert victim.read_text() == "not a photo"
    assert os.listdir(photos) == []


@pytest.mark.parametrize("kind", ["text/plain", "image/gif", "application/octet-stream"])
def test_file_that_is_not_sent_as_a_jpeg_or_png_makes_no_job(orch_client, photos, kind):
    r = orch_client.post("/jobs", files={"photo": ("photo", b"something else", kind)})

    assert r.status_code == 422
    assert orch_client.post("/jobs/claim").status_code == 204
    assert os.listdir(photos) == []


def test_photo_that_cannot_be_saved_makes_no_job(tmp_path):
    photos = tmp_path / "photos"
    orch = TestClient(create_app(str(tmp_path / "jobs.db"), str(photos)), raise_server_exceptions=False)
    photos.rmdir()

    r = orch.post("/jobs", files={"photo": ("photo", data_file("bobcat_1.jpg"), "image/jpeg")})

    assert r.status_code == 500
    assert orch.post("/jobs/claim").status_code == 204


def test_job_that_cannot_be_stored_keeps_no_photo(tmp_path, monkeypatch):
    photos = tmp_path / "photos"
    orch = TestClient(create_app(str(tmp_path / "jobs.db"), str(photos)), raise_server_exceptions=False)

    def locked(conn, photo_name, site_code):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "create_job", locked)
    r = orch.post("/jobs", files={"photo": ("photo", data_file("bobcat_1.jpg"), "image/jpeg")})

    assert r.status_code == 500
    assert os.listdir(photos) == []


def test_working_copy_goes_out_only_while_the_job_is_in_processing(orch_client, send_photo):
    job_id = send_photo(orch_client)
    assert orch_client.get(f"/jobs/{job_id}/photo").status_code == 409

    orch_client.post("/jobs/claim")
    r = orch_client.get(f"/jobs/{job_id}/photo")
    assert (r.status_code, r.headers["content-type"], r.content) == (200, "image/jpeg", data_file("bobcat_1.jpg"))

    finish(orch_client, job_id)
    assert orch_client.get(f"/jobs/{job_id}/photo").status_code == 409


def test_failed_job_gives_out_no_working_copy(orch_client, send_photo):
    job_id = send_photo(orch_client)
    orch_client.post("/jobs/claim")
    orch_client.post(f"/jobs/{job_id}/result", json={"error": "broken photo"})

    assert orch_client.get(f"/jobs/{job_id}/photo").status_code == 409


def test_photo_of_an_unknown_job(orch_client):
    assert orch_client.get("/jobs/nope/photo").status_code == 404
    assert orch_client.get("/review/nope/photo").status_code == 404


def test_review_photo_goes_out_only_while_the_job_is_on_the_review_list(orch_client, send_photo):
    job_id = send_photo(orch_client)
    assert orch_client.get(f"/review/{job_id}/photo").status_code == 409
    orch_client.post("/jobs/claim")
    assert orch_client.get(f"/review/{job_id}/photo").status_code == 409

    finish(orch_client, job_id)
    r = orch_client.get(f"/review/{job_id}/photo")
    assert (r.status_code, r.headers["content-type"], r.content) == (200, "image/jpeg", data_file("bobcat_1.jpg"))

    orch_client.post(f"/jobs/{job_id}/review", json={"label": "coyote"})
    assert orch_client.get(f"/review/{job_id}/photo").status_code == 409


def test_sure_tag_gives_out_no_review_photo(orch_client, send_photo):
    job_id = send_photo(orch_client)
    orch_client.post("/jobs/claim")
    finish(orch_client, job_id, probability=0.95, needs_review=False)

    assert orch_client.get(f"/review/{job_id}/photo").status_code == 409


def test_photo_can_only_be_asked_for_by_job_number_and_never_by_name(orch_client, send_photo, photos):
    marked_job(orch_client, send_photo)
    (name,) = os.listdir(photos)

    for guess in (f"/{name}", f"/photos/{name}", f"/data/photos/{name}", f"/jobs/{name}/photo",
                  f"/review/{name}/photo"):
        assert orch_client.get(guess).status_code == 404


def test_animal_photo_stays_after_the_check(orch_client, send_photo, photos, work):
    job_id = send_photo(orch_client)

    process_one(orch_client, lambda image: {"bobcat": 0.9, "coyote": 0.1}, work)

    assert orch_client.get(f"/jobs/{job_id}").json()["photo_name"] in os.listdir(photos)
    assert os.listdir(work) == []


def test_photo_of_a_checked_job_stays_for_the_next_training(orch_client, send_photo, photos):
    job_id = marked_job(orch_client, send_photo)

    orch_client.post(f"/jobs/{job_id}/review", json={"label": "coyote"})

    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["label"], [job["photo_name"]]) == ("coyote", os.listdir(photos))


def test_png_comes_back_as_a_png(orch_client):
    sent = io.BytesIO()
    Image.new("RGB", (200, 200)).save(sent, format="PNG")
    job_id = orch_client.post("/jobs", files={"photo": ("photo", sent.getvalue(), "image/png")}).json()["id"]

    orch_client.post("/jobs/claim")
    copy = orch_client.get(f"/jobs/{job_id}/photo")
    finish(orch_client, job_id)
    shown = orch_client.get(f"/review/{job_id}/photo")

    assert (copy.headers["content-type"], copy.content) == ("image/png", sent.getvalue())
    assert (shown.headers["content-type"], shown.content) == ("image/png", sent.getvalue())


def test_photo_of_a_broken_job_is_deleted(orch_client, send_photo, photos, work):
    job_id = send_photo(orch_client, "not_an_image.jpg")
    assert len(os.listdir(photos)) == 1

    process_one(orch_client, lambda image: {"bobcat": 0.9}, work)

    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["status"], job["photo_name"]) == ("failed", "")
    assert os.listdir(photos) == [] and os.listdir(work) == []


def test_photo_is_deleted_when_the_model_crashes_on_it(orch_client, send_photo, photos, work):
    job_id = send_photo(orch_client)

    process_one(orch_client, lambda image: 1 / 0, work)

    job = orch_client.get(f"/jobs/{job_id}").json()
    assert (job["status"], job["photo_name"]) == ("failed", "")
    assert os.listdir(photos) == []


def test_failed_job_gives_out_no_review_photo(orch_client, send_photo):
    job_id = send_photo(orch_client)
    orch_client.post("/jobs/claim")
    orch_client.post(f"/jobs/{job_id}/result", json={"error": "broken photo"})

    assert orch_client.get(f"/review/{job_id}/photo").status_code == 409


def test_photo_that_cannot_be_deleted_keeps_its_row_and_is_never_given_out(orch_client, send_photo, photos,
                                                                            monkeypatch):
    person, failed = send_photo(orch_client), send_photo(orch_client)
    orch_client.post("/jobs/claim")
    orch_client.post("/jobs/claim")
    monkeypatch.setattr(orch_app, "delete_photo", lambda path: False)  # as if somebody still reads the file

    assert finish(orch_client, person, "person", 1.0, False).status_code == 200
    assert orch_client.post(f"/jobs/{failed}/result", json={"error": "broken photo"}).status_code == 200

    kept = [orch_client.get(f"/jobs/{job_id}").json()["photo_name"] for job_id in (person, failed)]
    assert sorted(kept) == sorted(os.listdir(photos)) and len(kept) == 2
    for job_id in (person, failed):
        assert orch_client.get(f"/jobs/{job_id}/photo").status_code == 409
        assert orch_client.get(f"/review/{job_id}/photo").status_code == 409


def test_start_deletes_the_photos_that_a_failed_delete_left_behind(tmp_path, send_photo, monkeypatch):
    path, photos = str(tmp_path / "jobs.db"), tmp_path / "photos"
    before = TestClient(create_app(path, str(photos)))
    animal, person, failed, by_a_user = (send_photo(before) for _ in range(4))
    for _ in range(4):
        before.post("/jobs/claim")
    monkeypatch.setattr(orch_app, "delete_photo", lambda path: False)
    finish(before, animal, "bobcat", 0.9, False)
    finish(before, person, "person", 1.0, False)
    before.post(f"/jobs/{failed}/result", json={"error": "broken photo"})
    finish(before, by_a_user)
    before.post(f"/jobs/{by_a_user}/review", json={"label": "person"})
    assert len(os.listdir(photos)) == 4
    monkeypatch.undo()

    after = TestClient(create_app(path, str(photos)))

    assert os.listdir(photos) == [after.get(f"/jobs/{animal}").json()["photo_name"]]
    assert [after.get(f"/jobs/{job_id}").json()["photo_name"] for job_id in (person, failed, by_a_user)] == [""] * 3


def test_two_reviews_of_one_job_at_the_same_moment_delete_the_photo_once(orch_client, send_photo, photos, monkeypatch):
    job_id = marked_job(orch_client, send_photo)
    deleted, real_delete = [], orch_app.delete_photo
    monkeypatch.setattr(orch_app, "delete_photo", lambda path: deleted.append(path) or real_delete(path))
    start = threading.Barrier(2)
    answers = []

    def pick_person():
        client = TestClient(orch_client.app)
        start.wait()
        answers.append(client.post(f"/jobs/{job_id}/review", json={"label": "person"}).status_code)

    users = [threading.Thread(target=pick_person) for _ in range(2)]
    for user in users:
        user.start()
    for user in users:
        user.join(timeout=30)

    assert sorted(answers) == [200, 409]
    assert len(deleted) == 1 and os.listdir(photos) == []
    assert orch_client.get(f"/jobs/{job_id}").json()["photo_name"] == ""


def test_job_without_a_photo_on_disk_answers_with_404(orch_client, send_photo, photos):
    job_id = marked_job(orch_client, send_photo)
    os.remove(photos / os.listdir(photos)[0])

    assert orch_client.get(f"/review/{job_id}/photo").status_code == 404


def old_database(tmp_path, status):
    """A database from before the middle server kept photos, with one job that points at a file that is no photo."""
    path, victim = str(tmp_path / "old.db"), tmp_path / "notes.txt"
    victim.write_text("not a photo")
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE jobs (id TEXT PRIMARY KEY, image_path TEXT NOT NULL, status TEXT NOT NULL, label TEXT, "
                "probability REAL, needs_review INTEGER, error TEXT, site_code TEXT, model_version TEXT, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
    old.execute("INSERT INTO jobs (id, image_path, status, label, probability, needs_review, created_at, updated_at) "
                "VALUES ('old-job', ?, ?, 'cat', 0.6, 1, '2026-11-01', '2026-11-01')", (str(victim), status))
    old.commit()
    old.close()
    return TestClient(create_app(path, str(tmp_path / "photos"))), victim


def test_job_from_an_older_database_has_no_photo_and_can_still_be_reviewed(tmp_path):
    orch, victim = old_database(tmp_path, "done")

    assert orch.get("/review/old-job/photo").status_code == 404
    assert orch.post("/jobs/old-job/review", json={"label": "person"}).status_code == 200
    assert victim.read_text() == "not a photo"


def test_job_that_still_waited_in_an_older_database_fails_and_never_gives_out_a_file(tmp_path):
    orch, victim = old_database(tmp_path, "new")

    assert orch.post("/jobs/claim").json()["id"] == "old-job"
    assert orch.get("/jobs/old-job/photo").status_code == 404
    orch.post("/jobs/old-job/result", json={"error": "the photo did not come"})

    job = orch.get("/jobs/old-job").json()
    assert (job["status"], job["photo_name"]) == ("failed", "")
    assert victim.read_text() == "not a photo"


def test_worker_fails_a_job_that_still_waited_in_an_older_database(tmp_path, work):
    orch, victim = old_database(tmp_path, "new")

    assert process_one(orch, lambda image: {"bobcat": 0.9}, work) is True

    assert orch.get("/jobs/old-job").json()["status"] == "failed"
    assert victim.read_text() == "not a photo" and os.listdir(work) == []
