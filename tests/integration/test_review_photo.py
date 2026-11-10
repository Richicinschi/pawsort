import io
import os
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_server.worker import process_one
from interface.app import create_app


def photo() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def ui(orch_client):
    return TestClient(create_app(orch_client))


def tagged(ui, orch, work, scores):
    r = ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("trail.jpg", photo()))])
    process_one(orch, lambda image: scores, work)
    return r.json()["photos"][0]["job_id"]


def stored(photos) -> bytes:
    (name,) = os.listdir(photos)
    return (photos / name).read_bytes()


def test_website_passes_on_the_photo_of_a_job_on_the_review_list(ui, orch_client, work, photos):
    job_id = tagged(ui, orch_client, work, {"bobcat": 0.6, "coyote": 0.4})

    r = ui.get(f"/review/{job_id}/photo")

    assert (r.status_code, r.headers["content-type"]) == (200, "image/jpeg")
    assert r.content == stored(photos)


def test_sure_tag_never_comes_back_as_a_picture(ui, orch_client, work):
    job_id = tagged(ui, orch_client, work, {"bobcat": 0.95, "coyote": 0.05})
    assert ui.get(f"/review/{job_id}/photo").status_code == 404


def test_photo_stops_coming_back_once_a_person_checked_the_job(ui, orch_client, work):
    job_id = tagged(ui, orch_client, work, {"bobcat": 0.6, "coyote": 0.4})

    ui.post(f"/review/{job_id}", json={"label": "bobcat"})

    assert ui.get(f"/review/{job_id}/photo").status_code == 404


def test_job_that_is_still_waiting_gives_no_picture(ui):
    r = ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("trail.jpg", photo()))])
    assert ui.get(f"/review/{r.json()['photos'][0]['job_id']}/photo").status_code == 404


def test_job_that_failed_gives_no_picture(ui, orch_client, work):
    r = ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("trail.jpg", photo()))])
    process_one(orch_client, lambda image: 1 / 0, work)
    job_id = r.json()["photos"][0]["job_id"]

    assert orch_client.get(f"/jobs/{job_id}").json()["status"] == "failed"
    assert ui.get(f"/review/{job_id}/photo").status_code == 404


def test_png_comes_back_as_a_png(ui, orch_client, work, photos):
    sent = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 100, 50)).save(sent, format="PNG")
    r = ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("trail.png", sent.getvalue()))])
    process_one(orch_client, lambda image: {"bobcat": 0.6, "coyote": 0.4}, work)

    shown = ui.get(f"/review/{r.json()['photos'][0]['job_id']}/photo")

    assert (shown.status_code, shown.headers["content-type"]) == (200, "image/png")
    assert Image.open(io.BytesIO(shown.content)).format == "PNG" and shown.content == stored(photos)


@pytest.mark.parametrize("address", ["claim%3F", "nope"])
def test_photo_address_that_is_no_job_number_is_refused(ui, address):
    assert ui.get(f"/review/{address}/photo").status_code == 422


def test_unknown_job_gives_no_picture(ui):
    r = ui.get(f"/review/{uuid.uuid4()}/photo")
    assert r.status_code == 404 and "no photo" in r.json()["detail"]


def test_browser_is_told_not_to_keep_the_photo(ui, orch_client, work):
    job_id = tagged(ui, orch_client, work, {"bobcat": 0.6, "coyote": 0.4})
    assert ui.get(f"/review/{job_id}/photo").headers["cache-control"] == "no-store"


def test_website_gives_a_photo_back_at_no_other_address(ui, orch_client, work, photos):
    job_id = tagged(ui, orch_client, work, {"bobcat": 0.6, "coyote": 0.4})
    (name,) = os.listdir(photos)

    for guess in (f"/{name}", f"/photos/{name}", f"/data/photos/{name}", f"/static/{name}",
                  f"/jobs/{job_id}/photo", f"/result/{job_id}/photo", f"/photo/{job_id}"):
        assert ui.get(guess).status_code == 404
    assert ui.get(f"/review/{name}/photo").status_code == 422


def test_middle_server_down_gets_a_friendly_message():
    down = httpx.Client(base_url="http://127.0.0.1:9", timeout=1)
    r = TestClient(create_app(down)).get(f"/review/{uuid.uuid4()}/photo")
    assert r.status_code == 503 and "try again" in r.json()["detail"]
