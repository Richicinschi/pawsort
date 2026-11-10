import io
import os
import tempfile
import uuid

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import ExifTags, Image

from interface.app import create_app

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SECRET = "SECRET-PLACE"


def photo(fmt: str = "JPEG", size: tuple[int, int] = (200, 200), **extra) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 100, 50)).save(buf, format=fmt, **extra)
    return buf.getvalue()


def data_file(name: str) -> bytes:
    with open(os.path.join(DATA, name), "rb") as f:
        return f.read()


def upload(ui: TestClient, *files: tuple[str, bytes], site_code: str = "A12") -> list[dict]:
    r = ui.post("/upload", data={"site_code": site_code}, files=[("files", file) for file in files])
    assert r.status_code == 200
    return r.json()["photos"]


@pytest.fixture
def ui(orch_client):
    return TestClient(create_app(orch_client))


def stored(orch, photos, job_id) -> bytes:
    return (photos / orch.get(f"/jobs/{job_id}").json()["photo_name"]).read_bytes()


def test_several_photos_get_one_job_each(ui, orch_client):
    photos = upload(ui, ("bobcat_1.jpg", data_file("bobcat_1.jpg")), ("made.png", photo("PNG")))

    assert [p["file"] for p in photos] == ["bobcat_1.jpg", "made.png"]
    assert len({p["job_id"] for p in photos}) == 2
    for p in photos:
        job = orch_client.get(f"/jobs/{p['job_id']}").json()
        assert (job["status"], job["site_code"]) == ("new", "A12")
        assert ui.get(f"/result/{p['job_id']}").json() == {"status": "new"}


def test_bad_file_gets_a_reason_and_the_good_one_still_goes_through(ui, orch_client, photos):
    bad, good = upload(ui, ("not_an_image.jpg", data_file("not_an_image.jpg")), ("good.jpg", photo()))

    assert "cannot be opened" in bad["error"] and "job_id" not in bad
    assert orch_client.post("/jobs/claim").json()["id"] == good["job_id"]
    assert orch_client.post("/jobs/claim").status_code == 204
    assert len(os.listdir(photos)) == 1


REFUSED = {
    "empty.jpg": (lambda: b"", "empty"),
    "small.jpg": (lambda: photo(size=(127, 127)), "at least 128 x 128"),
    "large.jpg": (lambda: photo(size=(6001, 200)), "at most 6000 x 6000"),
    "moving.gif": (lambda: photo("GIF"), "Only JPEG and PNG"),
}


@pytest.mark.parametrize("name", REFUSED)
def test_refused_file_makes_no_job_and_no_file(ui, orch_client, photos, name):
    make_data, message = REFUSED[name]
    assert message in upload(ui, (name, make_data()))[0]["error"]
    assert orch_client.post("/jobs/claim").status_code == 204
    assert os.listdir(photos) == []


@pytest.mark.parametrize("site_code", ["", "two words"])
def test_bad_site_code_stops_the_whole_upload(ui, orch_client, photos, site_code):
    r = ui.post("/upload", data={"site_code": site_code}, files=[("files", ("good.jpg", photo()))])
    assert r.status_code == 400 and "camera site code" in r.json()["detail"]
    assert orch_client.post("/jobs/claim").status_code == 204
    assert os.listdir(photos) == []


def test_grey_night_photo_is_accepted(ui):
    buf = io.BytesIO()
    Image.new("L", (200, 200), 90).save(buf, format="JPEG")
    assert "job_id" in upload(ui, ("night.jpg", buf.getvalue()))[0]


def test_location_data_is_gone_from_a_stored_jpeg(ui, orch_client, photos):
    exif = Image.Exif()
    exif[ExifTags.IFD.GPSInfo] = {ExifTags.GPS.GPSLatitudeRef: "N", ExifTags.GPS.GPSAreaInformation: SECRET}
    sent = photo(exif=exif, xmp=SECRET.encode(), comment=SECRET.encode())
    assert sent.count(SECRET.encode()) == 3
    assert Image.open(io.BytesIO(sent)).getexif().get_ifd(ExifTags.IFD.GPSInfo)

    job_id = upload(ui, ("gps.jpg", sent))[0]["job_id"]

    kept = stored(orch_client, photos, job_id)
    assert SECRET.encode() not in kept
    assert not Image.open(io.BytesIO(kept)).getexif().get_ifd(ExifTags.IFD.GPSInfo)


def test_location_data_is_gone_from_a_stored_png(ui, orch_client, photos):
    exif = Image.Exif()
    exif[ExifTags.IFD.GPSInfo] = {ExifTags.GPS.GPSAreaInformation: SECRET}
    sent = photo("PNG", exif=exif)
    assert SECRET.encode() in sent

    job_id = upload(ui, ("gps.png", sent))[0]["job_id"]

    assert SECRET.encode() not in stored(orch_client, photos, job_id)


def test_stored_photo_is_still_the_same_picture(ui, orch_client, photos):
    job_id = upload(ui, ("bobcat_1.jpg", data_file("bobcat_1.jpg")))[0]["job_id"]

    kept = Image.open(io.BytesIO(stored(orch_client, photos, job_id)))

    assert (kept.format, kept.size) == ("JPEG", Image.open(io.BytesIO(data_file("bobcat_1.jpg"))).size)


def test_png_is_stored_as_a_png(ui, orch_client, photos):
    job_id = upload(ui, ("made.png", photo("PNG")))[0]["job_id"]

    assert orch_client.get(f"/jobs/{job_id}").json()["photo_name"].endswith(".png")
    assert Image.open(io.BytesIO(stored(orch_client, photos, job_id))).format == "PNG"


def test_saved_under_a_random_name_that_the_website_never_serves(ui, orch_client, photos):
    job_id = upload(ui, ("../../trail cam.jpg", photo()))[0]["job_id"]

    (name,) = os.listdir(photos)
    assert uuid.UUID(name.removesuffix(".jpg")).version == 4
    assert orch_client.get(f"/jobs/{job_id}").json()["photo_name"] == name
    for guess in (f"/{name}", f"/uploads/{name}", f"/photos/{name}", f"/data/photos/{name}", f"/static/{name}"):
        assert ui.get(guess).status_code == 404


def test_name_the_user_gave_never_leaves_the_website():
    sent = []

    def answer(request):
        sent.append(request.content)
        return httpx.Response(201, json={"id": "job-1"})

    middle = httpx.Client(base_url="http://middle", transport=httpx.MockTransport(answer))
    upload(TestClient(create_app(middle)), ("holiday of anna.jpg", photo()))

    assert b"anna" not in sent[0] and b"image/jpeg" in sent[0]


def test_website_keeps_nothing_on_disk(ui, tmp_path, monkeypatch):
    home, temp = tmp_path / "website", tmp_path / "temp"
    home.mkdir()
    temp.mkdir()
    monkeypatch.chdir(home)
    monkeypatch.setattr(tempfile, "tempdir", str(temp))  # where a large upload waits while it comes in
    large = io.BytesIO()
    Image.frombytes("RGB", (800, 800), os.urandom(800 * 800 * 3)).save(large, format="PNG")
    assert len(large.getvalue()) > 1024 * 1024

    answers = upload(ui, ("good.jpg", photo()), ("large.png", large.getvalue()),
                     ("not_an_image.jpg", data_file("not_an_image.jpg")))

    assert "job_id" in answers[1]
    assert os.listdir(home) == [] and os.listdir(temp) == []
    assert sorted(os.listdir(tmp_path)) == ["jobs.db", "photos", "temp", "website"]


def test_shows_still_working_then_the_label_and_percentage(ui, orch_client):
    job_id = upload(ui, ("cat.jpg", photo()))[0]["job_id"]
    orch_client.post("/jobs/claim")
    assert ui.get(f"/result/{job_id}").json() == {"status": "processing"}

    orch_client.post(f"/jobs/{job_id}/result",
                     json={"label": "mountain_lion", "probability": 0.87, "needs_review": False})

    assert ui.get(f"/result/{job_id}").json() == {
        "status": "done", "label": "mountain lion", "percent": 87, "needs_review": False}


def test_needs_review_never_shows_80_percent(ui, orch_client):
    job_id = upload(ui, ("cat.jpg", photo()))[0]["job_id"]
    orch_client.post("/jobs/claim")
    orch_client.post(f"/jobs/{job_id}/result", json={"label": "bobcat", "probability": 0.7996, "needs_review": True})

    assert ui.get(f"/result/{job_id}").json() == {
        "status": "done", "label": "bobcat", "percent": 79, "needs_review": True,
        "reason": "The model is less than 80% sure this is a bobcat.", "second_guess": None}


def test_failed_job_shows_a_friendly_message(ui, orch_client):
    job_id = upload(ui, ("cat.jpg", photo()))[0]["job_id"]
    orch_client.post("/jobs/claim")
    orch_client.post(f"/jobs/{job_id}/result", json={"error": "OSError: broken data stream in C:\\secret"})

    assert ui.get(f"/result/{job_id}").json() == {
        "status": "failed", "error": "We could not check this photo. Please try another one."}


def test_answer_never_names_the_stored_photo(ui, orch_client, photos):
    job_id = upload(ui, ("cat.jpg", photo()))[0]["job_id"]
    (name,) = os.listdir(photos)

    assert name not in ui.get(f"/result/{job_id}").text


def test_unknown_job(ui):
    assert ui.get(f"/result/{uuid.uuid4()}").status_code == 404


@pytest.mark.parametrize("address", ["claim%3F", "nope"])
def test_result_address_that_is_no_job_number_is_refused(ui, address):
    assert ui.get(f"/result/{address}").status_code == 422


def test_middle_server_down_makes_no_job_and_says_so():
    down = httpx.Client(base_url="http://127.0.0.1:9", timeout=1)
    ui = TestClient(create_app(down))

    assert "try again" in upload(ui, ("cat.jpg", photo()))[0]["error"]


def test_page_has_the_upload_form_and_the_notice_next_to_the_button(ui):
    r = ui.get("/")
    assert r.status_code == 200
    page = r.text
    assert 'id="site"' in page and 'id="files"' in page and "multiple" in page
    button, notice = page.index("<button"), page.index('id="notice"')
    assert 0 < notice - button < 200
