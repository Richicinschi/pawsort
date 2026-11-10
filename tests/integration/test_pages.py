import pytest
from fastapi.testclient import TestClient

from interface.app import create_app


@pytest.fixture
def ui(orch_client, tmp_path):
    return TestClient(create_app(orch_client, str(tmp_path / "uploads")))


@pytest.mark.parametrize("address", ["/", "/review"])
def test_page_loads_nothing_from_another_website(ui, address):
    page = ui.get(address).text
    assert "http://" not in page
    assert "https://" not in page
    assert "<link" not in page
    assert "<script src" not in page


def test_upload_page_has_one_box_to_drop_or_pick_photos(ui):
    page = ui.get("/").text
    box = page[page.index('id="drop"'):page.index('id="send"')]
    assert "Drop your photos here, or click to choose them" in box
    assert box.count("<input") == 1
    assert 'accept="image/jpeg,image/png" multiple required' in box


def test_small_pictures_come_from_the_browser_and_not_from_us(ui):
    page = ui.get("/").text
    assert "picture.src = URL.createObjectURL(file)" in page
    assert "The small pictures below come from your own computer" in page


def test_picture_of_a_person_goes_away(ui):
    assert 'if (job.label === "person") answer.querySelector("img").removeAttribute("src")' in ui.get("/").text


def test_website_never_gives_a_photo_back(ui, tmp_path):
    uploads = tmp_path / "uploads"
    (uploads / "trail.jpg").write_bytes(b"photo")

    for address in ["/uploads/trail.jpg", "/data/uploads/trail.jpg", "/static/trail.jpg", "/trail.jpg"]:
        assert ui.get(address).status_code == 404
