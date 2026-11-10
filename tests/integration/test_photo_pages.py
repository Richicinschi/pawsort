import pytest
from fastapi.testclient import TestClient

from interface.app import create_app


@pytest.fixture
def ui(orch_client):
    return TestClient(create_app(orch_client))


@pytest.fixture
def notice(ui):
    page = ui.get("/").text
    return page[page.index('id="notice"'):page.index("</form>")]


@pytest.fixture
def limit(ui):
    page = ui.get("/review").text
    return page[page.index('id="limit"'):page.index('id="message"')]


def test_notice_says_which_photos_we_keep_and_why(notice):
    assert "We keep your photos on our server" in notice
    assert "because a person may have to look at a photo to check its tag" in notice
    assert "because the next training of the model needs them" in notice


def test_notice_says_that_photos_of_people_are_deleted(notice):
    assert "A photo that shows a person is deleted at once, whatever the percentage is" in notice
    assert "also when one of us marks it as a photo of a person" in notice


def test_notice_says_that_a_photo_we_could_not_check_is_deleted(notice):
    assert "A photo that we could not check is deleted too" in notice


def test_notice_says_that_the_model_only_gets_a_copy(notice):
    assert "The model only gets a copy, and that copy is deleted right after the check" in notice
    assert "also when the check fails" in notice


def test_notice_says_what_we_never_store(notice):
    assert "We do not store names or contact details" in notice
    assert "we remove the location data before we keep a photo" in notice


@pytest.mark.parametrize("address", ["/", "/review"])
def test_no_page_says_any_more_that_every_photo_is_deleted(ui, address):
    page = ui.get(address).text
    assert "delete every photo" not in page
    assert "cannot show you the photo" not in page


def test_upload_page_still_says_when_a_photo_of_a_person_is_deleted(ui):
    assert "if (job.note) show(answer" in ui.get("/").text


def test_review_page_shows_the_photo_of_every_job(ui):
    page = ui.get("/review").text
    assert "picture.src = `/review/${job.job_id}/photo`" in page
    assert "line.append(photo(job), text, more" in page


def test_review_page_takes_only_the_answered_job_off_the_list(ui):
    page = ui.get("/review").text
    answer = page[page.index("async function send("):page.index("function button(")]
    assert "line.remove();" in answer
    assert "if (!reply.ok) { say(data.detail, \"error\"); return load(); }" in answer


def test_review_page_says_so_when_a_job_has_no_photo(ui):
    assert 'picture.replaceWith("We have no photo for this job.")' in ui.get("/review").text


def test_review_page_says_how_long_we_show_a_photo(limit):
    assert "We show a photo here only while its job waits for a person" in limit
    assert "the photo stays on our server for the next training" in limit


def test_review_page_says_that_picking_person_deletes_the_photo(limit):
    assert "If you pick person, we delete the photo at once" in limit
