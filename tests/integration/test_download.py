import csv
import io

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_server.worker import process_one
from interface.app import create_app, text_cell

HEADER = ["job number", "camera site", "label", "percentage", "needs review", "checked by a person", "model version",
          "time of the upload"]


def photo() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (200, 200), (200, 100, 50)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def ui(orch_client):
    return TestClient(create_app(orch_client))


@pytest.fixture
def tagged(ui, orch_client, work):
    def _tagged(scores, site="A12"):
        r = ui.post("/upload", data={"site_code": site}, files=[("files", ("trail.jpg", photo()))])
        process_one(orch_client, lambda image: scores, work, "wildlife-9")
        return r.json()["photos"][0]["job_id"]
    return _tagged


def set_rare(ui, label, rare=True):
    return ui.post("/rare", json={"label": label, "rare": rare})


def rows(ui):
    return list(csv.reader(io.StringIO(ui.get("/download").text)))


def row_of(ui, job_id):
    return next(row for row in rows(ui) if row[0] == job_id)


def test_download_is_a_csv_file_with_one_row_per_tag(ui, orch_client, tagged):
    sure, unsure = tagged({"mountain_lion": 0.9349, "bobcat": 0.05}), tagged({"bobcat": 0.6, "coyote": 0.4}, "B7")

    r = ui.get("/download")

    assert r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == 'attachment; filename="pawsort-tags.csv"'
    uploaded = [orch_client.get(f"/jobs/{job_id}").json()["created_at"] for job_id in (sure, unsure)]
    assert rows(ui) == [HEADER] + sorted([
        [sure, "A12", "mountain lion", "93", "no", "no", "wildlife-9", uploaded[0]],
        [unsure, "B7", "bobcat", "60", "yes", "no", "wildlife-9", uploaded[1]]])


def test_download_without_tags_has_only_the_header(ui):
    assert rows(ui) == [HEADER]


def test_job_that_failed_or_still_waits_is_not_in_the_download(ui, orch_client, tagged, work):
    done = tagged({"bobcat": 0.9})
    ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("trail.jpg", photo()))])
    process_one(orch_client, lambda image: 1 / 0, work)
    ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("trail.jpg", photo()))])

    assert [row[0] for row in rows(ui)[1:]] == [done]


def test_camera_site_of_a_rare_animal_is_not_in_the_download(ui, tagged):
    rare, common = tagged({"mountain_lion": 0.9}, "SECRET7"), tagged({"bobcat": 0.9}, "B7")
    set_rare(ui, "mountain_lion")

    assert (row_of(ui, rare)[1], row_of(ui, common)[1]) == ("", "B7")
    assert (row_of(ui, rare)[7], row_of(ui, common)[7] != "") == ("", True)
    assert "SECRET7" not in ui.get("/download").text


def test_camera_site_is_in_the_download_again_when_the_animal_is_taken_off_the_list(ui, tagged):
    job_id = tagged({"mountain_lion": 0.9}, "B7")
    set_rare(ui, "mountain_lion")
    set_rare(ui, "mountain_lion", False)
    assert row_of(ui, job_id)[1] == "B7"


def test_label_a_user_picked_counts_and_not_the_first_tag_of_the_model(ui, tagged):
    job_id = tagged({"bobcat": 0.6, "coyote": 0.4}, "SECRET7")
    set_rare(ui, "mountain_lion")
    assert row_of(ui, job_id)[1:6] == ["SECRET7", "bobcat", "60", "yes", "no"]

    ui.post(f"/review/{job_id}", json={"label": "mountain_lion"})

    assert row_of(ui, job_id)[1:6] == ["", "mountain lion", "60", "no", "yes"]
    assert "SECRET7" not in ui.get("/download").text


def test_rejected_correction_is_in_the_download_with_the_old_label(ui, tagged):
    job_id = tagged({"bobcat": 0.6, "coyote": 0.4})
    ui.post(f"/review/{job_id}", json={"label": "coyote"})
    ui.post(f"/corrections/{job_id}", json={"decision": "rejected"})

    assert row_of(ui, job_id)[2:6] == ["bobcat", "60", "no", "yes"]


def test_rare_page_lists_the_rare_animals_and_the_animals_that_can_be_added(ui):
    set_rare(ui, "mountain_lion")

    answer = ui.get("/rare/list").json()

    assert answer["rare"] == [{"value": "mountain_lion", "text": "mountain lion"}]
    assert len(answer["animals"]) == 17 and {"value": "badger", "text": "badger"} in answer["animals"]
    assert {"value": "mountain_lion", "text": "mountain lion"} not in answer["animals"]


def test_rare_page_adds_and_removes_an_animal(ui):
    assert set_rare(ui, "badger").json()["rare"] == [{"value": "badger", "text": "badger"}]
    r = set_rare(ui, "badger", False)
    assert (r.status_code, r.json()["rare"]) == (200, [])
    assert ui.get("/rare/list").json()["rare"] == []


@pytest.mark.parametrize("label", ["unicorn", "person", "empty"])
def test_rare_page_refuses_what_is_no_animal_in_plain_words(ui, label):
    r = set_rare(ui, label)
    assert r.status_code == 422 and "pick an animal from the list" in r.json()["detail"]
    assert ui.get("/rare/list").json()["rare"] == []


def test_rare_page_loads_with_a_link_back_and_says_who_it_is_for(ui):
    r = ui.get("/rare")
    assert r.status_code == 200
    assert 'href="/"' in r.text
    assert "This page is meant for the coordinator. There is no login yet, so everybody can use it." in r.text
    for outside in ("http://", "https://", "<link", "<script src"):
        assert outside not in r.text


def test_upload_page_links_to_the_download_and_to_the_rare_animals(ui):
    page = ui.get("/").text
    assert 'href="/download"' in page and 'href="/rare"' in page


def test_middle_server_down_gets_a_friendly_message():
    down = httpx.Client(base_url="http://127.0.0.1:9", timeout=1)
    ui = TestClient(create_app(down))

    for r in (ui.get("/download"), ui.get("/rare/list"), set_rare(ui, "badger")):
        assert r.status_code == 503 and "try again" in r.json()["detail"]


def test_rows_are_in_the_order_of_the_job_numbers_so_a_rare_row_does_not_sit_between_its_upload(ui, tagged):
    jobs = [tagged({"mountain_lion": 0.9} if number == 3 else {"bobcat": 0.9}, "SECRET7") for number in range(8)]
    set_rare(ui, "mountain_lion")

    assert [row[0] for row in rows(ui)[1:]] == sorted(jobs)


def test_site_code_that_starts_with_a_dash_is_not_read_as_a_formula(ui, tagged):
    job_id = tagged({"bobcat": 0.9}, "-A12")
    assert row_of(ui, job_id)[1] == "'-A12"
    assert f"{job_id},'-A12,bobcat" in ui.get("/download").text


@pytest.mark.parametrize("text,cell", [("=1+1", "'=1+1"), ("+A1", "'+A1"), ("-A12", "'-A12"), ("@SUM(A1)", "'@SUM(A1)"),
                                       ("A-12", "A-12"), ("wildlife-1.0", "wildlife-1.0"), ("", ""), (None, "")])
def test_text_cell_that_starts_like_a_formula_gets_a_mark_in_front(text, cell):
    assert text_cell(text) == cell


def test_cell_with_a_comma_and_a_quote_stays_one_cell(ui, orch_client, work):
    r = ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("trail.jpg", photo()))])
    process_one(orch_client, lambda image: {"bobcat": 0.9}, work, 'v1, the "new" one')

    line = ui.get("/download").text.splitlines()[1]

    assert row_of(ui, r.json()["photos"][0]["job_id"])[6] == 'v1, the "new" one'
    assert '"v1, the ""new"" one"' in line
