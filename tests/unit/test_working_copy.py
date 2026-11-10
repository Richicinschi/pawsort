import json
import os

import httpx
import pytest

from ai_server import detector
from ai_server.worker import delete_copies, main, process_one

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def stand_in(image):
    return {"bobcat": 0.9, "coyote": 0.1}


def fake_server(photo="bobcat_1.jpg", photo_status=200, report_status=200, claim=None):
    """A middle server that only exists in this test. It hands out one job and its photo, and keeps the report."""
    seen, reports = [], []

    def answer(request):
        seen.append(f"{request.method} {request.url.path}")
        if request.url.path == "/jobs/claim":
            return httpx.Response(200, json=claim or {"id": "job-1"})
        if request.url.path == "/jobs/job-1/photo":
            with open(os.path.join(DATA, photo), "rb") as f:
                return httpx.Response(photo_status, content=f.read())
        reports.append(json.loads(request.content))
        return httpx.Response(report_status, json={})

    return httpx.Client(base_url="http://middle", transport=httpx.MockTransport(answer)), seen, reports


def test_worker_asks_for_the_photo_of_the_job_it_claimed(work):
    client, seen, reports = fake_server()

    assert process_one(client, stand_in, work) is True

    assert seen == ["POST /jobs/claim", "GET /jobs/job-1/photo", "POST /jobs/job-1/result"]
    assert reports[0]["label"] == "bobcat"


def test_model_reads_a_copy_in_the_working_folder(work):
    client, _, _ = fake_server()
    copies = []

    def look(image):
        copies.append(os.listdir(work))
        return stand_in(image)

    process_one(client, look, work)

    assert len(copies[0]) == 1 and copies[0][0].endswith(".copy")


def test_copy_is_deleted_after_a_good_job(work):
    client, _, _ = fake_server()
    process_one(client, stand_in, work)
    assert os.listdir(work) == []


def test_copy_is_deleted_after_a_broken_photo(work):
    client, _, reports = fake_server(photo="not_an_image.jpg")

    process_one(client, stand_in, work)

    assert "error" in reports[0]
    assert os.listdir(work) == []


def test_copy_is_deleted_when_the_model_crashes(work):
    client, _, reports = fake_server()

    process_one(client, lambda image: 1 / 0, work)

    assert "ZeroDivisionError" in reports[0]["error"]
    assert os.listdir(work) == []


def test_copy_is_deleted_when_the_report_is_refused(work):
    client, _, _ = fake_server(report_status=409)

    with pytest.raises(httpx.HTTPStatusError):
        process_one(client, stand_in, work)

    assert os.listdir(work) == []


def test_photo_that_does_not_come_fails_the_job_and_leaves_no_copy(work):
    client, _, reports = fake_server(photo_status=409)

    process_one(client, stand_in, work)

    assert "error" in reports[0]
    assert os.listdir(work) == []


def test_worker_never_touches_a_file_that_a_job_points_at(work, tmp_path):
    victim = tmp_path / "notes.txt"
    victim.write_text("not a photo")
    client, _, _ = fake_server(claim={"id": "job-1", "image_path": str(victim), "photo_name": str(victim)})

    process_one(client, stand_in, work)

    assert victim.read_text() == "not a photo"


def test_two_jobs_never_share_a_copy(work):
    client, _, _ = fake_server()
    names = []

    def look(image):
        names.extend(os.listdir(work))
        return stand_in(image)

    process_one(client, look, work)
    process_one(client, look, work)

    assert len(set(names)) == 2


def left_behind(folder, *names):
    for name in names:
        with open(os.path.join(folder, name), "wb") as f:
            f.write(b"from before the crash")


def test_start_deletes_the_copies_a_crash_left_behind(work):
    left_behind(work, "1.copy", "2.copy")

    delete_copies(work)

    assert os.listdir(work) == []


def test_start_deletes_nothing_but_working_copies(work):
    left_behind(work, "1.copy", "thesis.docx", "pawsort.db")
    os.mkdir(os.path.join(work, "photos"))
    left_behind(os.path.join(work, "photos"), "holiday.jpg", "3.copy")

    delete_copies(work)

    assert sorted(os.listdir(work)) == ["pawsort.db", "photos", "thesis.docx"]
    assert sorted(os.listdir(os.path.join(work, "photos"))) == ["3.copy", "holiday.jpg"]


def test_start_makes_the_working_folder_when_it_is_not_there(tmp_path):
    folder = str(tmp_path / "data" / "work")
    delete_copies(folder)
    assert os.listdir(folder) == []


def test_ai_server_deletes_old_copies_before_it_loads_the_models(work, monkeypatch):
    left_behind(work, "1.copy", "notes.txt")

    class Stop(Exception):
        pass

    def stop():
        raise Stop

    monkeypatch.setenv("PAWSORT_WORK", work)
    monkeypatch.setattr(detector, "load", stop)
    with pytest.raises(Stop):
        main()

    assert os.listdir(work) == ["notes.txt"]
