import csv
import os

import httpx
import pytest
from PIL import Image

from training.export_corrections import export

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@pytest.fixture
def corrected(orch_client, send_photo):
    """A job whose tag a person changed from bobcat to the given label. With a decision, it also gets that."""
    def _corrected(new_label="coyote", decision=None, name="bobcat_1.jpg"):
        job_id = send_photo(orch_client, name, "A12")
        orch_client.post("/jobs/claim")
        orch_client.post(f"/jobs/{job_id}/result", json={"label": "bobcat", "probability": 0.6, "needs_review": True,
                                                         "model_version": "wildlife-9"})
        orch_client.post(f"/jobs/{job_id}/review", json={"label": new_label})
        if decision:
            orch_client.post(f"/corrections/{job_id}", json={"decision": decision})
        return job_id
    return _corrected


@pytest.fixture
def out(tmp_path):
    return tmp_path / "corrections"


def listed(out):
    with open(out / "corrections.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_export_fetches_the_photo_and_writes_the_list(orch_client, corrected, out):
    job_id = corrected(decision="approved", name="coyote_1.jpg")

    assert export(orch_client, str(out)) == 1

    assert listed(out) == [{"job": job_id, "photo": f"{job_id}.jpg", "label": "coyote", "model_version": "wildlife-9"}]
    with open(os.path.join(DATA, "coyote_1.jpg"), "rb") as f:
        assert (out / f"{job_id}.jpg").read_bytes() == f.read()


def test_an_unapproved_or_rejected_correction_can_never_be_in_that_folder(orch_client, corrected, out):
    approved = corrected(decision="approved")
    not_decided, rejected, confirmed = corrected("deer"), corrected("skunk", "rejected"), corrected("bobcat")

    export(orch_client, str(out))

    assert sorted(os.listdir(out)) == sorted(["corrections.csv", f"{approved}.jpg"])
    assert [row["job"] for row in listed(out)] == [approved]
    for job_id in (not_decided, rejected, confirmed):
        assert job_id not in (out / "corrections.csv").read_text()


def test_folder_is_made_and_gets_only_the_list_when_nothing_is_approved(orch_client, corrected, out):
    corrected()

    assert export(orch_client, str(out)) == 0

    assert os.listdir(out) == ["corrections.csv"]
    assert listed(out) == []


def test_second_export_does_not_fetch_the_same_correction_again(orch_client, corrected, out, tmp_path):
    corrected(decision="approved")
    export(orch_client, str(out))

    assert export(orch_client, str(tmp_path / "next")) == 0
    assert orch_client.get("/training/corrections").json()["jobs"] == []


def test_later_export_into_the_same_folder_adds_to_the_list(orch_client, corrected, out):
    first = corrected(decision="approved")
    export(orch_client, str(out))
    second = corrected("deer", "approved")

    export(orch_client, str(out))

    assert [(row["job"], row["label"]) for row in listed(out)] == [(first, "coyote"), (second, "deer")]


def test_correction_whose_photo_is_gone_is_left_out_and_not_counted_as_taken(orch_client, corrected, out, photos):
    gone, there = corrected(decision="approved"), corrected("deer", "approved")
    os.remove(photos / orch_client.get(f"/jobs/{gone}").json()["photo_name"])

    assert export(orch_client, str(out)) == 1

    assert [row["job"] for row in listed(out)] == [there]
    assert orch_client.get(f"/jobs/{gone}").json()["review"]["used_at"] is None
    assert orch_client.get(f"/jobs/{there}").json()["review"]["used_at"] is not None


def test_png_keeps_its_ending_in_the_folder(orch_client, out, tmp_path):
    path = tmp_path / "made.png"
    Image.new("RGB", (200, 200)).save(path)
    job_id = orch_client.post("/jobs", files={"photo": ("photo", path.read_bytes(), "image/png")}).json()["id"]
    orch_client.post("/jobs/claim")
    orch_client.post(f"/jobs/{job_id}/result", json={"label": "bobcat", "probability": 0.6, "needs_review": True})
    orch_client.post(f"/jobs/{job_id}/review", json={"label": "coyote"})
    orch_client.post(f"/corrections/{job_id}", json={"decision": "approved"})

    export(orch_client, str(out))

    assert listed(out)[0]["photo"] == f"{job_id}.png"
    assert (out / f"{job_id}.png").read_bytes() == path.read_bytes()


def stop_at_photo(orch, number, monkeypatch):
    """Lets the middle server go away when the export asks for the photo with this number."""
    real, asked = orch.get, []

    def get(address, **more):
        if address.endswith("/photo"):
            asked.append(address)
            if len(asked) == number:
                raise RuntimeError("the power went off")
        return real(address, **more)

    monkeypatch.setattr(orch, "get", get)


def test_export_that_stops_still_says_what_it_took_and_the_next_run_adds_no_doubles(orch_client, corrected, out,
                                                                                    monkeypatch):
    first, second = corrected(decision="approved"), corrected("deer", "approved")
    stop_at_photo(orch_client, 2, monkeypatch)

    with pytest.raises(RuntimeError):
        export(orch_client, str(out))

    assert [row["job"] for row in listed(out)] == [first]
    assert [job["id"] for job in orch_client.get("/training/corrections").json()["jobs"]] == [second]
    monkeypatch.undo()

    assert export(orch_client, str(out)) == 1

    assert [row["job"] for row in listed(out)] == [first, second]
    assert sorted(os.listdir(out)) == sorted(["corrections.csv", f"{first}.jpg", f"{second}.jpg"])


def test_job_that_is_already_in_the_list_gets_no_second_row(orch_client, corrected, out, monkeypatch):
    job_id = corrected(decision="approved")
    monkeypatch.setattr(orch_client, "post", lambda address, **more: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        export(orch_client, str(out))
    assert orch_client.get(f"/jobs/{job_id}").json()["review"]["used_at"] is None
    monkeypatch.undo()

    assert export(orch_client, str(out)) == 0

    assert [row["job"] for row in listed(out)] == [job_id]
    assert orch_client.get(f"/jobs/{job_id}").json()["review"]["used_at"] is not None


@pytest.mark.parametrize("job", ["../../outside", "nope", "ABCDEF12-0000-4000-8000-000000000000", ""])
def test_answer_of_the_server_only_becomes_a_file_name_when_it_is_a_job_number(out, tmp_path, job):
    asked = []

    def answer(request):
        asked.append(request.url.path)
        if request.url.path == "/training/corrections":
            return httpx.Response(200, json={"jobs": [{"id": job, "label": "coyote", "model_version": "v7"}]})
        return httpx.Response(200, content=b"photo", headers={"content-type": "image/jpeg"})

    client = httpx.Client(base_url="http://middle", transport=httpx.MockTransport(answer))

    assert export(client, str(out)) == 0

    assert asked == ["/training/corrections", "/training/corrections/used"]
    assert os.listdir(out) == ["corrections.csv"] and listed(out) == []
    assert not list(tmp_path.parent.rglob("outside*"))
