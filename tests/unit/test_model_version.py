import json
import os

import httpx
import pytest

from ai_server.model import read_version
from ai_server.worker import process_one

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def test_version_comes_from_the_file_next_to_the_model(tmp_path):
    (tmp_path / "model_version.txt").write_text("v2\n", encoding="utf-8")
    assert read_version(str(tmp_path / "other_model.pt")) == "v2"


def test_missing_version_file_stops_the_start(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_version(str(tmp_path / "model.pt"))


def test_empty_version_file_stops_the_start(tmp_path):
    (tmp_path / "model_version.txt").write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_version(str(tmp_path / "model.pt"))


def test_the_repo_has_a_version_next_to_the_model():
    assert read_version("models/pawsort_wildlife.pt")


def test_labels_file_stays_a_plain_list_of_labels():
    with open("models/labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    assert len(labels) == 18 and all(isinstance(label, str) for label in labels)


def report_of_one_job(photo, version, work):
    """Runs one job against a middle server that only exists in this test, and gives back what the worker sent."""
    sent = []

    def answer(request):
        if request.url.path == "/jobs/claim":
            return httpx.Response(200, json={"id": "job-1"})
        if request.url.path == "/jobs/job-1/photo":
            with open(os.path.join(DATA, photo), "rb") as f:
                return httpx.Response(200, content=f.read())
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={})

    client = httpx.Client(base_url="http://middle", transport=httpx.MockTransport(answer))
    process_one(client, lambda image: {"bobcat": 0.9, "coyote": 0.1}, work, version)
    return sent[0]


def test_worker_sends_the_version_with_a_good_result(work):
    body = report_of_one_job("bobcat_1.jpg", "v2", work)
    assert (body["label"], body["model_version"]) == ("bobcat", "v2")


def test_worker_sends_the_version_with_a_failed_result(work):
    body = report_of_one_job("not_an_image.jpg", "v2", work)
    assert "error" in body and body["model_version"] == "v2"
