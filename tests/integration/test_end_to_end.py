import os

import pytest
from fastapi.testclient import TestClient

from ai_server import detector
from ai_server.logic import decide
from ai_server.model import crop_to_box, load_model, predict_scores, preprocess, read_version
from ai_server.worker import process_one
from interface.app import create_app

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@pytest.fixture(scope="module")
def predict():
    finder = detector.load()
    model, labels = load_model("models/pawsort_wildlife.pt", "models/labels.json")

    def run(image):
        def classify(box):
            return predict_scores(model, labels, preprocess(crop_to_box(image, box)))
        return decide(detector.find(finder, image), classify)
    return run


@pytest.mark.parametrize("photo,label", [("bobcat_1.jpg", "bobcat"), ("coyote_1.jpg", "coyote"), ("empty_1.jpg", "empty")])
def test_real_photo_through_both_models(orch_client, send_photo, work, predict, photo, label):
    job_id = send_photo(orch_client, photo)
    process_one(orch_client, predict, work)
    assert orch_client.get(f"/jobs/{job_id}").json()["label"] == label


def test_real_model_reports_the_version_next_to_it(orch_client, send_photo, work, predict):
    version = read_version("models/pawsort_wildlife.pt")
    job_id = send_photo(orch_client)
    process_one(orch_client, predict, work, version)
    assert orch_client.get(f"/jobs/{job_id}").json()["model_version"] == version


def test_real_model_gives_an_animal_a_second_guess_and_an_empty_photo_none(orch_client, send_photo, work, predict):
    bobcat, empty = send_photo(orch_client, "bobcat_1.jpg"), send_photo(orch_client, "empty_1.jpg")
    process_one(orch_client, predict, work)
    process_one(orch_client, predict, work)

    job = orch_client.get(f"/jobs/{bobcat}").json()
    assert job["second_label"] not in (None, "bobcat") and 0 < job["second_probability"] < job["probability"]
    assert orch_client.get(f"/jobs/{empty}").json()["second_label"] is None


def test_upload_on_the_website_to_answer_on_the_page(orch_client, photos, work, predict):
    ui = TestClient(create_app(orch_client))
    with open(os.path.join(DATA, "bobcat_1.jpg"), "rb") as f:
        r = ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("bobcat_1.jpg", f.read()))])
    job_id = r.json()["photos"][0]["job_id"]

    assert process_one(orch_client, predict, work) is True

    answer = ui.get(f"/result/{job_id}").json()
    assert (answer["status"], answer["label"]) == ("done", "bobcat")
    assert os.listdir(work) == []
    assert os.listdir(photos) == [orch_client.get(f"/jobs/{job_id}").json()["photo_name"]]
