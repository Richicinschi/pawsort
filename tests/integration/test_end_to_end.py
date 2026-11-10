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
def test_real_photo_through_both_models(orch_client, copy_photo, predict, photo, label):
    job_id = orch_client.post("/jobs", json={"image_path": copy_photo(photo)}).json()["id"]
    process_one(orch_client, predict)
    assert orch_client.get(f"/jobs/{job_id}").json()["label"] == label


def test_real_model_reports_the_version_next_to_it(orch_client, copy_photo, predict):
    version = read_version("models/pawsort_wildlife.pt")
    job_id = orch_client.post("/jobs", json={"image_path": copy_photo("bobcat_1.jpg")}).json()["id"]
    process_one(orch_client, predict, version)
    assert orch_client.get(f"/jobs/{job_id}").json()["model_version"] == version


def test_real_model_gives_an_animal_a_second_guess_and_an_empty_photo_none(orch_client, copy_photo, predict):
    bobcat = orch_client.post("/jobs", json={"image_path": copy_photo("bobcat_1.jpg")}).json()["id"]
    empty = orch_client.post("/jobs", json={"image_path": copy_photo("empty_1.jpg")}).json()["id"]
    process_one(orch_client, predict)
    process_one(orch_client, predict)

    job = orch_client.get(f"/jobs/{bobcat}").json()
    assert job["second_label"] not in (None, "bobcat") and 0 < job["second_probability"] < job["probability"]
    assert orch_client.get(f"/jobs/{empty}").json()["second_label"] is None


def test_upload_on_the_website_to_answer_on_the_page(orch_client, tmp_path, predict):
    ui = TestClient(create_app(orch_client, str(tmp_path / "uploads")))
    with open(os.path.join(DATA, "bobcat_1.jpg"), "rb") as f:
        r = ui.post("/upload", data={"site_code": "A12"}, files=[("files", ("bobcat_1.jpg", f.read()))])
    job_id = r.json()["photos"][0]["job_id"]

    assert process_one(orch_client, predict) is True

    answer = ui.get(f"/result/{job_id}").json()
    assert (answer["status"], answer["label"]) == ("done", "bobcat")
    assert os.listdir(tmp_path / "uploads") == []
