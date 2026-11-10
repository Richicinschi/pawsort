import json
import os
from datetime import date

import pytest
import torch

from ai_server.model import read_version
from training.switch_model import EXTRA, FILES, Refused, switch, targets
from training.train import describe, save_candidate

SITE = {"accuracy": 0.80, "accuracy_when_sure": 0.85, "sent_to_person": 0.25, "animals_called_empty": 0.10,
        "per_label": {"bobcat": 0.8, "rodent": 0.6}}
RESULTS = {"crops": 3500, "labels": ["bobcat", "rodent"], "check_accuracy": 0.9, "train_seconds": 12,
           "known_sites": dict(SITE), "new_sites": dict(SITE)}
CROPS = {"bobcat": 3000, "rodent": 500}
DATA = ["3500 crops: 3498 from data/cache and 2 from the corrections in data/corrections/corrections.csv"]


def save(folder, version="v2", results=RESULTS):
    save_candidate(str(folder), version, {"weight": torch.ones(2)}, ["bobcat", "rodent"], results, CROPS, DATA)
    return str(folder)


@pytest.fixture
def models(tmp_path):
    """A stand-in for the models folder with version v1, which scores what RESULTS says or a bit less."""
    folder = tmp_path / "models"
    folder.mkdir()
    for name in FILES:
        (folder / name).write_text(f"{name} of v1")
    (folder / "model_version.txt").write_text("v1\n")
    report = targets({**RESULTS, "label_crops": CROPS})
    entry = {"trained_on": "2026-11-06", "training_data": "crops for v1", "report": report}
    (folder / "versions.json").write_text(json.dumps({"current": "v1", "previous": None, "versions": {"v1": entry}}))
    return str(folder)


def test_candidate_folder_holds_exactly_what_the_switch_script_reads(tmp_path):
    folder = save(tmp_path / "new_model")
    assert sorted(os.listdir(folder)) == sorted(FILES + EXTRA)


def test_candidate_has_the_model_the_labels_and_the_version_from_the_command_line(tmp_path):
    folder = save(tmp_path / "new_model", "wildlife-1.1")

    assert torch.equal(torch.load(os.path.join(folder, "pawsort_wildlife.pt"))["weight"], torch.ones(2))
    with open(os.path.join(folder, "labels.json")) as f:
        assert json.load(f) == ["bobcat", "rodent"]
    assert read_version(os.path.join(folder, "pawsort_wildlife.pt")) == "wildlife-1.1"


def test_report_has_the_scores_the_training_date_and_the_crops_per_label(tmp_path):
    with open(os.path.join(save(tmp_path / "new_model"), "report.json")) as f:
        report = json.load(f)

    assert report == {**RESULTS, "trained_on": date.today().isoformat(), "label_crops": CROPS}
    assert targets(report)["lowest_recall_known_sites"] == 0.6


def test_training_data_file_says_what_the_model_was_trained_on(tmp_path):
    with open(os.path.join(save(tmp_path / "new_model"), "training_data.txt")) as f:
        assert f.read().splitlines() == DATA


def test_description_names_the_corrections_file_and_how_many_corrections_were_in_it():
    parts = {"train": 12377, "cis_val": 900, "sample": 29000, "corrections": 2}
    assert describe(parts, "data/cache", "data/corrections", 5) == [
        "42279 crops: 42277 from data/cache and 2 from the corrections in data/corrections/corrections.csv",
        "train: 12377 crops", "cis_val: 900 crops", "sample: 29000 crops",
        "corrections: 2 crops from the 5 corrections in data/corrections/corrections.csv"]


def test_description_of_a_training_without_corrections():
    assert describe({"train": 10, "sample": 5}, "data/cache", None, 0) == [
        "15 crops: 15 from data/cache and no corrections", "train: 10 crops", "sample: 5 crops"]


@pytest.mark.parametrize("folder", ["models", "models/", "models/versions/v2", "./models"])
def test_saving_into_the_models_folder_is_refused_with_a_pointer_to_the_switch_script(folder, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("models")
    (tmp_path / "models" / "pawsort_wildlife.pt").write_text("the live model")

    with pytest.raises(SystemExit) as refusal:
        save(folder)

    assert str(refusal.value) == (
        "Refused: a training never writes into models, because a new model has to pass the switch first. "
        "Save it into a folder of its own, for example --save data/new_model, "
        "and then run python training/switch_model.py data/new_model")
    assert (tmp_path / "models" / "pawsort_wildlife.pt").read_text() == "the live model"
    assert os.listdir(tmp_path / "models") == ["pawsort_wildlife.pt"]


@pytest.mark.parametrize("version", [None, "", "  "])
def test_saving_without_a_version_name_is_refused(tmp_path, version):
    with pytest.raises(SystemExit, match="--version"):
        save(tmp_path / "new_model", version)
    assert not os.path.exists(tmp_path / "new_model")


def test_switch_script_accepts_a_saved_candidate_that_scores_the_same(tmp_path, models):
    message = switch(save(tmp_path / "new_model"), models)

    assert message.startswith("Switched from v1 to v2.")
    assert read_version(os.path.join(models, "pawsort_wildlife.pt")) == "v2"
    with open(os.path.join(models, "versions.json")) as f:
        entry = json.load(f)["versions"]["v2"]
    assert (entry["trained_on"], entry["training_data"]) == (date.today().isoformat(), DATA[0])


def test_switch_script_refuses_a_saved_candidate_that_scores_worse(tmp_path, models):
    worse = {**RESULTS, "known_sites": {**SITE, "accuracy": 0.5}}
    with pytest.raises(Refused, match="accuracy on known sites"):
        switch(save(tmp_path / "new_model", results=worse), models)
    assert read_version(os.path.join(models, "pawsort_wildlife.pt")) == "v1"
