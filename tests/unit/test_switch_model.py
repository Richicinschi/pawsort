import json
import os
import shutil

import pytest

from ai_server.model import read_version
from training.switch_model import FILES, TARGETS, Refused, back, switch, targets

SITE = {"accuracy": 0.80, "accuracy_when_sure": 0.85, "sent_to_person": 0.25, "animals_called_empty": 0.10,
        "per_label": {"bobcat": 0.8, "rodent": 0.6, "badger": 0.1}}
CROPS = {"bobcat": 3000, "rodent": 500, "badger": 499}


def run(**changes):
    """What a training run writes: the same scores for known and new sites, with the changes for the known sites."""
    return {"trained_on": "2026-11-12", "label_crops": CROPS, "known_sites": {**SITE, **changes},
            "new_sites": dict(SITE)}


def write(folder, version, report):
    folder.mkdir()
    for name in FILES:
        (folder / name).write_text(f"{name} of {version}")
    (folder / "model_version.txt").write_text(f"{version}\n")
    (folder / "report.json").write_text(json.dumps(report))
    (folder / "training_data.txt").write_text(f"crops for {version}\nphoto-1.jpg\nphoto-2.jpg\n")
    return str(folder)


@pytest.fixture
def models(tmp_path):
    """A models folder with stand-in files for the version v1, the way the repo has it for the real model."""
    folder = write(tmp_path / "models", "v1", {})
    for name in ("report.json", "training_data.txt"):
        os.remove(os.path.join(folder, name))
    entry = {"trained_on": "2026-11-06", "training_data": "crops for v1", "report": targets(run())}
    with open(os.path.join(folder, "versions.json"), "w") as f:
        json.dump({"current": "v1", "previous": None, "versions": {"v1": entry}}, f)
    return folder


def on_disk(folder):
    found = {}
    for path, _, names in os.walk(folder):
        for name in names:
            with open(os.path.join(path, name), "rb") as f:
                found[os.path.relpath(os.path.join(path, name), folder)] = f.read()
    return found


def versions(models):
    with open(os.path.join(models, "versions.json")) as f:
        return json.load(f)


WORSE = {
    "accuracy_known_sites": {"accuracy": 0.79},
    "accuracy_when_sure": {"accuracy_when_sure": 0.84},
    "sure_known_sites": {"sent_to_person": 0.26},
    "animals_called_empty_known_sites": {"animals_called_empty": 0.11},
    "lowest_recall_known_sites": {"per_label": {"bobcat": 0.8, "rodent": 0.59, "badger": 0.1}},
}


@pytest.mark.parametrize("target", WORSE)
def test_version_that_is_worse_on_one_target_is_refused_and_nothing_changes_on_disk(models, tmp_path, target):
    candidate = write(tmp_path / "new", "v2", run(**WORSE[target]))
    before = on_disk(models)

    with pytest.raises(Refused) as refusal:
        switch(candidate, models)

    words = next(words for name, words, _ in TARGETS if name == target)
    assert words in str(refusal.value)
    assert on_disk(models) == before


def test_refusal_names_the_first_target_that_is_worse_with_both_numbers(models, tmp_path):
    candidate = write(tmp_path / "new", "v2", run(accuracy=0.79, accuracy_when_sure=0.5))
    with pytest.raises(Refused) as refusal:
        switch(candidate, models)
    assert str(refusal.value) == ("Refused: the accuracy on known sites (AI-01) is 79.0% for the new model "
                                  "and 80.0% for the current one.")


def test_version_that_scores_the_same_is_accepted(models, tmp_path):
    message = switch(write(tmp_path / "new", "v2", run()), models)
    assert message == "Switched from v1 to v2. Start the AI server again, because it reads the model when it starts."
    assert versions(models)["current"] == "v2"


def test_version_that_scores_better_is_accepted(models, tmp_path):
    better = run(accuracy=0.9, accuracy_when_sure=0.95, sent_to_person=0.1, animals_called_empty=0.05)
    switch(write(tmp_path / "new", "v2", better), models)
    assert versions(models)["current"] == "v2"


def test_accepted_version_is_in_place_and_the_ai_server_reads_its_version(models, tmp_path):
    switch(write(tmp_path / "new", "v2", run()), models)

    for name in ("pawsort_wildlife.pt", "labels.json"):
        with open(os.path.join(models, name)) as f:
            assert f.read() == f"{name} of v2"
    assert read_version(os.path.join(models, "pawsort_wildlife.pt")) == "v2"


def test_accepted_version_gets_its_entry_with_date_training_data_and_report(models, tmp_path):
    switch(write(tmp_path / "new", "v2", run(accuracy=0.9)), models)

    saved = versions(models)
    assert (saved["current"], saved["previous"], sorted(saved["versions"])) == ("v2", "v1", ["v1", "v2"])
    entry = saved["versions"]["v2"]
    assert entry == {"trained_on": "2026-11-12", "training_data": "crops for v2",
                     "report": targets(run(accuracy=0.9))}
    with open(os.path.join(models, "versions", "v2", "training_data.txt")) as f:
        assert f.read().splitlines() == ["crops for v2", "photo-1.jpg", "photo-2.jpg"]


def test_switch_back_restores_the_exact_files(models, tmp_path):
    before = {name: on_disk(models)[name] for name in FILES}
    switch(write(tmp_path / "new", "v2", run()), models)
    assert {name: on_disk(models)[name] for name in FILES} != before

    message = back(models)

    assert message == ("Switched back from v2 to v1. Start the AI server again, because it reads the model when "
                       "it starts.")
    assert {name: on_disk(models)[name] for name in FILES} == before
    saved = versions(models)
    assert (saved["current"], saved["previous"], sorted(saved["versions"])) == ("v1", "v2", ["v1", "v2"])


def test_after_a_switch_back_the_newer_version_can_come_back_too(models, tmp_path):
    switch(write(tmp_path / "new", "v2", run()), models)
    back(models)
    back(models)
    assert read_version(os.path.join(models, "pawsort_wildlife.pt")) == "v2"


def test_switch_back_without_a_version_before_is_refused(models):
    before = on_disk(models)
    with pytest.raises(Refused, match="no version before this one"):
        back(models)
    assert on_disk(models) == before


def test_third_version_is_compared_with_the_one_in_place(models, tmp_path):
    switch(write(tmp_path / "new", "v2", run(accuracy=0.9)), models)
    with pytest.raises(Refused, match="85.0% for the new model and 90.0% for the current one"):
        switch(write(tmp_path / "newer", "v3", run(accuracy=0.85)), models)


def test_report_with_a_missing_target_is_refused(models, tmp_path):
    report = run()
    del report["label_crops"]
    candidate = write(tmp_path / "new", "v2", report)
    before = on_disk(models)

    with pytest.raises(Refused) as refusal:
        switch(candidate, models)

    assert str(refusal.value) == ("Refused: the lowest recall of a label with 500 or more training crops, known sites "
                                  "(AI-06) is measured for only one of the two models, so we cannot compare them.")
    assert on_disk(models) == before


def test_target_that_only_the_new_version_measured_is_refused_too(models, tmp_path):
    saved = versions(models)
    saved["versions"]["v1"]["report"]["accuracy_when_sure"] = None
    with open(os.path.join(models, "versions.json"), "w") as f:
        json.dump(saved, f)

    with pytest.raises(Refused, match="measured for only one of the two models"):
        switch(write(tmp_path / "new", "v2", run()), models)


def test_target_that_neither_version_measured_is_left_out(models, tmp_path):
    saved = versions(models)
    saved["versions"]["v1"]["report"].update(lowest_recall_known_sites=None, lowest_recall_new_sites=None)
    with open(os.path.join(models, "versions.json"), "w") as f:
        json.dump(saved, f)
    report = run()
    del report["label_crops"]

    switch(write(tmp_path / "new", "v2", report), models)

    assert versions(models)["current"] == "v2"


@pytest.mark.parametrize("name", FILES + ("report.json", "training_data.txt"))
def test_folder_with_a_missing_file_is_refused(models, tmp_path, name):
    candidate = write(tmp_path / "new", "v2", run())
    os.remove(os.path.join(candidate, name))
    before = on_disk(models)

    with pytest.raises(Refused, match=f"{name} is missing"):
        switch(candidate, models)
    assert on_disk(models) == before


def test_new_model_needs_a_version_of_its_own(models, tmp_path):
    candidate = write(tmp_path / "new", "v1", run(accuracy=0.9))
    before = on_disk(models)

    with pytest.raises(Refused, match="needs a version of its own"):
        switch(candidate, models)
    assert on_disk(models) == before


@pytest.mark.parametrize("version", ["../../outside", "a/b", "a\\b", "v 2", ".hidden", "", "v2;x"])
def test_version_that_is_no_plain_name_is_refused_and_writes_nothing(models, tmp_path, version):
    candidate = write(tmp_path / "new", version, run(accuracy=0.9))
    before = on_disk(tmp_path)

    with pytest.raises(Refused, match="is not a version name"):
        switch(candidate, models)
    assert on_disk(tmp_path) == before


def test_report_without_the_training_date_is_refused(models, tmp_path):
    report = run()
    del report["trained_on"]
    with pytest.raises(Refused, match="when the new model was trained"):
        switch(write(tmp_path / "new", "v2", report), models)


def test_lowest_recall_only_looks_at_labels_with_500_or_more_training_crops():
    report = targets(run())
    assert (report["lowest_recall_known_sites"], report["lowest_recall_new_sites"]) == (0.6, 0.6)


def test_report_has_a_number_for_every_target():
    assert targets(run()) == {
        "accuracy_known_sites": 0.8, "accuracy_new_sites": 0.8, "accuracy_when_sure": 0.85, "sure_known_sites": 0.75,
        "sure_new_sites": 0.75, "animals_called_empty_known_sites": 0.1, "animals_called_empty_new_sites": 0.1,
        "lowest_recall_known_sites": 0.6, "lowest_recall_new_sites": 0.6}
    assert list(targets(run())) == [name for name, _, _ in TARGETS]


def stop_at_copy(number, monkeypatch):
    """Lets the computer go off at this copy of a file, counted over the whole switch."""
    real, copies = shutil.copy2, []

    def copy(source, target):
        copies.append(target)
        if len(copies) == number:
            raise OSError("the power went off")
        return real(source, target)

    monkeypatch.setattr(shutil, "copy2", copy)


@pytest.mark.parametrize("copy", range(1, 12))
def test_switch_that_stops_at_any_file_and_runs_again_ends_complete_and_correct(models, tmp_path, monkeypatch, copy):
    old = {name: on_disk(models)[name] for name in FILES}
    candidate = write(tmp_path / "new", "v2", run())
    new = {name: on_disk(candidate)[name] for name in FILES}
    stop_at_copy(copy, monkeypatch)
    with pytest.raises(OSError):
        switch(candidate, models)
    assert versions(models)["current"] == "v1"
    monkeypatch.undo()

    switch(candidate, models)

    assert {name: on_disk(models)[name] for name in FILES} == new
    assert (versions(models)["current"], versions(models)["previous"]) == ("v2", "v1")
    assert not [name for name in on_disk(models) if ".tmp" in name]
    back(models)
    assert {name: on_disk(models)[name] for name in FILES} == old


def test_a_twelfth_copy_does_not_exist_so_the_test_above_stops_at_every_file(models, tmp_path, monkeypatch):
    stop_at_copy(12, monkeypatch)
    switch(write(tmp_path / "new", "v2", run()), models)
    assert versions(models)["current"] == "v2"


def test_version_that_is_kept_is_never_written_again(models, tmp_path):
    switch(write(tmp_path / "new", "v2", run()), models)
    back(models)
    kept = {name: data for name, data in on_disk(models).items() if name.startswith("versions" + os.sep)}
    with open(os.path.join(models, "pawsort_wildlife.pt"), "w") as f:
        f.write("half a file from a switch that stopped")

    switch(write(tmp_path / "newer", "v3", run()), models)

    after = on_disk(models)
    assert {name: after[name] for name in kept} == kept
    back(models)
    assert on_disk(models)["pawsort_wildlife.pt"] == b"pawsort_wildlife.pt of v1"


def test_kept_folder_with_other_files_under_the_same_name_is_refused(models, tmp_path):
    candidate = write(tmp_path / "new", "v2", run())
    os.makedirs(os.path.join(models, "versions"))
    write(tmp_path / "models" / "versions" / "v2", "v2", run(accuracy=0.99))
    before = on_disk(models)

    with pytest.raises(Refused, match="already holds other files"):
        switch(candidate, models)
    assert on_disk(models) == before


def test_versions_file_is_never_left_empty_or_half_written(models, tmp_path, monkeypatch):
    before = versions(models)

    def stops_while_writing(data, f, **more):
        f.write('{"current": "v2", "prev')
        raise OSError("the power went off")

    monkeypatch.setattr(json, "dump", stops_while_writing)
    with pytest.raises(OSError):
        switch(write(tmp_path / "new", "v2", run()), models)
    monkeypatch.undo()

    assert versions(models) == before
    switch(str(tmp_path / "new"), models)
    assert versions(models)["current"] == "v2"


@pytest.mark.parametrize("changes", [{"accuracy": 82.3}, {"accuracy": float("nan")}, {"accuracy": -0.1},
                                     {"animals_called_empty": 9.9}, {"sent_to_person": 21.1}])
def test_report_with_a_number_outside_0_to_1_is_refused(models, tmp_path, changes):
    candidate = write(tmp_path / "new", "v2", run(**changes))
    before = on_disk(models)

    with pytest.raises(Refused, match="only has numbers from 0 to 1"):
        switch(candidate, models)
    assert on_disk(models) == before


def test_the_repo_lists_the_model_it_ships_with_the_numbers_of_results_json():
    with open("models/versions.json", encoding="utf-8") as f:
        saved = json.load(f)
    with open("results.json", encoding="utf-8") as f:
        final = json.load(f)["final_model"]

    assert saved["current"] == read_version("models/pawsort_wildlife.pt")
    entry = saved["versions"][saved["current"]]
    assert entry["report"] == targets(final)
    assert all(os.path.exists(os.path.join("models", name)) for name in FILES)
    assert (entry["report"]["accuracy_known_sites"], entry["report"]["accuracy_new_sites"]) == (0.8234, 0.7383)
