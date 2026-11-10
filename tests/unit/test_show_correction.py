import pytest

from interface.app import show_correction, training_note


def test_correction_row_shows_both_labels_in_plain_words_and_the_version():
    row = show_correction({"id": "job-1", "site_code": "A12", "old_label": "mountain_lion", "new_label": "bobcat",
                           "model_version": "wildlife-1.0"})
    assert row == {"job_id": "job-1", "site_code": "A12", "old_label": "mountain lion", "new_label": "bobcat",
                   "model_version": "wildlife-1.0"}


def test_correction_row_without_a_site_code_or_a_version():
    row = show_correction({"id": "job-1", "site_code": None, "old_label": "cat", "new_label": "bobcat",
                           "model_version": None})
    assert (row["site_code"], row["model_version"]) == ("unknown", "unknown")


@pytest.mark.parametrize("waiting,text", [
    (0, "0 approved corrections wait for the next training."),
    (1, "1 approved correction waits for the next training."),
    (499, "499 approved corrections wait for the next training."),
])
def test_note_counts_the_approved_corrections(waiting, text):
    assert training_note({"for_training": waiting, "plan_training": False}) == text


def test_note_says_when_it_is_time_to_plan_a_training_round():
    assert training_note({"for_training": 500, "plan_training": True}) == (
        "500 approved corrections wait for the next training. It is time to plan a training round.")
