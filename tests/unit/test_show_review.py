import pytest

from interface.app import PERSON_NOTE, percent, show_job, show_label, show_review_job


def done(label, review=None):
    return {"status": "done", "label": label, "probability": 0.6, "needs_review": 1, "review": review}


@pytest.mark.parametrize("probability,expected", [(0.7996, 79), (0.62, 62), (0.5, 50), (1.0, 100)])
def test_percentage_never_rounds_up_to_the_limit(probability, expected):
    assert percent(probability) == expected


def test_label_in_plain_words():
    assert show_label("mountain_lion") == "mountain lion"


def test_list_row_keeps_the_real_label_for_the_answer():
    row = show_review_job({"id": "job-1", "site_code": "A12", "label": "mountain_lion", "probability": 0.7996})
    assert row == {"job_id": "job-1", "site_code": "A12", "label": "mountain_lion", "label_text": "mountain lion",
                   "percent": 79}


def test_list_row_without_a_site_code():
    row = show_review_job({"id": "job-1", "site_code": None, "label": "deer", "probability": 0.5})
    assert row["site_code"] == "unknown"


def test_job_that_waits_for_a_person_shows_the_mark():
    assert show_job(done("bobcat")) == {"status": "done", "label": "bobcat", "percent": 60, "needs_review": True}


def test_checked_job_loses_the_mark_and_the_percentage():
    assert show_job(done("coyote", review={"new_label": "coyote"})) == {
        "status": "done", "label": "coyote", "needs_review": False, "reviewed": True}


def test_job_changed_to_person_gets_the_note():
    assert show_job(done("person", review={"new_label": "person"}))["note"] == PERSON_NOTE
