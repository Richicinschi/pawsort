import pytest

from interface.app import PERSON_NOTE, percent, show_job, show_label, show_review_job


def done(label, review=None):
    return {"status": "done", "label": label, "probability": 0.6, "needs_review": 1, "review": review}


@pytest.mark.parametrize("probability,expected", [(0.7996, 79), (0.62, 62), (0.5, 50), (1.0, 100)])
def test_percentage_never_rounds_up_to_the_limit(probability, expected):
    assert percent(probability) == expected


def test_label_in_plain_words():
    assert show_label("mountain_lion") == "mountain lion"


def listed(label, probability, site_code="A12"):
    return {"id": "job-1", "site_code": site_code, "label": label, "probability": probability,
            "second_label": "bobcat", "second_probability": 0.15, "model_version": "v7"}


def test_list_row_keeps_the_real_label_for_the_answer():
    row = show_review_job(listed("mountain_lion", 0.7996))
    assert row == {"job_id": "job-1", "site_code": "A12", "label": "mountain_lion", "label_text": "mountain lion",
                   "percent": 79, "reason": "The model is less than 80% sure this is a mountain lion.",
                   "second_guess": "Its second guess is a bobcat, 15% sure.", "model_version": "v7"}


def test_list_row_without_a_site_code():
    assert show_review_job(listed("deer", 0.5, site_code=None))["site_code"] == "unknown"


def test_job_that_waits_for_a_person_shows_the_mark():
    assert show_job(done("bobcat")) == {
        "status": "done", "label": "bobcat", "percent": 60, "needs_review": True,
        "reason": "The model is less than 80% sure this is a bobcat.", "second_guess": None}


def test_checked_job_loses_the_mark_and_the_percentage():
    assert show_job(done("coyote", review={"new_label": "coyote"})) == {
        "status": "done", "label": "coyote", "needs_review": False, "reviewed": True}


def test_job_changed_to_person_gets_the_note():
    assert show_job(done("person", review={"new_label": "person"}))["note"] == PERSON_NOTE
