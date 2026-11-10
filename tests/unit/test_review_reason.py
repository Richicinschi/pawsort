from ai_server.logic import THRESHOLD
from interface.app import second_guess, show_job, show_review_job, why_review

RACCOON = "The model is less than 80% sure this is a raccoon."
WEAK_BOX = "The model saw something in this photo, but not clearly enough to say what it is."
SKUNK = "Its second guess is a skunk, 21% sure."


def unsure(label="raccoon", probability=0.57, second_label="skunk", second_probability=0.2149, **more):
    return {"status": "done", "label": label, "probability": probability, "needs_review": 1,
            "second_label": second_label, "second_probability": second_probability, **more}


def test_reason_names_the_animal_and_the_80_percent():
    assert why_review("raccoon") == RACCOON


def test_reason_says_an_before_a_vowel():
    assert why_review("opossum") == "The model is less than 80% sure this is an opossum."


def test_reason_shows_the_label_in_plain_words():
    assert why_review("mountain_lion") == "The model is less than 80% sure this is a mountain lion."


def test_reason_names_the_limit_the_ai_server_uses():
    assert f"less than {THRESHOLD:.0%} sure" in why_review("deer")


def test_weak_box_gets_its_own_reason():
    assert why_review("empty") == WEAK_BOX


def test_second_guess_names_the_label_and_its_percentage():
    assert second_guess(unsure()) == SKUNK


def test_second_guess_shows_the_label_in_plain_words():
    assert second_guess(unsure(second_label="mountain_lion")) == "Its second guess is a mountain lion, 21% sure."


def test_job_without_a_second_guess_gets_no_text_for_it():
    assert second_guess(unsure(second_label=None, second_probability=None)) is None
    assert second_guess({"status": "done", "label": "cat", "probability": 0.6, "needs_review": 1}) is None


def test_job_that_needs_review_says_why_and_shows_the_second_guess():
    assert show_job(unsure()) == {"status": "done", "label": "raccoon", "percent": 57, "needs_review": True,
                                  "reason": RACCOON, "second_guess": SKUNK}


def test_weak_box_job_says_its_own_reason_and_has_no_second_guess():
    answer = show_job(unsure("empty", 0.5, None, None))
    assert (answer["reason"], answer["second_guess"]) == (WEAK_BOX, None)


def test_sure_tag_gets_no_reason_and_no_second_guess():
    job = {**unsure(probability=0.93, second_probability=0.04), "needs_review": 0}
    assert show_job(job) == {"status": "done", "label": "raccoon", "percent": 93, "needs_review": False}


def test_checked_job_gets_no_reason_any_more():
    assert show_job(unsure(review={"new_label": "raccoon"})) == {
        "status": "done", "label": "raccoon", "needs_review": False, "reviewed": True}


def test_review_row_says_why_and_shows_the_second_guess_and_the_model_version():
    row = show_review_job(unsure(id="job-1", site_code="A12", model_version="wildlife-1.0"))
    assert (row["reason"], row["second_guess"], row["model_version"]) == (RACCOON, SKUNK, "wildlife-1.0")


def test_review_row_of_a_job_without_a_version_says_unknown():
    row = show_review_job(unsure(id="job-1", site_code="A12", model_version=None))
    assert row["model_version"] == "unknown"
