from interface.app import PERSON_NOTE, show_job


def done(label, probability, needs_review=False):
    return {"status": "done", "label": label, "probability": probability, "needs_review": int(needs_review)}


def test_person_gets_the_note():
    assert show_job(done("person", 1.0)) == {
        "status": "done", "label": "person", "percent": 100, "needs_review": False, "note": PERSON_NOTE}


def test_note_says_the_photo_is_deleted():
    assert "deleted" in PERSON_NOTE


def test_animal_gets_no_note():
    assert "note" not in show_job(done("bobcat", 0.9))


def test_label_is_shown_in_plain_words():
    assert show_job(done("mountain_lion", 0.87))["label"] == "mountain lion"


def test_failed_job_never_shows_the_technical_reason():
    answer = show_job({"status": "failed", "error": "no answer within 30 seconds"})
    assert answer == {"status": "failed", "error": "We could not check this photo. Please try another one."}


def test_job_still_working_shows_only_the_state():
    assert show_job({"status": "processing"}) == {"status": "processing"}
