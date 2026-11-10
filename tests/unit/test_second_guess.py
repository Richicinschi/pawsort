from ai_server.logic import decide, format_result


def classify(box):
    return {"raccoon": 0.57, "cat": 0.1234567, "skunk": 0.21}


def box(category="1", conf=0.9):
    return {"category": category, "conf": conf, "bbox": [0, 0, 1, 1]}


def second_guess(found):
    result = format_result(decide(found, classify))
    return result["second_label"], result["second_probability"]


def test_second_guess_is_the_next_most_likely_label():
    assert second_guess(box()) == ("skunk", 0.21)


def test_second_guess_does_not_change_the_first():
    result = format_result(decide(box(), classify))
    assert (result["label"], result["probability"], result["needs_review"]) == ("raccoon", 0.57, True)


def test_sure_tag_has_a_second_guess_too():
    result = format_result({"bobcat": 0.95, "coyote": 0.05})
    assert (result["needs_review"], result["second_label"]) == (False, "coyote")


def test_score_of_the_second_guess_is_rounded_like_the_first():
    assert format_result({"raccoon": 0.57, "cat": 0.1234567})["second_probability"] == 0.1235


def test_empty_has_no_second_guess():
    assert second_guess(None) == (None, None)


def test_person_has_no_second_guess():
    assert second_guess(box(category="2")) == (None, None)


def test_weak_box_has_no_second_guess():
    assert second_guess(box(conf=0.1)) == (None, None)
