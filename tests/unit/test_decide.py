from ai_server.logic import decide, format_result


def classify(box):
    return {"bobcat": 0.9, "coyote": 0.1}


def test_no_box_means_empty():
    assert decide(None, classify) == {"empty": 1.0}


def test_person_is_never_named_as_an_animal():
    assert decide({"category": "2", "conf": 0.9, "bbox": [0, 0, 1, 1]}, classify) == {"person": 1.0}


def test_animal_box_goes_to_the_classifier():
    assert decide({"category": "1", "conf": 0.9, "bbox": [0, 0, 1, 1]}, classify) == {"bobcat": 0.9, "coyote": 0.1}


def test_weak_box_is_empty_but_goes_to_a_person():
    scores = decide({"category": "1", "conf": 0.19, "bbox": [0, 0, 1, 1]}, classify)
    assert format_result(scores) == {"label": "empty", "probability": 0.5, "needs_review": True}


def test_box_of_exactly_the_limit_is_not_weak():
    assert decide({"category": "1", "conf": 0.2, "bbox": [0, 0, 1, 1]}, classify) == {"bobcat": 0.9, "coyote": 0.1}


def test_weak_box_of_a_person_also_goes_to_a_person():
    assert decide({"category": "2", "conf": 0.1, "bbox": [0, 0, 1, 1]}, classify) == {"empty": 0.5}
