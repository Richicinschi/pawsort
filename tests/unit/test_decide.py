from ai_server.logic import decide


def classify(box):
    return {"bobcat": 0.9, "coyote": 0.1}


def test_no_box_means_empty():
    assert decide(None, classify) == {"empty": 1.0}


def test_person_is_never_named_as_an_animal():
    assert decide({"category": "2", "bbox": [0, 0, 1, 1]}, classify) == {"person": 1.0}


def test_animal_box_goes_to_the_classifier():
    assert decide({"category": "1", "bbox": [0, 0, 1, 1]}, classify) == {"bobcat": 0.9, "coyote": 0.1}
