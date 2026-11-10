import pytest

from ai_server.logic import format_result


@pytest.mark.parametrize("scores,label,prob,flag,second,second_prob", [
    ({"bobcat": 0.90, "coyote": 0.10}, "bobcat", 0.9, False, "coyote", 0.1),
    ({"bobcat": 0.10, "coyote": 0.90}, "coyote", 0.9, False, "bobcat", 0.1),
    ({"bobcat": 0.60, "coyote": 0.40}, "bobcat", 0.6, True, "coyote", 0.4),
    ({"bobcat": 0.50, "coyote": 0.30, "deer": 0.20}, "bobcat", 0.5, True, "coyote", 0.3),
    ({"empty": 1.0}, "empty", 1.0, False, None, None),
])
def test_format_result(scores, label, prob, flag, second, second_prob):
    assert format_result(scores) == {"label": label, "probability": prob, "needs_review": flag,
                                     "second_label": second, "second_probability": second_prob}
