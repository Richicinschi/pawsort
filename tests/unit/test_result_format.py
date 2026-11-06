import pytest

from ai_server.logic import format_result


@pytest.mark.parametrize("scores,label,prob,flag", [
    ({"bobcat": 0.90, "coyote": 0.10}, "bobcat", 0.9, False),
    ({"bobcat": 0.10, "coyote": 0.90}, "coyote", 0.9, False),
    ({"bobcat": 0.60, "coyote": 0.40}, "bobcat", 0.6, True),
    ({"bobcat": 0.50, "coyote": 0.30, "deer": 0.20}, "bobcat", 0.5, True),
    ({"empty": 1.0}, "empty", 1.0, False),
])
def test_format_result(scores, label, prob, flag):
    assert format_result(scores) == {"label": label, "probability": prob, "needs_review": flag}
