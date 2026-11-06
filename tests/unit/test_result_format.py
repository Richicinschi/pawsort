import pytest

from ai_server.logic import format_result


@pytest.mark.parametrize("p_dog,label,prob,flag", [
    (0.90, "dog", 0.9, False),
    (0.10, "cat", 0.9, False),
    (0.60, "dog", 0.6, True),
    (0.50, "dog", 0.5, True),
    (0.03, "cat", 0.97, False),
])
def test_format_result(p_dog, label, prob, flag):
    assert format_result(p_dog) == {"label": label, "probability": prob, "needs_review": flag}
