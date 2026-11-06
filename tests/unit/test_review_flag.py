import pytest

from ai_server.logic import needs_review


@pytest.mark.parametrize("confidence,expected", [(0.79, True), (0.80, False), (0.95, False), (0.5, True)])
def test_needs_review_threshold(confidence, expected):
    assert needs_review(confidence) is expected
