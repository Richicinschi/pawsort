"""Turning the scores into the answer the user sees. No torch here on purpose."""
THRESHOLD = 0.80


def needs_review(confidence: float, threshold: float = THRESHOLD) -> bool:
    return confidence < threshold


def decide(found: dict | None, classify) -> dict[str, float]:
    if found is None:
        return {"empty": 1.0}
    if found["category"] == "2":  # 2 is a person
        return {"person": 1.0}
    return classify(found["bbox"])


def format_result(scores: dict[str, float], threshold: float = THRESHOLD) -> dict:
    label = max(scores, key=scores.get)
    confidence = scores[label]
    return {"label": label, "probability": round(confidence, 4), "needs_review": needs_review(confidence, threshold)}
