"""Turning the scores into the answer the user sees. No torch here on purpose."""
THRESHOLD = 0.80
MIN_BOX = 0.2
NOT_SURE = 0.5  # under the 80% rule, so the job goes to a person


def needs_review(confidence: float, threshold: float = THRESHOLD) -> bool:
    return confidence < threshold


def decide(found: dict | None, classify) -> dict[str, float]:
    if found is None:
        return {"empty": 1.0}
    if found["conf"] < MIN_BOX:  # a weak box is often an animal, so a person has to check it
        return {"empty": NOT_SURE}
    if found["category"] == "2":  # 2 is a person
        return {"person": 1.0}
    return classify(found["bbox"])


def format_result(scores: dict[str, float], threshold: float = THRESHOLD) -> dict:
    label = max(scores, key=scores.get)
    confidence = scores[label]
    return {"label": label, "probability": round(confidence, 4), "needs_review": needs_review(confidence, threshold)}
