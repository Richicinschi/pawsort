"""Turning a model probability into the answer the user sees. No torch here on purpose."""
THRESHOLD = 0.80


def needs_review(confidence: float, threshold: float = THRESHOLD) -> bool:
    return confidence < threshold


def format_result(p_dog: float, threshold: float = THRESHOLD) -> dict:
    label = "dog" if p_dog >= 0.5 else "cat"
    confidence = p_dog if label == "dog" else 1.0 - p_dog
    return {"label": label, "probability": round(confidence, 4), "needs_review": needs_review(confidence, threshold)}
