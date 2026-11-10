"""First step: MegaDetector looks for an animal in the photo."""
from megadetector.detection.run_detector import load_detector
from PIL import Image

WEAK_BOX = 0.05  # we also want the weak boxes, logic.py decides what they mean


def load():
    return load_detector("MDV5A")


def find(detector, image: Image.Image) -> dict | None:
    result = detector.generate_detections_one_image(image.convert("RGB"), "photo", detection_threshold=WEAK_BOX)
    if result.get("failure"):
        raise RuntimeError(result["failure"])
    found = result.get("detections") or []
    return max(found, key=lambda d: d["conf"]) if found else None
