"""Scoring for the two steps: the detector says empty or not, the model names the animal."""
import collections
import json
import os

import numpy as np

REVIEW = 0.80


def macro_f1(truth, guess):
    scores = []
    for label in sorted(set(truth) | set(guess)):
        tp = int(((truth == label) & (guess == label)).sum())
        fp = int(((truth != label) & (guess == label)).sum())
        fn = int(((truth == label) & (guess != label)).sum())
        scores.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(scores))


def detector_boxes(data: str) -> dict:
    """The best animal or vehicle box per photo, from the MegaDetector results file."""
    with open(os.path.join(data, "md_results.json"), encoding="utf-8") as f:
        boxes = {}
        for im in json.load(f)["images"]:
            found = [d for d in (im.get("detections") or []) if d["category"] in ("1", "3")]
            if found:
                boxes[os.path.basename(im["file"])] = max(found, key=lambda d: d["conf"])
    return boxes


def split_photos(data: str, split: str) -> list[dict]:
    """Photos of one CCT20 split with exactly one label."""
    path = os.path.join(data, "annotations", "eccv_18_annotation_files", f"{split}_annotations.json")
    with open(path, encoding="utf-8") as f:
        part = json.load(f)
    names = {c["id"]: c["name"] for c in part["categories"]}
    labels = collections.defaultdict(set)
    for a in part["annotations"]:
        labels[a["image_id"]].add(names[a["category_id"]])
    rows = []
    for im in sorted(part["images"], key=lambda i: i["id"]):
        if len(labels[im["id"]]) == 1:
            rows.append({"name": os.path.basename(im["file_name"]), "label": next(iter(labels[im["id"]])), "seq": im["seq_id"]})
    return rows


def evaluate(probs, row_of, photos, classes, boxes, min_conf=0.2) -> dict:
    """probs has one row per crop, row_of gives the row of a photo name."""
    truth, guess, sure = [], [], []
    bursts = collections.defaultdict(list)
    for p in photos:
        box = boxes.get(p["name"])
        has_box = box is not None and box["conf"] >= min_conf and p["name"] in row_of
        scores = probs[row_of[p["name"]]] if has_box else None
        truth.append(p["label"])
        guess.append(classes[int(scores.argmax())] if has_box else "empty")
        sure.append(float(scores.max()) >= REVIEW if has_box else (box is None or box["conf"] < 0.05))
        bursts[p["seq"]].append((p["label"], scores))
    truth, guess, sure = np.array(truth), np.array(guess), np.array(sure)
    right = truth == guess
    empty = truth == "empty"
    result = {
        "photos": len(photos),
        "accuracy": float(right.mean()),
        "macro_f1": macro_f1(truth, guess),
        "animal_accuracy": float(right[~empty].mean()),
        "empty_found": float(right[empty].mean()) if empty.any() else None,
        "animals_called_empty": float((guess[~empty] == "empty").mean()),
        "sent_to_person": float((~sure).mean()),
        "accuracy_when_sure": float(right[sure].mean()),
        "per_label": {c: round(float(right[truth == c].mean()), 3) for c in sorted(set(truth))},
    }
    burst_right, burst_sure = [], []
    for items in bursts.values():
        animals = [label for label, _ in items if label != "empty"]
        want = collections.Counter(animals).most_common(1)[0][0] if animals else "empty"
        got = [scores for _, scores in items if scores is not None]
        if got:
            mean = np.mean(got, axis=0)
            say, top = classes[int(mean.argmax())], float(mean.max())
        else:
            say, top = "empty", 1.0
        burst_right.append(say == want)
        burst_sure.append(top >= REVIEW)
    burst_right, burst_sure = np.array(burst_right), np.array(burst_sure)
    result.update({
        "bursts": len(burst_right),
        "burst_accuracy": float(burst_right.mean()),
        "burst_sent_to_person": float((~burst_sure).mean()),
        "burst_accuracy_when_sure": float(burst_right[burst_sure].mean()) if burst_sure.any() else None,
    })
    return result
