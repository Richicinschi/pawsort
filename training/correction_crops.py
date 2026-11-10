"""Turn the approved corrections that training/export_corrections.py fetched into training crops.

Run from the main folder: python training/correction_crops.py --corrections data/corrections
"""
import argparse
import csv
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_crops import MIN_CONF, save_part


def make_crops(folder: str, find) -> dict:
    """find gives the best box of a photo, like step 1 of the AI server. Returns how many were used and skipped."""
    with open(os.path.join(folder, "corrections.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = {"used": 0, "empty": 0, "no_box": 0}
    items, boxes = [], {}
    for row in rows:
        if row["label"] == "empty":  # step 2 never learns empty
            counts["empty"] += 1
            continue
        path = os.path.join(folder, row["photo"])
        with Image.open(path) as photo:
            box = find(photo)
        if box is None or box["conf"] < MIN_CONF:
            counts["no_box"] += 1
            continue
        boxes[row["photo"]] = box
        items.append((row["photo"], path, row["label"], row["job"]))
    for old in ("crops.npy", "crops.json"):
        if os.path.exists(os.path.join(folder, old)):
            os.remove(os.path.join(folder, old))
    if items:
        save_part(folder, "crops", items, boxes)
    counts["used"] = len(items)
    return counts


def summary(folder: str, counts: dict) -> str:
    return (f"Corrections in {folder}: {counts['used']} used as crops, {counts['empty']} skipped because the right "
            f"label is empty, {counts['no_box']} skipped because the detector found no box of at least {MIN_CONF}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrections", default="data/corrections")
    args = parser.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ai_server import detector

    finder = detector.load()
    print(summary(args.corrections, make_crops(args.corrections, lambda photo: detector.find(finder, photo))))


if __name__ == "__main__":
    main()
