import csv
import json
import os

import numpy as np
import pytest
from PIL import Image

from ai_server.model import crop_to_box
from training.correction_crops import make_crops, summary

BOX = {"category": "1", "conf": 0.9, "bbox": [0.25, 0.25, 0.5, 0.5]}


@pytest.fixture
def folder(tmp_path):
    """A folder the way training/export_corrections.py leaves it, with one photo per correction."""
    def _folder(*labels):
        with open(tmp_path / "corrections.csv", "w", newline="", encoding="utf-8") as f:
            rows = csv.writer(f)
            rows.writerow(["job", "photo", "label", "model_version"])
            for number, label in enumerate(labels):
                picture = Image.new("RGB", (400, 300), (10 * number, 100, 50))
                picture.paste((250, 250, 250), (100, 75, 300, 225))
                picture.save(tmp_path / f"job-{number}.png")
                rows.writerow([f"job-{number}", f"job-{number}.png", label, "v7"])
        return str(tmp_path)
    return _folder


def crops(folder):
    with open(os.path.join(folder, "crops.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return np.load(os.path.join(folder, "crops.npy"))[: len(meta)], meta


def test_crop_gets_the_corrected_label(folder):
    where = folder("skunk", "mountain_lion")

    counts = make_crops(where, lambda image: BOX)

    pictures, meta = crops(where)
    assert counts == {"used": 2, "empty": 0, "no_box": 0}
    assert [(row["name"], row["label"], row["seq"]) for row in meta] == [
        ("job-0.png", "skunk", "job-0"), ("job-1.png", "mountain_lion", "job-1")]
    assert pictures.shape == (2, 224, 224, 3)


def test_crop_is_cut_the_way_the_ai_server_cuts_it(folder):
    where = folder("skunk")

    make_crops(where, lambda image: BOX)

    with Image.open(os.path.join(where, "job-0.png")) as photo:
        expected = crop_to_box(photo.convert("RGB"), BOX["bbox"]).resize((224, 224), Image.BILINEAR)
    assert np.array_equal(crops(where)[0][0], np.asarray(expected))


def test_correction_to_empty_is_skipped_and_counted(folder):
    where = folder("skunk", "empty")

    counts = make_crops(where, lambda image: BOX)

    assert counts == {"used": 1, "empty": 1, "no_box": 0}
    assert [row["label"] for row in crops(where)[1]] == ["skunk"]


@pytest.mark.parametrize("found", [None, {**BOX, "conf": 0.19}])
def test_photo_without_a_box_of_at_least_02_is_skipped_and_counted(folder, found):
    where = folder("skunk", "cat")
    boxes = iter([BOX, found])

    counts = make_crops(where, lambda image: next(boxes))

    assert counts == {"used": 1, "empty": 0, "no_box": 1}
    assert [row["name"] for row in crops(where)[1]] == ["job-0.png"]


def test_box_of_exactly_02_is_used(folder):
    where = folder("skunk")
    assert make_crops(where, lambda image: {**BOX, "conf": 0.2})["used"] == 1


def test_detector_is_not_asked_about_a_correction_to_empty(folder):
    where = folder("empty")
    asked = []

    make_crops(where, asked.append)

    assert asked == []


def test_folder_where_nothing_can_be_used_gets_no_crops_and_loses_old_ones(folder):
    where = folder("skunk")
    make_crops(where, lambda image: BOX)
    assert os.path.exists(os.path.join(where, "crops.npy"))

    counts = make_crops(where, lambda image: None)

    assert counts == {"used": 0, "empty": 0, "no_box": 1}
    assert sorted(os.listdir(where)) == ["corrections.csv", "job-0.png"]


def test_summary_says_how_many_were_used_and_why_the_others_were_skipped():
    assert summary("data/corrections", {"used": 3, "empty": 1, "no_box": 2}) == (
        "Corrections in data/corrections: 3 used as crops, 1 skipped because the right label is empty, "
        "2 skipped because the detector found no box of at least 0.2.")
