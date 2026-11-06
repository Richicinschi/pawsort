"""Cut every animal out of its photo and save the crops at 224x224, so training does not reread the photos.

Run: python training/prepare_crops.py --data data --cache data/cache
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import detector_boxes, split_photos

SIZE, MARGIN, MIN_CONF = 224, 0.10, 0.2
SPLITS = ["train", "cis_val", "cis_test", "trans_test"]


def cut(item, boxes):
    name, path = item
    try:
        with Image.open(path) as image:
            photo = image.convert("RGB")
    except OSError as exc:
        print("skip", name, exc)
        return None
    width, height = photo.size
    x, y, w, h = boxes[name]["bbox"]
    pad = MARGIN * max(w, h)
    left = int(max(0.0, x - pad) * width)
    top = int(max(0.0, y - pad) * height)
    right = max(int(min(1.0, x + w + pad) * width), left + 1)
    bottom = max(int(min(1.0, y + h + pad) * height), top + 1)
    return np.asarray(photo.crop((left, top, right, bottom)).resize((SIZE, SIZE), Image.BILINEAR), dtype=np.uint8)


def save_part(cache, part, items, boxes):
    keep = [i for i in items if i[0] in boxes and boxes[i[0]]["conf"] >= MIN_CONF]
    array = np.lib.format.open_memmap(os.path.join(cache, f"{part}.npy"), mode="w+", dtype=np.uint8, shape=(len(keep), SIZE, SIZE, 3))
    meta = []
    started = time.time()
    with ThreadPoolExecutor(8) as pool:
        pictures = pool.map(lambda k: cut((k[0], k[1]), boxes), keep)
        for picture, row in zip(pictures, keep):
            if picture is not None:
                array[len(meta)] = picture
                meta.append({"name": row[0], "label": row[2], "seq": row[3]})
    array.flush()
    with open(os.path.join(cache, f"{part}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"{part}: {len(meta)} crops in {time.time() - started:.0f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--cache", default="data/cache")
    args = parser.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    boxes = detector_boxes(args.data)
    index = {}
    for folder, _, files in os.walk(os.path.join(args.data, "images")):
        for name in files:
            index[name] = os.path.join(folder, name)
    for split in SPLITS:
        items = [(p["name"], index[p["name"]], p["label"], p["seq"]) for p in split_photos(args.data, split)]
        save_part(args.cache, split, items, boxes)
    sample_list = os.path.join(args.data, "sample_list.json")
    if os.path.exists(sample_list):
        with open(sample_list, encoding="utf-8") as f:
            rows = json.load(f)
        items = []
        for r in rows:
            name = os.path.basename(r["file"])
            path = os.path.join(args.data, "caltech_sample", name)
            if os.path.exists(path):
                items.append((name, path, r["label"], r["seq"]))
        save_part(args.cache, "sample", items, boxes)


if __name__ == "__main__":
    main()
