"""Retrain MobileNetV3-Small on the animal crops and score the whole two step system.

Run: python training/train.py --cache data/cache --data data --save data/new_model --version wildlife-1.1
Add --corrections data/corrections to also learn from the approved corrections.
Add --max-crops 500 --epochs 1 for a quick check that everything still runs.
"""
import argparse
import collections
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import detector_boxes, evaluate, split_photos

SEED = 42
BATCH = 96
MIN_LABEL_CROPS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
REFUSED = ("Refused: a training never writes into models, because a new model has to pass the switch first. "
           "Save it into a folder of its own, for example --save data/new_model, "
           "and then run python training/switch_model.py data/new_model")


def load_part(cache, name):
    with open(os.path.join(cache, f"{name}.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return np.load(os.path.join(cache, f"{name}.npy"), mmap_mode="r")[: len(meta)], meta


def to_input(pictures, train):
    x = torch.as_tensor(np.ascontiguousarray(pictures), device=DEVICE).permute(0, 3, 1, 2).float().div_(255)
    if train:
        flip = torch.rand(len(x), device=DEVICE) < 0.5
        x[flip] = x[flip].flip(3)
        x = (x * (0.8 + 0.4 * torch.rand(len(x), 1, 1, 1, device=DEVICE))).clamp_(0, 1)
    return (x - MEAN) / STD


def predict(net, pictures, rows=None):
    net.eval()
    rows = list(range(len(pictures))) if rows is None else rows
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            out.append(torch.softmax(net(to_input(pictures[rows[i:i + 256]], False)), 1).cpu().numpy())
    return np.concatenate(out)


def check_candidate(folder: str, version: str | None) -> None:
    live, wanted = os.path.realpath("models"), os.path.realpath(folder)
    if wanted == live or wanted.startswith(live + os.sep):
        raise SystemExit(REFUSED)
    if not (version or "").strip():
        raise SystemExit("Refused: the new model needs a name of its own, for example --version wildlife-1.1")


def describe(parts: dict, cache: str, corrections: str | None, listed: int) -> list[str]:
    """What a model was trained on, for training_data.txt. The first line says it in short."""
    total, fixed = sum(parts.values()), parts.get("corrections", 0)
    file = f"{corrections}/corrections.csv"
    lines = [f"{total} crops: {total - fixed} from {cache} and "
             + (f"{fixed} from the corrections in {file}" if corrections else "no corrections")]
    lines += [f"{part}: {parts[part]} crops" for part in ("train", "cis_val", "sample") if part in parts]
    if corrections:
        lines.append(f"corrections: {fixed} crops from the {listed} corrections in {file}")
    return lines


def save_candidate(folder: str, version: str, state: dict, labels: list[str], results: dict, label_crops: dict,
                   training_data: list[str]) -> None:
    """Write everything training/switch_model.py reads into a folder of its own. Never into the live model."""
    check_candidate(folder, version)
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, "pawsort_wildlife.pt"))
    report = {**results, "trained_on": time.strftime("%Y-%m-%d"), "label_crops": label_crops}
    texts = {"labels.json": json.dumps(labels), "model_version.txt": version.strip() + "\n",
             "report.json": json.dumps(report, indent=2), "training_data.txt": "\n".join(training_data) + "\n"}
    for name, text in texts.items():
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-crops", type=int, default=0)
    parser.add_argument("--corrections", help="folder of export_corrections.py, after correction_crops.py")
    parser.add_argument("--save", help="folder for the new model, never models itself")
    parser.add_argument("--version", help="name of the new model version, needed with --save")
    args = parser.parse_args()
    if args.save:
        check_candidate(args.save, args.version)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_pictures, train_meta = load_part(args.cache, "train")
    val_pictures, val_meta = load_part(args.cache, "cis_val")
    sequences = sorted({m["seq"] for m in val_meta})
    random.shuffle(sequences)
    to_train = set(sequences[: len(sequences) // 2])  # a burst never sits on both sides
    sources = [(train_pictures, i, m["label"], "train") for i, m in enumerate(train_meta) if m["label"] != "empty"]
    sources += [(val_pictures, i, m["label"], "cis_val") for i, m in enumerate(val_meta)
                if m["label"] != "empty" and m["seq"] in to_train]
    check = [(i, m["label"]) for i, m in enumerate(val_meta) if m["label"] != "empty" and m["seq"] not in to_train]
    if os.path.exists(os.path.join(args.cache, "sample.npy")):
        extra_pictures, extra_meta = load_part(args.cache, "sample")
        sources += [(extra_pictures, i, m["label"], "sample") for i, m in enumerate(extra_meta)]
    if args.max_crops:
        random.shuffle(sources)
        sources = sources[: args.max_crops]
    listed = 0
    if args.corrections:  # after the cut above, so a quick run keeps every correction
        with open(os.path.join(args.corrections, "corrections.csv"), encoding="utf-8") as f:
            listed = sum(1 for _ in f) - 1
        if os.path.exists(os.path.join(args.corrections, "crops.npy")):
            fixed_pictures, fixed_meta = load_part(args.corrections, "crops")
            sources += [(fixed_pictures, i, m["label"], "corrections") for i, m in enumerate(fixed_meta)]
    counts = collections.Counter(s[2] for s in sources)
    labels = sorted(c for c, n in counts.items() if n >= (MIN_LABEL_CROPS if not args.max_crops else 1))
    index = {c: i for i, c in enumerate(labels)}
    sources = [s for s in sources if s[2] in index]
    training_data = describe(collections.Counter(s[3] for s in sources), args.cache, args.corrections, listed)
    print("crops", len(sources), "labels", len(labels), "on", DEVICE, flush=True)

    net = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, len(labels))
    net = net.to(DEVICE)
    weights = torch.tensor([(len(sources) / counts[c]) ** 0.5 for c in labels], device=DEVICE, dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = args.epochs * ((len(sources) + BATCH - 1) // BATCH)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=1e-3, total_steps=steps)

    best, best_state = -1.0, None
    started = time.time()
    rows = [i for i, label in check if label in index]
    truth = np.array([index[label] for _, label in check if label in index])
    for epoch in range(1, args.epochs + 1):
        net.train()
        random.shuffle(sources)
        for i in range(0, len(sources), BATCH):
            chunk = sources[i:i + BATCH]
            pictures = np.stack([s[0][s[1]] for s in chunk])
            y = torch.tensor([index[s[2]] for s in chunk], device=DEVICE)
            optimiser.zero_grad(set_to_none=True)
            loss_fn(net(to_input(pictures, True)), y).backward()
            optimiser.step()
            schedule.step()
        score = float((predict(net, val_pictures, rows).argmax(1) == truth).mean())
        print(f"epoch {epoch}: check accuracy {score:.4f}, {time.time() - started:.0f} s", flush=True)
        if score > best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)

    boxes = detector_boxes(args.data)
    results = {"crops": len(sources), "labels": labels, "check_accuracy": best, "train_seconds": round(time.time() - started)}
    for name, key in [("cis_test", "known_sites"), ("trans_test", "new_sites")]:
        pictures, meta = load_part(args.cache, name)
        probs = predict(net, pictures)
        full = np.concatenate([probs, np.zeros((len(probs), 1), dtype=probs.dtype)], axis=1)
        results[key] = evaluate(full, {m["name"]: i for i, m in enumerate(meta)}, split_photos(args.data, name), labels + ["empty"], boxes)
        print(key, "accuracy", round(results[key]["accuracy"], 3), "bursts", round(results[key]["burst_accuracy"], 3), flush=True)
    if args.save:
        save_candidate(args.save, args.version, best_state, labels, results, {c: counts[c] for c in labels},
                       training_data)
        print(f"saved {args.version} in {args.save}, now run: python training/switch_model.py {args.save}")


if __name__ == "__main__":
    main()
