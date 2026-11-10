"""Switch to a new model version, but only when it scores at least as well as the current one (AI-10, AI-11, FR-20).

Run from the main folder: python training/switch_model.py <folder with the new model>
Go back to the version before: python training/switch_model.py --back
"""
import argparse
import filecmp
import json
import os
import re
import shutil
import sys

FILES = ("pawsort_wildlife.pt", "labels.json", "model_version.txt")
EXTRA = ("report.json", "training_data.txt")
MIN_CROPS = 500
VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # it becomes a folder name
TARGETS = [  # the name in a report, the same in plain words, and if a higher number is the better one
    ("accuracy_known_sites", "accuracy on known sites (AI-01)", True),
    ("accuracy_new_sites", "accuracy on new sites (AI-02)", True),
    ("accuracy_when_sure", "accuracy on the sure tags (AI-03)", True),
    ("sure_known_sites", "share of sure tags on known sites (AI-04)", True),
    ("sure_new_sites", "share of sure tags on new sites (AI-04)", True),
    ("animals_called_empty_known_sites", "share of animals called empty on known sites (AI-05)", False),
    ("animals_called_empty_new_sites", "share of animals called empty on new sites (AI-05)", False),
    ("lowest_recall_known_sites", "lowest recall of a label with 500 or more training crops, known sites (AI-06)",
     True),
    ("lowest_recall_new_sites", "lowest recall of a label with 500 or more training crops, new sites (AI-06)",
     True),
]
RESTART = "Start the AI server again, because it reads the model when it starts."


class Refused(Exception):
    pass


def lowest_recall(site: dict, crops: dict | None) -> float | None:
    recalls = [recall for label, recall in site["per_label"].items() if (crops or {}).get(label, 0) >= MIN_CROPS]
    return min(recalls) if recalls else None


def targets(run: dict) -> dict:
    """The numbers the targets are checked against, from what training/train.py writes about a run."""
    known, new, crops = run["known_sites"], run["new_sites"], run.get("label_crops")
    numbers = {
        "accuracy_known_sites": known["accuracy"],
        "accuracy_new_sites": new["accuracy"],
        "accuracy_when_sure": known["accuracy_when_sure"],
        "sure_known_sites": 1 - known["sent_to_person"],
        "sure_new_sites": 1 - new["sent_to_person"],
        "animals_called_empty_known_sites": known["animals_called_empty"],
        "animals_called_empty_new_sites": new["animals_called_empty"],
        "lowest_recall_known_sites": lowest_recall(known, crops),
        "lowest_recall_new_sites": lowest_recall(new, crops),
    }
    return {name: None if number is None else round(number, 4) for name, number in numbers.items()}


def check(candidate: dict, current: dict) -> None:
    for name, words, higher in TARGETS:
        new, old = candidate.get(name), current.get(name)
        if new is not None and not 0 <= new <= 1:  # a report in percentages, or one with a broken number
            raise Refused(f"Refused: the {words} of the new model is {new}, but a report only has numbers "
                          "from 0 to 1.")
        if new is None and old is None:
            continue
        if new is None or old is None:
            raise Refused(f"Refused: the {words} is measured for only one of the two models, "
                          "so we cannot compare them.")
        if (new < old) if higher else (new > old):
            raise Refused(f"Refused: the {words} is {new:.1%} for the new model and {old:.1%} for the current one.")


def read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_versions(models: str, versions: dict) -> None:
    path = os.path.join(models, "versions.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(versions, f, indent=2)
        f.write("\n")
    os.replace(path + ".tmp", path)  # the file is the old one or the whole new one, never half of it


def keep(source: str, kept: str, names: tuple) -> None:
    """Keep the files of a version in a folder of its own. A version that is kept is never written again."""
    if os.path.isdir(kept):
        return
    shutil.rmtree(kept + ".tmp", ignore_errors=True)
    os.makedirs(kept + ".tmp")
    for name in names:
        shutil.copy2(os.path.join(source, name), os.path.join(kept + ".tmp", name))
    os.replace(kept + ".tmp", kept)


def put_in_place(kept: str, models: str) -> None:
    for name in FILES:
        shutil.copy2(os.path.join(kept, name), os.path.join(models, name + ".tmp"))
        os.replace(os.path.join(models, name + ".tmp"), os.path.join(models, name))


def switch(folder: str, models: str = "models") -> str:
    for name in FILES + EXTRA:
        if not os.path.exists(os.path.join(folder, name)):
            raise Refused(f"Refused: {name} is missing in {folder}.")
    with open(os.path.join(folder, "model_version.txt"), encoding="utf-8") as f:
        version = f.read().strip()
    with open(os.path.join(folder, "training_data.txt"), encoding="utf-8") as f:
        training_data = f.readline().strip()
    run = read_json(os.path.join(folder, "report.json"))
    versions = read_json(os.path.join(models, "versions.json"))
    current = versions["current"]
    if not VERSION.fullmatch(version):
        raise Refused(f"Refused: '{version}' in model_version.txt is not a version name. Use letters, numbers, "
                      "dots and dashes, for example wildlife-1.1.")
    if version in versions["versions"]:
        raise Refused(f"Refused: the new model needs a version of its own in model_version.txt, not '{version}'.")
    if not run.get("trained_on"):
        raise Refused("Refused: report.json does not say when the new model was trained (trained_on).")
    report = targets(run)
    check(report, versions["versions"][current]["report"])

    kept = os.path.join(models, "versions", version)
    if os.path.isdir(kept) and filecmp.cmpfiles(folder, kept, FILES + EXTRA, shallow=False)[0] != list(FILES + EXTRA):
        raise Refused(f"Refused: {kept} already holds other files. Give the new model another version name.")

    keep(models, os.path.join(models, "versions", current), FILES)
    keep(folder, kept, FILES + EXTRA)
    put_in_place(kept, models)  # a stop from here on is mended by running the same command again
    versions["versions"][version] = {"trained_on": run["trained_on"], "training_data": training_data,
                                     "report": report, "label_crops": run.get("label_crops")}
    versions.update(current=version, previous=current)
    write_versions(models, versions)
    return f"Switched from {current} to {version}. {RESTART}"


def back(models: str = "models") -> str:
    versions = read_json(os.path.join(models, "versions.json"))
    current, previous = versions["current"], versions["previous"]
    if previous is None:
        raise Refused("Refused: there is no version before this one to go back to.")
    kept = os.path.join(models, "versions", previous)
    for name in FILES:
        if not os.path.exists(os.path.join(kept, name)):
            raise Refused(f"Refused: {name} of {previous} is missing in {kept}.")
    put_in_place(kept, models)
    versions.update(current=previous, previous=current)
    write_versions(models, versions)
    return f"Switched back from {current} to {previous}. {RESTART}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?")
    parser.add_argument("--back", action="store_true")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()
    if args.back == bool(args.folder):
        parser.error("give the folder of the new model, or --back")
    try:
        print(back(args.models) if args.back else switch(args.folder, args.models))
    except Refused as refusal:
        sys.exit(str(refusal))


if __name__ == "__main__":
    main()
