"""Fetch the approved corrections with their photos for the next training, then tell the middle server they are taken.

Run from the main folder: python training/export_corrections.py --out data/corrections
"""
import argparse
import csv
import os
import uuid

import httpx

LIST = "corrections.csv"


def is_job_number(text: str) -> bool:
    try:
        return str(uuid.UUID(text)) == text
    except ValueError:
        return False


def listed_jobs(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["job"] for row in csv.DictReader(f)}


def export(client: httpx.Client, out: str) -> int:
    os.makedirs(out, exist_ok=True)
    r = client.get("/training/corrections")
    r.raise_for_status()
    path = os.path.join(out, LIST)
    listed = listed_jobs(path)
    new = not os.path.exists(path)
    taken, fetched = [], 0
    try:
        with open(path, "a", newline="", encoding="utf-8") as f:
            rows = csv.writer(f)
            if new:
                rows.writerow(["job", "photo", "label", "model_version"])
            for job in r.json()["jobs"]:
                if job["id"] in listed:  # a run that stopped before it could say what it took
                    taken.append(job["id"])
                    continue
                if not is_job_number(job["id"]):  # it becomes a file name
                    continue
                photo = client.get(f"/training/corrections/{job['id']}/photo")
                if photo.is_error:  # the photo is gone, so this correction cannot teach the model anything
                    continue
                name = f"{job['id']}.{'png' if photo.headers['content-type'] == 'image/png' else 'jpg'}"
                with open(os.path.join(out, name), "wb") as picture:
                    picture.write(photo.content)
                rows.writerow([job["id"], name, job["label"], job["model_version"]])
                taken.append(job["id"])
                fetched += 1
    finally:  # also after a stop, so the next run does not fetch the same corrections again
        client.post("/training/corrections/used", json={"jobs": taken}).raise_for_status()
    return fetched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/corrections")
    args = parser.parse_args()
    base = os.environ.get("PAWSORT_ORCH", "http://127.0.0.1:8001")
    with httpx.Client(base_url=base, timeout=30) as client:
        print(f"Approved corrections fetched into {args.out}: {export(client, args.out)}")


if __name__ == "__main__":
    main()
