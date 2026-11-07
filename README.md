# PawSort
 
Upload camera trap photos and find out if each one is empty or which animal it shows, with a "needs review" flag when the model is not sure.

PawSort is built from three servers: the interface (port 8000) talks to the orchestrator (port 8001, SQLite), and the AI server takes jobs from the orchestrator. The AI server works in two steps: MegaDetector finds the animal, then our MobileNet names it. The interface never talks to the AI server directly.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the main folder of the project:

```
python run_all.py
```

Then open http://127.0.0.1:8000, fill in a camera site code, pick one or more photos and press Upload photos. Ctrl+C stops all three servers.

The first photo can take a while, because the AI server loads the detector first. It can download the detector the very first time. Until then the page says "Still working...".

## Upload rules

The website refuses a file and says why when it is empty, larger than 10 MB, not a JPEG or PNG, cannot be opened, smaller than 128 x 128 pixels or larger than 6000 x 6000 pixels. The other photos of the same upload still go through.

The website removes all metadata from a photo, so also the location data, and saves it under a random name in `data/uploads`. The website never serves that folder.

## When something goes wrong

A job that stays in processing for more than 30 seconds goes to failed. The middle server looks for these jobs every second, also when the AI server is down, and it deletes the photo of that job. The page then shows the friendly message, and the next jobs in the queue just carry on.

When the middle server answers with an error, the AI server writes it down, waits a second and carries on. An answer that comes after the 30 seconds gets refused, so a failed job stays failed.

## Test

```
python -m pytest -q
```

Unit tests are in `tests/unit/` and integration tests are in `tests/integration/`.
