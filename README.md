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

## Photos of people

A photo with the label person gets deleted at any confidence, so also under 80%. The middle server does this as soon as it gets the label, even if the AI server did not delete the photo. It is the one exception to the needs review rule, so these jobs never get the mark. The job row keeps the label and the site code, and the path to the photo gets cleared. The page says that we deleted the photo.

The AI server still deletes every other photo right after the check, so the notice next to the upload button says that we delete every photo.

## Model version

The version of the model sits in `models/model_version.txt`, right next to the model. The AI server reads it at the start and stops there if the file is missing or empty. It then sends the version with every result, also with a failed one, and the middle server keeps it in the job row. A job that fails because it got stuck has no version, because the AI server never answered.

An older `data/pawsort.db` gets the new column by itself when the middle server starts, so you can keep your database.

## Review list

http://127.0.0.1:8000/review lists every job with the needs review mark, the oldest first. A user can confirm the label or pick another one, and the job then leaves the list. The `reviews` table keeps the old label, the new label, the model version and the time. A confirmation is a row where the old and the new label are the same. The job state stays done, because a review changes the tag and not the state.

We delete every photo right after the check, so the list only shows the job, the camera site, the label and the percentage, and the page says this. If a user picks person, the rule for photos of people runs too.

## When something goes wrong

A job that stays in processing for more than 30 seconds goes to failed. The middle server looks for these jobs every second, also when the AI server is down, and it deletes the photo of that job. The page then shows the friendly message, and the next jobs in the queue just carry on.

When the middle server answers with an error, the AI server writes it down, waits a second and carries on. An answer that comes after the 30 seconds gets refused, so a failed job stays failed.

## Test

```
python -m pytest -q
```

Unit tests are in `tests/unit/` and integration tests are in `tests/integration/`.
