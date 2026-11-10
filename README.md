# PawSort
 
Upload camera trap photos and find out if each one is empty or which animal it shows, with a "needs review" flag when the model is not sure.

PawSort is built from three servers: the interface (port 8000) talks to the orchestrator (port 8001, SQLite), and the AI server takes jobs from the orchestrator. The AI server works in two steps: MegaDetector finds the animal, then our MobileNet names it. The interface never talks to the AI server directly. The orchestrator, which we call the middle server, is the only one that keeps photos.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Use pip for this and not uv. On a Mac uv reads the extra index line in `requirements.txt` differently and cannot install the right detector.

## Run

From the main folder of the project:

```
python run_all.py
```

Then open http://127.0.0.1:8000, fill in a camera site code, drop your photos on the box or click it to pick them, and press Upload photos. Ctrl+C stops all three servers.

Every photo gets a line with a small picture, the answer and the file name, and the row above the list counts the answers. The small pictures come from the browser of the user. The website only sends a photo back where a person has to look at it: on the review list, on the corrections page and in the weekly sample.

The first photo can take a while, because the AI server loads the detector first. It can download the detector the very first time. Until then the page says "Still working...".

## Upload rules

The website refuses a file and says why when it is empty, larger than 10 MB, not a JPEG or PNG, cannot be opened, smaller than 128 x 128 pixels or larger than 6000 x 6000 pixels. The other photos of the same upload still go through.

The website removes all metadata from a photo, so also the location data. It does this in memory and then sends the photo to the middle server. The website keeps nothing on disk. Only an upload of more than 1 MB waits in a temporary file of the web framework while it comes in, and that file is gone when the upload is done. The same holds for the middle server when it takes the photo from the website.

## Where the photos are

The middle server saves a photo under a random name in `data/photos`, and the job row keeps only that name. To make a job, the website sends the photo itself and the site code. So no request can make a job for a file somewhere else, and nobody can ask for a photo by its name.

The AI server gets a working copy. It claims a job, asks the middle server for the photo of that job and puts the copy in its own folder `data/work`. The middle server only gives the copy while the job is in processing. The AI server deletes the copy right after the check, also when the check fails. Every copy has a name that ends on `.copy`. When the AI server starts, it first deletes the copies that are still in `data/work`, so a copy from before a crash goes too. It deletes nothing else in that folder.

A photo stays on the middle server when its job is done and the label is not person, because we need these photos for the review and for the next training. A job that fails keeps no photo: the middle server deletes it when the AI server reports an error and when a job got stuck. A failed job has no tag and never comes on the review list, so we have no use for its photo, and it could show a person.

The row of a job forgets the name of its photo only when the file is really gone. If a delete does not work, the middle server tries a few times, and at its next start it deletes every photo that still belongs to a failed job or to a job with the label person.

The three servers do not need the same disk any more. A second AI server on the same computer needs its own working folder, because every AI server deletes the copies in its folder when it starts. Set `PAWSORT_WORK` for it, for example to `data/work2`. `PAWSORT_PHOTOS` moves the photo folder of the middle server.

An older `data/pawsort.db` still works. Its jobs keep their tags but have no photo, because back then we deleted every photo. A job that was still waiting at the upgrade ends as failed, so upload that photo again. The old folder `data/uploads` is not used any more and you can delete it after the upgrade.

## Photos of people

A photo with the label person gets deleted at any confidence, so also under 80%. The middle server deletes it from `data/photos` as soon as it gets the label, from the AI server or from a user. An answer that comes too late gets refused, but then the job has already failed and its photo is already gone. It is the one exception to the needs review rule, so these jobs never get the mark. The job row keeps the label and the site code, and the name of the photo gets cleared once the file is gone. The page says that we deleted the photo.

The notice next to the upload button says which photos we keep and why, and that we delete photos of people and photos we could not check.

## Model version

The version of the model sits in `models/model_version.txt`, right next to the model. The AI server reads it at the start and stops there if the file is missing or empty. It then sends the version with every result, also with a failed one, and the middle server keeps it in the job row. A job that fails because it got stuck has no version, because the AI server never answered.

An older `data/pawsort.db` gets the new column by itself when the middle server starts, so you can keep your database.

## Switching to a new model

`models/versions.json` has one entry for every model version: the training date, one line about the training data, the file names and the test report. The test report holds the numbers we check our targets with: the accuracy on known and on new sites, the accuracy on the sure tags, the share of sure tags, the share of animals called empty, and the lowest recall of a label with 500 or more training crops. The entry of the current model comes from `results.json` and from the log of its training, `training/logs/wildlife-1.0.log`. The log has the number of training crops per label, and the entry keeps these numbers as `label_crops`, so everybody can check which labels have 500 or more.

A new model only goes live through this script:

```
python training/switch_model.py data/new_model
```

The folder holds `pawsort_wildlife.pt`, `labels.json`, `model_version.txt`, `report.json` and `training_data.txt`. `training/train.py --save` writes exactly these five files. The script compares the new report with the current one, target by target. If the new model is worse on one target, the script refuses, names that target with both numbers and changes nothing. A target that only one of the two models measured cannot be compared, so the script refuses then too. When it accepts, it keeps the files of the old and of the new version in `models/versions`, puts the new model in place and adds its entry to `versions.json`.

```
python training/switch_model.py --back
```

This puts the version before in place again. After a switch, start the AI server again, because it reads the model and its version when it starts. The script says so too.

If a switch stops in the middle, for example because the computer goes off, run the same command again. It then finishes the switch. The script never writes over a version it already keeps, so the copy of the old model stays good.

The kept versions in `models/versions` are files on this computer. They are not in git, so `--back` only works on the computer where the switch was made.

## Review list

http://127.0.0.1:8000/review lists every job with the needs review mark, the oldest first. A user can confirm the label or pick another one, and the job then leaves the list. The `reviews` table keeps the old label, the new label, the model version and the time. A confirmation is a row where the old and the new label are the same. The job state stays done, because a review changes the tag and not the state.

The list shows the photo of every job, with the job, the camera site, the label and the percentage, and under it why the tag needs review, the second guess and the model version. The review page gets a photo only while its job is on the review list. After a confirmation or a correction the photo stays on the middle server, but the page does not show it any more. The page then only takes that one job off the list and does not load the other photos again. If a user picks person, the rule for photos of people runs too and the photo is deleted at once.

## Why a tag needs review

The AI server also reports its second guess. That is the next most likely label of our MobileNet, with its score. Empty, person and a weak box have no second guess, because our MobileNet does not run for them. The middle server keeps the second guess in the job row, and an older `data/pawsort.db` gets the two new columns by itself.

When a tag needs review, the page says why in one sentence that names the animal and the 80%, and then shows the second guess. For example: "The model is less than 80% sure this is a raccoon. Its second guess is a skunk, 21% sure." A photo with only a weak box gets its own sentence: "The model saw something in this photo, but not clearly enough to say what it is." The website server makes these sentences, so unit tests check the exact words.

## Corrections

When a user picks another label on the review list, that is a correction, and it waits for a decision. A confirmation needs no decision. A change to person is final, because that photo is gone.

http://127.0.0.1:8000/corrections lists every correction that waits, the oldest first, with the photo, the old label, the new label and the model version. Approve keeps the new label. Reject puts the old label back on the job, because then the correction was wrong. A correction gets one decision, and a second one is refused. The page is meant for the coordinator. There is no login yet, so everybody can use it, and the page says so.

Only approved corrections go to the next training. Run this from the main folder of the project, with the middle server up:

```
python training/export_corrections.py --out data/corrections
```

The script asks the middle server for the approved corrections that no training took yet. It saves each photo in the folder and writes `corrections.csv` with the job, the photo file, the right label and the model version. Then it tells the middle server which ones it took, so the next run does not fetch them again. A correction that still waits or that got rejected can never be in that folder. A correction without a photo is left out.

The corrections page also counts the approved corrections that wait for a training. At 500 it says that it is time to plan a training round.

## Numbers per camera site

http://127.0.0.1:8000/numbers has a table with one row per camera site and week: the tags the model gave, the share that needed review, and the share of the checked tags that a person corrected. A week runs from Monday to Sunday and goes by the day of the upload, in UTC. A rejected correction does not count as corrected. A site and week with more than 40% needs review gets a warning on top of the page.

The page also has a box "Time to plan a training round". It lists every reason that holds: 500 approved corrections wait for a training, a camera site sent its first photo after the current model was trained, or a site got a warning. The middle server reads the training date from `models/versions.json`, and `PAWSORT_VERSIONS` moves that file. When that file has no training date, the page says that it cannot tell which camera sites are new.

## Weekly sample

http://127.0.0.1:8000/sample shows up to 50 tags that the model was sure about, picked at random. Nobody looks at these tags otherwise, so this is how we find mistakes the model is sure about. A user confirms the label or picks another one. The answer is stored like a review and marked as coming from the sample. A changed label goes to the corrections page like any other correction. The page counts how many tags of the sample are checked and how many of those were changed. A change that the coordinator rejects later still counts as changed there.

The sample belongs to a week, from Monday to Sunday in UTC. The middle server picks it the first time somebody opens the page in that week, from the sure tags that came in since the sample before, and it stays the same until the week is over. So when nobody opens the page for a week, no tag is lost, because these tags can still be picked the next time. A tag with the label person, a tag without a photo and a tag that a person already checked are never picked.

Two things to know. When more than 50 sure tags came in, the ones that were not picked are not offered again, because the next sample only looks at the tags after this pick. And when the page is opened in a week in which no sure tag waits, that week's sample stays empty, also when tags come in later that week. They are in the sample of the week after.

Both pages are meant for the team and have no login yet, and they say so.

## Rare animals and the download

http://127.0.0.1:8000/rare has the list of rare animals. An animal can be added from the labels we know and taken off the list again. The page is meant for the coordinator. There is no login yet, so everybody can use it, and the page says so.

http://127.0.0.1:8000/download gives all tags as a CSV file, `pawsort-tags.csv`. It has one row for every job that is done: the job number, the camera site, the label, the percentage, if the tag still needs review, if a person checked it, the model version and the time of the upload. When a user changed a tag, the label is the one the user picked. The percentage is always the one of the model. For a rare animal the camera site and the time of the upload stay empty, and the rows are in the order of the job numbers, which are random, and not in the order of the uploads. Otherwise the rows next to a rare animal, from the same upload, would give its camera site away. The middle server already leaves the site code and the time out, so they never reach the website. A text cell that starts with =, +, - or @ gets a ' in front, so a spreadsheet does not read a site code like -A12 as a formula. The upload page has a link to the download.

On the pages everybody still sees the camera site of every tag, also of a rare animal. Hiding it from volunteers needs a login, and the login is not built.

## From a correction to a new model

This is the whole loop, in order. Steps 3 to 6 run from the main folder of the project.

1. A user corrects a tag on the review list or in the weekly sample.
2. The coordinator approves the correction on http://127.0.0.1:8000/corrections.
3. `python training/export_corrections.py --out data/corrections` fetches the approved corrections with their photos. The middle server has to be up.
4. `python training/correction_crops.py --corrections data/corrections` cuts the animal out of every photo, the same way the AI server does it, and gives the crop the corrected label. A photo without a box of at least 0.2 is skipped, and so is a correction to empty, because our MobileNet never learns empty. The script says how many it used and how many it skipped.
5. `python training/train.py --data data --cache data/cache --corrections data/corrections --save data/new_model --version wildlife-1.1` trains on the old crops plus the corrections and writes a new model into `data/new_model`. It never writes into `models`, and it refuses when you ask for that.
6. `python training/switch_model.py data/new_model` puts the new model in place, but only when it scores at least as well as the current one on every target. Otherwise it says which target is worse and changes nothing.
7. Start the AI server again, because it reads the model when it starts.
8. If the new model turns out bad, `python training/switch_model.py --back` puts the version before in place again. Then start the AI server again.

## When something goes wrong

A job that stays in processing for more than 30 seconds goes to failed. The middle server looks for these jobs every second, also when the AI server is down, and it deletes the photo of that job. The page then shows the friendly message, and the next jobs in the queue just carry on.

When the middle server answers with an error, the AI server writes it down, waits a second and carries on. An answer that comes after the 30 seconds gets refused, so a failed job stays failed.

The middle server also refuses an answer with a confidence under 0 or over 1. It answers with 422 and stores nothing, so the job stays in processing until a good answer comes or the 30 seconds are over.

## Energy

We measured with CodeCarbon on our own PC, with the Dutch grid of 268 g CO2 per kWh. CodeCarbon estimates the whole PC and not only our code, so these numbers are an upper bound.

| Part | Energy | CO2 |
|---|---|---|
| Tagging 1,000 photos | about 66 Wh | about 18 g |
| One training run | about 32 to 44 Wh | about 9 to 12 g |
| 1,000 jobs through the middle server, without the AI | 0.5 Wh | 0.1 g |
| A PC that just waits, per hour | about 63 Wh | about 17 g |

We measured the row for the middle server before it kept the photos. It now also saves every photo, and one quick run on a busy PC gave about 0.6 Wh and 0.2 g.

To measure it yourself, close everything else first, because CodeCarbon counts all the work of the PC. Then run this from the main folder of the project:

```
python training/measure_energy.py --photos tests/data --count 30 --idle 60
```

## Test

```
python -m pytest -q
```

Unit tests are in `tests/unit/` and integration tests are in `tests/integration/`.

One integration test runs two AI workers at the same time on one middle server. It checks that every job is done exactly once and that no job goes to both workers.
