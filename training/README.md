# Training

The photos are not in the repo because they are too big. Get them from LILA BC:

- CCT20 benchmark images and the split files: https://lila.science/datasets/caltech-camera-traps/
- MegaDetector results for Caltech Camera Traps: https://lila.science/megadetector-results-for-camera-trap-datasets/

Put them in a `data` folder like this: `data/images`, `data/annotations/eccv_18_annotation_files`, `data/md_results.json`.
Extra photos from the other camera sites go in `data/caltech_sample`, with the list in `data/sample_list.json`.

```
python training/prepare_crops.py --data data --cache data/cache
python training/train.py --data data --cache data/cache --save
```

`--save` writes the new model over `models/pawsort_wildlife.pt` and `models/labels.json`, and the scores into `training/last_run.json`. A new model must not go live like that, it has to pass `training/switch_model.py` first. So after the training:

1. Make a folder, for example `data/new_model`, and move `models/pawsort_wildlife.pt` and `models/labels.json` into it. Put the old two files back with `git checkout models/pawsort_wildlife.pt models/labels.json`.
2. Write `model_version.txt` in that folder with the new version, for example `wildlife-1.1`.
3. Copy `training/last_run.json` into the folder as `report.json`. Add `"trained_on"` with the date, for example `"2026-11-12"`, and `"label_crops"` with the number of training crops of every label, for example `{"bobcat": 3254, "rodent": 2244}`. Without `label_crops` the lowest recall cannot be checked.
4. Write `training_data.txt`. The first line says in a few words what the model was trained on, the lines after it can list the photos.
5. Run `python training/switch_model.py data/new_model` from the main folder.

The script only switches when the new model scores at least as well as the current one on every target. The main README says how it decides and how to go back.

The approved corrections for a training come from `python training/export_corrections.py --out data/corrections`, with the middle server up.

A quick check that everything still runs:

```
python training/train.py --data data --cache data/cache --max-crops 500 --epochs 1
```
