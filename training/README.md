# Training

The photos are not in the repo because they are too big. Get them from LILA BC:

- CCT20 benchmark images and the split files: https://lila.science/datasets/caltech-camera-traps/
- MegaDetector results for Caltech Camera Traps: https://lila.science/megadetector-results-for-camera-trap-datasets/

Put them in a `data` folder like this: `data/images`, `data/annotations/eccv_18_annotation_files`, `data/md_results.json`.
Extra photos from the other camera sites go in `data/caltech_sample`, with the list in `data/sample_list.json`.

```
python training/prepare_crops.py --data data --cache data/cache
```

This cuts the animals out of all photos once. It takes a while, and you only repeat it when the photos change.

## A new model, step by step

Run every command from the main folder of the project.

1. Fetch the approved corrections, with the middle server up:
   `python training/export_corrections.py --out data/corrections`
2. Turn them into crops:
   `python training/correction_crops.py --corrections data/corrections`
   It runs the detector on every photo and cuts the box the same way as `prepare_crops.py`. It skips a photo without a box of at least 0.2 and a correction to empty, and it prints how many it used and skipped.
3. Train a new model into a folder of its own:
   `python training/train.py --data data --cache data/cache --corrections data/corrections --save data/new_model --version wildlife-1.1`
   Leave `--corrections` out to train without corrections. The folder gets `pawsort_wildlife.pt`, `labels.json`, `model_version.txt`, `report.json` with the scores, the training date and the crops per label, and `training_data.txt` with what the model was trained on. `--save models` is refused, because a new model has to pass the switch first.
4. Offer it to the switch:
   `python training/switch_model.py data/new_model`
   It only switches when the new model scores at least as well as the current one on every target. The main README says how it decides.
5. Start the AI server again.
6. To go back to the version before: `python training/switch_model.py --back`, and start the AI server again.

`training/logs/wildlife-1.0.log` is the log of the training of the model in the repo. Its first line has the training crops per label.

A quick check that everything still runs:

```
python training/train.py --data data --cache data/cache --max-crops 500 --epochs 1
```
