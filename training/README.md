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

A quick check that everything still runs:

```
python training/train.py --data data --cache data/cache --max-crops 500 --epochs 1
```
