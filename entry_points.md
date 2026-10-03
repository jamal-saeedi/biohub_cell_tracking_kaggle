# Entry points

Paths for inference come from `SETTINGS.json` (`--settings PATH` or `BIOHUB_SETTINGS`
selects another file); training paths are command-line arguments. Install first:
`pip install -r requirements.txt && pip install -e . --no-deps`.

## Inference

| step | command | reads | writes |
|---|---|---|---|
| 1. download the models | `python tools/download_models.py [--variant fin\|best\|all]` | Kaggle Models | `MODEL_DIR/<handle>/<version>/` (sha256-checked) |
| 2. predict | `biohub-predict [--variant fin\|best]` | `TEST_DATA_DIR/<stem>.zarr`, `MODEL_DIR` | `SUBMISSION_DIR/submission.csv`, per-movie predictions in `WORK_DIR` |

`fin` (default) is the winning submission, `best` the second one.
`notebooks/inference.ipynb` runs the same code, locally or on Kaggle.

## Training

`DATA` is the competition data directory (`train/` with `<stem>.zarr` + `<stem>.geff`,
`test/`), `OUT` an output directory.

| step | command | reads | writes |
|---|---|---|---|
| train a model | `biohub-train --recipe recipes/train/<run>.json --train-dir DATA/train --competition-dir DATA --out-dir OUT/training [--pseudo-dir DIR] [--init-checkpoint PT]` | movies, optionally a pseudo-label set and an initial checkpoint | `OUT/training/<run>/{best,last}.pt`, `history.json` |
| make pseudo-labels | `biohub-pseudo-labels --ensemble recipes/ensembles/<set>-teacher.json --runs OUT/training --rescorers OUT/rescorers --train-dir DATA/train --competition-dir DATA --out OUT/pseudo_labels/<set>` | a teacher ensemble | `OUT/pseudo_labels/<set>/<stem>.npz` |
| fit re-scorers | `biohub-train-rescorer --ensemble recipes/ensembles/<spec>.json --runs OUT/training --names <one per model> --train-dir DATA/train --competition-dir DATA --out OUT/rescorers` | trained models, validation movies | `OUT/rescorers/<name>.npz` |
| everything, in order | `DATA=DATA OUT=OUT bash tools/reproduce_training.sh` | the competition data | all of the above, for the whole lineage of both submissions |

## Predicting with retrained models

`biohub-predict --ensemble recipes/ensembles/fin.json --runs OUT/training --rescorers OUT/rescorers`

## Smoke test

`python tools/smoke_test.py [--out DIR]` runs every step above on synthetic movies on CPU
and validates the resulting `submission.csv`.
