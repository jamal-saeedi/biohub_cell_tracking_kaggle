# Biohub cell tracking — 6th place solution (inference)

Inference code for our 6th-place solution to the Kaggle competition
[Biohub Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).
It predicts a cell lineage graph (nodes = cells, edges = frame-to-frame links
and divisions) for every 3D time-lapse movie and writes `submission.csv`.

| Variant | Model set | Private LB | Public LB |
|---|---|---|---|
| `fin` (default) | final selected submission | **0.953** | 0.961 |
| `best` | best-validation checkpoints | 0.955 | 0.958 |

Both variants reproduce their Kaggle submissions byte for byte on the 4
public test movies (Kaggle 2×T4).

## Links

| | |
|---|---|
| Models (MIT) | https://www.kaggle.com/models/jamalsaeedi/biohub-cell-tracking |
| Code dataset (package + pinned wheels) | https://www.kaggle.com/datasets/jamalsaeedi/biohub-cell-tracking-kaggle |
| Kaggle notebook | https://www.kaggle.com/code/jamalsaeedi/biohub-cell-tracking-inference |

## Method

1. **Detection and linking networks.** Six 3D networks (three
   `IsotropicLineageNet`, three `MultiScaleLineageNet`) read a 3-frame window
   and output, for every frame, cell-centre heat-maps and, for every detected
   cell, an association over its candidate parents in the previous frame
   (link, no-parent and division logits). EMA weights, fp16 inference.
2. **Test-time augmentation.** Flip / transpose views per network
   (4, 4, 4, 3, 2, 2), with the encoder output cached across overlapping
   windows.
3. **Ensemble.** Detections are averaged across networks. Each network's
   candidate links are re-scored by its own gradient-boosted edge re-scorer
   (23 geometric and model features); the re-scored probabilities are
   averaged.
4. **Global solve.** An event ILP (appearance, disappearance, link, division)
   over the whole movie with [tracksdata](https://github.com/royerlab/tracksdata)
   and SCIP, solved LP-first; the link cost includes a stage-drift-corrected
   distance. A greedy solution is the fallback.
5. **Post-processing.** Drift-compensated line-fit smoothing of positions,
   coordinates clamped to the volume.
6. **Time safety.** A per-movie time budget against the 12 h limit; a movie
   that fails is re-run with fewer views, then with the primary network
   only, then with model-free tracking, so every movie gets a prediction.

## Layout

```
├── LICENSE                 MIT
├── SETTINGS.json           every input / output path (local and Kaggle)
├── pyproject.toml          package + pinned dependencies
├── notebooks/inference.ipynb
├── src/biohub_tracking/
│   ├── cli.py              biohub-predict
│   ├── recipe.py           the two variants, file hashes, shipped config
│   ├── settings.py         SETTINGS.json loader
│   ├── isotropic/          config, decoding, ensemble, re-scorer, ILP, pipeline, sharding
│   ├── models/             network definitions
│   └── postprocess/        smoothing
└── tools/
    ├── download_models.py      models from Kaggle Models into MODEL_DIR
    ├── build_kaggle_dataset.py the Kaggle code dataset
    └── kernel-metadata.json    the Kaggle notebook
```

## Run locally

Requirements: Linux, Python 3.12, an NVIDIA GPU with 16 GB (V100 and T4
tested).

```bash
pip install -e ".[download]"
python tools/download_models.py            # both variants into models/
```

Put the test movies (`<stem>.zarr`) in `data/test/`, or edit `SETTINGS.json`:

| Key | Default | |
|---|---|---|
| `TEST_DATA_DIR` | `data/test` | input movies |
| `MODEL_DIR` | `models` | model files (`<MODEL_DIR>/<variant>/<version>/...`) |
| `SUBMISSION_DIR` | `outputs` | `submission.csv` |
| `WORK_DIR` | `outputs/work` | per-movie predictions |

Then either

```bash
biohub-predict                       # variant fin
biohub-predict --variant best
```

or run `notebooks/inference.ipynb` (`VARIANT` in its first cell). Both run the
same code. Options: `--movies STEM ...`, `--num-shards N` (GPU processes;
default one per GPU), `--settings PATH` (or `BIOHUB_SETTINGS`).

On one V100 the 4 public test movies take about 13 minutes.

## Run on Kaggle

Open the [notebook](https://www.kaggle.com/code/jamalsaeedi/biohub-cell-tracking-inference)
and *Copy & Edit*, or push it with the Kaggle CLI:

```bash
kaggle kernels push -p tools
```

It attaches the competition data, the code dataset and both model variations,
installs the pinned wheels offline (internet off) and writes
`/kaggle/working/submission.csv`. Accelerator: GPU T4 ×2.

## Environment

The submissions ran in the Kaggle Python image
`gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`
(torch 2.10.0+cu128, numpy 2.0.2, scipy 1.16.3, pandas 2.3.3, Python 3.12)
plus the wheels in the code dataset (numba 0.65.1, polars 1.42.0,
pyscipopt 6.2.1, ilpy 0.6.0, zarr 3.2.1, tracksdata 0.1.0rc6.dev3+g980c2d30a).
`pyproject.toml` pins the same versions. Results are deterministic for a given
GPU architecture; on a V100 the coordinates differ slightly from the T4 runs
and the score on the public test movies is the same.

## Training

Training code and pseudo-label generation will be added in a later release.

## License

MIT, see [LICENSE](LICENSE).
