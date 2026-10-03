# Model Summary

Biohub - Cell Tracking During Development: 6th place solution.
The full method description, with figures, is in [SOLUTION.md](SOLUTION.md).

## A1. Background

| | |
|---|---|
| Competition | Biohub - Cell Tracking During Development |
| Private leaderboard | 0.953, 6th place (selected submission `fin`); the second submission `best` scored 0.955 |
| Public leaderboard | 0.961 (`fin`), 0.958 (`best`) |
| Author | Jamal Saeedi |
| Code | https://github.com/jamal-saeedi/biohub_cell_tracking_kaggle (MIT) |
| Models | https://www.kaggle.com/models/jamalsaeedi/biohub-cell-tracking (MIT) |

## A3. Summary

The solution is a learned tracker in three parts.

1. **Six 3D networks** of two architectures (three IsotropicLineageNet, three
   MultiScaleLineageNet, 2.5–5.7 M parameters) each detect cells *and* link them in one
   pass. They read 3-frame windows at native resolution, predict cell centres with
   sub-voxel offsets, and score every candidate link to the previous frame with a sparse
   graph-attention head: for every cell, a softmax over its candidate parents plus an
   explicit "new cell" class, a division score and a velocity.
2. **Ensemble.** The six centre maps are averaged into one set of cells. Each network
   scores the same candidate links with its own head and its own LightGBM re-scorer; the
   six link distributions are averaged.
3. **Global ILP** over the whole movie (cells, links, divisions, appearances,
   disappearances), solved LP-first, with a stage-drift-corrected link distance, followed
   by drift-compensated smoothing of the positions.

Only about 3 % of the cells are annotated, so training uses three-state detection targets
(positive, verified background, unknown), a count prior, and several **teacher → student
rounds in which the teacher is the whole pipeline**: its tracks on the training movies
become pseudo-labels for the next models. Tools: PyTorch, LightGBM, HiGHS / SCIP, numba.
Training: 4–17 h per model on one RTX 3090/4090. Inference: about 10 hours for the hidden
test set on Kaggle's 2× T4.

## A4. Features selection / engineering

- **The networks** take the raw volumes, each frame min–max scaled and z-scored on its
  own. No hand-made image features.
- **Edge geometry** inside the association head, 8 per candidate link: displacement (3),
  distance (1), displacement minus predicted velocity divided by the predicted
  uncertainty (3), elapsed frames (1).
- **The re-scorers** use 23 features per candidate link: the network's log-probability,
  rank, margin to the best rival, number of candidates and "new cell" probability; raw and
  drift-corrected distances split into z and lateral parts; velocity residuals; centre
  logits of both cells and the parent's division logit; competition for the parent (its
  best score elsewhere, how many cells rank it first, its number of candidates); local
  density; depth and relative time.
- **Candidate links:** the 4 nearest cells each way between consecutive frames, within
  20 µm (the true parent is a candidate for about 99.4 % of annotated links).

## A5. Training method(s)

- **Detection targets:** Gaussian targets within 3 µm of annotated centres; verified
  background (darker than the frame median and more than 6 µm from every annotation);
  everything else carries no loss. A count prior pulls the summed heatmap towards the
  organisers' per-movie cell-count estimate.
- **Losses:** penalty-reduced focal loss on centres, smooth L1 on sub-voxel offsets,
  cross-entropy over candidate parents + "new cell", division BCE, daughter-pair BCE,
  velocity Gaussian NLL, count prior. Centre and parent losses are computed separately on
  ground truth and pseudo-labels, L = L_GT + 0.5 · L_pseudo.
- **Linking samples:** the same k-NN candidate graphs as at inference; bright unannotated
  intensity peaks as distractors; 15 % of annotated parents removed and labelled
  "new cell".
- **Augmentation:** lateral D4, gamma, noise, synthetic stage drift, noisy-student noise
  (shot noise, depth gain, channel dropout), low-contrast haze.
- **Optimisation:** AdamW, warm-up + cosine decay, gradient clipping, EMA of the weights
  (0.999), bf16 mixed precision; 24–48 epochs of 2,048 full-frame 3-frame crops.
- **Pseudo-labels:** the teacher pipeline tracks every training movie; labels are the
  ILP topology with the raw detected positions, aligned per imaging cohort, with the
  teacher's probabilities as loss weights. Ground truth always wins. Seven label sets
  were made, the last two by 3-model ensemble teachers.
- **Splits:** every model has its own random movie-disjoint split (165 training, 30
  validation movies); the 4 public test movies are never trained on.
- **Ensembling:** equal-weight mean of the six models' centre logits (cells) and of their
  re-scored link probabilities; 2–4 test-time rotations per model (4/4/4/3/2/2).
- Every model has a recipe in `recipes/train/`; `tools/reproduce_training.sh` runs the
  whole lineage.

## A6. Interesting findings

- **Most linking errors are identity swaps between neighbours** (51–57 % of edge errors
  on validation), concentrated on frames where the microscope stage drifts. Correcting
  the link distance for drift was worth +0.009 on validation, drift-compensated
  smoothing +0.005.
- **An explicit "new cell" class makes post-processing unnecessary:** with every cell and
  every appearance priced in the ILP, gap closing and fragment repair gave +0.0000.
- **Validation loss does not rank tracking quality:** a model with better validation
  loss and recall scored lower end to end; every change was ranked on the tracking metric.
- **Test-time augmentation was the largest single gain** (+0.05 score, 1 → 4 views), because
  the averaged features also feed the linker.
- **The T4 has no native bf16:** switching from emulated bf16 to fp16 made prediction
  3.5× faster, which is what let the six-model ensemble fit in 12 hours.
- **Results are bit-identical on the same GPU architecture,** which made comparisons
  between changes exact; the Kaggle notebook reproduces both submissions byte for byte.
- Things that did not help: probability pruning of candidate links, a same-view mean
  teacher, weight averaging of models, temperature and cost-scale calibration (the
  association head is already calibrated).

## A7. Simple features and methods

A single model with the same solver scored 0.954 on the public leaderboard, against
0.961 for the six-model ensemble. It predicts the 4 public test movies in 5.4 minutes on
one T4, against about 15 minutes on two T4s for the ensemble.

## A8. Model execution time

| | time | hardware |
|---|---|---|
| training, one isotropic model | 4–10 h | 1× RTX 3090/4090 |
| training, one MultiScale model | about 17 h | 1× RTX 3090/4090 |
| whole training lineage (14 models, 7 pseudo-label sets, re-scorers) | about a week of single-GPU time | 1× RTX 3090/4090 |
| inference, 4 public test movies, six-model ensemble | about 15 min | Kaggle 2× T4 |
| inference, hidden test set | about 10 h | Kaggle 2× T4 |
| inference, single model | 5.4 min for the 4 public test movies | 1× T4 |

## A9. References

See the [References](SOLUTION.md#references) of the write-up.

## B. Submission model

| guideline item | where |
|---|---|
| code, data handling and trained models | this repository; models on [Kaggle Models](https://www.kaggle.com/models/jamalsaeedi/biohub-cell-tracking) (`tools/download_models.py`) |
| README (hardware, OS, software, how to train and predict) | [README.md](../README.md) |
| configuration files | `recipes/train/*.json`, `recipes/ensembles/*.json` |
| requirements.txt | [requirements.txt](../requirements.txt) |
| directory_structure.txt | [directory_structure.txt](../directory_structure.txt) |
| SETTINGS.json | [SETTINGS.json](../SETTINGS.json) |
| serialized trained models | Kaggle Models: variations `mres-fin-444322` (winning) and `mres-best-444322` |
| entry_points.md | [entry_points.md](../entry_points.md) |
