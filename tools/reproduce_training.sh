#!/usr/bin/env bash
# The whole training lineage of the shipped models, step by step.
#
#   DATA=data OUT=outputs bash tools/reproduce_training.sh
#
# DATA holds the competition data (train/ with <stem>.zarr + <stem>.geff for all
# 199 movies, test/ with the 4 public test movies). Every step needs one GPU;
# the steps are sequential except where noted, and each skips its work when its
# output exists. Training times on one RTX 3090/4090: 4-10 h per isotropic run,
# about 17 h per MultiScale run; each pseudo-label set needs one decode of the
# 195 training movies.
set -euo pipefail

DATA=${DATA:-data}
OUT=${OUT:-outputs}
RUNS=$OUT/training
LABELS=$OUT/pseudo_labels
TREES=$OUT/rescorers
SPLIT_A=314159   # model A's split
SPLIT_B=271828   # NS1's split

train() {  # train RECIPE [biohub-train options]
    local name=$1; shift
    [ -s "$RUNS/$name/last.pt" ] && return
    biohub-train --recipe "recipes/train/$name.json" --train-dir "$DATA/train" \
        --competition-dir "$DATA" --out-dir "$RUNS" "$@"
}

labels() {  # labels SET [biohub-pseudo-labels options]
    local name=$1; shift
    [ -d "$LABELS/$name" ] && return
    biohub-pseudo-labels --ensemble "recipes/ensembles/$name-teacher.json" --runs "$RUNS" \
        --rescorers "$TREES" --train-dir "$DATA/train" --competition-dir "$DATA" \
        --out "$LABELS/$name" "$@"
}

rescorers() {  # rescorers ENSEMBLE SPLIT_SEED NAME... (one tree file per model, primary first)
    local ensemble=$1 seed=$2; shift 2
    [ -s "$TREES/${!#}.npz" ] && return
    biohub-train-rescorer --ensemble "recipes/ensembles/$ensemble.json" --runs "$RUNS" \
        --names "$@" --split-seed "$seed" --train-dir "$DATA/train" --competition-dir "$DATA" \
        --out "$TREES"
}

# Round 0: ground truth only.
train W-link-s2

# Rounds 1-3: one-model teachers, each student fine-tuned or trained on its teacher's labels.
labels W-link-s2
train P-l50-s2 --pseudo-dir "$LABELS/W-link-s2"
labels P-l50-s2
train R2-ft24-noisy-s1 --pseudo-dir "$LABELS/P-l50-s2" --init-checkpoint "$RUNS/P-l50-s2/best.pt"
labels R2-ft24-noisy-s1
train R3-ft24-noisy-lc-s1 --pseudo-dir "$LABELS/R2-ft24-noisy-s1" \
    --init-checkpoint "$RUNS/R2-ft24-noisy-s1/best.pt"                       # model A

# Students on new splits and new architectures.
labels R3-ft24-noisy-lc-s1-full
train NS1-ft24-noisy-lc-s1 --pseudo-dir "$LABELS/R3-ft24-noisy-lc-s1-full" \
    --init-checkpoint "$RUNS/R3-ft24-noisy-lc-s1/best.pt"
# NS1's labels use NS1 with its own re-scorer, fitted on its own validation movies.
rescorers NS1-ft24-noisy-lc-s1-teacher $SPLIT_B NS1-ft24-noisy-lc-s1
labels NS1-ft24-noisy-lc-s1 --split-seed $SPLIT_B
train BX-wide-lc-s2 --pseudo-dir "$LABELS/NS1-ft24-noisy-lc-s1"            # also keeps best-e23.pt
train CX-deep-lc-s3 --pseudo-dir "$LABELS/R3-ft24-noisy-lc-s1-full"        # also keeps best-e20.pt

# Ensemble teacher v10: A + wide (epoch 23) + deep (epoch 20), member-own trees.
rescorers v10-teacher $SPLIT_A MRES-A3s-m0 MRES-A3s-m1 MRES-A3s-m2
labels v10
train B2-v10-e48-noisy-lc-s22 --pseudo-dir "$LABELS/v10" --init-checkpoint "$RUNS/BX-wide-lc-s2/best-e23.pt"
train DX-widedeep-e48-lc-s4 --pseudo-dir "$LABELS/v10"

# Ensemble teacher v11: A + fully trained wide + deep.
rescorers v11-teacher $SPLIT_A MRES-A3-m0 MRES-A3-m1 MRES-A3-m2
labels v11

# The five other members of the final ensemble (independent; may run in parallel).
train B3-v11-ft32-noisy-lc-s31 --pseudo-dir "$LABELS/v11" --init-checkpoint "$RUNS/B2-v10-e48-noisy-lc-s22/last.pt"
train D2-v11-ft32-noisy-lc-s33 --pseudo-dir "$LABELS/v11" --init-checkpoint "$RUNS/DX-widedeep-e48-lc-s4/last.pt"
train EX-ms-lc-s5 --pseudo-dir "$LABELS/v11"
train FX-ms-lc-s6 --pseudo-dir "$LABELS/v11"
train GX-ms-lc-s7 --pseudo-dir "$LABELS/v11"

# Member-own re-scorers of the two shipped ensembles.
rescorers fin $SPLIT_A MRES-FIN-444322-m{0,1,2,3,4,5}
rescorers best $SPLIT_A MRES-BEST-444322-m{0,1,2,3,4,5}

echo "done: predict with"
echo "  biohub-predict --ensemble recipes/ensembles/fin.json --runs $RUNS --rescorers $TREES"
