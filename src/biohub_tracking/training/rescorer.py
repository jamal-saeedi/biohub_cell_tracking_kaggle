"""Member-own edge re-scorers: one gradient-boosted tree ensemble per ensemble model.

    biohub-train-rescorer --ensemble recipes/ensembles/fin.json --runs outputs/training \\
        --names MRES-FIN-444322-m0 ... --train-dir data/train --out outputs/rescorers

For every movie the ensemble decodes the shared nodes (as at inference), and
each model scores the shared candidate edges with its own association head.
Rows are the candidate parents of every target whose annotated cell and
annotated parent both map one-to-one onto decoded cells (within `map_um`) and
whose true parent is among its candidates: label 1 on the true parent, 0 on the
others. Model k's trees are fitted on model k's own scores (`edge_rescorer.FEATURES`)
and exported as the flat arrays inference evaluates.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from biohub_tracking.isotropic.edge_rescorer import (
    FEATURES,
    TreeEnsemble,
    edge_features,
)

#: LightGBM settings of the shipped re-scorers.
LIGHTGBM_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
    "min_data_in_leaf": 50, "bagging_fraction": 0.8, "bagging_freq": 1,
    "feature_fraction": 0.8, "verbose": -1, "seed": 0,
}
TREES = 300


def greedy_map(gt_um: np.ndarray, cand_um: np.ndarray, radius: float) -> dict[int, int]:
    """One-to-one GT index -> candidate index, nearest pairs first, within `radius`."""
    if not len(gt_um) or not len(cand_um):
        return {}
    pairs = cKDTree(gt_um).sparse_distance_matrix(cKDTree(cand_um), radius, output_type="ndarray")
    out: dict[int, int] = {}
    used: set[int] = set()
    for i, j, _ in sorted(pairs.tolist(), key=lambda p: p[2]):
        i, j = int(i), int(j)
        if i in out or j in used:
            continue
        out[i] = j
        used.add(j)
    return out


def labelled_rows(features: np.ndarray, edge_index: np.ndarray, coords: np.ndarray,
                  tracks, spacing, map_um: float = 7.0) -> tuple[np.ndarray, np.ndarray] | None:
    """`(X, y)` rows of one movie and one model (see the module docstring)."""
    scale = np.asarray(spacing, dtype=np.float64)
    to_cand = np.full(len(tracks.t), -1)
    for t in np.unique(tracks.t):
        gi = np.flatnonzero(tracks.t == t)
        ci = np.flatnonzero(coords[:, 0] == t)
        for a, b in greedy_map(tracks.zyx[gi] * scale, coords[ci, 1:] * scale, map_um).items():
            to_cand[gi[a]] = ci[b]
    src, tgt = edge_index
    by_tgt: dict[int, list[int]] = {}
    for i, j in enumerate(tgt.tolist()):
        by_tgt.setdefault(j, []).append(i)
    rows, labels = [], []
    for s, d in zip(tracks.src, tracks.dst):
        cs, cd = to_cand[s], to_cand[d]
        if cs < 0 or cd < 0 or cd not in by_tgt:
            continue
        k = np.array(by_tgt[cd])
        hit = src[k] == cs
        if not hit.any():
            continue
        rows.append(k)
        labels.append(hit.astype(np.int8))
    if not rows:
        return None
    k = np.concatenate(rows)
    return features[k], np.concatenate(labels)


def fit_rescorer(X: np.ndarray, y: np.ndarray, *, trees: int = TREES, jobs: int = 8) -> TreeEnsemble:
    """Fit LightGBM and convert it, checking the two agree on every training row."""
    import lightgbm as lgb

    params = dict(LIGHTGBM_PARAMS, num_threads=jobs)
    booster = lgb.train(params, lgb.Dataset(X, y, feature_name=list(FEATURES)), trees)
    ensemble = TreeEnsemble.from_lightgbm_dump(booster.dump_model())
    gap = np.abs(ensemble.predict(X) - booster.predict(X, raw_score=True)).max()
    if gap > 1e-9:
        raise RuntimeError(f"exported trees disagree with LightGBM by {gap:.3g}")
    return ensemble


def member_rows(loaded, ds_path: Path, config, tracks, map_um: float = 7.0) -> list:
    """Per model, the labelled `(X, y)` rows of one movie (None when it has none)."""
    from biohub_tracking.isotropic.predict import member_edge_terms, movie_terms

    terms = movie_terms(loaded, ds_path, config, keep_members=True)
    out = []
    for k in range(1 + len(loaded.ensemble)):
        edge_index, logp, null, division, velocity = member_edge_terms(
            terms.member_pairs, k, terms.offset, terms.total)
        features = edge_features(terms.coords, edge_index, logp, null, terms.node_logit,
                                 division, velocity, terms.movie.spacing)
        out.append(labelled_rows(features, edge_index, terms.coords, tracks,
                                 terms.movie.spacing, map_um))
    return out


def train_rescorers(config, train_dir: Path, movies: list[str], out_dir: Path, names: list[str],
                    *, device=None, map_um: float = 7.0, trees: int = TREES,
                    jobs: int = 8) -> list[Path]:
    """Decode `movies` with the ensemble of `config` and write one tree file per model."""
    import torch

    from biohub_tracking.isotropic.predict import load_for_inference
    from biohub_tracking.training.corpus import load_tracks

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = replace(config, ensemble_rescorers=(), edge_rescorer=None)
    loaded = load_for_inference(config, device)
    models = 1 + len(loaded.ensemble)
    if len(names) != models:
        raise ValueError(f"{len(names)} names for {models} models")
    rows: list[list] = [[] for _ in range(models)]
    for index, stem in enumerate(movies):
        tracks = load_tracks(train_dir, stem)
        for k, r in enumerate(member_rows(loaded, Path(train_dir) / f"{stem}.zarr", config,
                                          tracks, map_um)):
            if r is not None:
                rows[k].append(r)
        print(f"[rescorer] {index + 1}/{len(movies)} {stem}: "
              f"{sum(len(r[1]) for r in rows[0])} rows so far", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for k, name in enumerate(names):
        if not rows[k]:
            raise RuntimeError(f"model {k}: no labelled rows")
        X = np.concatenate([r[0] for r in rows[k]])
        y = np.concatenate([r[1] for r in rows[k]])
        path = out_dir / f"{name}.npz"
        fit_rescorer(X, y, trees=trees, jobs=jobs).save(path)
        print(f"[rescorer] {path}: {len(y)} rows, {int(y.sum())} true parents", flush=True)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    from biohub_tracking.ensembles import EnsembleSpec, ensemble_config
    from biohub_tracking.training.splits import build_split_from_dir

    p = argparse.ArgumentParser(prog="biohub-train-rescorer", description=__doc__.splitlines()[0])
    p.add_argument("--ensemble", type=Path, required=True,
                   help="ensemble spec (recipes/ensembles); its own re-scorers are not used")
    p.add_argument("--runs", type=Path, required=True, help="directory of <run>/best.pt")
    p.add_argument("--names", nargs="+", required=True,
                   help="one output name per model, primary first (e.g. MRES-FIN-444322-m0 ...)")
    p.add_argument("--train-dir", type=Path, required=True, help="movies with .zarr + .geff")
    p.add_argument("--competition-dir", type=Path, default=None)
    p.add_argument("--movies", nargs="+", default=None,
                   help="default: the validation movies of split seed --split-seed")
    p.add_argument("--split-seed", type=int, default=314159)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--map-um", type=float, default=7.0)
    p.add_argument("--trees", type=int, default=TREES)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args(argv)

    spec = EnsembleSpec.load(args.ensemble)
    spec = EnsembleSpec(spec.runs, spec.member_tta_views, (), spec.overrides)
    config = ensemble_config(spec, args.runs)
    movies = args.movies
    if movies is None:
        competition = args.competition_dir or args.train_dir.parent
        movies = list(build_split_from_dir(args.train_dir, competition, seed=args.split_seed,
                                           fold_test=True).val)
    train_rescorers(config, args.train_dir, movies, args.out, args.names, map_um=args.map_um,
                    trees=args.trees, jobs=args.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
