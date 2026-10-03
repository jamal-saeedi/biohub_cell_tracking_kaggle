"""Teacher pseudo-labels: the whole inference pipeline's tracks on the training movies.

    biohub-pseudo-labels --ensemble recipes/ensembles/v11-teacher.json \\
        --runs outputs/training --rescorers outputs/rescorers \\
        --train-dir data/train --competition-dir data --out outputs/pseudo_labels/v11

Per movie: the teacher decodes, links, re-scores and solves exactly as at
inference. The labels are the ILP's topology with the decoded (unsmoothed)
positions; each node carries the teacher's centre probability and each edge its
re-scored parent probability, which the trainer uses as loss weights.

Positions are aligned to the annotation per cohort (`44b6` / `6bba`): the mean
signed offset between the teacher's cells and the matched annotated cells is
measured on the `--align-movies` (default: the validation movies of
`--split-seed`) and subtracted. The public test movies are never labelled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub_tracking.training.pseudo import PseudoLabels, save_pseudo


def labels_from_graph(graph, solver) -> PseudoLabels:
    """`PseudoLabels` of a solved graph (`predict_movie`): raw positions, the
    centre probability from the node cost, the parent edge's probability."""
    nodes = graph.node_attrs()
    ids = nodes["node_id"].to_numpy()
    order = np.argsort(ids, kind="stable")
    ids = ids[order]
    row = {int(i): k for k, i in enumerate(ids)}
    logit = -nodes["node_cost"].to_numpy()[order] / solver.node_weight_scale + solver.node_logit_bias
    parent = np.full(len(ids), -1, dtype=np.int64)
    edge_prob = np.zeros(len(ids))
    if graph.num_edges():
        edges = graph.edge_attrs()
        for s, t, p in zip(edges["source_id"].to_numpy(), edges["target_id"].to_numpy(),
                           edges["edge_prob"].to_numpy()):
            k = row[int(t)]
            if parent[k] >= 0:
                raise ValueError(f"node {int(t)} has two parents")
            parent[k] = row[int(s)]
            edge_prob[k] = float(p)
    return PseudoLabels(
        t=nodes["t"].to_numpy()[order].astype(np.int64),
        zyx=np.column_stack([nodes[a].to_numpy()[order] for a in ("z", "y", "x")]).astype(np.float64),
        node_prob=1.0 / (1.0 + np.exp(-logit)),
        parent=parent,
        edge_prob=edge_prob,
    )


def cohort_offsets(labelled: dict[str, PseudoLabels], train_dir: Path,
                   match_um: float = 4.0) -> dict[str, np.ndarray]:
    """Per-cohort mean signed (label - annotation) offset in native voxels, over
    one-to-one matches within `match_um` (as `pseudo.merge_pseudo` matches)."""
    from biohub_tracking.training.corpus import load_tracks

    found: dict[str, list] = {}
    for stem, labels in labelled.items():
        gt = load_tracks(train_dir, stem)
        spacing = np.asarray(gt.spacing)
        for frame in np.unique(gt.t):
            g = gt.frame_nodes(int(frame))
            p = np.flatnonzero(labels.t == frame)
            if not len(p):
                continue
            d = np.linalg.norm((labels.zyx[p][:, None] - gt.zyx[g][None]) * spacing, axis=-1)
            rows, cols = linear_sum_assignment(np.where(d <= match_um, d, 1e9))
            ok = d[rows, cols] <= match_um
            found.setdefault(stem[:4], []).append(labels.zyx[p[rows[ok]]] - gt.zyx[g[cols[ok]]])
    return {cohort: np.concatenate(v).mean(0) for cohort, v in found.items()}


def teacher_labels(config, train_dir: Path, stems: list[str], *,
                   device=None) -> dict[str, PseudoLabels]:
    """Run the teacher pipeline on `stems` and return its labels (unaligned)."""
    import torch

    from biohub_tracking.isotropic.predict import load_for_inference, predict_movie

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_for_inference(config, device)
    out = {}
    for index, stem in enumerate(stems):
        prediction = predict_movie(loaded, Path(train_dir) / f"{stem}.zarr", config)
        out[stem] = labels_from_graph(prediction.graph, config.solver)
        labels = out[stem]
        print(f"[labels] {index + 1}/{len(stems)} {stem}: {len(labels.t)} cells, "
              f"{int((labels.parent >= 0).sum())} links", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    from biohub_tracking.ensembles import (
        EnsembleSpec,
        apply_overrides,
        ensemble_config,
        parse_set,
    )
    from biohub_tracking.training.corpus import movie_stems
    from biohub_tracking.training.splits import build_split_from_dir, pinned_test_stems

    p = argparse.ArgumentParser(prog="biohub-pseudo-labels", description=__doc__.splitlines()[0])
    p.add_argument("--ensemble", type=Path, required=True, help="teacher spec (recipes/ensembles)")
    p.add_argument("--runs", type=Path, required=True, help="directory of <run>/best.pt")
    p.add_argument("--rescorers", type=Path, default=None, help="directory of <name>.npz")
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--competition-dir", type=Path, default=None,
                   help="directory with test/ (those movies are never labelled)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--movies", nargs="+", default=None,
                   help="movies to label (default: every annotated movie except the test movies)")
    p.add_argument("--align-movies", nargs="*", default=None,
                   help="movies the cohort offsets are measured on (default: the validation "
                        "movies of --split-seed; none: no alignment)")
    p.add_argument("--split-seed", type=int, default=314159)
    p.add_argument("--set", nargs="*", default=[], metavar="SECTION.KEY=VALUE",
                   help="further teacher config overrides")
    args = p.parse_args(argv)

    config = ensemble_config(EnsembleSpec.load(args.ensemble), args.runs, args.rescorers)
    config = apply_overrides(config, parse_set(args.set))
    competition = args.competition_dir or args.train_dir.parent
    pinned = set(pinned_test_stems(competition))
    stems = args.movies or [s for s in movie_stems(args.train_dir) if s not in pinned]
    if pinned & set(stems):
        p.error(f"refusing to label test movies: {sorted(pinned & set(stems))}")
    align = args.align_movies
    if align is None:
        align = list(build_split_from_dir(args.train_dir, competition, seed=args.split_seed,
                                          fold_test=True).val)

    labelled = teacher_labels(config, args.train_dir, sorted(set(stems) | set(align)))
    offsets = cohort_offsets({s: labelled[s] for s in align}, args.train_dir) if align else {}
    for cohort, offset in offsets.items():
        print(f"[labels] align {cohort}: subtract {np.round(offset, 3)} native voxels", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        labels = labelled[stem]
        offset = offsets.get(stem[:4])
        if align and offset is None:
            raise SystemExit(f"no aligned movie of cohort {stem[:4]} for {stem}")
        if offset is not None:
            labels = PseudoLabels(labels.t, labels.zyx - offset, labels.node_prob,
                                  labels.parent, labels.edge_prob)
        save_pseudo(args.out / f"{stem}.npz", labels, ensemble=str(args.ensemble),
                    offset_native=None if offset is None else offset.tolist())
    print(f"[labels] wrote {len(stems)} files to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
