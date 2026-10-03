"""Teacher pseudo-labels merged into the sparse annotation.

The teacher is the whole inference pipeline (`biohub_tracking.labels`): its ILP
tracks on the training movies. Merge rules, ground truth always wins:

* a pseudo node within `match_um` of a GT node in the same frame (one-to-one)
  is that GT node;
* a pseudo edge into a GT node that has a GT parent, or out of a GT node that
  has GT children, is dropped;
* pseudo nodes below `min_node_prob` and edges below `min_edge_prob` are
  dropped; a pseudo node without a surviving parent edge has an unknown parent.

Each merged node carries a detection weight (1 for GT, `node_prob` for pseudo)
and a parent-label weight (1 for a GT edge, `edge_prob` for a pseudo edge).
GT and pseudo terms are normalised separately in the loss and combined as
`L_gt + lambda * L_pseudo` (`LossWeights.pseudo`). Division and velocity labels
stay GT-only (`division_ok`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from biohub_tracking.training.corpus import MovieTracks


@dataclass(frozen=True)
class PseudoLabels:
    """One movie's teacher tracks, flat arrays indexed 0..M-1."""

    t: np.ndarray  # int64 (M,)
    zyx: np.ndarray  # float64 (M,3) native voxels
    node_prob: np.ndarray  # float64 (M,) teacher centre probability
    parent: np.ndarray  # int64 (M,) index into these arrays, -1 for none
    edge_prob: np.ndarray  # float64 (M,) probability of the parent edge, 0 if none

    def __post_init__(self) -> None:
        m = len(self.t)
        for name in ("zyx", "node_prob", "parent", "edge_prob"):
            if len(getattr(self, name)) != m:
                raise ValueError(f"pseudo {name} has {len(getattr(self, name))} rows, t has {m}")
        linked = self.parent >= 0
        if np.any(self.parent >= m):
            raise ValueError("pseudo parent index out of range")
        if linked.any() and not np.all(self.t[linked] == self.t[self.parent[linked]] + 1):
            raise ValueError("pseudo edges must advance exactly one frame")


def save_pseudo(path: Path | str, labels: PseudoLabels, **meta) -> None:
    np.savez_compressed(
        path, t=labels.t, zyx=labels.zyx, node_prob=labels.node_prob,
        parent=labels.parent, edge_prob=labels.edge_prob,
        meta=np.asarray(repr(meta)),
    )


def load_pseudo(path: Path | str) -> PseudoLabels:
    with np.load(path) as data:
        return PseudoLabels(
            t=data["t"].astype(np.int64),
            zyx=data["zyx"].astype(np.float64).reshape(-1, 3),
            node_prob=data["node_prob"].astype(np.float64),
            parent=data["parent"].astype(np.int64),
            edge_prob=data["edge_prob"].astype(np.float64),
        )


def merge_pseudo(
    tracks: MovieTracks,
    pseudo: PseudoLabels,
    *,
    match_um: float = 4.0,
    min_node_prob: float = 0.5,
    min_edge_prob: float = 0.5,
) -> MovieTracks:
    """GT nodes 0..N-1 unchanged, then the surviving pseudo nodes."""
    n_gt = tracks.n_nodes
    spacing = np.asarray(tracks.spacing, dtype=np.float64)
    keep = pseudo.node_prob >= min_node_prob
    # pseudo index -> merged index (GT index when matched), -1 when dropped
    to_merged = np.full(len(pseudo.t), -1, dtype=np.int64)

    for frame in np.unique(pseudo.t[keep]):
        candidates = np.flatnonzero(keep & (pseudo.t == frame))
        gt_nodes = tracks.frame_nodes(int(frame))
        if len(gt_nodes) == 0 or len(candidates) == 0:
            continue
        distance = np.linalg.norm(
            (pseudo.zyx[candidates][:, None, :] - tracks.zyx[gt_nodes][None].astype(np.float64))
            * spacing, axis=-1,
        )
        rows, cols = linear_sum_assignment(np.where(distance <= match_um, distance, 1e9))
        ok = distance[rows, cols] <= match_um
        to_merged[candidates[rows[ok]]] = gt_nodes[cols[ok]]

    fresh = np.flatnonzero(keep & (to_merged < 0))
    to_merged[fresh] = n_gt + np.arange(len(fresh))

    t = np.concatenate((tracks.t, pseudo.t[fresh]))
    zyx = np.concatenate((tracks.zyx, np.rint(pseudo.zyx[fresh]).astype(np.int64)))
    parent = np.concatenate((tracks.parent, np.full(len(fresh), -1, dtype=np.int64)))
    children = [list(c) for c in tracks.children] + [[] for _ in fresh]
    gt_has_children = np.array([len(c) > 0 for c in tracks.children], dtype=bool)

    node_weight = np.concatenate((np.ones(n_gt), pseudo.node_prob[fresh]))
    parent_weight = np.concatenate((
        (tracks.parent >= 0).astype(np.float64), np.zeros(len(fresh)),
    ))
    is_gt = np.concatenate((np.ones(n_gt, bool), np.zeros(len(fresh), bool)))

    # Candidate edges in merged indices. Among pseudo-only nodes the teacher's
    # graph is already a forest; only edges touching a GT node need checks.
    child = np.flatnonzero(keep & (pseudo.parent >= 0) & (pseudo.edge_prob >= min_edge_prob))
    a = to_merged[pseudo.parent[child]]
    b = to_merged[child]
    alive = (a >= 0) & (b >= 0)
    child, a, b = child[alive], a[alive], b[alive]
    touches_gt = (a < n_gt) | (b < n_gt)

    accepted = np.zeros(len(child), dtype=bool)
    accepted[~touches_gt] = True
    parent[b[~touches_gt]] = a[~touches_gt]
    for k in np.flatnonzero(~touches_gt):
        children[a[k]].append(int(b[k]))
    for k in np.flatnonzero(touches_gt):
        ak, bk = int(a[k]), int(b[k])
        if bk < n_gt and tracks.parent[bk] >= 0:
            continue  # GT already says who b's parent is
        if ak < n_gt and gt_has_children[ak]:
            continue  # GT already says what a became
        if parent[bk] >= 0 or len(children[ak]) >= 2:
            continue  # two pseudo nodes matched one GT node's neighbourhood
        parent[bk] = ak
        children[ak].append(bk)
        accepted[k] = True
    parent_weight[b[accepted]] = pseudo.edge_prob[child[accepted]]
    parent_gt = np.concatenate((tracks.parent >= 0, np.zeros(len(fresh), bool)))
    extra_src, extra_dst = a[accepted], b[accepted]

    return replace(
        tracks,
        t=t, zyx=zyx, parent=parent, children=children,
        src=np.concatenate((tracks.src, extra_src)).astype(np.int64),
        dst=np.concatenate((tracks.dst, extra_dst)).astype(np.int64),
        node_weight=node_weight, parent_weight=parent_weight, is_gt=is_gt,
        parent_gt=parent_gt,
        division_ok=np.concatenate((gt_has_children, np.zeros(len(fresh), bool))),
    )
