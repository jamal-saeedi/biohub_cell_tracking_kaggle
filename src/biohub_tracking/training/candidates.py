"""Candidate graphs and three-state association labels.

The annotated cells are sparse, so distractor nodes (bright peaks without an
annotation) are mixed in to give the linker competitors:

* a distractor as a source is a verified wrong parent for an annotated target;
* a distractor as a target has an unknown parent: context, no parent loss.

Labels: >= 0 is a class, -1 the explicit null (no parent), -2 unknown (no loss).
`n_gt_edges_in_graph / n_gt_edges` is the candidate recall; `n_forced` counts
true edges added because the k-NN graph missed them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNKNOWN = -2
NULL = -1


@dataclass(frozen=True)
class NodeSet:
    """Nodes of one frame, in crop-relative native voxels."""

    coords_native: np.ndarray  # (N,3) float64
    gt_index: np.ndarray  # (N,) int64 index into MovieTracks, -1 for a distractor

    def __len__(self) -> int:
        return int(len(self.gt_index))

    def um(self, spacing: tuple[float, float, float]) -> np.ndarray:
        return self.coords_native * np.asarray(spacing, dtype=np.float64)


@dataclass
class AssociationTargets:
    edge_index: np.ndarray  # (2,E) int64, unique, in range
    parent_edge: np.ndarray  # (n_tgt,) edge position of the true parent, NULL, UNKNOWN
    division_label: np.ndarray  # (n_src,) float32 in {0,1}
    division_mask: np.ndarray  # (n_src,) float32
    velocity_um: np.ndarray  # (n_src,3) float32, microns per interval
    velocity_mask: np.ndarray  # (n_src,) float32
    triplets: np.ndarray  # (P,3) int64 [parent, daughter_a, daughter_b]
    triplet_label: np.ndarray  # (P,) float32
    n_gt_edges: int
    n_gt_edges_in_graph: int
    n_forced: int
    n_null_targets: int
    #: (n_tgt,) float32 weight of each target's parent label; 0 where UNKNOWN.
    parent_weight: np.ndarray | None = None
    #: (n_tgt,) bool: the label (parent or null) came from a pseudo edge.
    parent_pseudo: np.ndarray | None = None


def build_candidate_edges(
    source_um: np.ndarray,
    target_um: np.ndarray,
    *,
    radius_um: float,
    max_per_node: int,
) -> np.ndarray:
    """Union of the k nearest sources of each target and the k nearest targets
    of each source, within `radius_um`: `(2,E)` int64, unique."""
    if len(source_um) == 0 or len(target_um) == 0:
        return np.empty((2, 0), dtype=np.int64)
    distance = np.linalg.norm(source_um[:, None, :] - target_um[None, :, :], axis=-1)
    pairs: list[np.ndarray] = []
    for axis in (0, 1):
        # axis 0: rank sources for each target; axis 1: rank targets per source.
        order = np.argsort(distance, axis=axis, kind="stable")
        keep = min(max_per_node, distance.shape[axis])
        picked = np.take(order, np.arange(keep), axis=axis)
        if axis == 0:
            rows, cols = picked, np.broadcast_to(
                np.arange(distance.shape[1]), picked.shape
            )
        else:
            rows, cols = (
                np.broadcast_to(np.arange(distance.shape[0])[:, None], picked.shape),
                picked,
            )
        rows, cols = rows.reshape(-1), cols.reshape(-1)
        within = distance[rows, cols] <= radius_um
        pairs.append(np.stack((rows[within], cols[within])))
    return _unique_edges(np.concatenate(pairs, axis=1))


def _unique_edges(edges: np.ndarray) -> np.ndarray:
    if edges.shape[1] == 0:
        return np.empty((2, 0), dtype=np.int64)
    flat = np.unique(edges[0].astype(np.int64) * (edges[1].max() + 1) + edges[1])
    stride = edges[1].max() + 1
    return np.stack((flat // stride, flat % stride)).astype(np.int64)


def build_association_targets(
    sources: NodeSet,
    targets: NodeSet,
    *,
    parent_of: np.ndarray,
    children_of: list[list[int]],
    spacing: tuple[float, float, float],
    radius_um: float = 12.0,
    max_per_node: int = 8,
    dropped_parents: frozenset[int] = frozenset(),
    max_triplet_negatives: int = 4,
    dt: float = 1.0,
    rng: np.random.Generator | None = None,
    label_weight: np.ndarray | None = None,
    division_ok: np.ndarray | None = None,
    label_gt: np.ndarray | None = None,
) -> AssociationTargets:
    """Candidate edges plus parent / division / velocity / daughter-pair labels.

    `parent_of` / `children_of` are indexed by the movie's node index
    (`NodeSet.gt_index`). Only parents in `dropped_parents` (removed on purpose)
    make their child's label the explicit null; a parent that is merely absent
    leaves it unknown.

    Movie-level options: `label_weight` weights each parent label (pseudo-labels
    carry the teacher's confidence), `division_ok` limits division, velocity and
    daughter labels to the sources it marks, `label_gt` marks annotated (not
    pseudo) labels so the loss normalises the two groups apart.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    source_um = sources.um(spacing)
    target_um = targets.um(spacing)
    edges = build_candidate_edges(
        source_um, target_um, radius_um=radius_um, max_per_node=max_per_node
    )

    source_row = {int(g): i for i, g in enumerate(sources.gt_index.tolist()) if g >= 0}
    target_row = {int(g): j for j, g in enumerate(targets.gt_index.tolist()) if g >= 0}

    # Which (source,target) pairs the loss needs to exist, and whether they did.
    required: list[tuple[int, int]] = []
    for gt_target, j in target_row.items():
        gt_parent = int(parent_of[gt_target])
        if gt_parent >= 0 and gt_parent in source_row:
            required.append((source_row[gt_parent], j))
    present = {(int(a), int(b)) for a, b in zip(*edges)} if edges.shape[1] else set()
    missing = [pair for pair in required if pair not in present]
    n_in_graph = len(required) - len(missing)
    if missing:
        edges = _unique_edges(
            np.concatenate((edges, np.asarray(missing, dtype=np.int64).T), axis=1)
        )
    edge_position = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(*edges))}

    parent_edge = np.full(len(targets), UNKNOWN, dtype=np.int64)
    parent_weight = np.zeros(len(targets), dtype=np.float32)
    parent_pseudo = np.zeros(len(targets), dtype=bool)
    n_null = 0
    for gt_target, j in target_row.items():
        gt_parent = int(parent_of[gt_target])
        if gt_parent < 0:
            continue  # annotation start, not a verified birth -- stays unknown
        w = 1.0 if label_weight is None else float(label_weight[gt_target])
        if w <= 0:
            continue
        parent_pseudo[j] = label_gt is not None and not bool(label_gt[gt_target])
        if gt_parent in source_row:
            parent_edge[j] = edge_position[(source_row[gt_parent], j)]
            parent_weight[j] = w
        elif gt_parent in dropped_parents:
            parent_edge[j] = NULL
            parent_weight[j] = w
            n_null += 1
    labelled_sources = (
        source_row if division_ok is None
        else {g: i for g, i in source_row.items() if division_ok[g]}
    )

    division_label = np.zeros(len(sources), dtype=np.float32)
    division_mask = np.zeros(len(sources), dtype=np.float32)
    velocity_um = np.zeros((len(sources), 3), dtype=np.float32)
    velocity_mask = np.zeros(len(sources), dtype=np.float32)
    positive_pairs: list[tuple[int, int, int]] = []
    for gt_source, i in labelled_sources.items():
        kids = [k for k in children_of[gt_source]]
        visible = [target_row[k] for k in kids if k in target_row]
        if len(kids) == 2 and len(visible) == 2:
            division_label[i] = 1.0
            division_mask[i] = 1.0
            positive_pairs.append((i, visible[0], visible[1]))
        elif len(kids) == 1 and len(visible) == 1:
            # An observed continuation: "did not divide" is a negative.
            division_mask[i] = 1.0
            velocity_mask[i] = 1.0
            velocity_um[i] = (target_um[visible[0]] - source_um[i]) / dt
        # out-degree 0, or a daughter outside the crop: unknown, mask stays 0.

    triplets, triplet_label = _daughter_triplets(
        positive_pairs, labelled_sources, target_row, children_of, edges,
        max_negatives=max_triplet_negatives, rng=rng,
    )
    return AssociationTargets(
        edge_index=edges,
        parent_edge=parent_edge,
        division_label=division_label,
        division_mask=division_mask,
        velocity_um=velocity_um,
        velocity_mask=velocity_mask,
        triplets=triplets,
        triplet_label=triplet_label,
        n_gt_edges=len(required),
        n_gt_edges_in_graph=n_in_graph,
        n_forced=len(missing),
        n_null_targets=n_null,
        parent_weight=parent_weight,
        parent_pseudo=parent_pseudo,
    )


def _daughter_triplets(
    positive_pairs: list[tuple[int, int, int]],
    source_row: dict[int, int],
    target_row: dict[int, int],
    children_of: list[list[int]],
    edges: np.ndarray,
    *,
    max_negatives: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Annotated daughter pairs as positives; negatives are other pairs among a
    parent's own candidate targets."""
    if edges.shape[1] == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.float32)
    neighbours: dict[int, list[int]] = {}
    for i, j in zip(*edges):
        neighbours.setdefault(int(i), []).append(int(j))

    rows: list[tuple[int, int, int]] = []
    labels: list[float] = []
    true_pairs = {(p, min(a, b), max(a, b)) for p, a, b in positive_pairs}
    for parent, a, b in positive_pairs:
        rows.append((parent, a, b))
        labels.append(1.0)
    for gt_source, i in source_row.items():
        options = [j for j in neighbours.get(i, ())]
        if len(options) < 2:
            continue
        kids = [target_row[k] for k in children_of[gt_source] if k in target_row]
        drawn = 0
        for a, b in _sample_pairs(options, max_negatives * 2, rng):
            key = (i, min(a, b), max(a, b))
            if key in true_pairs:
                continue
            if len(kids) == 2 and {a, b} == set(kids):
                continue
            rows.append((i, a, b))
            labels.append(0.0)
            drawn += 1
            if drawn >= max_negatives:
                break
    return (
        np.asarray(rows, dtype=np.int64).reshape(-1, 3),
        np.asarray(labels, dtype=np.float32),
    )


def _sample_pairs(
    options: list[int], count: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    """Up to `count` distinct unordered pairs from `options`."""
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    limit = len(options) * (len(options) - 1) // 2
    for _ in range(4 * count):
        if len(out) >= min(count, limit):
            break
        a, b = rng.choice(len(options), size=2, replace=False)
        pair = (min(options[a], options[b]), max(options[a], options[b]))
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out
