"""Gradient-boosted re-ranking of each target's candidate parents.

A tree ensemble re-ranks each target's candidates from the association head's
outputs plus geometry the head does not see (source-side competition, local
density, drift-corrected motion). The model's null mass is kept, so appearance
costs are unchanged:

    log P'(parent = e) = log(1 - P_null(target)) + log softmax_candidates(margin_e)

The trees are stored as flat numpy arrays and evaluated with numba (or numpy).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["FEATURES", "TreeEnsemble", "edge_features", "rescored_logp", "rescore_edges"]

FEATURES = (
    "logp", "rank", "margin", "n_cand", "null_logp",
    "dist_raw", "dist_drift", "drift_dz", "drift_lat", "raw_dz", "raw_lat",
    "vel_res", "vel_res_z", "logit_src", "logit_tgt", "div_src",
    "src_best_elsewhere", "src_top1_count", "src_out_deg", "density_tgt", "density_src",
    "z_tgt", "t_frac",
)


def edge_features(
    coords_tzyx: np.ndarray,
    edge_index: np.ndarray,
    edge_logp: np.ndarray,
    null_logp: np.ndarray,
    node_logit: np.ndarray,
    division_logit: np.ndarray,
    velocity_um: np.ndarray,
    spacing_um,
) -> np.ndarray:
    """(E, len(FEATURES)) float32 features of every candidate edge."""
    from scipy.spatial import cKDTree

    from biohub_tracking.isotropic.solver import drift_corrected_distance

    spacing = np.asarray(spacing_um, dtype=np.float64)
    coords = coords_tzyx
    src, tgt = edge_index
    logp = edge_logp
    n, e = len(coords), len(logp)
    um = coords[:, 1:] * spacing
    # per-target rank and margin to the best rival
    order = np.lexsort((-logp, tgt))
    rank = np.empty(e)
    first = np.ones(e, dtype=bool)
    first[1:] = tgt[order][1:] != tgt[order][:-1]
    start = np.maximum.accumulate(np.where(first, np.arange(e), 0))
    rank[order] = np.arange(e) - start
    best = np.full(n, -np.inf)
    np.maximum.at(best, tgt, logp)
    second = np.full(n, -np.inf)
    is_best = logp >= best[tgt]
    np.maximum.at(second, tgt[~is_best], logp[~is_best])
    with np.errstate(invalid="ignore"):
        margin = np.where(is_best, logp - second[tgt], logp - best[tgt])
    margin = np.clip(np.nan_to_num(margin, posinf=20.0, neginf=-20.0), -20, 20)
    n_cand = np.bincount(tgt, minlength=n)[tgt]
    # geometry: raw and drift-corrected displacement, split into z and lateral
    disp = um[tgt] - um[src]
    frame = coords[src, 0].astype(int)
    drift = drift_corrected_distance(coords, edge_index, logp, spacing)
    shift = np.zeros((frame.max() + 1, 3))
    top = order[first]
    top = top[np.exp(logp[top]) >= 0.7]
    for t in np.unique(frame[top]):
        shift[t] = np.median(disp[top[frame[top] == t]], axis=0)
    res = disp - shift[frame]
    vres = disp - velocity_um[src]
    # source-side competition: the source's best log-prob on any OTHER target
    src_best = np.full(n, -np.inf)
    np.maximum.at(src_best, src, logp)
    src_second = np.full(n, -np.inf)
    is_src_best = logp >= src_best[src]
    np.maximum.at(src_second, src[~is_src_best], logp[~is_src_best])
    elsewhere = np.where(is_src_best, src_second[src], src_best[src])
    elsewhere = np.clip(np.nan_to_num(elsewhere, neginf=-20.0), -20, 0)
    top1_count = np.bincount(src[order[first]], minlength=n)[src]
    out_deg = np.bincount(src, minlength=n)[src]
    density = np.zeros(n)
    for t in np.unique(coords[:, 0]):
        k = np.flatnonzero(coords[:, 0] == t)
        tree = cKDTree(um[k])
        density[k] = np.array([len(x) for x in tree.query_ball_point(um[k], 10.0)]) - 1
    t_max = max(coords[:, 0].max(), 1)
    return np.column_stack([
        logp, rank, margin, n_cand, null_logp[tgt],
        np.linalg.norm(disp, axis=1), drift, np.abs(res[:, 0]), np.linalg.norm(res[:, 1:], axis=1),
        np.abs(disp[:, 0]), np.linalg.norm(disp[:, 1:], axis=1),
        np.linalg.norm(vres, axis=1), np.abs(vres[:, 0]),
        node_logit[src], node_logit[tgt], division_logit[src],
        elsewhere, top1_count, out_deg, density[tgt], density[src],
        coords[tgt, 1], coords[tgt, 0] / t_max,
    ]).astype(np.float32)


def rescored_logp(margin: np.ndarray, tgt: np.ndarray, null_logp: np.ndarray) -> np.ndarray:
    """The re-ranked parent log-probabilities; each target's null mass is kept."""
    n = len(null_logp)
    peak = np.full(n, -np.inf)
    np.maximum.at(peak, tgt, margin)
    mass = np.zeros(n)
    np.add.at(mass, tgt, np.exp(margin - peak[tgt]))
    log_q = margin - peak[tgt] - np.log(mass[tgt])
    linked = np.log1p(-np.exp(np.minimum(null_logp[tgt], -1e-9)))
    return linked + log_q


@dataclass(frozen=True)
class TreeEnsemble:
    """A LightGBM binary booster as flat arrays; `predict` returns the raw margin.

    Node arrays are global over all trees. An internal node has `feature >= 0`
    and goes left when `x <= threshold`; a leaf has `feature == -1` and `value`.
    """

    feature: np.ndarray  # int32 (N,)
    threshold: np.ndarray  # float64 (N,)
    left: np.ndarray  # int32 (N,)
    right: np.ndarray  # int32 (N,)
    value: np.ndarray  # float64 (N,)
    roots: np.ndarray  # int32 (T,)
    feature_names: tuple[str, ...]

    @classmethod
    def load(cls, path: Path | str) -> TreeEnsemble:
        with np.load(path, allow_pickle=False) as blob:
            names = tuple(str(x) for x in blob["feature_names"])
            if names != FEATURES:
                raise ValueError(f"{path}: trained on unknown features {names}")
            return cls(blob["feature"], blob["threshold"], blob["left"], blob["right"],
                       blob["value"], blob["roots"], names)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Raw margin per row: numba when importable, else a numpy walk."""
        x = np.ascontiguousarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("edge features must be finite (the trees were trained without NaN)")
        kernel = _numba_kernel()
        if kernel is not None:
            return kernel(x, self.feature, self.threshold, self.left, self.right, self.value,
                          self.roots)
        rows = np.arange(len(x))
        out = np.zeros(len(x))
        for root in self.roots:
            node = np.full(len(x), root, dtype=np.int64)
            active = self.feature[node] >= 0
            while active.any():
                k = node[active]
                go_left = x[rows[active], self.feature[k]] <= self.threshold[k]
                node[active] = np.where(go_left, self.left[k], self.right[k])
                active = self.feature[node] >= 0
            out += self.value[node]
        return out


_KERNEL: list = []


def _numba_kernel():
    """The compiled tree walk, built once; `None` when numba is unavailable."""
    if not _KERNEL:
        try:  # numba refuses some numpy versions at import with other errors than ImportError
            import numba
        except Exception as error:  # noqa: BLE001 -- any failure means "use the numpy walk"
            # The numpy walk is ~20x slower, so say so in the log.
            print(f"[rescorer] tree kernel: numpy walk (numba unavailable: {error!r})", flush=True)
            _KERNEL.append(None)
        else:
            print(f"[rescorer] tree kernel: numba {numba.__version__}", flush=True)
            # rows are independent, so the parallel walk is deterministic
            @numba.njit(cache=False, nogil=True, parallel=True)
            def walk(x, feature, threshold, left, right, value, roots):
                out = np.zeros(x.shape[0])
                for r in numba.prange(x.shape[0]):
                    total = 0.0
                    for t in range(roots.shape[0]):
                        node = roots[t]
                        while feature[node] >= 0:
                            if x[r, feature[node]] <= threshold[node]:
                                node = left[node]
                            else:
                                node = right[node]
                        total += value[node]
                    out[r] = total
                return out

            _KERNEL.append(walk)
    return _KERNEL[0]


def rescore_edges(
    ensemble: TreeEnsemble,
    coords_tzyx, edge_index, edge_logp, null_logp, node_logit, division_logit, velocity_um,
    spacing_um,
) -> np.ndarray:
    """`edge_logp` re-ranked per target by `ensemble` (same shape)."""
    if edge_index.shape[1] == 0:
        return edge_logp
    feats = edge_features(coords_tzyx, edge_index, edge_logp, null_logp, node_logit,
                          division_logit, velocity_um, spacing_um)
    return rescored_logp(ensemble.predict(feats), edge_index[1], null_logp)
