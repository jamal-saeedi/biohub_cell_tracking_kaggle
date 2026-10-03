"""Model-free tracking for a movie the models could not predict (no GPU needed).

Used only when every model configuration failed for a movie, so the submission
still covers it:

* detection -- per frame: p1-p99.7 rescale, lateral block mean to ~1.625 um,
  difference of Gaussians at two scales, 3.2 um spherical non-maximum
  suppression, peaks brighter than the frame median, response >= 0.03;
* linking -- per frame pair, the median nearest-neighbour step removed (stage
  drift), then one-to-one matches cheapest first within `LINK_RADIUS_UM`.

If even that fails, `placeholder_movie` writes one node.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ISO_UM = 1.625
DOG_SCALES_UM = ((1.5, 4.0), (2.2, 5.5))
NMS_UM = 3.2
DOG_THRESHOLD = 0.03
LINK_RADIUS_UM = 6.0

__all__ = ["dog_frame", "link_nearest", "model_free_movie", "placeholder_movie"]


def dog_frame(frame: np.ndarray, spacing: tuple[float, float, float]):
    """(K,3) native zyx float and (K,) DoG response for one (Z,Y,X) frame."""
    from scipy.ndimage import gaussian_filter, maximum_filter

    f = np.asarray(frame, dtype=np.float32)
    lo, hi = np.percentile(f, (1.0, 99.7))
    f = np.clip((f - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    z, y, x = f.shape
    bins = [max(1, int(round(ISO_UM / float(s)))) for s in spacing[1:]]
    y, x = y - y % bins[0], x - x % bins[1]
    iso = f[:, :y, :x].reshape(z, y // bins[0], bins[0], x // bins[1], bins[1]).mean(axis=(2, 4))
    grid_um = np.array([spacing[0], spacing[1] * bins[0], spacing[2] * bins[1]], dtype=np.float64)
    response = np.max(
        [gaussian_filter(iso, s1 / grid_um, mode="nearest")
         - gaussian_filter(iso, s2 / grid_um, mode="nearest") for s1, s2 in DOG_SCALES_UM], axis=0)
    r = NMS_UM / grid_um
    k = np.floor(r).astype(int)
    offsets = np.mgrid[-k[0]:k[0] + 1, -k[1]:k[1] + 1, -k[2]:k[2] + 1]
    footprint = sum((offsets[i] / r[i]) ** 2 for i in range(3)) <= 1.0
    peak = response == maximum_filter(response, footprint=footprint, mode="constant", cval=-np.inf)
    peak &= response > 0
    peak &= iso > np.median(iso)
    idx = np.argwhere(peak)
    padded = np.pad(np.maximum(response, 0), 1)
    nbr = np.stack(np.mgrid[-1:2, -1:2, -1:2], -1).reshape(-1, 3)
    w = np.stack([padded[idx[:, 0] + 1 + o[0], idx[:, 1] + 1 + o[1], idx[:, 2] + 1 + o[2]]
                  for o in nbr], 1)
    sub = idx + (w @ nbr) / np.maximum(w.sum(1, keepdims=True), 1e-12)
    sub = np.clip(sub, 0, np.array(iso.shape) - 1)
    native = np.column_stack((sub[:, 0], sub[:, 1] * bins[0] + (bins[0] - 1) / 2,
                              sub[:, 2] * bins[1] + (bins[1] - 1) / 2))
    return native, response[tuple(idx.T)]


def link_nearest(t: np.ndarray, zyx_native: np.ndarray, spacing, radius_um: float = LINK_RADIUS_UM):
    """One-to-one gap-1 links, drift-corrected, cheapest first. Returns (E,2) index pairs."""
    from scipy.spatial import cKDTree

    um = np.asarray(zyx_native, dtype=np.float64) * np.asarray(spacing, dtype=np.float64)
    t = np.asarray(t, dtype=np.int64)
    by_frame = {int(f): np.flatnonzero(t == f) for f in np.unique(t)}
    links = []
    for f, src in by_frame.items():
        dst = by_frame.get(f + 1)
        if dst is None or len(src) == 0 or len(dst) == 0:
            continue
        a, b = um[src], um[dst]
        # Stage drift: the median nearest-neighbour displacement.
        d, j = cKDTree(b).query(a, distance_upper_bound=2.0 * radius_um)
        ok = np.isfinite(d)
        shift = np.median(b[j[ok]] - a[ok], axis=0) if ok.sum() >= 5 else np.zeros(3)
        pairs = cKDTree(a + shift).sparse_distance_matrix(cKDTree(b), radius_um, output_type="ndarray")
        if len(pairs) == 0:
            continue
        order = np.argsort(pairs["v"], kind="stable")
        used_a, used_b = set(), set()
        for i, jj in zip(pairs["i"][order].tolist(), pairs["j"][order].tolist()):
            if i in used_a or jj in used_b:
                continue
            used_a.add(i)
            used_b.add(jj)
            links.append((src[i], dst[jj]))
    return np.asarray(links, dtype=np.int64).reshape(-1, 2)


def _dicts(t, zyx, links, spacing):
    nodes = {i: {"node_id": i, "t": int(t[i]), "z": float(zyx[i, 0]), "y": float(zyx[i, 1]),
                 "x": float(zyx[i, 2])} for i in range(len(t))}
    step = (zyx[links[:, 1]] - zyx[links[:, 0]]) * np.asarray(spacing) if len(links) else np.zeros((0, 3))
    prob = np.exp(-np.linalg.norm(step, axis=1) / LINK_RADIUS_UM)
    edges = [{"source_id": int(s), "target_id": int(d), "edge_prob": float(p)}
             for (s, d), p in zip(links.tolist(), prob.tolist())]
    return nodes, edges


def model_free_movie(ds_path: Path, max_frames: int | None = None,
                     threshold: float = DOG_THRESHOLD, radius_um: float = LINK_RADIUS_UM):
    """`(nodes_by_id, edges)` for one movie from the image alone (DoG + nearest links)."""
    from biohub_tracking.tracking_io import open_dataset

    dataset = open_dataset(Path(ds_path))
    image = np.asarray(dataset.image)
    if max_frames is not None:
        image = image[:max_frames]
    spacing = tuple(float(v) for v in dataset.scale)[-3:]
    ts, zyx = [], []
    for f in range(image.shape[0]):
        native, response = dog_frame(image[f], spacing)
        keep = response >= threshold
        ts.append(np.full(int(keep.sum()), f, dtype=np.int64))
        zyx.append(native[keep])
    t, zyx = np.concatenate(ts), np.concatenate(zyx)
    if len(t) == 0:
        raise RuntimeError(f"{ds_path}: DoG found no cells")
    return _dicts(t, zyx, link_nearest(t, zyx, spacing, radius_um), spacing)


def placeholder_movie(shape: tuple[int, int, int] | None):
    """One node at the volume centre of frame 0: the CSV covers the movie, nothing more."""
    z, y, x = ((s - 1) / 2.0 for s in shape) if shape else (0.0, 0.0, 0.0)
    return {0: {"node_id": 0, "t": 0, "z": z, "y": y, "x": x}}, []
