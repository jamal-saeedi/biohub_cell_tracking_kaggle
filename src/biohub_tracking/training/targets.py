"""Three-state detection targets for sparse annotation.

About 3% of the cells are annotated, so a dense heatmap target would teach the
detector to suppress the other 97%. Every voxel is instead one of:

* positive -- within `positive_radius_um` of an annotated centre: an isotropic
  Gaussian, exactly 1.0 at the nearest grid cell;
* verified background -- darker than the frame's `background_quantile` AND
  further than `ignore_radius_um` from every annotation;
* unknown -- everything else, weight zero.

Coordinates are Z,Y,X. The detection grid is native/(1,2,2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DETECTION_STRIDE: tuple[int, int, int] = (1, 2, 2)


def detection_shape(
    native: tuple[int, int, int], stride: tuple[int, int, int] = DETECTION_STRIDE
) -> tuple[int, int, int]:
    """Spatial shape the centre/offset heads emit for a native crop."""
    return tuple(-(-n // s) for n, s in zip(native, stride))  # type: ignore[return-value]


def detection_spacing(
    spacing: tuple[float, float, float], stride: tuple[int, int, int] = DETECTION_STRIDE
) -> np.ndarray:
    return np.asarray(spacing, dtype=np.float64) * np.asarray(stride, dtype=np.float64)


@dataclass
class DetectionTargets:
    """Per-frame dense targets, all shaped `(T, C, Dz, Dy, Dx)` float32."""

    heatmap: np.ndarray  # T,1,D...  soft target in [0,1]
    weight: np.ndarray  # T,1,D...  1 supervised, 0 unknown
    offset: np.ndarray  # T,3,D...  nearest-cell residual in grid voxels
    offset_mask: np.ndarray  # T,1,D...  1 only at each annotation's nearest cell
    n_positive: int
    n_collisions: int  # annotations sharing one grid cell
    supervised_fraction: float
    #: T,1,D... weights of pseudo-labelled voxels, normalised separately in the
    #: loss (`L_gt + lambda * L_pseudo`); all zero without pseudo-labels.
    pseudo_weight: np.ndarray | None = None


def build_detection_targets(
    *,
    points: list[np.ndarray],
    native_shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    background: np.ndarray | None,
    sigma_um: float = 1.5,
    positive_radius_um: float = 3.0,
    ignore_radius_um: float = 6.0,
    stride: tuple[int, int, int] = DETECTION_STRIDE,
    point_weights: list[np.ndarray] | None = None,
    offset_weights: list[np.ndarray] | None = None,
    point_pseudo: list[np.ndarray] | None = None,
) -> DetectionTargets:
    """Targets for a `len(points)`-frame crop.

    `points[i]` is `(n_i, 3)` centres of frame `i` in native voxels relative to
    the crop origin; `background` is the boolean `(T, Dz, Dy, Dx)` verified
    background on the detection grid (`reduce_background`).

    `point_weights` (default 1) weight each point's positive ball. Points marked
    in `point_pseudo` go to `pseudo_weight` instead of `weight`. The ignore band
    is cleared around every point in both maps, and where GT or verified
    background supervises a voxel the pseudo map is zero. `offset_weights`
    (default: the point weights) set the offset mask; the heavier point keeps a
    shared grid cell's offset.
    """
    if positive_radius_um > ignore_radius_um:
        raise ValueError("positive_radius_um must not exceed ignore_radius_um")
    if sigma_um <= 0:
        raise ValueError("sigma_um must be positive")
    sigma_axes = np.full(3, sigma_um, dtype=np.float64)

    frames = len(points)
    grid = detection_shape(native_shape, stride)
    grid_um = detection_spacing(spacing, stride)
    shape = (frames, 1, *grid)

    heatmap = np.zeros(shape, dtype=np.float32)
    offset = np.zeros((frames, 3, *grid), dtype=np.float32)
    offset_mask = np.zeros(shape, dtype=np.float32)
    if background is None:
        weight = np.zeros(shape, dtype=np.float32)
    else:
        if background.shape != (frames, *grid):
            raise ValueError(
                f"background {background.shape} does not match grid {(frames, *grid)}"
            )
        weight = background.astype(np.float32).reshape(shape).copy()
    pseudo_weight = np.zeros(shape, dtype=np.float32)

    n_positive = 0
    n_collisions = 0
    stride_arr = np.asarray(stride, dtype=np.float64)
    sigma_cells = sigma_axes / grid_um
    gauss_radius = np.maximum(np.ceil(3.0 * sigma_cells), 1.0).astype(int)
    ignore_radius = np.maximum(np.ceil(ignore_radius_um / grid_um), 0.0).astype(int)
    positive_radius = np.maximum(np.ceil(positive_radius_um / grid_um), 0.0).astype(int)

    for frame, frame_points in enumerate(points):
        frame_points = np.asarray(frame_points, dtype=np.float64).reshape(-1, 3)
        pw = (np.ones(len(frame_points)) if point_weights is None
              else np.asarray(point_weights[frame], dtype=np.float64).reshape(-1))
        ow = pw if offset_weights is None else np.asarray(
            offset_weights[frame], dtype=np.float64).reshape(-1)
        pseudo = (np.zeros(len(frame_points), dtype=bool) if point_pseudo is None
                  else np.asarray(point_pseudo[frame], dtype=bool).reshape(-1))
        if len(pw) != len(frame_points) or len(ow) != len(frame_points) \
                or len(pseudo) != len(frame_points):
            raise ValueError("one weight per point is required")
        # Clear every ignore band first, so no band overwrites another's positives.
        for centre in frame_points:
            _apply_ball(weight[frame, 0], centre / stride_arr, ignore_radius, grid_um,
                        ignore_radius_um, value=0.0)
        for centre, w, w_off, is_pseudo in zip(frame_points, pw, ow, pseudo):
            cell = centre / stride_arr
            target_map = pseudo_weight if is_pseudo else weight
            _splat_gaussian(heatmap[frame, 0], cell, gauss_radius, grid_um, sigma_axes)
            _apply_ball(target_map[frame, 0], cell, positive_radius, grid_um,
                        positive_radius_um, value=float(w), combine=np.maximum)
            nearest = np.rint(cell).astype(int)
            if np.any(nearest < 0) or np.any(nearest >= np.asarray(grid)):
                continue  # centre rounds outside the crop; nothing to anchor
            index = (frame, 0, *nearest)
            if offset_mask[index] > 0:
                n_collisions += 1
            heatmap[index] = 1.0
            target_map[index] = max(target_map[index], float(w))
            if w_off >= offset_mask[index]:
                offset_mask[index] = float(w_off)
                offset[(frame, slice(None), *nearest)] = (cell - nearest).astype(np.float32)
            n_positive += 1

    pseudo_weight[weight > 0] = 0.0  # GT and verified background take precedence
    return DetectionTargets(
        pseudo_weight=pseudo_weight,
        heatmap=heatmap,
        weight=weight,
        offset=offset,
        offset_mask=offset_mask,
        n_positive=n_positive,
        n_collisions=n_collisions,
        supervised_fraction=float(weight.mean()),
    )


def _ball_slices(
    cell: np.ndarray, radius: np.ndarray, grid: tuple[int, ...]
) -> tuple[tuple[slice, ...], list[np.ndarray]] | None:
    """Index box around `cell` clipped to `grid`, plus per-axis cell offsets."""
    lower = np.maximum(np.floor(cell).astype(int) - radius, 0)
    upper = np.minimum(np.ceil(cell).astype(int) + radius + 1, np.asarray(grid))
    if np.any(lower >= upper):
        return None
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
    axes = [
        np.arange(a, b, dtype=np.float64) - c
        for a, b, c in zip(lower, upper, cell)
    ]
    return slices, axes


def _distance_um(axes: list[np.ndarray], grid_um: np.ndarray) -> np.ndarray:
    dz = (axes[0] * grid_um[0])[:, None, None]
    dy = (axes[1] * grid_um[1])[None, :, None]
    dx = (axes[2] * grid_um[2])[None, None, :]
    return np.sqrt(dz * dz + dy * dy + dx * dx)


def _splat_gaussian(
    field: np.ndarray,
    cell: np.ndarray,
    radius: np.ndarray,
    grid_um: np.ndarray,
    sigma_um: np.ndarray,
) -> None:
    box = _ball_slices(cell, radius, field.shape)
    if box is None:
        return
    slices, axes = box
    value = np.exp(-0.5 * (_distance_um(axes, grid_um) / float(sigma_um[0])) ** 2)
    np.maximum(field[slices], value.astype(field.dtype), out=field[slices])


def _apply_ball(
    field: np.ndarray,
    cell: np.ndarray,
    radius: np.ndarray,
    grid_um: np.ndarray,
    radius_um: float,
    *,
    value: float,
    combine=None,
) -> None:
    box = _ball_slices(cell, radius, field.shape)
    if box is None:
        return
    slices, axes = box
    inside = _distance_um(axes, grid_um) <= radius_um
    region = field[slices]
    region[inside] = value if combine is None else combine(region[inside], value)


def reduce_background(
    native_background: np.ndarray, stride: tuple[int, int, int] = DETECTION_STRIDE
) -> np.ndarray:
    """Native `(T,Z,Y,X)` boolean -> detection grid: a cell is background only
    when every native voxel under it is."""
    frames = native_background.shape[0]
    grid = detection_shape(native_background.shape[1:], stride)  # type: ignore[arg-type]
    padded = np.ones((frames, *(g * s for g, s in zip(grid, stride))), dtype=bool)
    padded[
        :, : native_background.shape[1], : native_background.shape[2],
        : native_background.shape[3]
    ] = native_background
    blocked = padded.reshape(
        frames, grid[0], stride[0], grid[1], stride[1], grid[2], stride[2]
    )
    return blocked.all(axis=(2, 4, 6))


def gaussian_mass(
    sigma_um: float,
    spacing: tuple[float, float, float],
    stride: tuple[int, int, int] = DETECTION_STRIDE,
) -> float:
    """Heatmap mass one cell contributes, summed on the grid: the count prior's
    conversion from an expected cell count to a predicted heatmap sum."""
    grid_um = detection_spacing(spacing, stride)
    total = 1.0
    for step, sigma in zip(grid_um, (sigma_um, sigma_um, sigma_um)):
        reach = int(np.ceil(4.0 * sigma / step))
        offsets = np.arange(-reach, reach + 1) * step
        total *= float(np.exp(-0.5 * (offsets / sigma) ** 2).sum())
    return total
