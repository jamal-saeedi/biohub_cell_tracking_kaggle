"""Augmentations applied jointly to a window's volume and its annotated points.

* Lateral D4 (the eight symmetries of the Y,X plane). Z is never flipped or
  rotated: the voxel is 4x taller than wide and attenuation makes z directional.
* Intensity: one gamma per window on the unit-scaled counts, a mean-preserving
  depth gain, shot noise and Gaussian noise after normalisation.
* Synthetic stage drift: frames translated laterally as a random walk.
* Low contrast: each frame blended with a lateral haze of itself.

Every optional augmentation draws nothing from the generator when disabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: (rotations of 90 degrees in the Y,X plane, flip Y afterwards)
D4_TRANSFORMS: tuple[tuple[int, bool], ...] = tuple(
    (k, flip) for k in range(4) for flip in (False, True)
)


@dataclass(frozen=True)
class AugmentConfig:
    lateral_d4: bool = True
    gamma_range: tuple[float, float] = (0.8, 1.25)
    #: Amplitude of a mean-preserving linear gain along Z (depth attenuation).
    depth_gain: float = 0.0
    #: Signal-dependent noise amplitude, scaled by sqrt(intensity).
    shot_noise: float = 0.0
    noise_sigma: float = 0.05  # z-scored units
    noise_probability: float = 0.5
    intensity_probability: float = 0.5
    #: Probability of synthetic stage drift; per-step shift uniform in
    #: [-drift_max_um, drift_max_um] on Y and X.
    drift_probability: float = 0.0
    drift_max_um: float = 6.0
    #: Probability of a low-contrast window: `c * x + (1 - c) * haze(x)` plus
    #: noise, re-standardised; `c` uniform in [low_contrast_min, 1], noise sigma
    #: uniform in [0, low_contrast_noise].
    low_contrast_probability: float = 0.0
    low_contrast_min: float = 0.1
    low_contrast_noise: float = 0.1


def apply_d4(
    volume: np.ndarray, points: np.ndarray, rotations: int, flip_y: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Transform a `(T,Z,Y,X)` volume and `(N,3)` Z,Y,X points together."""
    size_y, size_x = volume.shape[-2], volume.shape[-1]
    if rotations % 2 == 1 and size_y != size_x:
        raise ValueError("a 90 degree lateral rotation needs a square Y,X crop")
    out = volume
    coords = np.array(points, dtype=np.float64).reshape(-1, 3)
    z, y, x = coords[:, 0].copy(), coords[:, 1].copy(), coords[:, 2].copy()
    for _ in range(rotations % 4):
        out = np.rot90(out, k=1, axes=(-2, -1))
        # np.rot90 on (y,x) sends (y,x) -> (size_x-1-x, y) in the new frame.
        y, x = (size_x - 1) - x, y
        size_y, size_x = size_x, size_y
    if flip_y:
        out = out[..., ::-1, :]
        y = (size_y - 1) - y
    return np.ascontiguousarray(out), np.stack((z, y, x), axis=1)


def sample_d4(rng: np.random.Generator, config: AugmentConfig) -> tuple[int, bool]:
    if not config.lateral_d4:
        return 0, False
    rotations, flip_y = D4_TRANSFORMS[int(rng.integers(len(D4_TRANSFORMS)))]
    return rotations, bool(flip_y)


def sample_gamma(rng: np.random.Generator, config: AugmentConfig) -> float:
    """One gamma for the whole window, or exactly 1.0 for no change."""
    chance = config.intensity_probability
    if chance <= 0 or rng.random() >= chance:
        return 1.0
    return float(rng.uniform(*config.gamma_range))


def sample_depth_gain(rng: np.random.Generator, config: AugmentConfig) -> float:
    """Signed amplitude of the Z gain ramp, or 0.0 when disabled."""
    if config.depth_gain <= 0:
        return 0.0
    return float(rng.uniform(-config.depth_gain, config.depth_gain))


def depth_ramp(z_index: np.ndarray, depth: int, amplitude: float) -> np.ndarray:
    """`1 + amplitude * (2 * z / (depth - 1) - 1)`: mean 1, so it survives the z-score
    only as a gradient along Z."""
    if amplitude == 0.0 or depth < 2:
        return np.ones_like(z_index, dtype=np.float32)
    position = 2.0 * z_index.astype(np.float32) / (depth - 1) - 1.0
    return (1.0 + amplitude * position).astype(np.float32)


def add_shot_noise(
    normalized: np.ndarray, rng: np.random.Generator, config: AugmentConfig
) -> np.ndarray:
    """Noise on z-scored data scaled by `sqrt(relu(x) + 1)`; the `+1` keeps dim
    regions noisy."""
    if config.shot_noise <= 0:
        return normalized
    scale = float(rng.uniform(0.0, config.shot_noise))
    if scale == 0.0:
        return normalized
    amplitude = np.sqrt(np.maximum(normalized, 0.0) + 1.0, dtype=np.float32)
    return normalized + scale * amplitude * rng.standard_normal(
        normalized.shape, dtype=np.float32
    )


def apply_gamma(unit: np.ndarray, gamma: float) -> np.ndarray:
    """`unit ** gamma`, in place, for an array scaled to `[0, 1]`."""
    if gamma == 1.0:
        return unit
    return np.power(unit, gamma, out=unit, dtype=np.float32)


def add_noise(
    normalized: np.ndarray, rng: np.random.Generator, config: AugmentConfig
) -> np.ndarray:
    """Additive Gaussian noise in z-scored units."""
    if config.noise_sigma <= 0 or rng.random() >= config.noise_probability:
        return normalized
    scale = float(rng.uniform(0.0, config.noise_sigma))
    if scale == 0.0:
        return normalized
    noise = rng.standard_normal(normalized.shape, dtype=np.float32)
    return normalized + scale * noise


def sample_drift(
    rng: np.random.Generator, config: AugmentConfig, frames: int,
    spacing: tuple[float, float, float],
) -> np.ndarray | None:
    """`(frames, 2)` cumulative integer Y/X shifts in native voxels, or None."""
    if config.drift_probability <= 0 or frames < 2:
        return None
    if rng.random() >= config.drift_probability:
        return None
    step_um = rng.uniform(-config.drift_max_um, config.drift_max_um, (frames - 1, 2))
    steps = np.rint(step_um / np.asarray(spacing[1:], dtype=np.float64)).astype(np.int64)
    return np.concatenate((np.zeros((1, 2), dtype=np.int64), np.cumsum(steps, axis=0)))


def apply_drift(
    volume: np.ndarray, points: list[np.ndarray], shifts: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Translate frame t of a `(T,Z,Y,X)` volume by `shifts[t]` (Y, X voxels).

    The field is filled from the reflected border. Points move with their frame;
    points pushed outside are dropped, and the returned masks say which were kept.
    """
    size_y, size_x = volume.shape[-2], volume.shape[-1]
    reach = int(np.abs(shifts).max(initial=0))
    if reach == 0:
        return volume, points, [np.ones(len(p), dtype=bool) for p in points]
    if reach >= min(size_y, size_x):
        raise ValueError(f"drift of {reach} voxels exceeds the {size_y}x{size_x} field")
    padded = np.pad(volume, ((0, 0), (0, 0), (reach, reach), (reach, reach)), mode="reflect")
    out = np.empty_like(volume)
    moved: list[np.ndarray] = []
    kept: list[np.ndarray] = []
    for t in range(volume.shape[0]):
        dy, dx = (int(v) for v in shifts[t])
        out[t] = padded[t, :, reach - dy : reach - dy + size_y, reach - dx : reach - dx + size_x]
        shifted = points[t] + np.array([0.0, dy, dx])
        inside = (
            (shifted[:, 1] >= 0) & (shifted[:, 1] <= size_y - 1)
            & (shifted[:, 2] >= 0) & (shifted[:, 2] <= size_x - 1)
        )
        moved.append(shifted[inside])
        kept.append(inside)
    return out, moved, kept


def degrade_contrast(
    normalized: np.ndarray, rng: np.random.Generator, config: AugmentConfig
) -> np.ndarray:
    """Low-contrast window, `(T,Z,Y,X)` z-scored, applied to the model input only
    (targets come from the clean crop). The haze is a 41-voxel lateral box mean."""
    if config.low_contrast_probability <= 0 or rng.random() >= config.low_contrast_probability:
        return normalized
    from scipy.ndimage import uniform_filter

    keep = float(rng.uniform(config.low_contrast_min, 1.0))
    sigma = float(rng.uniform(0.0, config.low_contrast_noise))
    haze = uniform_filter(normalized, size=(1, 1, 41, 41), mode="nearest")
    out = keep * normalized + (1.0 - keep) * haze
    if sigma > 0:
        out += sigma * rng.standard_normal(out.shape, dtype=np.float32)
    axes = tuple(range(1, out.ndim))
    out -= out.mean(axis=axes, keepdims=True)
    out /= np.maximum(out.std(axis=axes, keepdims=True), 1e-6)
    return out.astype(np.float32, copy=False)
