"""Native-resolution movie reading and the input normalisation the models were trained with.

Per frame: scale to `[0, 1]` by the frame's min and range, then z-score with the
mean and standard deviation of the whole frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from biohub_tracking.isotropic.config import SPACING_UM
from biohub_tracking.tracking_io import open_dataset

__all__ = ["NativeMovie", "open_native_movie", "normalize_native"]


@dataclass
class NativeMovie:
    """A whole movie at native resolution, unnormalised."""

    stem: str
    image: np.ndarray  # (T,Z,Y,X), the zarr's own dtype
    spacing: tuple[float, float, float]
    path: Path

    @property
    def frames(self) -> int:
        return int(self.image.shape[0])

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.image.shape[1:])  # type: ignore[return-value]


def open_native_movie(ds_path: Path | str, max_frames: int | None = None) -> NativeMovie:
    """Open `<stem>.zarr` at full resolution, as stored."""
    ds_path = Path(ds_path)
    dataset = open_dataset(ds_path)
    image = dataset.image
    if image is None:  # pragma: no cover - open_dataset guarantees this
        raise RuntimeError(f"{ds_path}: no image data")
    image = np.asarray(image)
    if image.ndim != 4:
        raise ValueError(f"{ds_path}: expected (T,Z,Y,X), got {image.shape}")
    if max_frames is not None:
        image = image[:max_frames]

    spacing = tuple(float(v) for v in dataset.scale)
    if not np.allclose(spacing, SPACING_UM, rtol=1e-6):
        raise ValueError(
            f"{ds_path}: voxel spacing {spacing} differs from the {SPACING_UM} the models "
            "were trained on"
        )
    return NativeMovie(
        stem=Path(ds_path).stem, image=image, spacing=spacing, path=Path(ds_path)
    )


def normalize_native(frames: np.ndarray) -> np.ndarray:
    """Per-frame full-frame z-score of a `(T,Z,Y,X)` array, as float32.

    A constant frame becomes all zeros rather than dividing by zero.
    """
    out = np.empty(frames.shape, dtype=np.float32)
    for index in range(len(frames)):
        frame = frames[index]
        low = float(frame.min())
        span = max(float(frame.max()) - low, 1.0)
        block = frame.astype(np.float32)
        block -= low
        block /= span
        mean = float(block.mean())
        std = max(float(block.std()), 1e-6)
        out[index] = (block - mean) / std
    return out
