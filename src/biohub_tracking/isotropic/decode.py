"""Centre decoding: window plan, peak suppression, offsets and node descriptors.

A frame is decoded from one forward pass over the whole native volume (no
spatial tiling). Node descriptors are sampled at the decoded peaks right away,
so a movie's association state is a few hundred rows per frame rather than a
dense feature volume per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from biohub_tracking.isotropic.config import DetectionConfig, WindowConfig
from biohub_tracking.models.isotropic_lineage import DetectionOutput, nodes_from_peaks

__all__ = ["FrameNodes", "window_plan", "peak_indices", "decode_frames"]


@dataclass
class FrameNodes:
    """Decoded cells of one frame. Coordinates are native voxels, Z,Y,X."""

    frame: int
    coords_native: np.ndarray  # (N,3) float32
    scores: np.ndarray  # (N,) float32, sigmoid of the centre logit
    logits: np.ndarray  # (N,) float32, the centre logit (the solver's node cost)
    descriptors: Tensor  # (N, descriptor_channels) on the model's device
    member_descriptors: tuple = ()
    """The other ensemble members' descriptors at these same nodes, in
    `LoadedModel.ensemble` order."""

    def __len__(self) -> int:
        return int(len(self.scores))

    def um(self, spacing: tuple[float, float, float]) -> np.ndarray:
        return self.coords_native.astype(np.float64) * np.asarray(spacing)


def window_plan(n_frames: int, window: WindowConfig) -> list[tuple[int, list[int]]]:
    """`(start, positions)` pairs covering every frame exactly once.

    Each frame is read from the most-centred full-length window that fits:
    `[t-1, t, t+1]` in the interior, the first or last full window at the
    boundaries. A movie shorter than `window.frames` is one window over the
    frames it has.
    """
    size = min(window.frames, n_frames)
    if size < 1:
        return []
    last_start = n_frames - size
    grouped: dict[int, list[int]] = {}
    for frame in range(n_frames):
        start = min(max(frame - (size - 1) // 2, 0), last_start)
        grouped.setdefault(start, []).append(frame - start)
    return [(start, sorted(v)) for start, v in sorted(grouped.items())]


def peak_indices(
    probability: Tensor,
    *,
    threshold: float,
    window: tuple[int, int, int],
    max_peaks: int,
) -> tuple[Tensor, Tensor]:
    """Non-maximum suppression on one `(1,Dz,Dy,Dx)` probability volume.

    A voxel is kept only when the window's arg-max is that voxel, so an exact
    plateau yields one detection rather than one per voxel. Returns integer
    grid indices `(N,3)` and their probabilities `(N,)`.
    """
    probability = probability[None] if probability.ndim == 4 else probability
    padding = tuple(w // 2 for w in window)
    pooled, indices = F.max_pool3d(
        probability, window, stride=1, padding=padding, return_indices=True
    )
    shape = probability.shape[-3:]
    flat = torch.arange(
        int(np.prod(shape)), device=probability.device
    ).view(1, 1, *shape)
    keep = (indices == flat) & (probability >= threshold)
    cells = torch.nonzero(keep[0, 0], as_tuple=False)
    if len(cells) == 0:
        return cells, probability.new_zeros(0)
    scores = probability[0, 0][cells.unbind(-1)]
    if len(cells) > max_peaks:
        top = torch.topk(scores, max_peaks).indices
        cells, scores = cells[top], scores[top]
    return cells, scores


def _inside(coords: np.ndarray, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Centres the learned offset pushed outside the volume are dropped."""
    low = np.zeros(3)
    high = np.asarray(shape_zyx, dtype=np.float64) - 1.0
    return np.all((coords >= low) & (coords <= high), axis=1)


@torch.no_grad()
def decode_frames(
    output: DetectionOutput,
    *,
    positions: list[int],
    frame_ids: list[int],
    config: DetectionConfig,
    shape_zyx: tuple[int, int, int],
) -> list[FrameNodes]:
    """Turn one window's dense output into `FrameNodes` for the kept frames.

    `positions` indexes into the window; `frame_ids` gives each one's absolute
    frame number.
    """
    probability = output.center_logits.float().sigmoid()
    results: list[FrameNodes] = []
    for position, frame_id in zip(positions, frame_ids):
        cells, scores = peak_indices(
            probability[0, position],
            threshold=config.threshold,
            window=config.nms_window,
            max_peaks=config.max_peaks,
        )
        if len(cells) == 0:
            results.append(
                FrameNodes(
                    frame=frame_id,
                    coords_native=np.zeros((0, 3), dtype=np.float32),
                    scores=np.zeros(0, dtype=np.float32),
                    logits=np.zeros(0, dtype=np.float32),
                    descriptors=output.features.new_zeros(0, output.descriptor_channels),
                )
            )
            continue

        coords, descriptors = nodes_from_peaks(output, 0, position, cells)
        coords_np = coords.float().cpu().numpy()
        inside = _inside(coords_np.astype(np.float64), shape_zyx)
        z, y, x = cells.unbind(-1)
        logits = output.center_logits[0, position, 0, z, y, x].float().cpu().numpy()
        results.append(
            FrameNodes(
                frame=frame_id,
                coords_native=coords_np[inside],
                scores=scores.float().cpu().numpy()[inside],
                logits=logits[inside],
                descriptors=descriptors[torch.as_tensor(inside, device=descriptors.device)],
            )
        )
    return results
