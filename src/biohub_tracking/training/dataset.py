"""Native-resolution training windows for the lineage models.

Per example:

1. Read whole frames (the images are chunked one frame per chunk).
2. Gamma / depth-gain on the unit-scaled frame, z-score with whole-frame
   statistics, then crop (matching full-frame inference).
3. Lateral D4 and synthetic drift on the image and the points together.
4. Three-state detection targets (`targets.py`), the candidate graph and its
   association labels (`candidates.py`).

Most crops are centred on an annotated cell (a uniform crop usually holds none);
a share is uniform and a share is centred on a division.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset, Sampler

from biohub_tracking.training.augment import (
    AugmentConfig,
    add_noise,
    add_shot_noise,
    apply_d4,
    apply_drift,
    apply_gamma,
    degrade_contrast,
    depth_ramp,
    sample_d4,
    sample_depth_gain,
    sample_drift,
    sample_gamma,
)
from biohub_tracking.training.candidates import (
    NodeSet,
    build_association_targets,
)
from biohub_tracking.training.corpus import MovieTracks, load_tracks, raw_frames
from biohub_tracking.training.targets import (
    DETECTION_STRIDE,
    build_detection_targets,
    gaussian_mass,
    reduce_background,
)


@dataclass(frozen=True)
class DataConfig:
    """Everything that changes what a training example contains."""

    frames: int = 3
    crop_zyx: tuple[int, int, int] = (64, 128, 128)
    samples_per_epoch: int = 2048
    #: Share of crops placed uniformly rather than on an annotated cell.
    random_crop_probability: float = 0.15
    #: Share of crops centred on a dividing cell.
    division_anchor_probability: float = 0.25
    crop_jitter_fraction: float = 0.25

    sigma_um: float = 1.5
    positive_radius_um: float = 3.0
    ignore_radius_um: float = 6.0
    background_quantile: float = 0.5

    #: Image-derived competitors (bright peaks) added to each frame's nodes.
    distractors_per_frame: int = 64
    distractor_min_distance_um: float = 3.0
    distractor_quantile: float = 0.99
    #: k-NN candidate graph, the same as at inference.
    candidate_radius_um: float = 20.0
    candidate_max_per_node: int = 4
    #: Share of annotated parents removed from the source set, labelled "no parent".
    parent_drop_probability: float = 0.15
    coord_jitter_um: float = 0.5
    max_triplet_negatives: int = 4

    #: Teacher pseudo-labels (`pseudo.py`): a directory of `<stem>.npz`, one per
    #: training movie. `None` trains on the annotation alone.
    pseudo_dir: str | None = None
    pseudo_match_um: float = 4.0
    pseudo_min_node_prob: float = 0.5
    pseudo_min_edge_prob: float = 0.5

    augment: AugmentConfig = field(default_factory=AugmentConfig)

    def to_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != "augment"}
        out["crop_zyx"] = list(self.crop_zyx)
        out["augment"] = dict(self.augment.__dict__)
        out["augment"]["gamma_range"] = list(self.augment.gamma_range)
        return out


class EpochIndexSampler(Sampler):
    """Epoch `e` yields indices `[e * size, (e + 1) * size)`: the index is the
    example's seed, so the epoch reaches persistent dataloader workers."""

    def __init__(self, size: int, epoch: int = 0) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)
        self.epoch = int(epoch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        start = self.epoch * self.size
        return iter(range(start, start + self.size))

    def __len__(self) -> int:
        return self.size


@dataclass(frozen=True)
class Anchor:
    movie: int
    node: int
    frame: int


class LineageCropDataset(Dataset):
    """Map-style dataset; sample `i` is a pure function of `(seed, i)`
    (`i` carries the epoch, see `EpochIndexSampler`)."""

    def __init__(
        self,
        train_dir: Path | str,
        stems: list[str] | tuple[str, ...],
        config: DataConfig,
        *,
        seed: int = 0,
        train: bool = True,
    ) -> None:
        self.train_dir = Path(train_dir)
        self.stems = list(stems)
        self.config = config if train else replace(
            config,
            augment=AugmentConfig(
                lateral_d4=False, noise_sigma=0.0,
                noise_probability=0.0, intensity_probability=0.0,
            ),
            parent_drop_probability=0.0,
            coord_jitter_um=0.0,
        )
        self.seed = seed
        self.train = train
        self._tracks: dict[int, MovieTracks] = {}
        self.anchors, self.division_anchors = self._build_anchors()
        if not self.anchors:
            raise ValueError("no annotated nodes in the supplied movies")

    def __len__(self) -> int:
        return int(self.config.samples_per_epoch)

    def tracks(self, movie: int) -> MovieTracks:
        if movie not in self._tracks:
            tracks = load_tracks(self.train_dir, self.stems[movie])
            config = self.config
            if config.pseudo_dir is not None:
                from biohub_tracking.training.pseudo import load_pseudo, merge_pseudo

                path = Path(config.pseudo_dir) / f"{self.stems[movie]}.npz"
                if not path.exists():
                    raise FileNotFoundError(f"no pseudo-labels for {self.stems[movie]}: {path}")
                tracks = merge_pseudo(
                    tracks, load_pseudo(path), match_um=config.pseudo_match_um,
                    min_node_prob=config.pseudo_min_node_prob,
                    min_edge_prob=config.pseudo_min_edge_prob,
                )
            self._tracks[movie] = tracks
        return self._tracks[movie]

    def _build_anchors(self) -> tuple[list[Anchor], list[Anchor]]:
        anchors: list[Anchor] = []
        divisions: list[Anchor] = []
        for movie in range(len(self.stems)):
            tracks = self.tracks(movie)
            for node in range(tracks.n_nodes):
                if tracks.is_gt is not None and not tracks.is_gt[node]:
                    continue  # crops are centred on annotated cells only
                anchor = Anchor(movie, node, int(tracks.t[node]))
                anchors.append(anchor)
                # A GT division (a GT track end can gain two pseudo children).
                if len(tracks.children[node]) == 2 and (
                        tracks.division_ok is None or tracks.division_ok[node]):
                    divisions.append(anchor)
        return anchors, divisions

    # -- sampling ---------------------------------------------------------

    #: One independent random stream per stage, so changing one stage's settings
    #: leaves the other stages' draws unchanged.
    STREAMS = ("window", "image", "drop", "distractor", "graph")

    def _streams(self, index: int) -> dict[str, np.random.Generator]:
        """`index` is global: it already carries the epoch."""
        root = np.random.default_rng([self.seed, int(index)])
        return dict(zip(self.STREAMS, root.spawn(len(self.STREAMS))))

    def _choose_window(
        self, rng: np.random.Generator, index: int
    ) -> tuple[int, int, np.ndarray | None]:
        """Pick (movie, first frame, crop-centre anchor in native voxels)."""
        config = self.config
        uniform = rng.random() < config.random_crop_probability
        if uniform:
            movie = int(rng.integers(len(self.stems)))
            tracks = self.tracks(movie)
            first = self._window_start(rng, tracks, int(rng.integers(tracks.shape[0])))
            return movie, first, None
        dividing = bool(self.division_anchors) and rng.random() < config.division_anchor_probability
        pool = self.division_anchors if dividing else self.anchors
        anchor = pool[int(rng.integers(len(pool)))]
        tracks = self.tracks(anchor.movie)
        first = self._window_start(rng, tracks, anchor.frame, needs_next=dividing)
        return anchor.movie, first, tracks.zyx[anchor.node].astype(np.float64)

    def _window_start(
        self, rng: np.random.Generator, tracks: MovieTracks, frame: int,
        needs_next: bool = False,
    ) -> int:
        """First frame of a window containing `frame` (and `frame + 1` when
        `needs_next`, so a division anchor keeps its daughters)."""
        span = min(self.config.frames, tracks.shape[0])
        positions = span - 1 if needs_next and span > 1 else span
        offset = int(rng.integers(positions))
        return int(np.clip(frame - offset, 0, tracks.shape[0] - span))

    def _crop_origin(
        self,
        rng: np.random.Generator,
        volume_shape: tuple[int, int, int],
        centre: np.ndarray | None,
    ) -> np.ndarray:
        crop = np.minimum(np.asarray(self.config.crop_zyx), np.asarray(volume_shape))
        high = np.asarray(volume_shape) - crop
        if centre is None:
            return np.array(
                [int(rng.integers(h + 1)) for h in high], dtype=np.int64
            )
        jitter = self.config.crop_jitter_fraction * crop
        wanted = centre - crop / 2 + rng.uniform(-jitter, jitter)
        return np.clip(np.rint(wanted).astype(np.int64), 0, high)

    # -- the sample -------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        config = self.config
        streams = self._streams(index)
        rng = streams["window"]
        movie, first, centre = self._choose_window(rng, index)
        tracks = self.tracks(movie)
        span = min(config.frames, tracks.shape[0])
        frames = list(range(first, first + span))

        crop = np.minimum(np.asarray(config.crop_zyx), np.asarray(tracks.shape[1:]))
        origin = self._crop_origin(rng, tracks.shape[1:], centre)  # type: ignore[arg-type]
        volume, background_level = _prepare_volume(
            raw_frames(self.train_dir, self.stems[movie], frames),
            origin,
            crop,
            gamma=sample_gamma(streams["image"], config.augment),
            depth_gain=sample_depth_gain(streams["image"], config.augment),
            background_quantile=config.background_quantile,
        )
        volume = add_shot_noise(volume, streams["image"], config.augment)
        volume = add_noise(volume, streams["image"], config.augment)

        node_ids, node_points = _nodes_in_crop(tracks, frames, origin, crop)
        rotations, flip_y = sample_d4(streams["image"], config.augment)
        if volume.shape[-2] != volume.shape[-1]:
            rotations -= rotations % 2  # a quarter turn would change the crop shape
        flat = np.concatenate(node_points) if node_points else np.zeros((0, 3))
        volume, flat = apply_d4(volume, flat, rotations, flip_y)
        node_points = _resplit(flat, [len(n) for n in node_ids])
        shifts = sample_drift(streams["image"], config.augment, len(frames), tracks.spacing)
        if shifts is not None:
            volume, node_points, kept = apply_drift(volume, node_points, shifts)
            node_ids = [ids[keep] for ids, keep in zip(node_ids, kept)]

        background = reduce_background(volume < background_level[:, None, None, None])
        point_weights = offset_weights = point_pseudo = None
        if tracks.node_weight is not None:
            point_weights = [tracks.node_weight[ids] for ids in node_ids]
            point_pseudo = [~tracks.is_gt[ids] for ids in node_ids]
            # Offsets are supervised at annotated cells only.
            offset_weights = [tracks.is_gt[ids].astype(np.float64) for ids in node_ids]
        detection = build_detection_targets(
            points=node_points,
            native_shape=tuple(int(v) for v in volume.shape[1:]),  # type: ignore[arg-type]
            spacing=tracks.spacing,
            background=background,
            sigma_um=config.sigma_um,
            positive_radius_um=config.positive_radius_um,
            ignore_radius_um=config.ignore_radius_um,
            point_weights=point_weights,
            offset_weights=offset_weights,
            point_pseudo=point_pseudo,
        )

        node_sets, dropped = self._node_sets(
            streams, volume, node_ids, node_points, tracks.spacing
        )
        label_weight = None if tracks.parent_weight is None else tracks.parent_label_weight()
        label_gt = None if tracks.parent_gt is None else tracks.parent_label_gt()
        pairs = []
        for i in range(len(node_sets) - 1):
            association = build_association_targets(
                node_sets[i],
                node_sets[i + 1],
                parent_of=tracks.parent,
                children_of=tracks.children,
                spacing=tracks.spacing,
                radius_um=config.candidate_radius_um,
                max_per_node=config.candidate_max_per_node,
                dropped_parents=dropped,
                max_triplet_negatives=config.max_triplet_negatives,
                dt=1.0,
                rng=streams["graph"],
                label_weight=label_weight,
                division_ok=tracks.division_ok,
                label_gt=label_gt,
            )
            pairs.append(_pair_tensors(node_sets[i], node_sets[i + 1], association, i, i + 1))

        # Input-only degradation, drawn last from the image stream.
        volume = degrade_contrast(volume, streams["image"], config.augment)
        return {
            "stem": self.stems[movie],
            "first_frame": first,
            "points": [torch.from_numpy(p.astype(np.float32)) for p in node_points],
            "expected_cells": _expected_cells(tracks, crop),
            # Heatmap mass a correctly firing detector emits over one crop-frame.
            "expected_mass": _expected_cells(tracks, crop)
            * gaussian_mass(config.sigma_um, tracks.spacing),
            "origin": torch.from_numpy(origin),
            "spacing": torch.tensor(tracks.spacing, dtype=torch.float32),
            "image": torch.from_numpy(volume[:, None].copy()),
            "heatmap": torch.from_numpy(detection.heatmap),
            "weight": torch.from_numpy(detection.weight),
            "pseudo_weight": torch.from_numpy(detection.pseudo_weight),
            "offset": torch.from_numpy(detection.offset),
            "offset_mask": torch.from_numpy(detection.offset_mask),
            "n_positive": detection.n_positive,
            "n_collisions": detection.n_collisions,
            "supervised_fraction": detection.supervised_fraction,
            "pairs": pairs,
        }

    def _node_sets(
        self,
        streams: dict[str, np.random.Generator],
        volume: np.ndarray,
        node_ids: list[np.ndarray],
        node_points: list[np.ndarray],
        spacing: tuple[float, float, float],
    ) -> tuple[list[NodeSet], frozenset[int]]:
        """Annotated nodes (jittered, some dropped) plus image-derived distractors."""
        config = self.config
        drop_rng, distractor_rng = streams["drop"], streams["distractor"]
        dropped: set[int] = set()
        sets: list[NodeSet] = []
        for frame in range(volume.shape[0]):
            ids, points = node_ids[frame], node_points[frame]
            keep = np.ones(len(ids), dtype=bool)
            if config.parent_drop_probability > 0 and len(ids):
                keep = drop_rng.random(len(ids)) >= config.parent_drop_probability
                dropped.update(int(g) for g, k in zip(ids.tolist(), keep) if not k)
            kept_points = points[keep]
            if config.coord_jitter_um > 0 and len(kept_points):
                sigma = np.full(3, config.coord_jitter_um) / np.asarray(spacing)
                kept_points = kept_points + drop_rng.normal(
                    0.0, sigma, kept_points.shape
                )
            extra = _distractor_points(
                volume[frame], points, spacing, config, distractor_rng,
            )
            coords = np.concatenate((kept_points, extra)) if len(extra) else kept_points
            gt_index = np.concatenate(
                (ids[keep], np.full(len(extra), -1, dtype=np.int64))
            )
            sets.append(
                NodeSet(
                    coords_native=coords.reshape(-1, 3).astype(np.float64),
                    gt_index=gt_index.astype(np.int64),
                )
            )
        return sets, frozenset(dropped)


#: Every Nth voxel of a frame, for the normalisation statistics.
STATISTICS_STRIDE = 13


def _prepare_volume(
    frames: list[np.ndarray],
    origin: np.ndarray,
    crop: np.ndarray,
    *,
    gamma: float,
    depth_gain: float = 0.0,
    background_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Min-max scale, gamma / depth gain, and z-score a crop with whole-frame
    statistics (from a strided subsample transformed identically).

    Returns the crop and, per frame, the `background_quantile` level in the same
    z-scored units (the verified-background threshold).
    """
    volume = np.empty((len(frames), *(int(c) for c in crop)), dtype=np.float32)
    level = np.empty(len(frames), dtype=np.float32)
    box = tuple(slice(int(o), int(o + c)) for o, c in zip(origin, crop))
    for index, frame in enumerate(frames):
        low = float(frame.min())
        span = max(float(frame.max()) - low, 1.0)
        flat = frame.reshape(-1)
        sample = flat[::STATISTICS_STRIDE].astype(np.float32)
        sample -= low
        sample /= span
        block = frame[box].astype(np.float32)
        block -= low
        block /= span
        if depth_gain:
            # Z index of each strided sample, from its flat position.
            plane = frame.shape[1] * frame.shape[2]
            positions = np.arange(0, len(flat), STATISTICS_STRIDE)
            sample *= depth_ramp(positions // plane, frame.shape[0], depth_gain)
            crop_z = np.arange(box[0].start, box[0].stop)
            block *= depth_ramp(crop_z, frame.shape[0], depth_gain)[:, None, None]
        apply_gamma(sample, gamma)
        apply_gamma(block, gamma)
        mean = float(sample.mean())
        std = max(float(sample.std()), 1e-6)
        volume[index] = (block - mean) / std
        level[index] = (float(np.quantile(sample, background_quantile)) - mean) / std
    return volume, level


def _expected_cells(tracks: MovieTracks, crop: np.ndarray) -> float:
    """Expected cells per crop-frame: the movie's estimated cell count scaled by
    the crop's share of the volume (0.0 without an estimate)."""
    if not tracks.estimated_true_nodes:
        return 0.0
    frames = max(tracks.shape[0], 1)
    share = float(np.prod(crop)) / float(np.prod(tracks.shape[1:]))
    return float(tracks.estimated_true_nodes) / frames * share


def _nodes_in_crop(
    tracks: MovieTracks, frames: list[int], origin: np.ndarray, crop: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Annotated node ids and crop-relative native coordinates, per frame."""
    ids: list[np.ndarray] = []
    points: list[np.ndarray] = []
    for frame in frames:
        nodes = tracks.frame_nodes(frame)
        if len(nodes) == 0:
            ids.append(np.empty(0, dtype=np.int64))
            points.append(np.empty((0, 3), dtype=np.float64))
            continue
        local = tracks.zyx[nodes].astype(np.float64) - origin
        inside = np.all((local >= 0) & (local <= crop - 1), axis=1)
        ids.append(nodes[inside])
        points.append(local[inside])
    return ids, points


def _resplit(flat: np.ndarray, counts: list[int]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    start = 0
    for count in counts:
        out.append(flat[start : start + count].reshape(-1, 3))
        start += count
    return out


def _distractor_points(
    frame_volume: np.ndarray,
    gt_points: np.ndarray,
    spacing: tuple[float, float, float],
    config: DataConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bright local maxima (mostly unannotated cells) away from the annotated
    points, as linker competitors; peaks on the detection grid, (3,5,5) window."""
    if config.distractors_per_frame <= 0:
        return np.empty((0, 3), dtype=np.float64)
    stride = DETECTION_STRIDE
    usable = tuple(n - n % s for n, s in zip(frame_volume.shape, stride))
    blocked = frame_volume[: usable[0], : usable[1], : usable[2]].reshape(
        usable[0] // stride[0], stride[0],
        usable[1] // stride[1], stride[1],
        usable[2] // stride[2], stride[2],
    )
    coarse = blocked.mean(axis=(1, 3, 5))
    peak = coarse == ndimage.maximum_filter(coarse, size=(3, 5, 5), mode="nearest")
    level = np.quantile(coarse[::3, ::3, ::3], config.distractor_quantile)
    cells = np.argwhere(peak & (coarse >= level)).astype(np.float64)
    if len(cells) == 0:
        return np.empty((0, 3), dtype=np.float64)
    native = cells * np.asarray(stride, dtype=np.float64)
    if len(gt_points):
        scale = np.asarray(spacing)
        distance = np.linalg.norm(
            (native[:, None, :] - gt_points[None, :, :]) * scale, axis=-1
        )
        native = native[distance.min(axis=1) > config.distractor_min_distance_um]
    if len(native) > config.distractors_per_frame:
        native = native[
            rng.choice(len(native), config.distractors_per_frame, replace=False)
        ]
    return native


def _pair_tensors(
    sources: NodeSet, targets: NodeSet, association, source_frame: int,
    target_frame: int,
) -> dict:
    return {
        "index": source_frame,
        "source_frame": source_frame,
        "target_frame": target_frame,
        "dt": float(target_frame - source_frame),
        "source_coords": torch.from_numpy(sources.coords_native.astype(np.float32)),
        "target_coords": torch.from_numpy(targets.coords_native.astype(np.float32)),
        # Movie-level node index of each node, -1 for a distractor.
        "source_gt_index": torch.from_numpy(sources.gt_index),
        "target_gt_index": torch.from_numpy(targets.gt_index),
        "edge_index": torch.from_numpy(association.edge_index),
        "parent_edge": torch.from_numpy(association.parent_edge),
        "division_label": torch.from_numpy(association.division_label),
        "division_mask": torch.from_numpy(association.division_mask),
        "velocity_um": torch.from_numpy(association.velocity_um),
        "velocity_mask": torch.from_numpy(association.velocity_mask),
        "triplets": torch.from_numpy(association.triplets),
        "triplet_label": torch.from_numpy(association.triplet_label),
        "n_gt_edges": association.n_gt_edges,
        "n_gt_edges_in_graph": association.n_gt_edges_in_graph,
        "n_forced": association.n_forced,
        "n_null_targets": association.n_null_targets,
        "parent_weight": torch.from_numpy(association.parent_weight),
        "parent_pseudo": torch.from_numpy(association.parent_pseudo),
    }


def collate(samples: list[dict]) -> dict:
    """Stack the dense tensors; the sparse graphs stay per-example lists."""
    dense = ("image", "heatmap", "weight", "pseudo_weight", "offset", "offset_mask",
             "spacing", "origin")
    batch = {key: torch.stack([s[key] for s in samples]) for key in dense}
    batch["pairs"] = [s["pairs"] for s in samples]
    batch["points"] = [s["points"] for s in samples]
    batch["stem"] = [s["stem"] for s in samples]
    batch["first_frame"] = [s["first_frame"] for s in samples]
    for key in ("n_positive", "n_collisions", "supervised_fraction",
                "expected_cells", "expected_mass"):
        batch[key] = torch.tensor([float(s[key]) for s in samples])
    return batch
