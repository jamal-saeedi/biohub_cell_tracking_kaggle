"""Training movies: the ground-truth lineage (`.geff`) and frame reader (`.zarr`).

Coordinates are native voxel indices (Z,Y,X), as the `.geff` stores them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import zarr

#: Default Z,Y,X spacing in microns; `movie_spacing` reads the per-movie value.
SPACING_ZYX: tuple[float, float, float] = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class MovieTracks:
    """One movie's sparse annotation as flat arrays indexed 0..N-1.

    `parent[i]` is i's annotated parent or -1, `children` the inverse. An
    annotated track start is not a birth and an end is not a death.
    """

    stem: str
    shape: tuple[int, int, int, int]  # T,Z,Y,X of the image, not of the labels
    spacing: tuple[float, float, float]
    t: np.ndarray  # int64 (N,)
    zyx: np.ndarray  # int64 (N,3), native voxel indices
    src: np.ndarray  # int64 (E,), node index of the parent
    dst: np.ndarray  # int64 (E,), node index of the child, always t+1
    parent: np.ndarray  # int64 (N,), -1 where no annotated parent
    children: list[list[int]]
    estimated_true_nodes: float | None
    #: Set by `pseudo.merge_pseudo` (`None`: plain annotation, every node GT
    #: with weight 1). `node_weight` scales the detection target,
    #: `parent_weight` the parent label into the node, `division_ok` marks
    #: sources whose division/velocity label is GT, `parent_gt` parent labels
    #: that are annotated edges.
    node_weight: np.ndarray | None = None
    parent_weight: np.ndarray | None = None
    is_gt: np.ndarray | None = None
    division_ok: np.ndarray | None = None
    parent_gt: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return int(len(self.t))

    @property
    def n_edges(self) -> int:
        return int(len(self.src))

    @property
    def n_divisions(self) -> int:
        return int(sum(1 for c in self.children if len(c) == 2))

    def parent_label_weight(self) -> np.ndarray:
        """Weight of each node's parent label (0 where it has no parent)."""
        if self.parent_weight is None:
            return (self.parent >= 0).astype(np.float64)
        return np.where(self.parent >= 0, self.parent_weight.astype(np.float64), 0.0)

    def parent_label_gt(self) -> np.ndarray:
        """Whether each node's parent label is annotated (not pseudo)."""
        if self.parent_gt is None:
            return self.parent >= 0
        return self.parent_gt & (self.parent >= 0)

    def frame_nodes(self, frame: int) -> np.ndarray:
        """Node indices annotated at `frame`, ascending."""
        return self._by_frame.get(int(frame), _EMPTY)

    def um(self, node_indices: np.ndarray) -> np.ndarray:
        """Native voxel indices -> microns, Z,Y,X float64 (M,3)."""
        return self.zyx[node_indices].astype(np.float64) * np.asarray(self.spacing)

    def __post_init__(self) -> None:
        by_frame: dict[int, list[int]] = {}
        for index, frame in enumerate(self.t.tolist()):
            by_frame.setdefault(int(frame), []).append(index)
        object.__setattr__(
            self,
            "_by_frame",
            {f: np.asarray(v, dtype=np.int64) for f, v in by_frame.items()},
        )


_EMPTY = np.empty(0, dtype=np.int64)


def movie_stems(train_dir: Path | str) -> list[str]:
    """Stems that have BOTH an image and an annotation, sorted."""
    train_dir = Path(train_dir)
    images = {p.stem for p in train_dir.glob("*.zarr")}
    labels = {p.stem for p in train_dir.glob("*.geff")}
    return sorted(images & labels)


def movie_spacing(train_dir: Path | str, stem: str) -> tuple[float, float, float]:
    """Per-movie Z,Y,X spacing from the OME-NGFF transform, or the default."""
    attrs = dict(zarr.open_group(Path(train_dir) / f"{stem}.zarr", mode="r").attrs)
    try:
        transform = attrs["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ][0]
        if transform["type"] == "scale":
            return tuple(float(v) for v in transform["scale"][-3:])  # type: ignore[return-value]
    except (KeyError, IndexError, TypeError):
        pass
    return SPACING_ZYX


def find_estimated_true_nodes(geff_path: Path | str) -> float | None:
    """The organisers' estimate of the movie's cell count (`estimated_number_of_nodes`)."""
    geff_path = Path(geff_path)
    for candidate in (geff_path / "zarr.json", geff_path / ".zattrs"):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        found = _find_key(payload, "estimated_number_of_nodes")
        if found is not None:
            try:
                return float(found)
            except (TypeError, ValueError):
                continue
    return None


def _find_key(payload: object, key: str) -> object | None:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def load_tracks(train_dir: Path | str, stem: str) -> MovieTracks:
    """One movie's annotation graph and image shape. Every edge must advance one
    frame, no node may have two parents or more than two children."""
    train_dir = Path(train_dir)
    graph = zarr.open_group(train_dir / f"{stem}.geff", mode="r")
    ids = np.asarray(graph["nodes/ids"]).astype(np.int64)
    t = np.asarray(graph["nodes/props/t/values"]).astype(np.int64)
    zyx = np.stack(
        [np.asarray(graph[f"nodes/props/{a}/values"]).astype(np.int64) for a in "zyx"],
        axis=1,
    )
    raw_edges = np.asarray(graph["edges/ids"]).astype(np.int64).reshape(-1, 2)

    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    src = order[np.searchsorted(sorted_ids, raw_edges[:, 0])]
    dst = order[np.searchsorted(sorted_ids, raw_edges[:, 1])]
    if len(raw_edges) and not (
        np.array_equal(ids[src], raw_edges[:, 0])
        and np.array_equal(ids[dst], raw_edges[:, 1])
    ):
        raise ValueError(f"{stem}: edge references a node id that does not exist")

    delta = t[dst] - t[src]
    if len(delta) and not np.all(delta == 1):
        raise ValueError(f"{stem}: edges must advance exactly one frame")

    parent = np.full(len(ids), -1, dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(len(ids))]
    for a, b in zip(src.tolist(), dst.tolist()):
        if parent[b] != -1:
            raise ValueError(f"{stem}: node {b} has two annotated parents")
        parent[b] = a
        children[a].append(b)
        if len(children[a]) > 2:
            raise ValueError(f"{stem}: node {a} has more than two children")

    image = zarr.open_group(train_dir / f"{stem}.zarr", mode="r")["0"]
    return MovieTracks(
        stem=stem,
        shape=tuple(int(s) for s in image.shape),  # type: ignore[arg-type]
        spacing=movie_spacing(train_dir, stem),
        t=t,
        zyx=zyx,
        src=src,
        dst=dst,
        parent=parent,
        children=children,
        estimated_true_nodes=find_estimated_true_nodes(train_dir / f"{stem}.geff"),
    )


@lru_cache(maxsize=8)
def _image_handle(train_dir: str, stem: str):
    return zarr.open_group(Path(train_dir) / f"{stem}.zarr", mode="r")["0"]


@lru_cache(maxsize=24)
def _cached_frame(train_dir: str, stem: str, frame: int) -> np.ndarray:
    """One decoded frame, read-only (about 8 MiB; windows share frames)."""
    array = np.asarray(_image_handle(train_dir, stem)[frame])
    array.flags.writeable = False
    return array


def raw_frames(train_dir: Path | str, stem: str, frames: list[int]) -> list[np.ndarray]:
    """Whole frames as read-only `uint16` arrays (the images are chunked per frame)."""
    root = str(Path(train_dir))
    return [_cached_frame(root, stem, int(f)) for f in frames]


@dataclass(frozen=True)
class MovieSummary:
    """Cheap per-movie facts, enough to build a split without reading images."""

    stem: str
    cohort: str
    frames: int
    n_nodes: int
    n_edges: int
    n_divisions: int
    estimated_true_nodes: float | None

    @property
    def coverage(self) -> float | None:
        if not self.estimated_true_nodes:
            return None
        return self.n_nodes / self.estimated_true_nodes


def summarize(
    train_dir: Path | str, stems: list[str] | None = None
) -> list[MovieSummary]:
    """One `MovieSummary` per movie, in stem order (labels only)."""
    train_dir = Path(train_dir)
    rows = []
    for stem in stems if stems is not None else movie_stems(train_dir):
        tracks = load_tracks(train_dir, stem)
        rows.append(
            MovieSummary(
                stem=stem,
                cohort=stem.split("_")[0],
                frames=int(tracks.t.max()) + 1 if tracks.n_nodes else 0,
                n_nodes=tracks.n_nodes,
                n_edges=tracks.n_edges,
                n_divisions=tracks.n_divisions,
                estimated_true_nodes=tracks.estimated_true_nodes,
            )
        )
    return rows
