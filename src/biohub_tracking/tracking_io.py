"""Reading movies (`<stem>.zarr`) and writing / reading track graphs (`.geff`)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tracksdata as td
import zarr

DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)


@dataclass
class Dataset:
    path: Path
    image: np.ndarray | None  # (T, Z, Y, X), as stored
    scale: tuple[float, float, float]  # (Z, Y, X) voxel size in microns
    image_shape: tuple[int, ...]  # (T, Z, Y, X)


def open_dataset(ds_path: Path | str, load_image: bool = True) -> Dataset:
    """Open `<stem>.zarr` (path given with or without the extension).

    With `load_image=False` only the metadata is read.
    """
    ds_path = Path(ds_path)
    if ds_path.suffix in (".zarr", ".geff"):
        ds_path = ds_path.parent / ds_path.stem
    image_path = ds_path.parent / f"{ds_path.stem}.zarr"
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    group = zarr.open_group(image_path, mode="r")
    scale = _parse_scale(dict(group.attrs))
    shape = tuple(group["0"].shape)
    image = np.asarray(group["0"][...]) if load_image else None
    return Dataset(path=ds_path, image=image, scale=scale, image_shape=shape)


def _parse_scale(attrs: dict) -> tuple[float, float, float]:
    """(Z, Y, X) voxel scale from the OME-NGFF `multiscales` attributes."""
    if "multiscales" in attrs:
        transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if transform["type"] != "scale":
            raise ValueError(f"Transform type is not 'scale': {transform}")
        return tuple(transform["scale"][-3:])
    return DEFAULT_SCALE


def save_graph(graph: td.graph.BaseGraph, output_path: Path | str) -> None:
    """Write a tracksdata graph to a `.geff`, replacing any existing one."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix != ".geff":
        output_path = output_path.with_suffix(".geff")
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    graph.to_geff(output_path)


def graph_from_geff(path: Path) -> td.graph.IndexedRXGraph:
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph
