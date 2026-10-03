"""Write small synthetic movies in the competition format, for the smoke test.

    python tools/make_synthetic_data.py --out outputs/smoke/data

Creates `<out>/train/<stem>.zarr` + `<stem>.geff` (images and a sparse lineage
annotation) and `<out>/test/<stem>.zarr` for the last movie, which is therefore
pinned to the test split like the public test movies. Cells are Gaussian blobs
on a random walk; some divide.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

SPACING = (1.625, 0.40625, 0.40625)


def simulate(rng: np.random.Generator, frames: int, shape: tuple[int, int, int], cells: int):
    """Tracks as (t, z, y, x, track_id, parent_track) rows, positions in native voxels."""
    um = np.asarray(shape) * SPACING
    rows = []
    alive = [(k, rng.uniform(0.2, 0.8, 3) * um, -1) for k in range(cells)]
    next_id = cells
    for t in range(frames):
        survivors = []
        for track, pos, parent in alive:
            rows.append((t, *(pos / SPACING), track, parent))
            step = rng.normal(0, 1.0, 3) * (0.5, 1.0, 1.0)
            if t < frames - 2 and rng.random() < 0.03:  # division
                offset = rng.normal(0, 1.0, 3)
                offset *= 3.0 / np.linalg.norm(offset)
                for sign in (1, -1):
                    survivors.append((next_id, np.clip(pos + step + sign * offset, 0, um - 1), track))
                    next_id += 1
            else:
                survivors.append((track, np.clip(pos + step, 0, um - 1), track))
        alive = survivors
    return np.array(rows, dtype=np.float64)


def render(rows: np.ndarray, frames: int, shape: tuple[int, int, int],
           rng: np.random.Generator) -> np.ndarray:
    z, y, x = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    volume = np.zeros((frames, *shape), dtype=np.float32)
    sigma = 2.0 / np.asarray(SPACING)  # 2 um blobs
    for t, cz, cy, cx, *_ in rows:
        d2 = ((z - cz) / sigma[0]) ** 2 + ((y - cy) / sigma[1]) ** 2 + ((x - cx) / sigma[2]) ** 2
        volume[int(t)] += np.exp(-0.5 * d2)
    volume = 200 + 1500 * volume + rng.normal(0, 30, volume.shape)
    return np.clip(volume, 0, 65535).astype(np.uint16)


def write_movie(path: Path, image: np.ndarray) -> None:
    group = zarr.open_group(path, mode="w")
    group.attrs["multiscales"] = [{
        "version": "0.5",
        "axes": [{"name": a, "type": k} for a, k in
                 (("T", "time"), ("Z", "space"), ("Y", "space"), ("X", "space"))],
        "datasets": [{"path": "0", "coordinateTransformations": [
            {"type": "scale", "scale": [1.0, *SPACING]}]}],
    }]
    array = group.create_array("0", shape=image.shape, chunks=(1, *image.shape[1:]),
                               dtype="uint16")
    array[...] = image


def write_annotation(path: Path, rows: np.ndarray, annotated: set[int]) -> None:
    """The annotated tracks (whole lineages) as a geff graph."""
    root = {}
    for _, _, _, _, track, parent in rows:
        track, parent = int(track), int(parent)
        if track not in root:
            root[track] = track if parent < 0 or parent == track else root[parent]
    keep = np.array([root[int(r[4])] in annotated for r in rows])
    sub = rows[keep]
    ids = np.arange(len(sub))
    index = {(int(r[0]), int(r[4])): i for i, r in enumerate(sub)}
    edges = []
    for i, (t, *_, track, parent) in enumerate(sub):
        if t == 0:
            continue
        for source_track in (track, parent):
            j = index.get((int(t) - 1, int(source_track)))
            if j is not None:
                edges.append((j, i))
                break
    group = zarr.open_group(path, mode="w")
    group.attrs["geff"] = {"geff_version": "1.1", "directed": True,
                           "extra": {"estimated_number_of_nodes": int(len(rows))}}
    group.create_array("nodes/ids", data=ids.astype(np.int64))
    for k, name in enumerate("tzyx"):
        values = np.rint(sub[:, k]).astype(np.int64)
        group.create_array(f"nodes/props/{name}/values", data=values)
    group.create_array("edges/ids", data=np.asarray(edges, dtype=np.int64).reshape(-1, 2))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--movies", type=int, default=7)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--shape", type=int, nargs=3, default=(16, 64, 64))
    p.add_argument("--cells", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    shape = tuple(args.shape)
    (args.out / "train").mkdir(parents=True, exist_ok=True)
    (args.out / "test").mkdir(parents=True, exist_ok=True)
    for index in range(args.movies):
        cohort = "44b6" if index % 2 else "6bba"
        stem = f"{cohort}_{index:08x}"
        rows = simulate(rng, args.frames, shape, args.cells)
        image = render(rows, args.frames, shape, rng)
        tracks = sorted({int(r[4]) for r in rows if r[5] < 0})
        annotated = set(rng.choice(tracks, size=max(1, len(tracks) // 2), replace=False).tolist())
        write_movie(args.out / "train" / f"{stem}.zarr", image)
        write_annotation(args.out / "train" / f"{stem}.geff", rows, annotated)
        if index == args.movies - 1:
            write_movie(args.out / "test" / f"{stem}.zarr", image)
        print(f"{stem}: {len(rows)} cells over {args.frames} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
