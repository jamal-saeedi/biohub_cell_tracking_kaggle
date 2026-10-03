"""Top level: predict every movie, enforce the submission contract, smooth, write the CSV.

The structural contract: in-degree <= 1, out-degree <= 2, every edge joins
consecutive frames, no dangling edges. Every movie gets a prediction: a movie
without model output gets the model-free tracking (`fallback`).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from biohub_tracking.isotropic import predict as iso_predict
from biohub_tracking.isotropic import shard as iso_shard
from biohub_tracking.isotropic.config import IsotropicConfig
from biohub_tracking.tracking_io import graph_from_geff
from biohub_tracking.submission import MovieResult, validate_submission, write_test_submission

__all__ = [
    "graph_to_dicts", "minimal_repair", "run_pipeline", "smooth_positions",
    "stage_offsets",
]


def graph_to_dicts(
    geff_path: Path,
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    """Load a solved `.geff` (the solution subgraph) into plain node/edge structures."""
    graph = graph_from_geff(geff_path)
    nodes_by_id = {
        int(row["node_id"]): {
            "node_id": int(row["node_id"]),
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
        for row in graph.node_attrs().iter_rows(named=True)
    }
    edges = [
        {
            "source_id": int(row["source_id"]),
            "target_id": int(row["target_id"]),
            "edge_prob": (
                float(row["edge_prob"]) if row.get("edge_prob") is not None else None
            ),
        }
        for row in graph.edge_attrs().iter_rows(named=True)
    ]
    return nodes_by_id, edges


def minimal_repair(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]], dict[str, int]]:
    """Enforce the submission's structural contract; change nothing else."""
    stats = {
        "nodes_in": len(nodes_by_id),
        "edges_in": len(edges),
        "dropped_dangling": 0,
        "dropped_degree": 0,
    }

    kept_edges: list[dict[str, object]] = []
    incoming: dict[int, int] = {}
    outgoing: dict[int, int] = {}
    # Highest probability first, so a degree violation drops the least confident edge.
    ordered = sorted(
        edges,
        key=lambda e: (-(e.get("edge_prob") or 0.0), e["source_id"], e["target_id"]),
    )
    for edge in ordered:
        source, target = int(edge["source_id"]), int(edge["target_id"])
        if source not in nodes_by_id or target not in nodes_by_id:
            stats["dropped_dangling"] += 1
            continue
        if int(nodes_by_id[target]["t"]) != int(nodes_by_id[source]["t"]) + 1:
            stats["dropped_dangling"] += 1
            continue
        if incoming.get(target, 0) >= 1 or outgoing.get(source, 0) >= 2:
            stats["dropped_degree"] += 1
            continue
        incoming[target] = incoming.get(target, 0) + 1
        outgoing[source] = outgoing.get(source, 0) + 1
        kept_edges.append(edge)

    kept_edges.sort(key=lambda e: (e["source_id"], e["target_id"]))
    stats["nodes_out"] = len(nodes_by_id)
    stats["edges_out"] = len(kept_edges)
    stats["divisions"] = sum(1 for v in outgoing.values() if v == 2)
    return nodes_by_id, kept_edges, stats


def stage_offsets(nodes_by_id, edges) -> dict[int, np.ndarray]:
    """Cumulative stage shift per frame, in native voxels, from the solved tracks.

    The step of frame t -> t+1 is the median displacement over the solved
    one-to-one links leaving frame t (division edges excluded); a frame with no
    such link contributes zero. Frame `t`'s offset is the sum of the steps
    before it, so the first frame sits at zero.
    """
    out_degree: dict = {}
    for edge in edges:
        out_degree[edge["source_id"]] = out_degree.get(edge["source_id"], 0) + 1
    steps: dict[int, list] = {}
    for edge in edges:
        if out_degree[edge["source_id"]] != 1:
            continue
        a, b = nodes_by_id[edge["source_id"]], nodes_by_id[edge["target_id"]]
        if int(b["t"]) != int(a["t"]) + 1:
            continue
        steps.setdefault(int(a["t"]), []).append(
            (float(b["z"]) - float(a["z"]), float(b["y"]) - float(a["y"]),
             float(b["x"]) - float(a["x"])))
    frames = [int(v["t"]) for v in nodes_by_id.values()]
    offsets, total = {}, np.zeros(3)
    for t in range(min(frames), max(frames) + 1):
        offsets[t] = total.copy()
        if t in steps:
            total = total + np.median(np.asarray(steps[t]), axis=0)
    return offsets


def _moved(nodes_by_id, offsets, sign: float) -> dict:
    return {
        k: {**v, "z": float(v["z"]) + sign * offsets[int(v["t"])][0],
            "y": float(v["y"]) + sign * offsets[int(v["t"])][1],
            "x": float(v["x"]) + sign * offsets[int(v["t"])][2]}
        for k, v in nodes_by_id.items()
    }


def smooth_positions(nodes_by_id, edges, cfg, stats: dict, *,
                     drift_compensated: bool = False) -> dict:
    """Line-fit smoothing of every track; coordinates only.

    With `drift_compensated`, the cumulative stage shift (`stage_offsets`) is
    removed first and added back after, so the fit sees each cell's own motion.
    """
    from collections import defaultdict

    from biohub_tracking.postprocess.smoothing import linefit_smooth_output_graph

    if not cfg.enabled:
        return nodes_by_id
    counters = defaultdict(int)
    if drift_compensated and edges and nodes_by_id:
        offsets = stage_offsets(nodes_by_id, edges)
        smoothed = linefit_smooth_output_graph(_moved(nodes_by_id, offsets, -1.0), edges, cfg,
                                               counters)
        smoothed = _moved(smoothed, offsets, 1.0)
    else:
        smoothed = linefit_smooth_output_graph(nodes_by_id, edges, cfg, counters)
    stats.update({f"smooth_{k}": v for k, v in counters.items()})
    return smoothed


def _volume_shape(ds_path: Path) -> tuple[int, int, int] | None:
    """(Z, Y, X) of a movie from its zarr metadata (no pixels read); None if unreadable."""
    from biohub_tracking.tracking_io import open_dataset

    try:
        shape = open_dataset(ds_path, load_image=False).image_shape
    except Exception:  # noqa: BLE001 -- clamping is a refinement, never a failure
        return None
    return tuple(int(v) for v in shape[-3:]) if shape and len(shape) >= 3 else None


def clamp_to_volume(nodes_by_id: dict, shape: tuple[int, int, int] | None) -> dict:
    """Clamp node centres into [0, dim - 1] per axis, in place: smoothing can
    extrapolate a chain end past the volume."""
    if shape is None:
        return nodes_by_id
    for node in nodes_by_id.values():
        for axis, dim in zip(("z", "y", "x"), shape):
            value = float(node[axis])
            if value > dim - 1:
                node[axis] = float(dim - 1)
            elif value < 0.0:
                node[axis] = 0.0
    return nodes_by_id


def _stems(data_dir: Path, dataset_stems: list[str] | None) -> list[str]:
    """Every movie the submission must cover, in the writer's order."""
    if dataset_stems is not None:
        return sorted(dataset_stems)
    return sorted(p.stem for p in Path(data_dir).glob("*.zarr"))


def _fallback(ds_path: Path, max_frames: int | None):
    """The model-free tracking (`fallback.model_free_movie`), else one placeholder node."""
    from biohub_tracking.isotropic import fallback

    try:
        nodes_by_id, edges = fallback.model_free_movie(ds_path, max_frames=max_frames)
        print(f"[pipeline] {ds_path.stem}: model-free FALLBACK wrote {len(nodes_by_id)} "
              f"nodes, {len(edges)} edges", flush=True)
        return nodes_by_id, edges
    except Exception as error:  # noqa: BLE001 -- the last resort must not raise
        print(f"[pipeline] {ds_path.stem}: model-free FALLBACK failed ({error!r}); writing "
              "one placeholder node", flush=True)
        return fallback.placeholder_movie(_volume_shape(ds_path))


def _movie_or_fallback(geff_path: Path | None, ds_path: Path, max_frames: int | None):
    """A movie's solved graph, or the model-free fallback when there is none."""
    if geff_path is not None:
        try:
            return graph_to_dicts(geff_path)
        except Exception as error:  # noqa: BLE001 -- a corrupt file is a missing movie
            print(f"[pipeline] {ds_path.stem}: cannot read {geff_path.name} ({error!r})",
                  flush=True)
    else:
        print(f"[pipeline] {ds_path.stem}: no model output", flush=True)
    return _fallback(ds_path, max_frames)


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    submission_path: Path,
    config: IsotropicConfig,
    dataset_stems: list[str] | None = None,
    device=None,
    max_frames: int | None = None,
    num_shards: int | None = None,
    session_deadline_seconds: float | None = None,
    ilp_reserve_fraction: float = 0.3,
    timing: dict | None = None,
) -> Path:
    """Predict every movie, enforce the contract, smooth, write and validate the CSV.

    `output_dir` receives the per-movie `.geff` predictions. `timing`, if given,
    is filled with wall-clock seconds and counts.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)

    start = time.perf_counter()
    if num_shards is not None and num_shards > 1:
        geff_paths = iso_shard.predict_sharded(
            data_dir, output_dir, config,
            dataset_stems=dataset_stems, max_frames=max_frames,
            num_shards=num_shards,
            session_deadline_seconds=session_deadline_seconds,
            ilp_reserve_fraction=ilp_reserve_fraction,
        )
    else:
        try:
            geff_paths = iso_predict.predict(
                data_dir, output_dir, config,
                dataset_stems=dataset_stems, device=device, max_frames=max_frames,
                session_deadline_seconds=session_deadline_seconds,
                ilp_reserve_fraction=ilp_reserve_fraction, skip_failures=True,
            )
        except Exception as error:  # noqa: BLE001 -- e.g. a model failed to load
            print(f"[pipeline] predict FAILED before any movie ({error!r}); re-running "
                  "every movie on the fallback configurations", flush=True)
            geff_paths = []
        done = {Path(g).stem for g in geff_paths}
        failed = [s for s in _stems(data_dir, dataset_stems) if s not in done]
        if failed:
            iso_shard.rerun_missing(data_dir, output_dir, config, failed, max_frames=max_frames)
            geff_paths = [output_dir / f"{s}.geff" for s in _stems(data_dir, dataset_stems)
                          if (output_dir / f"{s}.geff").exists()]
    predict_seconds = time.perf_counter() - start

    movies: list[MovieResult] = []
    movie_stats: dict[str, dict[str, int]] = {}
    by_stem = {Path(g).stem: Path(g) for g in geff_paths}
    for stem in _stems(data_dir, dataset_stems):
        nodes_by_id, edges = _movie_or_fallback(by_stem.get(stem), data_dir / stem, max_frames)
        nodes_by_id, edges, stats = minimal_repair(nodes_by_id, edges)
        nodes_by_id = smooth_positions(
            nodes_by_id, edges, config.smoothing, stats,
            drift_compensated=config.smooth_drift_compensated,
        )
        nodes_by_id = clamp_to_volume(nodes_by_id, _volume_shape(data_dir / stem))
        if not nodes_by_id:  # the writer refuses an empty movie
            print(f"[pipeline] {stem}: no nodes left after post-processing; writing the "
                  "model-free FALLBACK", flush=True)
            nodes_by_id, edges = _fallback(data_dir / stem, max_frames)
            nodes_by_id = clamp_to_volume(nodes_by_id, _volume_shape(data_dir / stem))
        movie_stats[stem] = stats
        movies.append((stem, nodes_by_id, edges, stats))

    write_test_submission(movies, Path(submission_path))
    validate_submission(Path(submission_path),
                        expected_datasets=_stems(data_dir, dataset_stems))

    if timing is not None:
        timing["predict_seconds"] = predict_seconds
        timing["total_seconds"] = time.perf_counter() - start
        timing["nodes"] = int(sum(s["nodes_out"] for s in movie_stats.values()))
        timing["edges"] = int(sum(s["edges_out"] for s in movie_stats.values()))
        timing["divisions"] = int(sum(s["divisions"] for s in movie_stats.values()))
    return Path(submission_path)
