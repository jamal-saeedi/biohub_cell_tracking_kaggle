"""Line-fit smoothing of track positions (no topology change).

Each node with at least two chain neighbours within `window` frames is moved
toward the degree-1 least-squares fit of its chain at its own time, blended at
`weight`. A chain stops at a division or a merge.
"""

from __future__ import annotations

import numpy as np

from biohub_tracking.isotropic.config import SmoothingConfig


def _line_intercept(
    neighbourhood: list[tuple[int, int]], original_pos: dict[int, np.ndarray]
) -> np.ndarray:
    """The degree-1 least-squares fit of every axis over `neighbourhood`, at dt=0."""
    dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
    coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
    mean_dt = dts.mean()
    mean_coords = coords.mean(axis=0)
    dt_centered = dts - mean_dt
    sxx = float(dt_centered @ dt_centered)
    slope = (dt_centered @ (coords - mean_coords)) / sxx
    return mean_coords - slope * mean_dt


def linefit_smooth_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    cfg: SmoothingConfig,
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    """Smooth `nodes_by_id` in place along the tracks in `edges`; returns it."""
    if not cfg.enabled or cfg.weight <= 0 or cfg.window <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)

    original_pos = {
        node_id: np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64)
        for node_id, node in nodes_by_id.items()
    }
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(cfg.weight, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]
        for links, sign in ((predecessor, -1), (successor, 1)):
            current = node_id
            for step in range(1, cfg.window + 1):
                linked = links.get(current, [])
                if len(linked) != 1:
                    break
                current = linked[0]
                if current not in original_pos:
                    break
                neighbourhood.append((sign * step, current))
        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue
        fitted = _line_intercept(neighbourhood, original_pos)
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]["z"] = float(pos[0])
        nodes_by_id[node_id]["y"] = float(pos[1])
        nodes_by_id[node_id]["x"] = float(pos[2])

    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id
