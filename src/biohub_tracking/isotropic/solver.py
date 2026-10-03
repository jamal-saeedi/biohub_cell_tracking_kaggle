"""Event ILP: the model's likelihoods as costs on node, edge and event variables.

`tracksdata.solvers.ILPSolver` constrains, per node,

    appear_j    + sum_i edge_ij == node_j                 (one parent, or none)
    disappear_i + sum_j edge_ij == node_i + division_i     (one child, or two)
    node_i >= division_i

and minimises the sum of the costs written onto the graph here:

| variable      | cost                                      |
|---------------|-------------------------------------------|
| `edge_ij`     | `-log P(parent = i | j)` + distance term  |
| `appear_j`    | `-log P(no parent | j)` (+ appearance bias)|
| `node_j`      | `-(centre_logit_j - node_logit_bias)`     |
| `division_i`  | `max(floor, -division_logit_i)`           |
| `disappear_i` | `disappearance_weight`                    |

The node term is the only one that can be negative; without it the empty
graph would be optimal. The LP relaxation is nearly integral, so the problem is
first solved by LP + a small restricted MIP (`lpfix`), falling back to SCIP.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from dataclasses import dataclass

import numpy as np
import polars as pl
import tracksdata as td

from biohub_tracking.isotropic.config import SolverConfig

LOG = logging.getLogger(__name__)

__all__ = [
    "TrackGraph", "build_event_graph", "drift_corrected_distance", "greedy_solution",
    "solve_event_ilp",
]


@contextlib.contextmanager
def suppress_output():
    """Silence stdout/stderr (the SCIP solver is extremely chatty)."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield




@dataclass
class TrackGraph:
    """The candidate graph handed to the solver, with its costs already on it."""

    graph: td.graph.InMemoryGraph
    node_ids: np.ndarray  # tracksdata ids, in the order nodes were added
    n_nodes: int
    n_edges: int
    # The arrays that produced the graph, for the LP-first and greedy paths.
    edge_ids: np.ndarray | None = None  # tracksdata edge ids, in edge_index order
    edge_index: np.ndarray | None = None
    node_cost: np.ndarray | None = None
    appear_cost: np.ndarray | None = None
    div_cost: np.ndarray | None = None
    edge_cost: np.ndarray | None = None


def drift_corrected_distance(
    coords_tzyx: np.ndarray,
    edge_index: np.ndarray,
    edge_logp: np.ndarray,
    spacing_um,
    min_prob: float = 0.7,
) -> np.ndarray:
    """Each edge's displacement minus its frame's global shift, in microns.

    The shift of frame t -> t+1 is the median displacement of each target's
    most probable parent, over targets whose parent is confident
    (P >= `min_prob`): the model's own confident links, no solve, no GT. A
    frame with no confident link gets zero shift, i.e. the raw distance.
    `coords_tzyx` is in native voxels; `spacing_um` converts Z, Y, X.
    """
    if edge_index.shape[1] == 0:
        return np.zeros(0)
    scale = np.asarray(spacing_um, dtype=np.float64)
    source, target = edge_index
    displacement = (coords_tzyx[target, 1:] - coords_tzyx[source, 1:]) * scale
    frame = coords_tzyx[source, 0].astype(np.int64)
    shift = np.zeros((int(frame.max()) + 1, 3))
    # First row per target after sorting by (target, -logp) is its argmax parent.
    order = np.lexsort((-edge_logp, target))
    first = np.ones(len(order), dtype=bool)
    first[1:] = target[order][1:] != target[order][:-1]
    best = order[first]
    best = best[np.exp(edge_logp[best]) >= min_prob]
    for t in np.unique(frame[best]):
        shift[t] = np.median(displacement[best[frame[best] == t]], axis=0)
    residual = displacement - shift[frame]
    return np.linalg.norm(residual, axis=1)


def temper_parent_softmax(
    edge_index: np.ndarray, edge_logp: np.ndarray, null_logp: np.ndarray, temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    """Each target's parent distribution (candidates + null) at `temperature`.

    `log p / T`, renormalised per target over the same outcomes. A node with no
    candidates keeps its null log-probability (0 when it is the only outcome).
    """
    if temperature <= 0:
        raise ValueError(f"association_temperature must be > 0, got {temperature}")
    cols = edge_index[1]
    z_edge, z_null = edge_logp / temperature, null_logp / temperature
    peak = z_null.copy()
    np.maximum.at(peak, cols, z_edge)
    mass = np.exp(z_null - peak)
    np.add.at(mass, cols, np.exp(z_edge - peak[cols]))
    norm = peak + np.log(mass)
    has_edges = np.zeros(len(null_logp), dtype=bool)
    has_edges[cols] = True
    return z_edge - norm[cols], np.where(has_edges, z_null - norm, null_logp)


def build_event_graph(
    coords_tzyx: np.ndarray,
    node_logit: np.ndarray,
    null_logp: np.ndarray,
    division_logit: np.ndarray,
    edge_index: np.ndarray,
    edge_logp: np.ndarray,
    edge_distance_um: np.ndarray,
    config: SolverConfig,
) -> TrackGraph:
    """Assemble a `tracksdata` graph with one cost per variable class.

    All arrays are movie-global: `coords_tzyx` is `(N,4)`, the per-node arrays
    are `(N,)`, and `edge_index` is `(2,E)` of indices into them.
    """
    if len(coords_tzyx) == 0:
        raise ValueError("cannot solve an empty movie")

    model_edge_logp = edge_logp  # `edge_prob` keeps the untempered probability
    if config.association_temperature != 1.0:
        edge_logp, null_logp = temper_parent_softmax(
            edge_index, edge_logp, null_logp, config.association_temperature
        )
    node_cost = -config.node_weight_scale * (node_logit - config.node_logit_bias)
    appear_cost = -null_logp
    if config.appearance_bias:
        appear_cost = appear_cost + config.appearance_bias * (
            coords_tzyx[:, 0] > coords_tzyx[:, 0].min()
        )
    div_cost = np.maximum(
        config.division_cost_floor,
        -config.division_weight_scale * division_logit + config.division_bias,
    )
    edge_cost = -edge_logp + config.edge_distance_weight * edge_distance_um

    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    for key in ("node_cost", "appear_cost", "div_cost"):
        graph.add_node_attr_key(key, pl.Float64, 0.0)

    rows = coords_tzyx.tolist()
    node_ids = graph.bulk_add_nodes(
        [
            {
                "t": int(t),
                "z": float(z),
                "y": float(y),
                "x": float(x),
                "node_cost": float(nc),
                "appear_cost": float(ac),
                "div_cost": float(dc),
            }
            for (t, z, y, x), nc, ac, dc in zip(
                rows, node_cost.tolist(), appear_cost.tolist(), div_cost.tolist()
            )
        ]
    )

    n_edges = int(edge_index.shape[1])
    edge_ids: list[int] = []
    if n_edges:
        graph.add_edge_attr_key("edge_cost", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        probability = np.exp(model_edge_logp)
        edge_ids = graph.bulk_add_edges(
            [
                {
                    "source_id": node_ids[int(s)],
                    "target_id": node_ids[int(t)],
                    "edge_cost": float(c),
                    "edge_prob": float(p),
                    "edge_dist": float(d),
                }
                for s, t, c, p, d in zip(
                    edge_index[0].tolist(),
                    edge_index[1].tolist(),
                    edge_cost.tolist(),
                    probability.tolist(),
                    edge_distance_um.tolist(),
                )
            ],
            # Insertion order: `edge_ids[i]` is `edge_index[:, i]`.
            return_ids=True,
        )

    return TrackGraph(
        graph=graph,
        node_ids=np.asarray(node_ids),
        n_nodes=len(node_ids),
        n_edges=n_edges,
        edge_ids=np.asarray(edge_ids) if n_edges else np.empty(0, dtype=np.int64),
        edge_index=edge_index,
        node_cost=node_cost,
        appear_cost=appear_cost,
        div_cost=div_cost,
        edge_cost=edge_cost,
    )


def solve_event_ilp(
    track: TrackGraph, config: SolverConfig, timeout: float | None = None
) -> td.graph.BaseGraph:
    """Run the event ILP and return the solution subgraph.

    LP-first, then SCIP if the LP-first answer is not within `lp_first_gap` of
    the LP bound. `timeout` is one deadline for the whole solve. Whatever fails,
    a feasible answer comes back: the LP-first incumbent, else a greedy solution
    of the same objective.
    """
    if track.n_edges == 0:
        return track.graph

    deadline = None if timeout is None else time.perf_counter() + float(timeout)
    budget = timeout

    solved, incumbent = _solve_lp_first(track, config, deadline=deadline)
    if solved is not None:
        return solved
    if deadline is not None:
        budget = max(1.0, deadline - time.perf_counter())
        # Little time left: a feasible LP-first answer beats a starved SCIP run.
        if incumbent is not None and budget < max(60.0, 3.0 * incumbent[1].seconds):
            LOG.warning("lp_first: %.0fs left, accepting its %.4f%% incumbent instead of "
                        "starting SCIP", budget, 100 * incumbent[1].rel_gap)
            return _lpfix_graph(track, incumbent[1])

    solver = td.solvers.ILPSolver(
        edge_weight=td.EdgeAttr("edge_cost"),
        node_weight=td.NodeAttr("node_cost"),
        appearance_weight=td.NodeAttr("appear_cost"),
        disappearance_weight=float(config.disappearance_weight),
        division_weight=td.NodeAttr("div_cost"),
        timeout=budget,
        gap=0.0,
        num_threads=1,
    )
    started = time.perf_counter()
    try:
        with suppress_output():
            solved = solver.solve(track.graph)
    except Exception as error:  # noqa: BLE001 -- fall back below
        LOG.warning("SCIP raised %r", error)
        print(f"[solver] SCIP raised {error!r}", flush=True)
        solved = None
    # SCIP stopped by its time limit: the LP-first answer has a known gap, so prefer it.
    if (incumbent is not None and deadline is not None
            and time.perf_counter() - started >= 0.98 * budget):
        LOG.warning("SCIP hit its %.0fs limit; using the lp_first incumbent (%.4f%%)",
                    budget, 100 * incumbent[1].rel_gap)
        return _lpfix_graph(track, incumbent[1])
    empty = solved is None or (solved.num_nodes() == 0 and track.n_nodes > 0)
    # An empty solution would fail the submission; keep the LP-first answer.
    if incumbent is not None and empty:
        LOG.warning("SCIP returned no solution; using the lp_first incumbent")
        return _lpfix_graph(track, incumbent[1])
    if empty:
        # No incumbent either: the greedy answer to the same objective.
        kept = greedy_solution(track, float(config.disappearance_weight))
        LOG.warning("SCIP returned no solution and there is no incumbent; using "
                    "the greedy solution (%d nodes, %d edges)",
                    len(kept.kept_nodes), len(kept.kept_edges))
        print(f"[solver] EMPTY solution, no incumbent: greedy fallback kept "
              f"{len(kept.kept_nodes)} nodes, {len(kept.kept_edges)} edges", flush=True)
        return _lpfix_graph(track, kept)
    return solved


@dataclass
class _Kept:
    """The two fields `_lpfix_graph` reads from an LP-fix result."""

    kept_nodes: np.ndarray
    kept_edges: np.ndarray


def greedy_solution(track: TrackGraph, disappear_cost: float) -> _Kept:
    """A feasible answer to the event ILP in one pass, for when no solver produced one.

    Nodes: every detection the model prices as real (`node_cost < 0`), or all of them
    if none is, less the ones left unlinked whose node + appear + disappear cost is
    positive. Edges: cheapest first, each taken iff it keeps the tracking legal (one
    parent, at most two children) and LOWERS the objective -- `edge - appear(target)`,
    minus the source's disappearance on its first child, plus its division cost on the
    second. Same costs as the ILP, so it is the optimum's greedy approximation rather
    than a different model.
    """
    n = track.n_nodes
    node_cost = np.asarray(track.node_cost, dtype=np.float64)
    keep = node_cost < 0
    if not keep.any():
        keep[:] = True
    if track.n_edges == 0 or track.edge_index is None:
        return _Kept(kept_nodes=np.flatnonzero(keep), kept_edges=np.empty(0, dtype=np.int64))
    src = np.asarray(track.edge_index[0], dtype=np.int64)
    dst = np.asarray(track.edge_index[1], dtype=np.int64)
    edge_cost = np.asarray(track.edge_cost, dtype=np.float64)
    appear = np.asarray(track.appear_cost, dtype=np.float64)
    division = np.asarray(track.div_cost, dtype=np.float64)
    has_parent = np.zeros(n, dtype=bool)
    children = np.zeros(n, dtype=np.int64)
    chosen = []
    for e in np.argsort(edge_cost, kind="stable").tolist():
        s, t = src[e], dst[e]
        if not (keep[s] and keep[t]) or has_parent[t] or children[s] >= 2:
            continue
        delta = edge_cost[e] - appear[t] + (-disappear_cost if children[s] == 0 else division[s])
        if delta >= 0:
            continue
        has_parent[t] = True
        children[s] += 1
        chosen.append(e)
    # A node left without any link pays node + appear + disappear; drop it if positive.
    alone = keep & ~has_parent & (children == 0)
    keep &= ~(alone & (node_cost + appear + disappear_cost > 0))
    return _Kept(kept_nodes=np.flatnonzero(keep), kept_edges=np.asarray(chosen, dtype=np.int64))


def _solve_lp_first(
    track: TrackGraph, config: SolverConfig, *, deadline: float | None = None
) -> td.graph.BaseGraph | None:
    """`(solution graph, None)` when the LP-first solve is within `lp_first_gap` of
    the LP bound, else `(None, incumbent)`: `incumbent` is `(None, result)` for a
    feasible answer outside the gap, or `None`.
    """
    from biohub_tracking.isotropic.lpfix import solve_lpfix

    start = time.perf_counter()
    try:
        result = solve_lpfix(
            track.n_nodes,
            track.edge_index[0], track.edge_index[1],
            track.node_cost, track.appear_cost,
            float(config.disappearance_weight),
            track.div_cost, track.edge_cost,
            hops=int(config.lp_first_hops),
            # Half the remaining budget, so a fallback still leaves SCIP time.
            time_limit=(
                None if deadline is None
                else max(1.0, 0.5 * (deadline - time.perf_counter()))
            ),
        )
    except (RuntimeError, ImportError, MemoryError) as error:
        LOG.warning("lp_first failed (%s); falling back to SCIP",
                    error)
        return None, None

    gap = result.rel_gap
    if gap > config.lp_first_gap:
        LOG.warning(
            "lp_first missed the LP bound by %.4f%% > lp_first_gap %.4f%% after "
            "%.1fs; falling back to SCIP",
            100 * gap, 100 * config.lp_first_gap, time.perf_counter() - start)
        feasible = len(result.kept_nodes) > 0 and np.isfinite(gap)
        return None, ((None, result) if feasible else None)

    graph = _lpfix_graph(track, result)
    LOG.info(
        "lp_first solved in %.1fs (LP %.1fs + MIP %.1fs): %s",
        result.seconds, result.lp_seconds, result.mip_seconds,
        "proven optimal (objective == LP bound)" if result.optimal
        else f"{100 * gap:.4f}% above the LP bound, within lp_first_gap",
    )
    return graph, None


def _lpfix_graph(track: TrackGraph, result):
    """`track`'s graph filtered to an LP-fix solution, as `ILPSolver.solve()` returns it."""
    from tracksdata.attrs import EdgeAttr, NodeAttr

    graph = track.graph
    if "solution" not in graph.node_attr_keys():
        graph.add_node_attr_key("solution", pl.Boolean)
    if "solution" not in graph.edge_attr_keys():
        graph.add_edge_attr_key("solution", pl.Boolean)
    graph.update_node_attrs(attrs={"solution": False})
    graph.update_edge_attrs(attrs={"solution": False})
    graph.update_node_attrs(
        node_ids=track.node_ids[result.kept_nodes].tolist(),
        attrs={"solution": True},
    )
    graph.update_edge_attrs(
        edge_ids=track.edge_ids[result.kept_edges].tolist(),
        attrs={"solution": True},
    )
    return graph.filter(
        NodeAttr("solution") == True,  # noqa: E712 -- tracksdata's expression API
        EdgeAttr("solution") == True,  # noqa: E712
    ).subgraph()
