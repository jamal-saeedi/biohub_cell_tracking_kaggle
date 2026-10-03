"""LP-first solve of the event ILP.

The constraint matrix is a network matrix except for the `division <= node`
rows, so the LP relaxation is nearly integral. Solve the LP, fix every variable
it decided, and solve the MIP over the fractional variables plus their
constraint neighbourhood (`hops` steps through the variable-constraint graph).
The LP objective is a lower bound, so `rel_gap` bounds how far the answer can be
from the optimum.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


__all__ = ["LPFixResult", "solve_lpfix"]


@dataclass
class LPFixResult:
    """One solve; `optimal` means the objective equals the LP bound."""

    kept_edges: np.ndarray
    """Row indices into the edge arrays that the solve selected."""
    kept_nodes: np.ndarray
    objective: float
    bound: float
    optimal: bool
    lp_seconds: float
    mip_seconds: float

    @property
    def seconds(self) -> float:
        return self.lp_seconds + self.mip_seconds

    @property
    def rel_gap(self) -> float:
        return abs(self.objective - self.bound) / max(1.0, abs(self.bound))


def solve_lpfix(
    n: int,
    src: np.ndarray,
    dst: np.ndarray,
    node_cost: np.ndarray,
    appear_cost: np.ndarray,
    disappear_cost: float,
    div_cost: np.ndarray,
    edge_cost: np.ndarray,
    *,
    hops: int = 3,
    time_limit: float | None = None,
) -> LPFixResult:
    """Solve the event ILP LP-first with scipy's HiGHS.

    `time_limit` bounds the whole solve (LP plus restricted MIP). A failed or
    infeasible solve raises; the caller then falls back to SCIP.
    """
    import scipy.sparse as sp
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp

    e = len(src)
    cost = np.concatenate([
        node_cost, appear_cost, np.full(n, float(disappear_cost)),
        div_cost, edge_cost,
    ])
    i = np.arange(n)
    node, app, dis, div = i, n + i, 2 * n + i, 3 * n + i
    edge = 4 * n + np.arange(e)
    nvar = 4 * n + e

    # Flow in: appear + incoming edges == node. Flow out: disappear + outgoing
    # edges == node + division (a dividing node emits two children).
    r_in = sp.coo_matrix(
        (np.r_[np.ones(n), np.ones(e), -np.ones(n)],
         (np.r_[i, dst, i], np.r_[app, edge, node])), shape=(n, nvar))
    r_out = sp.coo_matrix(
        (np.r_[np.ones(n), np.ones(e), -np.ones(n), -np.ones(n)],
         (np.r_[i, src, i, i], np.r_[dis, edge, node, div])), shape=(n, nvar))
    # division <= node: the rows that break total unimodularity.
    cpl = sp.coo_matrix(
        (np.r_[np.ones(n), -np.ones(n)], (np.r_[i, i], np.r_[div, node])),
        shape=(n, nvar))
    eq, ub = sp.vstack([r_in, r_out]).tocsr(), cpl.tocsr()

    t0 = time.perf_counter()
    lp_options = {} if time_limit is None else {"time_limit": float(time_limit)}
    lp = linprog(cost, A_eq=eq, b_eq=np.zeros(2 * n), A_ub=ub, b_ub=np.zeros(n),
                 bounds=(0, 1), method="highs", options=lp_options)
    lp_seconds = time.perf_counter() - t0
    if lp.x is None:
        raise RuntimeError(f"LP relaxation failed: {lp.message}")

    fractional = np.abs(lp.x - np.round(lp.x)) > 1e-6

    a = sp.vstack([eq, ub]).tocsr()
    free = fractional.copy()
    for _ in range(hops):
        touched = a[:, free].getnnz(axis=1) > 0
        grown = free | (a[touched].getnnz(axis=0) > 0)
        if grown.sum() == free.sum():
            break  # the neighbourhood closed
        free = grown
    lo = np.where(free, 0.0, np.round(lp.x))
    hi = np.where(free, 1.0, np.round(lp.x))

    t0 = time.perf_counter()
    mip_options = {"mip_rel_gap": 0.0, "disp": False}
    if time_limit is not None:
        # Whatever the LP did not spend, floored above zero.
        mip_options["time_limit"] = max(0.1, float(time_limit) - lp_seconds)
    res = milp(cost,
               constraints=[LinearConstraint(eq, 0, 0),
                            LinearConstraint(ub, -np.inf, 0)],
               integrality=np.ones(nvar), bounds=Bounds(lo, hi),
               options=mip_options)
    mip_seconds = time.perf_counter() - t0
    if res.x is None:
        raise RuntimeError(
            f"restricted MIP infeasible ({res.message})")

    x = np.round(res.x).astype(bool)
    return LPFixResult(
        kept_edges=np.flatnonzero(x[edge]),
        kept_nodes=np.flatnonzero(x[node]),
        objective=float(res.fun),
        bound=float(lp.fun),
        optimal=abs(res.fun - lp.fun) <= 1e-6 * max(1.0, abs(lp.fun)),
        lp_seconds=lp_seconds,
        mip_seconds=mip_seconds,
    )
