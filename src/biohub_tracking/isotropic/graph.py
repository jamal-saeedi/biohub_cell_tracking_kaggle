"""Candidate parent graph between consecutive frames, and association scoring.

Candidates are the union of "k nearest sources for each target" and "k nearest
targets for each source" within a radius (KD-tree, deduplicated). Every
candidate reaches the solver with its probability; nothing is thresholded here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree

from biohub_tracking.isotropic.config import CandidateConfig
from biohub_tracking.isotropic.decode import FrameNodes
from biohub_tracking.models.isotropic_lineage import (
    IsotropicLineageNet,
    parent_log_probabilities,
)

__all__ = ["PairAssociation", "candidate_edges_kdtree", "associate_pair"]


@dataclass
class PairAssociation:
    """Scored candidate parents for one ordered frame pair `(source, target)`."""

    source_frame: int
    target_frame: int
    edge_index: np.ndarray  # (2,E) int64, local indices into each frame's nodes
    edge_logp: np.ndarray  # (E,) log P(parent = source_i | target_j)
    null_logp: np.ndarray  # (n_target,) log P(no parent | target_j)
    division_logit: np.ndarray  # (n_source,) raw logit, not a probability
    distance_um: np.ndarray  # (E,) Euclidean, microns
    velocity_um: np.ndarray  # (n_source,3) predicted displacement per interval

    @property
    def n_edges(self) -> int:
        return int(self.edge_index.shape[1])


def candidate_edges_kdtree(
    source_um: np.ndarray,
    target_um: np.ndarray,
    *,
    radius_um: float,
    max_per_node: int,
) -> np.ndarray:
    """Symmetric k-nearest-within-radius graph as `(2,E)` int64, unique.

    Returns local indices: row 0 into `source_um`, row 1 into `target_um`.
    """
    n_source, n_target = len(source_um), len(target_um)
    if n_source == 0 or n_target == 0:
        return np.empty((2, 0), dtype=np.int64)

    pairs: list[np.ndarray] = []
    # k nearest sources for each target.
    k = min(max_per_node, n_source)
    distance, index = cKDTree(source_um).query(
        target_um, k=k, distance_upper_bound=radius_um
    )
    distance, index = np.atleast_2d(distance.T).T, np.atleast_2d(index.T).T
    valid = np.isfinite(distance) & (index < n_source)
    rows = index[valid]
    cols = np.broadcast_to(np.arange(n_target)[:, None], index.shape)[valid]
    pairs.append(np.stack((rows, cols)))

    # k nearest targets for each source.
    k = min(max_per_node, n_target)
    distance, index = cKDTree(target_um).query(
        source_um, k=k, distance_upper_bound=radius_um
    )
    distance, index = np.atleast_2d(distance.T).T, np.atleast_2d(index.T).T
    valid = np.isfinite(distance) & (index < n_target)
    cols = index[valid]
    rows = np.broadcast_to(np.arange(n_source)[:, None], index.shape)[valid]
    pairs.append(np.stack((rows, cols)))

    edges = np.concatenate(pairs, axis=1).astype(np.int64)
    if edges.shape[1] == 0:
        return np.empty((2, 0), dtype=np.int64)
    # Duplicate candidates would double-count parent mass.
    flat = np.unique(edges[0] * np.int64(n_target) + edges[1])
    return np.stack((flat // n_target, flat % n_target)).astype(np.int64)


@torch.no_grad()
def associate_pair(
    model: IsotropicLineageNet,
    source: FrameNodes,
    target: FrameNodes,
    *,
    spacing: tuple[float, float, float],
    config: CandidateConfig,
) -> PairAssociation:
    """Score every candidate parent of `target` among `source` (coordinates in microns)."""
    gap = target.frame - source.frame
    if gap <= 0:
        raise ValueError(f"target frame {target.frame} must follow {source.frame}")

    source_um = source.um(spacing)
    target_um = target.um(spacing)
    edges = candidate_edges_kdtree(
        source_um,
        target_um,
        radius_um=config.radius_um * gap,
        max_per_node=config.max_per_node,
    )

    device = source.descriptors.device
    edge_index = torch.as_tensor(edges, dtype=torch.long, device=device)
    src_um = torch.as_tensor(source_um, dtype=torch.float32, device=device)
    tgt_um = torch.as_tensor(target_um, dtype=torch.float32, device=device)

    output = model.association(
        source.descriptors, target.descriptors, src_um, tgt_um, edge_index, dt=float(gap)
    )
    edge_logp, null_logp = parent_log_probabilities(
        output.edge_logits, output.no_parent_logits, edge_index
    )
    edge_logp_np = edge_logp.float().cpu().numpy()

    if edges.shape[1]:
        distance = np.linalg.norm(source_um[edges[0]] - target_um[edges[1]], axis=1)
    else:
        distance = np.zeros(0, dtype=np.float64)

    return PairAssociation(
        source_frame=source.frame,
        target_frame=target.frame,
        edge_index=edges,
        edge_logp=edge_logp_np,
        null_logp=null_logp.float().cpu().numpy(),
        division_logit=output.division_logits.float().cpu().numpy(),
        distance_um=distance,
        velocity_um=output.velocity_um.float().cpu().numpy(),
    )
