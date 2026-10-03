"""Losses for three-state supervision.

Every term divides by the number of examples it supervised. The centre term
normalises its positive and negative sums separately (a crop holds a handful of
annotated centres and a few hundred thousand verified-background voxels).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor
from torch.nn import functional as F

from biohub_tracking.models.isotropic_lineage import (
    AssociationOutput,
    parent_log_probabilities,
)


@dataclass(frozen=True)
class LossWeights:
    center: float = 1.0
    offset: float = 1.0
    parent: float = 1.0
    division: float = 0.5
    velocity: float = 0.1
    daughter: float = 0.25
    #: Count prior (`count_prior_loss`).
    count: float = 0.0
    #: Weight of the verified-background voxels inside the centre term.
    background: float = 1.0
    #: lambda in `L_gt + lambda * L_pseudo` for the centre and parent terms.
    pseudo: float = 0.5
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    division_pos_weight: float = 20.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class LossTerms:
    """Every term, plus the counts that produced it, for the training log."""

    total: Tensor
    parts: dict[str, Tensor] = field(default_factory=dict)
    counts: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: Tensor, weight: float, count: float) -> None:
        self.parts[name] = value.detach()
        self.counts[name] = float(count)
        if weight:
            self.total = self.total + weight * value


def center_focal_loss(
    logits: Tensor,
    target: Tensor,
    weight: Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    background_weight: float = 1.0,
) -> tuple[Tensor, float, float]:
    """Penalty-reduced focal loss over supervised voxels only.

    `weight` is the three-state mask (0 = unknown, no loss); soft weights in
    (0, 1] make each branch a weighted mean (pseudo-labels carry the teacher's
    confidence). The positive sum is normalised by its weight and the negative
    sum by the negative weight. Returns the loss and the positive / negative
    voxel counts.
    """
    logits = logits.float()
    target = target.float()
    weight = weight.float()
    supervised = weight > 0
    positive = supervised & (target >= 1.0)
    negative = supervised & ~positive
    n_pos = float(positive.sum())
    n_neg = float(negative.sum())

    log_p = F.logsigmoid(logits)
    log_1mp = F.logsigmoid(-logits)
    probability = torch.sigmoid(logits)

    pos_term = -((1 - probability) ** alpha) * log_p
    focal = ((1 - target) ** beta) * (probability**alpha)
    neg_term = -focal * log_1mp
    w_pos = weight[positive]
    w_neg = weight[negative]
    loss = (pos_term[positive] * w_pos).sum() / w_pos.sum().clamp_min(1.0)
    if n_neg:
        denominator = w_neg.sum().detach().clamp_min(1e-6)
        loss = loss + background_weight * (neg_term[negative] * w_neg).sum() / denominator
    return loss, n_pos, n_neg


def offset_loss(predicted: Tensor, target: Tensor, mask: Tensor) -> tuple[Tensor, float]:
    """Smooth L1 (beta 0.1 grid cells) on the nearest-cell residual, where an
    annotation sits."""
    count = float(mask.sum())
    if count == 0:
        return predicted.sum() * 0.0, 0.0
    error = F.smooth_l1_loss(
        predicted.float(), target.float(), beta=0.1, reduction="none"
    )
    return (error * mask).sum() / (3.0 * count), count


def parent_loss(
    output: AssociationOutput, edge_index: Tensor, parent_edge: Tensor,
    parent_weight: Tensor | None = None,
) -> tuple[Tensor, float, float]:
    """Cross entropy over each target's candidate parents plus the null class.

    `parent_edge[j]` is the edge position of j's true parent, -1 when the parent
    was removed on purpose (null is correct), -2 when unknown (no loss).
    `parent_weight` (default 1) makes the term a weighted mean. Returns the
    loss, the summed weight and the top-1 accuracy.
    """
    edge_logp, null_logp = parent_log_probabilities(
        output.edge_logits, output.no_parent_logits, edge_index
    )
    if parent_weight is None:
        parent_weight = torch.ones_like(null_logp)
    parent_weight = parent_weight.to(null_logp.dtype)
    linked = (parent_edge >= 0) & (parent_weight > 0)
    nulled = (parent_edge == -1) & (parent_weight > 0)
    count = float(parent_weight[linked].sum() + parent_weight[nulled].sum())
    if count == 0:
        return edge_logp.sum() * 0.0, 0.0, float("nan")
    total = edge_logp.new_zeros(())
    if linked.any():
        total = total - (edge_logp[parent_edge[linked]] * parent_weight[linked]).sum()
    if nulled.any():
        total = total - (null_logp[nulled] * parent_weight[nulled]).sum()
    return total / count, count, _parent_accuracy(
        edge_logp, null_logp, edge_index, parent_edge, linked, nulled
    )


@torch.no_grad()
def _parent_accuracy(
    edge_logp: Tensor,
    null_logp: Tensor,
    edge_index: Tensor,
    parent_edge: Tensor,
    linked: Tensor,
    nulled: Tensor,
) -> float:
    """Fraction of supervised targets whose argmax over {parents, null} is right."""
    best = null_logp.clone()
    if edge_index.shape[1]:
        best.scatter_reduce_(
            0, edge_index[1], edge_logp, reduce="amax", include_self=True
        )
    correct = 0.0
    if bool(linked.any()):
        chosen = edge_logp[parent_edge[linked]]
        correct += float((chosen >= best[edge_index[1][parent_edge[linked]]]).sum())
    if bool(nulled.any()):
        correct += float((null_logp[nulled] >= best[nulled]).sum())
    return correct / float(linked.sum() + nulled.sum())


def division_loss(
    logits: Tensor, label: Tensor, mask: Tensor, *, pos_weight: float
) -> tuple[Tensor, float]:
    """Weighted BCE on sources whose children are observed: two annotated
    daughters (positive) or one annotated child (negative)."""
    count = float(mask.sum())
    if count == 0:
        return logits.sum() * 0.0, 0.0
    scale = logits.new_full((), float(pos_weight))
    weights = torch.where(label > 0.5, scale, torch.ones_like(scale)) * mask
    loss = F.binary_cross_entropy_with_logits(
        logits.float(), label.float(), weight=weights, reduction="sum"
    )
    return loss / weights.sum().clamp_min(1e-6), count


def velocity_nll(
    velocity: Tensor, log_variance: Tensor, target: Tensor, mask: Tensor
) -> tuple[Tensor, float]:
    """Diagonal Gaussian NLL `0.5 * ((v - v_gt)^2 * exp(-logvar) + logvar)` on
    annotated one-to-one continuations."""
    count = float(mask.sum())
    if count == 0:
        return velocity.sum() * 0.0, 0.0
    residual = (velocity.float() - target.float()) ** 2
    logvar = log_variance.float()
    per_node = 0.5 * (residual * torch.exp(-logvar) + logvar)
    return (per_node.sum(-1) * mask).sum() / (3.0 * count), count


def daughter_loss(logits: Tensor, label: Tensor) -> tuple[Tensor, float]:
    if logits.numel() == 0:
        return logits.sum() * 0.0, 0.0
    return (
        F.binary_cross_entropy_with_logits(logits.float(), label.float()),
        float(logits.numel()),
    )


def count_prior_loss(
    logits: Tensor, expected_mass: Tensor, *, beta: float = 0.2
) -> tuple[Tensor, float]:
    """Pull the predicted heatmap mass of each crop-frame toward its expected
    mass: the organisers' per-movie cell-count estimate scaled to the crop,
    times `targets.gaussian_mass`. A relative, two-sided smooth L1; frames
    without an estimate (`expected_mass <= 0`) are skipped.

    The unknown region has no other loss term; without this prior the detector
    over-fires there.
    """
    valid = expected_mass > 0
    count = float(valid.sum())
    if count == 0:
        return logits.sum() * 0.0, 0.0
    predicted = torch.sigmoid(logits.float()).flatten(1).sum(1)
    ratio = predicted[valid] / expected_mass[valid].float()
    return (
        F.smooth_l1_loss(ratio, torch.ones_like(ratio), beta=beta),
        count,
    )
