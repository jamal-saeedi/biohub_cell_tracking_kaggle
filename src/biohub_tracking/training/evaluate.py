"""The shared forward/loss step (training and validation), peak decoding and
validation metrics.

With sparse annotation an unmatched detection is not a false positive, so
validation reports recall against annotated cells and `node_ratio`: decoded
peaks over the organisers' cell-count estimate scaled to the crop.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from biohub_tracking.models.isotropic_lineage import (
    DetectionOutput,
    node_descriptors,
)
from biohub_tracking.training.losses import (
    LossTerms,
    LossWeights,
    center_focal_loss,
    count_prior_loss,
    daughter_loss,
    division_loss,
    offset_loss,
    parent_loss,
    velocity_nll,
)
from biohub_tracking.training.targets import DETECTION_STRIDE

#: Suppression window on the detection grid, about 3.25 um on every axis.
NMS_WINDOW: tuple[int, int, int] = (3, 5, 5)


@dataclass
class BatchStats:
    """Non-differentiable numbers worth logging from one batch."""

    values: dict[str, float]
    counts: dict[str, float]


def decode_peaks(
    probability: Tensor,
    offsets: Tensor,
    *,
    threshold: float,
    window: tuple[int, int, int] = NMS_WINDOW,
    stride: tuple[int, int, int] = DETECTION_STRIDE,
    max_peaks: int = 20000,
) -> list[tuple[Tensor, Tensor]]:
    """Non-maximum suppression on `(B,1,Dz,Dy,Dx)` probabilities: a voxel is kept
    when it is its window's arg-max (one peak per plateau), at most `max_peaks`
    per frame. Returns `(coords_native, score)` per batch element, offsets applied.
    """
    padding = tuple(w // 2 for w in window)
    pooled, indices = F.max_pool3d(
        probability, window, stride=1, padding=padding, return_indices=True
    )
    shape = probability.shape[-3:]
    count = int(np.prod(shape))
    flat = torch.arange(count, device=probability.device).view(1, 1, *shape)
    keep = (indices == flat) & (probability >= threshold)
    scale = probability.new_tensor(stride)
    out: list[tuple[Tensor, Tensor]] = []
    for index in range(probability.shape[0]):
        cells = torch.nonzero(keep[index, 0], as_tuple=False)
        if len(cells) == 0:
            out.append((probability.new_zeros(0, 3), probability.new_zeros(0)))
            continue
        if len(cells) > max_peaks:
            scores = probability[index, 0][cells.unbind(-1)]
            cells = cells[torch.topk(scores, max_peaks).indices]
        z, y, x = cells.unbind(-1)
        delta = offsets[index, :, z, y, x].transpose(0, 1)
        out.append(((cells.to(delta) + delta) * scale, probability[index, 0, z, y, x]))
    return out


def match_points(
    predicted_um: np.ndarray, truth_um: np.ndarray, radius_um: float
) -> tuple[int, float]:
    """Greedy one-to-one matching over globally sorted distances; returns the
    number of matches and their mean distance."""
    if len(predicted_um) == 0 or len(truth_um) == 0:
        return 0, float("nan")
    distance = np.linalg.norm(truth_um[:, None, :] - predicted_um[None, :, :], axis=-1)
    order = np.argsort(distance, axis=None)
    used_truth, used_prediction = set(), set()
    distances: list[float] = []
    for flat in order:
        row, column = divmod(int(flat), distance.shape[1])
        if distance[row, column] > radius_um:
            break
        if row in used_truth or column in used_prediction:
            continue
        used_truth.add(row)
        used_prediction.add(column)
        distances.append(float(distance[row, column]))
    return len(distances), float(np.mean(distances)) if distances else float("nan")


class ProposalCoords:
    """Move annotated linker nodes onto the detector's own matched detections.

    At inference the linker sees decoded peaks; in training it sees jittered
    annotations. For each (example, frame) this decodes the current peaks (NMS
    on the detached centre map), matches annotated nodes to them one-to-one
    within `match_um`, and moves each matched node to its peak with probability
    `fraction`. The peak coordinate keeps the offset head in the graph.
    Distractors and unmatched annotations keep their coordinates; decisions are
    cached per (example, frame) so a frame shared by two pairs stays one node set.
    """

    def __init__(
        self,
        dense: DetectionOutput,
        batch: dict,
        device: torch.device,
        *,
        fraction: float,
        threshold: float = 0.3,
        match_um: float = 3.0,
        max_peaks: int = 20000,
    ) -> None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("proposal fraction must be in [0, 1]")
        self.dense, self.batch, self.device = dense, batch, device
        self.fraction, self.threshold, self.match_um = fraction, threshold, match_um
        self.max_peaks = max_peaks
        self.cache: dict[tuple[int, int], tuple[Tensor, Tensor, Tensor]] = {}
        self.real = 0.0
        self.matched = 0.0
        self.replaced = 0.0

    def coords(self, example: int, frame: int, coords: Tensor, gt_index: Tensor) -> Tensor:
        key = (example, frame)
        if key in self.cache:
            cached_in, cached_index, cached_out = self.cache[key]
            if cached_in.shape != coords.shape or not torch.equal(cached_in, coords):
                raise RuntimeError(f"frame {frame} of example {example} has two node sets")
            return cached_out
        out = self._replace(example, frame, coords, gt_index.to(coords.device))
        self.cache[key] = (coords, gt_index, out)
        return out

    def _replace(self, example: int, frame: int, coords: Tensor, gt_index: Tensor) -> Tensor:
        real = torch.nonzero(gt_index >= 0).flatten()
        self.real += float(len(real))
        if len(real) == 0 or self.fraction <= 0:
            return coords
        dense = self.dense
        # (1, Dz, Dy, Dx): the channel axis doubles as `peak_cells`' batch axis.
        probability = torch.sigmoid(dense.center_logits[example, frame].detach().float())
        peaks, _scores = peak_cells(probability, threshold=self.threshold,
                                    max_peaks=self.max_peaks)[0]
        if len(peaks) == 0:
            return coords
        z, y, x = peaks.unbind(-1)
        offsets = dense.offsets_zyx[example, frame, :, z, y, x].transpose(0, 1).float()
        stride = torch.tensor(dense.stride_zyx, dtype=torch.float32, device=offsets.device)
        proposal = (peaks.float() + offsets) * stride  # differentiable in the offsets
        spacing = self.batch["spacing"][example].to(coords.device).float()
        distance = torch.cdist(coords[real].float() * spacing, proposal.detach() * spacing)
        rows, cols = torch.nonzero(distance <= self.match_um, as_tuple=True)
        if len(rows) == 0:
            return coords
        order = torch.argsort(distance[rows, cols]).tolist()
        rows, cols = rows.tolist(), cols.tolist()
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for flat in order:
            r, c = rows[flat], cols[flat]
            if r in used_rows or c in used_cols:
                continue
            used_rows.add(r)
            used_cols.add(c)
            pairs.append((r, c))
        self.matched += float(len(pairs))
        keep = torch.rand(len(pairs)) < self.fraction
        chosen = [p for p, k in zip(pairs, keep.tolist()) if k]
        self.replaced += float(len(chosen))
        if not chosen:
            return coords
        node = real[torch.tensor([r for r, _ in chosen], device=real.device)]
        peak = torch.tensor([c for _, c in chosen], device=proposal.device)
        return coords.index_copy(0, node, proposal[peak].to(coords.dtype))

    def values(self) -> dict[str, float]:
        return {"proposal_real": self.real, "proposal_matched": self.matched,
                "proposal_replaced": self.replaced}


def peak_cells(
    probability: Tensor,
    *,
    threshold: float,
    window: tuple[int, int, int] = NMS_WINDOW,
    max_peaks: int = 20000,
) -> list[tuple[Tensor, Tensor]]:
    """`decode_peaks`' suppression on `(B,Dz,Dy,Dx)`, returning grid cells and scores."""
    padding = tuple(w // 2 for w in window)
    grid = probability[:, None]
    _pooled, indices = F.max_pool3d(grid, window, stride=1, padding=padding, return_indices=True)
    shape = probability.shape[-3:]
    flat = torch.arange(int(np.prod(shape)), device=probability.device).view(1, 1, *shape)
    keep = (indices == flat) & (grid >= threshold)
    out = []
    for index in range(probability.shape[0]):
        cells = torch.nonzero(keep[index, 0], as_tuple=False)
        scores = probability[index][cells.unbind(-1)] if len(cells) else probability.new_zeros(0)
        if len(cells) > max_peaks:
            top = torch.topk(scores, max_peaks).indices
            cells, scores = cells[top], scores[top]
        out.append((cells, scores))
    return out


def batch_losses(
    model: nn.Module,
    batch: dict,
    device: torch.device,
    weights: LossWeights,
    *,
    association: bool = True,
    stride: tuple[int, int, int] = DETECTION_STRIDE,
    proposal_fraction: float = 0.0,
    proposal_threshold: float = 0.3,
    proposal_match_um: float = 3.0,
) -> tuple[LossTerms, DetectionOutput, BatchStats]:
    """One forward pass and every loss term. Association runs per example and
    per frame pair, unbatched. `proposal_fraction > 0` links matched annotations
    at the detector's own detections (`ProposalCoords`)."""
    images = batch["image"].to(device, non_blocking=True)
    dense = model(images)
    terms = LossTerms(total=images.new_zeros(()))

    center, n_pos, n_neg = center_focal_loss(
        dense.center_logits.flatten(0, 1),
        batch["heatmap"].to(device).flatten(0, 1),
        batch["weight"].to(device).flatten(0, 1),
        alpha=weights.focal_alpha,
        beta=weights.focal_beta,
        background_weight=weights.background,
    )
    terms.add("center", center, weights.center, n_pos)
    # Pseudo-labelled voxels, normalised on their own: L_gt + lambda * L_pseudo.
    pseudo_weight = batch.get("pseudo_weight")
    if pseudo_weight is not None and bool((pseudo_weight > 0).any()):
        center_pseudo, n_pseudo, _ = center_focal_loss(
            dense.center_logits.flatten(0, 1),
            batch["heatmap"].to(device).flatten(0, 1),
            pseudo_weight.to(device).flatten(0, 1),
            alpha=weights.focal_alpha,
            beta=weights.focal_beta,
            background_weight=weights.background,
        )
        terms.add("center_pseudo", center_pseudo, weights.center * weights.pseudo, n_pseudo)
    offset, n_offset = offset_loss(
        dense.offsets_zyx.flatten(0, 1),
        batch["offset"].to(device).flatten(0, 1),
        batch["offset_mask"].to(device).flatten(0, 1),
    )
    terms.add("offset", offset, weights.offset, n_offset)

    frames = images.shape[1]
    expected_mass = (
        batch["expected_mass"].to(device).repeat_interleave(frames)
    )
    count, n_count = count_prior_loss(
        dense.center_logits.flatten(0, 1), expected_mass
    )
    terms.add("count", count, weights.count, n_count)

    values: dict[str, float] = {"supervised_voxels": n_pos + n_neg}
    counts: dict[str, float] = {}
    if not association:
        return terms, dense, BatchStats(values, counts)

    proposals = None
    if proposal_fraction > 0:
        proposals = ProposalCoords(dense, batch, device, fraction=proposal_fraction,
                                   threshold=proposal_threshold, match_um=proposal_match_um)
    accumulator = association_terms(model, dense, batch, device, weights, proposals)
    accumulator.emit(terms, weights)
    values.update(accumulator.values())
    if proposals is not None:
        values.update(proposals.values())
    counts.update(accumulator.counts)
    return terms, dense, BatchStats(values, counts)


def association_terms(
    model: nn.Module,
    dense: DetectionOutput,
    batch: dict,
    device: torch.device,
    weights: LossWeights,
    proposals: ProposalCoords | None = None,
) -> _Accumulator:
    """Every association term of one batch, count-weighted, unemitted."""
    zero = dense.center_logits.new_zeros((), dtype=torch.float32)
    accumulator = _Accumulator(zero)
    for example, pairs in enumerate(batch["pairs"]):
        spacing = batch["spacing"][example].to(device)
        for pair in pairs:
            _association_terms(
                model, dense, example, pair, spacing, device, weights, accumulator,
                proposals,
            )
    return accumulator


class _Accumulator:
    """Count-weighted sums of the per-pair means over a batch."""

    def __init__(self, zero: Tensor) -> None:
        self.sums = {k: zero.clone() for k in
                     ("parent", "parent_pseudo", "division", "velocity", "daughter")}
        self.counts = dict.fromkeys(self.sums, 0.0)
        self.extra = {
            "parent_correct": 0.0, "gt_edges": 0.0, "gt_edges_in_graph": 0.0,
            "forced_edges": 0.0, "candidate_edges": 0.0, "division_positive": 0.0,
            "division_hit": 0.0, "null_targets": 0.0,
        }

    def add(self, name: str, value: Tensor, count: float) -> None:
        if count > 0:
            self.sums[name] = self.sums[name] + value * count
            self.counts[name] += count

    def emit(self, terms: LossTerms, weights: LossWeights) -> None:
        for name in self.sums:
            count = self.counts[name]
            value = self.sums[name] / count if count else self.sums[name]
            if name == "parent_pseudo":
                if not count:
                    continue
                weight = weights.parent * weights.pseudo
            else:
                weight = getattr(weights, name)
            terms.add(name, value, weight, count)

    def values(self) -> dict[str, float]:
        out = dict(self.extra)
        if self.counts["parent"]:
            out["parent_accuracy"] = (
                self.extra["parent_correct"] / self.counts["parent"]
            )
        if self.extra["gt_edges"]:
            out["candidate_recall"] = (
                self.extra["gt_edges_in_graph"] / self.extra["gt_edges"]
            )
        if self.extra["division_positive"]:
            out["division_recall"] = (
                self.extra["division_hit"] / self.extra["division_positive"]
            )
        return out


def _association_terms(
    model: nn.Module,
    dense: DetectionOutput,
    example: int,
    pair: dict,
    spacing: Tensor,
    device: torch.device,
    weights: LossWeights,
    accumulator: _Accumulator,
    proposals: ProposalCoords | None = None,
) -> None:
    frame = int(pair["source_frame"])
    later = int(pair["target_frame"])
    dt = float(pair["dt"])
    source_coords = pair["source_coords"].to(device)
    target_coords = pair["target_coords"].to(device)
    if proposals is not None:
        source_coords = proposals.coords(example, frame, source_coords, pair["source_gt_index"])
        target_coords = proposals.coords(example, later, target_coords, pair["target_gt_index"])
    edge_index = pair["edge_index"].to(device)
    source = node_descriptors(dense, example, frame, source_coords)
    target = node_descriptors(dense, example, later, target_coords)
    source_um = source_coords * spacing
    target_um = target_coords * spacing
    output = model.association(
        source, target, source_um, target_um, edge_index, dt=dt
    )

    parent_edge = pair["parent_edge"].to(device)
    weight = pair.get("parent_weight")
    weight = (torch.ones(parent_edge.shape, device=device) if weight is None
              else weight.to(device).float())
    pseudo = pair.get("parent_pseudo")
    pseudo = (torch.zeros(parent_edge.shape, dtype=torch.bool, device=device)
              if pseudo is None else pseudo.to(device))
    # GT and pseudo parent labels are normalised apart (L_gt + lambda * L_pseudo).
    loss, count, accuracy = parent_loss(
        output, edge_index, parent_edge, torch.where(pseudo, 0.0, weight),
    )
    accumulator.add("parent", loss, count)
    if count and accuracy == accuracy:  # NaN-safe
        accumulator.extra["parent_correct"] += accuracy * count
    if bool(pseudo.any()):
        loss, count, _ = parent_loss(
            output, edge_index, parent_edge, torch.where(pseudo, weight, 0.0),
        )
        accumulator.add("parent_pseudo", loss, count)
    accumulator.extra["gt_edges"] += float(pair["n_gt_edges"])
    accumulator.extra["gt_edges_in_graph"] += float(pair["n_gt_edges_in_graph"])
    accumulator.extra["forced_edges"] += float(pair["n_forced"])
    accumulator.extra["candidate_edges"] += float(edge_index.shape[1])
    accumulator.extra["null_targets"] += float(pair["n_null_targets"])

    division_mask = pair["division_mask"].to(device)
    division_label = pair["division_label"].to(device)
    loss, count = division_loss(
        output.division_logits, division_label, division_mask,
        pos_weight=weights.division_pos_weight,
    )
    accumulator.add("division", loss, count)
    positive = (division_label > 0.5) & (division_mask > 0)
    accumulator.extra["division_positive"] += float(positive.sum())
    if bool(positive.any()):
        hit = (output.division_logits[positive] > 0).sum()
        accumulator.extra["division_hit"] += float(hit)

    loss, count = velocity_nll(
        output.velocity_um, output.log_variance,
        pair["velocity_um"].to(device), pair["velocity_mask"].to(device),
    )
    accumulator.add("velocity", loss, count)

    triplets = pair["triplets"].to(device)
    if triplets.numel():
        logits = model.association.score_daughter_pairs(
            output, triplets, source_um, target_um
        )
        loss, count = daughter_loss(logits, pair["triplet_label"].to(device))
        accumulator.add("daughter", loss, count)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    device: torch.device,
    weights: LossWeights,
    *,
    detection_threshold: float = 0.3,
    match_radius_um: float = 3.0,
    proposal_metrics: bool = False,
) -> dict[str, float]:
    """Metrics over a fixed sample of validation crops. The loss links the
    annotated coordinates; `proposal_metrics` also reports parent loss and
    accuracy with every matched annotation at its decoded detection."""
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    counts: dict[str, float] = {}
    matched = missed = predicted = frame_count = 0
    expected = 0.0
    distances: list[float] = []

    for batch in loader:
        terms, dense, stats = batch_losses(model, batch, device, weights)
        _accumulate(totals, counts, "loss", float(terms.total.detach()), 1.0)
        for name, value in terms.parts.items():
            _accumulate(totals, counts, f"loss_{name}", float(value), 1.0)
        for name, value in stats.values.items():
            _accumulate(totals, counts, name, float(value), 1.0)
        if proposal_metrics:
            proposals = ProposalCoords(dense, batch, device, fraction=1.0,
                                       threshold=detection_threshold,
                                       match_um=match_radius_um)
            accumulator = association_terms(model, dense, batch, device, weights, proposals)
            if accumulator.counts["parent"]:
                parent = accumulator.sums["parent"] / accumulator.counts["parent"]
                _accumulate(totals, counts, "proposal_loss_parent", float(parent), 1.0)
            extra = accumulator.values()
            if "parent_accuracy" in extra:
                _accumulate(totals, counts, "proposal_parent_accuracy",
                            extra["parent_accuracy"], 1.0)
            for name, value in proposals.values().items():
                _accumulate(totals, counts, name, value, 1.0)

        probability = torch.sigmoid(dense.center_logits.flatten(0, 1).float())
        peaks = decode_peaks(
            probability,
            dense.offsets_zyx.flatten(0, 1).float(),
            threshold=detection_threshold,
            stride=dense.stride_zyx,
        )
        frames = dense.center_logits.shape[1]
        for flat, (coords, _score) in enumerate(peaks):
            example, frame = divmod(flat, frames)
            scale = batch["spacing"][example].numpy()
            truth = batch["points"][example][frame].numpy() * scale
            hits, mean_distance = match_points(
                coords.cpu().numpy() * scale, truth, match_radius_um
            )
            matched += hits
            missed += len(truth) - hits
            predicted += len(coords)
            expected += float(batch["expected_cells"][example])
            frame_count += 1
            if mean_distance == mean_distance:
                distances.append(mean_distance * hits)

    out = {name: totals[name] / counts[name] for name in totals}
    total_truth = matched + missed
    out["det_recall"] = matched / total_truth if total_truth else float("nan")
    out["det_peaks_per_frame"] = (
        predicted / frame_count if frame_count else float("nan")
    )
    out["node_ratio"] = predicted / expected if expected else float("nan")
    out["loc_error_um"] = (sum(distances) / matched) if matched else float("nan")
    out["matched_nodes"] = float(matched)
    out["annotated_nodes"] = float(total_truth)
    model.train(was_training)
    return out


def _accumulate(
    totals: dict[str, float],
    counts: dict[str, float],
    name: str,
    value: float,
    weight: float,
) -> None:
    if value != value:  # skip NaN
        return
    totals[name] = totals.get(name, 0.0) + value * weight
    counts[name] = counts.get(name, 0.0) + weight
