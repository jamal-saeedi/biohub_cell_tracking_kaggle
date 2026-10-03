"""Pipeline configuration. The defaults are the submitted recipe.

One frozen dataclass per stage, composed into `IsotropicConfig`. The model
files (checkpoints and edge re-scorers) are attached by `recipe.shipped_config`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

#: Native voxel spacing in microns, Z, Y, X.
SPACING_UM: tuple[float, float, float] = (1.625, 0.40625, 0.40625)


@dataclass(frozen=True)
class WindowConfig:
    """Frames per forward pass. Each frame is decoded from its own most-centred
    window (`decode.window_plan`)."""

    frames: int = 3

    def __post_init__(self) -> None:
        if self.frames < 1:
            raise ValueError("frames must be >= 1")


@dataclass(frozen=True)
class DetectionConfig:
    """Centre decoding and test-time augmentation."""

    threshold: float = 0.5
    """Probability floor for a peak."""

    nms_window: tuple[int, int, int] = (3, 5, 5)
    """Peak-suppression window on the `(1, 2, 2)`-strided detection grid, Z, Y, X."""

    max_peaks: int = 20000
    """Per-frame cap on decoded peaks."""

    tta_views: int = 4
    """D4 views averaged per model when `member_tta_views` is empty (1 to 8)."""

    member_tta_views: tuple[int, ...] = (4, 4, 4, 3, 2, 2)
    """D4 views per ensemble model, primary first, each 1 to 8."""


@dataclass(frozen=True)
class CandidateConfig:
    """Candidate parents of each node in the previous frame."""

    radius_um: float = 20.0
    max_per_node: int = 4
    """k of the symmetric k-nearest-within-radius graph."""


@dataclass(frozen=True)
class SolverConfig:
    """Event-ILP costs, in negative-log-likelihood units (see `solver`)."""

    node_logit_bias: float = 0.0
    """Subtracted from every centre logit: raising it makes nodes more expensive."""

    node_weight_scale: float = 1.0
    disappearance_weight: float = 12.0
    """Cost of a track ending (the model has no disappearance head)."""

    division_weight_scale: float = 1.0
    division_bias: float = 0.0
    division_cost_floor: float = 0.0
    """Division costs are clamped at this floor, so a division never pays the solver."""

    edge_distance_weight: float = 0.2
    """Weight (per micron) of each edge's drift-corrected displacement: the
    displacement minus the frame's global stage shift."""

    association_temperature: float = 0.9
    """Temperature of each target's parent softmax (candidates + the null class)."""

    appearance_bias: float = 0.5
    """Added to the appearance cost of every node after the first frame."""

    lp_first_hops: int = 3
    """How far the LP-first solve frees variables around the fractional ones."""

    lp_first_gap: float = 0.005
    """Relative gap to the LP bound the LP-first answer may have before SCIP is run."""


@dataclass(frozen=True)
class SmoothingConfig:
    """Line-fit smoothing of track positions (coordinates only, never topology)."""

    enabled: bool = True
    weight: float = 0.8
    """Blend between the fitted and the decoded position."""
    window: int = 2
    """Chain neighbours on either side in the degree-1 fit."""


@dataclass(frozen=True)
class IsotropicConfig:
    """Everything one inference run needs."""

    checkpoint: Path | None = None
    """The primary model."""

    ensemble_checkpoints: tuple[Path, ...] = ()
    """The other ensemble members. Their centre logits and offsets are averaged
    with the primary's into one node set; each scores the shared candidate edges
    with its own association head."""

    edge_rescorer: Path | None = None
    """One re-scorer for a single model (the primary-only fallback)."""

    ensemble_rescorers: tuple[Path, ...] = ()
    """One re-scorer per model, primary first: each model's own association terms
    are re-scored by its own trees, then averaged in probability space."""

    window: WindowConfig = field(default_factory=WindowConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)

    smooth_drift_compensated: bool = True
    """Smooth in the stage's frame of reference: the per-frame stage shift is
    removed before smoothing and restored after (`pipeline.smooth_positions`)."""

    amp_dtype: str = "auto"
    """Autocast precision: ``"auto"`` (bf16 on GPUs with native bf16, else
    fp16), ``"bf16"`` or ``"fp16"``."""

    def __post_init__(self) -> None:
        views = self.detection.member_tta_views
        if views:
            if any(int(v) != v or not 1 <= v <= 8 for v in views):
                raise ValueError(f"detection.member_tta_views must each be an integer 1..8: {views}")
            if self.ensemble_checkpoints and len(views) != 1 + len(self.ensemble_checkpoints):
                raise ValueError(f"detection.member_tta_views has {len(views)} view counts for "
                                 f"{1 + len(self.ensemble_checkpoints)} models (primary first)")
        if not self.ensemble_rescorers:
            return
        object.__setattr__(self, "ensemble_rescorers",
                           tuple(Path(p) for p in self.ensemble_rescorers))
        if not self.ensemble_checkpoints:
            raise ValueError("ensemble_rescorers needs an ensemble; one model uses edge_rescorer")
        if len(self.ensemble_rescorers) != 1 + len(self.ensemble_checkpoints):
            raise ValueError(
                f"ensemble_rescorers has {len(self.ensemble_rescorers)} re-scorers for "
                f"{1 + len(self.ensemble_checkpoints)} models (one per model, primary first)")
        if self.edge_rescorer is not None:
            raise ValueError("ensemble_rescorers replaces edge_rescorer; set edge_rescorer=None")

    def with_detection(self, **overrides) -> IsotropicConfig:
        return replace(self, detection=replace(self.detection, **overrides))


def config_to_dict(config: IsotropicConfig) -> dict:
    """JSON-safe snapshot; how a shard worker receives its configuration."""
    from dataclasses import asdict

    payload = asdict(config)
    payload["checkpoint"] = str(config.checkpoint) if config.checkpoint else None
    payload["ensemble_checkpoints"] = [str(p) for p in config.ensemble_checkpoints]
    payload["edge_rescorer"] = str(config.edge_rescorer) if config.edge_rescorer else None
    payload["ensemble_rescorers"] = [str(p) for p in config.ensemble_rescorers]
    return payload


def config_from_dict(payload: dict) -> IsotropicConfig:
    """Inverse of `config_to_dict`; unknown keys are an error."""
    payload = dict(payload)
    nested = {
        "window": WindowConfig,
        "detection": DetectionConfig,
        "candidates": CandidateConfig,
        "solver": SolverConfig,
        "smoothing": SmoothingConfig,
    }
    kwargs: dict = {}
    for name, cls in nested.items():
        if name in payload:
            section = dict(payload.pop(name))
            for key in ("nms_window", "member_tta_views"):
                if key in section:
                    section[key] = tuple(section[key])
            kwargs[name] = cls(**section)
    if payload.get("checkpoint"):
        payload["checkpoint"] = Path(payload["checkpoint"])
    if payload.get("edge_rescorer"):
        payload["edge_rescorer"] = Path(payload["edge_rescorer"])
    payload["ensemble_checkpoints"] = tuple(Path(p) for p in payload.get("ensemble_checkpoints", ()))
    payload["ensemble_rescorers"] = tuple(Path(p) for p in payload.get("ensemble_rescorers", ()))
    return IsotropicConfig(**payload, **kwargs)
