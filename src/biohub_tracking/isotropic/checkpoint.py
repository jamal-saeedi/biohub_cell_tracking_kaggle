"""Build a model from a checkpoint: the weights (`model`) and the constructor
arguments of its architecture (`config.model`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from biohub_tracking.models.isotropic_lineage import IsotropicLineageNet
from biohub_tracking.models.multiscale_lineage import build_lineage_model

__all__ = ["LoadedModel", "load_isotropic_model"]


@dataclass
class LoadedModel:
    """An eval-mode model and the other ensemble members."""

    model: IsotropicLineageNet  # or MultiScaleLineageNet: the same call contract
    ensemble: tuple = ()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


def load_isotropic_model(checkpoint: Path | str, device: torch.device) -> LoadedModel:
    """Build the architecture the checkpoint records and load its weights."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"no checkpoint at {checkpoint}")
    blob = torch.load(checkpoint, map_location="cpu", weights_only=True)

    model = build_lineage_model(**blob["config"]["model"])

    missing, unexpected = model.load_state_dict(blob["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{checkpoint}: state dict does not match the recorded architecture "
            f"(missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]})"
        )

    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedModel(model=model)
