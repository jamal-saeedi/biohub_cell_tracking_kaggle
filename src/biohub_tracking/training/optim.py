"""AdamW parameter groups (no weight decay on biases and normalisation gains)
and the linear warm-up + cosine learning-rate schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class OptimConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    warmup_steps: int = 500
    total_steps: int = 20000
    min_lr_factor: float = 0.05
    grad_clip: float = 1.0

    def to_dict(self) -> dict:
        out = dict(self.__dict__)
        out["betas"] = list(self.betas)
        return out


def build_optimizer(model: nn.Module, config: OptimConfig) -> torch.optim.AdamW:
    decayed, plain = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (plain if parameter.ndim <= 1 else decayed).append((name, parameter))
    return torch.optim.AdamW(
        [
            {"params": [p for _, p in decayed], "weight_decay": config.weight_decay},
            {"params": [p for _, p in plain], "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.betas,
    )


def learning_rate_at(step: int, config: OptimConfig) -> float:
    """Linear warm-up to `learning_rate`, then cosine down to `min_lr_factor`."""
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    span = max(config.total_steps - config.warmup_steps, 1)
    progress = min(max(step - config.warmup_steps, 0) / span, 1.0)
    floor = config.min_lr_factor
    return config.learning_rate * (
        floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def autocast_dtype(device: torch.device) -> torch.dtype | None:
    """bf16 where supported, fp16 on older CUDA GPUs (with a GradScaler), None on CPU."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
