"""Exponential moving average of the weights, kept in float32 (used at inference)."""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn


class WeightAverage:
    """`decay`-weighted average of every floating parameter and buffer."""

    def __init__(
        self, model: nn.Module, decay: float = 0.999, warmup: int = 1000
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must lie strictly between 0 and 1")
        self.decay = decay
        self.warmup = max(int(warmup), 0)
        self.steps = 0
        self.shadow = {
            name: tensor.detach().clone().float()
            for name, tensor in _tracked(model)
        }

    def current_decay(self) -> float:
        """Ramped decay, `min(decay, (1 + steps) / (warmup + steps))`."""
        if self.warmup == 0:
            return self.decay
        return min(self.decay, (1.0 + self.steps) / (self.warmup + self.steps))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        decay = self.current_decay()
        for name, tensor in _tracked(model):
            shadow = self.shadow[name]
            if shadow.shape != tensor.shape:
                raise ValueError(f"EMA shape mismatch for {name}")
            shadow.mul_(decay).add_(tensor.detach().float(), alpha=1.0 - decay)
        self.steps += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, tensor in _tracked(model):
            tensor.copy_(self.shadow[name].to(tensor.dtype))

    @contextmanager
    def evaluated(self, model: nn.Module):
        """Temporarily install the averaged weights, then restore the live ones."""
        backup = {name: t.detach().clone() for name, t in _tracked(model)}
        self.copy_to(model)
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, tensor in _tracked(model):
                    tensor.copy_(backup[name])


def _tracked(model: nn.Module):
    """Floating parameters and buffers, in a stable order."""
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point():
            yield name, parameter
    for name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            yield f"buffer.{name}", buffer
