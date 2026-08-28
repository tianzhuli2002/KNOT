"""Continuous Flow Matching probability paths."""

from __future__ import annotations

import torch


def expand_time(time: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if time.shape != (len(target),):
        raise ValueError(f"time must have shape [B], found {tuple(time.shape)}")
    return time.reshape((len(target),) + (1,) * (target.ndim - 1))


def linear_path(
    source: torch.Tensor,
    target: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Straight conditional path x_t=(1-t)x_0+t x_1 and its velocity."""
    if source.shape != target.shape:
        raise ValueError(
            f"Flow source and target shapes differ: {tuple(source.shape)} vs {tuple(target.shape)}"
        )
    t = expand_time(time, target)
    return (1 - t) * source + t * target, target - source
