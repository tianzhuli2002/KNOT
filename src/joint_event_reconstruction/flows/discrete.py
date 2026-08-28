"""Masked discrete Flow Matching over complete feasible assignment states."""

from __future__ import annotations

import torch
from torch.nn import functional as F


class MaskedDiscreteFlow:
    """Absorbing unmasking path from MASK at t=0 to a clean state at t=1."""

    def __init__(self, *, eps: float = 1.0e-4):
        if not 0 < eps < 1:
            raise ValueError("Discrete Flow Matching eps must lie in (0,1)")
        self.eps = eps

    def sample_time(self, batch_size: int, *, device: torch.device, dtype: torch.dtype, generator: torch.Generator | None = None) -> torch.Tensor:
        return torch.rand(batch_size, device=device, dtype=dtype, generator=generator) * (1 - self.eps)

    def training_state(self, target: torch.Tensor, time: torch.Tensor, mask_state: int, *, generator: torch.Generator | None = None) -> torch.Tensor:
        if target.shape != time.shape:
            raise ValueError("Discrete target and time must have identical [B] shape")
        reveal = torch.rand(len(target), device=target.device, dtype=time.dtype, generator=generator) < time
        return torch.where(reveal, target, torch.full_like(target, mask_state))

    def loss(self, clean_logits: torch.Tensor, target: torch.Tensor, current_state: torch.Tensor, time: torch.Tensor, *, mask_state: int) -> torch.Tensor:
        """Clean-state posterior prediction on states that remain masked."""
        masked = current_state == mask_state
        if not masked.any():
            return clean_logits.sum() * 0
        return F.cross_entropy(clean_logits[masked], target[masked])

    @torch.no_grad()
    def step(self, state: torch.Tensor, clean_logits: torch.Tensor, time: torch.Tensor, next_time: torch.Tensor, *, mask_state: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if state.shape != time.shape or time.shape != next_time.shape:
            raise ValueError("Discrete state/time/next_time must all have shape [B]")
        if (next_time < time).any() or (time < 0).any() or (next_time > 1).any():
            raise ValueError("Discrete flow times must satisfy 0 <= t <= t_next <= 1")
        masked = state == mask_state
        if not masked.any():
            return state
        reveal_probability = ((next_time - time) / (1 - time).clamp_min(self.eps)).clamp(0, 1)
        reveal = torch.rand(len(state), device=state.device, dtype=time.dtype, generator=generator) < reveal_probability
        reveal &= masked
        if not reveal.any():
            return state
        probabilities = clean_logits.softmax(dim=1)
        sampled = torch.multinomial(probabilities[reveal], 1, generator=generator).squeeze(1)
        updated = state.clone()
        updated[reveal] = sampled
        return updated
