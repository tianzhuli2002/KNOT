"""Gaussian VP diffusion with cosine schedule and deterministic DDIM sampling."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class StandardDiffusion(nn.Module):
    def __init__(self, denoiser: nn.Module, training_steps: int = 1000, cosine_offset: float = 0.008):
        super().__init__()
        if training_steps <= 1:
            raise ValueError("Standard diffusion needs at least two training steps")
        self.denoiser = denoiser
        self.training_steps = training_steps
        t = torch.linspace(0.0, 1.0, training_steps + 1, dtype=torch.float64)
        values = torch.cos(((t + cosine_offset) / (1 + cosine_offset)) * math.pi / 2).square()
        values = (values / values[0]).clamp(1.0e-8, 1.0)
        self.register_buffer("alpha_bar", values.to(torch.float32))

    def loss(self, target: torch.Tensor, visible: torch.Tensor, mask: torch.Tensor, parent: torch.Tensor) -> dict:
        batch = len(target)
        timestep = torch.randint(1, self.training_steps + 1, (batch,), device=target.device)
        epsilon = torch.randn_like(target)
        alpha = self.alpha_bar[timestep].view(batch, 1, 1)
        noisy = alpha.sqrt() * target + (1 - alpha).sqrt() * epsilon
        prediction = self.denoiser(
            noisy, timestep.to(target.dtype) / self.training_steps, visible, mask, parent,
        )
        return {"loss": F.mse_loss(prediction, epsilon), "noise": epsilon, "prediction": prediction}

    @torch.no_grad()
    def sample(
        self, shape: tuple[int, int, int], visible: torch.Tensor, mask: torch.Tensor,
        parent: torch.Tensor, *, inference_steps: int, generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, int]:
        if inference_steps <= 0 or inference_steps > self.training_steps:
            raise ValueError("DDIM inference_steps must be in [1, training_steps]")
        x = torch.randn(shape, device=visible.device, dtype=visible.dtype, generator=generator)
        times = torch.linspace(self.training_steps, 0, inference_steps + 1, dtype=torch.float64)
        times = times.round().to(torch.long).to(visible.device)
        nfe = 0
        for current, following in zip(times[:-1], times[1:]):
            t = current.expand(shape[0])
            epsilon = self.denoiser(
                x, t.to(x.dtype) / self.training_steps, visible, mask, parent,
            )
            nfe += 1
            alpha = self.alpha_bar[current].to(x.dtype)
            alpha_next = self.alpha_bar[following].to(x.dtype)
            clean = (x - (1 - alpha).sqrt() * epsilon) / alpha.sqrt().clamp_min(1.0e-8)
            x = alpha_next.sqrt() * clean + (1 - alpha_next).sqrt() * epsilon
        return x, nfe
