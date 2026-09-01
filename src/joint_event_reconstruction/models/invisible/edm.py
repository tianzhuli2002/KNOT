"""EDM preconditioning, weighted training and deterministic Heun sampling."""

from __future__ import annotations

import torch
from torch import nn


class EDMDiffusion(nn.Module):
    def __init__(
        self, denoiser: nn.Module, *, sigma_data: float = 1.0, p_mean: float = -0.5,
        p_std: float = 1.2, sigma_min: float = 0.004, sigma_max: float = 160.0,
        rho: float = 7.0, active_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        if not 0 < sigma_min < sigma_max or sigma_data <= 0 or p_std <= 0 or rho <= 0:
            raise ValueError("EDM sigma and distribution parameters must be positive and ordered")
        self.denoiser = denoiser
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        if active_mask is None:
            active_mask = torch.ones(denoiser.state_token_count, 3, dtype=torch.bool)
        self.register_buffer("active_mask", torch.as_tensor(active_mask, dtype=torch.bool))

    def coefficients(self, sigma: torch.Tensor) -> tuple[torch.Tensor, ...]:
        data = self.sigma_data
        denominator = sigma.square() + data ** 2
        return data ** 2 / denominator, sigma * data / denominator.sqrt(), denominator.rsqrt(), sigma.log() / 4

    def denoise(
        self, noisy: torch.Tensor, sigma: torch.Tensor, visible: torch.Tensor,
        mask: torch.Tensor, parent: torch.Tensor,
    ) -> torch.Tensor:
        if (sigma <= 0).any():
            raise ValueError("EDM denoising requires strictly positive sigma")
        c_skip, c_out, c_in, c_noise = self.coefficients(sigma)
        expand = (len(sigma),) + (1,) * (noisy.ndim - 1)
        active = self.active_mask.to(dtype=noisy.dtype)
        network = self.denoiser(
            c_in.view(expand) * noisy, c_noise, visible, mask, parent,
        ) * active
        return (c_skip.view(expand) * noisy + c_out.view(expand) * network) * active

    def loss(self, target: torch.Tensor, visible: torch.Tensor, mask: torch.Tensor, parent: torch.Tensor) -> dict:
        sigma = (torch.randn(len(target), device=target.device, dtype=target.dtype) * self.p_std + self.p_mean).exp()
        active = self.active_mask.to(dtype=target.dtype)
        noise = torch.randn_like(target) * active
        noisy = target + sigma.view(len(target), 1, 1) * noise
        denoised = self.denoise(noisy, sigma, visible, mask, parent)
        weight = (sigma.square() + self.sigma_data ** 2) / (sigma * self.sigma_data).square()
        per_event = (denoised - target).square().flatten(1).sum(dim=1) / active.sum()
        return {"loss": (weight * per_event).mean(), "sigma": sigma, "denoised": denoised}

    def schedule(self, nfe: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if nfe < 3 or nfe % 2 == 0:
            raise ValueError("Deterministic EDM Heun NFE must be an odd integer of at least 3")
        positive_points = (nfe + 1) // 2
        ramp = torch.linspace(0, 1, positive_points, dtype=torch.float64, device=device)
        high = self.sigma_max ** (1 / self.rho)
        low = self.sigma_min ** (1 / self.rho)
        positive = (high + ramp * (low - high)).pow(self.rho)
        return torch.cat((positive, positive.new_zeros(1))).to(dtype)

    @torch.no_grad()
    def sample(
        self, shape: tuple[int, int, int], visible: torch.Tensor, mask: torch.Tensor,
        parent: torch.Tensor, *, nfe: int, generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, int]:
        sigmas = self.schedule(nfe, device=visible.device, dtype=visible.dtype)
        active = self.active_mask.to(device=visible.device, dtype=visible.dtype)
        x = (
            torch.randn(shape, device=visible.device, dtype=visible.dtype, generator=generator)
            * sigmas[0] * active
        )
        evaluations = 0
        for index in range(len(sigmas) - 1):
            sigma, following = sigmas[index], sigmas[index + 1]
            batch_sigma = sigma.expand(shape[0])
            denoised = self.denoise(x, batch_sigma, visible, mask, parent)
            evaluations += 1
            derivative = (x - denoised) / sigma
            proposed = x + (following - sigma) * derivative
            if following > 0:
                next_denoised = self.denoise(
                    proposed, following.expand(shape[0]), visible, mask, parent,
                )
                evaluations += 1
                next_derivative = (proposed - next_denoised) / following
                x = x + (following - sigma) * (derivative + next_derivative) / 2
            else:
                x = proposed
            x = x * active
        if evaluations != nfe:
            raise RuntimeError(f"EDM sampler requested {nfe} NFE but executed {evaluations}")
        return x, evaluations
