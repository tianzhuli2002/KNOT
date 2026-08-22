"""Common Transformer-native denoiser used by VP diffusion and EDM."""

from __future__ import annotations

import math

import torch
from torch import nn


def noise_embedding(level: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        torch.arange(half, device=level.device, dtype=level.dtype)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = level.unsqueeze(1) * frequencies.unsqueeze(0)
    result = torch.cat((angles.sin(), angles.cos()), dim=1)
    if result.shape[1] < width:
        result = torch.cat((result, result.new_zeros((len(result), 1))), dim=1)
    return result


class InvisibleDenoiserBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_feedforward: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.cross_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ff_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.self_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
        )
        self.modulation = nn.Linear(d_model, 6 * d_model)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self, state: torch.Tensor, visible: torch.Tensor, parent: torch.Tensor,
        visible_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        shift_self, scale_self, gate_self, shift_ff, scale_ff, gate_ff = self.modulation(parent).chunk(6, dim=-1)
        normalised = self.self_norm(state) * (1 + scale_self) + shift_self
        update = self.self_attention(normalised, normalised, normalised, need_weights=False)[0]
        state = state + torch.sigmoid(gate_self) * update
        normalised = self.cross_norm(state)
        state = state + self.cross_attention(
            normalised, visible, visible, key_padding_mask=visible_padding_mask, need_weights=False,
        )[0]
        normalised = self.ff_norm(state) * (1 + scale_ff) + shift_ff
        return state + torch.sigmoid(gate_ff) * self.feedforward(normalised)


class InvisibleDenoiser(nn.Module):
    def __init__(
        self, role_count: int, *, visible_dim: int, parent_dim: int, d_model: int,
        n_heads: int, layers: int, d_feedforward: int, dropout: float,
        zero_init_output: bool = True,
    ):
        super().__init__()
        if role_count <= 0:
            raise ValueError("InvisibleDenoiser requires at least one invisible role")
        self.role_count = role_count
        self.input_projection = nn.Linear(3, d_model)
        self.role_embedding = nn.Parameter(torch.empty(role_count, d_model))
        nn.init.normal_(self.role_embedding, std=0.02)
        self.noise_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.visible_projection = nn.Linear(visible_dim, d_model)
        self.parent_projection = nn.Linear(parent_dim, d_model)
        self.blocks = nn.ModuleList([
            InvisibleDenoiserBlock(d_model, n_heads, d_feedforward, dropout) for _ in range(layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, 3)
        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def forward(
        self, noisy: torch.Tensor, noise_level: torch.Tensor, visible: torch.Tensor,
        visible_padding_mask: torch.Tensor, parent_context: torch.Tensor,
    ) -> torch.Tensor:
        if noisy.ndim != 3 or noisy.shape[1:] != (self.role_count, 3):
            raise ValueError(f"Noisy invisible state must be [B,{self.role_count},3], found {tuple(noisy.shape)}")
        if parent_context.shape[:2] != noisy.shape[:2]:
            raise ValueError("Parent context and invisible role axes do not match")
        if noise_level.shape != (len(noisy),):
            raise ValueError(f"Noise level must have shape [B], found {tuple(noise_level.shape)}")
        noise = self.noise_projection(noise_embedding(noise_level, self.role_embedding.shape[1]))
        visible_state = self.visible_projection(visible)
        parent = self.parent_projection(parent_context)
        # Direct parent addition keeps assignment conditioning effective from
        # the first optimisation step; the zero-initialised adaptive modulation
        # then learns a finer role-specific conditioning on top.
        state = (
            self.input_projection(noisy) + self.role_embedding.unsqueeze(0)
            + noise.unsqueeze(1) + parent
        )
        for block in self.blocks:
            state = block(state, visible_state, parent, visible_padding_mask)
        return self.output_projection(self.output_norm(state))
