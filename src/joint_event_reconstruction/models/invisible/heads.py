"""Optional topology-sized point and parent four-vector heads."""

from __future__ import annotations

import torch
from torch import nn


class DirectInvisiblePointHead(nn.Module):
    """Predict one scaled Cartesian three-momentum per compiled invisible role."""

    def __init__(self, d_model: int, role_count: int):
        super().__init__()
        if role_count <= 0:
            raise ValueError("A direct invisible point head requires at least one role")
        self.role_count = int(role_count)
        self.output = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), 3 * self.role_count),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2:
            raise ValueError(f"Point-head context must be [B,D], found {tuple(pooled.shape)}")
        return self.output(pooled).reshape(len(pooled), self.role_count, 3)


class DirectParentFourVectorHead(nn.Module):
    """Predict physical E,px,py,pz for configured parent target roles.

    The head has no nominal mass-shell parameterisation.  It predicts an
    unconstrained spatial momentum and a positive invariant-mass offset, so
    ``E >= |p|`` is guaranteed while the parent mass remains event-dependent.
    """

    def __init__(self, d_model: int, parent_count: int):
        super().__init__()
        if parent_count <= 0:
            raise ValueError("A parent head requires at least one configured parent role")
        self.parent_count = int(parent_count)
        self.output = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), 4 * self.parent_count),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2:
            raise ValueError(f"Parent-head context must be [B,D], found {tuple(pooled.shape)}")
        raw = self.output(pooled).reshape(len(pooled), self.parent_count, 4)
        spatial = raw[..., 1:]
        mass_offset = torch.nn.functional.softplus(raw[..., :1])
        energy = (spatial.square().sum(dim=-1, keepdim=True) + mass_offset.square()).sqrt()
        return torch.cat((energy, spatial), dim=-1)
