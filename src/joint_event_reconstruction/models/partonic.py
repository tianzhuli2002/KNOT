"""Topology-generic direct truth-level four-vector inference head."""

from __future__ import annotations

import torch
from torch import nn


class PartonicFourVectorHead(nn.Module):
    """Predict scaled ``px, py, pz, log1p(invariant_mass)`` features.

    The training scaler supplies the physical units.  Its inverse uses
    ``expm1`` for a positive event-dependent mass before constructing energy;
    no nominal mass or reconstructed constituent is used.
    """

    def __init__(self, d_model: int, roles: tuple[str, ...]) -> None:
        super().__init__()
        if not roles or len(set(roles)) != len(roles):
            raise ValueError("Partonic head requires unique configured target roles")
        self.roles = tuple(str(role) for role in roles)
        self.output = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), 4),
        )

    def forward(self, query_states: torch.Tensor, query_names: tuple[str, ...]) -> torch.Tensor:
        if query_states.ndim != 3:
            raise ValueError(f"Partonic query states must be [B,Q,D], found {tuple(query_states.shape)}")
        locations = []
        for role in self.roles:
            if role not in query_names:
                raise ValueError(f"Partonic target role {role!r} is absent from topology queries")
            locations.append(query_names.index(role))
        selected = query_states[:, locations]
        return self.output(selected)


def partonic_features(four_vectors: torch.Tensor) -> torch.Tensor:
    """Convert physical ``(E,px,py,pz)`` to ``(px,py,pz,m)`` features."""
    if four_vectors.ndim != 3 or four_vectors.shape[-1] != 4:
        raise ValueError("Partonic truth must have shape [B,P,4] in E_px_py_pz order")
    energy = four_vectors[..., 0]
    spatial = four_vectors[..., 1:]
    mass_squared = energy.square() - spatial.square().sum(dim=-1)
    if not torch.isfinite(four_vectors).all() or (energy < 0).any() or (mass_squared < -1.0e-3).any():
        raise ValueError("Partonic truth contains an invalid physical four-vector")
    mass = mass_squared.clamp_min(0.0).sqrt()
    return torch.cat((spatial, torch.log1p(mass).unsqueeze(-1)), dim=-1)


def four_vectors_from_partonic_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3 or features.shape[-1] != 4:
        raise ValueError("Partonic features must have shape [B,P,4]")
    spatial = features[..., :3]
    mass = torch.expm1(features[..., 3]).clamp_min(0.0)
    energy = torch.sqrt(spatial.square().sum(dim=-1) + mass.square())
    return torch.cat((energy.unsqueeze(-1), spatial), dim=-1)


__all__ = [
    "PartonicFourVectorHead",
    "four_vectors_from_partonic_features",
    "partonic_features",
]
