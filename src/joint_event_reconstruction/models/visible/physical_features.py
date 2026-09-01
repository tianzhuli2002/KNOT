"""Token-aligned physical inputs shared by physics-aware visible encoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PhysicalEncoderInput:
    fourmomenta: torch.Tensor
    scalars: torch.Tensor
    type_ids: torch.Tensor
    padding_mask: torch.Tensor


def physical_encoder_input(batch: dict, schema, config) -> PhysicalEncoderInput:
    required = {"tokens", "type_ids", "padding_mask", "physics_vectors"}
    missing = sorted(required - set(batch))
    if missing:
        raise KeyError(f"Physical visible encoder is missing batch fields {missing}")
    tokens = batch["tokens"]
    type_ids = batch["type_ids"]
    padding = batch["padding_mask"]
    vectors = batch["physics_vectors"]
    if vectors.shape != tokens.shape[:2] + (4,):
        raise ValueError(
            "physics_vectors must be token-aligned [B,N,4] in (E,px,py,pz)"
        )
    if type_ids.shape != tokens.shape[:2] or padding.shape != tokens.shape[:2]:
        raise ValueError("type_ids and padding_mask must align with visible tokens")
    if not torch.isfinite(vectors).all():
        raise ValueError("physics_vectors contain non-finite values")
    names = tuple(str(name) for name in config.scalar_features)
    scalars = tokens.new_zeros(tokens.shape[:2] + (len(names),))
    for scalar_index, name in enumerate(names):
        for source, plan in schema.sources.items():
            feature = next(
                (item for item in plan.features if item.variable == name), None
            )
            if feature is None:
                continue
            if len(feature.indices) != 1:
                raise ValueError(f"Physical scalar feature {name!r} is not scalar")
            source_mask = type_ids == int(schema.source_type_ids[source])
            scalars[..., scalar_index] = torch.where(
                source_mask, tokens[..., int(feature.indices[0])], scalars[..., scalar_index]
            )
    scalars = scalars.masked_fill(padding.unsqueeze(-1), 0.0)
    scale = float(config.momentum_scale_gev)
    if not scale > 0:
        raise ValueError("momentum_scale_gev must be positive")
    return PhysicalEncoderInput(
        fourmomenta=(vectors / scale).masked_fill(padding.unsqueeze(-1), 0.0),
        scalars=scalars,
        type_ids=type_ids,
        padding_mask=padding,
    )
