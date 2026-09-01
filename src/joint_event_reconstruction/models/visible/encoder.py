"""Explicit construction of the maintained visible-event backends."""

from __future__ import annotations

from joint_event_reconstruction.models.visible.transformer import (
    TransformerVisibleEncoder,
)


def build_visible_encoder(config, *, n_features: int, n_object_types: int):
    backend = str(config.type)
    if backend == "transformer":
        return TransformerVisibleEncoder(
            n_features=n_features,
            d_model=int(config.d_model),
            n_heads=int(config.n_heads),
            n_layers=int(config.layers),
            d_feedforward=int(config.d_feedforward),
            dropout=float(config.dropout),
            n_object_types=n_object_types,
        )
    raise ValueError(f"Unknown visible encoder type {backend!r}")
