"""Composition root for structured visible-event reconstruction."""

from __future__ import annotations

from torch import nn

from joint_event_reconstruction.models.reconstruction.structured import StructuredReconstruction
from joint_event_reconstruction.models.visible.encoder import VisibleEventEncoder
from joint_event_reconstruction.topology.spec import TopologySpec


class StructuredEventModel(nn.Module):
    def __init__(
        self, spec: TopologySpec, *, n_features: int, n_object_types: int,
        d_model: int, n_heads: int, n_layers: int, d_feedforward: int, dropout: float,
    ):
        super().__init__()
        self.visible = VisibleEventEncoder(
            n_features, d_model, n_heads, n_layers, d_feedforward, dropout, n_object_types,
        )
        self.reconstruction = StructuredReconstruction(spec, d_model, dropout)

    def forward(self, batch: dict) -> dict:
        visible = self.visible(batch["tokens"], batch["type_ids"], batch["padding_mask"])
        reconstruction = self.reconstruction(visible["tokens"], visible["pooled"], batch["structured"])
        return {"visible": visible, **reconstruction, "loss": reconstruction["reconstruction_loss"]}
