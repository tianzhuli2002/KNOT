"""The preserved standard-Transformer visible-event baseline."""

from __future__ import annotations

import torch
from torch import nn


class TransformerVisibleEncoder(nn.Module):
    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_feedforward: int,
        dropout: float,
        n_object_types: int,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.input_projection = nn.Linear(n_features, d_model)
        self.type_embedding = nn.Embedding(n_object_types, d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, batch: dict) -> dict:
        tokens = batch["tokens"]
        type_ids = batch["type_ids"]
        padding_mask = batch["padding_mask"]
        x = self.input_projection(tokens) + self.type_embedding(type_ids)
        cls = self.cls_token.expand(len(tokens), -1, -1)
        x = torch.cat((cls, x), dim=1)
        cls_mask = torch.zeros(
            (len(tokens), 1), dtype=torch.bool, device=padding_mask.device
        )
        encoded = self.output_norm(
            self.encoder(
                x,
                src_key_padding_mask=torch.cat((cls_mask, padding_mask), dim=1),
            )
        )
        return {
            "pooled": encoded[:, 0],
            "tokens": encoded[:, 1:],
            "padding_mask": padding_mask,
        }
