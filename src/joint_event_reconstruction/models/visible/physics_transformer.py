"""Physics-aware non-equivariant Transformer control."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from joint_event_reconstruction.models.visible.physical_features import (
    physical_encoder_input,
)


def _pair_features(fourmomenta: torch.Tensor) -> torch.Tensor:
    _, px, py, pz = fourmomenta.unbind(dim=-1)
    pt = torch.sqrt(px.square() + py.square()).clamp_min(1.0e-8)
    eta = torch.asinh(pz / pt)
    phi = torch.atan2(py, px)
    delta_eta = eta[:, :, None] - eta[:, None, :]
    delta_phi = phi[:, :, None] - phi[:, None, :]
    wrapped = torch.atan2(torch.sin(delta_phi), torch.cos(delta_phi))
    delta_r = torch.sqrt(delta_eta.square() + wrapped.square())
    minkowski = (
        fourmomenta[:, :, None, 0] * fourmomenta[:, None, :, 0]
        - (fourmomenta[:, :, None, 1:] * fourmomenta[:, None, :, 1:]).sum(-1)
    )
    pair = fourmomenta[:, :, None] + fourmomenta[:, None, :]
    mass2 = pair[..., 0].square() - pair[..., 1:].square().sum(-1)
    signed_log_mass2 = torch.sign(mass2) * torch.log1p(mass2.abs())
    return torch.stack(
        (
            delta_eta,
            torch.sin(delta_phi),
            torch.cos(delta_phi),
            torch.log(delta_r + 1.0e-6),
            minkowski,
            signed_log_mass2,
        ),
        dim=-1,
    )


class _PhysicsAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_feedforward, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
        )
        self.dropout = dropout

    def forward(self, x, bias, padding):
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            batch, tokens, 3, self.n_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        mask = bias.masked_fill(padding[:, None, None, :], -torch.inf)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2).reshape(batch, tokens, width)
        x = x + self.out(attended)
        x = x + self.ff(self.norm2(x))
        return x.masked_fill(padding.unsqueeze(-1), 0.0)


class RelationalTransformerVisibleEncoder(nn.Module):
    def __init__(self, config, schema, n_object_types: int):
        super().__init__()
        d_model = int(config.d_model)
        n_heads = int(config.n_heads)
        if d_model % n_heads:
            raise ValueError("Physics Transformer width must be divisible by heads")
        self.config = config
        self.schema = schema
        scalar_count = len(config.inputs.object_features) + len(config.inputs.event_features)
        self.object_projection = nn.Linear(4 + scalar_count, d_model)
        self.type_embedding = nn.Embedding(n_object_types, d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        self.pair_bias = None
        if bool(config.relations.enabled):
            self.pair_bias = nn.Sequential(
                nn.Linear(6, int(config.relations.hidden_dim)),
                nn.GELU(),
                nn.Linear(int(config.relations.hidden_dim), n_heads),
            )
        self.blocks = nn.ModuleList(
            _PhysicsAttentionBlock(
                d_model,
                n_heads,
                int(config.d_feedforward),
                float(config.dropout),
            )
            for _ in range(int(config.layers))
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, batch: dict) -> dict:
        physical = physical_encoder_input(batch, self.schema, self.config)
        objects = self.object_projection(
            torch.cat((physical.fourmomenta, physical.scalars), dim=-1)
        ) + self.type_embedding(physical.type_ids)
        cls = self.cls_token.expand(len(objects), -1, -1)
        x = torch.cat((cls, objects), dim=1)
        cls_padding = torch.zeros(
            (len(objects), 1), dtype=torch.bool, device=objects.device
        )
        padding = torch.cat((cls_padding, physical.padding_mask), dim=1)
        bias = objects.new_zeros(
            len(objects), int(self.config.n_heads), x.shape[1], x.shape[1]
        )
        if self.pair_bias is not None:
            object_bias = self.pair_bias(_pair_features(physical.fourmomenta)).permute(
                0, 3, 1, 2
            )
            bias[:, :, 1:, 1:] = object_bias
        for block in self.blocks:
            x = block(x, bias, padding)
        encoded = self.output_norm(x)
        return {
            "pooled": encoded[:, 0],
            "tokens": encoded[:, 1:].masked_fill(
                physical.padding_mask.unsqueeze(-1), 0.0
            ),
            "padding_mask": physical.padding_mask,
        }
