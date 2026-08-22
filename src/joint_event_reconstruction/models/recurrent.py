"""Weight-shared global reasoning over visible and topology tokens."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.topology.spec import TopologySpec


class RecurrentReasoner(nn.Module):
    def __init__(
        self, spec: TopologySpec, d_model: int, n_heads: int, block_layers: int,
        d_feedforward: int, dropout: float, *, fixed_input_reinjection: bool,
    ):
        super().__init__()
        self.spec = spec
        self.fixed_input_reinjection = fixed_input_reinjection
        self.topology_tokens = nn.Parameter(torch.empty(len(spec.recurrent_tokens), d_model))
        nn.init.normal_(self.topology_tokens, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.shared_block = nn.TransformerEncoder(layer, num_layers=block_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def initialise(self, pooled: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
        topology = self.topology_tokens.unsqueeze(0).expand(len(visible), -1, -1)
        return torch.cat((pooled.unsqueeze(1), visible, topology), dim=1)

    def mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        unmasked = torch.zeros(
            (len(padding_mask), 1 + len(self.spec.recurrent_tokens)),
            dtype=torch.bool, device=padding_mask.device,
        )
        return torch.cat((unmasked[:, :1], padding_mask, unmasked[:, 1:]), dim=1)

    def step(self, state: torch.Tensor, fixed: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.fixed_input_reinjection:
            visible_end = fixed.shape[1]
            reinjected = torch.cat(
                (state[:, :visible_end] + fixed, state[:, visible_end:]), dim=1,
            )
        else:
            reinjected = state
        return self.output_norm(self.shared_block(reinjected, src_key_padding_mask=mask))

    def split(self, state: torch.Tensor, visible_count: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        pooled = state[:, 0]
        visible = state[:, 1:1 + visible_count]
        topology = state[:, 1 + visible_count:]
        node_context = {
            name: topology[:, index] for index, name in enumerate(self.spec.recurrent_tokens)
        }
        return pooled, visible, node_context

    def replace_topology(self, state: torch.Tensor, visible_count: int, node_tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        ordered = torch.stack([node_tokens[name] for name in self.spec.recurrent_tokens], dim=1)
        return torch.cat((state[:, :1 + visible_count], ordered), dim=1)
