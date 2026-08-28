"""Topology-shaped inference state conditioned on fixed event tokens."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.models.invisible.denoiser import noise_embedding
from joint_event_reconstruction.topology.spec import TopologySpec


class TopologyFlowReasoner(nn.Module):
    """One shared reasoning block; only topology tokens persist between flow steps."""

    def __init__(self, spec: TopologySpec, d_model: int, n_heads: int, block_layers: int, d_feedforward: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.spec = spec
        self.d_model = d_model
        self.topology_tokens = nn.Parameter(torch.empty(len(spec.recurrent_tokens), d_model))
        nn.init.normal_(self.topology_tokens, std=0.02)
        self.time_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.invisible_projection = nn.Linear(3, d_model)
        self.invisible_role_embedding = nn.Parameter(torch.empty(len(spec.invisible_roles), d_model))
        nn.init.normal_(self.invisible_role_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_feedforward, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.shared_block = nn.TransformerEncoder(layer, num_layers=block_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.topology_tokens.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)

    def _mask(self, visible_padding_mask: torch.Tensor) -> torch.Tensor:
        fixed = torch.zeros((len(visible_padding_mask), 1), dtype=torch.bool, device=visible_padding_mask.device)
        topology = torch.zeros((len(visible_padding_mask), len(self.spec.recurrent_tokens)), dtype=torch.bool, device=visible_padding_mask.device)
        return torch.cat((fixed, visible_padding_mask, topology), dim=1)

    def step(self, visible: dict, assignment_context: dict[str, torch.Tensor], invisible_state: torch.Tensor | None, time: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        batch = len(visible["tokens"])
        if time.shape != (batch,):
            raise ValueError(f"Flow time must have shape [B], found {tuple(time.shape)}")
        topology = self.initial_state(batch, device=visible["tokens"].device, dtype=visible["tokens"].dtype) if state is None else state
        if topology.shape != (batch, len(self.spec.recurrent_tokens), self.d_model):
            raise ValueError("Persistent topology state has incompatible shape")
        time_context = self.time_projection(noise_embedding(time, self.d_model)).unsqueeze(1)
        dynamic = topology + time_context
        for index, name in enumerate(self.spec.recurrent_tokens):
            dynamic[:, index] = dynamic[:, index] + assignment_context[name]
        if invisible_state is not None:
            if invisible_state.shape != (batch, len(self.spec.invisible_roles), 3):
                raise ValueError(f"Invisible flow state must be [B,{len(self.spec.invisible_roles)},3], found {tuple(invisible_state.shape)}")
            for role_index, role in enumerate(self.spec.invisible_roles):
                node_index = self.spec.recurrent_tokens.index(role.node)
                invisible = self.invisible_projection(invisible_state[:, role_index]) + self.invisible_role_embedding[role_index]
                dynamic[:, node_index] = dynamic[:, node_index] + invisible
        sequence = torch.cat((visible["pooled"].unsqueeze(1), visible["tokens"], dynamic), dim=1)
        encoded = self.output_norm(self.shared_block(sequence, src_key_padding_mask=self._mask(visible["padding_mask"])))
        return encoded[:, 1 + visible["tokens"].shape[1]:]

    def as_dict(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: state[:, index] for index, name in enumerate(self.spec.recurrent_tokens)}
