"""Detached posterior-moment feedback into topology tokens."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.topology.spec import TopologySpec


class InvisiblePosteriorFeedback(nn.Module):
    def __init__(self, spec: TopologySpec, d_model: int, gate_initial_bias: float = -2.0):
        super().__init__()
        self.spec = spec
        self.projections = nn.ModuleDict({name: nn.Linear(6, d_model) for name in spec.recurrent_tokens})
        self.gates = nn.ModuleDict({name: nn.Linear(2 * d_model, d_model) for name in spec.recurrent_tokens})
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, gate_initial_bias)

    def forward(
        self, node_tokens: dict[str, torch.Tensor], samples: torch.Tensor, *, detach: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if detach:
            samples = samples.detach()
        mean = samples.mean(dim=1)
        variance = samples.var(dim=1, unbiased=False)
        accumulated: dict[str, list[torch.Tensor]] = {}
        for role_index, role in enumerate(self.spec.invisible_roles):
            accumulated.setdefault(role.node, []).append(
                torch.cat((mean[:, role_index], variance[:, role_index]), dim=1)
            )
        updated = dict(node_tokens)
        gates = {}
        for node, moments in accumulated.items():
            moment = torch.stack(moments, dim=0).mean(dim=0)
            delta = self.projections[node](moment)
            gate = torch.sigmoid(self.gates[node](torch.cat((node_tokens[node], delta), dim=1)))
            updated[node] = node_tokens[node] + gate * delta
            gates[node] = gate.mean(dim=1)
        return updated, gates
