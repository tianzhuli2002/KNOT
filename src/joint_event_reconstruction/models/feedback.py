"""Differentiable soft assignment feedback for persistent topology tokens."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.topology.spec import TopologySpec


class SoftReconstructionFeedback(nn.Module):
    def __init__(self, spec: TopologySpec, d_model: int, gate_initial_bias: float = -2.0):
        super().__init__()
        width = 2 * d_model + 2
        self.spec = spec
        self.updates = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(width, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            for name in spec.recurrent_tokens
        })
        self.gates = nn.ModuleDict({name: nn.Linear(width, d_model) for name in spec.recurrent_tokens})
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, gate_initial_bias)

    def forward(
        self,
        node_tokens: dict[str, torch.Tensor],
        candidate_embeddings: dict[str, torch.Tensor],
        probabilities: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
        updated = {}
        diagnostics = {}
        for name in self.spec.recurrent_tokens:
            probability = probabilities[name]
            non_null = probability[:, 1:]
            summary = torch.sum(non_null.unsqueeze(-1) * candidate_embeddings[name], dim=1)
            null = probability[:, :1]
            entropy = -(probability * probability.clamp_min(1.0e-12).log()).sum(dim=1, keepdim=True)
            features = torch.cat((node_tokens[name], summary, null, entropy), dim=1)
            gate = torch.sigmoid(self.gates[name](features))
            updated[name] = node_tokens[name] + gate * self.updates[name](features)
            top_two = probability.topk(min(2, probability.shape[1]), dim=1).values
            margin = top_two[:, 0] - top_two[:, 1] if top_two.shape[1] == 2 else top_two[:, 0]
            diagnostics[name] = {
                "null_probability": null.squeeze(1), "entropy": entropy.squeeze(1),
                "margin": margin, "gate_mean": gate.mean(dim=1),
            }
        return updated, diagnostics
