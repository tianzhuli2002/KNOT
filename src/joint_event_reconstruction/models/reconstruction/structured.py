"""Hierarchical visible assignments with exact within-role exchange symmetry."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from joint_event_reconstruction.topology.spec import TopologySpec


def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim != 2:
        raise ValueError(
            f"Static candidate indices must have shape [C,M], found {tuple(indices.shape)}"
        )
    candidates, members = indices.shape
    safe = indices.clamp_min(0)
    return values[:, safe.reshape(-1)].reshape(
        len(values), candidates, members, values.shape[-1]
    )


class SymmetricPairEncoder(nn.Module):
    """Exact permutation-invariant representation for a two-object role."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, members: torch.Tensor) -> torch.Tensor:
        if members.shape[2] != 2:
            raise ValueError(
                f"SymmetricPairEncoder requires two members, found {members.shape[2]}"
            )
        first, second = members[:, :, 0], members[:, :, 1]
        return self.projection(
            torch.cat(
                (first + second, (first - second).abs(), first * second), dim=-1
            )
        )


class SymmetricSetEncoder(nn.Module):
    """Permutation-invariant representation for repeated roles with >2 members."""

    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.member = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, members: torch.Tensor) -> torch.Tensor:
        if members.ndim != 4 or members.shape[2] < 3:
            raise ValueError(
                "SymmetricSetEncoder expects [B,C,M,D] with M>=3, "
                f"found {tuple(members.shape)}"
            )
        encoded = self.member(members)
        summary = torch.cat(
            (encoded.sum(dim=2), encoded.amax(dim=2)), dim=-1
        )
        return self.projection(summary)


class StructuredReconstruction(nn.Module):
    """Encode topology candidates and score a candidate-or-NULL assignment per node."""

    def __init__(self, spec: TopologySpec, d_model: int, dropout: float):
        super().__init__()
        self.spec = spec
        self.pair_encoders = nn.ModuleDict()
        self.set_encoders = nn.ModuleDict()
        self.node_encoders = nn.ModuleDict()
        self.scorers = nn.ModuleDict()
        self.null_embeddings = nn.ParameterDict()
        for node_name in spec.dag_order:
            node = spec.nodes[node_name]
            role_count = len(
                [role for role in node.leaf_roles if not role.invisible]
            ) + len(node.composite_roles)
            if role_count == 0:
                raise ValueError(
                    f"Visible candidate node {node_name} has no visible support"
                )
            for role in node.leaf_roles:
                if role.invisible or role.multiplicity == 1:
                    continue
                key = role.key.replace(".", "__")
                if role.multiplicity == 2:
                    self.pair_encoders[key] = SymmetricPairEncoder(d_model, dropout)
                else:
                    self.set_encoders[key] = SymmetricSetEncoder(d_model, dropout)
            self.node_encoders[node_name] = nn.Sequential(
                nn.Linear(role_count * d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            self.scorers[node_name] = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            self.null_embeddings[node_name] = nn.Parameter(torch.empty(d_model))
            nn.init.normal_(self.null_embeddings[node_name], std=0.02)

    def encode_candidates(
        self, encoded_tokens: torch.Tensor, structured: dict
    ) -> dict[str, torch.Tensor]:
        embeddings = {}
        for node_name in self.spec.dag_order:
            pool = structured["pools"][node_name]
            parts = []
            node = self.spec.nodes[node_name]
            for role in node.leaf_roles:
                if role.invisible:
                    continue
                gathered = _gather(
                    encoded_tokens, pool["leaf_indices"][role.key]
                )
                if role.multiplicity == 1:
                    parts.append(gathered[:, :, 0])
                elif role.multiplicity == 2:
                    parts.append(
                        self.pair_encoders[role.key.replace(".", "__")](gathered)
                    )
                else:
                    parts.append(
                        self.set_encoders[role.key.replace(".", "__")](gathered)
                    )
            for role in node.composite_roles:
                child = embeddings[role.child]
                child_index = pool["child_indices"][role.key]
                if child_index.ndim != 1:
                    raise ValueError(
                        "Static child indices must have shape [C], "
                        f"found {tuple(child_index.shape)}"
                    )
                parts.append(child[:, child_index.clamp_min(0)])
            candidate = self.node_encoders[node_name](torch.cat(parts, dim=-1))
            candidate = candidate * pool["candidate_mask"].unsqueeze(-1).to(
                candidate.dtype
            )
            embeddings[node_name] = candidate
        return embeddings

    def score_candidates(
        self,
        embeddings: dict[str, torch.Tensor],
        pooled: torch.Tensor,
        structured: dict,
        node_context: dict[str, torch.Tensor] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        logits = {}
        probabilities = {}
        for node_name in self.spec.dag_order:
            candidate = embeddings[node_name]
            pool = structured["pools"][node_name]
            scoring_context = (
                pooled if node_context is None else node_context[node_name]
            )
            context = scoring_context.unsqueeze(1).expand(
                -1, candidate.shape[1], -1
            )
            candidate_logits = self.scorers[node_name](
                torch.cat((candidate, context), dim=-1)
            ).squeeze(-1)
            candidate_logits = candidate_logits.masked_fill(
                ~pool["candidate_mask"], -torch.inf
            )
            null = self.null_embeddings[node_name].unsqueeze(0).expand(
                len(pooled), -1
            )
            null_logits = self.scorers[node_name](
                torch.cat((null, scoring_context), dim=-1)
            ).squeeze(-1)
            role_logits = torch.cat(
                (null_logits.unsqueeze(1), candidate_logits), dim=1
            )
            logits[node_name] = role_logits
            probabilities[node_name] = role_logits.softmax(dim=1)
        return logits, probabilities

    def assignment_loss(
        self, logits: dict[str, torch.Tensor], structured: dict
    ) -> dict:
        node_losses = {}
        for node_name in self.spec.dag_order:
            pool = structured["pools"][node_name]
            if (
                pool["reconstructable"] & (pool["truth_index"] < 0)
            ).any():
                raise RuntimeError(
                    f"Reconstructable node {node_name} has no truth candidate"
                )
            target = torch.where(
                pool["reconstructable"],
                pool["truth_index"] + 1,
                torch.zeros_like(pool["truth_index"]),
            )
            losses = F.cross_entropy(logits[node_name], target, reduction="none")
            support = structured.get(
                "objective_support", torch.ones_like(target, dtype=torch.bool)
            )
            if support.dtype != torch.bool or support.shape != target.shape:
                raise ValueError(f"Invalid assignment objective support for node {node_name}")
            node_losses[node_name] = (
                losses[support].mean() if support.any() else losses.sum() * 0
            )
        return {
            "assignment_loss": torch.stack(tuple(node_losses.values())).mean(),
            "node_losses": node_losses,
        }

    def forward(
        self,
        encoded_tokens: torch.Tensor,
        pooled: torch.Tensor,
        structured: dict,
        *,
        node_context: dict[str, torch.Tensor] | None = None,
        calculate_loss: bool = True,
    ) -> dict:
        embeddings = self.encode_candidates(encoded_tokens, structured)
        logits, probabilities = self.score_candidates(
            embeddings, pooled, structured, node_context
        )
        result = {
            "candidate_embeddings": embeddings,
            "assignment_logits": logits,
            "assignment_probabilities": probabilities,
        }
        if calculate_loss:
            result.update(self.assignment_loss(logits, structured))
        return result
