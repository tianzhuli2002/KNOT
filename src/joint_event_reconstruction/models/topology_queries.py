"""Small topology-defined latent query block for the shared event context.

The maintained L-GATr encoder currently exposes scalar channels to the JER
heads.  This module therefore operates on those scalar channels only.  It is
deliberately not advertised as a Lorentz-equivariant query transformer: the
physical four-vector head below has its own explicitly documented output
parameterisation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def _normalise_role(entry: Mapping[str, object]) -> dict[str, object]:
    name = str(entry.get("name", "")).strip()
    if not name:
        raise ValueError("Every topology query requires a non-empty name")
    node = entry.get("node")
    node = None if node is None else str(node)
    kind = str(entry.get("kind", "role"))
    if not kind:
        raise ValueError(f"Topology query {name!r} has an empty kind")
    related = tuple(str(value) for value in entry.get("related", ()))
    return {"name": name, "node": node, "kind": kind, "related": related}


class TopologyQueryBlock(nn.Module):
    """One-pass event-to-query and query-to-query scalar processing.

    ``query_roles`` is a list of resolved topology metadata dictionaries.  A
    role query is initialised only from learned scalar role/type embeddings and
    the encoded measured event.  No truth tensor is accepted by this module.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        query_roles: Sequence[Mapping[str, object]],
        interaction: str = "full",
        layers: int = 1,
        dropout: float = 0.0,
        relation_edges: Sequence[tuple[int, int]] = (),
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError("Topology-query d_model must be positive and divisible by n_heads")
        if interaction not in {"full", "topology", "topology_plus_global"}:
            raise ValueError(f"Unknown topology-query interaction policy {interaction!r}")
        if layers <= 0:
            raise ValueError("Topology-query layers must be positive")
        normalised = tuple(_normalise_role(role) for role in query_roles)
        if not normalised:
            raise ValueError("Enabled topology queries require at least one role")
        names = tuple(str(role["name"]) for role in normalised)
        if len(set(names)) != len(names):
            raise ValueError(f"Topology query names must be unique: {names}")
        kinds = tuple(sorted({str(role["kind"]) for role in normalised}))
        kind_index = {kind: index for index, kind in enumerate(kinds)}

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.query_roles = normalised
        self.query_names = names
        self.interaction = interaction
        self.layers = int(layers)
        self.role_embedding = nn.Parameter(torch.empty(len(names), d_model))
        self.kind_embedding = nn.Embedding(len(kinds), d_model)
        self.event_query = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.query_attention = nn.ModuleList(
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(layers)
        )
        self.query_norm = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(layers + 1))
        self.query_ffn = nn.ModuleList(
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
            )
            for _ in range(layers)
        )
        self.summary = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.event_context = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model))
        self.register_buffer(
            "kind_indices",
            torch.tensor([kind_index[str(role["kind"])] for role in normalised], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "query_attention_mask",
            self._build_attention_mask(len(names), interaction, relation_edges, normalised),
            persistent=False,
        )
        nn.init.normal_(self.role_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.kind_embedding.weight, mean=0.0, std=0.02)

    @staticmethod
    def _build_attention_mask(
        count: int,
        interaction: str,
        relation_edges: Sequence[tuple[int, int]],
        roles: Sequence[Mapping[str, object]],
    ) -> torch.Tensor:
        # True means blocked for torch MultiheadAttention.
        allowed = torch.eye(count, dtype=torch.bool)
        if interaction == "full":
            allowed[:] = True
        else:
            for left, right in relation_edges:
                if not (0 <= int(left) < count and 0 <= int(right) < count):
                    raise ValueError("Topology-query relation edge is outside the query list")
                allowed[int(left), int(right)] = True
                allowed[int(right), int(left)] = True
            # Queries declared for the same compiled node share local
            # information even when no explicit edge was supplied.
            for left, left_role in enumerate(roles):
                for right, right_role in enumerate(roles):
                    if left_role.get("node") is not None and left_role.get("node") == right_role.get("node"):
                        allowed[left, right] = True
                    if str(right_role["name"]) in tuple(left_role.get("related", ())):
                        allowed[left, right] = True
            if interaction == "topology_plus_global":
                global_indices = [
                    index for index, role in enumerate(roles)
                    if str(role.get("kind")) == "global"
                ]
                for index in global_indices:
                    allowed[index, :] = True
                    allowed[:, index] = True
            # A declared policy must never create a query with no admissible
            # key.  Self attention is always retained above.
            if not allowed.any(dim=1).all():
                raise ValueError("Topology-query attention policy produced an isolated query")
        return ~allowed

    def forward(
        self,
        pooled: torch.Tensor,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[str, ...]]:
        if pooled.ndim != 2 or tokens.ndim != 3 or padding_mask.ndim != 2:
            raise ValueError("Topology queries expect pooled [B,D], tokens [B,N,D], padding_mask [B,N]")
        if pooled.shape[0] != tokens.shape[0] or padding_mask.shape[:2] != tokens.shape[:2]:
            raise ValueError("Topology-query event and token batch shapes do not agree")
        if pooled.shape[1] != self.d_model or tokens.shape[2] != self.d_model:
            raise ValueError("Topology-query feature width does not match d_model")
        event = torch.cat((pooled.unsqueeze(1), tokens), dim=1)
        event_padding = torch.cat(
            (torch.zeros((len(pooled), 1), dtype=torch.bool, device=padding_mask.device), padding_mask),
            dim=1,
        )
        query = self.role_embedding.unsqueeze(0).expand(len(pooled), -1, -1)
        query = query + self.kind_embedding(self.kind_indices).unsqueeze(0)
        query_delta, _ = self.event_query(
            query, event, event, key_padding_mask=event_padding
        )
        query = self.query_norm[0](query + query_delta)
        for attention, norm, ffn in zip(
            self.query_attention, self.query_norm[1:], self.query_ffn, strict=True
        ):
            delta, _ = attention(query, query, query, attn_mask=self.query_attention_mask)
            query = norm(query + delta)
            query = query + ffn(query)
        summary = query.mean(dim=1)
        context = pooled + self.event_context(summary)
        return {
            "states": query,
            "summary": self.summary(summary),
            "context": context,
            "roles": self.query_names,
        }


__all__ = ["TopologyQueryBlock"]
