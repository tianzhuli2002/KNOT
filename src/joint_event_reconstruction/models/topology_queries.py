"""Small topology-defined latent query block for the shared event context.

The maintained L-GATr encoder currently exposes scalar channels to the JER
heads.  This module therefore operates on those scalar channels only.  It is
deliberately not advertised as a Lorentz-equivariant query transformer: the
physical four-vector head below has its own explicitly documented output
parameterisation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TopologyQueryState:
    """Stable output contract for the one shared topology-query pass."""

    query_states: torch.Tensor
    global_context: torch.Tensor
    query_summary: torch.Tensor
    node_query_indices: tuple[tuple[str, int], ...]
    node_contexts: dict[str, torch.Tensor]
    invisible_role_names: tuple[str, ...]
    invisible_query_indices: tuple[int, ...]
    invisible_contexts: torch.Tensor
    parent_role_names: tuple[str, ...]
    parent_query_indices: tuple[int, ...]
    parent_contexts: torch.Tensor
    role_names: tuple[str, ...]
    topology_identity: str

    # Keep the old string-key access for the existing direct-head experiment.
    def __getitem__(self, name: str):
        aliases = {
            "states": self.query_states,
            "summary": self.query_summary,
            "context": self.global_context,
            "roles": self.role_names,
        }
        if name in aliases:
            return aliases[name]
        raise KeyError(name)


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
        required_node_names: Sequence[str] = (),
        invisible_role_names: Sequence[str] = (),
        invisible_role_nodes: Sequence[str] = (),
        topology_identity: str = "",
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
        node_entries = tuple(
            (str(role["node"]), index)
            for index, role in enumerate(normalised)
            if role["node"] is not None
        )
        node_names = tuple(name for name, _ in node_entries)
        if len(set(node_names)) != len(node_names):
            raise ValueError(f"Each compiled node may map to only one query: {node_entries}")
        missing_nodes = sorted(set(str(value) for value in required_node_names) - set(node_names))
        if missing_nodes:
            raise ValueError(
                "Topology-query mode requires one query for every compiled node; "
                f"missing={missing_nodes}"
            )
        invisible_names = tuple(str(value) for value in invisible_role_names)
        invisible_nodes = tuple(str(value) for value in invisible_role_nodes)
        if len(invisible_names) != len(invisible_nodes):
            raise ValueError("Invisible query roles and compiled nodes must have equal lengths")
        if len(set(invisible_names)) != len(invisible_names):
            raise ValueError("Invisible role names must be unique")
        node_to_query = dict(node_entries)
        absent_invisible_nodes = sorted(set(invisible_nodes) - set(node_to_query))
        if absent_invisible_nodes:
            raise ValueError(
                "Invisible roles require queries for their compiled decay nodes; "
                f"missing={absent_invisible_nodes}"
            )
        self.node_query_indices = node_entries
        self.invisible_role_names = invisible_names
        self.invisible_query_indices = tuple(node_to_query[node] for node in invisible_nodes)
        parent_entries = tuple(
            (str(role["name"]), index)
            for index, role in enumerate(normalised)
            if str(role["kind"]) == "parent"
        )
        self.parent_role_names = tuple(name for name, _ in parent_entries)
        self.parent_query_indices = tuple(index for _, index in parent_entries)
        self.topology_identity = str(topology_identity)
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
        if len(invisible_names):
            self.invisible_projection = nn.Sequential(
                nn.LayerNorm(d_model), nn.Linear(d_model, d_model)
            )
            self.invisible_role_embedding = nn.Parameter(
                torch.empty(len(invisible_names), d_model)
            )
            nn.init.normal_(self.invisible_role_embedding, mean=0.0, std=0.02)
        else:
            self.invisible_projection = None
            self.register_parameter("invisible_role_embedding", None)
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
            query, event, event, key_padding_mask=event_padding, need_weights=False
        )
        query = self.query_norm[0](query + query_delta)
        for attention, norm, ffn in zip(
            self.query_attention, self.query_norm[1:], self.query_ffn, strict=True
        ):
            delta, _ = attention(
                query, query, query, attn_mask=self.query_attention_mask,
                need_weights=False,
            )
            query = norm(query + delta)
            query = query + ffn(query)
        summary = query.mean(dim=1)
        context = pooled + self.event_context(summary)
        node_contexts = {
            name: query[:, index] for name, index in self.node_query_indices
        }
        if self.invisible_query_indices:
            assert self.invisible_projection is not None
            invisible_source = query[:, self.invisible_query_indices]
            invisible_contexts = self.invisible_projection(invisible_source)
            invisible_contexts = invisible_contexts + self.invisible_role_embedding.unsqueeze(0)
        else:
            invisible_contexts = query.new_empty((len(query), 0, self.d_model))
        if self.parent_query_indices:
            parent_contexts = query[:, self.parent_query_indices]
        else:
            parent_contexts = query.new_empty((len(query), 0, self.d_model))
        return TopologyQueryState(
            query_states=query,
            global_context=context,
            query_summary=self.summary(summary),
            node_query_indices=self.node_query_indices,
            node_contexts=node_contexts,
            invisible_role_names=self.invisible_role_names,
            invisible_query_indices=self.invisible_query_indices,
            invisible_contexts=invisible_contexts,
            parent_role_names=self.parent_role_names,
            parent_query_indices=self.parent_query_indices,
            parent_contexts=parent_contexts,
            role_names=self.query_names,
            topology_identity=self.topology_identity,
        )


__all__ = ["TopologyQueryBlock", "TopologyQueryState"]
