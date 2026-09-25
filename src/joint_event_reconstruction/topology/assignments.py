"""Tensorised complete assignment states compiled from topology candidate pools."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping

import torch

from joint_event_reconstruction.topology.candidates import CandidateRecord
from joint_event_reconstruction.topology.spec import TopologySpec

NULL_CANDIDATE = -1
MASK_NODE = -2


@dataclass(frozen=True)
class AssignmentTemplate:
    roots: tuple[str, ...]
    nodes: tuple[str, ...]
    root_candidate_indices: torch.Tensor
    node_candidate_indices: torch.Tensor
    non_null_mask: torch.Tensor

    @property
    def state_count(self) -> int:
        return int(self.root_candidate_indices.shape[0])

    def batch(self, pools: Mapping[str, Mapping[str, torch.Tensor]]) -> dict:
        batch_size = int(pools[self.roots[0]]["candidate_mask"].shape[0])
        valid = torch.ones((batch_size, self.state_count), dtype=torch.bool)
        for root_index, root in enumerate(self.roots):
            selected = self.root_candidate_indices[:, root_index]
            non_null = selected >= 0
            candidate_mask = pools[root]["candidate_mask"]
            if candidate_mask.shape[1] == 0:
                if non_null.any():
                    raise RuntimeError(f"Assignment template selects candidates from empty root pool {root}")
                continue
            selected_valid = candidate_mask[:, selected.clamp_min(0)]
            valid &= (~non_null).unsqueeze(0) | selected_valid

        truth_roots = torch.stack([
            torch.where(
                pools[root]["reconstructable"],
                pools[root]["truth_index"],
                torch.full_like(pools[root]["truth_index"], NULL_CANDIDATE),
            )
            for root in self.roots
        ], dim=1)
        matches = (
            self.root_candidate_indices.unsqueeze(0) == truth_roots.unsqueeze(1)
        ).all(dim=2) & valid
        counts = matches.sum(dim=1)
        if (counts != 1).any():
            rows = (counts != 1).nonzero(as_tuple=False).flatten()[:10].tolist()
            raise RuntimeError(
                "Every event must map to exactly one complete assignment truth state; "
                f"rows={rows}, counts={counts[rows].tolist()}"
            )
        return {
            "roots": self.roots,
            "nodes": self.nodes,
            "root_candidate_indices": self.root_candidate_indices,
            "node_candidate_indices": self.node_candidate_indices,
            "non_null_mask": self.non_null_mask,
            "valid_mask": valid,
            "truth_state": matches.to(torch.long).argmax(dim=1),
        }


@dataclass(frozen=True)
class AssignmentSpace:
    roots: tuple[str, ...]
    nodes: tuple[str, ...]
    root_candidate_indices: torch.Tensor
    node_candidate_indices: torch.Tensor
    non_null_mask: torch.Tensor
    valid_mask: torch.Tensor
    truth_state: torch.Tensor

    @classmethod
    def from_structured(cls, structured: Mapping) -> "AssignmentSpace":
        if "assignments" not in structured:
            raise KeyError("Structured batch has no complete assignment state space")
        values = structured["assignments"]
        return cls(
            roots=tuple(values["roots"]),
            nodes=tuple(values["nodes"]),
            root_candidate_indices=values["root_candidate_indices"],
            node_candidate_indices=values["node_candidate_indices"],
            non_null_mask=values["non_null_mask"],
            valid_mask=values["valid_mask"],
            truth_state=values["truth_state"],
        )

    @property
    def state_count(self) -> int:
        return int(self.root_candidate_indices.shape[0])

    @property
    def mask_state(self) -> int:
        return self.state_count

    def state_logits(self, node_logits: Mapping[str, torch.Tensor]) -> torch.Tensor:
        sample = next(iter(node_logits.values()))
        scores = torch.zeros(
            (self.valid_mask.shape[0], self.state_count),
            dtype=sample.dtype, device=sample.device,
        )
        for root_index, root in enumerate(self.roots):
            selected = self.root_candidate_indices[:, root_index]
            scores = scores + node_logits[root][:, selected + 1]
        return scores.masked_fill(~self.valid_mask, -torch.inf)

    def state_probabilities(self, node_logits: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.state_logits(node_logits).softmax(dim=1)

    def node_indices(self, state: torch.Tensor) -> torch.Tensor:
        if ((state < 0) | (state > self.mask_state)).any():
            raise ValueError("Assignment state index lies outside clean states plus MASK")
        is_mask = state == self.mask_state
        safe = state.clamp_max(self.state_count - 1)
        selected = self.node_candidate_indices[safe.reshape(-1)].reshape(
            *state.shape, len(self.nodes)
        )
        return torch.where(
            is_mask.unsqueeze(-1),
            torch.full_like(selected, MASK_NODE),
            selected,
        )

    def topk_non_null(
        self, node_logits: Mapping[str, torch.Tensor], k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if k <= 0:
            raise ValueError("top-k assignment count must be positive")
        logits = self.state_logits(node_logits)
        allowed = self.valid_mask & self.non_null_mask.unsqueeze(0)
        counts = allowed.sum(dim=1)
        if (counts < k).any():
            rows = (counts < k).nonzero(as_tuple=False).flatten()[:10].tolist()
            raise RuntimeError(f"Fewer than k={k} complete non-null assignments for rows={rows}")
        restricted = logits.masked_fill(~allowed, -torch.inf)
        values, states = restricted.topk(k, dim=1)
        return states, values, torch.logsumexp(restricted, dim=1)


def build_assignment_template(
    spec: TopologySpec,
    pools: Mapping[str, tuple[CandidateRecord, ...]],
    *,
    max_states: int | None = None,
) -> AssignmentTemplate:
    if max_states is not None and int(max_states) <= 0:
        raise ValueError("max_states must be positive or null")
    roots = tuple(spec.roots)
    nodes = tuple(spec.dag_order)
    node_position = {name: index for index, name in enumerate(nodes)}
    root_states: list[tuple[int, ...]] = []
    node_states: list[tuple[int, ...]] = []
    options = [tuple([NULL_CANDIDATE, *range(len(pools[root]))]) for root in roots]

    for root_selection in product(*options):
        ownership: set[tuple[str, int]] = set()
        selected_nodes: dict[str, int] = {}
        valid = True

        def select_node(node_name: str, candidate_index: int) -> None:
            nonlocal valid
            previous = selected_nodes.get(node_name)
            if previous is not None:
                if previous != candidate_index:
                    valid = False
                return
            selected_nodes[node_name] = candidate_index
            candidate = pools[node_name][candidate_index]
            for role_key, child_index in candidate.children:
                select_node(spec.composite_roles[role_key].child, int(child_index))

        for root, candidate_index in zip(roots, root_selection):
            if candidate_index == NULL_CANDIDATE:
                continue
            candidate = pools[root][candidate_index]
            if ownership & candidate.leaf_ownership:
                valid = False
                break
            ownership.update(candidate.leaf_ownership)
            select_node(root, int(candidate_index))
            if not valid:
                break

        if valid:
            root_states.append(tuple(int(index) for index in root_selection))
            if max_states is not None and len(root_states) > int(max_states):
                root_sizes = {root: len(pools[root]) for root in roots}
                raise RuntimeError(
                    "Complete assignment state count exceeds the configured limit: "
                    f"observed_at_least={len(root_states)}, max_states={int(max_states)}, "
                    f"root_candidates={root_sizes}, topology={spec.signature}"
                )
            node_state = [NULL_CANDIDATE] * len(nodes)
            for node, candidate_index in selected_nodes.items():
                node_state[node_position[node]] = int(candidate_index)
            node_states.append(tuple(node_state))

    if not root_states:
        raise RuntimeError("Topology produced no globally feasible assignment states")
    if len(set(root_states)) != len(root_states):
        raise RuntimeError("Complete assignment state enumeration produced duplicates")

    root_tensor = torch.tensor(root_states, dtype=torch.long)
    return AssignmentTemplate(
        roots=roots,
        nodes=nodes,
        root_candidate_indices=root_tensor,
        node_candidate_indices=torch.tensor(node_states, dtype=torch.long),
        non_null_mask=(root_tensor >= 0).all(dim=1),
    )
