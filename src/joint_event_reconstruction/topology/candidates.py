"""Reference candidate construction from a compiled topology.

The CPU builder is used for topology inspection and decoder validation. Neural-model
candidate pools use the same semantics with batched index tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Mapping

from joint_event_reconstruction.topology.spec import TopologySpec
from joint_event_reconstruction.topology.truth import TopologyTruth


ObjectKey = tuple[str, int]


@dataclass(frozen=True)
class CandidateRecord:
    node: str
    index: int
    constituents: tuple[tuple[str, tuple[int, ...]], ...]
    children: tuple[tuple[str, int], ...]
    leaf_ownership: frozenset[ObjectKey]
    truth: bool


def build_event_candidates(
    spec: TopologySpec,
    valid_indices: Mapping[str, tuple[int, ...]],
    truth: TopologyTruth | None = None,
    *,
    event: int = 0,
) -> dict[str, tuple[CandidateRecord, ...]]:
    pools: dict[str, tuple[CandidateRecord, ...]] = {}
    for node_name in spec.dag_order:
        node = spec.nodes[node_name]
        options: list[tuple[str, tuple]] = []
        for role in node.leaf_roles:
            if role.invisible:
                continue
            available = valid_indices.get(role.source)
            if available is None:
                raise KeyError(f"Candidate construction for {role.key} requires source {role.source}")
            options.append((role.key, tuple(combinations(available, role.multiplicity))))
        for role in node.composite_roles:
            options.append((role.key, pools[role.child]))

        records = []
        combinations_by_role = product(*(values for _, values in options)) if options else [()]
        for choices in combinations_by_role:
            ownership: set[ObjectKey] = set()
            constituents = []
            children = []
            is_truth = truth is not None
            valid = True
            for (role_key, _), choice in zip(options, choices):
                if isinstance(choice, CandidateRecord):
                    if ownership & choice.leaf_ownership:
                        valid = False
                        break
                    ownership.update(choice.leaf_ownership)
                    children.append((role_key, choice.index))
                    is_truth = is_truth and choice.truth
                else:
                    role = spec.leaf_roles[role_key]
                    objects = frozenset((role.source, int(index)) for index in choice)
                    if ownership & objects:
                        valid = False
                        break
                    ownership.update(objects)
                    constituents.append((role_key, tuple(int(index) for index in choice)))
                    if truth is not None:
                        expected = truth.leaves[role_key].indices[event]
                        expected_values = tuple(sorted(int(index) for index in expected.tolist() if index >= 0))
                        is_truth = is_truth and tuple(sorted(choice)) == expected_values
            if valid:
                records.append(CandidateRecord(
                    node=node_name, index=len(records), constituents=tuple(constituents),
                    children=tuple(children), leaf_ownership=frozenset(ownership), truth=bool(is_truth),
                ))
        if truth is not None and truth.nodes[node_name].visible_reconstructable[event]:
            matches = sum(candidate.truth for candidate in records)
            if matches != 1:
                raise RuntimeError(
                    f"Reconstructable node {node_name} has {matches} truth candidates; event={event}, "
                    f"candidate_count={len(records)}"
                )
        pools[node_name] = tuple(records)
    return pools
