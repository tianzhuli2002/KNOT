"""Small immutable objects produced by the topology compiler."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


VISIBLE_SOURCES = frozenset({"JET", "LEPTON", "ELECTRON", "MUON"})
INVISIBLE_SOURCES = frozenset({"NEUTRINO"})
KNOWN_SOURCES = VISIBLE_SOURCES | INVISIBLE_SOURCES


@dataclass(frozen=True)
class LeafRole:
    key: str
    node: str
    name: str
    source: str
    truth_names: tuple[str, ...]
    truth_indices: tuple[int | None, ...]

    @property
    def multiplicity(self) -> int:
        return len(self.truth_names)

    @property
    def exchangeable(self) -> bool:
        return self.multiplicity > 1

    @property
    def invisible(self) -> bool:
        return self.source in INVISIBLE_SOURCES

    @property
    def ownership_keys(self) -> tuple[str, ...]:
        if self.multiplicity == 1:
            return (self.key,)
        return tuple(f"{self.key}[{slot}]" for slot in range(self.multiplicity))


@dataclass(frozen=True)
class CompositeRole:
    key: str
    node: str
    name: str
    child: str


@dataclass(frozen=True)
class DecayNode:
    name: str
    leaf_roles: tuple[LeafRole, ...]
    composite_roles: tuple[CompositeRole, ...]

    @property
    def roles(self) -> tuple[LeafRole | CompositeRole, ...]:
        return self.leaf_roles + self.composite_roles


@dataclass(frozen=True)
class CandidatePoolSpec:
    node: str
    visible_leaf_roles: tuple[str, ...]
    child_roles: tuple[str, ...]
    leaf_ownership: tuple[str, ...]


@dataclass(frozen=True)
class TopologySpec:
    nodes: Mapping[str, DecayNode]
    leaf_roles: Mapping[str, LeafRole]
    composite_roles: Mapping[str, CompositeRole]
    dag_order: tuple[str, ...]
    roots: tuple[str, ...]
    parents: Mapping[str, tuple[str, ...]]
    children: Mapping[str, tuple[str, ...]]
    descendants: Mapping[str, tuple[str, ...]]
    visible_support: Mapping[str, tuple[str, ...]]
    invisible_completion: Mapping[str, tuple[str, ...]]
    candidate_pools: Mapping[str, CandidatePoolSpec]
    recurrent_tokens: tuple[str, ...]
    signature: str

    @property
    def invisible_roles(self) -> tuple[LeafRole, ...]:
        return tuple(role for role in self.leaf_roles.values() if role.invisible)

    @property
    def visible_leaf_roles(self) -> tuple[LeafRole, ...]:
        return tuple(role for role in self.leaf_roles.values() if not role.invisible)


def frozen_mapping(values: dict) -> Mapping:
    return MappingProxyType(dict(values))
