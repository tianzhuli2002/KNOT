"""Compile the deliberately small decay grammar into explicit topology semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

from omegaconf import OmegaConf

from joint_event_reconstruction.topology.spec import (
    CandidatePoolSpec, CompositeRole, DecayNode, KNOWN_SOURCES, LeafRole,
    TopologySpec, frozen_mapping,
)


def _leaf(value: object, context: str) -> tuple[str, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{context} must be exactly [SOURCE, TRUTH_INDEX], found {value!r}")
    source, raw_index = value
    if not isinstance(source, str) or source != source.upper() or source not in KNOWN_SOURCES:
        raise ValueError(f"{context} has unknown or non-canonical source {source!r}; expected {sorted(KNOWN_SOURCES)}")
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        raise ValueError(f"{context} truth index must be an integer, found {raw_index!r}")
    if source == "NEUTRINO" and raw_index < 0:
        raise ValueError(f"{context} invisible truth index must be non-negative")
    if source != "NEUTRINO" and raw_index <= 0:
        raise ValueError(f"{context} visible truth labels must be positive; zero is unassigned")
    return source, raw_index


def compile_topology(raw: Mapping[str, object]) -> TopologySpec:
    if set(raw) != {"topology"}:
        raise ValueError(f"Topology file must contain only the 'topology' mapping, found keys={sorted(raw)}")
    definitions = raw["topology"]
    if not isinstance(definitions, Mapping) or not definitions:
        raise ValueError("topology must be a non-empty mapping of decay-node names")
    node_names = tuple(definitions)
    if any(not isinstance(name, str) or not name for name in node_names):
        raise ValueError("Every decay node requires a non-empty string name")

    leaves: dict[str, LeafRole] = {}
    composites: dict[str, CompositeRole] = {}
    parsed_nodes: dict[str, DecayNode] = {}
    truth_bindings: set[tuple[str, int]] = set()
    for node_name, roles_raw in definitions.items():
        if not isinstance(roles_raw, Mapping) or not roles_raw:
            raise ValueError(f"Decay node {node_name!r} must contain a non-empty role mapping")
        node_leaves = []
        node_composites = []
        for role_name, value in roles_raw.items():
            if not isinstance(role_name, str) or not role_name:
                raise ValueError(f"Decay node {node_name!r} contains an invalid role name {role_name!r}")
            key = f"{node_name}.{role_name}"
            if isinstance(value, str):
                if value not in definitions:
                    raise ValueError(f"Composite role {key} references undefined decay node {value!r}")
                role = CompositeRole(key, node_name, role_name, value)
                composites[key] = role
                node_composites.append(role)
                continue
            repeated = isinstance(value, (list, tuple)) and value and isinstance(value[0], (list, tuple))
            members = list(value) if repeated else [value]
            parsed = tuple(_leaf(member, key) for member in members)
            sources = {source for source, _ in parsed}
            if len(sources) != 1:
                raise ValueError(f"Repeated role {key} must contain one source, found {sorted(sources)}")
            indices = tuple(sorted(index for _, index in parsed))
            if len(set(indices)) != len(indices):
                raise ValueError(f"Repeated role {key} contains duplicate truth indices {indices}")
            source = parsed[0][0]
            for binding in parsed:
                if binding in truth_bindings:
                    raise ValueError(f"Truth binding {binding} is owned by more than one leaf role")
                truth_bindings.add(binding)
            role = LeafRole(key, node_name, role_name, source, indices)
            leaves[key] = role
            node_leaves.append(role)
        parsed_nodes[node_name] = DecayNode(
            node_name,
            tuple(sorted(node_leaves, key=lambda role: role.name)),
            tuple(sorted(node_composites, key=lambda role: role.name)),
        )

    children = {
        name: tuple(sorted(role.child for role in parsed_nodes[name].composite_roles))
        for name in node_names
    }
    parents_lists = {name: [] for name in node_names}
    for parent, child_names in children.items():
        for child in child_names:
            parents_lists[child].append(parent)

    state: dict[str, int] = {}
    order: list[str] = []
    def visit(name: str, trail: tuple[str, ...]) -> None:
        marker = state.get(name, 0)
        if marker == 1:
            raise ValueError(f"Topology contains a composite cycle: {' -> '.join(trail + (name,))}")
        if marker == 2:
            return
        state[name] = 1
        for child in children[name]:
            visit(child, trail + (name,))
        state[name] = 2
        order.append(name)
    for name in sorted(node_names):
        visit(name, ())

    descendants: dict[str, tuple[str, ...]] = {}
    visible_support: dict[str, tuple[str, ...]] = {}
    invisible_completion: dict[str, tuple[str, ...]] = {}
    pools: dict[str, CandidatePoolSpec] = {}
    for name in order:
        child_descendants = {child for child in children[name]}
        for child in children[name]:
            child_descendants.update(descendants[child])
        descendants[name] = tuple(sorted(child_descendants))
        visible = [role.key for role in parsed_nodes[name].leaf_roles if not role.invisible]
        invisible = [role.key for role in parsed_nodes[name].leaf_roles if role.invisible]
        ownership = [key for role in parsed_nodes[name].leaf_roles if not role.invisible for key in role.ownership_keys]
        for child in children[name]:
            visible.extend(visible_support[child])
            invisible.extend(invisible_completion[child])
            ownership.extend(pools[child].leaf_ownership)
        visible_support[name] = tuple(visible)
        invisible_completion[name] = tuple(invisible)
        pools[name] = CandidatePoolSpec(
            node=name,
            visible_leaf_roles=tuple(role.key for role in parsed_nodes[name].leaf_roles if not role.invisible),
            child_roles=tuple(role.key for role in parsed_nodes[name].composite_roles),
            leaf_ownership=tuple(ownership),
        )

    roots = tuple(sorted(name for name in node_names if not parents_lists[name]))
    canonical = {
        "nodes": {
            name: {
                role.name: ({"child": role.child} if isinstance(role, CompositeRole)
                            else {"source": role.source, "truth_indices": role.truth_indices})
                for role in parsed_nodes[name].roles
            } for name in sorted(node_names)
        }
    }
    signature = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
    return TopologySpec(
        nodes=frozen_mapping(parsed_nodes), leaf_roles=frozen_mapping(leaves),
        composite_roles=frozen_mapping(composites), dag_order=tuple(order), roots=roots,
        parents=frozen_mapping({name: tuple(sorted(values)) for name, values in parents_lists.items()}),
        children=frozen_mapping(children), descendants=frozen_mapping(descendants),
        visible_support=frozen_mapping(visible_support), invisible_completion=frozen_mapping(invisible_completion),
        candidate_pools=frozen_mapping(pools), recurrent_tokens=tuple(order), signature=signature,
    )


def load_topology(path: str | Path) -> TopologySpec:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Topology configuration does not exist: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return compile_topology(raw)
