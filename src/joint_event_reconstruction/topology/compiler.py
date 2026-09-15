"""Compile the deliberately small decay grammar into explicit topology semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Mapping

from omegaconf import OmegaConf

from joint_event_reconstruction.topology.spec import (
    CandidatePoolSpec,
    CompositeRole,
    DecayNode,
    KNOWN_SOURCES,
    LeafRole,
    TopologySpec,
    frozen_mapping,
)
from joint_event_reconstruction.topology.physics import compile_physics


def _leaf(
    value: object,
    context: str,
    truth_bindings: Mapping[str, object] | None,
) -> tuple[str, str, int | None]:
    if not isinstance(value, Mapping) or set(value) != {"source", "truth"}:
        raise ValueError(
            f"{context} must be exactly {{source: SOURCE, truth: SEMANTIC_NAME}}, "
            f"found {value!r}. Raw [SOURCE, TRUTH_INDEX] storage bindings are obsolete."
        )
    source = value["source"]
    truth_name = value["truth"]
    if (
        not isinstance(source, str)
        or source != source.upper()
        or source not in KNOWN_SOURCES
    ):
        raise ValueError(
            f"{context} has unknown or non-canonical source {source!r}; "
            f"expected {sorted(KNOWN_SOURCES)}"
        )
    if not isinstance(truth_name, str) or not truth_name:
        raise ValueError(f"{context}.truth must be a non-empty semantic name")
    if truth_bindings is None:
        return source, truth_name, None
    if truth_name not in truth_bindings:
        raise ValueError(
            f"{context} refers to truth binding {truth_name!r}, which is absent "
            "from the selected data config"
        )
    binding = truth_bindings[truth_name]
    if not isinstance(binding, Mapping) or set(binding) != {"source", "index"}:
        raise ValueError(
            f"Truth binding {truth_name!r} must contain exactly source and index"
        )
    if str(binding["source"]) != source:
        raise ValueError(
            f"Truth binding {truth_name!r} source mismatch: topology={source}, "
            f"data={binding['source']}"
        )
    raw_index = binding["index"]
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        raise ValueError(f"Truth binding {truth_name!r} index must be an integer")
    if source == "NEUTRINO" and raw_index < 0:
        raise ValueError(f"Truth binding {truth_name!r} invisible index must be non-negative")
    if source != "NEUTRINO" and raw_index <= 0:
        raise ValueError(
            f"Truth binding {truth_name!r} visible label must be positive; zero is unassigned"
        )
    return source, truth_name, raw_index


def compile_topology(
    raw: Mapping[str, object],
    truth_bindings: Mapping[str, object] | None = None,
) -> TopologySpec:
    if set(raw) - {"topology", "physics"} or "topology" not in raw:
        raise ValueError(
            "Topology file must contain a 'topology' mapping and may contain only "
            "the optional 'physics' mapping, "
            f"found keys={sorted(raw)}"
        )
    definitions = raw["topology"]
    if not isinstance(definitions, Mapping) or not definitions:
        raise ValueError("topology must be a non-empty mapping of decay-node names")
    node_names = tuple(definitions)
    if any(not isinstance(name, str) or not name for name in node_names):
        raise ValueError("Every decay node requires a non-empty string name")

    leaves: dict[str, LeafRole] = {}
    composites: dict[str, CompositeRole] = {}
    parsed_nodes: dict[str, DecayNode] = {}
    owned_truth_names: set[tuple[str, str]] = set()
    for node_name, roles_raw in definitions.items():
        if not isinstance(roles_raw, Mapping) or not roles_raw:
            raise ValueError(
                f"Decay node {node_name!r} must contain a non-empty role mapping"
            )
        node_leaves = []
        node_composites = []
        for role_name, value in roles_raw.items():
            if not isinstance(role_name, str) or not role_name:
                raise ValueError(
                    f"Decay node {node_name!r} contains an invalid role name "
                    f"{role_name!r}"
                )
            key = f"{node_name}.{role_name}"
            if isinstance(value, str):
                if value not in definitions:
                    raise ValueError(
                        f"Composite role {key} references undefined decay node "
                        f"{value!r}"
                    )
                role = CompositeRole(key, node_name, role_name, value)
                composites[key] = role
                node_composites.append(role)
                continue
            repeated = (
                isinstance(value, (list, tuple))
                and value
                and isinstance(value[0], Mapping)
            )
            members = list(value) if repeated else [value]
            parsed = tuple(_leaf(member, key, truth_bindings) for member in members)
            sources = {source for source, _, _ in parsed}
            if len(sources) != 1:
                raise ValueError(
                    f"Repeated role {key} must contain one source, "
                    f"found {sorted(sources)}"
                )
            truth_names = tuple(name for _, name, _ in parsed)
            indices = tuple(index for _, _, index in parsed)
            if len(set(truth_names)) != len(truth_names):
                raise ValueError(
                    f"Repeated role {key} contains duplicate truth names {truth_names}"
                )
            source = parsed[0][0]
            if source == "NEUTRINO" and len(indices) != 1:
                raise ValueError(
                    f"Invisible role {key} must identify exactly one neutrino. "
                    "Use separate named NEUTRINO roles for multiple invisible "
                    "particles so their semantic order is explicit."
                )
            for source_name, truth_name, index in parsed:
                binding = (source_name, truth_name)
                if binding in owned_truth_names:
                    raise ValueError(
                        f"Truth binding {binding} is owned by more than one leaf role"
                    )
                owned_truth_names.add(binding)
            role = LeafRole(key, node_name, role_name, source, truth_names, indices)
            leaves[key] = role
            node_leaves.append(role)
        parsed_nodes[node_name] = DecayNode(
            node_name,
            tuple(sorted(node_leaves, key=lambda role: role.name)),
            tuple(sorted(node_composites, key=lambda role: role.name)),
        )

    children = {
        name: tuple(
            sorted(role.child for role in parsed_nodes[name].composite_roles)
        )
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
            raise ValueError(
                f"Topology contains a composite cycle: "
                f"{' -> '.join(trail + (name,))}"
            )
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
        visible = [
            role.key for role in parsed_nodes[name].leaf_roles if not role.invisible
        ]
        invisible = [
            role.key for role in parsed_nodes[name].leaf_roles if role.invisible
        ]
        ownership = [
            key
            for role in parsed_nodes[name].leaf_roles
            if not role.invisible
            for key in role.ownership_keys
        ]
        for child in children[name]:
            visible.extend(visible_support[child])
            invisible.extend(invisible_completion[child])
            ownership.extend(pools[child].leaf_ownership)
        visible_support[name] = tuple(visible)
        invisible_completion[name] = tuple(invisible)
        pools[name] = CandidatePoolSpec(
            node=name,
            visible_leaf_roles=tuple(
                role.key
                for role in parsed_nodes[name].leaf_roles
                if not role.invisible
            ),
            child_roles=tuple(
                role.key for role in parsed_nodes[name].composite_roles
            ),
            leaf_ownership=tuple(ownership),
        )

    roots = tuple(sorted(name for name in node_names if not parents_lists[name]))
    canonical = {
        "nodes": {
            name: {
                role.name: (
                    {"child": role.child}
                    if isinstance(role, CompositeRole)
                    else {
                        "source": role.source,
                        "truth": role.truth_names,
                    }
                )
                for role in parsed_nodes[name].roles
            }
            for name in sorted(node_names)
        }
    }
    signature = hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode()
    ).hexdigest()
    physics = None
    if "physics" in raw:
        physics = compile_physics(
            raw["physics"],
            node_names=node_names,
            leaf_names=tuple(leaves),
            topology_composite_names=tuple(composites),
        )
    return TopologySpec(
        nodes=frozen_mapping(parsed_nodes),
        leaf_roles=frozen_mapping(leaves),
        composite_roles=frozen_mapping(composites),
        dag_order=tuple(order),
        roots=roots,
        parents=frozen_mapping(
            {
                name: tuple(sorted(values))
                for name, values in parents_lists.items()
            }
        ),
        children=frozen_mapping(children),
        descendants=frozen_mapping(descendants),
        visible_support=frozen_mapping(visible_support),
        invisible_completion=frozen_mapping(invisible_completion),
        candidate_pools=frozen_mapping(pools),
        recurrent_tokens=tuple(order),
        signature=signature,
        physics=physics,
    )


def load_topology(
    path: str | Path,
    truth_bindings: Mapping[str, object] | None = None,
) -> TopologySpec:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Topology configuration does not exist: {path}"
        )
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return compile_topology(raw, truth_bindings=truth_bindings)
