"""Topology-derived four-vector composition for selected event hypotheses."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from joint_event_reconstruction.topology.spec import TopologySpec


def massless_invisible_four_vector(momentum: torch.Tensor) -> torch.Tensor:
    if momentum.shape[-1] != 3:
        raise ValueError("Invisible momentum must end in Cartesian [px,py,pz]")
    energy = momentum.square().sum(dim=-1, keepdim=True).sqrt()
    return torch.cat((energy, momentum), dim=-1)


def compose_four_vectors(
    spec: TopologySpec,
    visible_leaves: Mapping[str, torch.Tensor],
    invisible_momenta: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Recursively sum `(E,px,py,pz)` leaves without topology-name branches."""
    nodes: dict[str, torch.Tensor] = {}
    for node_name in spec.dag_order:
        pieces = []
        node = spec.nodes[node_name]
        for role in node.leaf_roles:
            if role.invisible:
                if role.key not in invisible_momenta:
                    raise KeyError(f"Missing invisible four-vector input {role.key}")
                pieces.append(massless_invisible_four_vector(invisible_momenta[role.key]))
            else:
                if role.key not in visible_leaves:
                    raise KeyError(f"Missing visible four-vector input {role.key}")
                vector = visible_leaves[role.key]
                if vector.ndim < 2 or vector.shape[-2:] != (role.multiplicity, 4):
                    raise ValueError(
                        f"Visible role {role.key} must end in "
                        f"[{role.multiplicity},4] (E,px,py,pz)"
                    )
                pieces.append(vector.sum(dim=-2))
        for role in node.composite_roles:
            pieces.append(nodes[role.child])
        if not pieces:
            raise RuntimeError(f"Compiled decay node {node_name} contains no physical pieces")
        nodes[node_name] = torch.stack(pieces, dim=0).sum(dim=0)
    return nodes


def compose_assignment_state(
    spec: TopologySpec,
    pools: Mapping[str, Mapping[str, torch.Tensor]],
    physics_vectors: torch.Tensor,
    node_candidate_indices: torch.Tensor,
    invisible_momenta: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compose one complete assignment state, preserving physical NULL branches.

    ``node_candidate_indices`` follows ``spec.dag_order`` and uses ``-1`` for a
    topology node absent because its root is physically NULL. Such a node receives
    a NaN four-vector; selected independent roots remain composable.
    """
    if physics_vectors.ndim != 2 or physics_vectors.shape[-1] != 4:
        raise ValueError("physics_vectors must have shape [tokens,4]")
    if node_candidate_indices.shape != (len(spec.dag_order),):
        raise ValueError(
            "node_candidate_indices must contain one candidate index per DAG node"
        )

    nan = torch.full(
        (4,), float("nan"), device=physics_vectors.device, dtype=physics_vectors.dtype
    )
    nodes: dict[str, torch.Tensor] = {}
    for node_position, node_name in enumerate(spec.dag_order):
        candidate_index = int(node_candidate_indices[node_position])
        if candidate_index < 0:
            nodes[node_name] = nan.clone()
            continue

        pieces = []
        node = spec.nodes[node_name]
        pool = pools[node_name]
        for role in node.leaf_roles:
            if role.invisible:
                if role.key not in invisible_momenta:
                    raise KeyError(f"Missing invisible four-vector input {role.key}")
                pieces.append(
                    massless_invisible_four_vector(invisible_momenta[role.key])
                )
            else:
                indices = pool["leaf_indices"][role.key][candidate_index]
                pieces.append(physics_vectors[indices].sum(dim=0))
        for role in node.composite_roles:
            child = nodes[role.child]
            if not torch.isfinite(child).all():
                raise RuntimeError(
                    f"Selected node {node_name} has absent composite child {role.child}"
                )
            pieces.append(child)
        if not pieces:
            raise RuntimeError(f"Compiled decay node {node_name} contains no physical pieces")
        nodes[node_name] = torch.stack(pieces, dim=0).sum(dim=0)
    return nodes


def invariant_mass(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 4:
        raise ValueError("Four-vector must end in (E,px,py,pz)")
    mass2 = vector[..., 0].square() - vector[..., 1:].square().sum(dim=-1)
    return mass2.clamp_min(0).sqrt()
