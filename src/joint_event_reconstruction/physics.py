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
                        f"Visible role {role.key} must end in [{role.multiplicity},4] (E,px,py,pz)"
                    )
                pieces.append(vector.sum(dim=-2))
        for role in node.composite_roles:
            pieces.append(nodes[role.child])
        if not pieces:
            raise RuntimeError(f"Compiled decay node {node_name} contains no physical pieces")
        nodes[node_name] = torch.stack(pieces, dim=0).sum(dim=0)
    return nodes


def invariant_mass(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 4:
        raise ValueError("Four-vector must end in (E,px,py,pz)")
    mass2 = vector[..., 0].square() - vector[..., 1:].square().sum(dim=-1)
    return mass2.clamp_min(0).sqrt()
