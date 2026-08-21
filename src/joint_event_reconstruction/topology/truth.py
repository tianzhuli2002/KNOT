"""Compile raw HDF5 labels into semantic, hierarchical topology truth."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from joint_event_reconstruction.topology.spec import TopologySpec


@dataclass(frozen=True)
class LeafTruth:
    indices: torch.Tensor
    physical_present: torch.Tensor
    selected_present: torch.Tensor
    reconstructable: torch.Tensor


@dataclass(frozen=True)
class NodeTruth:
    visible_reconstructable: torch.Tensor
    full_reconstructable: torch.Tensor


@dataclass(frozen=True)
class TopologyTruth:
    leaves: Mapping[str, LeafTruth]
    nodes: Mapping[str, NodeTruth]


def compile_truth(
    spec: TopologySpec,
    labels: Mapping[str, torch.Tensor],
    object_masks: Mapping[str, torch.Tensor],
    *,
    selected_widths: Mapping[str, int],
) -> TopologyTruth:
    """Interpret raw labels once at the data/topology boundary.

    ``labels`` and ``object_masks`` retain the full stored object width. A selected
    truth object outside ``selected_widths[source]`` is physically present but not
    reconstructable by the model.
    """
    batch_size = None
    for source, values in labels.items():
        if values.ndim != 2:
            raise ValueError(f"Raw {source} labels must have shape [batch, object], found {tuple(values.shape)}")
        mask = object_masks.get(source)
        if mask is None or mask.shape != values.shape or mask.dtype != torch.bool:
            raise ValueError(f"Object mask for {source} must be boolean with shape {tuple(values.shape)}")
        if batch_size is None:
            batch_size = values.shape[0]
        elif values.shape[0] != batch_size:
            raise ValueError("Raw topology-label sources have inconsistent batch sizes")
        valid_values = values[mask]
        if valid_values.numel() and not torch.isfinite(valid_values).all():
            raise ValueError(f"Real selected objects in {source} contain non-finite raw labels")
        finite = values[torch.isfinite(values)]
        if finite.numel() and not torch.equal(finite, finite.round()):
            raise ValueError(f"Raw {source} labels must be integers, zero, or NaN padding")
    if batch_size is None:
        raise ValueError("No raw topology-label sources were supplied")

    leaf_truth: dict[str, LeafTruth] = {}
    for role in spec.leaf_roles.values():
        if role.invisible and role.source not in labels:
            indices = torch.full((batch_size, role.multiplicity), -1, dtype=torch.long)
            missing = torch.zeros((batch_size, role.multiplicity), dtype=torch.bool)
            leaf_truth[role.key] = LeafTruth(indices, missing, missing, missing.all(dim=1))
            continue
        if role.source not in labels:
            raise KeyError(f"Visible role {role.key} requires missing label source {role.source}")
        source_labels = labels[role.source]
        source_mask = object_masks[role.source]
        width = int(selected_widths.get(role.source, 0))
        if width < 0 or width > source_labels.shape[1]:
            raise ValueError(
                f"Selected width for {role.source} is {width}, stored width is {source_labels.shape[1]}"
            )
        role_indices = []
        physical = []
        selected = []
        positions = torch.arange(source_labels.shape[1], device=source_labels.device).unsqueeze(0)
        for truth_index in role.truth_indices:
            matches = source_mask & (source_labels == truth_index)
            counts = matches.sum(dim=1)
            if (counts > 1).any():
                event = int((counts > 1).nonzero(as_tuple=False)[0])
                raise ValueError(
                    f"Truth label {truth_index} for {role.key} occurs more than once in {role.source}; event={event}"
                )
            present = counts == 1
            selected_match = matches & (positions < width)
            is_selected = selected_match.any(dim=1)
            index = selected_match.to(torch.int64).argmax(dim=1)
            index = torch.where(is_selected, index, torch.full_like(index, -1))
            role_indices.append(index)
            physical.append(present)
            selected.append(is_selected)
        physical_tensor = torch.stack(physical, dim=1)
        selected_tensor = torch.stack(selected, dim=1)
        leaf_truth[role.key] = LeafTruth(
            indices=torch.stack(role_indices, dim=1),
            physical_present=physical_tensor,
            selected_present=selected_tensor,
            reconstructable=selected_tensor.all(dim=1),
        )

    node_truth: dict[str, NodeTruth] = {}
    for node_name in spec.dag_order:
        node = spec.nodes[node_name]
        visible_terms = [leaf_truth[role.key].reconstructable for role in node.leaf_roles if not role.invisible]
        invisible_terms = [leaf_truth[role.key].reconstructable for role in node.leaf_roles if role.invisible]
        for role in node.composite_roles:
            visible_terms.append(node_truth[role.child].visible_reconstructable)
            invisible_terms.append(node_truth[role.child].full_reconstructable)
        visible = torch.stack(visible_terms, dim=1).all(dim=1) if visible_terms else torch.ones(batch_size, dtype=torch.bool)
        full_terms = [visible, *invisible_terms]
        full = torch.stack(full_terms, dim=1).all(dim=1)
        node_truth[node_name] = NodeTruth(visible, full)

    return TopologyTruth(MappingProxyType(leaf_truth), MappingProxyType(node_truth))
