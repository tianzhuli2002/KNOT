"""Topology-level physics composition and hypothesis compatibility.

The module intentionally contains a small physics grammar rather than a general
expression language.  It is usable by prediction and offline analysis without
depending on a particular neural model.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from joint_event_reconstruction.topology.spec import (
    LeafRole,
    PhysicsCompositeSpec,
    PhysicsObservableSpec,
    PhysicsSpec,
    TopologySpec,
)


_OBSERVABLE_TYPES = frozenset(
    {"invariant_mass", "mass_difference", "transverse_sum_residual"}
)


def _finite_reference(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must be a finite number")
    return result


def _as_nonempty_names(value: object, context: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{context} must be a non-empty list of names")
    names = tuple(value)
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError(f"{context} must contain non-empty string names")
    return names


def compile_physics(
    raw: Mapping[str, object],
    *,
    node_names: Sequence[str],
    leaf_names: Sequence[str],
    topology_composite_names: Sequence[str],
) -> PhysicsSpec:
    """Compile the deliberately small optional physics block.

    The compiler resolves names but does not attach any hard mass-shell or MET
    constraint.  Those are evaluated later as compatibility residuals.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("physics must be a mapping")
    unknown = set(raw) - {"composites", "observables"}
    if unknown:
        raise ValueError(f"physics contains unknown keys: {sorted(unknown)}")
    composites_raw = raw.get("composites", {})
    observables_raw = raw.get("observables", {})
    if not isinstance(composites_raw, Mapping):
        raise ValueError("physics.composites must be a mapping")
    if not isinstance(observables_raw, Mapping):
        raise ValueError("physics.observables must be a mapping")

    topology_names = set(node_names) | set(leaf_names) | set(topology_composite_names)
    raw_composite_names = tuple(composites_raw)
    if any(not isinstance(name, str) or not name for name in raw_composite_names):
        raise ValueError("physics composite names must be non-empty strings")
    names = tuple(sorted(raw_composite_names))
    if len(names) != len(set(names)):
        raise ValueError("physics.composites contains duplicate names")
    if any(not name for name in names):
        raise ValueError("physics composite names must be non-empty strings")
    collisions = sorted(set(names) & topology_names)
    if collisions:
        raise ValueError(
            "physics composite names collide with topology objects: "
            f"{collisions}"
        )

    parsed_composites: dict[str, PhysicsCompositeSpec] = {}
    physics_names = set(names)
    for name in names:
        value = composites_raw[name]
        if not isinstance(value, Mapping) or set(value) != {"sum"}:
            raise ValueError(
                f"physics.composites.{name} must contain exactly the 'sum' key"
            )
        terms = _as_nonempty_names(value["sum"], f"physics.composites.{name}.sum")
        for term in terms:
            if term not in topology_names and term not in physics_names:
                raise ValueError(
                    f"physics.composites.{name} refers to unknown object {term!r}"
                )
        parsed_composites[name] = PhysicsCompositeSpec(name, terms)

    visit_state: dict[str, int] = {}
    order: list[str] = []

    def visit(name: str, trail: tuple[str, ...]) -> None:
        marker = visit_state.get(name, 0)
        if marker == 1:
            raise ValueError(
                "Physics composite cycle: " + " -> ".join(trail + (name,))
            )
        if marker == 2:
            return
        visit_state[name] = 1
        for term in parsed_composites[name].terms:
            if term in parsed_composites:
                visit(term, trail + (name,))
        visit_state[name] = 2
        order.append(name)

    for name in names:
        visit(name, ())

    raw_observable_names = tuple(observables_raw)
    if any(not isinstance(name, str) or not name for name in raw_observable_names):
        raise ValueError("physics observable names must be non-empty strings")
    observable_names = tuple(sorted(raw_observable_names))
    if len(observable_names) != len(set(observable_names)):
        raise ValueError("physics.observables contains duplicate names")
    if set(observable_names) & (topology_names | physics_names):
        raise ValueError("physics observable names must be unique from all objects")

    parsed_observables: list[PhysicsObservableSpec] = []
    known = topology_names | physics_names
    for name in observable_names:
        value = observables_raw[name]
        if not isinstance(value, Mapping) or "type" not in value:
            raise ValueError(f"physics.observables.{name} needs a type")
        kind = value["type"]
        if kind not in _OBSERVABLE_TYPES:
            raise ValueError(
                f"physics.observables.{name} has unsupported type {kind!r}; "
                f"expected {sorted(_OBSERVABLE_TYPES)}"
            )
        if kind == "invariant_mass":
            if set(value) != {"type", "object", "reference"}:
                raise ValueError(
                    f"physics.observables.{name} has invalid invariant_mass keys"
                )
            if isinstance(value["object"], (list, tuple)):
                objects = _as_nonempty_names(value["object"], f"{name}.object")
            elif isinstance(value["object"], str) and value["object"]:
                objects = (value["object"],)
            else:
                raise ValueError(f"{name}.object must identify one named object")
            if len(objects) != 1:
                raise ValueError(f"{name}.object must identify exactly one object")
            reference = _finite_reference(value["reference"], f"{name}.reference")
            measurement = None
        elif kind == "mass_difference":
            if set(value) != {"type", "objects", "reference"}:
                raise ValueError(
                    f"physics.observables.{name} has invalid mass_difference keys"
                )
            objects = _as_nonempty_names(value["objects"], f"{name}.objects")
            if len(objects) != 2:
                raise ValueError(f"{name}.objects must contain exactly two objects")
            reference = _finite_reference(value["reference"], f"{name}.reference")
            measurement = None
        else:
            if set(value) != {"type", "objects", "measurement"}:
                raise ValueError(
                    f"physics.observables.{name} has invalid transverse_sum_residual keys"
                )
            objects = _as_nonempty_names(value["objects"], f"{name}.objects")
            reference = None
            measurement = value["measurement"]
            if measurement != "MET":
                raise ValueError(
                    f"{name}.measurement must be the supported 'MET' measurement"
                )
        for object_name in objects:
            if object_name not in known:
                raise ValueError(
                    f"physics.observables.{name} refers to unknown object {object_name!r}"
                )
        parsed_observables.append(
            PhysicsObservableSpec(name, kind, objects, reference, measurement)
        )

    signature_payload = {
        "composites": [
            {"name": item.name, "terms": item.terms}
            for item in parsed_composites.values()
        ],
        "observables": [
            {
                "name": item.name,
                "kind": item.kind,
                "objects": item.objects,
                "reference": item.reference,
                "measurement": item.measurement,
            }
            for item in parsed_observables
        ],
        "order": tuple(order),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PhysicsSpec(
        composites=tuple(parsed_composites[name] for name in names),
        observables=tuple(parsed_observables),
        order=tuple(order),
        signature=signature,
    )


def _massless_four_vector(momentum: torch.Tensor) -> torch.Tensor:
    if momentum.shape[-1] != 3:
        raise ValueError("Invisible momentum must end in [px, py, pz]")
    energy = momentum.square().sum(dim=-1).clamp_min(0).sqrt()
    return torch.cat((energy.unsqueeze(-1), momentum), dim=-1)


def _normalise_role_values(
    value: torch.Tensor | Mapping[str, torch.Tensor],
    roles: tuple[LeafRole, ...],
    *,
    batch_size: int,
    hypotheses: int,
    last_dim: int,
) -> torch.Tensor:
    if isinstance(value, Mapping):
        pieces = []
        for role in roles:
            if role.key not in value:
                raise KeyError(f"Missing invisible role {role.key}")
            pieces.append(value[role.key])
        value = torch.stack(pieces, dim=-2)
    if not torch.is_tensor(value):
        raise TypeError("Invisible posterior must be a tensor or role mapping")
    if value.shape[-1] != last_dim:
        raise ValueError(
            f"Invisible posterior must end in {last_dim} components, found {value.shape}"
        )
    role_count = len(roles)
    if value.ndim == 4:
        if value.shape[0] != batch_size or value.shape[2] != role_count:
            raise ValueError(
                "Unconditioned invisible posterior must have shape [B,M,R,D]"
            )
        return value.unsqueeze(1).expand(
            batch_size, hypotheses, value.shape[1], role_count, last_dim
        )
    if value.ndim == 5:
        expected = (batch_size, hypotheses, role_count)
        if value.shape[0] != expected[0] or value.shape[1] not in {1, expected[1]}:
            raise ValueError(
                "Hypothesis-conditioned invisible posterior must have shape "
                "[B,K,M,R,D]"
            )
        if value.shape[3] != expected[2]:
            raise ValueError("Invisible posterior role count disagrees with topology")
        if value.shape[1] == 1 and hypotheses != 1:
            return value.expand(batch_size, hypotheses, *value.shape[2:])
        return value
    raise ValueError("Invisible posterior must have four or five dimensions")


class TopologyKinematics:
    """Vectorised physical p4 construction for complete assignment hypotheses."""

    def __init__(self, spec: TopologySpec):
        self.spec = spec
        self._node_position = {name: i for i, name in enumerate(spec.dag_order)}

    def _invisible_p4(
        self,
        invisible_momenta: torch.Tensor | Mapping[str, torch.Tensor] | None,
        invisible_four_vectors: torch.Tensor | Mapping[str, torch.Tensor] | None,
        *,
        batch_size: int,
        hypotheses: int,
        mass_shells: Mapping[str, object] | None,
        masses: Mapping[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        roles = self.spec.invisible_roles
        if invisible_four_vectors is not None:
            values = _normalise_role_values(
                invisible_four_vectors,
                roles,
                batch_size=batch_size,
                hypotheses=hypotheses,
                last_dim=4,
            )
            if not torch.isfinite(values).all():
                raise ValueError("Invisible physical four-vectors must be finite")
            return values
        if invisible_momenta is None:
            raise ValueError("An invisible posterior is required by this topology")
        values = _normalise_role_values(
            invisible_momenta,
            roles,
            batch_size=batch_size,
            hypotheses=hypotheses,
            last_dim=3,
        )
        pieces = []
        for index, role in enumerate(roles):
            momentum = values[..., index, :]
            shell = None if mass_shells is None else mass_shells.get(role.key)
            mass = None if masses is None else masses.get(role.key)
            if shell is None:
                physical = _massless_four_vector(momentum)
            else:
                physical = shell.to_four_vector(momentum, generated_mass=mass)
            if physical.shape[-1] != 4:
                raise ValueError(f"Mass-shell conversion for {role.key} did not return p4")
            pieces.append(physical)
        result = torch.stack(pieces, dim=-2)
        if not torch.isfinite(result).all():
            raise ValueError("Invisible physical four-vectors are not finite")
        return result

    @staticmethod
    def _select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Select the candidate axis and turn NULL/MASK into NaN p4s."""
        batch, hypotheses, candidates, samples, components = values.shape
        if candidates <= 0:
            raise ValueError("A topology candidate pool is empty")
        indices = indices.to(device=values.device, dtype=torch.long)
        valid = (indices >= 0) & (indices < candidates)
        safe = indices.clamp(0, candidates - 1)
        gather = safe.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            batch, hypotheses, 1, samples, components
        )
        selected = values.gather(2, gather).squeeze(2)
        nan = torch.full_like(selected, float("nan"))
        return torch.where(valid.unsqueeze(-1).unsqueeze(-1), selected, nan)

    def compose(
        self,
        pools: Mapping[str, Mapping[str, torch.Tensor]],
        physics_vectors: torch.Tensor,
        node_candidate_indices: torch.Tensor,
        *,
        invisible_momenta: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
        invisible_four_vectors: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
        invisible_mass_shells: Mapping[str, object] | None = None,
        invisible_masses: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        if physics_vectors.ndim != 3 or physics_vectors.shape[-1] != 4:
            raise ValueError("physics_vectors must have shape [B,T,4]")
        if node_candidate_indices.ndim != 3:
            raise ValueError("node_candidate_indices must have shape [B,K,N]")
        batch_size, hypotheses, node_count = node_candidate_indices.shape
        if node_count != len(self.spec.dag_order):
            raise ValueError("Assignment node count disagrees with topology")
        if physics_vectors.shape[0] != batch_size:
            raise ValueError("physics_vectors and assignments disagree on batch size")
        posterior = self._invisible_p4(
            invisible_momenta,
            invisible_four_vectors,
            batch_size=batch_size,
            hypotheses=hypotheses,
            mass_shells=invisible_mass_shells,
            masses=invisible_masses,
        )
        samples = posterior.shape[2]
        candidate_values: dict[str, torch.Tensor] = {}
        for node_name in self.spec.dag_order:
            pool = pools[node_name]
            candidate_mask = pool.get("candidate_mask")
            if candidate_mask is None:
                raise KeyError(f"Candidate pool {node_name} has no candidate_mask")
            candidate_mask = candidate_mask.to(device=physics_vectors.device)
            if candidate_mask.ndim != 2 or candidate_mask.shape[0] != batch_size:
                raise ValueError(
                    f"Candidate mask for {node_name} must have shape [B,C] with B={batch_size}"
                )
            candidates = int(candidate_mask.shape[1])
            if candidates <= 0:
                raise ValueError(f"Candidate pool {node_name} is empty")
            values = torch.zeros(
                (batch_size, hypotheses, candidates, samples, 4),
                dtype=physics_vectors.dtype,
                device=physics_vectors.device,
            )
            node = self.spec.nodes[node_name]
            for role in node.leaf_roles:
                if role.invisible:
                    role_p4 = posterior[..., self.spec.invisible_roles.index(role), :]
                    values = values + role_p4.unsqueeze(2).expand_as(values)
                else:
                    indices = pool["leaf_indices"][role.key].to(
                        device=physics_vectors.device, dtype=torch.long
                    )
                    if indices.shape != (candidates, role.multiplicity):
                        raise ValueError(
                            f"Leaf indices for {role.key} must have shape "
                            f"[{candidates},{role.multiplicity}]"
                        )
                    gathered = physics_vectors[:, indices].sum(dim=-2)
                    values = values + gathered[:, None, :, None, :].expand_as(values)
            for role in node.composite_roles:
                child = candidate_values[role.child]
                child_indices = pool["child_indices"][role.key].to(
                    device=physics_vectors.device, dtype=torch.long
                )
                if child_indices.shape != (candidates,):
                    raise ValueError(f"Child indices for {role.key} must have shape [{candidates}]")
                safe = child_indices.clamp_min(0)
                gather = safe.view(1, 1, candidates, 1, 1).expand(
                    batch_size, hypotheses, candidates, samples, 4
                )
                child_values = child.gather(2, gather)
                child_valid = child_indices.ge(0).view(1, 1, candidates, 1, 1)
                values = values + torch.where(child_valid, child_values, torch.nan)
            valid_candidates = candidate_mask[:, None, :, None, None]
            values = torch.where(valid_candidates, values, torch.nan)
            candidate_values[node_name] = values

        objects: dict[str, torch.Tensor] = {}
        for role in self.spec.invisible_roles:
            objects[role.key] = posterior[..., self.spec.invisible_roles.index(role), :]
        for role in self.spec.visible_leaf_roles:
            pool = pools[role.node]
            candidates = int(pool["candidate_mask"].shape[1])
            indices = pool["leaf_indices"][role.key].to(
                device=physics_vectors.device, dtype=torch.long
            )
            gathered = physics_vectors[:, indices].sum(dim=-2)
            expanded = gathered[:, None, :, None, :].expand(
                batch_size, hypotheses, candidates, samples, 4
            )
            objects[role.key] = self._select(
                expanded, node_candidate_indices[..., self._node_position[role.node]]
            )
        for node_name, values in candidate_values.items():
            objects[node_name] = self._select(
                values, node_candidate_indices[..., self._node_position[node_name]]
            )
        for role in self.spec.composite_roles.values():
            objects[role.key] = objects[role.child]

        if self.spec.physics is not None:
            for name in self.spec.physics.order:
                composite = self.spec.physics.composite_map[name]
                terms = [objects[term] for term in composite.terms]
                objects[name] = torch.stack(terms, dim=0).sum(dim=0)
        return objects

    def compose_assignment_states(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Explicit alias used by offline analyses and future model code."""
        return self.compose(*args, **kwargs)


@dataclass(frozen=True)
class PhysicsFeatureResult:
    tensor: torch.Tensor
    feature_names: tuple[str, ...]
    physics_signature: str
    diagnostics: Mapping[str, float]


class PhysicsFeatureEngine:
    """Convert reconstructed p4 objects into named physical residuals."""

    def __init__(self, spec: PhysicsSpec):
        self.spec = spec

    @staticmethod
    def _mass(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if vector.shape[-1] != 4:
            raise ValueError("Physics objects must end in [E,px,py,pz]")
        mass2 = vector[..., 0].square() - vector[..., 1:].square().sum(dim=-1)
        return mass2.clamp_min(0).sqrt(), mass2 < 0

    def evaluate(
        self,
        objects: Mapping[str, torch.Tensor],
        *,
        observed_met: torch.Tensor | None = None,
    ) -> PhysicsFeatureResult:
        if not self.spec.observables:
            raise ValueError("Physics feature evaluation requires observables")
        pieces: list[torch.Tensor] = []
        feature_names: list[str] = []
        spacelike = []
        for observable in self.spec.observables:
            for name in observable.objects:
                if name not in objects:
                    raise KeyError(f"Physics observable {observable.name} needs {name}")
            if observable.kind == "invariant_mass":
                mass, bad = self._mass(objects[observable.objects[0]])
                pieces.append(mass - float(observable.reference))
                feature_names.append(observable.name)
                spacelike.append(bad)
            elif observable.kind == "mass_difference":
                first, bad_first = self._mass(objects[observable.objects[0]])
                second, bad_second = self._mass(objects[observable.objects[1]])
                pieces.append(first - second - float(observable.reference))
                feature_names.append(observable.name)
                spacelike.extend((bad_first, bad_second))
            elif observable.kind == "transverse_sum_residual":
                if observable.measurement != "MET":
                    raise ValueError(
                        f"Unsupported transverse measurement {observable.measurement!r}"
                    )
                if observed_met is None:
                    raise ValueError(f"{observable.name} requires observed MET")
                if observed_met.shape[-1] == 4:
                    measurement = observed_met[..., 1:3]
                elif observed_met.shape[-1] == 2:
                    measurement = observed_met
                else:
                    raise ValueError("Observed MET must end in [px,py] or [E,px,py,pz]")
                total = torch.stack(
                    [objects[name] for name in observable.objects], dim=0
                ).sum(dim=0)
                measurement = measurement.to(device=total.device, dtype=total.dtype)
                measurement = measurement.reshape(
                    measurement.shape[0], *([1] * (total.ndim - 2)), 2
                )
                residual = total[..., 1:3] - measurement
                pieces.extend((residual[..., 0], residual[..., 1]))
                feature_names.extend((f"{observable.name}.px", f"{observable.name}.py"))
            else:  # pragma: no cover - compiler prevents this
                raise ValueError(f"Unsupported physics observable {observable.kind}")
        tensor = torch.stack(pieces, dim=-1)
        diagnostics = {
            "nonfinite_fraction": float((~torch.isfinite(tensor)).float().mean().detach()),
            "spacelike_fraction": float(
                torch.stack([item.reshape(-1) for item in spacelike]).float().mean().detach()
            ) if spacelike else 0.0,
        }
        return PhysicsFeatureResult(
            tensor=tensor,
            feature_names=tuple(feature_names),
            physics_signature=self.spec.signature,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class PhysicsFeatureScaler:
    """Train-derived residual scales; means are retained but not subtracted by default."""

    feature_names: tuple[str, ...]
    mean: torch.Tensor
    scale: torch.Tensor
    topology_signature: str
    physics_signature: str
    source_data_identity: str
    split: str
    selection: str
    fit_count: int
    epsilon: float = 1.0e-6
    center: bool = False

    @classmethod
    def fit(
        cls,
        residuals: torch.Tensor,
        feature_names: Sequence[str],
        *,
        topology_signature: str,
        physics_signature: str,
        source_data_identity: str,
        split: str = "train",
        selection: str = "correct_assignment_truth_invisible",
        epsilon: float = 1.0e-6,
        center: bool = False,
        row_mask: torch.Tensor | None = None,
    ) -> "PhysicsFeatureScaler":
        if split != "train":
            raise ValueError("PhysicsFeatureScaler may only be fitted on split='train'")
        if selection != "correct_assignment_truth_invisible":
            raise ValueError(
                "PhysicsFeatureScaler requires the correct-assignment truth-invisible selection"
            )
        if residuals.ndim < 2:
            raise ValueError("Physics residuals must have a feature dimension")
        names = tuple(feature_names)
        if residuals.shape[-1] != len(names) or len(set(names)) != len(names):
            raise ValueError("Feature names do not match residual dimensions")
        rows = residuals.reshape(-1, residuals.shape[-1])
        if row_mask is not None:
            if row_mask.shape != residuals.shape[:-1]:
                raise ValueError(
                    "PhysicsFeatureScaler row_mask must match residual leading dimensions"
                )
            row_mask = row_mask.reshape(-1).to(device=residuals.device, dtype=torch.bool)
            rows = rows[row_mask]
        if rows.shape[0] == 0:
            raise ValueError("No finite training residuals were available for scaling")
        if not torch.isfinite(rows).all():
            raise ValueError(
                "PhysicsFeatureScaler received non-finite selected training residuals"
            )
        epsilon = float(epsilon)
        if epsilon <= 0:
            raise ValueError("Scaler epsilon must be positive")
        mean = rows.mean(dim=0)
        scale = rows.std(dim=0, unbiased=False).clamp_min(epsilon)
        return cls(
            names,
            mean.detach().cpu(),
            scale.detach().cpu(),
            str(topology_signature),
            str(physics_signature),
            str(source_data_identity),
            split,
            selection,
            int(rows.shape[0]),
            epsilon,
            bool(center),
        )

    def transform(
        self,
        residuals: torch.Tensor,
        *,
        feature_names: Sequence[str] | None = None,
        topology_signature: str | None = None,
        physics_signature: str | None = None,
    ) -> torch.Tensor:
        if residuals.shape[-1] != len(self.feature_names):
            raise ValueError("Residual feature count disagrees with PhysicsFeatureScaler")
        if feature_names is not None and tuple(feature_names) != self.feature_names:
            raise ValueError("Residual feature names disagree with PhysicsFeatureScaler")
        if (
            topology_signature is not None
            and topology_signature != self.topology_signature
        ):
            raise ValueError("Topology signature disagrees with PhysicsFeatureScaler")
        if physics_signature is not None and physics_signature != self.physics_signature:
            raise ValueError("Physics signature disagrees with PhysicsFeatureScaler")
        mean = self.mean.to(device=residuals.device, dtype=residuals.dtype)
        scale = self.scale.to(device=residuals.device, dtype=residuals.dtype)
        return (residuals - mean if self.center else residuals) / scale

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "topology_signature": self.topology_signature,
            "physics_signature": self.physics_signature,
            "source_data_identity": self.source_data_identity,
            "split": self.split,
            "selection": self.selection,
            "fit_count": self.fit_count,
            "epsilon": self.epsilon,
            "center": self.center,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "PhysicsFeatureScaler":
        required = {
            "feature_names", "mean", "scale", "topology_signature",
            "physics_signature", "source_data_identity", "split", "selection",
            "fit_count",
        }
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(f"PhysicsFeatureScaler is missing fields {missing}")
        names = tuple(values["feature_names"])
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("PhysicsFeatureScaler feature_names are invalid")
        mean = torch.as_tensor(values["mean"], dtype=torch.float32)
        scale = torch.as_tensor(values["scale"], dtype=torch.float32)
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or mean.shape[0] != len(names)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError("PhysicsFeatureScaler mean/scale values are invalid")
        split = str(values["split"])
        selection = str(values["selection"])
        if split != "train" or selection != "correct_assignment_truth_invisible":
            raise ValueError("PhysicsFeatureScaler provenance is not a train truth fit")
        epsilon = float(values.get("epsilon", 1.0e-6))
        fit_count = int(values["fit_count"])
        if not math.isfinite(epsilon) or epsilon <= 0 or fit_count <= 0:
            raise ValueError("PhysicsFeatureScaler fit metadata is invalid")
        return cls(
            names,
            mean,
            scale,
            str(values["topology_signature"]),
            str(values["physics_signature"]),
            str(values["source_data_identity"]),
            split,
            selection,
            fit_count,
            epsilon,
            bool(values.get("center", False)),
        )

    def save(self, path: str | Path) -> None:
        torch.save(self.as_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "PhysicsFeatureScaler":
        return cls.from_dict(torch.load(Path(path), map_location="cpu", weights_only=True))

    @property
    def identity_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class PosteriorReduction:
    mean: torch.Tensor
    std: torch.Tensor
    rms: torch.Tensor
    analytic_log_evidence: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        return torch.cat(
            (self.mean, self.std, self.rms, self.analytic_log_evidence.unsqueeze(-1)),
            dim=-1,
        )


class PosteriorReducer:
    """Permutation-invariant reduction over posterior samples."""

    def reduce(
        self,
        features: torch.Tensor,
        valid_samples: torch.Tensor | None = None,
        valid_hypotheses: torch.Tensor | None = None,
    ) -> PosteriorReduction:
        if features.ndim != 4:
            raise ValueError("Posterior features must have shape [B,K,M,F]")
        finite = torch.isfinite(features).all(dim=-1)
        if valid_samples is None:
            valid = finite
        else:
            if valid_samples.ndim == 2:
                valid_samples = valid_samples.unsqueeze(1)
            if valid_samples.shape != features.shape[:3]:
                raise ValueError("valid_samples must have shape [B,K,M]")
            valid = valid_samples.to(torch.bool) & finite
        any_samples = valid.any(dim=2)
        if valid_hypotheses is None:
            valid_hypotheses = torch.ones(
                features.shape[:2], dtype=torch.bool, device=features.device
            )
        elif valid_hypotheses.shape != features.shape[:2]:
            raise ValueError("valid_hypotheses must have shape [B,K]")
        else:
            valid_hypotheses = valid_hypotheses.to(torch.bool)
        if (valid_hypotheses & ~any_samples).any():
            raise ValueError(
                "Each valid hypothesis must have at least one finite posterior sample"
            )
        safe = torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))
        count = valid.sum(dim=2).clamp_min(1).to(features.dtype)
        mean = safe.sum(dim=2) / count.unsqueeze(-1)
        second = safe.square().sum(dim=2) / count.unsqueeze(-1)
        variance = (second - mean.square()).clamp_min(0)
        std = variance.sqrt()
        rms = second.clamp_min(0).sqrt()
        chi2 = safe.square().sum(dim=-1)
        log_terms = torch.where(
            valid,
            -0.5 * chi2,
            torch.full_like(chi2, -torch.inf),
        )
        evidence = torch.logsumexp(log_terms, dim=2) - count.log()
        invalid = ~valid_hypotheses
        mean = mean.masked_fill(invalid.unsqueeze(-1), 0.0)
        std = std.masked_fill(invalid.unsqueeze(-1), 0.0)
        rms = rms.masked_fill(invalid.unsqueeze(-1), 0.0)
        evidence = evidence.masked_fill(invalid, -torch.inf)
        return PosteriorReduction(mean, std, rms, evidence)


class AnalyticPhysicsScorer:
    """Interpretable compatibility score, not a normalised event likelihood."""

    normalised_likelihood = False

    def __call__(self, reduction: PosteriorReduction) -> torch.Tensor:
        return reduction.analytic_log_evidence

    @property
    def metadata(self) -> dict[str, object]:
        return {"type": "analytic_chi2_log_evidence", "normalised_likelihood": False}


class LearnedHypothesisScorer(nn.Module):
    """Small residual scorer for already-computed reduced physics features."""

    def __init__(self, input_features: int, hidden: int = 32):
        super().__init__()
        if input_features <= 0 or hidden <= 0:
            raise ValueError("Learned scorer dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, reduced_features: torch.Tensor) -> torch.Tensor:
        return self.network(reduced_features).squeeze(-1)


class MatchedVisibleOnlyScorer(LearnedHypothesisScorer):
    """Capacity-matched control using only pre-invisible visible summaries.

    The four inputs are, in order, visible log score, visible probability,
    event assignment entropy, and the top-1/top-2 margin.  Keeping this input
    contract explicit prevents a control study from accidentally receiving
    neutrino-derived physics features.
    """

    input_names = (
        "visible_log_score",
        "visible_probability",
        "assignment_entropy",
        "top1_top2_margin",
    )

    def __init__(self, hidden: int = 32):
        super().__init__(len(self.input_names), hidden=hidden)

    def forward(self, visible_features: torch.Tensor) -> torch.Tensor:
        if visible_features.shape[-1] != len(self.input_names):
            raise ValueError(
                "MatchedVisibleOnlyScorer expects exactly four visible-only features: "
                f"{self.input_names}"
            )
        return super().forward(visible_features)


def reuse_unconditioned_posterior(
    posterior: torch.Tensor, hypotheses: int
) -> tuple[torch.Tensor, int]:
    """Broadcast one q(nu|X) across K assignments without K generator calls."""
    if posterior.ndim < 2 or hypotheses <= 0:
        raise ValueError("posterior must have [B,M,...] and hypotheses must be positive")
    return posterior.unsqueeze(1).expand(-1, hypotheses, *posterior.shape[1:]), 1


@dataclass(frozen=True)
class HypothesisEvaluation:
    visible_log_score: torch.Tensor
    visible_probability: torch.Tensor
    feature_result: PhysicsFeatureResult | None
    reduction: PosteriorReduction | None
    analytic_score: torch.Tensor
    learned_delta: torch.Tensor
    final_log_score: torch.Tensor
    final_probability: torch.Tensor
    ranking: torch.Tensor
    objects: Mapping[str, torch.Tensor]


class HypothesisEvaluator:
    """One orchestration point for physics-aware assignment hypothesis scoring."""

    def __init__(
        self,
        spec: TopologySpec,
        *,
        scaler: PhysicsFeatureScaler | None = None,
        mode: str = "none",
        alpha: float = 1.0,
        scorer: nn.Module | None = None,
    ):
        if mode not in {"none", "physics", "learned"}:
            raise ValueError("HypothesisEvaluator mode must be none, physics, or learned")
        if spec.physics is None and mode != "none":
            raise ValueError("A physics spec is required for physics-aware evaluation")
        if mode != "none" and scaler is None:
            raise ValueError("Physics-aware evaluation requires a train-derived scaler")
        self.spec = spec
        self.scaler = scaler
        self.mode = mode
        self.alpha = float(alpha)
        self.kinematics = TopologyKinematics(spec)
        self.features = None if spec.physics is None else PhysicsFeatureEngine(spec.physics)
        self.reducer = PosteriorReducer()
        self.analytic = AnalyticPhysicsScorer()
        self.scorer = scorer
        if mode == "learned" and scorer is None:
            raise ValueError("Learned hypothesis evaluation requires a scorer")

    def evaluate(
        self,
        visible_log_score: torch.Tensor,
        pools: Mapping[str, Mapping[str, torch.Tensor]],
        physics_vectors: torch.Tensor,
        node_candidate_indices: torch.Tensor,
        *,
        invisible_momenta: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
        invisible_four_vectors: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
        observed_met: torch.Tensor | None = None,
        valid_hypotheses: torch.Tensor | None = None,
        invisible_mass_shells: Mapping[str, object] | None = None,
        invisible_masses: Mapping[str, torch.Tensor] | None = None,
    ) -> HypothesisEvaluation:
        if visible_log_score.ndim != 2:
            raise ValueError("visible_log_score must have shape [B,K]")
        batch_size, hypotheses = visible_log_score.shape
        if valid_hypotheses is None:
            valid_hypotheses = torch.isfinite(visible_log_score)
        if valid_hypotheses.shape != visible_log_score.shape:
            raise ValueError("valid_hypotheses must have shape [B,K]")
        valid_hypotheses = valid_hypotheses.to(
            device=visible_log_score.device, dtype=torch.bool
        )
        if (~valid_hypotheses.any(dim=1)).any():
            raise ValueError("Each event must have at least one valid assignment hypothesis")
        visible_score = visible_log_score.masked_fill(~valid_hypotheses, -torch.inf)
        visible_probability = torch.softmax(visible_score, dim=1)
        objects: Mapping[str, torch.Tensor] = {}
        feature_result = None
        reduction = None
        analytic_score = torch.zeros_like(visible_score)
        learned_delta = torch.zeros_like(visible_score)
        if self.mode != "none":
            objects = self.kinematics.compose(
                pools,
                physics_vectors,
                node_candidate_indices,
                invisible_momenta=invisible_momenta,
                invisible_four_vectors=invisible_four_vectors,
                invisible_mass_shells=invisible_mass_shells,
                invisible_masses=invisible_masses,
            )
            feature_result = self.features.evaluate(objects, observed_met=observed_met)
            scaled = self.scaler.transform(
                feature_result.tensor,
                feature_names=feature_result.feature_names,
                topology_signature=self.spec.signature,
                physics_signature=feature_result.physics_signature,
            )
            valid_samples = valid_hypotheses.unsqueeze(-1).expand(
                -1, -1, scaled.shape[2]
            )
            reduction = self.reducer.reduce(
                scaled,
                valid_samples=valid_samples,
                valid_hypotheses=valid_hypotheses,
            )
            raw_analytic = self.analytic(reduction)
            if not torch.isfinite(raw_analytic[valid_hypotheses]).all():
                raise ValueError(
                    "Physics compatibility produced a non-finite score for a valid hypothesis"
                )
            analytic_score = raw_analytic.masked_fill(
                ~valid_hypotheses, -torch.inf
            )
            if self.mode == "physics":
                # Keep the invalid support out of the correction before applying
                # alpha.  In particular, ``0 * -inf`` would otherwise create a
                # NaN for the natural visible-only (alpha=0) control.
                safe_analytic = torch.where(
                    valid_hypotheses, raw_analytic, torch.zeros_like(raw_analytic)
                )
                learned_delta = self.alpha * safe_analytic
            else:
                scorer_input = torch.nan_to_num(
                    reduction.as_tensor(), nan=0.0, posinf=0.0, neginf=0.0
                )
                learned_delta = self.scorer(scorer_input)
                learned_delta = learned_delta.masked_fill(~valid_hypotheses, 0.0)
        final_score = (visible_score + learned_delta).masked_fill(
            ~valid_hypotheses, -torch.inf
        )
        final_probability = torch.softmax(final_score, dim=1)
        ranking = torch.argsort(final_score, dim=1, descending=True, stable=True)
        return HypothesisEvaluation(
            visible_log_score=visible_score,
            visible_probability=visible_probability,
            feature_result=feature_result,
            reduction=reduction,
            analytic_score=analytic_score,
            learned_delta=learned_delta,
            final_log_score=final_score,
            final_probability=final_probability,
            ranking=ranking,
            objects=objects,
        )


__all__ = [
    "AnalyticPhysicsScorer",
    "HypothesisEvaluation",
    "HypothesisEvaluator",
    "LearnedHypothesisScorer",
    "MatchedVisibleOnlyScorer",
    "PhysicsFeatureEngine",
    "PhysicsFeatureResult",
    "PhysicsFeatureScaler",
    "PhysicsSpec",
    "PosteriorReducer",
    "PosteriorReduction",
    "TopologyKinematics",
    "compile_physics",
    "reuse_unconditioned_posterior",
]
