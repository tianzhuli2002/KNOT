#!/usr/bin/env python3
"""Run bounded coordinate and explicit-source CFM checks on validation truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

from joint_event_reconstruction.models.invisible.coordinates import MetResidualCoordinates
from joint_event_reconstruction.models.invisible.couplings import IndependentCoupling
from joint_event_reconstruction.models.invisible.flow_matching import ContinuousFlowMatching
from joint_event_reconstruction.models.partonic import (
    four_vectors_from_partonic_features,
    partonic_features,
)


PARENT_ROLES = ("top", "antitop")
NEUTRINO_ROLES = ("top.nu", "antitop.nu")


class _FixedVelocity(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, state, time, visible, mask, parent, **kwargs):
        del time, visible, mask, parent, kwargs
        return torch.full_like(state, self.value)


class _ZeroVelocity(nn.Module):
    def forward(self, state, time, visible, mask, parent, **kwargs):
        del time, visible, mask, parent, kwargs
        return torch.zeros_like(state)


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max()) if left.numel() else 0.0


def run(event_table: Path, output_path: Path) -> dict:
    if not event_table.is_file():
        raise FileNotFoundError(f"Validation event table is missing: {event_table}")
    with h5py.File(event_table, "r") as source:
        if source.attrs.get("schema") != "realnu_event_metrics_v1":
            raise ValueError("Unexpected accepted R2 event-table schema")
        if str(source.attrs.get("sealed_test_used", "")).lower() != "false":
            raise ValueError("Validation source does not establish sealed_test_used=false")
        if source.attrs.get("parent_role_order") != '["top", "antitop"]':
            raise ValueError("Parent truth role order is not [top, antitop]")
        if source.attrs.get("neutrino_role_order") != '["top.nu", "antitop.nu"]':
            raise ValueError("Neutrino truth role order is not [top.nu, antitop.nu]")
        ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        parent_values = torch.from_numpy(np.asarray(source["truth/parents"][:], dtype=np.float64))
        neutrino_values = torch.from_numpy(np.asarray(source["truth/neutrinos"][:], dtype=np.float64))
        met_values = torch.from_numpy(np.asarray(source["truth/met"][:], dtype=np.float64))
        if len(np.unique(ids)) != len(ids) or not np.isfinite(parent_values.numpy()).all():
            raise ValueError("Validation truth contains duplicate IDs or invalid parent vectors")
        if not np.isfinite(neutrino_values.numpy()).all() or not np.isfinite(met_values.numpy()).all():
            raise ValueError("Validation neutrino truth or MET contains non-finite values")

    encoded_parent = partonic_features(parent_values)
    decoded_parent = four_vectors_from_partonic_features(encoded_parent)
    pair = decoded_parent[:, 0] + decoded_parent[:, 1]
    pair_mass_squared = pair[:, 0].square() - pair[:, 1:].square().sum(dim=-1)
    if (pair[:, 0] <= 0).any() or (pair_mass_squared < -1e-8).any():
        raise ValueError("Decoded parent pair is not future-timelike")
    parent_delta = (decoded_parent - parent_values).abs()
    if not torch.allclose(decoded_parent, parent_values, rtol=2e-10, atol=2e-7):
        raise AssertionError("Parton parent coordinate round trip failed")

    coordinates = MetResidualCoordinates(NEUTRINO_ROLES)
    encoded_neutrino = coordinates.encode(neutrino_values, met_values)
    decoded_neutrino = coordinates.decode(encoded_neutrino, met_values)
    if not torch.allclose(decoded_neutrino, neutrino_values, rtol=1e-12, atol=1e-9):
        raise AssertionError("MET-residual neutrino coordinate round trip failed")
    met_closure = decoded_neutrino[..., :2].sum(dim=-2) + encoded_neutrino[..., -1, :2]
    if not torch.allclose(met_closure, met_values, rtol=1e-12, atol=1e-9):
        raise AssertionError("MET-residual coordinate closure failed")

    torch.manual_seed(117)
    target = torch.randn(16, 8, dtype=torch.float32)
    explicit_source = torch.randn_like(target)
    explicit_time = torch.linspace(0.05, 0.95, len(target), dtype=target.dtype)
    visible = torch.zeros(16, 2, 4, dtype=target.dtype)
    mask = torch.zeros(16, 2, dtype=torch.bool)
    parent_context = torch.zeros(16, 2, 4, dtype=target.dtype)
    flow = ContinuousFlowMatching(
        _ZeroVelocity(), coupling=IndependentCoupling(), active_mask=torch.ones(8, dtype=torch.bool)
    )
    loss_result = flow.loss(
        target,
        visible,
        mask,
        parent_context,
        source_state=explicit_source,
        flow_time=explicit_time,
    )
    expected_state = explicit_source * (1.0 - explicit_time[:, None]) + target * explicit_time[:, None]
    expected_velocity = target - explicit_source
    if not torch.equal(loss_result["source"], explicit_source):
        raise AssertionError("Explicit source was not preserved by independent CFM coupling")
    if not torch.allclose(loss_result["state"], expected_state, rtol=0.0, atol=2e-7):
        raise AssertionError("CFM linear path did not use the explicit source and time")
    if not torch.equal(loss_result["target_velocity"], expected_velocity):
        raise AssertionError("CFM target velocity differs from the linear path derivative")

    expected_velocity_value = 1.75
    solver_results = {}
    fixed_flow = ContinuousFlowMatching(
        _FixedVelocity(expected_velocity_value),
        coupling=IndependentCoupling(),
        active_mask=torch.ones(8, dtype=torch.bool),
    )
    initial = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).reshape(4, 8)
    for solver, step_counts in (("euler", (1, 8)), ("heun", (1, 2, 8))):
        for steps in step_counts:
            endpoint, nfe = fixed_flow.sample(
                tuple(initial.shape),
                visible[:4],
                mask[:4],
                parent_context[:4],
                steps=steps,
                solver=solver,
                initial_state=initial,
            )
            expected = initial + expected_velocity_value
            if not torch.allclose(endpoint, expected, rtol=0.0, atol=3e-7):
                raise AssertionError(f"{solver}{steps} failed the constant-velocity endpoint")
            solver_results[f"{solver}_{steps}"] = {
                "nfe": nfe,
                "max_abs_endpoint_error": _max_abs(endpoint, expected),
            }

    result = {
        "status": "passed",
        "event_count": len(ids),
        "event_id_unique": True,
        "sealed_test_used": False,
        "truth_roles": {"parents": list(PARENT_ROLES), "neutrinos": list(NEUTRINO_ROLES)},
        "parent_coordinate_roundtrip": {
            "representation": "px,py,pz,log1p(mass)",
            "max_abs_four_vector_error_GeV": float(parent_delta.max()),
            "roles_preserved": True,
            "positive_energy": bool((decoded_parent[..., 0] > 0).all()),
            "future_timelike_pairs": bool((pair_mass_squared >= -1e-8).all()),
        },
        "neutrino_coordinate_roundtrip": {
            "representation": coordinates.coordinate_type,
            "token_order": list(coordinates.token_names),
            "max_abs_momentum_error_GeV": _max_abs(decoded_neutrino, neutrino_values),
            "max_abs_MET_closure_error_GeV": _max_abs(met_closure, met_values),
        },
        "explicit_source_time_fixture": {
            "batch": len(target),
            "state_dimension": int(target.shape[1]),
            "source_max_abs_error": _max_abs(loss_result["source"], explicit_source),
            "state_max_abs_error": _max_abs(loss_result["state"], expected_state),
            "velocity_max_abs_error": _max_abs(loss_result["target_velocity"], expected_velocity),
        },
        "constant_velocity_solver_toy": solver_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.event_table, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
