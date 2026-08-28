#!/usr/bin/env python3
"""File-free CPU/CUDA smoke of every maintained reconstruction formulation.

This is a software/mechanism test only. It constructs topology-consistent random visible
objects and invisible truth in memory, then exercises forward/backward, prediction HDF5,
evaluation and plotting. No metric from this script is a physics-performance result.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.assignment_semantics import (
    NON_NULL_SUPPORT,
    NULL_INCLUDING_SUPPORT,
)
from joint_event_reconstruction.configuration import validate_experiment
from joint_event_reconstruction.data.invisible import InvisibleScaler
from joint_event_reconstruction.data.structured import StructuredCollator
from joint_event_reconstruction.evaluation import evaluate_predictions
from joint_event_reconstruction.evaluation_discrete import evaluate_discrete_predictions
from joint_event_reconstruction.evaluation_joint import evaluate_joint_predictions
from joint_event_reconstruction.m0.data.module import build_token_schema
from joint_event_reconstruction.plotting import plot_comparison, plot_prediction
from joint_event_reconstruction.prediction import predict_hdf5
from joint_event_reconstruction.prediction_discrete import predict_discrete_flow_hdf5
from joint_event_reconstruction.prediction_joint import predict_joint_flow_hdf5
from joint_event_reconstruction.topology.assignments import AssignmentSpace
from joint_event_reconstruction.topology.compiler import load_topology
from joint_event_reconstruction.training.joint import JointEventModule
from scripts.smoke_all import MODES, run_mode as run_factorized_mode
from scripts.smoke_flows import FLOW_MODES, configure as configure_flow
from scripts.smoke_flows import run_mode as run_flow_mode


CONTRACTS = {
    "synthetic_contract": {"minimum_invisible": 1, "minimum_repeat": 2},
    "synthetic_multinu": {"minimum_invisible": 3, "minimum_repeat": 3},
}


def _synthetic_items(
    experiment, visible, *, events: int, seed: int
) -> tuple[list[dict], object]:
    if events < 2:
        raise ValueError("Synthetic smoke needs at least two events for finite scaling")
    schema = build_token_schema(visible.dataset)
    spec = load_topology(experiment.topology_config)
    generator = torch.Generator().manual_seed(seed)

    source_positions: dict[str, torch.Tensor] = {}
    type_ids = []
    cursor = 0
    for source, plan in schema.sources.items():
        width = 1 if source == "GLOBAL" else int(plan.num_particles or 0)
        if width <= 0:
            continue
        positions = torch.arange(cursor, cursor + width, dtype=torch.long)
        source_positions[source] = positions
        type_ids.extend([int(schema.source_type_ids[source])] * width)
        cursor += width
    if cursor == 0:
        raise RuntimeError("Visible schema produced no synthetic token slots")

    label_slots: dict[tuple[str, int], int] = {}
    by_source: dict[str, set[int]] = {}
    for role in spec.visible_leaf_roles:
        by_source.setdefault(role.source, set()).update(
            int(index) for index in role.truth_indices
        )
    for source, labels in by_source.items():
        if source not in source_positions:
            raise RuntimeError(
                f"Topology source {source} is absent from visible token schema"
            )
        positions = source_positions[source]
        if len(labels) > len(positions):
            raise RuntimeError(
                f"Synthetic schema has {len(positions)} {source} slots but topology "
                f"needs {len(labels)} distinct truth objects"
            )
        for local, label in enumerate(sorted(labels)):
            label_slots[(source, label)] = int(positions[local])

    common_type_ids = torch.tensor(type_ids, dtype=torch.long)
    items = []
    for event in range(events):
        tokens = torch.randn((cursor, schema.dim), generator=generator)
        tokens += event / max(events, 1) * 0.05
        raw_labels = torch.full((cursor,), -100.0, dtype=torch.float32)
        for (_, label), position in label_slots.items():
            raw_labels[position] = float(label)
        items.append(
            {
                "tokens": tokens,
                "type_ids": common_type_ids.clone(),
                "padding_mask": torch.zeros(cursor, dtype=torch.bool),
                "labels": torch.tensor(0.0, dtype=torch.float32),
                "event_index": torch.tensor(event, dtype=torch.long),
                "reco_raw_labels": raw_labels,
                "reco_source_ids": common_type_ids.to(torch.int8).clone(),
            }
        )
    return items, schema


def _synthetic_physics_vectors(batch: dict, schema, *, seed: int) -> torch.Tensor:
    """Build finite `(E,px,py,pz)` tokens solely to exercise physics I/O paths."""
    generator = torch.Generator().manual_seed(seed)
    batch_size, token_count = batch["type_ids"].shape
    momentum = 20.0 * torch.randn(
        (batch_size, token_count, 3), generator=generator
    )
    masses = torch.full((batch_size, token_count), 0.5)
    met_type = schema.source_type_ids.get("MET")
    if met_type is not None:
        met = batch["type_ids"] == int(met_type)
        momentum[..., 2] = torch.where(
            met, torch.zeros_like(momentum[..., 2]), momentum[..., 2]
        )
        masses = torch.where(met, torch.zeros_like(masses), masses)
    energy = torch.sqrt(momentum.square().sum(dim=-1) + masses.square())
    return torch.cat((energy.unsqueeze(-1), momentum), dim=-1)


def build_synthetic_batch(experiment, *, events: int, device: str):
    visible = OmegaConf.load(experiment.visible_config)
    items, schema = _synthetic_items(
        experiment,
        visible,
        events=events,
        seed=int(experiment.experiment.seed),
    )
    collator = StructuredCollator(
        experiment.topology_config,
        dict(schema.source_type_ids),
        build_assignments=True,
        max_assignment_states=int(experiment.data.max_assignment_states),
    )
    batch = collator(items)
    spec = collator.spec

    generator = torch.Generator().manual_seed(int(experiment.experiment.seed) + 17)
    invisible_truth = torch.randn(
        (events, len(spec.invisible_roles), 3), generator=generator
    )
    role_offsets = torch.arange(
        len(spec.invisible_roles), dtype=invisible_truth.dtype
    ).reshape(1, -1, 1)
    invisible_truth = invisible_truth + 0.25 * role_offsets
    scaler = InvisibleScaler.fit(
        invisible_truth, tuple(role.key for role in spec.invisible_roles)
    )
    batch["invisible_truth"] = invisible_truth
    batch["invisible_truth_scaled"] = scaler.transform(invisible_truth)
    batch["physics_vectors"] = _synthetic_physics_vectors(
        batch, schema, seed=int(experiment.experiment.seed) + 31
    )

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    return visible, move(batch), scaler


class _SyntheticPredictionData:
    """Minimal data facade used by the maintained HDF5 prediction writers."""

    def __init__(self, batch, visible, source_path: Path):
        self._batch = batch
        self.schema = build_token_schema(visible.dataset)
        self.h5_path = source_path

    def train_dataloader(self):
        return [self._batch]

    def val_dataloader(self):
        return [self._batch]

    def test_dataloader(self):
        return [self._batch]


def _factorized_cfm_config(base, conditioning: str):
    config = configure_flow(base, "cfm_no_assignment")
    config.model.invisible.conditioning.assignment = conditioning
    config.model.invisible.conditioning.allow_truth_oracle = False
    config.inference.top_k_assignments = 2
    config.inference.posterior_samples = 2
    validate_experiment(config)
    return config


def _assert_plot_outputs(paths: list[Path], *, name: str) -> None:
    if not paths or not all(
        path.is_file() and path.stat().st_size > 0 for path in paths
    ):
        raise RuntimeError(f"{name} plotting did not produce non-empty outputs")
    if not any(path.name == "metrics.json" for path in paths):
        raise RuntimeError(f"{name} plotting did not write metrics.json")
    if not any(path.suffix == ".png" for path in paths):
        raise RuntimeError(f"{name} plotting did not write a PNG diagnostic")


def _assert_support(metrics: dict, expected: str, name: str) -> None:
    observed = metrics["assignment_support"]["name"]
    if observed != expected:
        raise RuntimeError(
            f"{name} assignment support changed: expected={expected}, observed={observed}"
        )


def _prediction_roundtrip(base, visible, batch, scaler, device):
    """Exercise every maintained posterior schema through predict/evaluate/plot."""
    with tempfile.TemporaryDirectory(prefix="jer-smoke-") as directory:
        directory = Path(directory)
        source_path = directory / "source.h5"
        with h5py.File(source_path, "w") as source:
            source.create_dataset(
                "EVENT_ID", data=batch["event_index"].detach().cpu().numpy()
            )
        data = _SyntheticPredictionData(batch, visible, source_path)
        prediction_paths = {}
        summaries = {}

        for conditioning in ("none", "hypothesis"):
            config = _factorized_cfm_config(base, conditioning)
            module = JointEventModule(
                config,
                visible,
                invisible_scaler=scaler,
                synthetic_data=True,
            ).to(device)
            path = directory / f"cfm_{conditioning}.h5"
            predict_hdf5(
                module,
                data,
                scaler,
                path,
                split="test",
                top_k=2,
                posterior_samples=2,
                device=device,
            )
            metrics = evaluate_predictions(path)
            if metrics["events"] != len(batch["tokens"]):
                raise RuntimeError(
                    f"CFM {conditioning} prediction round-trip changed event count"
                )
            if metrics["invisible_assignment_conditioning"] != conditioning:
                raise RuntimeError(
                    f"CFM {conditioning} evaluation lost conditioning semantics"
                )
            _assert_support(metrics, NON_NULL_SUPPORT, f"CFM {conditioning}")
            prediction_paths[f"cfm_{conditioning}"] = path
            summaries[f"cfm_{conditioning}"] = {
                "events": metrics["events"],
                "samples": metrics["posterior_samples"],
                "top_k": metrics["top_k_assignments"],
                "conditioning": metrics["invisible_assignment_conditioning"],
                "assignment_support": metrics["assignment_support"]["name"],
            }

        dfm_config = configure_flow(base, "dfm_assignment")
        dfm = JointEventModule(
            dfm_config,
            visible,
            invisible_scaler=None,
            synthetic_data=True,
        ).to(device)
        dfm_path = directory / "dfm.h5"
        predict_discrete_flow_hdf5(
            dfm,
            data,
            dfm_path,
            split="test",
            posterior_samples=2,
            steps=2,
            device=device,
        )
        dfm_metrics = evaluate_discrete_predictions(dfm_path)
        if dfm_metrics["events"] != len(batch["tokens"]):
            raise RuntimeError("DFM prediction round-trip changed event count")
        _assert_support(dfm_metrics, NULL_INCLUDING_SUPPORT, "DFM")
        prediction_paths["dfm"] = dfm_path
        summaries["dfm"] = {
            "events": dfm_metrics["events"],
            "samples": dfm_metrics["posterior_samples"],
            "steps": dfm_metrics["flow_steps"],
            "assignment_support": dfm_metrics["assignment_support"]["name"],
        }

        joint_config = configure_flow(base, "joint_flow")
        joint = JointEventModule(
            joint_config,
            visible,
            invisible_scaler=scaler,
            synthetic_data=True,
        ).to(device)
        joint_path = directory / "joint.h5"
        predict_joint_flow_hdf5(
            joint,
            data,
            scaler,
            joint_path,
            split="test",
            posterior_samples=2,
            steps=2,
            sample_chunk_size=2,
            device=device,
        )
        joint_metrics = evaluate_joint_predictions(joint_path)
        if joint_metrics["events"] != len(batch["tokens"]):
            raise RuntimeError("Joint prediction round-trip changed event count")
        _assert_support(joint_metrics, NULL_INCLUDING_SUPPORT, "joint flow")
        prediction_paths["joint"] = joint_path
        summaries["joint"] = {
            "events": joint_metrics["events"],
            "samples": joint_metrics["posterior_samples"],
            "steps": joint_metrics["flow_steps"],
            "assignment_support": joint_metrics["assignment_support"]["name"],
        }

        for name, path in prediction_paths.items():
            written = plot_prediction(
                path,
                directory / f"plots_{name}",
                formats=("png",),
            )
            _assert_plot_outputs(written, name=name)
            summaries[name]["plot_files"] = len(written)

        comparison = plot_comparison(
            [
                prediction_paths["cfm_none"],
                prediction_paths["cfm_hypothesis"],
                prediction_paths["joint"],
            ],
            directory / "comparison",
            labels=["CFM-none", "CFM-hypothesis", "joint-flow"],
            formats=("png",),
        )
        if not comparison or not all(
            path.is_file() and path.stat().st_size > 0 for path in comparison
        ):
            raise RuntimeError("Prediction comparison plotting produced no outputs")
        summaries["comparison_plot_files"] = len(comparison)
        return summaries


def run_synthetic_smoke(
    experiment_path: Path,
    *,
    device: str = "cpu",
    events: int = 4,
    factorized: bool = True,
    flows: bool = True,
):
    torch.set_num_threads(1)
    base = OmegaConf.load(experiment_path)
    visible, batch, scaler = build_synthetic_batch(
        base, events=events, device=device
    )
    results = {"factorized": {}, "flows": {}}

    if factorized:
        for mode in MODES:
            results["factorized"][mode] = {
                "status": "PASS",
                **run_factorized_mode(base, visible, batch, scaler, mode, device),
            }
    if flows:
        for mode in FLOW_MODES:
            results["flows"][mode] = {
                "status": "PASS",
                **run_flow_mode(base, visible, batch, scaler, mode, device),
            }
        results["prediction_roundtrip"] = {
            "status": "PASS",
            **_prediction_roundtrip(base, visible, batch, scaler, device),
        }

    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def run_topology_contract(
    topology_name: str, *, device: str = "cpu", events: int = 4
):
    """Prove topology and Rnu genericity without a process-name model branch."""
    requirements = CONTRACTS[topology_name]
    base = copy.deepcopy(OmegaConf.load(ROOT / "configs/experiments/ttbar1L.yaml"))
    base.experiment.name = topology_name
    base.topology_config = str(ROOT / "configs/topology" / f"{topology_name}.yaml")
    base.visible_config = str(ROOT / "configs/m0_ttbar1L.yaml")
    visible, batch, scaler = build_synthetic_batch(
        base, events=events, device=device
    )
    spec = load_topology(base.topology_config)
    space = AssignmentSpace.from_structured(batch["structured"])

    if len(spec.roots) < 2:
        raise RuntimeError("Synthetic topology contract must exercise multiple roots")
    if max(role.multiplicity for role in spec.visible_leaf_roles) < int(
        requirements["minimum_repeat"]
    ):
        raise RuntimeError(
            "Synthetic topology contract does not exercise the requested repeated-role "
            "multiplicity"
        )
    if not any(spec.nodes[node].composite_roles for node in spec.dag_order):
        raise RuntimeError("Synthetic topology contract must contain a hierarchy")
    if len(spec.invisible_roles) < int(requirements["minimum_invisible"]):
        raise RuntimeError(
            "Synthetic topology contract does not exercise enough invisible roles"
        )
    if batch["invisible_truth"].shape != (
        events,
        len(spec.invisible_roles),
        3,
    ):
        raise RuntimeError("Synthetic invisible truth violates [B,Rnu,3]")
    if space.state_count <= 1 or not space.non_null_mask.any():
        raise RuntimeError(
            "Synthetic topology contract produced no useful complete assignments"
        )

    results = {
        "topology": {
            "roots": list(spec.roots),
            "nodes": list(spec.dag_order),
            "invisible_roles": [role.key for role in spec.invisible_roles],
            "invisible_count": len(spec.invisible_roles),
            "maximum_visible_role_multiplicity": max(
                role.multiplicity for role in spec.visible_leaf_roles
            ),
            "assignment_states": space.state_count,
        },
        "visible_m1": run_factorized_mode(
            base, visible, batch, scaler, "visible_m1", device
        ),
        "edm_no_assignment": run_factorized_mode(
            base, visible, batch, scaler, "edm_no_assignment", device
        ),
        "cfm_no_assignment": run_flow_mode(
            base, visible, batch, scaler, "cfm_no_assignment", device
        ),
        "dfm_assignment": run_flow_mode(
            base, visible, batch, scaler, "dfm_assignment", device
        ),
        "joint_flow": run_flow_mode(
            base, visible, batch, scaler, "joint_flow", device
        ),
        "prediction_roundtrip": _prediction_roundtrip(
            base, visible, batch, scaler, device
        ),
    }
    print(json.dumps({"topology_contract": results}, indent=2, sort_keys=True))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology",
        required=True,
        choices=("ttbar1L", "ttbar2L", "ttH1L", *CONTRACTS),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--events", type=int, default=4)
    parser.add_argument("--factorized-only", action="store_true")
    parser.add_argument("--flows-only", action="store_true")
    args = parser.parse_args()
    if args.factorized_only and args.flows_only:
        raise ValueError("Choose at most one of --factorized-only and --flows-only")
    if args.topology in CONTRACTS:
        if args.factorized_only or args.flows_only:
            raise ValueError(
                "Synthetic topology contracts have their own fixed validation subset"
            )
        run_topology_contract(args.topology, device=args.device, events=args.events)
        return
    run_synthetic_smoke(
        Path(f"configs/experiments/{args.topology}.yaml"),
        device=args.device,
        events=args.events,
        factorized=not args.flows_only,
        flows=not args.factorized_only,
    )


if __name__ == "__main__":
    main()
