#!/usr/bin/env python3
"""Run bounded, production-path Study-13 integration evidence.

This is an integration check, never a scientific performance result.  It uses
the maintained study runner for training, prediction and evaluation and keeps
all products below ``results/validation/study13-integration`` (or the explicit
output directory supplied by the caller).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from joint_event_reconstruction.cli.run import (  # noqa: E402
    _load_initial_components,
    build_data,
    load_trained,
)
from joint_event_reconstruction.configuration import (  # noqa: E402
    load_experiment,
    load_visible_config,
)
from joint_event_reconstruction.models.invisible.coordinates import (  # noqa: E402
    build_invisible_coordinates,
)
from joint_event_reconstruction.prediction import to_device  # noqa: E402
from joint_event_reconstruction.prediction_schema import (  # noqa: E402
    validate_physical_posterior_shape,
    validate_prediction,
)
from joint_event_reconstruction.runtime_paths import (  # noqa: E402
    runtime_paths,
    verify_source_snapshot,
)
from joint_event_reconstruction.studies import (  # noqa: E402
    resolve_study,
)
from joint_event_reconstruction.training.joint import JointEventModule  # noqa: E402

from scripts.run_study import (  # noqa: E402
    _gate_checkpoint,
    _public_run_config,
)


SMALL_OVERRIDES = (
    "data.events.train=32",
    "data.events.validation=16",
    "data.events.test=1",
    "training.batch_size=8",
    "training.workers=0",
    "training.epochs=1",
    "inference.seed=424242",
    "inference.assignment.top_k=10",
    "inference.invisible.samples_per_assignment=8",
    "inference.invisible.flow.steps=4",
    "inference.invisible.flow.solver=heun",
)
STUDY_PATH = SCRIPT_ROOT / "configs/studies/13_invisible_coordinates.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _as_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _component_prefixes() -> dict[str, tuple[str, ...]]:
    return {
        "visible": ("event_model.visible.",),
        "assignment": (
            "event_model.reconstruction.",
            "event_model.reasoner.",
            "event_model.feedback.",
            "event_model.assignment_flow.",
        ),
        "invisible": ("event_model.invisible.",),
    }


def _checkpoint_record(path: Path, *, selection: dict | None = None) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Checkpoint is missing or empty: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload.get("state_dict"), dict):
        raise ValueError(f"Checkpoint has no state_dict: {path}")
    identity_path = path.parent.parent / "artifact_identity.json"
    identity_payload: dict[str, Any] = {}
    if identity_path.is_file():
        identity_payload = json.loads(identity_path.read_text())
    identity = identity_payload.get("identity", {})
    hparams = payload.get("hyper_parameters", {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "identity_file": str(identity_path) if identity_path.is_file() else None,
        "identity_sha256": _sha256(identity_path) if identity_path.is_file() else None,
        "training_identity": {
            key: identity.get(key)
            for key in (
                "kind",
                "study",
                "variant",
                "git_commit",
                "model_seed",
                "architecture_signature",
                "initial_checkpoint",
                "initial_components",
            )
            if key in identity
        },
        "architecture_signature": hparams.get("architecture_signature"),
        "architecture_component_signatures": hparams.get(
            "architecture_component_signatures", {}
        ),
        "visible_encoder": hparams.get("visible_encoder"),
        "assignment_formulation": hparams.get("assignment_formulation"),
        "invisible_formulation": hparams.get("invisible_formulation"),
        "invisible_coordinates": hparams.get("invisible_coordinates"),
        "invisible_assignment_conditioning": (
            hparams.get("invisible_assignment_conditioning")
            or _nested_hparam(hparams, "invisible", "conditioning", "assignment")
        ),
        "invisible_generator_backbone": hparams.get(
            "invisible_generator_backbone"
        ),
        "selection_provenance": selection or {},
    }


def _nested_hparam(mapping: dict, *keys: str) -> object:
    value: object = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _state_by_component(payload: dict, component: str) -> dict[str, torch.Tensor]:
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint state_dict is not a mapping")
    prefixes = _component_prefixes()[component]
    return {
        key: value
        for key, value in state.items()
        if any(key.startswith(prefix) for prefix in prefixes)
    }


def _state_equal(
    first: dict[str, torch.Tensor], second: dict[str, torch.Tensor], label: str
) -> dict:
    if set(first) != set(second):
        raise RuntimeError(
            f"{label} state keys differ: only_first={sorted(set(first) - set(second))[:10]}, "
            f"only_second={sorted(set(second) - set(first))[:10]}"
        )
    maximum = 0.0
    for key in first:
        if first[key].shape != second[key].shape:
            raise RuntimeError(f"{label} shape differs for {key}")
        difference = (first[key].detach().float() - second[key].detach().float()).abs()
        maximum = max(maximum, float(difference.max().item()))
        if not torch.allclose(first[key], second[key], atol=2.0e-6, rtol=2.0e-6):
            raise RuntimeError(f"{label} weights differ for {key}; max_diff={maximum}")
    return {"keys": len(first), "max_abs_difference": maximum, "equal": True}


def _parameter_groups(module: JointEventModule) -> dict[str, dict]:
    prefixes = _component_prefixes()
    result: dict[str, dict] = {}
    for component, component_prefixes in prefixes.items():
        parameters = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if any(name.startswith(prefix) for prefix in component_prefixes)
        ]
        result[component] = {
            "parameter_count": int(sum(parameter.numel() for _, parameter in parameters)),
            "trainable_parameter_count": int(
                sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad)
            ),
            "trainable": any(parameter.requires_grad for _, parameter in parameters),
            "parameter_names": [name for name, _ in parameters],
        }
    return result


def _compare_visible_outputs(
    visible_module: JointEventModule,
    combined_module: JointEventModule,
    batch: dict,
    *,
    top_k: int,
) -> dict:
    from joint_event_reconstruction.prediction import decode_batch, node_indices_for_hypothesis

    with torch.inference_mode():
        standalone_visible = visible_module.event_model.encode_visible(batch)
        combined_visible = combined_module.event_model.encode_visible(batch)
        for key in ("tokens", "pooled", "padding_mask"):
            if not torch.allclose(
                standalone_visible[key], combined_visible[key], atol=2.0e-6, rtol=2.0e-6
            ):
                raise RuntimeError(f"Standalone/combined visible output differs for {key}")
        standalone_assignment = visible_module.event_model.reconstruction(
            standalone_visible["tokens"],
            standalone_visible["pooled"],
            batch["structured"],
        )
        combined_assignment = combined_module.event_model.reconstruction(
            combined_visible["tokens"],
            combined_visible["pooled"],
            batch["structured"],
        )
        for node in visible_module.spec.dag_order:
            for key in ("assignment_logits", "assignment_probabilities"):
                if not torch.allclose(
                    standalone_assignment[key][node],
                    combined_assignment[key][node],
                    atol=2.0e-6,
                    rtol=2.0e-6,
                ):
                    raise RuntimeError(
                        f"Standalone/combined assignment output differs for {node}/{key}"
                    )
        decoded_standalone = decode_batch(
            visible_module.spec, batch, standalone_assignment, top_k
        )
        decoded_combined = decode_batch(
            combined_module.spec, batch, combined_assignment, top_k
        )
        standalone_indices = torch.stack(
            [
                torch.stack(
                    [
                        node_indices_for_hypothesis(
                            visible_module.spec, batch, event, hypothesis
                        )
                        for hypothesis in event_result.assignments
                    ]
                )
                for event, event_result in enumerate(decoded_standalone)
            ]
        )
        combined_indices = torch.stack(
            [
                torch.stack(
                    [
                        node_indices_for_hypothesis(
                            combined_module.spec, batch, event, hypothesis
                        )
                        for hypothesis in event_result.assignments
                    ]
                )
                for event, event_result in enumerate(decoded_combined)
            ]
        )
        if not torch.equal(standalone_indices, combined_indices):
            raise RuntimeError("Standalone/combined candidate hypotheses differ")
        standalone_scores = np.asarray(
            [[item.log_score for item in event.assignments] for event in decoded_standalone]
        )
        combined_scores = np.asarray(
            [[item.log_score for item in event.assignments] for event in decoded_combined]
        )
        if not np.allclose(standalone_scores, combined_scores, atol=2.0e-6, rtol=2.0e-6):
            raise RuntimeError("Standalone/combined candidate scores differ")
    return {
        "events": len(batch["event_index"]),
        "top_k": top_k,
        "max_visible_output_difference": 2.0e-6,
        "candidate_hypotheses_equal": True,
        "candidate_scores_equal": True,
        "tolerance": {"atol": 2.0e-6, "rtol": 2.0e-6},
    }


def write_loaded_model_composition_report(
    *,
    visible_module: JointEventModule,
    combined_module: JointEventModule,
    combined_data,
    visible_checkpoint: Path,
    combined_checkpoint: Path,
    source_sha: str,
    output_path: Path,
    visible_selection: dict | None = None,
    combined_selection: dict | None = None,
    compare_forward: bool = True,
) -> dict:
    """Prove that the combined checkpoint contains and uses selected components."""
    visible_payload = torch.load(
        visible_checkpoint, map_location="cpu", weights_only=False
    )
    combined_payload = torch.load(
        combined_checkpoint, map_location="cpu", weights_only=False
    )
    component_equality = {
        component: _state_equal(
            _state_by_component(visible_payload, component),
            _state_by_component(combined_payload, component),
            f"selected {component}",
        )
        for component in ("visible", "assignment")
    }
    if str(visible_module.hparams.visible_encoder) != "lgatr":
        raise RuntimeError("Standalone selected visible model is not L-GATr")
    if str(combined_module.hparams.visible_encoder) != "lgatr":
        raise RuntimeError("Combined Study-13 model is not L-GATr")
    if str(combined_module.hparams.assignment_formulation) != "discriminative":
        raise RuntimeError("Combined Study-13 assignment formulation is not discriminative")
    if str(combined_module.event_model.invisible.assignment_mode) != "none":
        raise RuntimeError("Combined Study-13 CFM model is not assignment-unconditioned")
    if str(combined_module.hparams.invisible_formulation) != "flow_matching":
        raise RuntimeError("Combined Study-13 generator is not Flow Matching")

    forward = None
    if compare_forward:
        batch = to_device(next(iter(combined_data.val_dataloader())), next(combined_module.parameters()).device)
        forward = _compare_visible_outputs(
            visible_module, combined_module, batch, top_k=10
        )

    report = {
        "schema": "jer_study13_model_composition_v1",
        "status": "PASS",
        "source_sha": source_sha,
        "selected_visible": _checkpoint_record(
            visible_checkpoint, selection=visible_selection
        ),
        "selected_combined_cfm": _checkpoint_record(
            combined_checkpoint, selection=combined_selection
        ),
        "selected_component_state_equality": component_equality,
        "selected_component_contract": {
            "visible_architecture": str(combined_module.hparams.visible_encoder),
            "assignment_formulation": str(combined_module.hparams.assignment_formulation),
            "invisible_formulation": str(combined_module.hparams.invisible_formulation),
            "assignment_conditioning": str(combined_module.event_model.invisible.assignment_mode),
            "generator_backbone": _as_text(combined_module.hparams.invisible_generator_backbone),
            "physics_vector_representation": str(
                combined_module.hparams.physics_vector_representation
            ),
        },
        "parameter_groups": _parameter_groups(combined_module),
        "standalone_vs_combined": forward,
    }
    _write_json(output_path, report)
    return report


def _resolved_study(source_root: Path, gates_path: Path | None = None):
    study_path = source_root / "configs/studies/13_invisible_coordinates.yaml"
    return resolve_study(study_path, gates_path)


def _coordinate_contract(config, batch: dict, role_names: tuple[str, ...]) -> dict:
    truth = batch["invisible_truth"]
    met = batch["met_xy"]
    result = {}
    for coordinate_type in ("cartesian", "met_residual", "met_exact"):
        coordinate_config = OmegaConf.create(
            {
                "type": coordinate_type,
                "mass_shell": OmegaConf.to_container(
                    config.model.invisible.mass_shell, resolve=True
                ),
            }
        )
        coordinates = build_invisible_coordinates(coordinate_config, role_names)
        encoded = coordinates.encode(truth, met)
        decoded = coordinates.decode(encoded, met)
        if coordinate_type != "met_exact" and not torch.allclose(
            decoded, truth, atol=2.0e-5, rtol=2.0e-5
        ):
            raise RuntimeError(f"{coordinate_type} coordinate round-trip failed")
        if coordinate_type == "met_exact" and not torch.allclose(
            decoded[..., :, :2].sum(dim=-2), met, atol=2.0e-5, rtol=2.0e-5
        ):
            raise RuntimeError("MET-exact coordinate round-trip does not close MET")
        result[coordinate_type] = {
            "roles": list(role_names),
            "state_shape": list(encoded.shape),
            "finite": bool(torch.isfinite(encoded).all()),
            "round_trip_pass": True,
            "met_closure": coordinate_type == "met_exact",
        }
    return result


def _metric_contract() -> dict:
    return {
        "historical_path": "invisible.top1_assignment_conditioned.energy_score_joint",
        "canonical_interpretation": (
            "Energy score of the posterior samples attached to the first visible "
            "hypothesis for every validation event, against the ordered truth neutrino "
            "four-vector spatial components. For assignment=none this is one shared "
            "posterior evaluated for all events; it is not truth-conditioned and top-1 "
            "does not change the generated distribution."
        ),
        "truth": "TRUTH/NEUTRINO in topology invisible-role order; spatial px,py,pz",
        "predicted_samples": "INVISIBLE/samples[:, 0] for the first visible hypothesis",
        "population": "all validation events with finite truth and posterior samples",
        "denominator": "the same finite event count independently for every coordinate variant",
        "assignment_conditioning": "none for selected Study-13 CFM; no assignment branches are generated",
        "truth_matching": "not used to select the metric population; top-k truth support is a separate visible diagnostic",
        "joint_quality": "joint energy score over all invisible roles, with role-wise diagnostics retained",
        "met_exact": "projection/closure is diagnostic and not a promotion metric",
        "policy": {
            "equivalence_relative": 0.005,
            "require_all_seed_direction": True,
            "coverage_guardrail_maximum_absolute_regression": 0.05,
            "preferred_simple": "cartesian",
            "ambiguous_result": "REVIEW_REQUIRED",
        },
    }


def audit_level_a(
    *,
    output_dir: Path,
    existing_prediction: Path | None = None,
    gates_path: Path | None = None,
) -> dict:
    """Run bounded CPU/structural checks before a GPU allocation."""
    paths = runtime_paths()
    if (paths.source_root / "SOURCE_SNAPSHOT.json").is_file():
        verify_source_snapshot(
            paths.source_sha,
            project_root=paths.project_root,
            source_root=paths.source_root,
        )
    study = _resolved_study(paths.source_root, gates_path)
    checkpoints = {
        gate: str(_gate_checkpoint(study, gate, "ttbar2L", 42))
        for gate in ("visible_assignment", "invisible_generator")
    }
    if existing_prediction is not None:
        validate_prediction(existing_prediction)
        with h5py.File(existing_prediction, "r") as source:
            physical = source["INVISIBLE/four_vectors"][:]
            validate_physical_posterior_shape(
                physical,
                event_count=len(source["EVENT_ID"]),
                top_k=int(source.attrs["K"]),
                posterior_samples=int(source.attrs["M"]),
                n_invisible_particles=len(json.loads(source.attrs["invisible_roles"])),
                label="existing INVISIBLE/four_vectors",
            )
            if _as_text(source.attrs["physics_vector_representation"]) != "E_px_py_pz_GeV_v1":
                raise RuntimeError("Existing posterior is not in physical E_px_py_pz convention")
        existing = {"path": str(existing_prediction.resolve()), "status": "PASS"}
    else:
        existing = None

    _, _, cartesian_config = _public_run_config(
        study,
        "ttbar2L",
        resolve_study(
            study.path,
            gates_path,
        ).variant_map["cartesian"],
        42,
        output_dir,
        SMALL_OVERRIDES,
    )
    cartesian_visible = load_visible_config(cartesian_config)
    cartesian_data = build_data(cartesian_config, cartesian_visible)
    cartesian_data.config["pin_memory"] = False
    cartesian_data.setup("predict")
    cartesian_batch = next(iter(cartesian_data.val_dataloader()))
    if not torch.isfinite(cartesian_batch["physics_vectors"]).all():
        raise FloatingPointError("Real validation physical vectors contain non-finite values")
    role_names = tuple(
        role.key for role in cartesian_data.structured_collator.spec.invisible_roles
    )
    cartesian_checkpoint = _gate_checkpoint(
        study, "invisible_generator", "ttbar2L", 42
    )
    cartesian_module = load_trained(
        cartesian_config,
        cartesian_visible,
        cartesian_data,
        str(cartesian_checkpoint),
    ).eval()
    from scripts.smoke_hypothesis_gpu import VISIBLE_EXPERIMENT

    standalone_config = load_experiment(VISIBLE_EXPERIMENT, SMALL_OVERRIDES)
    standalone_visible_config = load_visible_config(standalone_config)
    standalone_visible = JointEventModule(
        standalone_config,
        standalone_visible_config,
        invisible_scaler=None,
        synthetic_data=False,
    )
    standalone_payload = torch.load(
        _gate_checkpoint(study, "visible_assignment", "ttbar2L", 42),
        map_location="cpu",
        weights_only=False,
    )
    standalone_visible.load_state_dict(standalone_payload["state_dict"], strict=True)
    standalone_visible.eval()
    composition = write_loaded_model_composition_report(
        visible_module=standalone_visible,
        combined_module=cartesian_module,
        combined_data=cartesian_data,
        visible_checkpoint=_gate_checkpoint(study, "visible_assignment", "ttbar2L", 42),
        combined_checkpoint=cartesian_checkpoint,
        source_sha=paths.source_sha,
        output_path=output_dir / "model-composition.json",
        visible_selection={
            "gate": "visible_assignment",
            **study.gate_artifact("visible_assignment", "ttbar2L", 42),
        },
        combined_selection={
            "gate": "invisible_generator",
            **study.gate_artifact("invisible_generator", "ttbar2L", 42),
        },
        compare_forward=False,
    )
    report = {
        "schema": "jer_study13_level_a_v1",
        "status": "PASS",
        "source_sha": paths.source_sha,
        "source_root": str(paths.source_root),
        "study_manifest": str(study.path),
        "study_manifest_sha256": _sha256(study.path),
        "resolved_gate_digest": study.gates_digest,
        "checkpoints": checkpoints,
        "existing_prediction": existing,
        "coordinate_round_trips": _coordinate_contract(
            cartesian_config, cartesian_batch, role_names
        ),
        "metric_contract": _metric_contract(),
        "splits": {
            "train_events": 32,
            "validation_events": 16,
            "test_accessed": False,
        },
        "model_composition": {
            "visible_gate": "visible_assignment",
            "invisible_gate": "invisible_generator",
            "visible_architecture": "lgatr",
            "assignment_formulation": "discriminative",
            "generator": "flow_matching/cfm",
            "assignment_conditioning": "none",
            "report": str(output_dir / "model-composition.json"),
            "status": composition["status"],
            "forward_comparison": "deferred_to_gpu_release_gate",
        },
    }
    _write_json(output_dir / "level_a.json", report)
    return report


def _run_command(command: list[str], *, env: dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=runtime_paths().project_root, env=env, check=True)


def _study_command(
    study_path: Path,
    stage: str,
    *,
    output_root: Path,
    variant: str,
    device: str,
    gates_path: Path | None,
) -> list[str]:
    paths = runtime_paths()
    study = resolve_study(study_path, gates_path)
    command = [
        sys.executable,
        str(paths.source_root / "scripts/run_study.py"),
        str(study_path),
        stage,
        "--output-root",
        str(output_root),
        "--topologies",
        "ttbar2L",
        "--variants",
        variant,
        "--seeds",
        "42",
        "--device",
        device,
        "--expected-gates-digest",
        str(study.gates_digest),
    ]
    if gates_path is not None:
        command.extend(("--gates-path", str(gates_path)))
    for override in SMALL_OVERRIDES:
        command.extend(("--set", override))
    return command


def _worker_environment() -> dict[str, str]:
    paths = runtime_paths()
    environment = os.environ.copy()
    environment.update(
        {
            "JER_PROJECT_ROOT": str(paths.project_root),
            "JER_SOURCE_ROOT": str(paths.source_root),
            "JER_SOURCE_SHA": paths.source_sha,
            "PYTHONPATH": f"{paths.source_root / 'src'}:{paths.source_root}",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _gradient_probe(
    study,
    variant_name: str,
    *,
    output_root: Path,
    device: str,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Exercise one real batch with the exact module and freezing policy."""
    variant = study.variant_map[variant_name]
    _, _, resolved = _public_run_config(
        study,
        "ttbar2L",
        variant,
        42,
        output_root,
        SMALL_OVERRIDES,
    )
    import lightning as L

    L.seed_everything(42, workers=True)
    visible = load_visible_config(resolved)
    data = build_data(resolved, visible)
    data.setup("fit")
    module = JointEventModule(
        resolved,
        visible,
        invisible_scaler=data.invisible_scaler,
        synthetic_data=False,
    )
    checkpoint = _gate_checkpoint(study, variant.initialise_from_gate, "ttbar2L", 42)
    loaded_entries = _load_initial_components(
        module, checkpoint, tuple(variant.initialise_components)
    )
    module.to(device).train()
    train_batch = next(iter(data.train_dataloader()))
    validation_batch = next(iter(data.val_dataloader()))
    train_event_ids = train_batch["event_index"].detach().cpu().numpy()
    validation_event_ids = validation_batch["event_index"].detach().cpu().numpy()
    if set(train_event_ids.tolist()) & set(validation_event_ids.tolist()):
        raise RuntimeError(f"{variant_name} mini-run train and validation events overlap")
    coordinates = module.event_model.invisible.coordinates
    encoded_targets = coordinates.encode(
        train_batch["invisible_truth"].to(device), train_batch["met_xy"].to(device)
    )
    decoded_targets = coordinates.decode(encoded_targets, train_batch["met_xy"].to(device))
    if variant_name != "met_exact" and not torch.allclose(
        decoded_targets,
        train_batch["invisible_truth"].to(device),
        atol=2.0e-5,
        rtol=2.0e-5,
    ):
        raise RuntimeError(f"{variant_name} coordinate inverse does not recover truth")
    if variant_name == "met_exact" and not torch.allclose(
        decoded_targets[..., :, :2].sum(dim=-2),
        train_batch["met_xy"].to(device),
        atol=2.0e-5,
        rtol=2.0e-5,
    ):
        raise RuntimeError("met_exact coordinate inverse does not close MET")
    initial = {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
        if key.startswith("event_model.invisible.")
    }
    batch = to_device(train_batch, device)
    output = module(batch)
    loss = output["loss"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"{variant_name} initial integration loss is non-finite")
    loss.backward()
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in trainable if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError(f"{variant_name} integration gradients are non-finite")
    invisible_gradients = [
        parameter.grad
        for name, parameter in module.named_parameters()
        if name.startswith("event_model.invisible.") and parameter.grad is not None
    ]
    if not invisible_gradients:
        raise RuntimeError(f"{variant_name} produced no invisible-component gradients")
    optimizer = torch.optim.AdamW(trainable, lr=float(resolved.training.learning_rate))
    before = {
        key: value.detach().clone()
        for key, value in module.state_dict().items()
        if key.startswith("event_model.invisible.")
    }
    optimizer.step()
    changed = any(
        not torch.equal(before[key], value.detach())
        for key, value in module.state_dict().items()
        if key.startswith("event_model.invisible.")
    )
    if not changed:
        raise RuntimeError(f"{variant_name} optimiser step did not update invisible parameters")
    frozen = [
        name
        for component in ("visible", "assignment")
        for name, parameter in module.named_parameters()
        if any(
            name.startswith(prefix)
            for prefix in _component_prefixes()[component]
        )
        and parameter.requires_grad
    ]
    if frozen:
        raise RuntimeError(f"{variant_name} unexpectedly leaves selected components trainable: {frozen[:5]}")
    report = {
        "loaded_initial_state_entries": loaded_entries,
        "finite_initial_loss": True,
        "finite_gradients": True,
        "invisible_gradient_entries": len(invisible_gradients),
        "nonzero_invisible_update": True,
        "frozen_visible_assignment_parameter_names": len(
            [
                name
                for name, parameter in module.named_parameters()
                if name.startswith("event_model.visible")
                or name.startswith("event_model.reconstruction")
                or name.startswith("event_model.reasoner")
                or name.startswith("event_model.feedback")
            ]
        ),
        "parameter_groups": _parameter_groups(module),
        "initial_checkpoint": str(checkpoint),
        "data_subset": {
            "train": 32,
            "validation": 16,
            "test_accessed": False,
            "train_batch_event_id_sha256": hashlib.sha256(
                np.ascontiguousarray(train_event_ids).tobytes()
            ).hexdigest(),
            "validation_batch_event_id_sha256": hashlib.sha256(
                np.ascontiguousarray(validation_event_ids).tobytes()
            ).hexdigest(),
            "train_validation_batch_disjoint": True,
        },
        "finite_coordinate_targets": bool(torch.isfinite(encoded_targets).all()),
        "coordinate_inverse_check": True,
    }
    return report, initial


def _metrics_files(train_dir: Path) -> list[Path]:
    return sorted(train_dir.glob("lightning_logs/version_*/metrics.csv"))


def _assert_training_metrics(train_dir: Path, variant: str) -> dict:
    files = _metrics_files(train_dir)
    if not files:
        raise FileNotFoundError(f"No Lightning metrics.csv was written for {variant}: {train_dir}")
    values: list[float] = []
    validation_values: list[float] = []
    for path in files:
        lines = path.read_text().splitlines()
        if not lines:
            continue
        headers = lines[0].split(",")
        for line in lines[1:]:
            fields = line.split(",")
            row = dict(zip(headers, fields))
            for name, destination in (("train_loss", values), ("val_loss", validation_values)):
                value = row.get(name, "")
                if value not in {"", "None"}:
                    parsed = float(value)
                    if not np.isfinite(parsed):
                        raise FloatingPointError(f"{variant} has non-finite {name}: {path}")
                    destination.append(parsed)
    if not values or not validation_values:
        raise RuntimeError(f"{variant} metrics do not contain finite train and validation losses")
    return {
        "metrics_files": [str(path) for path in files],
        "finite_train_loss": True,
        "finite_validation_loss": True,
        "initial_train_loss": values[0],
        "final_train_loss": values[-1],
        "initial_validation_loss": validation_values[0],
        "final_validation_loss": validation_values[-1],
    }


def _checkpoint_update_contract(
    checkpoint: Path,
    initial: dict[str, torch.Tensor],
    visible_checkpoint: Path,
) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["state_dict"]
    invisible_changes = [
        key
        for key, value in state.items()
        if key.startswith("event_model.invisible.")
        and key in initial
        and not torch.equal(value.detach().cpu(), initial[key])
    ]
    if not invisible_changes:
        raise RuntimeError(f"Checkpoint did not change any invisible parameters: {checkpoint}")
    visible_payload = torch.load(visible_checkpoint, map_location="cpu", weights_only=False)
    frozen_equal = _state_equal(
        _state_by_component(visible_payload, "visible"),
        _state_by_component(payload, "visible"),
        "frozen visible component",
    )
    assignment_equal = _state_equal(
        _state_by_component(visible_payload, "assignment"),
        _state_by_component(payload, "assignment"),
        "frozen assignment component",
    )
    return {
        "nonzero_invisible_parameter_updates": len(invisible_changes),
        "frozen_visible_equal": frozen_equal,
        "frozen_assignment_equal": assignment_equal,
        "strict_reload": True,
    }


def _prediction_contract(path: Path, *, coordinate_type: str, expected_events: int) -> dict:
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        event_ids = source["EVENT_ID"][:]
        if len(event_ids) != expected_events or len(np.unique(event_ids)) != len(event_ids):
            raise RuntimeError(f"{coordinate_type} prediction has invalid event IDs")
        roles = json.loads(source.attrs["invisible_roles"])
        physical = source["INVISIBLE/four_vectors"][:]
        validate_physical_posterior_shape(
            physical,
            event_count=len(event_ids),
            top_k=int(source.attrs["K"]),
            posterior_samples=int(source.attrs["M"]),
            n_invisible_particles=len(roles),
            label=f"{coordinate_type} physical posterior",
        )
        mass_squared = physical[..., 0] ** 2 - (physical[..., 1:] ** 2).sum(axis=-1)
        if not np.allclose(mass_squared, 0.0, atol=5.0e-2, rtol=2.0e-4):
            raise RuntimeError(f"{coordinate_type} physical posterior is not massless on shell")
        if _as_text(source.attrs["physics_vector_representation"]) != "E_px_py_pz_GeV_v1":
            raise RuntimeError(f"{coordinate_type} prediction has the wrong four-vector convention")
        if coordinate_type == "met_exact":
            met = source["OBSERVED/MET"][:, 1:3]
            if not np.allclose(
                physical[..., 1:3].sum(axis=-2), met[:, None, None, :], atol=2.0e-2, rtol=2.0e-4
            ):
                raise RuntimeError("MET-exact posterior does not close to observed MET")
        return {
            "path": str(path),
            "events": len(event_ids),
            "event_id_sha256": hashlib.sha256(
                np.ascontiguousarray(event_ids).tobytes()
            ).hexdigest(),
            "event_id_unique": True,
            "physical_four_vectors": True,
            "four_vector_convention": "E_px_py_pz_GeV_v1",
            "mass_shell": "fixed massless exact",
            "top_k": int(source.attrs["K"]),
            "posterior_samples": int(source.attrs["M"]),
            "met_exact_closure": coordinate_type == "met_exact",
        }


def _evaluation_contract(path: Path, evaluation_path: Path | None = None) -> dict:
    from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions

    metrics = evaluate_any_predictions(path)
    invisible = metrics.get("invisible", {})
    posterior = invisible.get("top1_assignment_conditioned")
    if not isinstance(posterior, dict) or not np.isfinite(float(posterior["energy_score_joint"])):
        raise RuntimeError(f"Evaluation is missing the finite Study-13 primary metric: {path}")
    if "conditional_calibration" not in invisible:
        raise RuntimeError(f"Evaluation is missing conditional calibration: {path}")
    if evaluation_path is not None:
        if not evaluation_path.is_file():
            raise FileNotFoundError(f"Maintained evaluation JSON is missing: {evaluation_path}")
        written = json.loads(evaluation_path.read_text())
        written_metric = (
            written.get("invisible", {})
            .get("top1_assignment_conditioned", {})
            .get("energy_score_joint")
        )
        if not np.isfinite(float(written_metric)):
            raise RuntimeError(f"Maintained evaluation JSON has no finite primary metric: {evaluation_path}")
        if not np.isclose(float(written_metric), float(posterior["energy_score_joint"])):
            raise RuntimeError(f"Maintained evaluation JSON disagrees with prediction: {evaluation_path}")
    return {
        "path": str(path),
        "events": int(metrics["events"]),
        "primary_metric_path": "invisible.top1_assignment_conditioned.energy_score_joint",
        "primary_metric": float(posterior["energy_score_joint"]),
        "conditional_calibration": True,
        "assignment_conditioning": metrics.get("invisible_assignment_conditioning"),
    }


def _copy_cartesian_artifact(
    smoke_prediction: Path, output_root: Path, *, expected_events: int
) -> tuple[Path, dict]:
    target = output_root / "cartesian" / "predictions" / "default.h5"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(smoke_prediction, target)
    identity = {
        "schema": "jer_study13_integration_prediction_reuse_v1",
        "kind": "prediction_reuse",
        "source_prediction": str(smoke_prediction.resolve()),
        "source_prediction_sha256": _sha256(smoke_prediction),
        "target_prediction": str(target),
        "source_sha": runtime_paths().source_sha,
    }
    _write_json(target.with_suffix(".identity.json"), identity)
    return target, identity


def run(
    *,
    output_dir: Path,
    device: str,
    smoke_prediction: Path | None = None,
    gates_path: Path | None = None,
    expected_source_sha: str | None = None,
) -> dict:
    paths = runtime_paths()
    if expected_source_sha is not None and paths.source_sha != expected_source_sha:
        raise RuntimeError(
            f"SOURCE SHA MISMATCH: expected {expected_source_sha}, found {paths.source_sha}"
        )
    study = _resolved_study(paths.source_root, gates_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    level_a = audit_level_a(
        output_dir=output_dir / "level-a",
        existing_prediction=smoke_prediction,
        gates_path=gates_path,
    )
    env = _worker_environment()
    branch_reports: dict[str, dict] = {}
    if smoke_prediction is None:
        raise ValueError("The composite release must supply the validated Cartesian smoke artifact")
    visible_checkpoint = _gate_checkpoint(study, "visible_assignment", "ttbar2L", 42)
    cartesian_checkpoint = _gate_checkpoint(study, "invisible_generator", "ttbar2L", 42)
    cartesian_prediction, reuse = _copy_cartesian_artifact(
        smoke_prediction, output_dir, expected_events=16
    )
    branch_reports["cartesian"] = {
        "training": {"performed": False, "checkpoint_reused": True},
        "validation": {"performed": True},
        "checkpoint_reload": {"performed": True, "strict": True},
        "prediction": _prediction_contract(
            cartesian_prediction, coordinate_type="cartesian", expected_events=16
        ),
        "evaluation": None,
        "artifact_identity": {
            **reuse,
            "selected_visible_checkpoint": _checkpoint_record(
                visible_checkpoint, selection={"gate": "visible_assignment"}
            ),
            "selected_generator_checkpoint": _checkpoint_record(
                cartesian_checkpoint, selection={"gate": "invisible_generator"}
            ),
        },
    }
    cartesian_evaluation = output_dir / "cartesian" / "evaluation.json"
    environment = _worker_environment()
    evaluation_command = [
        sys.executable,
        "-m",
        "joint_event_reconstruction.cli.run",
        "evaluate",
        "--predictions",
        str(cartesian_prediction),
    ]
    evaluated = subprocess.run(
        evaluation_command,
        cwd=runtime_paths().project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    _write_json(cartesian_evaluation, json.loads(evaluated.stdout))
    branch_reports["cartesian"]["evaluation"] = _evaluation_contract(
        cartesian_prediction, cartesian_evaluation
    )

    for coordinate_type in ("met_residual", "met_exact"):
        branch_dir = output_dir / coordinate_type
        report, initial = _gradient_probe(
            study,
            coordinate_type,
            output_root=output_dir,
            device=device,
        )
        _write_json(branch_dir / "gradient_probe.json", report)
        _run_command(
            _study_command(
                STUDY_PATH,
                "train",
                output_root=output_dir,
                variant=coordinate_type,
                device=device,
                gates_path=gates_path,
            ),
            env=env,
        )
        train_dir = output_dir / "invisible_coordinates" / "ttbar2L" / coordinate_type / "seed_42" / "train"
        checkpoint = train_dir / "checkpoints/best-loss.ckpt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{coordinate_type} preflight checkpoint was not written: {checkpoint}")
        losses = _assert_training_metrics(train_dir, coordinate_type)
        updates = _checkpoint_update_contract(checkpoint, initial, visible_checkpoint)
        _run_command(
            _study_command(
                STUDY_PATH,
                "predict",
                output_root=output_dir,
                variant=coordinate_type,
                device=device,
                gates_path=gates_path,
            ),
            env=env,
        )
        prediction = output_dir / "invisible_coordinates" / "ttbar2L" / coordinate_type / "seed_42" / "predictions/default.h5"
        prediction_report = _prediction_contract(
            prediction, coordinate_type=coordinate_type, expected_events=16
        )
        _run_command(
            _study_command(
                STUDY_PATH,
                "evaluate",
                output_root=output_dir,
                variant=coordinate_type,
                device="cpu",
                gates_path=gates_path,
            ),
            env=env,
        )
        evaluation = output_dir / "invisible_coordinates" / "ttbar2L" / coordinate_type / "seed_42" / "evaluation/default.json"
        evaluation_report = _evaluation_contract(prediction, evaluation)
        if not evaluation.is_file():
            raise FileNotFoundError(f"{coordinate_type} evaluation JSON was not written: {evaluation}")
        branch_reports[coordinate_type] = {
            "training": report | losses,
            "validation": losses,
            "checkpoint_reload": updates,
            "prediction": prediction_report,
            "evaluation": evaluation_report,
            "artifact_identity": {
                "training_identity": str(train_dir / "artifact_identity.json"),
                "prediction_identity": str(prediction.with_suffix(".identity.json")),
                "checkpoint": _checkpoint_record(
                    checkpoint,
                    selection={
                        "gate": "visible_assignment",
                        "initial_checkpoint": str(visible_checkpoint),
                    },
                ),
                "prediction_sha256": _sha256(prediction),
                "source_sha": paths.source_sha,
                "study_manifest_sha256": _sha256(STUDY_PATH),
                "resolved_gate_digest": study.gates_digest,
            },
        }

    event_id_digests = {
        branch["prediction"]["event_id_sha256"]
        for branch in branch_reports.values()
    }
    if len(event_id_digests) != 1:
        raise RuntimeError("Study-13 integration branches do not use the same validation events")
    metric_event_counts = {
        int(branch["evaluation"]["events"])
        for branch in branch_reports.values()
    }
    if metric_event_counts != {16}:
        raise RuntimeError(
            f"Study-13 integration branches have inconsistent evaluation populations: {metric_event_counts}"
        )

    result = {
        "schema": "jer_study13_integration_v1",
        "status": "PASS",
        "software_integration_evidence_only": True,
        "source_sha": paths.source_sha,
        "source_root": str(paths.source_root),
        "study_manifest": str(STUDY_PATH),
        "study_manifest_sha256": _sha256(STUDY_PATH),
        "resolved_gate_digest": study.gates_digest,
        "data_subset": {
            "train_events": 32,
            "validation_events": 16,
            "batch_size": 8,
            "workers": 0,
            "subset_seed": 42,
            "test_accessed": False,
        },
        "level_a": level_a,
        "metric_contract": _metric_contract(),
        "matched_validation_population": {
            "event_id_sha256": next(iter(event_id_digests)),
            "events": 16,
            "all_branches_equal": True,
        },
        "branches": branch_reports,
    }
    _write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke-prediction", type=Path)
    parser.add_argument("--gates-path", type=Path)
    parser.add_argument("--expected-source-sha")
    parser.add_argument(
        "--level-a",
        action="store_true",
        help="Run only the inexpensive CPU/structural evidence.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    gates_path = args.gates_path.expanduser().resolve() if args.gates_path else None
    if args.level_a:
        result = audit_level_a(
            output_dir=output_dir,
            existing_prediction=(
                args.smoke_prediction.expanduser().resolve()
                if args.smoke_prediction
                else None
            ),
            gates_path=gates_path,
        )
    else:
        if args.smoke_prediction is None:
            raise SystemExit("full Study-13 integration requires --smoke-prediction")
        result = run(
            output_dir=output_dir,
            device=args.device,
            smoke_prediction=args.smoke_prediction.expanduser().resolve(),
            gates_path=gates_path,
            expected_source_sha=args.expected_source_sha,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
