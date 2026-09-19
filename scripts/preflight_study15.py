#!/usr/bin/env python3
"""Run the bounded, real-data Study-15 production-path preflight.

This is software integration evidence only.  It deliberately exercises the
maintained Study-15 runner for both the locked conventional consumer and one
seed of the Conditional L-GATr training branch; it is never written below
``results/studies``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.cli.run import build_data, load_trained, load_visible_config  # noqa: E402
from joint_event_reconstruction.runtime_paths import runtime_paths  # noqa: E402
from joint_event_reconstruction.studies import resolve_study  # noqa: E402
from scripts import preflight_study13 as p13  # noqa: E402
from scripts.run_study import _gate_checkpoint, _public_run_config  # noqa: E402


STUDY_PATH = ROOT / "configs/studies/15_lgatr_invisible_generator.yaml"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run(command: list[str], env: dict[str, str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(
            f"Study-15 maintained command failed ({completed.returncode}):\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


def _command(stage: str, output_root: Path, variant: str, study, gates_path: Path, device: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_study.py"),
        str(STUDY_PATH),
        stage,
        "--output-root", str(output_root),
        "--topologies", "ttbar2L",
        "--variants", variant,
        "--seeds", "42",
        "--device", device,
        "--expected-gates-digest", str(study.gates_digest),
        "--gates-path", str(gates_path),
    ]
    command.extend(item for override in SMALL_OVERRIDES for item in ("--set", override))
    return command


def _checkpoint_entry(study, gate: str, seed: int) -> dict:
    entry = study.gate_artifact(gate, "ttbar2L", seed)
    checkpoint = _gate_checkpoint(study, gate, "ttbar2L", seed)
    return {
        "gate": gate,
        "seed": seed,
        "path": str(checkpoint),
        "sha256": _sha256(checkpoint),
        "expected_sha256": entry["checkpoint_sha256"],
        "identity_file": str(Path(entry["identity_file"]).expanduser().resolve()),
        "identity_sha256": entry["identity_sha256"],
        "training_sha": entry["git_sha"],
        "architecture_identity": entry["architecture_identity"],
    }


def _resolved_configs(study, output_root: Path) -> tuple[Any, Any, dict]:
    conventional = study.variant_map["conventional"]
    conditional = study.variant_map["conditional_lgatr"]
    _, _, conventional_config = _public_run_config(
        study, "ttbar2L", conventional, 42, output_root, SMALL_OVERRIDES
    )
    _, _, conditional_config = _public_run_config(
        study, "ttbar2L", conditional, 42, output_root, SMALL_OVERRIDES
    )
    return conventional_config, conditional_config, {
        "conventional": OmegaConf.to_container(conventional_config, resolve=True),
        "conditional_lgatr": OmegaConf.to_container(conditional_config, resolve=True),
    }


def _remove_allowed(value: object, path: tuple[str, ...], allowed: set[tuple[str, ...]]) -> object:
    if any(path[: len(prefix)] == prefix for prefix in allowed):
        return None
    if isinstance(value, dict):
        return {
            key: _remove_allowed(child, path + (str(key),), allowed)
            for key, child in value.items()
            if _remove_allowed(child, path + (str(key),), allowed) is not None
        }
    if isinstance(value, list):
        return [_remove_allowed(child, path + (str(index),), allowed) for index, child in enumerate(value)]
    return value


def _config_diff(conventional: dict, conditional: dict) -> dict:
    allowed = {
        ("model", "invisible", "generator", "backbone"),
        ("training", "trainable"),
        ("training", "output"),
        ("experiment", "seed"),
    }
    left = _remove_allowed(conventional, (), allowed)
    right = _remove_allowed(conditional, (), allowed)
    if left != right:
        raise RuntimeError("Study-15 conventional and Conditional L-GATr configs differ outside the whitelist")
    return {
        "status": "PASS",
        "allowed_differences": ["model.invisible.generator.backbone", "training.trainable", "training.output", "experiment.seed"],
        "unexpected_differences": [],
    }


def _strict_reload(config, checkpoint: Path, expected_backbone: str) -> dict:
    visible = load_visible_config(config)
    data = build_data(config, visible)
    data.setup("predict")
    module = load_trained(config, visible, data, str(checkpoint)).eval()
    backbone = str(config.model.invisible.generator.backbone.type)
    if backbone != expected_backbone:
        raise RuntimeError(f"Study-15 reloaded backbone is {backbone!r}, expected {expected_backbone!r}")
    trainable = [name for name, parameter in module.named_parameters() if parameter.requires_grad]
    if any(name.startswith("event_model.visible.") for name in trainable):
        raise RuntimeError("Study-15 reload unexpectedly leaves visible parameters trainable")
    if any(name.startswith(prefix) for name in trainable for prefix in p13._component_prefixes()["assignment"]):
        raise RuntimeError("Study-15 reload unexpectedly leaves assignment parameters trainable")
    invisible = [name for name in trainable if name.startswith("event_model.invisible.")]
    if not invisible:
        raise RuntimeError("Study-15 reload has no trainable invisible generator parameters")
    return {
        "strict_reload": True,
        "backbone": backbone,
        "trainable_parameter_count": len(trainable),
        "trainable_invisible_parameter_count": len(invisible),
        "frozen_visible_assignment": True,
    }


def run(*, output_dir: Path, device: str, gates_path: Path, expected_source_sha: str, campaign_id: str) -> dict:
    paths = runtime_paths()
    if paths.source_sha != expected_source_sha:
        raise RuntimeError(f"SOURCE SHA MISMATCH: expected {expected_source_sha}, found {paths.source_sha}")
    study = resolve_study(STUDY_PATH, gates_path)
    if study.gates_digest is None:
        raise RuntimeError("Study-15 has no resolved gate digest")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the source/runtime environment explicit when called directly and from
    # the immutable Slurm worker alike.
    env = os.environ.copy()
    env.update({
        "JER_PROJECT_ROOT": str(paths.project_root),
        "JER_SOURCE_ROOT": str(paths.source_root),
        "JER_SOURCE_SHA": paths.source_sha,
        "PYTHONPATH": f"{paths.source_root / 'src'}:{paths.source_root}",
        "PYTHONDONTWRITEBYTECODE": "1",
    })

    conventional_config, conditional_config, configs = _resolved_configs(study, output_dir)
    config_diff = _config_diff(configs["conventional"], configs["conditional_lgatr"])
    if str(conventional_config.model.invisible.coordinates.type) != "met_residual":
        raise RuntimeError("Study-15 did not inherit the selected met_residual coordinates")
    if str(conditional_config.model.invisible.generator.backbone.type) != "lgatr":
        raise RuntimeError("Study-15 Conditional L-GATr branch did not construct an lgatr generator")
    if list(conditional_config.training.trainable) != ["invisible"]:
        raise RuntimeError(f"Study-15 trainable component policy changed: {conditional_config.training.trainable}")

    selected = {
        gate: _checkpoint_entry(study, gate, 42)
        for gate in ("visible_assignment", "invisible_generator", "invisible_coordinates", "assignment_formulation")
    }
    for gate in ("invisible_coordinates",):
        for seed in (42, 43, 44):
            _checkpoint_entry(study, gate, seed)

    conventional_dir = output_dir / "conventional"
    _run(_command("predict", output_dir, "conventional", study, gates_path, device), env)
    conventional_prediction = output_dir / "lgatr_invisible_generator/ttbar2L/conventional/seed_42/predictions/default.h5"
    conventional_evaluation = output_dir / "lgatr_invisible_generator/ttbar2L/conventional/seed_42/evaluation/default.json"
    _run(_command("evaluate", output_dir, "conventional", study, gates_path, "cpu"), env)
    conventional_report = {
        "training_performed": False,
        "checkpoint_reused": True,
        "checkpoint": selected["invisible_coordinates"],
        "prediction": p13._prediction_contract(conventional_prediction, coordinate_type="met_residual", expected_events=16),
        "evaluation": p13._evaluation_contract(conventional_prediction, conventional_evaluation),
        "test_accessed": False,
    }
    _write_json(conventional_dir / "result.json", {"schema": "jer_study15_conventional_v1", "status": "PASS", **conventional_report})

    old_overrides = p13.SMALL_OVERRIDES
    p13.SMALL_OVERRIDES = SMALL_OVERRIDES
    try:
        gradient, initial = p13._gradient_probe(
            study, "conditional_lgatr", output_root=output_dir, device=device
        )
        _run(_command("train", output_dir, "conditional_lgatr", study, gates_path, device), env)
    finally:
        p13.SMALL_OVERRIDES = old_overrides

    train_dir = output_dir / "lgatr_invisible_generator/ttbar2L/conditional_lgatr/seed_42/train"
    checkpoint = train_dir / "checkpoints/best-loss.ckpt"
    visible_checkpoint = _gate_checkpoint(study, "visible_assignment", "ttbar2L", 42)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Study-15 Conditional L-GATr checkpoint is missing: {checkpoint}")
    losses = p13._assert_training_metrics(train_dir, "conditional_lgatr")
    updates = p13._checkpoint_update_contract(checkpoint, initial, visible_checkpoint)
    reload_report = _strict_reload(conditional_config, checkpoint, "lgatr")
    old_overrides = p13.SMALL_OVERRIDES
    p13.SMALL_OVERRIDES = SMALL_OVERRIDES
    try:
        _run(_command("predict", output_dir, "conditional_lgatr", study, gates_path, device), env)
        _run(_command("evaluate", output_dir, "conditional_lgatr", study, gates_path, "cpu"), env)
    finally:
        p13.SMALL_OVERRIDES = old_overrides
    conditional_prediction = output_dir / "lgatr_invisible_generator/ttbar2L/conditional_lgatr/seed_42/predictions/default.h5"
    conditional_evaluation = output_dir / "lgatr_invisible_generator/ttbar2L/conditional_lgatr/seed_42/evaluation/default.json"
    conditional_report = {
        "training": {**gradient, **losses},
        "checkpoint_reload": {**updates, **reload_report},
        "prediction": p13._prediction_contract(conditional_prediction, coordinate_type="met_residual", expected_events=16),
        "evaluation": p13._evaluation_contract(conditional_prediction, conditional_evaluation),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
            "identity_file": str(train_dir / "artifact_identity.json"),
            "identity_sha256": _sha256(train_dir / "artifact_identity.json"),
            "training_sha": expected_source_sha,
        },
        "gpu_memory_max_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device == "cuda" and torch.cuda.is_available() else None
        ),
        "test_accessed": False,
    }
    _write_json(output_dir / "conditional_lgatr/result.json", {"schema": "jer_study15_conditional_v1", "status": "PASS", **conditional_report})

    result = {
        "schema": "jer_study15_integration_v1",
        "status": "PASS",
        "software_integration_evidence_only": True,
        "source_sha": expected_source_sha,
        "campaign_id": campaign_id,
        "study_manifest": str(STUDY_PATH),
        "study_manifest_sha256": _sha256(STUDY_PATH),
        "resolved_gate_digest": study.gates_digest,
        "data_subset": {"train_events": 32, "validation_events": 16, "batch_size": 8, "workers": 0, "subset_seed": 42, "test_accessed": False},
        "selected_checkpoint_hashes": {gate: value["sha256"] for gate, value in selected.items()},
        "selected_checkpoint_records": selected,
        "required_components": {
            "conventional_checkpoint_reuse": True,
            "conditional_lgatr_constructed": True,
            "visible_assignment_initialisation": True,
            "visible_assignment_frozen": True,
            "invisible_generator_trainable": True,
            "finite_training_and_validation_losses": True,
            "strict_checkpoint_reload": True,
            "prediction_and_evaluation": True,
            "config_diff_contract": True,
            "test_split_sealed": True,
        },
        "config_diff": config_diff,
        "conditional_lgatr_backbone": OmegaConf.to_container(
            conditional_config.model.invisible.generator.backbone, resolve=True
        ),
        "branches": {
            "conventional": conventional_report,
            "conditional_lgatr": conditional_report,
        },
        "subresults": {
            "conventional": {"path": str(conventional_dir / "result.json"), "schema": "jer_study15_conventional_v1"},
            "conditional_lgatr": {"path": str(output_dir / "conditional_lgatr/result.json"), "schema": "jer_study15_conditional_v1"},
        },
    }
    _write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gates-path", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    result = run(
        output_dir=args.output_dir.expanduser().resolve(),
        device=args.device,
        gates_path=args.gates_path.expanduser().resolve(),
        expected_source_sha=args.expected_source_sha,
        campaign_id=args.campaign_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
