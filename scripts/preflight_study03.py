#!/usr/bin/env python3
"""Run the conditional Study-03 preflight when Study-10 released coupling."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.runtime_paths import runtime_paths  # noqa: E402
from joint_event_reconstruction.studies import resolve_study  # noqa: E402
from scripts import preflight_study13 as p13  # noqa: E402
from scripts.run_study import _gate_checkpoint, _public_run_config  # noqa: E402


STUDY_PATH = ROOT / "configs/studies/03_assignment_conditioning.yaml"
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


def _command(stage: str, output: Path, variant: str, study, gates_path: Path, device: str) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts/run_study.py"), str(STUDY_PATH), stage,
        "--output-root", str(output), "--topologies", "ttbar2L", "--variants", variant,
        "--seeds", "42", "--device", device,
        "--expected-gates-digest", str(study.gates_digest), "--gates-path", str(gates_path),
    ]
    command.extend(item for override in SMALL_OVERRIDES for item in ("--set", override))
    return command


def run(*, output_dir: Path, gates_path: Path, expected_source_sha: str, campaign_id: str) -> dict:
    paths = runtime_paths()
    if paths.source_sha != expected_source_sha:
        raise RuntimeError(f"SOURCE SHA MISMATCH: expected {expected_source_sha}, found {paths.source_sha}")
    gates = OmegaConf.to_container(OmegaConf.load(gates_path), resolve=True)
    coupling = gates.get("assignment_coupling", {})
    if coupling.get("status") != "locked" or coupling.get("selected_decision") in {None, "none"}:
        raise RuntimeError("Study-03 is not applicable: Study-10 did not release a deployable coupling")
    study = resolve_study(STUDY_PATH, gates_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **__import__("os").environ,
        "JER_PROJECT_ROOT": str(paths.project_root),
        "JER_SOURCE_ROOT": str(paths.source_root),
        "JER_SOURCE_SHA": paths.source_sha,
        "PYTHONPATH": f"{paths.source_root / 'src'}:{paths.source_root}",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    refined = study.variant_map["refined"]
    _, _, resolved = _public_run_config(study, "ttbar2L", refined, 42, output_dir, SMALL_OVERRIDES)
    if str(resolved.model.invisible.conditioning.assignment) != "hypothesis":
        raise RuntimeError("Study-03 refined branch does not use model-predicted hypothesis conditioning")
    if bool(resolved.model.invisible.conditioning.allow_truth_oracle):
        raise RuntimeError("Study-03 refined branch enables truth-oracle conditioning")

    old_overrides = p13.SMALL_OVERRIDES
    p13.SMALL_OVERRIDES = SMALL_OVERRIDES
    try:
        gradient, initial = p13._gradient_probe(study, "refined", output_root=output_dir, device="cuda")
        subprocess.run(_command("train", output_dir, "refined", study, gates_path, "cuda"), cwd=ROOT, env=env, check=True)
    finally:
        p13.SMALL_OVERRIDES = old_overrides
    train_dir = output_dir / "assignment_conditioning/ttbar2L/refined/seed_42/train"
    checkpoint = train_dir / "checkpoints/best-loss.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Study-03 refined checkpoint is missing: {checkpoint}")
    losses = p13._assert_training_metrics(train_dir, "refined")
    reload_report = p13._checkpoint_update_contract(
        checkpoint, initial, _gate_checkpoint(study, "visible_assignment", "ttbar2L", 42)
    )
    old_overrides = p13.SMALL_OVERRIDES
    p13.SMALL_OVERRIDES = SMALL_OVERRIDES
    try:
        subprocess.run(_command("predict", output_dir, "refined", study, gates_path, "cuda"), cwd=ROOT, env=env, check=True)
        subprocess.run(_command("evaluate", output_dir, "refined", study, gates_path, "cpu"), cwd=ROOT, env=env, check=True)
        subprocess.run(_command("predict", output_dir, "none", study, gates_path, "cuda"), cwd=ROOT, env=env, check=True)
        subprocess.run(_command("evaluate", output_dir, "none", study, gates_path, "cpu"), cwd=ROOT, env=env, check=True)
    finally:
        p13.SMALL_OVERRIDES = old_overrides
    refined_prediction = output_dir / "assignment_conditioning/ttbar2L/refined/seed_42/predictions/default.h5"
    refined_evaluation = output_dir / "assignment_conditioning/ttbar2L/refined/seed_42/evaluation/default.json"
    none_prediction = output_dir / "assignment_conditioning/ttbar2L/none/seed_42/predictions/default.h5"
    none_evaluation = output_dir / "assignment_conditioning/ttbar2L/none/seed_42/evaluation/default.json"
    refined_report = p13._prediction_contract(refined_prediction, coordinate_type="met_residual", expected_events=16)
    none_report = p13._prediction_contract(none_prediction, coordinate_type="met_residual", expected_events=16)
    with __import__("h5py").File(refined_prediction, "r") as source:
        conditioning = str(source.attrs.get("invisible_assignment_conditioning", ""))
    if conditioning != "hypothesis":
        raise RuntimeError(f"Study-03 refined prediction conditioning is {conditioning!r}, not hypothesis")
    result = {
        "schema": "jer_study03_integration_v1",
        "status": "PASS",
        "software_integration_evidence_only": True,
        "source_sha": expected_source_sha,
        "campaign_id": campaign_id,
        "study_manifest_sha256": _sha256(STUDY_PATH),
        "resolved_gate_digest": study.gates_digest,
        "selected_coupling": coupling.get("selected_decision"),
        "anti_leakage": {
            "refined_conditioning": "hypothesis",
            "truth_assignment_in_production_path": False,
            "truth_oracle_mode": False,
        },
        "branches": {
            "none": {"prediction": none_report, "evaluation": p13._evaluation_contract(none_prediction, none_evaluation), "checkpoint_from_gate": "invisible_backbone"},
            "refined": {"training": {**gradient, **losses}, "checkpoint_reload": reload_report, "prediction": refined_report, "evaluation": p13._evaluation_contract(refined_prediction, refined_evaluation)},
        },
        "required_components": {
            "model_predicted_refined_assignment": True,
            "no_truth_assignment_leakage": True,
            "generator_checkpoint_loaded": True,
            "conditioning_reaches_generator": True,
            "finite_training_reload_prediction_evaluation": True,
            "matched_validation_population": True,
            "test_split_sealed": True,
        },
        "test_accessed": False,
    }
    _write_json(output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gates-path", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir.expanduser().resolve(), gates_path=args.gates_path.expanduser().resolve(), expected_source_sha=args.expected_source_sha, campaign_id=args.campaign_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
