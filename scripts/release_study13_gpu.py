#!/usr/bin/env python3
"""Run the composite Study-13 GPU release gate.

The release gate is deliberately small and explicit: the selected-model
hypothesis smoke must pass, and the same allocation must complete the real
Study-13 integration preflight for Cartesian, MET-residual and MET-exact.
This artifact is evidence for the campaign controller, not a scientific
performance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

from joint_event_reconstruction.runtime_paths import runtime_paths

from scripts.preflight_study13 import run as run_integration
from scripts.smoke_hypothesis_gpu import run as run_smoke


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _required_components(smoke: dict, integration: dict) -> dict[str, bool]:
    branches = integration.get("branches", {})
    return {
        "model_composition": smoke.get("model_composition_status") == "PASS"
        and Path(smoke.get("model_composition", "")).is_file(),
        "hypothesis_smoke": smoke.get("status") == "PASS",
        "cartesian_integration": (
            isinstance(branches.get("cartesian"), dict)
            and branches["cartesian"].get("prediction", {}).get("physical_four_vectors") is True
            and isinstance(branches["cartesian"].get("evaluation"), dict)
        ),
        "met_residual_integration": (
            isinstance(branches.get("met_residual"), dict)
            and branches["met_residual"].get("training", {}).get("finite_initial_loss") is True
            and branches["met_residual"].get("training", {}).get("finite_gradients") is True
            and branches["met_residual"].get("training", {}).get("nonzero_invisible_update") is True
            and branches["met_residual"].get("training", {}).get("finite_validation_loss") is True
            and branches["met_residual"].get("checkpoint_reload", {}).get("strict_reload") is True
            and branches["met_residual"].get("checkpoint_reload", {}).get("frozen_visible_equal", {}).get("equal") is True
            and branches["met_residual"].get("checkpoint_reload", {}).get("frozen_assignment_equal", {}).get("equal") is True
            and isinstance(branches["met_residual"].get("evaluation"), dict)
        ),
        "met_exact_integration": (
            isinstance(branches.get("met_exact"), dict)
            and branches["met_exact"].get("training", {}).get("finite_initial_loss") is True
            and branches["met_exact"].get("training", {}).get("finite_gradients") is True
            and branches["met_exact"].get("training", {}).get("nonzero_invisible_update") is True
            and branches["met_exact"].get("training", {}).get("finite_validation_loss") is True
            and branches["met_exact"].get("checkpoint_reload", {}).get("strict_reload") is True
            and branches["met_exact"].get("checkpoint_reload", {}).get("frozen_visible_equal", {}).get("equal") is True
            and branches["met_exact"].get("checkpoint_reload", {}).get("frozen_assignment_equal", {}).get("equal") is True
            and branches["met_exact"].get("prediction", {}).get("met_exact_closure") is True
            and isinstance(branches["met_exact"].get("evaluation"), dict)
        ),
        "schema_and_metric": all(
            isinstance(branches.get(name), dict)
            and isinstance(branches[name].get("evaluation"), dict)
            and branches[name]["evaluation"].get("primary_metric_path")
            == "invisible.top1_assignment_conditioned.energy_score_joint"
            for name in ("cartesian", "met_residual", "met_exact")
        ),
    }


def run(
    *,
    output_dir: Path,
    device: str = "cuda",
    expected_source_sha: str | None = None,
    gates_path: Path | None = None,
) -> dict:
    paths = runtime_paths()
    if expected_source_sha is not None and paths.source_sha != expected_source_sha:
        raise RuntimeError(
            f"SOURCE SHA MISMATCH: expected {expected_source_sha}, found {paths.source_sha}"
        )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_dir = output_dir / "hypothesis-smoke"
    integration_dir = output_dir / "study13-integration"
    result_path = output_dir / "result.json"
    try:
        run_smoke(smoke_dir, device, expected_source_sha=expected_source_sha)
        smoke_path = smoke_dir / "smoke_result.json"
        smoke = json.loads(smoke_path.read_text())
        prediction = smoke_dir / "prediction.h5"
        integration = run_integration(
            output_dir=integration_dir,
            device=device,
            smoke_prediction=prediction,
            gates_path=gates_path,
            expected_source_sha=expected_source_sha,
        )
        required = _required_components(smoke, integration)
        if not all(required.values()):
            raise RuntimeError(f"Composite release checks are incomplete: {required}")
        composition_path = Path(smoke["model_composition"])
        composition = json.loads(composition_path.read_text())
        result = {
            "schema": "jer_study13_release_v1",
            "status": "PASS",
            "source_sha": paths.source_sha,
            "source_root": str(paths.source_root),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_terminal_state": "RUNNING",
            "topology": "ttbar2L",
            "study_manifest": integration["study_manifest"],
            "study_manifest_sha256": integration["study_manifest_sha256"],
            "resolved_gate_digest": integration["resolved_gate_digest"],
            "checkpoint_hashes_and_training_identities": {
                "selected_visible": composition["selected_visible"],
                "selected_combined_cfm": composition["selected_combined_cfm"],
            },
            "data_subset": integration["data_subset"],
            "model_composition": str(composition_path),
            "smoke": {
                "result": str(smoke_path),
                "prediction": str(prediction),
                "prediction_sha256": _sha256(prediction),
                "status": smoke["status"],
            },
            "study13_integration": {
                "result": str(integration_dir / "result.json"),
                "status": integration["status"],
                "branches": {
                    name: {
                        "training": branch.get("training", {}),
                        "validation": branch.get("validation", {}),
                        "checkpoint_reload": branch.get("checkpoint_reload", {}),
                        "prediction": branch.get("prediction", {}),
                        "evaluation": branch.get("evaluation", {}),
                    }
                    for name, branch in integration["branches"].items()
                },
            },
            "required_components": required,
            "metric_contract": integration["metric_contract"],
            "software_integration_evidence_only": True,
        }
        _write_json(result_path, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    except Exception as error:
        failure = {
            "schema": "jer_study13_release_v1",
            "status": "FAIL",
            "source_sha": paths.source_sha,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_terminal_state": "FAILED",
            "error": str(error),
            "traceback": traceback.format_exc(),
            "required_components": {},
        }
        _write_json(result_path, failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--expected-source-sha")
    parser.add_argument("--gates-path", type=Path)
    args = parser.parse_args()
    result = run(
        output_dir=args.output_dir,
        device=args.device,
        expected_source_sha=args.expected_source_sha,
        gates_path=args.gates_path.expanduser().resolve() if args.gates_path else None,
    )
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
