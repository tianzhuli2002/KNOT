"""Run a truth-side metric pass over bounded events from reused predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def _load_runner() -> object:
    source_root = Path(os.environ["JER_REALNU_SOURCE_ROOT"]).expanduser().resolve()
    project_root = Path(os.environ["JER_REALNU_PROJECT_ROOT"]).expanduser().resolve()
    os.environ["JER_REALNU_EXPECTED_SHA"] = "cb2b60445499f78421503d4fccc4a43cae1444fd"
    os.environ["JER_PROJECT_ROOT"] = str(project_root)
    os.environ["JER_SOURCE_ROOT"] = str(source_root)
    os.environ["JER_SOURCE_SHA"] = os.environ["JER_REALNU_EXPECTED_SHA"]
    sys.path.insert(0, str(source_root / "src"))
    runner_path = Path(__file__).resolve().parents[1] / "production/run_realnu_production.py"
    spec = importlib.util.spec_from_file_location("realnu_recovery_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import recovery evaluator: {runner_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(manifest_path: Path, modes: list[str], event_limit: int) -> dict:
    runner = _load_runner()
    source_sha = "cb2b60445499f78421503d4fccc4a43cae1444fd"
    recovery_sha = os.environ.get("JER_EVALUATION_RECOVERY_SHA", "working-tree")
    manifest = runner.load_run_manifest(manifest_path, source_sha)
    results = {}
    for mode in modes:
        entry = runner._mode_entry(manifest, mode)
        training_path = Path(entry["training_output"])
        training = runner._read_json(training_path / "production_training.json")
        config, _, _ = runner.load_and_validate_config(
            mode, Path(entry["config_path"]), training_path
        )
        prediction_path = Path(entry["validation_output"]) / "standard/validation.h5"
        reusable = runner.validate_reusable_standard_prediction(
            prediction_path,
            mode=mode,
            expected_sha=source_sha,
            manifest=manifest,
            training_record=training,
            config=config,
        )
        provenance = runner.artifact_provenance(
            mode,
            config,
            manifest,
            gpu_type="not used; bounded evaluation only",
            training_record=training,
        )
        metrics = runner.evaluate_standard_prediction(
            prediction_path,
            config,
            provenance,
            expected_event_count=runner.EXPECTED_EVENT_COUNT,
            event_limit=event_limit,
        )
        if metrics.get("events") != event_limit:
            raise RuntimeError(
                f"{mode} bounded evaluation returned {metrics.get('events')} events, "
                f"expected {event_limit}"
            )
        results[mode] = {
            "prediction_path": str(prediction_path.resolve()),
            "prediction_sha256": reusable["prediction_sha256"],
            "evaluation_recovery_sha": recovery_sha,
            "events": metrics["events"],
            "jet_multiplicity": metrics["jet_multiplicity"],
            "assignment_supported_events": metrics["assignment"]["supported_events"],
            "neutrino_roles": metrics["neutrino"]["role_order"],
        }
    return {
        "status": "passed",
        "source_commit": source_sha,
        "evaluation_recovery_sha": recovery_sha,
        "event_limit": event_limit,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mode", action="append", choices=("Q0", "Q1", "J0", "J_A", "J_AN"), required=True)
    parser.add_argument("--events", type=int, default=512)
    args = parser.parse_args()
    result = run(args.manifest.expanduser().resolve(), args.mode, args.events)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
