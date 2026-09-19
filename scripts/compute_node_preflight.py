#!/usr/bin/env python3
"""Cheap compute-node contract check used before allocating an L40S."""

from __future__ import annotations

import argparse
import json
import os

import h5py

from joint_event_reconstruction.runtime_paths import runtime_paths, verify_source_snapshot


def _finite_tensors(value) -> bool:
    import torch

    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tensors(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_tensors(item) for item in value)
    return True


def run(events: int) -> dict:
    paths = runtime_paths()
    source = verify_source_snapshot(
        paths.source_sha,
        project_root=paths.project_root,
        source_root=paths.source_root,
    )
    dataset = paths.project("data/real/ttbar2L_canonical.h5")
    if not dataset.is_file() or dataset.stat().st_size <= 0:
        raise FileNotFoundError(f"Canonical ttbar2L dataset is missing or empty: {dataset}")
    with h5py.File(dataset, "r") as handle:
        required = {"EVENT_ID", "SPLIT", "INPUTS", "LABELS"}
        missing = sorted(required - set(handle))
        if missing:
            raise KeyError(f"Canonical dataset is missing groups/datasets {missing}: {dataset}")
    if not paths.cache_root.is_dir() or not os.access(paths.cache_root, os.R_OK):
        raise PermissionError(f"Canonical cache root is not accessible: {paths.cache_root}")
    cache_file = paths.cache_root / "ttbar2L_canonical.pt"
    if not cache_file.is_file() or not os.access(cache_file, os.R_OK):
        raise FileNotFoundError(f"Canonical tensor cache is not accessible: {cache_file}")

    from joint_event_reconstruction.cli.run import build_data
    from joint_event_reconstruction.configuration import load_experiment, load_visible_config
    from joint_event_reconstruction.studies import resolve_study
    from scripts.run_study import _gate_checkpoint

    experiment_path = paths.source("configs/experiments/ttbar2L.yaml")
    config = load_experiment(
        experiment_path,
        overrides=[
            f"data.events.train={int(events)}",
            f"data.events.validation={int(events)}",
            f"data.events.test={int(events)}",
            "data.cache=null",
            "training.workers=0",
        ],
    )
    visible = load_visible_config(config)
    data = build_data(config, visible)
    data.setup("predict")
    batch = next(iter(data.val_dataloader()))
    if not _finite_tensors(batch):
        raise FloatingPointError("Compute-node validation batch contains non-finite values")

    study = resolve_study(paths.source("configs/studies/13_invisible_coordinates.yaml"))
    checkpoint_contracts = {}
    for gate_name in ("visible_assignment", "invisible_generator"):
        checkpoint = _gate_checkpoint(study, gate_name, "ttbar2L", 42)
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload.get("state_dict"), dict):
            raise ValueError(f"Checkpoint has no state_dict: {checkpoint}")
        checkpoint_contracts[gate_name] = str(checkpoint)

    from scripts.preflight_study13 import audit_level_a

    level_a_directory = (
        paths.results_root
        / "validation/study13-level-a"
        / paths.source_sha
        / f"job-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    level_a = audit_level_a(output_dir=level_a_directory)

    result = {
        "status": "PASS",
        "source_sha": paths.source_sha,
        "source_root": str(source),
        "project_root": str(paths.project_root),
        "dataset": str(dataset),
        "cache_root": str(paths.cache_root),
        "cache_file": str(cache_file),
        "validation_events": int(events),
        "checkpoint_contracts": checkpoint_contracts,
        "study13_level_a": level_a,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=int, default=2)
    args = parser.parse_args()
    if args.events <= 0:
        raise SystemExit("--events must be positive")
    try:
        run(args.events)
    except Exception as error:
        print(f"COMPUTE_NODE_PREFLIGHT_FAIL: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
