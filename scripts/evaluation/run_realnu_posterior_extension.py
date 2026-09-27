#!/usr/bin/env python3
"""Generate one nested 64-sample validation posterior per trained realnu model."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

MODES = ("Q0", "Q1", "PCFM_Q", "J0", "J_A", "J_AN")
STANDARD = ("Q0", "Q1", "J0", "J_A", "J_AN")
PARENT = ("PCFM_Q", "J0", "J_A", "J_AN")
SAMPLE_COUNT = 64


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _import_production_runner(source_root: Path):
    runner_path = source_root / "scripts/production/run_realnu_production.py"
    spec = importlib.util.spec_from_file_location(
        "realnu_production_runner", runner_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot import the accepted production runner: {runner_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(manifest_path: Path, output: Path, samples: int = SAMPLE_COUNT) -> dict:
    if samples < 64:
        raise ValueError(
            "The requested nested posterior study requires at least 64 samples"
        )
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "Full posterior extension must run inside its dependent Slurm job"
        )
    manifest_path = manifest_path.resolve()
    output = output.resolve()
    manifest = read_json(manifest_path)
    if manifest.get("source_commit") != manifest.get("g0_source_commit"):
        raise RuntimeError("Production manifest source differs from accepted G0 source")
    if (
        manifest.get("test_rows_read") != 0
        or manifest.get("inputs", {}).get("reserved_test_rows_materialised") != 0
    ):
        raise RuntimeError("Production manifest does not prove sealed-test exclusion")
    source_root = Path(manifest["source_root"]).resolve()
    project_root = source_root.parents[2]
    source_meta = read_json(source_root / "SOURCE_SNAPSHOT.json")
    if source_meta.get("commit_sha") != manifest["source_commit"]:
        raise RuntimeError(
            "Accepted source snapshot metadata and campaign manifest disagree"
        )
    os.environ["JER_REALNU_SOURCE_ROOT"] = str(source_root)
    os.environ["JER_REALNU_PROJECT_ROOT"] = str(project_root)
    os.environ["JER_REALNU_EXPECTED_SHA"] = str(manifest["source_commit"])
    os.environ["JER_REALNU_PRODUCTION_SOURCE_SHA"] = str(manifest["source_commit"])
    os.environ["JER_REALNU_PRODUCTION_RUN_MANIFEST"] = str(manifest_path)
    os.environ["JER_PROJECT_ROOT"] = str(project_root)
    os.environ["JER_SOURCE_ROOT"] = str(source_root)
    os.environ["JER_SOURCE_SHA"] = str(manifest["source_commit"])
    os.environ["JER_SOURCE_BRANCH"] = str(manifest["source_branch"])
    os.environ["PYTHONPATH"] = (
        str(source_root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    sys.path.insert(0, str(source_root / "src"))
    runner = _import_production_runner(source_root)
    runner.load_run_manifest(manifest_path, manifest["source_commit"])
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated posterior-extension job has no CUDA device")
    device_name = torch.cuda.get_device_name(0)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite posterior extension output: {output}"
        )
    output.mkdir(parents=True)
    payload = {
        "status": "running",
        "campaign": manifest["campaign"],
        "source_commit": manifest["source_commit"],
        "run_manifest": str(manifest_path),
        "sample_count": int(samples),
        "sample_prefixes": [1, 4, 8, 16, 32, 64],
        "gpu": device_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "started_utc": datetime.now(UTC).isoformat(),
        "products": {},
        "reserved_test_rows_read": 0,
    }
    atomic_json(output / "posterior64_manifest.json", payload)
    try:
        for mode in MODES:
            entry = manifest["modes"][mode]
            existing_validation = read_json(
                Path(entry["validation_output"]) / "validation.json"
            )
            if existing_validation.get("status") != "passed":
                raise RuntimeError(
                    f"{mode} full-production validation is not complete and passed"
                )
            training_output = Path(entry["training_output"])
            training = read_json(training_output / "production_training.json")
            if (
                training.get("status") != "passed"
                or training.get("source_commit") != manifest["source_commit"]
            ):
                raise RuntimeError(
                    f"{mode} has no successful checkpoint from the accepted source"
                )
            checkpoint = Path(training["checkpoint"])
            if not checkpoint.is_file() or runner.sha256(checkpoint) != training.get(
                "checkpoint_sha256"
            ):
                raise RuntimeError(
                    f"{mode} production checkpoint is missing or its SHA changed"
                )
            config, _, _ = runner.load_and_validate_config(
                mode, Path(entry["config_path"]), training_output
            )
            config.inference.posterior_samples = int(samples)
            if mode == "J_AN" and (
                str(config.model.parent_cfm.conditioning) != "P2"
                or int(config.model.parent_cfm.p2_neutrino_steps) != 8
                or str(config.model.parent_cfm.p2_neutrino_solver) != "heun"
            ):
                raise RuntimeError(
                    "J_AN must retain live P2 conditioning with one 8-step Heun neutrino sample"
                )
            mode_root = output / mode
            mode_root.mkdir()
            provenance = runner.artifact_provenance(
                mode,
                config,
                manifest,
                gpu_type=device_name,
                training_record=training,
            )
            product = {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": training["checkpoint_sha256"],
                "posterior_samples": int(samples),
            }
            if mode in STANDARD:
                standard_root = mode_root / "standard"
                standard_root.mkdir()
                standard = runner._make_standard_prediction(
                    mode, config, checkpoint, standard_root, provenance
                )
                product["standard"] = standard
            if mode in PARENT:
                parent_root = mode_root / "parent"
                parent_root.mkdir()
                parent = runner._make_parent_prediction(
                    mode, config, checkpoint, parent_root, provenance
                )
                if mode == "J_AN":
                    prediction_path = Path(parent["prediction"])
                    import h5py

                    with h5py.File(prediction_path, "r") as prediction:
                        contract = json.loads(
                            str(prediction.attrs["conditioning_contract"])
                        )
                        invisible_indices = np.asarray(
                            prediction["CONDITIONING/invisible_sample_index"][:],
                            dtype=np.int64,
                        )
                        parent_indices = np.asarray(
                            prediction["CONDITIONING/parent_sample_index"][:],
                            dtype=np.int64,
                        )
                        assignment_indices = np.asarray(
                            prediction["CONDITIONING/assignment_hypothesis_index"][:],
                            dtype=np.int64,
                        )
                        if (
                            prediction.attrs.get("conditioning") != "P2"
                            or contract.get("invisible_context")
                            != "predicted_CFM_sample_top_antitop_role_order_one_sample_per_event"
                            or int(prediction.attrs.get("solver_steps", -1)) != 32
                            or int(prediction.attrs.get("nfe", -1)) != 64
                            or json.loads(
                                str(prediction.attrs.get("invisible_role_order", "[]"))
                            )
                            != ["top.nu", "antitop.nu"]
                            or not np.all(invisible_indices == 0)
                            or not np.all(assignment_indices == -1)
                            or not np.array_equal(
                                parent_indices,
                                np.broadcast_to(
                                    np.arange(samples, dtype=np.int64),
                                    parent_indices.shape,
                                ),
                            )
                        ):
                            raise RuntimeError(
                                "J_AN full posterior lost live P2 sample association or 8-step role contract"
                            )
                    parent["p2_invisible_context"] = contract["invisible_context"]
                    parent["p2_invisible_sample_index"] = (
                        "invisible_sample_index=0; one live predicted sample reused for all parent draws"
                    )
                product["parent"] = parent
            payload["products"][mode] = product
            payload["completed_modes"] = list(payload["products"])
            payload["updated_utc"] = datetime.now(UTC).isoformat()
            atomic_json(output / "posterior64_manifest.json", payload)
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "status": "passed",
                        "posterior_samples": samples,
                        "products": product,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        payload["status"] = "passed"
        payload["completed_utc"] = datetime.now(UTC).isoformat()
        atomic_json(output / "posterior64_manifest.json", payload)
        return payload
    except Exception as error:
        payload["status"] = "failed"
        payload["error"] = f"{type(error).__name__}: {error}"
        payload["failed_utc"] = datetime.now(UTC).isoformat()
        atomic_json(output / "posterior64_manifest.json", payload)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=SAMPLE_COUNT)
    args = parser.parse_args()
    payload = run(args.manifest, args.output, args.samples)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output.resolve()),
                "gpu": payload["gpu"],
                "source_sha": payload["source_commit"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
