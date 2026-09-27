#!/usr/bin/env python3
"""Resolve and validate the immutable realnu G0 execution contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

SOURCE_BRANCH = "research/realnu-training-hotpath-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_file(value: str, project_root: Path, source_root: Path) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [project_root / path, source_root / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Configured scaler/checkpoint file is missing: {value}")


def _external_file_references(config_path: Path, project_root: Path, source_root: Path) -> tuple[list[dict], list[dict]]:
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    checkpoints = []
    scalers = []

    def visit(value, key_path: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{key_path}.{key}" if key_path else str(key)
                visit(child, child_path)
            return
        if not isinstance(value, str):
            return
        lowered = key_path.lower()
        if not any(token in lowered for token in ("checkpoint", "scaler", "initialization", "init_from")):
            return
        path_like = "/" in value or "\\" in value or Path(value).suffix.lower() in {
            ".ckpt", ".pt", ".pth", ".npz", ".json"
        }
        if not path_like:
            return
        resolved = _resolve_file(value, project_root, source_root)
        record = {"config_key": key_path, "path": str(resolved), "sha256": sha256(resolved)}
        (checkpoints if "checkpoint" in lowered or "initialization" in lowered or "init_from" in lowered else scalers).append(record)

    visit(config)
    return checkpoints, scalers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--g0-root", type=Path, required=True)
    parser.add_argument("--contract-output", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve(strict=True)
    project_root = args.project_root.expanduser().resolve(strict=True)
    python_bin = args.python.expanduser().resolve(strict=True)
    attempt_root = args.attempt_root.expanduser().resolve()
    g0_root = args.g0_root.expanduser().resolve()
    contract_output = args.contract_output.expanduser().resolve()
    if len(args.expected_sha) != 40 or any(char not in "0123456789abcdef" for char in args.expected_sha):
        raise ValueError("--expected-sha must be a full lowercase Git SHA")
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise FileNotFoundError(f"Configured Python interpreter is missing or not executable: {python_bin}")
    runner = source_root / "scripts/production/run_realnu_gpu_g0.py"
    if not runner.is_file():
        raise FileNotFoundError(f"G0 runner is missing under the explicit source root: {runner}")
    if source_root != project_root / "runtime/source" / args.expected_sha:
        raise RuntimeError("G0 source root must be the immutable runtime/source/<SHA> snapshot")
    if g0_root != attempt_root / "G0":
        raise RuntimeError("G0 output root must be the G0 child of the unique attempt root")
    if contract_output != attempt_root / "resolved_execution_contract.json":
        raise RuntimeError("Resolved contract must be stored at the fixed attempt-root path")
    if attempt_root.exists():
        raise FileExistsError(f"Refusing to reuse realnu G0 attempt root: {attempt_root}")
    if g0_root.exists():
        raise FileExistsError(f"Refusing to reuse realnu G0 output root: {g0_root}")

    metadata_path = source_root / "SOURCE_SNAPSHOT.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Immutable source snapshot metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    project_sha = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    observed_branch = subprocess.check_output(
        ["git", "-C", str(project_root), "branch", "--show-current"], text=True
    ).strip()
    try:
        observed_tree = subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", f"{args.expected_sha}^{{tree}}"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Expected release commit is unavailable in the canonical project repository: {args.expected_sha}"
        ) from error
    dirty = subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(
            f"Canonical project checkout is dirty: sha={project_sha}, branch={observed_branch}, dirty={dirty!r}"
        )
    if (
        metadata.get("schema") != "jer_source_snapshot_v1"
        or metadata.get("commit_sha") != args.expected_sha
        or metadata.get("tree_sha") != observed_tree
        or (source_root / ".git").exists()
    ):
        raise RuntimeError("Immutable source snapshot metadata does not match the expected commit and tree")

    os.environ["JER_REALNU_PROJECT_ROOT"] = str(project_root)
    os.environ["JER_REALNU_SOURCE_ROOT"] = str(source_root)
    os.environ["JER_PROJECT_ROOT"] = str(project_root)
    os.environ["JER_SOURCE_ROOT"] = str(source_root)
    os.environ["PYTHONPATH"] = str(source_root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    sys.path.insert(0, str(source_root / "scripts/production"))
    sys.path.insert(0, str(source_root / "src"))
    import run_realnu_gpu_g0 as g0

    g0._assert_source(args.expected_sha)
    resolved = g0._preflight()
    if resolved.get("status") != "passed":
        raise RuntimeError("The authoritative realnu G0 input/config preflight did not pass")
    if resolved.get("invisible_roles") != ["top.nu", "antitop.nu"]:
        raise RuntimeError("Resolved G0 preflight has the wrong neutrino role order")
    if resolved.get("parent_roles") != ["top", "antitop"]:
        raise RuntimeError("Resolved G0 preflight has the wrong parent role order")
    if int(resolved.get("reserved_test_rows_materialised", -1)) != 0:
        raise RuntimeError("Reserved test rows are materialised in the G0 input contract")

    scaler_files = []
    initialisation_checkpoints = []
    all_config_records = {
        f"G0_{mode}": record for mode, record in resolved["configs"].items()
    } | {
        f"PRODUCTION_{mode}": record
        for mode, record in resolved["production_configs"].items()
    }
    for config_record in all_config_records.values():
        config_path = Path(config_record["path"]).resolve()
        if not config_path.is_file() or sha256(config_path) != config_record["sha256"]:
            raise RuntimeError(f"G0 config changed after preflight: {config_path}")
        checkpoints, scalers = _external_file_references(config_path, project_root, source_root)
        initialisation_checkpoints.extend(checkpoints)
        scaler_files.extend(scalers)
    initialisation_manifest = resolved["initialisation_manifest"]
    initialisation_checkpoints.extend(
        {
            "config_key": f"initialisation.{mode}",
            "path": record["path"],
            "sha256": record["sha256"],
        }
        for mode, record in resolved["initialisation_checkpoints"].items()
    )

    measured = Path(resolved["measured_input"]).resolve()
    truth = Path(resolved["truth_store"]).resolve()
    cache = Path(resolved["cache"]).resolve()
    source_manifest = project_root / "results/experiments/partonic_ttbar2L_2M_realnu_v1/source_manifest_v1.json"
    split_membership = project_root / "results/experiments/partonic_ttbar2L_2M_realnu_v1/split_membership_v1.npz"
    cache_metadata = cache.with_suffix(".json")
    for required in (measured, truth, cache, source_manifest, split_membership, cache_metadata):
        if not required.is_file():
            raise FileNotFoundError(f"Resolved G0 input is missing: {required}")

    site = OmegaConf.to_container(OmegaConf.load(project_root / "configs/site/csf3.yaml"), resolve=True)
    contract = {
        "schema": "realnu_g0_execution_contract_v1",
        "source_root": str(source_root),
        "project_root": str(project_root),
        "source_sha": args.expected_sha,
        "source_branch": SOURCE_BRANCH,
        "project_root_git_sha": project_sha,
        "project_root_git_branch": observed_branch,
        "python": str(python_bin),
        "g0_runner": str(runner.resolve()),
        "attempt_root": str(attempt_root),
        "g0_root": str(g0_root),
        "contract_path": str(contract_output),
        "measured_cache": {"path": str(cache), "sha256": sha256(cache), "metadata_sha256": sha256(cache_metadata)},
        "measured_input": {"path": str(measured), "sha256": resolved["measured_input_sha256"]},
        "truth_sidecar": {"path": str(truth), "sha256": resolved["truth_store_sha256"]},
        "source_manifest": {"path": str(source_manifest), "sha256": resolved["source_manifest_sha256"]},
        "split_membership": {"path": str(split_membership), "sha256": resolved["split_membership_sha256"]},
        "invisible_role_order": resolved["invisible_roles"],
        "parent_role_order": resolved["parent_roles"],
        "configs": resolved["configs"],
        "production_configs": resolved["production_configs"],
        "initialisation_manifest": initialisation_manifest,
        "initialisation_checkpoints": initialisation_checkpoints,
        "scaler_files": scaler_files,
        "sealed_test_materialised": False,
        "resources": {
            "partition": str(site["slurm"]["gpu_partition"]),
            "gres": str(site["slurm"]["gpu_gres"]),
            "cpus_per_task": int(site["slurm"]["gpu_cpus"]),
            "memory": str(site["slurm"]["gpu_memory"]),
            "time_limit": str(site["slurm"]["gpu_time"]),
        },
        "g0_preflight": resolved,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    contract_output.parent.mkdir(parents=True, exist_ok=False)
    contract_output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
