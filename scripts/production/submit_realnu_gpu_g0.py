#!/usr/bin/env python3
"""Preflight and submit exactly one immutable realnu GPU G0 attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON_DEFAULT = Path("/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python")
WRAPPER = ROOT / "scripts/production/run_realnu_gpu_g0.sh"
PREFLIGHT = ROOT / "scripts/production/preflight_realnu_gpu_g0.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=PYTHON_DEFAULT)
    args = parser.parse_args()

    project_root = ROOT.resolve()
    source_root = args.source_root.expanduser().resolve(strict=True)
    expected_sha = args.expected_sha
    python_bin = args.python.expanduser().resolve(strict=True)
    attempt_root = args.attempt_root.expanduser().resolve()
    g0_root = attempt_root / "G0"
    contract_path = attempt_root / "resolved_execution_contract.json"
    submission_path = attempt_root / "submission.json"
    if not WRAPPER.is_file() or not os.access(WRAPPER, os.X_OK):
        raise FileNotFoundError(f"G0 Slurm wrapper is missing or not executable: {WRAPPER}")
    if not PREFLIGHT.is_file():
        raise FileNotFoundError(f"G0 preflight runner is missing: {PREFLIGHT}")

    preflight_env = os.environ.copy()
    preflight_env.update(
        {
            "JER_REALNU_PROJECT_ROOT": str(project_root),
            "JER_REALNU_SOURCE_ROOT": str(source_root),
            "JER_REALNU_EXPECTED_SHA": expected_sha,
            "JER_PROJECT_ROOT": str(project_root),
            "JER_SOURCE_ROOT": str(source_root),
            "PYTHONPATH": str(source_root / "src"),
        }
    )
    preflight_command = [
        str(python_bin), str(PREFLIGHT),
        "--source-root", str(source_root),
        "--project-root", str(project_root),
        "--expected-sha", expected_sha,
        "--python", str(python_bin),
        "--attempt-root", str(attempt_root),
        "--g0-root", str(g0_root),
        "--contract-output", str(contract_path),
    ]
    completed = subprocess.run(
        preflight_command, cwd=project_root, env=preflight_env,
        check=False, text=True, capture_output=True,
    )
    if completed.returncode:
        raise RuntimeError(
            "Realnu G0 submission preflight failed before sbatch.\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    contract = json.loads(contract_path.read_text())
    if contract.get("source_sha") != expected_sha or contract.get("g0_root") != str(g0_root):
        raise RuntimeError("Written G0 execution contract differs from the requested submission")

    scheduler = contract["resources"]
    job_name = f"realnu-g0-{expected_sha[:8]}"
    for command in (
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-n", job_name, "-o", "%i|%j|%T"],
        ["sacct", "-X", "-n", "-P", "-u", os.environ.get("USER", ""), "--name", job_name,
         "--format=JobIDRaw,JobName,State,ExitCode"],
    ):
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"Cannot reconcile the unique G0 job name before submission: {command!r}\n"
                f"{result.stderr.strip()}"
            )
        if result.returncode == 0 and result.stdout.strip():
            raise FileExistsError(
                f"A Slurm job with the frozen G0 name already exists; refusing duplicate submission: "
                f"{job_name}\n{result.stdout.strip()}"
            )
    if submission_path.exists():
        raise FileExistsError(f"Refusing to reuse a realnu G0 submission record: {submission_path}")

    log_dir = attempt_root / "slurm"
    log_dir.mkdir()
    command = [
        "sbatch", "--parsable",
        "--partition", scheduler["partition"],
        "--gres", scheduler["gres"],
        "--cpus-per-task", str(scheduler["cpus_per_task"]),
        "--mem", scheduler["memory"],
        "--time", scheduler["time_limit"],
        "--account", "gpu-free", "--qos", "gpu-free",
        "--job-name", job_name,
        "--comment", f"realnu_g0:{expected_sha}",
        "--chdir", str(project_root),
        "--output", str(log_dir / "G0-%j.out"),
        "--export=" + ",".join(
            (
                "ALL",
                f"JER_REALNU_SOURCE_ROOT={source_root}",
                f"JER_REALNU_PROJECT_ROOT={project_root}",
                f"JER_REALNU_EXPECTED_SHA={expected_sha}",
                f"JER_REALNU_G0_ROOT={g0_root}",
                f"JER_REALNU_PREFLIGHT_CONTRACT={contract_path}",
                f"JER_PYTHON={python_bin}",
            )
        ),
        str(WRAPPER),
    ]
    record = {
        "schema": "realnu_g0_submission_v1",
        "attempt_root": str(attempt_root),
        "source_sha": expected_sha,
        "source_root": str(source_root),
        "execution_contract": str(contract_path),
        "execution_contract_sha256": sha256(contract_path),
        "job_name": job_name,
        "command": command,
        "submission_state": "SUBMITTING",
        "submitted_utc": datetime.now(UTC).isoformat(),
    }
    atomic_json(submission_path, record)
    submitted = subprocess.run(command, cwd=project_root, text=True, capture_output=True, check=False)
    if submitted.returncode:
        record.update(
            {
                "submission_state": "SUBMISSION_UNCERTAIN",
                "stdout": submitted.stdout,
                "stderr": submitted.stderr,
            }
        )
        atomic_json(submission_path, record)
        raise RuntimeError(
            f"sbatch did not return a job ID. Reconcile Slurm by job name {job_name}; do not retry.\n"
            f"stdout:\n{submitted.stdout}\nstderr:\n{submitted.stderr}"
        )
    job_id = submitted.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        record.update(
            {
                "submission_state": "SUBMISSION_UNCERTAIN",
                "stdout": submitted.stdout,
                "stderr": submitted.stderr,
            }
        )
        atomic_json(submission_path, record)
        raise RuntimeError(f"Unexpected sbatch response for {job_name}: {submitted.stdout!r}")
    record.update(
        {
            "job_id": job_id,
            "sbatch_response": submitted.stdout.strip(),
            "submission_state": "ACCEPTED",
            "accepted_utc": datetime.now(UTC).isoformat(),
        }
    )
    atomic_json(submission_path, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
