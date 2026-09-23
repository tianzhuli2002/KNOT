#!/usr/bin/env python3
"""Submit the matched P1--P5 campaign after a validated GPU smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "results/experiments/vyper_jer_30_4_matched_20260923"
MANIFEST = EXPERIMENT / "data_contract/matched_event_manifest.h5"
VYPER_ROOT = ROOT.parent / "VyPER_matched_upstream"
VYPER_CONFIG = VYPER_ROOT / "configs/matched_30_4.yaml"
VYPER_PYTHON = Path(
    "/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/envs/VyPER/bin/python"
)
JER_PYTHON = Path(
    "/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def submit(command: list[str], wrap: str) -> str:
    result = subprocess.run(
        ["sbatch", "--parsable", *command, "--wrap", wrap],
        check=True,
        text=True,
        capture_output=True,
    )
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected sbatch response: {result.stdout!r}")
    return job_id


def empty_or_absent(path: Path) -> None:
    if path.exists() and any(item.is_file() for item in path.rglob("*")):
        raise FileExistsError(f"Production output is not empty: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-smoke-job", required=True)
    parser.add_argument("--partition", default="gpuL")
    parser.add_argument("--gres", default="gpu:1")
    parser.add_argument("--cpus-per-task", type=int, default=12)
    parser.add_argument("--mem", default="96G")
    parser.add_argument("--time", default="4-00:00:00")
    parser.add_argument("--account", default="gpu-free")
    parser.add_argument("--qos", default="gpu-free")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.gpu_smoke_job.isdigit():
        raise ValueError("--gpu-smoke-job must be a numeric Slurm job ID")
    if not MANIFEST.is_file() or not VYPER_CONFIG.is_file():
        raise FileNotFoundError("Matched manifest or upstream production config is missing")
    source_sha = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip():
        raise RuntimeError("JER checkout has tracked modifications; freeze it before submission")
    manifest_sha = sha256(MANIFEST)
    config_names = {
        "P1": "matched_upstream",
        "P2": "jer_visible_assignment.public.yaml",
        "P3": "jer_reference_factorized.public.yaml",
        "P4": "jer_joint.public.yaml",
        "P5": "jer_continued_control.public.yaml",
    }
    configs = {
        key: (VYPER_CONFIG if key == "P1" else EXPERIMENT / "configs" / name)
        for key, name in config_names.items()
    }
    config_hashes = {key: sha256(path) for key, path in configs.items()}
    outputs = {
        "P1": EXPERIMENT / "training/vyper_reference",
        "P2": EXPERIMENT / "training/jer_visible_assignment",
        "P3": EXPERIMENT / "training/jer_factorized",
        "P4": EXPERIMENT / "training/jer_joint",
        "P5": EXPERIMENT / "training/jer_continued_control",
    }
    for output in outputs.values():
        empty_or_absent(output)

    common_slurm = [
        "--partition", args.partition,
        "--gres", args.gres,
        "--cpus-per-task", str(args.cpus_per_task),
        "--mem", args.mem,
        "--time", args.time,
        "--account", args.account,
        "--qos", args.qos,
    ]
    submission_dir = EXPERIMENT / "submission_records"
    submission_dir.mkdir(parents=True, exist_ok=True)
    log_dir = submission_dir / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)
    jer_wrapper = ROOT / "scripts/production/run_jer_production_stage.py"
    vyper_wrapper = ROOT / "scripts/production/run_vyper_production.py"
    manifest_arg = f"--manifest {MANIFEST} --manifest-sha256 {manifest_sha}"

    p1_wrap = (
        f"cd {ROOT} && {JER_PYTHON} {vyper_wrapper} "
        f"--vyper-root {VYPER_ROOT} --config {VYPER_CONFIG} "
        f"--config-sha256 {config_hashes['P1']} --upstream-sha 7a6263c366e06dd31cb714c52dec243f66d03f8b "
        f"{manifest_arg} --python {VYPER_PYTHON} --config-name matched_30_4 "
        f"--output {outputs['P1']}"
    )
    p2_wrap = (
        f"cd {ROOT} && {JER_PYTHON} {jer_wrapper} --repo-root {ROOT} "
        f"--config {configs['P2']} --config-sha256 {config_hashes['P2']} "
        f"{manifest_arg} --source-sha {source_sha} --expected-output {outputs['P2']} --device cuda"
    )
    visible_checkpoint = outputs["P2"] / "checkpoints/best-loss.ckpt"
    downstream = {}
    for key in ("P3", "P4", "P5"):
        downstream[key] = (
            f"cd {ROOT} && {JER_PYTHON} {jer_wrapper} --repo-root {ROOT} "
            f"--config {configs[key]} --config-sha256 {config_hashes[key]} "
            f"{manifest_arg} --source-sha {source_sha} --expected-output {outputs[key]} --device cuda "
            f"--initial-checkpoint {visible_checkpoint} "
            f"--initial-checkpoint-sha256-from-file {visible_checkpoint} "
            f"--initial-components visible assignment"
        )
    # Replace the convenience token with a value fixed at submission time.  It
    # is deliberately not computed by a dependent job after the checkpoint is
    # produced; the dependent wrapper checks the same path and metadata again.
    # The token is expanded below once P2 has a submitted, immutable path.
    for key in downstream:
        downstream[key] = downstream[key].replace(
            f"--initial-checkpoint-sha256-from-file {visible_checkpoint}",
            f"--initial-checkpoint-sha256 $(sha256sum {visible_checkpoint} | cut -d' ' -f1)",
        )

    base = {
        "source_sha": source_sha,
        "upstream_vyper_sha": "7a6263c366e06dd31cb714c52dec243f66d03f8b",
        "manifest": str(MANIFEST),
        "manifest_sha256": manifest_sha,
        "gpu_smoke_job": args.gpu_smoke_job,
        "configs": {key: {"path": str(path), "sha256": config_hashes[key]} for key, path in configs.items()},
        "outputs": {key: str(path) for key, path in outputs.items()},
        "slurm": {
            "partition": args.partition,
            "gres": args.gres,
            "cpus_per_task": args.cpus_per_task,
            "mem": args.mem,
            "time": args.time,
            "account": args.account,
            "qos": args.qos,
        },
    }
    if args.dry_run:
        print(json.dumps({**base, "wraps": {"P1": p1_wrap, "P2": p2_wrap, **downstream}}, indent=2, sort_keys=True))
        return

    jobs = {}
    jobs["P1"] = submit(
        [*common_slurm, "--dependency", f"afterok:{args.gpu_smoke_job}", "--job-name", "vyper30-reference",
         "--output", str(log_dir / "P1-vyper-%j.out")],
        p1_wrap,
    )
    jobs["P2"] = submit(
        [*common_slurm, "--dependency", f"afterok:{args.gpu_smoke_job}", "--job-name", "jer30-visible",
         "--output", str(log_dir / "P2-visible-%j.out")],
        p2_wrap,
    )
    # P2's checkpoint hash is unknown until that job completes.  The shell
    # expansion is performed only by the dependent job after its afterok gate;
    # the immutable checkpoint path is still verified by the JER wrapper.
    for key in ("P3", "P4", "P5"):
        jobs[key] = submit(
            [*common_slurm, "--dependency", f"afterok:{jobs['P2']}", "--job-name", f"jer30-{key.lower()}",
             "--output", str(log_dir / f"{key}-%j.out")],
            downstream[key],
        )

    record = {
        **base,
        "jobs": jobs,
        "dependencies": {
            "P1": [args.gpu_smoke_job],
            "P2": [args.gpu_smoke_job],
            "P3": [jobs["P2"]],
            "P4": [jobs["P2"]],
            "P5": [jobs["P2"]],
        },
        "status_at_submission": {
            key: subprocess.check_output(
                ["squeue", "-h", "-j", job, "-o", "%T"], text=True
            ).strip()
            for key, job in jobs.items()
        },
    }
    path = submission_dir / "primary_jobs.json"
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite submission record: {path}")
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
