"""Submit the realnu R2 validation and evaluation dependency graph."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

VALIDATION_MODES = ("Q0", "Q1", "J0", "J_A", "J_AN")
MODES = ("Q0", "Q1", "PCFM_Q", "J0", "J_A", "J_AN")
TRAINING_SHA = "cb2b60445499f78421503d4fccc4a43cae1444fd"
GPU = {
    "partition": "gpuL",
    "gres": "gpu:l40s:1",
    "cpus": 12,
    "memory": "64G",
    "time": "4-00:00:00",
    "account": "gpu-free",
    "qos": "gpu-free",
}
CPU = {
    "partition": "multicore",
    "cpus": 8,
    "memory": "32G",
    "time": "2-00:00:00",
    "account": "fatpou01",
    "qos": "fatpou01",
}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _snapshot_sha(source_root: Path) -> str:
    metadata = json.loads((source_root / "SOURCE_SNAPSHOT.json").read_text())
    return str(metadata.get("commit_sha", ""))


def _validate_inputs(
    project_root: Path,
    training_source_root: Path,
    evaluation_source_root: Path,
    manifest_path: Path,
) -> tuple[dict, Path]:
    manifest = json.loads(manifest_path.read_text())
    evaluation_recovery = manifest.get("evaluation_recovery", {})
    training_sha = str(manifest.get("source_commit", ""))
    evaluation_sha = str(evaluation_recovery.get("source_sha", ""))
    if (
        training_sha != TRAINING_SHA
        or manifest.get("g0_source_commit") != TRAINING_SHA
        or manifest.get("source_branch") != "research/realnu-training-hotpath-v1"
        or evaluation_recovery.get("branch") != "research/realnu-evaluation-recovery-v2"
    ):
        raise RuntimeError("R2 manifest does not identify the accepted training and recovery releases")
    if set(manifest.get("modes", {})) != set(MODES):
        raise RuntimeError("R2 manifest does not contain exactly the six frozen modes")
    if training_source_root != project_root / "runtime/source" / training_sha:
        raise RuntimeError("Training source root is not the accepted immutable snapshot")
    if evaluation_source_root != project_root / "runtime/source" / evaluation_sha:
        raise RuntimeError("Evaluation source root is not the committed recovery snapshot")
    if _snapshot_sha(training_source_root) != training_sha:
        raise RuntimeError("Training source snapshot metadata differs from the R2 manifest")
    if _snapshot_sha(evaluation_source_root) != evaluation_sha:
        raise RuntimeError("Evaluation source snapshot metadata differs from the R2 manifest")
    r2_root = Path(evaluation_recovery.get("r2_root", "")).expanduser().resolve()
    if not r2_root.is_dir() or Path(evaluation_recovery.get("source_root", "")).resolve() != evaluation_source_root:
        raise RuntimeError("R2 output root or recovery source root differs from the manifest")
    if (
        manifest.get("test_rows_read") != 0
        or manifest.get("inputs", {}).get("reserved_test_rows_materialised") != 0
        or evaluation_recovery.get("sealed_test_used") is not False
    ):
        raise RuntimeError("R2 manifest does not prove sealed-test exclusion")
    if set(manifest.get("job_ids", {}).get("validation_r2", {})):
        raise RuntimeError("R2 validation jobs are already recorded; refusing duplicate submission")
    for mode in VALIDATION_MODES:
        entry = manifest["modes"][mode]
        prediction = Path(entry.get("reused_standard_prediction", ""))
        if not prediction.is_file():
            raise FileNotFoundError(f"{mode} standard prediction for strict reuse is missing: {prediction}")
        validation_root = Path(entry["validation_output"])
        if validation_root.exists():
            raise FileExistsError(
                f"Refusing to submit over existing {mode} R2 validation artifacts: {validation_root}"
            )
    pcfm_validation_path = Path(manifest["modes"]["PCFM_Q"]["validation_output"]) / "validation.json"
    pcfm_validation = json.loads(pcfm_validation_path.read_text())
    if (
        pcfm_validation.get("status") != "passed"
        or pcfm_validation.get("source_commit") != training_sha
        or pcfm_validation.get("reserved_test_rows_read") != 0
    ):
        raise RuntimeError("The existing PCFM_Q parent validation no longer passes its source contract")
    for mode in MODES:
        training_path = Path(manifest["modes"][mode]["training_output"])
        training = json.loads((training_path / "production_training.json").read_text())
        checkpoint = Path(training.get("checkpoint", ""))
        if training.get("status") != "passed" or not checkpoint.is_file():
            raise RuntimeError(f"{mode} accepted checkpoint is missing or not passed")
    if (r2_root.parent / "R_VALIDATION_R2.json").exists() or (
        r2_root.parent / "R_VALIDATION_R2.md"
    ).exists():
        raise FileExistsError("Refusing to replace an existing R_VALIDATION_R2 aggregate")
    return manifest, r2_root


def _slurm_command(
    *,
    project_root: Path,
    evaluation_source_root: Path,
    training_source_root: Path,
    manifest_path: Path,
    r2_root: Path,
    source_sha: str,
    evaluation_sha: str,
    python_bin: Path,
    log_path: Path,
    task: str,
    stage: str,
    resources: dict,
    mode: str | None = None,
    dependency: str | None = None,
) -> list[str]:
    label = f"rn-r2-{task.lower().replace('_', '-')}-{mode.lower()}" if mode else f"rn-r2-{task.lower().replace('_', '-')}"
    command = [
        "sbatch",
        "--parsable",
        f"--partition={resources['partition']}",
        f"--cpus-per-task={resources['cpus']}",
        f"--mem={resources['memory']}",
        f"--time={resources['time']}",
        f"--account={resources['account']}",
        f"--qos={resources['qos']}",
    ]
    if "gres" in resources:
        command.append(f"--gres={resources['gres']}")
    command.extend(
        [
            f"--job-name={label}",
            f"--chdir={project_root}",
            f"--output={log_path}",
            f"--comment=realnu_recovery_r2:{task}:{mode or 'all'}:{evaluation_sha[:8]}",
        ]
    )
    if dependency:
        command.append(f"--dependency={dependency}")
    exported = [
        "ALL",
        f"JER_REALNU_PROJECT_ROOT={project_root}",
        f"JER_REALNU_TRAINING_SOURCE_ROOT={training_source_root}",
        f"JER_REALNU_EVALUATION_SOURCE_ROOT={evaluation_source_root}",
        f"JER_REALNU_SOURCE_SHA={source_sha}",
        f"JER_REALNU_EVALUATION_SHA={evaluation_sha}",
        f"JER_REALNU_R2_MANIFEST={manifest_path}",
        f"JER_REALNU_R2_ROOT={r2_root}",
        f"JER_REALNU_RECOVERY_STAGE={stage}",
        f"JER_PYTHON={python_bin}",
    ]
    if mode:
        exported.append(f"JER_REALNU_RECOVERY_MODE={mode}")
    command.append("--export=" + ",".join(exported))
    command.append(str(evaluation_source_root / "scripts/evaluation/run_realnu_recovery_job.sh"))
    return command


def _submit(command: list[str], project_root: Path) -> str:
    output = subprocess.check_output(command, cwd=project_root, text=True).strip()
    job_id = output.split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned an invalid Slurm job ID: {output!r}")
    return job_id


def submit(
    project_root: Path,
    training_source_root: Path,
    evaluation_source_root: Path,
    manifest_path: Path,
    python_bin: Path,
    *,
    dry_run: bool,
) -> dict:
    manifest, r2_root = _validate_inputs(
        project_root, training_source_root, evaluation_source_root, manifest_path
    )
    source_sha = manifest["source_commit"]
    evaluation_sha = manifest["evaluation_recovery"]["source_sha"]
    attempt_root = Path(manifest["attempt_root"])
    log_root = attempt_root / "slurm"
    launcher = evaluation_source_root / "scripts/evaluation/run_realnu_recovery_job.sh"
    if not launcher.is_file():
        raise FileNotFoundError(f"R2 Slurm launcher is missing from snapshot: {launcher}")
    python_bin = python_bin.expanduser().resolve(strict=True)
    plan = []

    def add(task, stage, *, mode=None, dependency=None, resources=CPU):
        command = _slurm_command(
            project_root=project_root,
            evaluation_source_root=evaluation_source_root,
            training_source_root=training_source_root,
            manifest_path=manifest_path,
            r2_root=r2_root,
            source_sha=source_sha,
            evaluation_sha=evaluation_sha,
            python_bin=python_bin,
            log_path=log_root / f"realnu-r2-{task}-{mode or 'all'}-%j.out",
            task=task,
            stage=stage,
            resources=resources,
            mode=mode,
            dependency=dependency,
        )
        plan.append(
            {
                "task": task,
                "mode": mode,
                "dependency": dependency,
                "resources": resources,
                "command": command,
            }
        )
        return command

    validation_commands = {}
    for mode in VALIDATION_MODES:
        resources = GPU if mode in ("J0", "J_A", "J_AN") else CPU
        validation_commands[mode] = add(
            "validate",
            "validate-r2",
            mode=mode,
            resources=resources,
        )
    aggregate_dependency = "afterok:" + ":".join("<" + mode + ">" for mode in VALIDATION_MODES)
    aggregate_command = add(
        "aggregate",
        "aggregate-r2",
        dependency=aggregate_dependency,
    )
    core_command = add(
        "core-evaluation",
        "core-evaluation",
        dependency="afterok:<aggregate>",
    )
    posterior_command = add(
        "posterior64",
        "posterior64",
        dependency="afterok:<aggregate>",
        resources=GPU,
    )
    final_command = add(
        "final-evaluation",
        "final-evaluation",
        dependency="afterok:<core-evaluation>:<posterior64>",
    )
    if dry_run:
        return {"status": "ready_not_submitted", "r2_root": str(r2_root), "plan": plan}

    log_root.mkdir(parents=True, exist_ok=True)
    ids = {"validation_r2": {}, "aggregate_r2": None, "core_evaluation": None, "posterior64": None, "final_evaluation": None}
    manifest["recovery_status"] = "submitting"
    manifest["job_ids"].update(ids)
    atomic_json(manifest_path, manifest)
    try:
        for mode, command in validation_commands.items():
            ids["validation_r2"][mode] = _submit(command, project_root)
            manifest["job_ids"]["validation_r2"] = ids["validation_r2"].copy()
            atomic_json(manifest_path, manifest)

        aggregate_dependency = "afterok:" + ":".join(ids["validation_r2"].values())
        aggregate_command = _slurm_command(
            project_root=project_root,
            evaluation_source_root=evaluation_source_root,
            training_source_root=training_source_root,
            manifest_path=manifest_path,
            r2_root=r2_root,
            source_sha=source_sha,
            evaluation_sha=evaluation_sha,
            python_bin=python_bin,
            log_path=log_root / "realnu-r2-aggregate-%j.out",
            task="aggregate",
            stage="aggregate-r2",
            resources=CPU,
            dependency=aggregate_dependency,
        )
        plan[5]["dependency"] = aggregate_dependency
        plan[5]["command"] = aggregate_command
        ids["aggregate_r2"] = _submit(aggregate_command, project_root)
        manifest["job_ids"]["aggregate_r2"] = ids["aggregate_r2"]
        atomic_json(manifest_path, manifest)

        core_dependency = f"afterok:{ids['aggregate_r2']}"
        core_command = _slurm_command(
            project_root=project_root,
            evaluation_source_root=evaluation_source_root,
            training_source_root=training_source_root,
            manifest_path=manifest_path,
            r2_root=r2_root,
            source_sha=source_sha,
            evaluation_sha=evaluation_sha,
            python_bin=python_bin,
            log_path=log_root / "realnu-r2-core-%j.out",
            task="core-evaluation",
            stage="core-evaluation",
            resources=CPU,
            dependency=core_dependency,
        )
        plan[6]["dependency"] = core_dependency
        plan[6]["command"] = core_command
        posterior_command = _slurm_command(
            project_root=project_root,
            evaluation_source_root=evaluation_source_root,
            training_source_root=training_source_root,
            manifest_path=manifest_path,
            r2_root=r2_root,
            source_sha=source_sha,
            evaluation_sha=evaluation_sha,
            python_bin=python_bin,
            log_path=log_root / "realnu-r2-posterior64-%j.out",
            task="posterior64",
            stage="posterior64",
            resources=GPU,
            dependency=core_dependency,
        )
        plan[7]["dependency"] = core_dependency
        plan[7]["command"] = posterior_command
        ids["core_evaluation"] = _submit(core_command, project_root)
        manifest["job_ids"]["core_evaluation"] = ids["core_evaluation"]
        atomic_json(manifest_path, manifest)
        ids["posterior64"] = _submit(posterior_command, project_root)
        manifest["job_ids"]["posterior64"] = ids["posterior64"]
        atomic_json(manifest_path, manifest)

        final_dependency = f"afterok:{ids['core_evaluation']}:{ids['posterior64']}"
        final_command = _slurm_command(
            project_root=project_root,
            evaluation_source_root=evaluation_source_root,
            training_source_root=training_source_root,
            manifest_path=manifest_path,
            r2_root=r2_root,
            source_sha=source_sha,
            evaluation_sha=evaluation_sha,
            python_bin=python_bin,
            log_path=log_root / "realnu-r2-final-%j.out",
            task="final-evaluation",
            stage="final-evaluation",
            resources=CPU,
            dependency=final_dependency,
        )
        plan[8]["dependency"] = final_dependency
        plan[8]["command"] = final_command
        ids["final_evaluation"] = _submit(final_command, project_root)
        manifest["job_ids"].update(ids)
        manifest["recovery_status"] = "submitted"
        manifest["recovery_submitted_utc"] = datetime.now(UTC).isoformat()
        atomic_json(manifest_path, manifest)
    except Exception as error:
        manifest["job_ids"].update(ids)
        manifest["recovery_status"] = "partial_submission_failure"
        manifest["submission_error"] = f"{type(error).__name__}: {error}"
        manifest["recovery_updated_utc"] = datetime.now(UTC).isoformat()
        atomic_json(manifest_path, manifest)
        raise
    submitted_plan = {
        "status": "submitted",
        "training_source_sha": source_sha,
        "evaluation_recovery_sha": evaluation_sha,
        "r2_root": str(r2_root),
        "jobs": ids,
        "plan": plan,
        "submitted_utc": manifest["recovery_submitted_utc"],
    }
    atomic_json(r2_root / "submission_plan.json", submitted_plan)
    return submitted_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--training-source-root", type=Path, required=True)
    parser.add_argument("--evaluation-source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(os.environ.get("JER_PYTHON", sys.executable)))
    parser.add_argument("--submit", action="store_true", help="submit after all recovery checks pass")
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve(strict=True)
    training_source_root = args.training_source_root.expanduser().resolve(strict=True)
    evaluation_source_root = args.evaluation_source_root.expanduser().resolve(strict=True)
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    if project_root != Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries").resolve():
        raise RuntimeError("The project root differs from the frozen realnu production project")
    if (project_root / ".git").is_dir() and subprocess.check_output(
        ["git", "-C", str(project_root), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).strip():
        raise RuntimeError("The canonical project checkout must be clean before R2 submission")
    result = submit(
        project_root,
        training_source_root,
        evaluation_source_root,
        manifest_path,
        args.python,
        dry_run=not args.submit,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
