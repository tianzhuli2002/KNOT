"""Submit six realnu training jobs, dependent validations, and R_VALIDATION."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from run_realnu_production import (
    ATTEMPT_ROOT,
    EXPECTED_G0_SHA,
    INITIALISATION_MANIFEST_PATH,
    INITIALISATION_ROOT,
    MODE_CONFIGS,
    PARENT_MODES,
    PRODUCTION_ROOT,
    ROOT,
    SOURCE_BRANCH,
    SOURCE_ROOT,
    STANDARD_MODES,
    assert_source,
    load_and_validate_config,
    sha256,
    verify_g0,
    verify_inputs,
)

ALLOWED_POST_G0_FILES: set[str] = set()
ACCOUNT = "gpu-free"
QOS = "gpu-free"


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def post_g0_source_diff(expected_sha: str) -> list[str]:
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", EXPECTED_G0_SHA, expected_sha],
        cwd=ROOT,
        text=True,
    ).splitlines()
    unexpected = sorted(set(changed) - ALLOWED_POST_G0_FILES)
    if unexpected:
        raise RuntimeError(
            "Production source contains changes outside the production orchestration scripts: "
            f"{unexpected}"
        )
    if set(changed) != ALLOWED_POST_G0_FILES:
        raise RuntimeError(
            "Production must use the exact source accepted by G0; "
            f"found {sorted(set(changed))}"
        )
    return sorted(changed)


def site_resources() -> dict:
    from omegaconf import OmegaConf

    site = OmegaConf.load(ROOT / "configs/site/csf3.yaml")
    return {
        "gpu_partition": str(site.slurm.gpu_partition),
        "gpu_gres": str(site.slurm.gpu_gres),
        "gpu_cpus": int(site.slurm.gpu_cpus),
        "gpu_memory": str(site.slurm.gpu_memory),
        "gpu_time": str(site.slurm.gpu_time),
        "cpu_partition": str(site.slurm.cpu_partition),
        "cpu_cpus": int(site.slurm.cpu_cpus),
        "cpu_memory": str(site.slurm.cpu_memory),
        "cpu_time": str(site.slurm.cpu_time),
    }


def initialisation_contract() -> dict:
    if not INITIALISATION_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Realnu initialisation manifest is missing: {INITIALISATION_MANIFEST_PATH}"
        )
    manifest_sha = sha256(INITIALISATION_MANIFEST_PATH)
    initialisation_manifest = json.loads(INITIALISATION_MANIFEST_PATH.read_text())
    if set(initialisation_manifest.get("modes", {})) != set(MODE_CONFIGS):
        raise RuntimeError("Realnu initialisation manifest does not contain exactly the six release modes")
    if (
        initialisation_manifest.get("q0_q1_shared_initialisation_equal") is not True
        or initialisation_manifest.get("q0_q1_shared_components")
        != ["visible", "assignment", "invisible"]
    ):
        raise RuntimeError("Realnu production initialisation equivalence contract failed")

    modes = {}
    for mode, record in initialisation_manifest["modes"].items():
        checkpoint = Path(record["checkpoint"]).expanduser().resolve()
        expected_checkpoint = (INITIALISATION_ROOT / f"{mode}_initial.ckpt").resolve()
        if checkpoint != expected_checkpoint or not checkpoint.is_file():
            raise FileNotFoundError(
                f"{mode} initial checkpoint path differs from the frozen release path: {checkpoint}"
            )
        components = sorted(record.get("component_sha256", {}))
        if not components:
            raise RuntimeError(f"{mode} initial checkpoint has no complete model components")
        modes[mode] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "components": components,
            "component_sha256": record["component_sha256"],
            "initialisation": record["initialisation"],
            "seed": int(record["initialisation"]["seed"]),
        }
    for component in ("visible", "assignment", "invisible"):
        if (
            modes["Q0"]["component_sha256"].get(component)
            != modes["Q1"]["component_sha256"].get(component)
        ):
            raise RuntimeError(f"Q0/Q1 initial {component} checkpoint states differ")
    for mode in ("J0", "J_A", "J_AN"):
        if (
            modes[mode]["component_sha256"].get("invisible")
            != modes["Q1"]["component_sha256"].get("invisible")
        ):
            raise RuntimeError(
                f"{mode} invisible initialization differs from the shared Q1 predictor"
            )
    parent_hashes = {
        mode: modes[mode]["component_sha256"].get("partonic_cfm")
        for mode in PARENT_MODES
    }
    if None in parent_hashes.values() or len(set(parent_hashes.values())) != 1:
        raise RuntimeError(
            "Parent-flow modes must share the approved parent-CFM initial weights: "
            f"{parent_hashes}"
        )
    return {
        "manifest_path": str(INITIALISATION_MANIFEST_PATH.resolve()),
        "manifest_sha256": manifest_sha,
        "seed": int(initialisation_manifest["seed"]),
        "modes": modes,
    }


def slurm_command(
    *,
    stage: str,
    source_sha: str,
    run_manifest: Path,
    python_bin: Path,
    output_log: Path,
    resources: dict,
    mode: str | None = None,
    dependency: str | None = None,
) -> list[str]:
    command = ["sbatch", "--parsable"]
    if stage in {"train", "validate"}:
        command.extend(
            [
                f"--partition={resources['gpu_partition']}",
                f"--gres={resources['gpu_gres']}",
                f"--cpus-per-task={resources['gpu_cpus']}",
                f"--mem={resources['gpu_memory']}",
                f"--time={resources['gpu_time']}",
                f"--account={ACCOUNT}",
                f"--qos={QOS}",
            ]
        )
    else:
        command.extend(
            [
                f"--partition={resources['cpu_partition']}",
                f"--cpus-per-task={resources['cpu_cpus']}",
                f"--mem={resources['cpu_memory']}",
                f"--time={resources['cpu_time']}",
            ]
        )
    label = f"rn-{mode.lower()}-{stage}" if mode else "rn-r-validation"
    command.extend(
        [
            f"--job-name={label}",
            f"--chdir={ROOT}",
            f"--output={output_log}",
            f"--comment=realnu_production:{stage}:{mode or 'R_VALIDATION'}:{source_sha[:8]}",
        ]
    )
    if dependency:
        command.append(f"--dependency={dependency}")
    exported = [
        "ALL",
        f"JER_REALNU_PROJECT_ROOT={ROOT}",
        f"JER_REALNU_SOURCE_ROOT={SOURCE_ROOT}",
        f"JER_REALNU_EXPECTED_SHA={source_sha}",
        f"JER_PROJECT_ROOT={ROOT}",
        f"JER_SOURCE_ROOT={SOURCE_ROOT}",
        f"JER_PYTHON={python_bin}",
        f"JER_REALNU_PRODUCTION_STAGE={stage}",
        f"JER_REALNU_PRODUCTION_SOURCE_SHA={source_sha}",
        f"JER_REALNU_PRODUCTION_RUN_MANIFEST={run_manifest}",
    ]
    if mode:
        exported.append(f"JER_REALNU_PRODUCTION_MODE={mode}")
    command.append("--export=" + ",".join(exported))
    command.append(str(SOURCE_ROOT / "scripts/production/run_realnu_production.sh"))
    return command


def submit(command: list[str]) -> str:
    output = subprocess.check_output(command, cwd=ROOT, text=True).strip()
    job_id = output.split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"sbatch returned an invalid job ID: {output!r}")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--g0-acceptance", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(os.environ.get("JER_PYTHON", sys.executable)))
    parser.add_argument("--submit", action="store_true", help="submit after all release checks pass")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve(strict=True)
    source_root = args.source_root.expanduser().resolve(strict=True)
    if project_root != ROOT or source_root != SOURCE_ROOT:
        raise RuntimeError("Submission roots differ from the explicit production environment")
    if source_root != project_root / "runtime/source" / EXPECTED_G0_SHA:
        raise RuntimeError("Production source must be the immutable accepted G0 snapshot")
    source_sha = EXPECTED_G0_SHA
    assert_source(source_sha)
    changed_files = post_g0_source_diff(source_sha)
    acceptance_path = args.g0_acceptance.expanduser().resolve()
    python_bin = args.python.expanduser().resolve()
    if not python_bin.is_file():
        raise FileNotFoundError(f"Production Python interpreter is missing: {python_bin}")

    acceptance_shell = {
        "campaign": "realnu_top_antitop_v1",
        "g0_acceptance_path": str(acceptance_path),
        "g0_acceptance_sha256": sha256(acceptance_path),
        "source_commit": source_sha,
        "source_branch": SOURCE_BRANCH,
        "source_root": str(source_root),
        "g0_source_commit": EXPECTED_G0_SHA,
        "changed_files_after_g0": changed_files,
    }
    inputs = verify_inputs()
    initialisations = initialisation_contract()
    acceptance = verify_g0(
        acceptance_path,
        source_sha,
        {
            "inputs": inputs,
            "g0_acceptance_path": str(acceptance_path),
            "g0_acceptance_sha256": acceptance_shell["g0_acceptance_sha256"],
        },
    )
    if acceptance.get("source_commit") != EXPECTED_G0_SHA:
        raise RuntimeError("The required realnu G0 artifact is not from the frozen release SHA")
    modes = {}
    for mode, config_path in MODE_CONFIGS.items():
        config, _, training_output = load_and_validate_config(mode, config_path)
        initial = initialisations["modes"][mode]
        validation_output = training_output / "validation"
        if training_output.exists():
            raise FileExistsError(f"Refusing to reuse {mode} production output: {training_output}")
        if validation_output.exists():
            raise FileExistsError(f"Refusing to reuse {mode} validation output: {validation_output}")
        modes[mode] = {
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256(config_path),
            "experiment_name": str(config.experiment.name),
            "training_output": str(training_output),
            "validation_output": str(validation_output),
            "trainable_components": sorted(str(value) for value in config.training.trainable),
            "standard_validation": mode in STANDARD_MODES,
            "parent_validation": mode in PARENT_MODES,
            "conditioning": (
                str(config.model.parent_cfm.conditioning)
                if bool(config.model.parent_cfm.enabled)
                else None
            ),
            "initialisation": initial,
        }

    training_outputs = {entry["training_output"] for entry in modes.values()}
    if len(training_outputs) != len(MODE_CONFIGS):
        raise RuntimeError("Production modes must use distinct full-training output paths")

    if ATTEMPT_ROOT.exists():
        raise FileExistsError(f"Refusing to reuse a realnu production attempt: {ATTEMPT_ROOT}")
    if (PRODUCTION_ROOT / "R_VALIDATION.json").exists() or (PRODUCTION_ROOT / "R_VALIDATION.md").exists():
        raise FileExistsError(f"Refusing to replace existing R_VALIDATION under {PRODUCTION_ROOT}")
    run_manifest_path = ATTEMPT_ROOT / "run_manifest.json"
    manifest = {
        **acceptance_shell,
        "status": "preparing" if args.submit else "validated_dry_run",
        "attempt_root": str(ATTEMPT_ROOT.resolve()),
        "run_manifest_path": str(run_manifest_path.resolve()),
        "submitted_utc": None,
        "resources": site_resources(),
        "python": str(python_bin),
        "inputs": inputs,
        "initialisation_manifest": initialisations["manifest_path"],
        "initialisation_manifest_sha256": initialisations["manifest_sha256"],
        "initialisation_seed": initialisations["seed"],
        "modes": modes,
        "job_ids": {"training": {}, "validation": {}, "r_validation": None},
        "production_jobs_submitted": 0,
        "test_rows_read": 0,
    }
    resources = manifest["resources"]
    command_plan = []
    for mode in MODE_CONFIGS:
        command_plan.append(
            {
                "stage": "train",
                "mode": mode,
                "command": slurm_command(
                    stage="train",
                    source_sha=source_sha,
                    run_manifest=run_manifest_path,
                    python_bin=python_bin,
                    output_log=ATTEMPT_ROOT / f"slurm/train-{mode}-%j.out",
                    mode=mode,
                    resources=resources,
                ),
            }
        )
    command_plan.append(
        {
            "stage": "validate",
            "mode": "each after its own training job",
            "dependency": "afterok:<training-job-id>",
        }
    )
    command_plan.append(
        {
            "stage": "aggregate",
            "mode": "R_VALIDATION",
            "dependency": "afterany:<all-six-validation-job-ids>",
        }
    )
    if not args.submit:
        print(json.dumps({"status": "ready_not_submitted", "manifest": manifest, "plan": command_plan}, indent=2, sort_keys=True))
        return

    ATTEMPT_ROOT.mkdir(parents=True)
    (ATTEMPT_ROOT / "slurm").mkdir()
    manifest["status"] = "submitting"
    atomic_json(run_manifest_path, manifest)
    try:
        for mode in MODE_CONFIGS:
            command = slurm_command(
                stage="train",
                source_sha=source_sha,
                run_manifest=run_manifest_path,
                python_bin=python_bin,
                output_log=ATTEMPT_ROOT / f"slurm/train-{mode}-%j.out",
                mode=mode,
                resources=resources,
            )
            manifest["job_ids"]["training"][mode] = submit(command)
            manifest["production_jobs_submitted"] += 1
            atomic_json(run_manifest_path, manifest)
        for mode in MODE_CONFIGS:
            dependency = f"afterok:{manifest['job_ids']['training'][mode]}"
            command = slurm_command(
                stage="validate",
                source_sha=source_sha,
                run_manifest=run_manifest_path,
                python_bin=python_bin,
                output_log=ATTEMPT_ROOT / f"slurm/validate-{mode}-%j.out",
                mode=mode,
                dependency=dependency,
                resources=resources,
            )
            manifest["job_ids"]["validation"][mode] = submit(command)
            manifest["production_jobs_submitted"] += 1
            atomic_json(run_manifest_path, manifest)
        validation_ids = ":".join(manifest["job_ids"]["validation"].values())
        aggregate = slurm_command(
            stage="aggregate",
            source_sha=source_sha,
            run_manifest=run_manifest_path,
            python_bin=python_bin,
            output_log=ATTEMPT_ROOT / "slurm/R-VALIDATION-%j.out",
            dependency=f"afterany:{validation_ids}",
            resources=resources,
        )
        manifest["job_ids"]["r_validation"] = submit(aggregate)
        manifest["status"] = "submitted"
        manifest["submitted_utc"] = datetime.now(UTC).isoformat()
        manifest["production_jobs_submitted"] += 1
        atomic_json(run_manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "partial_submission_failure"
        manifest["submission_error"] = f"{type(error).__name__}: {error}"
        manifest["updated_utc"] = datetime.now(UTC).isoformat()
        atomic_json(run_manifest_path, manifest)
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
