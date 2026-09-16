#!/usr/bin/env python3
"""Submit one JER study as dependency-ordered Slurm arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.studies import StudySpec, VariantSpec, resolve_study


_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _selected(value: str, allowed: tuple, label: str) -> tuple:
    if value == "all":
        return allowed
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; allowed={list(allowed)}")
    return selected


def _levels(study: StudySpec, variants: tuple[VariantSpec, ...]) -> list[tuple[VariantSpec, ...]]:
    remaining = {variant.name: variant for variant in variants}
    complete: set[str] = set()
    levels: list[tuple[VariantSpec, ...]] = []
    while remaining:
        ready = tuple(
            variant
            for variant in study.variants
            if variant.name in remaining
            and (
                variant.initialise_from is None
                or variant.initialise_from in complete
                or variant.initialise_from not in remaining
                and variant.initialise_from not in {item.name for item in variants}
            )
        )
        if not ready:
            raise RuntimeError("Selected study dependency graph has no schedulable variant")
        levels.append(ready)
        for variant in ready:
            complete.add(variant.name)
            del remaining[variant.name]
    return levels


def _submit(command: list[str], *, dry_run: bool, fake_id: int) -> str:
    print(shlex.join(command), flush=True)
    if dry_run:
        return str(900000 + fake_id)
    result = subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=True
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job id from {result.stdout!r}")
    print(f"submitted job {job_id}")
    return job_id


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )
    return commit, dirty


def _save_submission(path: Path, submitted: dict) -> None:
    path.write_text(json.dumps(submitted, indent=2, sort_keys=True) + "\n")


def _write_task_file(path: Path, tasks: list[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{topology}\t{variant}\t{seed}\n" for topology, variant, seed in tasks)
    )


def _write_postprocess_task_file(
    path: Path, tasks: list[tuple[str, str, int, str]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{topology}\t{variant}\t{seed}\t{prediction}\n"
            for topology, variant, seed, prediction in tasks
        )
    )


def _postprocess_task_lists(
    study: StudySpec,
    topologies: tuple[str, ...],
    variants: tuple[VariantSpec, ...],
    seeds: tuple[int, ...],
    output_root: Path,
) -> tuple[
    list[tuple[str, str, int, str]],
    list[tuple[str, str, int, str]],
    list[tuple[str, str, int, str]],
    list[tuple[str, str, int, str]],
]:
    """Split resumable work into missing evaluation and plot tasks.

    Evaluation-complete predictions with missing plots are deliberately kept out of
    the evaluation array.  This prevents a retry from touching valid evaluation
    JSONs while still allowing the plot product to be repaired independently.
    """
    from scripts.run_study import (
        _evaluation_complete,
        _evaluation_path,
        _plot_complete,
        _prediction_path,
        _validate_prediction_artifact,
    )

    evaluation_tasks: list[tuple[str, str, int, str]] = []
    plot_tasks: list[tuple[str, str, int, str]] = []
    skipped_evaluation: list[tuple[str, str, int, str]] = []
    skipped_plot: list[tuple[str, str, int, str]] = []
    for topology in topologies:
        for variant in variants:
            if variant.reporting == "setup":
                continue
            for seed in seeds:
                for prediction in variant.predictions:
                    task = (topology, variant.name, seed, prediction.name)
                    source = _prediction_path(
                        output_root, study, topology, variant.name, seed, prediction.name
                    )
                    if not source.is_file():
                        raise FileNotFoundError(
                            f"Postprocessing requires prediction artifact {source}"
                        )
                    _validate_prediction_artifact(source)
                    evaluation = _evaluation_path(
                        output_root, study, topology, variant.name, seed, prediction.name
                    )
                    plot = (
                        output_root / study.name / topology / variant.name
                        / f"seed_{seed}" / "plots" / prediction.name
                    )
                    evaluation_ok = _evaluation_complete(evaluation)
                    plot_ok = _plot_complete(plot)
                    if evaluation_ok:
                        skipped_evaluation.append(task)
                    else:
                        evaluation_tasks.append(task)
                    if plot_ok:
                        skipped_plot.append(task)
                    elif evaluation_ok:
                        plot_tasks.append(task)
    return evaluation_tasks, plot_tasks, skipped_evaluation, skipped_plot


def _prediction_task_complete(
    study: StudySpec,
    topology: str,
    variant: VariantSpec,
    seed: int,
    *,
    output_root: Path,
    overrides: tuple[str, ...],
    git_commit: str,
) -> bool:
    """Check every prediction identity for one Slurm task before resubmission."""
    from scripts.run_study import (
        _prediction_checkpoint,
        _prediction_identity,
        _prediction_path,
        _public_run_config,
        _validate_prediction_artifact,
        _verify_artifact_record,
    )

    for prediction in variant.predictions:
        target = _prediction_path(
            output_root, study, topology, variant.name, seed, prediction.name
        )
        record = target.with_suffix(".identity.json")
        checkpoint = _prediction_checkpoint(study, variant, topology, seed, output_root)
        source_record = checkpoint.parent.parent / "artifact_identity.json"
        if not target.is_file() or not record.is_file() or not source_record.is_file():
            return False
        try:
            _validate_prediction_artifact(target)
            source_payload = json.loads(source_record.read_text())
            source_identity = source_payload["identity"]
            checkpoint_training_sha = source_identity["git_commit"]
            _, _, resolved = _public_run_config(
                study,
                topology,
                variant,
                seed,
                output_root,
                overrides,
                prediction,
            )
            expected = _prediction_identity(
                study,
                topology,
                variant,
                prediction,
                seed,
                resolved,
                checkpoint,
                checkpoint_training_sha,
                prediction.recurrent_steps,
                git_commit,
            )
            _verify_artifact_record(record, target, expected, "prediction")
        except (KeyError, OSError, TypeError, ValueError, RuntimeError):
            return False
    return True


def _worker_script(
    path: Path,
    *,
    task_file: Path,
    study_path: Path,
    stage: str,
    output_root: Path,
    device: str,
    global_overrides: tuple[str, ...],
    expected_commit: str,
    expected_manifest_sha256: str,
    expected_experiments: dict[str, dict[str, str]],
    expected_gate_digest: str | None = None,
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_study.py"),
        str(study_path),
        stage,
        "--output-root", str(output_root),
        "--topologies", '"$topology"',
        "--variants", '"$variant"',
        "--seeds", '"$seed"',
        "--device", device,
    ]
    if expected_gate_digest is not None:
        command += ["--expected-gates-digest", expected_gate_digest]
    if stage == "train":
        command.append("--exact-variants")
    for override in global_overrides:
        command += ["--set", override]
    rendered = " ".join(
        token if token.startswith('"$') else shlex.quote(token) for token in command
    )
    experiment_checks = "".join(
        "if [[ \"$topology\" == "
        + shlex.quote(topology)
        + " ]]; then expected_experiment_path="
        + shlex.quote(identity["path"])
        + "; expected_experiment_sha256="
        + shlex.quote(identity["sha256"])
        + "; fi\n"
        for topology, identity in sorted(expected_experiments.items())
    )
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"repo_root={shlex.quote(str(ROOT))}\n"
        f"export PYTHONPATH={shlex.quote(str(ROOT / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        f"expected_git_commit={shlex.quote(expected_commit)}\n"
        'actual_git_commit=$(git -C "$repo_root" rev-parse HEAD)\n'
        'if [[ "$actual_git_commit" != "$expected_git_commit" ]]; then\n'
        '    echo "Checkout changed after Slurm submission: expected $expected_git_commit, found $actual_git_commit" >&2\n'
        "    exit 3\n"
        "fi\n"
        'if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then\n'
        '    echo "Checkout became dirty after Slurm submission" >&2\n'
        "    exit 3\n"
        "fi\n"
        f"manifest_snapshot={shlex.quote(str(study_path))}\n"
        f"expected_manifest_sha256={shlex.quote(expected_manifest_sha256)}\n"
        'actual_manifest_sha256=$(sha256sum "$manifest_snapshot" | cut -d" " -f1)\n'
        'if [[ "$actual_manifest_sha256" != "$expected_manifest_sha256" ]]; then\n'
        '    echo "Study manifest snapshot changed after submission" >&2\n'
        "    exit 3\n"
        "fi\n"
        f"task_file={shlex.quote(str(task_file))}\n"
        'line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$task_file")\n'
        "if [[ -z \"$line\" ]]; then echo \"Missing study task for index $SLURM_ARRAY_TASK_ID\" >&2; exit 2; fi\n"
        "IFS=$'\\t' read -r topology variant seed <<< \"$line\"\n"
        'expected_experiment_path=""\n'
        'expected_experiment_sha256=""\n'
        + experiment_checks
        + 'if [[ -z "$expected_experiment_path" ]]; then\n'
        '    echo "Unknown topology in submitted task" >&2\n'
        "    exit 2\n"
        "fi\n"
        'actual_experiment_sha256=$(sha256sum "$expected_experiment_path" | cut -d" " -f1)\n'
        'if [[ "$actual_experiment_sha256" != "$expected_experiment_sha256" ]]; then\n'
        '    echo "Base experiment changed after Slurm submission" >&2\n'
        "    exit 3\n"
        "fi\n"
        f"exec {rendered}\n"
    )
    path.chmod(0o755)


def _postprocess_worker_script(
    path: Path,
    *,
    task_file: Path,
    study_path: Path,
    stages: tuple[str, ...],
    output_root: Path,
    expected_commit: str,
    expected_manifest_sha256: str,
    expected_gate_digest: str | None,
    expected_experiments: dict[str, dict[str, str]],
    overrides: tuple[str, ...],
) -> None:
    """Write one prediction-level CPU worker with strict source checks."""
    commands = []
    for stage in stages:
        command = [
            sys.executable,
            str(ROOT / "scripts/run_study.py"),
            str(study_path),
            stage,
            "--output-root", str(output_root),
            "--topologies", '"$topology"',
            "--variants", '"$variant"',
            "--seeds", '"$seed"',
            "--device", "cpu",
        ]
        if stage in {"evaluate", "plot"}:
            command += ["--predictions", '"$prediction"']
        if expected_gate_digest is not None:
            command += ["--expected-gates-digest", expected_gate_digest]
        for override in overrides:
            command += ["--set", override]
        commands.append(
            " ".join(
                token if token.startswith('"$') else shlex.quote(token)
                for token in command
            )
        )
    experiment_checks = "".join(
        "if [[ \"$topology\" == "
        + shlex.quote(topology)
        + " ]]; then expected_experiment_path="
        + shlex.quote(identity["path"])
        + "; expected_experiment_sha256="
        + shlex.quote(identity["sha256"])
        + "; fi\n"
        for topology, identity in sorted(expected_experiments.items())
    )
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"repo_root={shlex.quote(str(ROOT))}\n"
        f"export PYTHONPATH={shlex.quote(str(ROOT / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        f"expected_git_commit={shlex.quote(expected_commit)}\n"
        'actual_git_commit=$(git -C "$repo_root" rev-parse HEAD)\n'
        'if [[ "$actual_git_commit" != "$expected_git_commit" ]]; then\n'
        '    echo "Checkout changed after Slurm submission: expected $expected_git_commit, found $actual_git_commit" >&2\n'
        "    exit 3\n"
        "fi\n"
        'if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then\n'
        '    echo "Checkout became dirty after Slurm submission" >&2\n'
        "    exit 3\n"
        "fi\n"
        f"manifest_snapshot={shlex.quote(str(study_path))}\n"
        f"expected_manifest_sha256={shlex.quote(expected_manifest_sha256)}\n"
        'actual_manifest_sha256=$(sha256sum "$manifest_snapshot" | cut -d" " -f1)\n'
        'if [[ "$actual_manifest_sha256" != "$expected_manifest_sha256" ]]; then\n'
        '    echo "Study manifest snapshot changed after submission" >&2\n'
        "    exit 3\n"
        "fi\n"
        f"task_file={shlex.quote(str(task_file))}\n"
        'line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$task_file")\n'
        'if [[ -z "$line" ]]; then echo "Missing study task for index $SLURM_ARRAY_TASK_ID" >&2; exit 2; fi\n'
        "IFS=$'\\t' read -r topology variant seed prediction <<< \"$line\"\n"
        'expected_experiment_path=""\n'
        'expected_experiment_sha256=""\n'
        + experiment_checks
        + 'if [[ -z "$expected_experiment_path" ]]; then\n'
        '    echo "Unknown topology in submitted task" >&2\n'
        "    exit 2\n"
        "fi\n"
        'actual_experiment_sha256=$(sha256sum "$expected_experiment_path" | cut -d" " -f1)\n'
        'if [[ "$actual_experiment_sha256" != "$expected_experiment_sha256" ]]; then\n'
        '    echo "Base experiment changed after submission" >&2\n'
        "    exit 3\n"
        "fi\n"
        + " && ".join(commands)
        + "\n"
    )
    path.chmod(0o755)


def _post_script(
    path: Path,
    commands: list[list[str]],
    expected_commit: str,
    manifest_snapshot: Path,
    expected_manifest_sha256: str,
    expected_experiments: dict[str, dict[str, str]],
) -> None:
    experiment_checks = "".join(
        "experiment_path="
        + shlex.quote(identity["path"])
        + "; expected_experiment_sha256="
        + shlex.quote(identity["sha256"])
        + "; actual_experiment_sha256=$(sha256sum \"$experiment_path\" | cut -d\" \" -f1); "
        + "if [[ \"$actual_experiment_sha256\" != \"$expected_experiment_sha256\" ]]; then "
        + "echo \"Base experiment changed after Slurm submission: "
        + topology
        + "\" >&2; exit 3; fi\n"
        for topology, identity in sorted(expected_experiments.items())
    )
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"repo_root={shlex.quote(str(ROOT))}\n"
        f"export PYTHONPATH={shlex.quote(str(ROOT / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        f"expected_git_commit={shlex.quote(expected_commit)}\n"
        'actual_git_commit=$(git -C "$repo_root" rev-parse HEAD)\n'
        'if [[ "$actual_git_commit" != "$expected_git_commit" ]]; then\n'
        '    echo "Checkout changed after Slurm submission: expected $expected_git_commit, found $actual_git_commit" >&2\n'
        "    exit 3\n"
        "fi\n"
        'if [[ -n "$(git -C "$repo_root" status --porcelain)" ]]; then\n'
        '    echo "Checkout became dirty after Slurm submission" >&2\n'
        "    exit 3\n"
        "fi\n"
        f"manifest_snapshot={shlex.quote(str(manifest_snapshot))}\n"
        f"expected_manifest_sha256={shlex.quote(expected_manifest_sha256)}\n"
        'actual_manifest_sha256=$(sha256sum "$manifest_snapshot" | cut -d" " -f1)\n'
        'if [[ "$actual_manifest_sha256" != "$expected_manifest_sha256" ]]; then\n'
        '    echo "Study manifest snapshot changed after submission" >&2\n'
        "    exit 3\n"
        "fi\n"
        + experiment_checks
        + " && ".join(shlex.join(command) for command in commands)
        + "\n"
    )
    path.chmod(0o755)


def _sbatch_array(
    *,
    worker: Path,
    task_count: int,
    job_name: str,
    log_dir: Path,
    partition: str,
    gres: str | None,
    cpus: int,
    memory: str,
    time_limit: str,
    dependency: str | None,
    extra: tuple[str, ...],
) -> list[str]:
    if task_count <= 0:
        raise ValueError("Cannot submit an empty Slurm array")
    command = [
        "sbatch", "--parsable",
        f"--job-name={job_name}",
        f"--array=0-{task_count - 1}",
        f"--partition={partition}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time={time_limit}",
        f"--output={log_dir}/%x-%A_%a.out",
        f"--error={log_dir}/%x-%A_%a.err",
    ]
    if gres:
        command.append(f"--gres={gres}")
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.extend(extra)
    command.append(str(worker))
    return command


def _post_command(
    study_path: Path,
    stage: str,
    *,
    output_root: Path,
    topologies: tuple[str, ...],
    variants: tuple[VariantSpec, ...],
    seeds: tuple[int, ...],
    overrides: tuple[str, ...],
    expected_gate_digest: str | None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_study.py"),
        str(study_path),
        stage,
        "--output-root", str(output_root),
        "--topologies", ",".join(topologies),
        "--variants", ",".join(variant.name for variant in variants),
        "--seeds", ",".join(map(str, seeds)),
        "--device", "cpu",
    ]
    for override in overrides:
        command += ["--set", override]
    if expected_gate_digest is not None:
        command += ["--expected-gates-digest", expected_gate_digest]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a JER study to Slurm")
    parser.add_argument("study", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/studies"))
    parser.add_argument("--topologies", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--gpu-partition", default="gpuL")
    parser.add_argument("--gpu-gres", default="gpu:l40s:1")
    parser.add_argument("--gpu-cpus", type=int, default=12)
    parser.add_argument("--gpu-mem", default="64G")
    parser.add_argument("--gpu-time", default="4-00:00:00")
    parser.add_argument("--cpu-partition", default="multicore")
    parser.add_argument("--cpu-cpus", type=int, default=4)
    parser.add_argument("--cpu-mem", default="16G")
    parser.add_argument("--cpu-time", default="04:00:00")
    parser.add_argument("--set", dest="global_overrides", action="append", default=[])
    parser.add_argument(
        "--extra-gpu-sbatch",
        action="append",
        default=[],
        help="Pass an additional sbatch argument to GPU training/prediction jobs.",
    )
    parser.add_argument(
        "--extra-cpu-sbatch",
        action="append",
        default=[],
        help="Pass an additional sbatch argument to CPU postprocessing jobs.",
    )
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument(
        "--prediction-only",
        action="store_true",
        help="Skip all training layers and submit only missing prediction tasks.",
    )
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Submit only missing prediction-level evaluation/plot tasks and aggregation.",
    )
    parser.add_argument(
        "--dependency-afterok",
        help="External job ID that must succeed before the first training layer starts.",
    )
    parser.add_argument(
        "--campaign-id",
        help="Human-readable immutable Slurm-attempt identifier; defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--submission-identity",
        help="Campaign-level immutable identity used for idempotent resubmission checks.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.prediction_only and args.postprocess_only:
        raise ValueError("--prediction-only and --postprocess-only cannot be combined")

    study = resolve_study(args.study)
    topologies = _selected(args.topologies, study.topologies, "topologies")
    variant_names = tuple(variant.name for variant in study.variants)
    requested_names = _selected(args.variants, variant_names, "variants")
    requested = tuple(study.variant_map[name] for name in requested_names)
    if args.prediction_only or args.postprocess_only:
        training = ()
    elif not study.training:
        training = ()
    elif args.variants == "all":
        training = study.training_order()
    else:
        training = study.training_order(requested_names)
    seeds = (
        study.seeds
        if args.seeds == "all"
        else tuple(int(value) for value in args.seeds.split(",") if value)
    )
    unknown_seeds = sorted(set(seeds) - set(study.seeds))
    if unknown_seeds:
        raise ValueError(f"Seeds {unknown_seeds} are not declared by the study")

    from scripts.run_study import _gate_checkpoint

    for variant in training:
        if variant.initialise_from_gate is None:
            continue
        for topology in topologies:
            for seed in seeds:
                _gate_checkpoint(study, variant.initialise_from_gate, topology, seed)

    study_path = study.path
    output_root = args.output_root.expanduser().resolve()
    manifest_sha256 = _sha256_file(study_path)
    submitted_at = datetime.now(timezone.utc).isoformat()
    campaign_id = args.campaign_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + f"-{manifest_sha256[:8]}"
    )
    if not _CAMPAIGN_ID.fullmatch(campaign_id):
        raise ValueError(
            "--campaign-id must contain only letters, numbers, '.', '_' and '-'"
        )
    slurm_base = output_root / study.name / "_slurm"
    slurm_root = slurm_base / campaign_id
    if slurm_root.exists():
        raise FileExistsError(
            f"Slurm campaign attempt already exists and is immutable: {slurm_root}"
        )
    log_dir = slurm_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest_snapshot = slurm_root / "study_manifest.yaml"
    shutil.copyfile(study_path, manifest_snapshot)
    if _sha256_file(manifest_snapshot) != manifest_sha256:
        raise RuntimeError(
            "Study manifest snapshot checksum does not match the submitted manifest"
        )
    overrides = tuple(args.global_overrides)
    gpu_extra = tuple(args.extra_gpu_sbatch)
    cpu_extra = tuple(args.extra_cpu_sbatch)
    git_commit, git_dirty = _git_state()
    if git_dirty and not args.dry_run:
        raise RuntimeError("Study submission requires a clean committed checkout")
    base_experiments = {
        topology: {
            "path": str(study.experiment_path(ROOT, topology)),
            "sha256": _sha256_file(study.experiment_path(ROOT, topology)),
        }
        for topology in topologies
    }

    submitted: dict[str, object] = {
        "campaign_id": campaign_id,
        "submission_identity": args.submission_identity,
        "submission_timestamp": submitted_at,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "study": study.name,
        "study_manifest": str(study_path),
        "study_manifest_sha256": manifest_sha256,
        "study_manifest_snapshot": str(manifest_snapshot),
        "study_manifest_snapshot_sha256": _sha256_file(manifest_snapshot),
        "selection_split": study.split,
        "output_root": str(output_root),
        "slurm_root": str(slurm_root),
        "log_directory": str(log_dir),
        "topologies": list(topologies),
        "base_experiments": {
            topology: identity for topology, identity in base_experiments.items()
        },
        "requested_variants": [variant.name for variant in requested],
        "training_closure": [variant.name for variant in training],
        "seeds": list(seeds),
        "overrides": list(overrides),
        "extra_gpu_sbatch": list(gpu_extra),
        "extra_cpu_sbatch": list(cpu_extra),
        "required_gates": list(study.requires),
        "resolved_gate_variants": {
            name: gate.get("selected_variant")
            for name, gate in (study.resolved_gates or {}).items()
        },
        "resolved_gate_digest": study.gates_digest,
        "checkpoint_sources": {
            variant.name: {
                "study": variant.checkpoint_from[0],
                "variant": variant.checkpoint_from[1],
            }
            for variant in requested
            if variant.checkpoint_from is not None
        },
        "resources": {
            "gpu": {
                "partition": args.gpu_partition,
                "gres": args.gpu_gres,
                "cpus": args.gpu_cpus,
                "memory": args.gpu_mem,
                "time": args.gpu_time,
            },
            "cpu": {
                "partition": args.cpu_partition,
                "cpus": args.cpu_cpus,
                "memory": args.cpu_mem,
                "time": args.cpu_time,
            },
        },
        "train_levels": [],
        "submission_mode": (
            "postprocess_only" if args.postprocess_only
            else "prediction_only" if args.prediction_only
            else "normal"
        ),
    }
    submission_path = slurm_root / "submission.json"
    _save_submission(submission_path, submitted)

    if args.postprocess_only:
        if args.no_postprocess:
            raise ValueError("--postprocess-only cannot be combined with --no-postprocess")
        prediction_variants = tuple(
            variant for variant in requested if variant.reporting != "setup"
        )
        evaluation_tasks, plot_tasks, skipped_evaluation, skipped_plot = (
            _postprocess_task_lists(
                study, topologies, prediction_variants, seeds, output_root
            )
        )
        submitted["evaluation_tasks"] = len(evaluation_tasks)
        submitted["plot_tasks"] = len(plot_tasks)
        submitted["postprocess_tasks"] = len(evaluation_tasks) + len(plot_tasks)
        submitted["skipped_complete_evaluation_tasks"] = [
            list(task) for task in skipped_evaluation
        ]
        submitted["skipped_complete_plot_tasks"] = [list(task) for task in skipped_plot]
        fake_id = 0
        evaluation_job = None
        plot_job = None
        if evaluation_tasks:
            evaluation_file = slurm_root / "evaluate.tsv"
            evaluation_worker = slurm_root / "evaluate.sh"
            _write_postprocess_task_file(evaluation_file, evaluation_tasks)
            _postprocess_worker_script(
                evaluation_worker,
                task_file=evaluation_file,
                study_path=manifest_snapshot,
                stages=("evaluate", "plot"),
                output_root=output_root,
                expected_commit=git_commit,
                expected_manifest_sha256=manifest_sha256,
                expected_gate_digest=study.gates_digest,
                expected_experiments=base_experiments,
                overrides=overrides,
            )
            evaluation_job = _submit(
                _sbatch_array(
                    worker=evaluation_worker,
                    task_count=len(evaluation_tasks),
                    job_name=f"jer-{study.name[:18]}-eval",
                    log_dir=log_dir,
                    partition=args.cpu_partition,
                    gres=None,
                    cpus=args.cpu_cpus,
                    memory=args.cpu_mem,
                    time_limit=args.cpu_time,
                    dependency=args.dependency_afterok,
                    extra=cpu_extra,
                ),
                dry_run=args.dry_run,
                fake_id=fake_id,
            )
            fake_id += 1
            submitted["evaluation_task_file"] = str(evaluation_file)
            submitted["evaluation_worker"] = str(evaluation_worker)
        if plot_tasks:
            plot_file = slurm_root / "plot.tsv"
            plot_worker = slurm_root / "plot.sh"
            _write_postprocess_task_file(plot_file, plot_tasks)
            _postprocess_worker_script(
                plot_worker,
                task_file=plot_file,
                study_path=manifest_snapshot,
                stages=("plot",),
                output_root=output_root,
                expected_commit=git_commit,
                expected_manifest_sha256=manifest_sha256,
                expected_gate_digest=study.gates_digest,
                expected_experiments=base_experiments,
                overrides=overrides,
            )
            plot_dependency = args.dependency_afterok
            if evaluation_job is not None:
                plot_dependency = evaluation_job
            plot_job = _submit(
                _sbatch_array(
                    worker=plot_worker,
                    task_count=len(plot_tasks),
                    job_name=f"jer-{study.name[:18]}-plot",
                    log_dir=log_dir,
                    partition=args.cpu_partition,
                    gres=None,
                    cpus=args.cpu_cpus,
                    memory=args.cpu_mem,
                    time_limit=args.cpu_time,
                    dependency=plot_dependency,
                    extra=cpu_extra,
                ),
                dry_run=args.dry_run,
                fake_id=fake_id,
            )
            fake_id += 1
            submitted["plot_task_file"] = str(plot_file)
            submitted["plot_worker"] = str(plot_worker)
        submitted["evaluation_job"] = evaluation_job
        submitted["plot_job"] = plot_job

        aggregate_commands = [
            _post_command(
                manifest_snapshot,
                stage,
                output_root=output_root,
                topologies=topologies,
                variants=prediction_variants,
                seeds=seeds,
                overrides=overrides,
                expected_gate_digest=study.gates_digest,
            )
            for stage in ("compare", "summary", "analyze")
        ]
        aggregate_path = output_root / study.name / "summary_aggregate.json"
        aggregate_job = None
        if not (
            not evaluation_tasks
            and not plot_tasks
            and aggregate_path.is_file()
            and aggregate_path.stat().st_size > 0
        ):
            aggregate_worker = slurm_root / "aggregate.sh"
            _post_script(
                aggregate_worker,
                aggregate_commands,
                git_commit,
                manifest_snapshot,
                manifest_sha256,
                base_experiments,
            )
            dependencies = [job for job in (evaluation_job, plot_job) if job is not None]
            if not dependencies and args.dependency_afterok:
                dependencies.append(args.dependency_afterok)
            dependency_text = ":".join(dependencies) if dependencies else None
            aggregate_command = [
                "sbatch", "--parsable",
                f"--job-name=jer-{study.name[:18]}-agg",
                f"--partition={args.cpu_partition}",
                f"--cpus-per-task={args.cpu_cpus}",
                f"--mem={args.cpu_mem}",
                f"--time={args.cpu_time}",
                f"--output={log_dir}/%x-%j.out",
                f"--error={log_dir}/%x-%j.err",
            ]
            if dependency_text:
                aggregate_command.append(f"--dependency=afterok:{dependency_text}")
            aggregate_command.extend((*cpu_extra, str(aggregate_worker)))
            aggregate_job = _submit(
                aggregate_command,
                dry_run=args.dry_run,
                fake_id=fake_id,
            )
            submitted["aggregate_worker"] = str(aggregate_worker)
        submitted["aggregate_job"] = aggregate_job
        submitted["aggregate_depends_on"] = [
            job for job in (evaluation_job, plot_job) if job is not None
        ]
        _save_submission(submission_path, submitted)
        slurm_base.mkdir(parents=True, exist_ok=True)
        (slurm_base / "latest.txt").write_text(str(slurm_root) + "\n")
        print(json.dumps(submitted, indent=2, sort_keys=True))
        return

    dependency = args.dependency_afterok
    fake_id = 0
    for level_index, variants in enumerate(_levels(study, training)):
        tasks = [
            (topology, variant.name, seed)
            for topology in topologies
            for variant in variants
            for seed in seeds
        ]
        task_file = slurm_root / f"train_level_{level_index}.tsv"
        worker = slurm_root / f"train_level_{level_index}.sh"
        _write_task_file(task_file, tasks)
        _worker_script(
            worker,
            task_file=task_file,
            study_path=manifest_snapshot,
            stage="train",
            output_root=output_root,
            device="cuda",
            global_overrides=overrides,
            expected_commit=git_commit,
            expected_manifest_sha256=manifest_sha256,
            expected_gate_digest=study.gates_digest,
            expected_experiments=base_experiments,
        )
        job_id = _submit(
            _sbatch_array(
                worker=worker,
                task_count=len(tasks),
                job_name=f"jer-{study.name[:18]}-tr{level_index}",
                log_dir=log_dir,
                partition=args.gpu_partition,
                gres=args.gpu_gres,
                cpus=args.gpu_cpus,
                memory=args.gpu_mem,
                time_limit=args.gpu_time,
                dependency=dependency,
                extra=gpu_extra,
            ),
            dry_run=args.dry_run,
            fake_id=fake_id,
        )
        fake_id += 1
        submitted["train_levels"].append(
            {
                "level": level_index,
                "variants": [variant.name for variant in variants],
                "job_id": job_id,
                "depends_on": dependency,
                "task_file": str(task_file),
                "worker": str(worker),
            }
        )
        dependency = job_id
        _save_submission(submission_path, submitted)

    prediction_variants = tuple(
        variant for variant in requested if variant.reporting != "setup"
    )
    if prediction_variants:
        all_prediction_tasks = [
            (topology, variant.name, seed)
            for topology in topologies
            for variant in prediction_variants
            for seed in seeds
        ]
        if args.prediction_only:
            prediction_tasks = [
                task
                for task in all_prediction_tasks
                if not _prediction_task_complete(
                    study,
                    task[0],
                    study.variant_map[task[1]],
                    task[2],
                    output_root=output_root,
                    overrides=overrides,
                    git_commit=git_commit,
                )
            ]
            submitted["skipped_complete_prediction_tasks"] = [
                list(task) for task in all_prediction_tasks if task not in prediction_tasks
            ]
        else:
            prediction_tasks = all_prediction_tasks
        if not prediction_tasks:
            submitted["prediction_tasks"] = 0
            submitted["prediction_job"] = None
            submitted["postprocess_job"] = None
            _save_submission(submission_path, submitted)
            slurm_base.mkdir(parents=True, exist_ok=True)
            (slurm_base / "latest.txt").write_text(str(slurm_root) + "\n")
            print("All requested prediction identities are already complete; no jobs submitted.")
            print(json.dumps(submitted, indent=2, sort_keys=True))
            return
        prediction_file = slurm_root / "predict.tsv"
        prediction_worker = slurm_root / "predict.sh"
        _write_task_file(prediction_file, prediction_tasks)
        _worker_script(
            prediction_worker,
            task_file=prediction_file,
            study_path=manifest_snapshot,
            stage="predict",
            output_root=output_root,
            device="cuda",
            global_overrides=overrides,
            expected_commit=git_commit,
            expected_manifest_sha256=manifest_sha256,
            expected_gate_digest=study.gates_digest,
            expected_experiments=base_experiments,
        )
        prediction_job = _submit(
            _sbatch_array(
                worker=prediction_worker,
                task_count=len(prediction_tasks),
                job_name=f"jer-{study.name[:18]}-pred",
                log_dir=log_dir,
                partition=args.gpu_partition,
                gres=args.gpu_gres,
                cpus=args.gpu_cpus,
                memory=args.gpu_mem,
                time_limit=args.gpu_time,
                dependency=dependency,
                extra=gpu_extra,
            ),
            dry_run=args.dry_run,
            fake_id=fake_id,
        )
        fake_id += 1
        submitted["prediction_job"] = prediction_job
        submitted["prediction_tasks"] = len(prediction_tasks)
        submitted["prediction_depends_on"] = dependency
        submitted["prediction_task_file"] = str(prediction_file)
        submitted["prediction_worker"] = str(prediction_worker)
        _save_submission(submission_path, submitted)

        if not args.no_postprocess:
            all_postprocess_tasks = [
                (topology, variant.name, seed, prediction.name)
                for topology in topologies
                for variant in prediction_variants
                for seed in seeds
                for prediction in variant.predictions
            ]
            postprocess_file = slurm_root / "evaluate.tsv"
            postprocess_worker = slurm_root / "evaluate.sh"
            _write_postprocess_task_file(postprocess_file, all_postprocess_tasks)
            _postprocess_worker_script(
                postprocess_worker,
                task_file=postprocess_file,
                study_path=manifest_snapshot,
                stages=("evaluate", "plot"),
                output_root=output_root,
                expected_commit=git_commit,
                expected_manifest_sha256=manifest_sha256,
                expected_gate_digest=study.gates_digest,
                expected_experiments=base_experiments,
                overrides=overrides,
            )
            evaluation_job = _submit(
                _sbatch_array(
                    worker=postprocess_worker,
                    task_count=len(all_postprocess_tasks),
                    job_name=f"jer-{study.name[:18]}-eval",
                    log_dir=log_dir,
                    partition=args.cpu_partition,
                    gres=None,
                    cpus=args.cpu_cpus,
                    memory=args.cpu_mem,
                    time_limit=args.cpu_time,
                    dependency=prediction_job,
                    extra=cpu_extra,
                ),
                dry_run=args.dry_run,
                fake_id=fake_id,
            )
            fake_id += 1
            post_commands = [
                _post_command(
                    manifest_snapshot,
                    stage,
                    output_root=output_root,
                    topologies=topologies,
                    variants=prediction_variants,
                    seeds=seeds,
                    overrides=overrides,
                    expected_gate_digest=study.gates_digest,
                )
                for stage in ("compare", "summary", "analyze")
            ]
            aggregate_worker = slurm_root / "aggregate.sh"
            _post_script(
                aggregate_worker,
                post_commands,
                git_commit,
                manifest_snapshot,
                manifest_sha256,
                base_experiments,
            )
            aggregate_job = _submit(
                [
                    "sbatch", "--parsable",
                    f"--job-name=jer-{study.name[:18]}-agg",
                    f"--partition={args.cpu_partition}",
                    f"--cpus-per-task={args.cpu_cpus}",
                    f"--mem={args.cpu_mem}",
                    f"--time={args.cpu_time}",
                    f"--dependency=afterok:{evaluation_job}",
                    f"--output={log_dir}/%x-%j.out",
                    f"--error={log_dir}/%x-%j.err",
                    *cpu_extra,
                    str(aggregate_worker),
                ],
                dry_run=args.dry_run,
                fake_id=fake_id,
            )
            submitted["evaluation_job"] = evaluation_job
            submitted["evaluation_tasks"] = len(all_postprocess_tasks)
            submitted["evaluation_task_file"] = str(postprocess_file)
            submitted["evaluation_worker"] = str(postprocess_worker)
            submitted["aggregate_job"] = aggregate_job
            submitted["aggregate_depends_on"] = evaluation_job
            submitted["aggregate_worker"] = str(aggregate_worker)
            # Preserve the historical key for consumers of old submission records.
            submitted["postprocess_job"] = aggregate_job
            submitted["postprocess_depends_on"] = evaluation_job
            submitted["postprocess_worker"] = str(aggregate_worker)
            _save_submission(submission_path, submitted)

    _save_submission(submission_path, submitted)
    slurm_base.mkdir(parents=True, exist_ok=True)
    (slurm_base / "latest.txt").write_text(str(slurm_root) + "\n")
    print(json.dumps(submitted, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
