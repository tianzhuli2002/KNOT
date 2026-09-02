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

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.studies import StudySpec, VariantSpec, load_study


_CAMPAIGN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
SELECTIONS_PATH = ROOT / "configs/studies/UPSTREAM_SELECTIONS.yaml"
_GATE_KEYS = {
    "status",
    "source_study",
    "selected_variant",
    "validated_sha",
    "required_by",
}
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


def _scientific_gates() -> dict:
    if not SELECTIONS_PATH.is_file():
        raise FileNotFoundError(f"Scientific selection gates do not exist: {SELECTIONS_PATH}")
    raw = OmegaConf.to_container(OmegaConf.load(SELECTIONS_PATH), resolve=True)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Scientific selection gates must be a non-empty mapping: {SELECTIONS_PATH}")
    study_paths = {
        load_study(path).name: path
        for path in sorted((ROOT / "configs/studies").glob("[0-9][0-9]_*.yaml"))
    }
    for name, gate in raw.items():
        if not isinstance(gate, dict) or set(gate) != _GATE_KEYS:
            raise ValueError(
                f"Scientific gate {name!r} must contain exactly {sorted(_GATE_KEYS)}"
            )
        if gate["status"] not in {"pending", "locked"}:
            raise ValueError(f"Scientific gate {name!r} has invalid status {gate['status']!r}")
        source_study = str(gate["source_study"]).strip()
        if not source_study:
            raise ValueError(f"Scientific gate {name!r} has no source_study")
        if source_study not in study_paths:
            raise ValueError(
                f"Scientific gate {name!r} refers to unknown source study "
                f"{source_study!r}"
            )
        required_by = gate["required_by"]
        if (
            not isinstance(required_by, list)
            or not required_by
            or any(not str(study).strip() for study in required_by)
            or len(set(required_by)) != len(required_by)
        ):
            raise ValueError(f"Scientific gate {name!r} has invalid required_by")
        unknown_dependents = sorted(set(required_by) - set(study_paths))
        if unknown_dependents:
            raise ValueError(
                f"Scientific gate {name!r} refers to unknown dependent studies "
                f"{unknown_dependents}"
            )
        if gate["status"] == "locked":
            selected_variant = str(gate["selected_variant"]).strip()
            if not selected_variant:
                raise ValueError(f"Locked scientific gate {name!r} has no selected_variant")
            if selected_variant not in load_study(
                study_paths[source_study]
            ).variant_map:
                raise ValueError(
                    f"Locked scientific gate {name!r} selects unknown variant "
                    f"{selected_variant!r} from {source_study!r}"
                )
            if not isinstance(gate["validated_sha"], str) or not _COMMIT_SHA.fullmatch(
                gate["validated_sha"]
            ):
                raise ValueError(
                    f"Locked scientific gate {name!r} must record a 40-character validated_sha"
                )
    return raw


def _require_locked_scientific_gates(study_name: str, selections: dict | None = None) -> None:
    selections = _scientific_gates() if selections is None else selections
    pending = [
        name
        for name, gate in selections.items()
        if study_name in gate["required_by"] and gate["status"] != "locked"
    ]
    if pending:
        raise RuntimeError(
            f"Study {study_name!r} is decision-gated by pending selections {pending}. "
            f"Record the human-reviewed winner in {SELECTIONS_PATH} and update the "
            "study's explicit reference patch before submission."
        )


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
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a JER study to Slurm")
    parser.add_argument("study", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/studies"))
    parser.add_argument("--topologies", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--gpu-partition", default="gpuL")
    parser.add_argument("--gpu-gres", default="gpu:1")
    parser.add_argument("--gpu-cpus", type=int, default=12)
    parser.add_argument("--gpu-mem", default="64G")
    parser.add_argument("--gpu-time", default="4-00:00:00")
    parser.add_argument("--cpu-partition", default="multicore")
    parser.add_argument("--cpu-cpus", type=int, default=4)
    parser.add_argument("--cpu-mem", default="16G")
    parser.add_argument("--cpu-time", default="04:00:00")
    parser.add_argument("--set", dest="global_overrides", action="append", default=[])
    parser.add_argument(
        "--extra-sbatch",
        action="append",
        default=[],
        help="Pass one additional sbatch argument, e.g. --extra-sbatch=--account=project.",
    )
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument(
        "--dependency-afterok",
        help="External job ID that must succeed before the first training layer starts.",
    )
    parser.add_argument(
        "--campaign-id",
        help="Human-readable immutable Slurm-attempt identifier; defaults to a UTC timestamp.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    study = load_study(args.study)
    selections = _scientific_gates()
    if not args.dry_run:
        _require_locked_scientific_gates(study.name, selections)
    topologies = _selected(args.topologies, study.topologies, "topologies")
    variant_names = tuple(variant.name for variant in study.variants)
    requested_names = _selected(args.variants, variant_names, "variants")
    requested = tuple(study.variant_map[name] for name in requested_names)
    training = (
        study.dependency_order()
        if args.variants == "all"
        else study.dependency_closure(requested_names)
    )
    seeds = (
        study.seeds
        if args.seeds == "all"
        else tuple(int(value) for value in args.seeds.split(",") if value)
    )
    unknown_seeds = sorted(set(seeds) - set(study.seeds))
    if unknown_seeds:
        raise ValueError(f"Seeds {unknown_seeds} are not declared by the study")

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
    extra = tuple(args.extra_sbatch)
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
        "extra_sbatch": list(extra),
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
    }
    submission_path = slurm_root / "submission.json"
    _save_submission(submission_path, submitted)
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
                extra=extra,
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
        prediction_tasks = [
            (topology, variant.name, seed)
            for topology in topologies
            for variant in prediction_variants
            for seed in seeds
        ]
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
                extra=extra,
            ),
            dry_run=args.dry_run,
            fake_id=fake_id,
        )
        fake_id += 1
        submitted["prediction_job"] = prediction_job
        submitted["prediction_depends_on"] = dependency
        submitted["prediction_task_file"] = str(prediction_file)
        submitted["prediction_worker"] = str(prediction_worker)
        _save_submission(submission_path, submitted)

        if not args.no_postprocess:
            post_worker = slurm_root / "postprocess.sh"
            post_commands = [
                _post_command(
                    manifest_snapshot,
                    stage,
                    output_root=output_root,
                    topologies=topologies,
                    variants=prediction_variants,
                    seeds=seeds,
                    overrides=overrides,
                )
                for stage in ("evaluate", "plot", "compare", "summary", "analyze")
            ]
            _post_script(
                post_worker,
                post_commands,
                git_commit,
                manifest_snapshot,
                manifest_sha256,
                base_experiments,
            )
            post_job = _submit(
                [
                    "sbatch", "--parsable",
                    f"--job-name=jer-{study.name[:18]}-post",
                    f"--partition={args.cpu_partition}",
                    f"--cpus-per-task={args.cpu_cpus}",
                    f"--mem={args.cpu_mem}",
                    f"--time={args.cpu_time}",
                    f"--dependency=afterok:{prediction_job}",
                    f"--output={log_dir}/%x-%j.out",
                    f"--error={log_dir}/%x-%j.err",
                    *extra,
                    str(post_worker),
                ],
                dry_run=args.dry_run,
                fake_id=fake_id,
            )
            submitted["postprocess_job"] = post_job
            submitted["postprocess_depends_on"] = prediction_job
            submitted["postprocess_worker"] = str(post_worker)
            _save_submission(submission_path, submitted)

    _save_submission(submission_path, submitted)
    slurm_base.mkdir(parents=True, exist_ok=True)
    (slurm_base / "latest.txt").write_text(str(slurm_root) + "\n")
    print(json.dumps(submitted, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
