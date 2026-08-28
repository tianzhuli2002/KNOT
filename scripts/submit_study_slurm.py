#!/usr/bin/env python3
"""Submit one study manifest as dependency-ordered Slurm arrays.

Training variants are grouped by initialisation dependency depth. Predictions run only
after every required training level succeeds; evaluation/plotting/summary run afterwards
on CPU. The generated task files make the submitted campaign auditable and restartable.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.studies import StudySpec, VariantSpec, load_study


def _levels(study: StudySpec) -> list[tuple[VariantSpec, ...]]:
    remaining = dict(study.variant_map)
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
            )
        )
        if not ready:
            raise RuntimeError("Study dependency graph has no schedulable variant")
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
        command,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch job id from {result.stdout!r}")
    print(f"submitted job {job_id}")
    return job_id


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
) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_study.py"),
        str(study_path),
        stage,
        "--output-root",
        str(output_root),
        "--topologies",
        '"$topology"',
        "--variants",
        '"$variant"',
        "--seeds",
        '"$seed"',
        "--device",
        device,
    ]
    for override in global_overrides:
        command += ["--set", override]
    rendered = " ".join(
        token if token.startswith('"$') else shlex.quote(token) for token in command
    )
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"task_file={shlex.quote(str(task_file))}\n"
        'line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$task_file")\n'
        "if [[ -z \"$line\" ]]; then echo \"Missing study task for index $SLURM_ARRAY_TASK_ID\" >&2; exit 2; fi\n"
        "IFS=$'\\t' read -r topology variant seed <<< \"$line\"\n"
        f"exec {rendered}\n"
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
    command = [
        "sbatch",
        "--parsable",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a JER study to Slurm")
    parser.add_argument("study", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/studies"))
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    study = load_study(args.study)
    study_path = study.path
    output_root = args.output_root.expanduser().resolve()
    slurm_root = output_root / study.name / "_slurm"
    log_dir = slurm_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    overrides = tuple(args.global_overrides)
    extra = tuple(args.extra_sbatch)

    submitted: dict[str, object] = {"study": study.name, "train_levels": []}
    dependency = None
    fake_id = 0
    for level_index, variants in enumerate(_levels(study)):
        tasks = [
            (topology, variant.name, seed)
            for topology in study.topologies
            for variant in variants
            for seed in study.seeds
        ]
        task_file = slurm_root / f"train_level_{level_index}.tsv"
        worker = slurm_root / f"train_level_{level_index}.sh"
        _write_task_file(task_file, tasks)
        _worker_script(
            worker,
            task_file=task_file,
            study_path=study_path,
            stage="train",
            output_root=output_root,
            device="cuda",
            global_overrides=overrides,
        )
        command = _sbatch_array(
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
        )
        job_id = _submit(command, dry_run=args.dry_run, fake_id=fake_id)
        fake_id += 1
        submitted["train_levels"].append(
            {"level": level_index, "variants": [v.name for v in variants], "job_id": job_id}
        )
        dependency = job_id

    prediction_variants = tuple(
        variant for variant in study.variants if variant.reporting != "setup"
    )
    prediction_tasks = [
        (topology, variant.name, seed)
        for topology in study.topologies
        for variant in prediction_variants
        for seed in study.seeds
    ]
    prediction_file = slurm_root / "predict.tsv"
    prediction_worker = slurm_root / "predict.sh"
    _write_task_file(prediction_file, prediction_tasks)
    _worker_script(
        prediction_worker,
        task_file=prediction_file,
        study_path=study_path,
        stage="predict",
        output_root=output_root,
        device="cuda",
        global_overrides=overrides,
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

    if not args.no_postprocess:
        post = " && ".join(
            shlex.join(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_study.py"),
                    str(study_path),
                    stage,
                    "--output-root",
                    str(output_root),
                    "--device",
                    "cpu",
                    *sum((["--set", value] for value in overrides), []),
                ]
            )
            for stage in ("evaluate", "plot", "compare", "summary")
        )
        command = [
            "sbatch",
            "--parsable",
            f"--job-name=jer-{study.name[:18]}-post",
            f"--partition={args.cpu_partition}",
            f"--cpus-per-task={args.cpu_cpus}",
            f"--mem={args.cpu_mem}",
            f"--time={args.cpu_time}",
            f"--dependency=afterok:{prediction_job}",
            f"--output={log_dir}/%x-%j.out",
            f"--error={log_dir}/%x-%j.err",
            *extra,
            "--wrap",
            post,
        ]
        post_job = _submit(command, dry_run=args.dry_run, fake_id=fake_id)
        submitted["postprocess_job"] = post_job

    (slurm_root / "submission.json").write_text(
        json.dumps(submitted, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(submitted, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
