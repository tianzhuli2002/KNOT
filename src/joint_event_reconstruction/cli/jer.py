"""Human-facing JER project, validation and campaign command line interface."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import h5py
from omegaconf import OmegaConf

from joint_event_reconstruction.runtime_paths import (
    ensure_source_snapshot,
    runtime_paths,
    verify_source_snapshot,
)
from joint_event_reconstruction.studies import (
    ScientificGateNotApplicable,
    ScientificGatePending,
    load_scientific_gates,
    load_study,
    resolve_study,
)


_LEGACY_COMMANDS = {
    "train",
    "predict",
    "evaluate",
    "plot",
    "validate-data",
    "show-config",
    "config",
    "inputs",
    "cache",
    "topology",
    "checkpoint",
}
_ACTIVE_PAPER1 = {
    "invisible_coordinates",
    "lgatr_invisible_generator",
    "compatibility_reranking",
    "assignment_conditioning",
}
_PAPER1_IDS = ("02", "08", "13", "15", "10", "03", "04", "06", "07", "09")
_JOB_STATE_TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED"}


def _paths():
    return runtime_paths()


def _run(command: list[str], *, env: dict[str, str] | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=_paths().project_root,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def _git(*args: str, check: bool = True) -> str:
    result = _run(["git", *args], check=False)
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _study_paths() -> dict[str, Path]:
    root = _paths().source_root / "configs/studies"
    result: dict[str, Path] = {}
    for path in sorted(root.glob("[0-9][0-9]_*.yaml")):
        study = load_study(path)
        if study.name in result:
            raise ValueError(f"Duplicate maintained study {study.name!r}")
        result[study.name] = path
    return result


def _study_ids() -> dict[str, str]:
    return {
        path.name[:2]: name
        for name, path in _study_paths().items()
    }


def _resolve_study(value: str):
    paths = _study_paths()
    value = value.strip()
    if value in _study_ids():
        value = _study_ids()[value]
    if value.endswith(".yaml"):
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = _paths().source(candidate)
        return load_study(candidate.resolve())
    if value not in paths:
        matches = [name for name in paths if name.startswith(value)]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous study {value!r}; use 13 or a maintained name")
        value = matches[0]
    return load_study(paths[value])


def _study_number(study_name: str) -> str:
    for number, name in _study_ids().items():
        if name == study_name:
            return number
    return "--"


def _submission_records(study_name: str) -> list[tuple[Path, dict]]:
    root = _paths().results_root / "studies" / study_name / "_slurm"
    records: list[tuple[Path, dict]] = []
    for path in sorted(root.glob("*/submission.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        records.append((path, payload))
    return records


def _latest_submission(study_name: str, source_sha: str | None = None) -> tuple[Path, dict] | None:
    candidates = _submission_records(study_name)
    if source_sha is not None:
        candidates = [
            item for item in candidates
            if item[1].get("source_sha", item[1].get("git_commit")) == source_sha
        ]
    return candidates[-1] if candidates else None


def _job_state(job_id: object) -> tuple[str, str]:
    if not job_id:
        return "NONE", ""
    job = str(job_id)
    queued = _run(["squeue", "-h", "-j", job, "-o", "%T|%R"], check=False)
    if queued.returncode == 0 and queued.stdout.strip():
        fields = queued.stdout.strip().split("|", 1)
        return fields[0].strip().upper(), fields[1].strip() if len(fields) > 1 else ""
    accounting = _run(
        ["sacct", "-X", "-n", "-P", "-j", job, "--format=State,ExitCode"],
        check=False,
    )
    rows = [line.strip() for line in accounting.stdout.splitlines() if line.strip()]
    if not rows:
        return "UNKNOWN", ""
    state, _, exit_code = rows[-1].partition("|")
    return state.split("+", 1)[0].split(" ", 1)[0].upper(), exit_code


def _record_job_ids(record: dict) -> list[str]:
    ids: list[str] = []
    for level in record.get("train_levels", []):
        if level.get("job_id"):
            ids.append(str(level["job_id"]))
    for key in ("prediction_job", "evaluation_job", "plot_job", "aggregate_job", "postprocess_job"):
        if record.get(key):
            ids.append(str(record[key]))
    return ids


def _active_jobs() -> list[str]:
    result = _run(
        ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%.18i %.36j %.10T %.12M %R"],
        check=False,
    )
    return [line for line in result.stdout.splitlines() if re.search(r"jer|invis|study|hypot", line, re.I)]


def _outputs_for(study_name: str) -> tuple[int, int, int]:
    root = _paths().results_root / "studies" / study_name
    predictions = list(root.glob("ttbar2L/*/seed_*/predictions/*.h5"))
    identities = list(root.glob("ttbar2L/*/seed_*/predictions/*.identity.json"))
    return len(predictions), len(identities), len(list(root.glob("study*_human_review.json")))


def _historical_status(number: str, name: str) -> tuple[str, str] | None:
    root = _paths().results_root / "studies"
    if number == "02":
        review = root / "generator_formulations/study02_human_review.json"
        predictions, identities, _ = _outputs_for(name)
        if review.is_file() and predictions >= 12 and identities >= 12:
            return "COMPLETE", "validated 12 prediction products and Study-02 review"
    if number == "08":
        review = root / "generator_inference/study08_human_review.json"
        predictions, identities, _ = _outputs_for(name)
        if review.is_file() and predictions >= 33 and identities >= 33:
            return "COMPLETE_WITH_WARNING", "33 validated products exist; historical aggregate 20608915 timed out after outputs"
    if number == "12":
        review = root / "encoder_backbone/study12_human_review.json"
        if review.is_file() and (root / "encoder_backbone/summary.json").is_file():
            return "COMPLETE", "validated Study-12 review and summary"
    if number == "01":
        if (root / "visible_recurrence/summary.json").is_file() or (root / "visible_recurrence-final-20260913/visible_recurrence/summary.json").is_file():
            return "COMPLETE", "validated historical Study-01 products"
    if number == "05" and (root / "discrete_assignment/assignment_formulation_review.json").is_file():
        return "COMPLETE", "validated Study-05 review; discriminative assignment retained"
    return None


def _campaign_status(study, *, source_sha: str) -> tuple[str, str]:
    number = _study_number(study.name)
    if not study.campaign_enabled:
        return study.campaign_state.upper(), study.campaign_reason
    historical = _historical_status(number, study.name)
    if historical is not None:
        return historical
    try:
        resolve_study(study.path)
    except ScientificGatePending as error:
        return "BLOCKED_GATE", str(error)
    except ScientificGateNotApplicable as error:
        return "NOT_APPLICABLE", str(error)
    record_item = _latest_submission(study.name, source_sha)
    if record_item is None:
        return "READY", "no submission for current source SHA"
    path, record = record_item
    ids = _record_job_ids(record)
    states = [_job_state(job)[0] for job in ids]
    if any(state in {"PENDING", "CONFIGURING"} for state in states):
        return "PENDING", f"{path.parent.name}; scheduler state {next(state for state in states if state in {'PENDING', 'CONFIGURING'})}"
    if any(state in {"RUNNING", "COMPLETING"} for state in states):
        return "RUNNING", f"{path.parent.name}; real jobs are active"
    if any(state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "PREEMPTED"} for state in states):
        return "FAILED", f"{path.parent.name}; Slurm state {next(state for state in states if state in _JOB_STATE_TERMINAL - {'COMPLETED'})}"
    return "SUBMITTED", f"{path.parent.name}; awaiting validated outputs"


def _print_gates() -> None:
    gates = load_scientific_gates()
    for name, gate in gates.items():
        selected = gate.get("selected_variant") or "-"
        print(f"  {name:24s} {str(gate['status']).upper():14s} {selected}")


def _print_campaign() -> None:
    source_sha = _git("rev-parse", "HEAD")
    print("Paper-1 studies")
    for number in _PAPER1_IDS:
        name = _study_ids().get(number)
        if name is None:
            continue
        study = _resolve_study(name)
        status, reason = _campaign_status(study, source_sha=source_sha)
        print(f"  Study {number:>2s} {name:28s} {status:20s} {reason}")


def _print_status() -> None:
    paths = _paths()
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    origin_main = _git("rev-parse", "origin/main", check=False) or "unavailable"
    dirty = bool(_git("status", "--porcelain", check=False))
    divergence = _git("rev-list", "--left-right", "--count", "origin/main...HEAD", check=False) or "unavailable"
    print("Repository")
    print(f"  branch             {branch or '(detached)'}")
    print(f"  HEAD               {head}")
    print(f"  origin/main        {origin_main}")
    print(f"  dirty?             {'YES' if dirty else 'no'}")
    print(f"  origin/main...HEAD {divergence}")
    print("  authoritative      main (after fast-forward consolidation)")
    print("\nEnvironment")
    for label, value in (
        ("project root", paths.project_root),
        ("source root", paths.source_root),
        ("data path", paths.data_root),
        ("cache path", paths.cache_root),
        ("results path", paths.results_root),
        ("runtime path", paths.runtime_root),
    ):
        print(f"  {label:18s} {value}")
    dataset = paths.project("data/real/ttbar2L_canonical.h5")
    print(f"  dataset exists?    {'yes' if dataset.is_file() else 'NO'}")
    print("\nScientific gates")
    _print_gates()
    print()
    _print_campaign()
    jobs = _active_jobs()
    print("\nSlurm")
    if jobs:
        for line in jobs:
            print(f"  {line}")
    else:
        print("  no active JER jobs")
    smoke = _successful_smoke(head)
    print("\nNext action")
    if not smoke:
        print("  Run: jer doctor && jer smoke gpu --watch")
    else:
        status, _ = _campaign_status(_resolve_study("13"), source_sha=head)
        if status in {"READY", "BLOCKED_SMOKE"}:
            print("  Run: jer study plan 13 && jer study submit 13 --watch")
        else:
            print("  Monitor Study 13, then review invisible_coordinates before Study 15.")


def _validate_gate_artifacts() -> list[str]:
    failures: list[str] = []
    gates = load_scientific_gates()
    for gate_name, gate in gates.items():
        if gate["status"] != "locked":
            continue
        for topology, entries in (gate.get("artifacts") or {}).items():
            for seed, entry in entries.items():
                checkpoint = Path(entry["checkpoint"])
                identity = Path(entry["identity_file"])
                if not checkpoint.is_file():
                    failures.append(f"{gate_name}:{topology}:{seed}: missing checkpoint {checkpoint}")
                    continue
                if not identity.is_file():
                    failures.append(f"{gate_name}:{topology}:{seed}: missing identity {identity}")
                    continue
                if _sha256(checkpoint) != entry["checkpoint_sha256"]:
                    failures.append(f"{gate_name}:{topology}:{seed}: checkpoint hash mismatch")
                if _sha256(identity) != entry["identity_sha256"]:
                    failures.append(f"{gate_name}:{topology}:{seed}: identity hash mismatch")
                try:
                    import torch

                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    if not isinstance(payload.get("state_dict"), dict):
                        failures.append(f"{gate_name}:{topology}:{seed}: checkpoint has no state_dict")
                except Exception as error:  # pragma: no cover - diagnostic boundary
                    failures.append(f"{gate_name}:{topology}:{seed}: checkpoint load failed: {error}")
    return failures


def _local_data_check() -> None:
    paths = _paths()
    dataset = paths.project("data/real/ttbar2L_canonical.h5")
    with h5py.File(dataset, "r") as source:
        missing = sorted({"EVENT_ID", "SPLIT", "INPUTS", "LABELS"} - set(source))
        if missing:
            raise RuntimeError(f"Canonical dataset missing groups/datasets {missing}: {dataset}")
    from joint_event_reconstruction.cli.run import build_data
    from joint_event_reconstruction.configuration import load_experiment, load_visible_config

    config = load_experiment(
        paths.source("configs/experiments/ttbar2L.yaml"),
        [
            "data.events.train=2",
            "data.events.validation=2",
            "data.events.test=2",
            "data.cache=null",
            "training.workers=0",
        ],
    )
    data = build_data(config, load_visible_config(config))
    data.setup("predict")
    # The maintained structured data module exposes split loaders rather than
    # Lightning's optional predict loader. ``setup("predict")`` still forces
    # the full prediction-time validation path; read one validation batch here
    # to exercise the canonical input contract without a GPU.
    next(iter(data.val_dataloader()))


def _doctor_local() -> list[str]:
    failures: list[str] = []
    paths = _paths()
    # A local doctor is deliberately CPU-only.  Keep it independent of an
    # accidentally visible or unavailable login-node GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for label, path in (
        ("project root", paths.project_root),
        ("data root", paths.data_root),
        ("cache root", paths.cache_root),
        ("results root", paths.results_root),
        ("runtime root", paths.runtime_root),
    ):
        if not path.is_dir():
            failures.append(f"{label} does not exist: {path}")
    if not (paths.cache_root.is_dir() and os.access(paths.cache_root, os.W_OK)):
        failures.append(f"cache root is not writable: {paths.cache_root}")
    cache_file = paths.cache_root / "ttbar2L_canonical.pt"
    if not (cache_file.is_file() and os.access(cache_file, os.R_OK)):
        failures.append(f"canonical tensor cache is not readable: {cache_file}")
    try:
        _git("rev-parse", "--is-inside-work-tree")
    except RuntimeError as error:
        failures.append(str(error))
    if _run(["git", "ls-remote", "--heads", "origin"], check=False).returncode:
        failures.append("origin remote is not readable")
    try:
        load_scientific_gates()
    except Exception as error:
        failures.append(f"gate registry: {error}")
    failures.extend(_validate_gate_artifacts())
    try:
        _local_data_check()
    except Exception as error:
        failures.append(f"data.setup('predict'): {error}")
    try:
        snapshot = ensure_source_snapshot(_git("rev-parse", "HEAD"), project_root=paths.project_root)
        verify_source_snapshot(paths.source_sha if paths.source_root != paths.project_root else _git("rev-parse", "HEAD"), project_root=paths.project_root, source_root=snapshot)
    except Exception as error:
        failures.append(f"source snapshot: {error}")
    for path in _study_paths().values():
        try:
            load_study(path)
        except Exception as error:
            failures.append(f"study manifest {path.name}: {error}")
    for path in _study_paths().values():
        study = load_study(path)
        if study.campaign_enabled and study.split == "test":
            failures.append(f"active study selects real test split: {study.name}")
    return failures


def _site_slurm() -> dict:
    paths = _paths()
    config_path = paths.project("configs/site/csf3.yaml")
    raw = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return raw["slurm"]


def _worker_environment(source_root: Path, source_sha: str) -> dict[str, str]:
    paths = _paths()
    environment = os.environ.copy()
    environment.update(
        {
            "JER_PROJECT_ROOT": str(paths.project_root),
            "JER_SOURCE_ROOT": str(source_root),
            "JER_SOURCE_SHA": source_sha,
            "PYTHONPATH": f"{source_root / 'src'}:{source_root}{os.pathsep}{environment.get('PYTHONPATH', '')}",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _wait_job(job_id: str, *, cycles: int = 72, interval: float = 5.0) -> tuple[str, str]:
    last = ("UNKNOWN", "")
    for _ in range(cycles):
        last = _job_state(job_id)
        print(f"  job {job_id}: {last[0]} {last[1]}" if last[1] else f"  job {job_id}: {last[0]}", flush=True)
        if last[0] in _JOB_STATE_TERMINAL:
            return last
        time.sleep(interval)
    return last


def _run_compute_node_preflight(*, watch: bool = True) -> tuple[str, str, Path]:
    paths = _paths()
    sha = _git("rev-parse", "HEAD")
    previous = _latest_successful_preflight(sha)
    if previous is not None:
        print(f"CPU preflight already PASS for source SHA {sha}: logs={previous}")
        return "existing", "COMPLETED", previous
    source = ensure_source_snapshot(sha, project_root=paths.project_root)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = paths.logs_root / "preflight" / sha / attempt
    directory.mkdir(parents=True, exist_ok=False)
    result_path = directory / "result.json"
    worker = directory / "worker.sh"
    worker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"export JER_PROJECT_ROOT={paths.project_root}\n"
        f"export JER_SOURCE_ROOT={source}\n"
        f"export JER_SOURCE_SHA={sha}\n"
        f"export PYTHONPATH={source / 'src'}:{source}\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        f"cd {paths.project_root}\n"
        f"exec {sys.executable} {source / 'scripts/compute_node_preflight.py'} --events 2\n"
    )
    worker.chmod(0o755)
    slurm = _site_slurm()
    submitted = _run(
        [
            "sbatch", "--parsable", "--job-name=jer-preflight",
            f"--partition={slurm['cpu_partition']}", f"--cpus-per-task={slurm['cpu_cpus']}",
            f"--mem={slurm['cpu_memory']}", f"--time={slurm['cpu_time']}",
            f"--output={directory}/%x-%j.out", f"--error={directory}/%x-%j.err",
            str(worker),
        ],
        env=_worker_environment(source, sha),
        check=False,
    )
    if submitted.returncode:
        raise RuntimeError(submitted.stderr.strip() or "CPU preflight sbatch failed")
    job_id = submitted.stdout.strip().split(";", 1)[0]
    print(f"CPU preflight submitted: {job_id}")
    state = _wait_job(job_id) if watch else ("SUBMITTED", "")
    if state[0] == "COMPLETED":
        output = next(iter(directory.glob("jer-preflight-*.out")), None)
        if output is not None:
            result_path.write_text(output.read_text())
        return job_id, state[0], directory
    error = next(iter(directory.glob("jer-preflight-*.err")), None)
    if error is not None and error.read_text().strip():
        print(error.read_text().rstrip(), file=sys.stderr)
    return job_id, state[0], directory


def _successful_smoke(source_sha: str) -> dict | None:
    path = _paths().results_root / "validation/gpu-smoke/registry.json"
    if not path.is_file():
        return None
    try:
        records = json.loads(path.read_text()).get("smokes", [])
    except (OSError, json.JSONDecodeError):
        return None
    for record in reversed(records):
        if record.get("source_sha") == source_sha and record.get("status") == "PASS":
            return record
    return None


def _write_registry(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"schema": "jer_gpu_smoke_v1", "smokes": records}, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _smoke_gpu(watch: bool) -> int:
    paths = _paths()
    sha = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", check=False):
        raise RuntimeError("GPU smoke requires a clean committed checkout; commit the source first")
    source = ensure_source_snapshot(sha, project_root=paths.project_root)
    if not _successful_preflight(sha):
        _, state, _ = _run_compute_node_preflight(watch=True)
        if state != "COMPLETED":
            if state in {"PENDING", "RUNNING"}:
                print(f"CPU compute-node preflight is still {state}; no GPU smoke submitted.")
                return 0
            raise RuntimeError(f"CPU compute-node preflight did not complete successfully: {state}")
    registry_path = paths.results_root / "validation/gpu-smoke/registry.json"
    records: list[dict] = []
    if registry_path.is_file():
        records = json.loads(registry_path.read_text()).get("smokes", [])
    active = [record for record in records if record.get("source_sha") == sha and record.get("status") in {"SUBMITTED", "PENDING", "RUNNING"}]
    if active:
        record = active[-1]
        print(f"GPU smoke already submitted for source SHA {sha}: job {record['job_id']}")
        if not watch:
            return 0
        job_id = str(record["job_id"])
        state = _wait_job(job_id)
        record["scheduler_state"] = state[0]
        if state[0] == "COMPLETED":
            result = Path(record["artifact_path"]) / "smoke_result.json"
            record["status"] = "PASS" if result.is_file() and json.loads(result.read_text()).get("status") == "PASS" else "FAILED"
        _write_registry(registry_path, records)
        return 0 if record["status"] == "PASS" else 1
    attempt = len(list((paths.results_root / "validation/gpu-smoke" / sha).glob("attempt-*"))) + 1
    output = paths.results_root / "validation/gpu-smoke" / sha / f"attempt-{attempt:03d}"
    output.mkdir(parents=True, exist_ok=False)
    worker = output / "worker.sh"
    worker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"export JER_PROJECT_ROOT={paths.project_root}\n"
        f"export JER_SOURCE_ROOT={source}\n"
        f"export JER_SOURCE_SHA={sha}\n"
        f"export PYTHONPATH={source / 'src'}:{source}\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        f"cd {paths.project_root}\n"
        f"exec {sys.executable} {source / 'scripts/smoke_hypothesis_gpu.py'} --output-dir {output} --expected-source-sha {sha}\n"
    )
    worker.chmod(0o755)
    slurm = _site_slurm()
    submitted = _run(
        [
            "sbatch", "--parsable", "--job-name=jer-gpu-smoke",
            f"--partition={slurm['gpu_partition']}", f"--gres={slurm['gpu_gres']}",
            f"--cpus-per-task={slurm['gpu_cpus']}", f"--mem={slurm['gpu_memory']}",
            f"--time={slurm['gpu_time']}", f"--output={output}/%x-%j.out", f"--error={output}/%x-%j.err",
            str(worker),
        ],
        env=_worker_environment(source, sha),
        check=False,
    )
    if submitted.returncode:
        raise RuntimeError(submitted.stderr.strip() or "GPU smoke sbatch failed")
    job_id = submitted.stdout.strip().split(";", 1)[0]
    record = {
        "source_sha": sha,
        "job_id": job_id,
        "status": "SUBMITTED",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "artifact_path": str(output),
    }
    records.append(record)
    _write_registry(registry_path, records)
    print(f"GPU smoke submitted: {job_id}\n  source SHA: {sha}\n  artifacts: {output}")
    if not watch:
        return 0
    state = _wait_job(job_id)
    record["scheduler_state"] = state[0]
    result = output / "smoke_result.json"
    record["status"] = "PASS" if state[0] == "COMPLETED" and result.is_file() and json.loads(result.read_text()).get("status") == "PASS" else ("PENDING" if state[0] not in _JOB_STATE_TERMINAL else "FAILED")
    _write_registry(registry_path, records)
    return 0 if record["status"] == "PASS" else (0 if record["status"] == "PENDING" else 1)


def _successful_preflight(source_sha: str) -> bool:
    return _latest_successful_preflight(source_sha) is not None


def _latest_successful_preflight(source_sha: str) -> Path | None:
    root = _paths().logs_root / "preflight" / source_sha
    candidates = sorted(
        [*root.glob("*/result.json"), *root.glob("*/jer-preflight-*.out")],
        reverse=True,
    )
    for result in candidates:
        try:
            if json.loads(result.read_text()).get("status") == "PASS":
                return result.parent
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _study_plan(study_arg: str) -> int:
    study = _resolve_study(study_arg)
    if study.name not in _ACTIVE_PAPER1:
        raise RuntimeError(f"Study {study.name} is not on the active Paper-1 submission allowlist")
    source_sha = _git("rev-parse", "HEAD")
    print(f"Study { _study_number(study.name) } {study.name}")
    print(f"  source SHA: {source_sha}")
    print(f"  manifest: {study.path}")
    print(f"  manifest SHA256: {_sha256(study.path)}")
    print(f"  split: {study.split}")
    print(f"  topologies: {', '.join(study.topologies)}")
    print(f"  seeds: {', '.join(map(str, study.seeds))}")
    training = len(study.training_order()) * len(study.topologies) * len(study.seeds) if study.training else 0
    predictions = sum(len(variant.predictions) * len(study.topologies) * len(study.seeds) for variant in study.variants if variant.reporting != "setup")
    print(f"  training tasks: {training}")
    print(f"  prediction tasks: {predictions}")
    print(f"  evaluation tasks: {predictions}")
    print("  aggregate tasks: 1")
    if study.name == "invisible_coordinates":
        print("  Cartesian: reused selected CFM checkpoint; 0 training tasks")
        print("  MET residual: 3 new training tasks")
        print("  MET exact: 3 new diagnostic training tasks")
        print("  inference: Heun, 4 steps, M=32")
    if study.resolved_gates is None:
        try:
            resolved = resolve_study(study.path)
            print(f"  gate digest: {resolved.gates_digest}")
            for name, gate in resolved.resolved_gates.items():
                print(f"  gate {name}: {gate.get('selected_variant')} validated {gate.get('validated_sha')}")
                for topology, artifacts in sorted((gate.get("artifacts") or {}).items()):
                    for seed, artifact in sorted(artifacts.items()):
                        print(
                            f"    {topology}/{seed}: "
                            f"checkpoint_sha256={artifact.get('checkpoint_sha256')} "
                            f"identity_sha256={artifact.get('identity_sha256')}"
                        )
        except (ScientificGatePending, ScientificGateNotApplicable) as error:
            raise RuntimeError(str(error)) from error
    return 0


def _study_submit(study_arg: str, watch: bool) -> int:
    study = _resolve_study(study_arg)
    if study.name not in _ACTIVE_PAPER1:
        raise RuntimeError(f"Study {study.name} is deferred/superseded or outside the Paper-1 allowlist")
    if not study.campaign_enabled:
        raise RuntimeError(f"Study {study.name} is {study.campaign_state}: {study.campaign_reason}")
    sha = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", check=False):
        raise RuntimeError("Study submission requires a clean committed checkout")
    smoke = _successful_smoke(sha)
    if smoke is None:
        print(f"No successful GPU smoke exists for source SHA {sha}.")
        print("Run:\n    jer smoke gpu --watch")
        return 2
    existing = _latest_submission(study.name, sha)
    identity = hashlib.sha256(f"paper1:{study.name}:{sha}:{_sha256(study.path)}".encode()).hexdigest()
    for path, record in _submission_records(study.name):
        if record.get("submission_identity") == identity:
            print(f"Study {study.name} already submitted exactly once: {path}")
            return 0
    script = _paths().project("scripts/submit_study_slurm.py")
    source = ensure_source_snapshot(sha, project_root=_paths().project_root)
    command = [
        sys.executable, str(script), str(study.path),
        "--output-root", str(_paths().results_root / "studies"),
        "--campaign-id", f"paper1-{study.name}-{sha[:12]}",
        "--submission-identity", identity,
    ]
    result = _run(command, env=_worker_environment(source, sha), check=False)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "study submission failed")
    print(f"Study {study.name} submitted from source SHA {sha}")
    if not watch:
        return 0
    record_item = _latest_submission(study.name, sha)
    if record_item is None:
        raise RuntimeError("Submission completed without a submission record")
    _, record = record_item
    ids = _record_job_ids(record)
    for job in ids[:4]:
        _wait_job(job, cycles=6, interval=5.0)
    return 0


def _study_status(study_arg: str) -> int:
    study = _resolve_study(study_arg)
    status, reason = _campaign_status(study, source_sha=_git("rev-parse", "HEAD"))
    print(f"Study {_study_number(study.name)} {study.name}: {status}")
    print(reason)
    item = _latest_submission(study.name)
    if item:
        print(f"submission: {item[0]}")
        print(f"jobs: {' '.join(_record_job_ids(item[1])) or 'none'}")
    return 0


def _study_logs(study_arg: str) -> int:
    study = _resolve_study(study_arg)
    item = _latest_submission(study.name)
    if item is None:
        print(f"No submission logs for Study {study.name}")
        return 0
    path, record = item
    print(f"submission: {path}")
    print(f"log directory: {record.get('log_directory', path.parent / 'logs')}")
    log_directory = Path(record.get("log_directory", path.parent / "logs"))
    for log in sorted(log_directory.glob("*.out"))[-8:]:
        print(log)
    return 0


def _cleanup() -> int:
    parent = _paths().project_root.parent
    candidates = sorted(path for path in parent.iterdir() if re.search(r"jer|JointEvent", path.name, re.I))
    print("Cleanup dry-run; no files removed.")
    for path in candidates:
        if path.resolve() == _paths().project_root:
            continue
        print(f"  stale external candidate: {path}")
    worktrees = _git("worktree", "list", "--porcelain", check=False)
    print("  registered worktrees:")
    print(worktrees or "    canonical checkout only")
    return 0


def _main_new(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="jer", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("gates")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--compute-node", action="store_true")
    smoke = sub.add_parser("smoke")
    smoke_sub = smoke.add_subparsers(dest="smoke_command", required=True)
    gpu = smoke_sub.add_parser("gpu")
    gpu.add_argument("--watch", action="store_true")
    study = sub.add_parser("study")
    study_sub = study.add_subparsers(dest="study_command", required=True)
    for action in ("plan", "submit", "status", "logs"):
        command = study_sub.add_parser(action)
        command.add_argument("study")
        if action == "submit":
            command.add_argument("--watch", action="store_true")
    campaign = sub.add_parser("campaign")
    campaign_sub = campaign.add_subparsers(dest="campaign_command", required=True)
    campaign_status = campaign_sub.add_parser("status")
    campaign_status.add_argument("campaign", nargs="?", default="paper1")
    campaign_status.add_argument("--campaign-root", type=Path)
    campaign_launch = campaign_sub.add_parser("launch")
    campaign_launch.add_argument("campaign", nargs="?", default="paper1")
    campaign_launch.add_argument("--auto", action="store_true")
    campaign_launch.add_argument("--campaign-root", type=Path)
    campaign_watch = campaign_sub.add_parser("watch")
    campaign_watch.add_argument("campaign", nargs="?", default="paper1")
    campaign_watch.add_argument("--interval", type=float, default=15.0)
    campaign_watch.add_argument("--campaign-root", type=Path)
    campaign_resume = campaign_sub.add_parser("resume")
    campaign_resume.add_argument("campaign", nargs="?", default="paper1")
    campaign_resume.add_argument("--campaign-root", type=Path)
    campaign_review = campaign_sub.add_parser("review")
    campaign_review.add_argument("campaign", nargs="?", default="paper1")
    campaign_review.add_argument("--campaign-root", type=Path)
    campaign_approve = campaign_sub.add_parser("approve")
    campaign_approve.add_argument("campaign", nargs="?", default="paper1")
    campaign_approve.add_argument("gate")
    campaign_approve.add_argument("variant")
    campaign_approve.add_argument("--campaign-root", type=Path)
    campaign_continue = campaign_sub.add_parser("continue")
    campaign_continue.add_argument("campaign", nargs="?", default="paper1")
    campaign_continue.add_argument("--from-campaign", type=Path, required=True)
    campaign_continue.add_argument("--campaign-root", type=Path, required=True)
    campaign_preflight = campaign_sub.add_parser("preflight")
    campaign_preflight.add_argument("campaign", nargs="?", default="paper1")
    campaign_preflight.add_argument(
        "stage", choices=("invisible_backbone", "assignment_coupling", "assignment_conditioning")
    )
    campaign_preflight.add_argument("--campaign-root", type=Path, required=True)
    campaign_sub.add_parser("next")
    cleanup = sub.add_parser("cleanup")
    cleanup.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "status":
        _print_status()
        return 0
    if args.command == "gates":
        _print_gates()
        return 0
    if args.command == "doctor":
        failures = _doctor_local()
        if not failures:
            print("PASS: local JER doctor")
        else:
            for failure in failures:
                print(f"FAIL: {failure}", file=sys.stderr)
        if args.compute_node:
            _, state, directory = _run_compute_node_preflight(watch=True)
            print(f"compute-node preflight: {state}; logs={directory}")
            if state != "COMPLETED":
                failures.append(f"compute-node preflight state {state}")
        return 1 if failures else 0
    if args.command == "smoke":
        return _smoke_gpu(args.watch)
    if args.command == "study":
        if args.study_command == "plan":
            return _study_plan(args.study)
        if args.study_command == "submit":
            return _study_submit(args.study, args.watch)
        if args.study_command == "status":
            return _study_status(args.study)
        return _study_logs(args.study)
    if args.command == "campaign":
        if args.campaign_command == "status":
            if args.campaign != "paper1":
                raise RuntimeError(f"Unknown campaign {args.campaign!r}; only paper1 is maintained")
            from joint_event_reconstruction import campaign_controller
            campaign_controller.status(args.campaign_root)
            return 0
        if args.campaign_command == "launch":
            if args.campaign != "paper1":
                raise RuntimeError(f"Unknown campaign {args.campaign!r}; only paper1 is maintained")
            if not args.auto:
                raise RuntimeError("Paper-1 launch requires the explicit --auto flag")
            from joint_event_reconstruction import campaign_controller

            state = campaign_controller.launch(args.campaign_root, auto=args.auto)
            print(
                f"Paper-1 campaign armed: {state['campaign_id']}\n"
                f"  source SHA: {state['source_sha']}\n"
                f"  smoke: {state['smoke'].get('job_id')} {state['smoke'].get('status')}\n"
                f"  controller: {state['smoke'].get('controller_job', '-')}\n"
                "  continuation: Slurm dependency; safe to log out"
            )
            return 0
        if args.campaign_command == "watch":
            if args.campaign != "paper1":
                raise RuntimeError(f"Unknown campaign {args.campaign!r}; only paper1 is maintained")
            from joint_event_reconstruction import campaign_controller

            campaign_controller.watch(args.campaign_root, interval=args.interval)
            return 0
        if args.campaign_command == "resume":
            if args.campaign != "paper1":
                raise RuntimeError(f"Unknown campaign {args.campaign!r}; only paper1 is maintained")
            from joint_event_reconstruction import campaign_controller

            state = campaign_controller.resume(args.campaign_root)
            print(f"Paper-1 campaign resumed: {state['state']}")
            return 0
        if args.campaign_command == "review":
            if args.campaign != "paper1":
                raise RuntimeError(f"Unknown campaign {args.campaign!r}; only paper1 is maintained")
            from joint_event_reconstruction import campaign_controller

            campaign_controller.review(args.campaign_root)
            return 0
        if args.campaign_command == "approve":
            if args.campaign != "paper1":
                raise RuntimeError(f"Unknown campaign {args.campaign!r}; only paper1 is maintained")
            from joint_event_reconstruction import campaign_controller

            state = campaign_controller.approve(
                args.campaign_root, args.gate, args.variant
            )
            print(f"Paper-1 campaign approved {args.gate}={args.variant}: {state['state']}")
            return 0
        if args.campaign_command == "continue":
            if args.campaign != "paper1":
                raise RuntimeError(f"Unknown campaign {args.campaign!r}; only paper1 is maintained")
            from joint_event_reconstruction import campaign_controller

            state = campaign_controller.continue_campaign(args.from_campaign, args.campaign_root)
            print(
                f"Paper-1 campaign continuation created: {state['campaign_id']}\n"
                f"  source SHA: {state['source_sha']}\n"
                f"  parent: {state['parent_campaign']['campaign_id']}\n"
                f"  state: {state['state']}"
            )
            return 0
        if args.campaign_command == "preflight":
            if args.campaign != "paper1":
                raise RuntimeError(f"Unknown campaign {args.campaign!r}; only paper1 is maintained")
            from joint_event_reconstruction import campaign_controller

            state = campaign_controller.preflight(args.campaign_root, args.stage)
            release = state["stage_releases"][args.stage]
            print(
                f"Paper-1 {args.stage} preflight armed: job={release.get('preflight_job_id')} "
                f"controller={state['stages'][args.stage].get('controller_job')} state={state['state']}"
            )
            return 0
        _print_status()
        return 0
    if args.command == "cleanup":
        if not args.dry_run:
            raise RuntimeError("Only --dry-run cleanup is maintained")
        return _cleanup()
    raise RuntimeError(f"Unhandled jer command {args.command!r}")


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] in _LEGACY_COMMANDS:
        from joint_event_reconstruction.cli import run as legacy

        legacy.main()
        return
    try:
        exit_code = _main_new(argv)
    except (FileNotFoundError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise SystemExit(f"jer: {error}") from None
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
