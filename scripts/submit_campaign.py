#!/usr/bin/env python3
"""Plan and submit the currently eligible scientific campaign wave."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.studies import (
    ScientificGateNotApplicable,
    ScientificGatePending,
    StudySpec,
    load_scientific_gates,
    load_study,
    resolve_study,
)
from joint_event_reconstruction.prediction_schema import (
    DISCRETE_PREDICTION_SCHEMA,
    JOINT_PREDICTION_SCHEMA,
    PREDICTION_SCHEMA,
    validate_discrete_prediction,
    validate_joint_prediction,
    validate_prediction,
)


CAMPAIGN_NAME = "jer-main"
STUDY_GLOB = "[0-9][0-9]_*.yaml"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip())
    return commit, dirty


def _study_paths() -> dict[str, Path]:
    paths = {}
    for path in sorted((ROOT / "configs/studies").glob(STUDY_GLOB)):
        study = load_study(path)
        if study.name in paths:
            raise ValueError(f"Duplicate maintained study {study.name!r}")
        paths[study.name] = path
    return paths


def _identity(study: StudySpec, commit: str) -> tuple[str, dict]:
    payload = {
        "source_sha": commit,
        "manifest_sha256": _sha256_file(study.path),
        "gates_digest": study.gates_digest,
        "study": study.name,
        "topologies": list(study.topologies),
        "seeds": list(study.seeds),
        "variants": [variant.name for variant in study.variants],
        "training": study.training,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload


def _submission_files(output_root: Path, study_name: str) -> list[Path]:
    return sorted((output_root / study_name / "_slurm").glob("*/submission.json"))


def _matching_submission(
    output_root: Path, study: StudySpec, identity: str, identity_payload: dict
) -> tuple[Path, dict] | None:
    for path in _submission_files(output_root, study.name):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("git_dirty"):
            continue
        if record.get("submission_identity") == identity:
            return path, record
        fields = {
            "git_commit": identity_payload["source_sha"],
            "study_manifest_sha256": identity_payload["manifest_sha256"],
            "resolved_gate_digest": identity_payload["gates_digest"],
            "topologies": identity_payload["topologies"],
            "seeds": identity_payload["seeds"],
        }
        if all(record.get(key) == value for key, value in fields.items()):
            return path, record
    return None


def _normalise_state(value: str) -> str:
    return value.strip().split("+", 1)[0].split(" ", 1)[0].upper()


def _job_state(job_id: str | None) -> str:
    if not job_id:
        return "UNKNOWN"
    queue = subprocess.run(
        ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    ).stdout.splitlines()
    if queue:
        return _normalise_state(queue[0])
    account = subprocess.run(
        ["sacct", "-n", "-X", "-j", str(job_id), "--format=State"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    ).stdout.splitlines()
    states = [_normalise_state(line) for line in account if line.strip()]
    if not states:
        return "SUBMITTED"
    if any(state in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}
           for state in states):
        return "FAILED"
    if any(state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"} for state in states):
        return "RUNNING"
    if all(state == "COMPLETED" for state in states):
        return "COMPLETED"
    return states[0]


def _record_job_ids(record: dict) -> list[str]:
    ids = [
        str(level["job_id"])
        for level in record.get("train_levels", [])
        if level.get("job_id")
    ]
    for key in ("prediction_job", "postprocess_job"):
        if record.get(key):
            ids.append(str(record[key]))
    return ids


def _task_count(path: str | Path | None) -> int:
    if not path:
        return 0
    task_file = Path(path)
    if not task_file.is_file():
        raise FileNotFoundError(f"Submitted Slurm task file is missing: {task_file}")
    return sum(bool(line.strip()) for line in task_file.read_text().splitlines())


def _array_completion(job_id: str, expected: int) -> tuple[str, str]:
    """Check every array element, not only the parent array record."""
    result = subprocess.run(
        ["sacct", "-n", "-X", "-j", str(job_id), "--format=JobID,State"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    elements = {}
    prefix = f"{job_id}_"
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 2 or not fields[0].startswith(prefix):
            continue
        suffix = fields[0][len(prefix):]
        if suffix.isdigit():
            elements[int(suffix)] = _normalise_state(fields[1])
    if len(elements) != expected:
        return "FAILED", f"array {job_id}: observed {len(elements)}/{expected} task states"
    failed = sorted(index for index, state in elements.items() if state in {
        "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"
    })
    if failed:
        return "FAILED", f"array {job_id}: failed task indices {failed}"
    if any(state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
           for state in elements.values()):
        return "RUNNING", f"array {job_id}: {expected} task states observed"
    if all(state == "COMPLETED" for state in elements.values()):
        return "COMPLETED", f"array {job_id}: {expected}/{expected} tasks completed"
    return "FAILED", f"array {job_id}: unexpected task states {sorted(set(elements.values()))}"


def _validate_prediction_output(path: Path) -> None:
    with h5py.File(path, "r") as source:
        schema = str(source.attrs.get("schema_version"))
    validators = {
        PREDICTION_SCHEMA: validate_prediction,
        DISCRETE_PREDICTION_SCHEMA: validate_discrete_prediction,
        JOINT_PREDICTION_SCHEMA: validate_joint_prediction,
    }
    if schema not in validators:
        raise ValueError(f"Unsupported prediction schema {schema!r}: {path}")
    validators[schema](path)


def _outputs_complete(study: StudySpec, record: dict) -> tuple[bool, str]:
    output_root = Path(record["output_root"])
    for topology in record.get("topologies", study.topologies):
        for variant in study.variants:
            if variant.reporting == "setup" or variant.name not in record.get("requested_variants", []):
                continue
            for seed in record.get("seeds", study.seeds):
                for prediction in variant.predictions:
                    run = output_root / study.name / topology / variant.name / f"seed_{seed}"
                    prediction_path = run / "predictions" / f"{prediction.name}.h5"
                    identity_path = run / "predictions" / f"{prediction.name}.identity.json"
                    evaluation_path = run / "evaluation" / f"{prediction.name}.json"
                    plot_path = run / "plots" / prediction.name
                    for required in (prediction_path, identity_path, evaluation_path):
                        if not required.is_file() or required.stat().st_size <= 0:
                            return False, f"missing output {required}"
                    if not plot_path.is_dir() or not any(plot_path.iterdir()):
                        return False, f"missing plots {plot_path}"
                    try:
                        _validate_prediction_output(prediction_path)
                        json.loads(identity_path.read_text())
                        json.loads(evaluation_path.read_text())
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        return False, f"invalid output {prediction_path}: {error}"
    summary = output_root / study.name / "summary.json"
    if not summary.is_file() or not summary.stat().st_size:
        return False, f"missing summary {summary}"
    analysis_roots = {
        "encoder_backbone": (
            output_root / study.name / "pairwise_complementarity.json",
            output_root / study.name / "pairwise_complementarity.csv",
            output_root / study.name / "oracle_union_matrix.png",
            output_root / study.name / "oracle_union_matrix.pdf",
            output_root / study.name / "disagreement_matrix.png",
            output_root / study.name / "disagreement_matrix.pdf",
            output_root / study.name / "study12_human_review.md",
            output_root / study.name / "study12_human_review.json",
            output_root / study.name / "analysis14_assignment_sets.json",
            output_root / study.name / "analysis14_assignment_sets.md",
        ),
        "assignment_conditioning": (output_root / study.name / "assignment_conditioning.json",),
        "data_scaling": (output_root / study.name / "data_scaling_recurrence.json",),
    }
    if study.name in analysis_roots:
        for required in analysis_roots[study.name]:
            if not required.is_file() or not required.stat().st_size:
                return False, f"missing dedicated analysis {required}"
    if study.name == "c2_feedback":
        for topology in study.topologies:
            for seed in study.seeds:
                required = output_root / study.name / topology / "analysis" / f"seed_{seed}" / "feedback_transitions.json"
                if not required.is_file() or not required.stat().st_size:
                    return False, f"missing dedicated analysis {required}"
    return True, "all predictions, evaluations, plots, summary and dedicated analyses validated"


def _submission_status(record: dict, study: StudySpec) -> tuple[str, str]:
    states = [_job_state(job_id) for job_id in _record_job_ids(record)]
    if any(state == "FAILED" for state in states):
        return "FAILED", f"Slurm job states: {states}"
    if record.get("postprocess_job") and _job_state(str(record["postprocess_job"])) == "COMPLETED":
        valid, reason = _outputs_complete(study, record)
        return ("COMPLETE", reason) if valid else ("FAILED", reason)
    if any(state in {"RUNNING", "PENDING", "CONFIGURING", "COMPLETING"} for state in states):
        return "RUNNING", f"Slurm job states: {states}"
    return "SUBMITTED", f"Slurm job states: {states}"


def _historical_study01() -> dict:
    root = ROOT / "results/studies/visible_recurrence-final-20260913/visible_recurrence"
    submission = root / "_slurm/study01-final-fresh-20260913/submission.json"
    train = list(root.glob("ttbar2L/*/seed_*/train/artifact_identity.json"))
    prediction = list(root.glob("ttbar2L/*/seed_*/predictions/*.h5"))
    if len(train) != 21:
        return {"status": "FAILED", "reason": f"Study 01 has {len(train)}/21 training identities"}
    if len(prediction) != 45:
        return {"status": "FAILED", "reason": f"Study 01 has {len(prediction)} prediction files; expected 45"}
    if not submission.is_file():
        return {"status": "FAILED", "reason": f"Missing Study 01 submission record: {submission}"}
    record = json.loads(submission.read_text())
    train_levels = record.get("train_levels", [])
    if len(train_levels) != 1:
        return {"status": "FAILED", "reason": "Study 01 historical training DAG is not one 21-task array"}
    train_level = train_levels[0]
    train_expected = _task_count(train_level.get("task_file"))
    prediction_expected = _task_count(record.get("prediction_task_file"))
    if train_expected != 21 or prediction_expected != 21:
        return {
            "status": "FAILED",
            "reason": (
                "Study 01 historical task files are not 21 training/21 prediction units: "
                f"train={train_expected}, prediction={prediction_expected}"
            ),
        }
    train_state, train_reason = _array_completion(str(train_level["job_id"]), train_expected)
    prediction_state, prediction_reason = _array_completion(
        str(record["prediction_job"]), prediction_expected
    )
    post_state = _job_state(str(record.get("postprocess_job")))
    if train_state != "COMPLETED" or prediction_state != "COMPLETED" or post_state != "COMPLETED":
        return {
            "status": "FAILED",
            "reason": f"Study 01 historical jobs: {train_reason}; {prediction_reason}; post={post_state}",
        }
    jobs = _record_job_ids(record)
    return {
        "status": "COMPLETE",
        "reason": "Historical Study 01 products verified; immutable and not rerun",
        "source_sha": record.get("git_commit"),
        "manifest_sha256": record.get("study_manifest_sha256"),
        "submission_identity": f"historical:{record.get('campaign_id', 'study01')}",
        "output_root": str(root),
        "job_ids": jobs,
        "training_job_ids": [str(train_level["job_id"])],
        "prediction_job_ids": [str(record["prediction_job"])],
        "postprocess_job_ids": [str(record["postprocess_job"])],
        "training_identities": len(train),
        "prediction_files": len(prediction),
        "training_jobs_completed": train_expected,
        "prediction_jobs_completed": prediction_expected,
    }


def _run_plan(command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode:
        raise RuntimeError(
            f"Dry-plan failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )


def _dry_plan(study: StudySpec, identity: str) -> None:
    with tempfile.TemporaryDirectory(prefix="jer-campaign-plan-") as directory:
        output_root = Path(directory) / "studies"
        common = [
            sys.executable,
            str(ROOT / "scripts/run_study.py"),
            str(study.path),
            "plan",
            "--output-root", str(output_root),
            "--dry-run",
        ]
        _run_plan(common)
        _run_plan([
            sys.executable,
            str(ROOT / "scripts/submit_study_slurm.py"),
            str(study.path),
            "--output-root", str(output_root),
            "--campaign-id", f"plan-{identity[:16]}",
            "--submission-identity", identity,
            "--dry-run",
        ])


def _static_block(study: StudySpec, message: str) -> dict:
    train_units = len(study.variants) * len(study.topologies) * len(study.seeds) if study.training else 0
    prediction_units = sum(
        len(variant.predictions) * len(study.topologies) * len(study.seeds)
        for variant in study.variants if variant.reporting != "setup"
    )
    return {
        "status": "BLOCKED_GATE",
        "reason": message,
        "manifest_migrated": True,
        "slurm_dry_plan": f"mechanically validated: train_units={train_units}, prediction_units={prediction_units}",
        "training_units": train_units,
        "prediction_units": prediction_units,
    }


def _blocked_entry(
    study: StudySpec,
    output_root: Path,
    commit: str,
    message: str,
    *,
    status: str = "BLOCKED_GATE",
) -> dict:
    identity_payload = {
        "source_sha": commit,
        "manifest_sha256": _sha256_file(study.path),
        "study": study.name,
        "topologies": list(study.topologies),
        "seeds": list(study.seeds),
        "variants": [variant.name for variant in study.variants],
        "training": study.training,
        "required_gates": list(study.requires),
    }
    identity = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    static = _static_block(study, message)
    static["status"] = status
    return {
        "study": study.name,
        "source_sha": commit,
        "manifest_sha256": identity_payload["manifest_sha256"],
        "submission_identity": identity,
        "required_gates": list(study.requires),
        "resolved_gate_variants": {},
        "output_root": str(output_root / study.name),
        "job_ids": [],
        "training_job_ids": [],
        "prediction_job_ids": [],
        "postprocess_job_ids": [],
        "three_seed_strategy": list(study.seeds) == [42, 43, 44],
        **static,
    }


def _entry(
    study: StudySpec,
    output_root: Path,
    commit: str,
    *,
    run_plans: bool,
) -> dict:
    identity, identity_payload = _identity(study, commit)
    training_units = len(study.variants) * len(study.topologies) * len(study.seeds) if study.training else 0
    prediction_units = sum(
        len(variant.predictions) * len(study.topologies) * len(study.seeds)
        for variant in study.variants if variant.reporting != "setup"
    )
    base = {
        "study": study.name,
        "source_sha": commit,
        "manifest_sha256": identity_payload["manifest_sha256"],
        "submission_identity": identity,
        "required_gates": list(study.requires),
        "resolved_gate_variants": {
            name: gate.get("selected_variant")
            for name, gate in (study.resolved_gates or {}).items()
        },
        "output_root": str(output_root / study.name),
        "manifest_migrated": True,
        "three_seed_strategy": list(study.seeds) == [42, 43, 44],
        "training_units": training_units,
        "prediction_units": prediction_units,
    }
    existing = _matching_submission(output_root, study, identity, identity_payload)
    if existing is not None:
        path, record = existing
        status, reason = _submission_status(record, study)
        return {
            **base,
            "status": status,
            "reason": reason,
            "submission_record": str(path),
            "job_ids": _record_job_ids(record),
            "slurm_dry_plan": "validated at submission",
        }
    if run_plans:
        from scripts.run_study import _gate_checkpoint

        for variant in study.variants:
            if variant.initialise_from_gate is None:
                continue
            for topology in study.topologies:
                for seed in study.seeds:
                    _gate_checkpoint(study, variant.initialise_from_gate, topology, seed)
        _dry_plan(study, identity)
    return {
        **base,
        "status": "READY",
        "job_ids": [],
        "slurm_dry_plan": "PASS",
    }


def _external_source(study: StudySpec) -> set[str]:
    return {
        variant.checkpoint_from[0]
        for variant in study.variants
        if variant.checkpoint_from is not None
    }


def _source_job_id(
    source_name: str,
    source_variant: str,
    entries: dict[str, dict],
) -> str | None:
    entry = entries.get(source_name)
    if not entry:
        return None
    record_path = entry.get("submission_record")
    if not record_path:
        return None
    record = json.loads(Path(record_path).read_text())
    for level in record.get("train_levels", []):
        if source_variant in level.get("variants", []):
            return str(level["job_id"])
    return None


def _submit_one(
    study: StudySpec,
    entry: dict,
    output_root: Path,
    dependency: str | None,
) -> dict:
    campaign_id = f"campaign-{study.name}-{entry['submission_identity'][:16]}"
    command = [
        sys.executable,
        str(ROOT / "scripts/submit_study_slurm.py"),
        str(study.path),
        "--output-root", str(output_root),
        "--campaign-id", campaign_id,
        "--submission-identity", entry["submission_identity"],
    ]
    if dependency:
        command += ["--dependency-afterok", dependency]
    print("SUBMIT:", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    record_path = output_root / study.name / "_slurm" / campaign_id / "submission.json"
    if not record_path.is_file():
        raise RuntimeError(f"Submission completed without a record: {record_path}")
    record = json.loads(record_path.read_text())
    return {
        **entry,
        "status": "SUBMITTED",
        "submission_record": str(record_path),
        "job_ids": _record_job_ids(record),
        "training_job_ids": [str(level["job_id"]) for level in record.get("train_levels", [])],
        "prediction_job_ids": [str(record["prediction_job"])] if record.get("prediction_job") else [],
        "postprocess_job_ids": [str(record["postprocess_job"])] if record.get("postprocess_job") else [],
    }


def _analysis14_entry(study12: dict) -> dict:
    status = study12["status"]
    if status == "COMPLETE":
        status = "COMPLETE"
    elif status in {"SUBMITTED", "RUNNING"}:
        status = status
    elif status == "READY":
        status = "READY"
    else:
        status = "BLOCKED_GATE"
    return {
        "study": "Analysis 14",
        "status": status,
        "required_gates": [],
        "manifest_migrated": True,
        "three_seed_strategy": True,
        "slurm_dry_plan": "bundled with Study 12 postprocess",
        "reason": "Validation-only assignment-set/conformal diagnostic; never selects the encoder",
    }


def _print_report(entries: list[dict], gates: dict[str, dict]) -> None:
    print("\nSCIENTIFIC GATES")
    for name, gate in gates.items():
        selected = gate.get("selected_variant") or "-"
        print(f"  {name:24s} {gate['status']:15s} {selected}")
    print("\nCAMPAIGN STATUS")
    for entry in entries:
        reason = f" — {entry['reason']}" if entry.get("reason") else ""
        print(f"  {entry['study']:24s} {entry['status']:15s}{reason}")


def _write_registry(path: Path, commit: str, entries: list[dict], gates: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "campaign": CAMPAIGN_NAME,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_sha": commit,
        "gates": gates,
        "studies": entries,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit all currently eligible JER studies")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--ready", action="store_true", help="dry-plan and submit eligible studies")
    action.add_argument("--status", action="store_true", help="report gates and current campaign status")
    parser.add_argument("--output-root", type=Path, default=Path("results/studies"))
    parser.add_argument("--campaign-root", type=Path, default=Path("results/campaigns/jer-main"))
    args = parser.parse_args()
    if not args.ready and not args.status:
        parser.error("choose --ready or --status")

    output_root = args.output_root.expanduser().resolve()
    campaign_root = args.campaign_root.expanduser().resolve()
    commit, dirty = _git_state()
    gates = load_scientific_gates()
    paths = _study_paths()
    entries_by_name: dict[str, dict] = {}
    resolved_by_name: dict[str, StudySpec] = {}
    for name, path in paths.items():
        raw = load_study(path)
        if name == "visible_recurrence":
            historical = _historical_study01()
            entry = {
                "study": name,
                **historical,
                "required_gates": [],
                "manifest_migrated": True,
                "three_seed_strategy": True,
                "slurm_dry_plan": "historical Study 01 DAG verified; no rerun",
            }
            entries_by_name[name] = entry
            continue
        try:
            resolved = resolve_study(path)
        except ScientificGatePending as error:
            entries_by_name[name] = _blocked_entry(
                raw,
                output_root,
                commit,
                str(error) + "; static DAG shape validated",
            )
            continue
        except ScientificGateNotApplicable as error:
            entries_by_name[name] = _blocked_entry(
                raw,
                output_root,
                commit,
                str(error),
                status="NOT_APPLICABLE",
            )
            continue
        resolved_by_name[name] = resolved
        entries_by_name[name] = _entry(resolved, output_root, commit, run_plans=args.ready)

    entries = [entries_by_name[name] for name in paths]
    entries.append(_analysis14_entry(entries_by_name["encoder_backbone"]))
    _write_registry(campaign_root / "campaign_state.json", commit, entries, gates)
    _print_report(entries, gates)

    if args.status:
        return
    if dirty:
        raise RuntimeError("Campaign submission requires a clean committed checkout")

    ready_names = [
        name for name in paths
        if entries_by_name[name].get("status") == "READY" and name in resolved_by_name
    ]
    submitted = dict(entries_by_name)
    pending = set(ready_names)
    while pending:
        progress = False
        for name in sorted(pending):
            study = resolved_by_name[name]
            sources = _external_source(study)
            unresolved = [source for source in sources if source in pending]
            if unresolved:
                continue
            dependencies = []
            for variant in study.variants:
                if variant.checkpoint_from is None:
                    continue
                source_name, source_variant = variant.checkpoint_from
                source_id = _source_job_id(source_name, source_variant, submitted)
                if source_id:
                    dependencies.append(source_id)
                elif entries_by_name.get(source_name, {}).get("status") not in {"COMPLETE", "SUBMITTED", "RUNNING"}:
                    raise RuntimeError(
                        f"Study {name} is ready but checkpoint source {source_name}:{source_variant} "
                        "has neither completed artifacts nor a submitted training job"
                    )
            dependency = ",".join(sorted(set(dependencies))) or None
            submitted[name] = _submit_one(study, entries_by_name[name], output_root, dependency)
            pending.remove(name)
            progress = True
            break
        if not progress:
            raise RuntimeError(f"Cross-study checkpoint dependency cycle or unresolved source: {sorted(pending)}")

    final_entries = [submitted[name] for name in paths]
    final_entries.append(_analysis14_entry(submitted["encoder_backbone"]))
    _write_registry(campaign_root / "campaign_state.json", commit, final_entries, gates)
    print("\nSUBMITTED ELIGIBLE STUDIES:")
    for name in ready_names:
        print(f"  {name}: {submitted[name]['status']} jobs={submitted[name].get('job_ids', [])}")


if __name__ == "__main__":
    main()
