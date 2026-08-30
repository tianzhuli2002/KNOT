#!/usr/bin/env python3
"""Run one reproducible JER study manifest through the maintained CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.configuration import validate_experiment
from joint_event_reconstruction.data.subsets import (
    SUBSET_RANKING,
    deterministic_subset_positions,
    event_id_digest,
)
from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions
from joint_event_reconstruction.plotting import plot_comparison, plot_prediction
from joint_event_reconstruction.prediction_schema import (
    DISCRETE_PREDICTION_SCHEMA,
    JOINT_PREDICTION_SCHEMA,
    PREDICTION_SCHEMA,
    validate_discrete_prediction,
    validate_joint_prediction,
    validate_prediction,
)
from joint_event_reconstruction.public_config import resolve_public_experiment
from joint_event_reconstruction.studies import PredictionSpec, StudySpec, VariantSpec, load_study
from joint_event_reconstruction.study_metrics import (
    assert_same_events,
    common_non_null_truth_assignment,
)


SUMMARY_FIELDS = (
    "study", "topology", "variant", "variant_label", "prediction",
    "prediction_label", "reporting", "seed", "events", "assignment_support",
    "assignment_accuracy", "assignment_accuracy_standard_error",
    "common_non_null_events", "common_non_null_assignment_accuracy",
    "common_non_null_assignment_standard_error", "compatibility_assignment_only_top1",
    "compatibility_combined_top1", "energy_score", "sample_equivalent_nfe_per_event",
    "prediction_seed", "prediction_file",
)
ARTIFACT_IDENTITY_SCHEMA = "joint_event_reconstruction_artifact_identity_v1"


def _run_dir(root: Path, study: StudySpec, topology: str, variant: str, seed: int) -> Path:
    return root / study.name / topology / variant / f"seed_{seed}"


def _checkpoint(root: Path, study: StudySpec, topology: str, variant: str, seed: int) -> Path:
    return _run_dir(root, study, topology, variant, seed) / "train/checkpoints/best-loss.ckpt"


def _prediction_path(root, study, topology, variant, seed, prediction) -> Path:
    return _run_dir(root, study, topology, variant, seed) / "predictions" / f"{prediction}.h5"


def _evaluation_path(root, study, topology, variant, seed, prediction) -> Path:
    return _run_dir(root, study, topology, variant, seed) / "evaluation" / f"{prediction}.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_digest(identity: dict) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _file_identity(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise RuntimeError(f"Artifact is absent or empty: {path}")
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(path),
    }


def _repo_path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


@lru_cache(maxsize=None)
def _data_subset_identity_cached(
    data_config_path: str,
    subset_seed: int,
    train_count: int | None,
    validation_count: int | None,
    test_count: int | None,
) -> dict:
    data = OmegaConf.load(data_config_path)
    source_path = _repo_path(data.source)
    split_path = _repo_path(data.split)
    requested = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    split_identity = {}
    with np.load(split_path) as split_file, h5py.File(source_path, "r") as source:
        for split, count in requested.items():
            rows = np.asarray(split_file[split], dtype=np.int64)
            event_ids = np.asarray(source["EVENT_ID"][rows], dtype=np.int64)
            positions = deterministic_subset_positions(event_ids, count, subset_seed)
            selected = event_ids[positions]
            split_identity[split] = {
                "available_events": int(len(event_ids)),
                "selected_events": int(len(selected)),
                "ordered_event_id_sha256": event_id_digest(selected),
            }
    source_stat = source_path.stat()
    return {
        "ranking": SUBSET_RANKING,
        "subset_seed": int(subset_seed),
        "source": str(source_path),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "split_indices": str(split_path),
        "split_indices_sha256": _sha256_file(split_path),
        "splits": split_identity,
    }


def _data_subset_identity(resolved) -> dict:
    limits = resolved.data.max_events
    return _data_subset_identity_cached(
        str(Path(resolved.data_config).resolve()),
        int(resolved.data.subset_seed),
        None if limits.train is None else int(limits.train),
        None if limits.val is None else int(limits.val),
        None if limits.test is None else int(limits.test),
    )


def _require_clean_git() -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "Scientific study execution requires a clean committed checkout; "
            f"dirty entries={status.splitlines()[:20]}"
        )
    commit = _git_commit()
    if commit is None:
        raise RuntimeError("Scientific study execution requires a Git commit")
    return commit


def _training_identity(
    study,
    topology,
    variant,
    seed,
    experiment,
    resolved,
    initial,
    git_commit,
) -> dict:
    return {
        "schema": ARTIFACT_IDENTITY_SCHEMA,
        "kind": "training",
        "git_commit": git_commit,
        "study": study.name,
        "study_manifest": str(study.path),
        "study_manifest_sha256": _sha256_file(study.path),
        "topology": topology,
        "variant": variant.name,
        "variant_label": variant.label,
        "variant_patch": variant.patch,
        "model_seed": int(seed),
        "data_subset": _data_subset_identity(resolved),
        "base_experiment": str(experiment),
        "base_experiment_sha256": _sha256_file(experiment),
        "resolved_config": OmegaConf.to_container(resolved, resolve=True),
        "initial_checkpoint": None if initial is None else _file_identity(initial),
        "initial_components": list(variant.initialise_components),
    }


def _prediction_identity(
    study,
    topology,
    variant,
    prediction,
    seed,
    resolved,
    checkpoint,
    recurrent_steps,
    git_commit,
) -> dict:
    return {
        "schema": ARTIFACT_IDENTITY_SCHEMA,
        "kind": "prediction",
        "git_commit": git_commit,
        "study": study.name,
        "study_manifest": str(study.path),
        "study_manifest_sha256": _sha256_file(study.path),
        "topology": topology,
        "variant": variant.name,
        "prediction": prediction.name,
        "model_seed": int(seed),
        "data_subset": _data_subset_identity(resolved),
        "split": study.split,
        "checkpoint": _file_identity(checkpoint),
        "resolved_config": OmegaConf.to_container(resolved, resolve=True),
        "prediction_seed": int(resolved.inference.seed),
        "recurrent_steps": recurrent_steps,
    }


def _write_artifact_record(path: Path, artifact: Path, identity: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "identity": identity,
                "identity_sha256": _identity_digest(identity),
                "artifact": _file_identity(artifact),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _verify_artifact_record(
    record_path: Path,
    artifact: Path,
    expected_identity: dict,
    label: str,
) -> None:
    if not record_path.is_file():
        raise RuntimeError(
            f"{label} exists without artifact identity: {artifact}. "
            "Use --force to regenerate it explicitly."
        )
    record = json.loads(record_path.read_text())
    expected_digest = _identity_digest(expected_identity)
    if (
        record.get("identity") != expected_identity
        or record.get("identity_sha256") != expected_digest
    ):
        raise RuntimeError(
            f"Stale {label} identity for {artifact}: expected={expected_digest}, "
            f"found={record.get('identity_sha256')}. Use --force to regenerate."
        )
    observed = _file_identity(artifact)
    if record.get("artifact") != observed:
        raise RuntimeError(
            f"{label} checksum/size changed after completion: {artifact}. "
            "Use --force to regenerate."
        )


def _validate_prediction_artifact(path: Path) -> None:
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


def _invalidate_aggregate_outputs(root: Path, study, topology: str, seed: int) -> None:
    study_root = root / study.name
    comparison = study_root / topology / "comparisons" / f"seed_{seed}"
    if comparison.exists():
        shutil.rmtree(comparison)
    for name in ("summary.csv", "summary.json", "summary_aggregate.json"):
        path = study_root / name
        if path.exists():
            path.unlink()


def _depends_on(study: StudySpec, variant: VariantSpec, ancestor: str) -> bool:
    parent = variant.initialise_from
    while parent is not None:
        if parent == ancestor:
            return True
        parent = study.variant_map[parent].initialise_from
    return False


def _invalidate_after_training(
    root: Path,
    study,
    topology: str,
    variant: str,
    seed: int,
) -> None:
    affected = [
        candidate
        for candidate in study.variants
        if candidate.name == variant or _depends_on(study, candidate, variant)
    ]
    for candidate in affected:
        run_dir = _run_dir(root, study, topology, candidate.name, seed)
        for name in ("train", "predictions", "evaluation", "plots"):
            path = run_dir / name
            if path.exists():
                shutil.rmtree(path)
    _invalidate_aggregate_outputs(root, study, topology, seed)


def _invalidate_after_prediction(
    root: Path,
    study,
    topology: str,
    run_dir: Path,
    seed: int,
    prediction: str,
) -> None:
    for path in (
        run_dir / "evaluation" / f"{prediction}.json",
        run_dir / "predictions" / f"{prediction}.identity.json",
    ):
        if path.exists():
            path.unlink()
    plot_dir = run_dir / "plots" / prediction
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    _invalidate_aggregate_outputs(root, study, topology, seed)


def _selected(value: str, allowed: tuple, label: str) -> tuple:
    if value == "all":
        return allowed
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; allowed={list(allowed)}")
    return selected


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _public_run_config(
    study: StudySpec,
    topology: str,
    variant: VariantSpec,
    seed: int,
    output_root: Path,
    global_overrides: tuple[str, ...],
    prediction: PredictionSpec | None = None,
):
    experiment = study.experiment_path(ROOT, topology)
    public = OmegaConf.load(experiment)
    public = OmegaConf.merge(public, OmegaConf.create(variant.patch))
    if global_overrides:
        public = OmegaConf.merge(public, OmegaConf.from_dotlist(list(global_overrides)))
    public = OmegaConf.merge(
        public,
        {
            "experiment": {"seed": seed},
            "training": {
                "output": str(_run_dir(output_root, study, topology, variant.name, seed) / "train")
            },
        },
    )
    if prediction is not None:
        public = OmegaConf.merge(public, OmegaConf.create(prediction.patch))
    resolved = resolve_public_experiment(public, source_path=experiment)
    validate_experiment(resolved)
    return experiment, public, resolved


def _write_yaml(path: Path, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(config, resolve=True))


def _execute(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _train_one(study, topology, variant, seed, *, root, global_overrides, device, force, dry_run):
    run_dir = _run_dir(root, study, topology, variant.name, seed)
    checkpoint = _checkpoint(root, study, topology, variant.name, seed)
    experiment, public, resolved = _public_run_config(
        study, topology, variant, seed, root, global_overrides
    )
    initial = None
    if variant.initialise_from is not None:
        initial = _checkpoint(root, study, topology, variant.initialise_from, seed)
        if not initial.is_file() and not dry_run:
            raise FileNotFoundError(
                f"Variant {variant.name!r} requires checkpoint {initial}"
            )

    identity = None
    identity_path = run_dir / "train/artifact_identity.json"
    if not dry_run:
        git_commit = _require_clean_git()
        identity = _training_identity(
            study,
            topology,
            variant,
            seed,
            experiment,
            resolved,
            initial,
            git_commit,
        )
        if checkpoint.is_file() and not force:
            _verify_artifact_record(identity_path, checkpoint, identity, "checkpoint")
            print(f"SKIP train (identity matched): {checkpoint}")
            return
        train_dir = run_dir / "train"
        if train_dir.exists() and not force:
            raise RuntimeError(
                f"Incomplete training directory exists without a reusable checkpoint: "
                f"{train_dir}. Use --force to regenerate."
            )
        if force:
            _invalidate_after_training(
                root, study, topology, variant.name, seed
            )

    public_path = run_dir / "experiment.yaml"
    if not dry_run:
        _write_yaml(public_path, public)
        _write_yaml(run_dir / "resolved.yaml", resolved)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "study_run.json").write_text(
            json.dumps(
                {
                    "study": study.name,
                    "study_manifest": str(study.path),
                    "topology": topology,
                    "variant": variant.name,
                    "variant_label": variant.label,
                    "reporting": variant.reporting,
                    "seed": seed,
                    "base_experiment": str(experiment),
                    "variant_patch": variant.patch,
                    "initial_checkpoint": None if initial is None else str(initial),
                    "initial_components": list(variant.initialise_components),
                    "model_seed": int(seed),
                    "data_subset_seed": int(resolved.data.subset_seed),
                    "data_subset": identity["data_subset"],
                    "prediction_seed": int(resolved.inference.seed),
                    "git_commit": identity["git_commit"],
                    "training_identity_sha256": _identity_digest(identity),
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )

    command = [
        sys.executable, "-m", "joint_event_reconstruction.cli.run", "train",
        "--experiment", str(public_path), "--device", device,
    ]
    if initial is not None:
        command += ["--initial-checkpoint", str(initial), "--initial-components", *variant.initialise_components]
    _execute(command, dry_run=dry_run)
    if not dry_run:
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
            raise RuntimeError(f"Training completed without a non-empty checkpoint: {checkpoint}")
        _write_artifact_record(identity_path, checkpoint, identity)


def _predict_one(study, topology, variant, prediction, seed, *, root, global_overrides, device, force, dry_run):
    run_dir = _run_dir(root, study, topology, variant.name, seed)
    checkpoint = _checkpoint(root, study, topology, variant.name, seed)
    if not checkpoint.is_file() and not dry_run:
        raise FileNotFoundError(f"Prediction requires checkpoint {checkpoint}")
    target = _prediction_path(root, study, topology, variant.name, seed, prediction.name)
    _, public, resolved = _public_run_config(
        study, topology, variant, seed, root, global_overrides, prediction
    )
    recurrent_steps = prediction.recurrent_steps
    identity = None
    identity_path = target.with_suffix(".identity.json")
    if not dry_run:
        git_commit = _require_clean_git()
        training_identity_path = run_dir / "train/artifact_identity.json"
        training_record = json.loads(training_identity_path.read_text()) if training_identity_path.is_file() else None
        if training_record is None:
            raise RuntimeError(
                f"Prediction checkpoint has no completed training identity: {checkpoint}"
            )
        if training_record.get("artifact") != _file_identity(checkpoint):
            raise RuntimeError(f"Prediction checkpoint changed after training: {checkpoint}")
        identity = _prediction_identity(
            study,
            topology,
            variant,
            prediction,
            seed,
            resolved,
            checkpoint,
            recurrent_steps,
            git_commit,
        )
        if target.is_file() and not force:
            _validate_prediction_artifact(target)
            _verify_artifact_record(identity_path, target, identity, "prediction")
            print(f"SKIP predict (identity matched): {target}")
            return
        if force:
            _invalidate_after_prediction(
                root, study, topology, run_dir, seed, prediction.name
            )

    config_path = (
        run_dir / "predictions" / f"{prediction.name}.yaml"
    )
    if not dry_run:
        _write_yaml(config_path, public)
        _write_yaml(config_path.with_suffix(".resolved.yaml"), resolved)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
    command = [
        sys.executable, "-m", "joint_event_reconstruction.cli.run", "predict",
        "--experiment", str(config_path), "--checkpoint", str(checkpoint),
        "--output", str(target), "--split", study.split, "--device", device,
    ]
    if recurrent_steps is not None:
        command += ["--recurrent-steps", str(recurrent_steps)]
    _execute(command, dry_run=dry_run)
    if not dry_run:
        _validate_prediction_artifact(target)
        with h5py.File(target, "r+") as output:
            output.attrs["study_name"] = study.name
            output.attrs["study_variant"] = variant.name
            output.attrs["study_prediction"] = prediction.name
            output.attrs["study_seed"] = seed
            output.attrs["data_subset_seed"] = int(resolved.data.subset_seed)
            output.attrs["artifact_identity_sha256"] = _identity_digest(identity)
        _write_artifact_record(identity_path, target, identity)


def _evaluate_one(study, topology, variant, prediction, seed, *, root, force, dry_run):
    source = _prediction_path(root, study, topology, variant.name, seed, prediction.name)
    target = _evaluation_path(root, study, topology, variant.name, seed, prediction.name)
    if dry_run:
        print(f"evaluate {source} -> {target}")
        return
    del force
    _validate_prediction_artifact(source)
    metrics = evaluate_any_predictions(source)
    metrics["common_non_null_truth_assignment"] = common_non_null_truth_assignment(source)
    with h5py.File(source, "r") as handle:
        prediction_seed = int(handle.attrs.get("prediction_seed", -1))
        data_subset_seed = int(handle.attrs.get("data_subset_seed", -1))
    metrics["study"] = {
        "name": study.name,
        "topology": topology,
        "variant": variant.name,
        "variant_label": variant.label,
        "prediction": prediction.name,
        "prediction_label": prediction.label,
        "reporting": variant.reporting,
        "model_seed": seed,
        "data_subset_seed": data_subset_seed,
        "prediction_seed": prediction_seed,
        "selection_split": study.split,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _plot_one(study, topology, variant, prediction, seed, *, root, formats, dry_run):
    source = _prediction_path(root, study, topology, variant.name, seed, prediction.name)
    target = _run_dir(root, study, topology, variant.name, seed) / "plots" / prediction.name
    print(f"plot {source} -> {target}")
    if not dry_run:
        plot_prediction(source, target, formats=formats)


def _metric_row(study, topology, variant, prediction, seed, metrics):
    if metrics["assignment_formulation"] == "discriminative":
        assignment = metrics["visible"]["complete"]
        accuracy = assignment["top1_efficiency"]
        accuracy_error = assignment["top1_efficiency_standard_error"]
    else:
        assignment = metrics["assignment"]
        accuracy = assignment["empirical_mode_accuracy"]
        accuracy_error = assignment["empirical_mode_accuracy_standard_error"]

    invisible = metrics.get("invisible", {})
    energy = None
    if metrics["model_mode"] == "factorized":
        posterior = (
            invisible.get("truncated_assignment_weighted_mixture")
            or invisible.get("top1_assignment_conditioned")
        )
        if posterior:
            energy = posterior.get("energy_score_joint")
    else:
        energy = invisible.get("energy_score_joint")
    common = metrics.get("common_non_null_truth_assignment", {})
    compatibility = metrics.get("compatibility", {})
    return {
        "study": study.name,
        "topology": topology,
        "variant": variant.name,
        "variant_label": variant.label,
        "prediction": prediction.name,
        "prediction_label": prediction.label,
        "reporting": variant.reporting,
        "seed": seed,
        "events": metrics["events"],
        "assignment_support": metrics["assignment_support"]["name"],
        "assignment_accuracy": accuracy,
        "assignment_accuracy_standard_error": accuracy_error,
        "common_non_null_events": common.get("events"),
        "common_non_null_assignment_accuracy": common.get("accuracy"),
        "common_non_null_assignment_standard_error": common.get("standard_error"),
        "compatibility_assignment_only_top1": compatibility.get("assignment_only_top1"),
        "compatibility_combined_top1": compatibility.get("combined_top1"),
        "energy_score": energy,
        "sample_equivalent_nfe_per_event": (
            metrics.get("sample_equivalent_nfe_per_event")
            or metrics.get("network_calls_per_event")
        ),
        "prediction_seed": metrics.get("study", {}).get("prediction_seed"),
        "prediction_file": metrics["prediction_file"],
    }


def _summary(study, topologies, variants, seeds, *, root, dry_run):
    rows = []
    for topology in topologies:
        for variant in variants:
            if variant.reporting == "setup":
                continue
            for seed in seeds:
                for prediction in variant.predictions:
                    path = _evaluation_path(root, study, topology, variant.name, seed, prediction.name)
                    if not path.is_file():
                        if dry_run:
                            continue
                        raise FileNotFoundError(path)
                    rows.append(
                        _metric_row(
                            study, topology, variant, prediction, seed,
                            json.loads(path.read_text()),
                        )
                    )
    destination = root / study.name
    print(f"summarize {len(rows)} results -> {destination}")
    if dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["topology"], row["variant"], row["prediction"],
                row["reporting"], row["assignment_support"],
            )
        ].append(row)
    aggregate = []
    for key, values in sorted(groups.items()):
        item = {
            "topology": key[0], "variant": key[1], "prediction": key[2],
            "reporting": key[3], "assignment_support": key[4], "seeds": len(values),
        }
        for metric in (
            "assignment_accuracy", "common_non_null_assignment_accuracy",
            "energy_score", "sample_equivalent_nfe_per_event",
        ):
            numbers = [float(value[metric]) for value in values if value[metric] is not None]
            item[f"{metric}_mean"] = statistics.mean(numbers) if numbers else None
            item[f"{metric}_seed_std"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        aggregate.append(item)
    (destination / "summary_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )


def _compare(study, topology, variants, seed, *, root, formats, include_diagnostics, dry_run):
    paths, labels = [], []
    for variant in variants:
        if variant.reporting == "setup" or (
            variant.reporting != "primary" and not include_diagnostics
        ):
            continue
        for prediction in variant.predictions:
            path = _prediction_path(root, study, topology, variant.name, seed, prediction.name)
            if path.is_file() or dry_run:
                paths.append(path)
                labels.append(f"{variant.label}: {prediction.label}")
    if len(paths) < 2:
        return
    if not dry_run:
        assert_same_events(paths)
        plot_comparison(
            paths,
            root / study.name / topology / "comparisons" / f"seed_{seed}",
            labels=labels,
            formats=formats,
        )


def _study_specific_analysis(study, topologies, variants, seeds, *, root, dry_run):
    if study.name != "visible_recurrence":
        print(f"No dedicated analysis is maintained for {study.name}.")
        return
    required = {
        "m1",
        "m2",
        "m3",
        "m2_parameter_control",
        "m2_compute_control",
        "m3_parameter_control",
        "m3_compute_control",
    }
    selected = {variant.name for variant in variants}
    if not required <= selected:
        print(
            "Skip Study 01 dedicated analysis for partial variant selection; "
            f"missing={sorted(required - selected)}"
        )
        return
    for topology in topologies:
        for seed in seeds:
            command = [
                sys.executable,
                str(ROOT / "scripts/analyze_study_01.py"),
                "--output-root",
                str(root),
                "--topology",
                topology,
                "--seed",
                str(seed),
            ]
            print(shlex.join(command))
            if not dry_run:
                subprocess.run(command, cwd=ROOT, check=True)


def _plan(study, topologies, requested, training, seeds, root):
    print(
        f"study: {study.name}\n"
        f"topologies: {', '.join(topologies)}\n"
        f"seeds: {', '.join(map(str, seeds))}\n"
        f"output: {root / study.name}\n"
        f"requested: {', '.join(variant.name for variant in requested)}\n"
        f"training closure: {', '.join(variant.name for variant in training)}"
    )
    for variant in requested:
        print(
            f"  {variant.name:24s} {variant.reporting:10s} "
            f"init={variant.initialise_from or '-':20s} predictions={len(variant.predictions)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a validated JER study manifest")
    parser.add_argument("study", type=Path)
    parser.add_argument(
        "stage",
        choices=(
            "plan",
            "train",
            "predict",
            "evaluate",
            "plot",
            "compare",
            "summary",
            "analyze",
            "all",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/studies"))
    parser.add_argument("--topologies", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--set", dest="global_overrides", action="append", default=[])
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument(
        "--exact-variants",
        action="store_true",
        help="Do not expand training dependencies; intended for dependency-ordered batch workers.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    study = load_study(args.study)
    topologies = _selected(args.topologies, study.topologies, "topologies")
    names = tuple(variant.name for variant in study.variants)
    requested_names = _selected(args.variants, names, "variants")
    requested = tuple(study.variant_map[name] for name in requested_names)
    if args.variants == "all":
        training = study.dependency_order()
    elif args.exact_variants:
        training = requested
    else:
        training = study.dependency_closure(requested_names)

    seeds = (
        study.seeds
        if args.seeds == "all"
        else tuple(int(value) for value in args.seeds.split(",") if value)
    )
    unknown_seeds = sorted(set(seeds) - set(study.seeds))
    if unknown_seeds:
        raise ValueError(f"Seeds {unknown_seeds} are not declared by the study")
    root = args.output_root.expanduser().resolve()
    global_overrides = tuple(args.global_overrides)
    formats = tuple(args.formats)
    _plan(study, topologies, requested, training, seeds, root)

    if args.stage == "plan":
        for topology in topologies:
            for variant in training:
                for seed in seeds:
                    _public_run_config(study, topology, variant, seed, root, global_overrides)
            for variant in requested:
                for seed in seeds:
                    for prediction in variant.predictions:
                        _public_run_config(
                            study, topology, variant, seed, root,
                            global_overrides, prediction,
                        )
        print("All selected compact training and prediction configurations validate.")
        return

    stages = (
        ("train", "predict", "evaluate", "plot", "compare", "summary", "analyze")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        if stage == "train":
            for topology in topologies:
                for variant in training:
                    for seed in seeds:
                        _train_one(
                            study, topology, variant, seed, root=root,
                            global_overrides=global_overrides, device=args.device,
                            force=args.force, dry_run=args.dry_run,
                        )
        elif stage in {"predict", "evaluate", "plot"}:
            for topology in topologies:
                for variant in requested:
                    if variant.reporting == "setup":
                        continue
                    for seed in seeds:
                        for prediction in variant.predictions:
                            if stage == "predict":
                                _predict_one(
                                    study, topology, variant, prediction, seed,
                                    root=root, global_overrides=global_overrides,
                                    device=args.device, force=args.force,
                                    dry_run=args.dry_run,
                                )
                            elif stage == "evaluate":
                                _evaluate_one(
                                    study, topology, variant, prediction, seed,
                                    root=root, force=args.force, dry_run=args.dry_run,
                                )
                            else:
                                _plot_one(
                                    study, topology, variant, prediction, seed,
                                    root=root, formats=formats, dry_run=args.dry_run,
                                )
        elif stage == "compare":
            for topology in topologies:
                for seed in seeds:
                    _compare(
                        study, topology, requested, seed, root=root, formats=formats,
                        include_diagnostics=args.include_diagnostics,
                        dry_run=args.dry_run,
                    )
        elif stage == "summary":
            _summary(study, topologies, requested, seeds, root=root, dry_run=args.dry_run)
        elif stage == "analyze":
            _study_specific_analysis(
                study,
                topologies,
                requested,
                seeds,
                root=root,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
