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

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

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
from joint_event_reconstruction.studies import (
    PredictionSpec,
    StudySpec,
    VariantSpec,
    load_study,
    resolve_study,
)
from joint_event_reconstruction.study_metrics import (
    assert_same_events,
    common_non_null_truth_assignment,
)
from joint_event_reconstruction.runtime_paths import runtime_paths, verify_source_snapshot


RUNTIME_PATHS = runtime_paths()
ROOT = RUNTIME_PATHS.source_root
PROJECT_ROOT = RUNTIME_PATHS.project_root


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


def _training_config_identity(resolved) -> dict:
    """Keep prediction-only settings out of the checkpoint identity."""
    config = OmegaConf.to_container(resolved, resolve=True)
    config.pop("inference", None)
    training = config.get("training")
    if isinstance(training, dict):
        # These switches affect execution mechanics only.  They must not make
        # an otherwise identical checkpoint scientifically incompatible.
        training.pop("runtime", None)
        training.pop("output", None)
    return config


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
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _maintained_study_path(name: str) -> Path:
    matches = []
    for path in sorted((ROOT / "configs/studies").glob("[0-9][0-9]_*.yaml")):
        if load_study(path).name == name:
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(
            f"Checkpoint source study {name!r} must resolve to exactly one maintained "
            f"manifest; matches={matches}"
        )
    return matches[0]


def _gate_checkpoint(
    study: StudySpec, gate_name: str, topology: str, seed: int
) -> Path:
    """Resolve and verify one immutable human-selected checkpoint artifact."""
    entry = study.gate_artifact(gate_name, topology, seed)
    checkpoint = _repo_path(entry["checkpoint"])
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise FileNotFoundError(
            f"Selected gate artifact is missing or empty: gate={gate_name}, "
            f"topology={topology}, model_seed={seed}, checkpoint={checkpoint}"
        )
    observed_checkpoint = _file_identity(checkpoint)
    if observed_checkpoint["sha256"] != entry["checkpoint_sha256"]:
        raise RuntimeError(
            f"Selected gate checkpoint checksum mismatch: gate={gate_name}, "
            f"topology={topology}, model_seed={seed}, expected="
            f"{entry['checkpoint_sha256']}, observed={observed_checkpoint['sha256']}"
        )
    identity_file = _repo_path(entry["identity_file"])
    if not identity_file.is_file():
        raise FileNotFoundError(
            f"Selected gate artifact identity is missing: gate={gate_name}, "
            f"topology={topology}, model_seed={seed}, identity={identity_file}"
        )
    if _sha256_file(identity_file) != entry["identity_sha256"]:
        raise RuntimeError(
            f"Selected gate identity checksum mismatch: gate={gate_name}, "
            f"topology={topology}, model_seed={seed}, identity={identity_file}"
        )
    record = json.loads(identity_file.read_text())
    identity = record.get("identity")
    artifact = record.get("artifact")
    if not isinstance(identity, dict) or not isinstance(artifact, dict):
        raise ValueError(f"Selected gate identity is not a training artifact record: {identity_file}")
    if artifact != observed_checkpoint:
        raise RuntimeError(
            f"Selected gate artifact record does not describe its checkpoint: {identity_file}"
        )
    expected_identity = {
        "git_commit": entry["git_sha"],
        "study": entry["source_study"],
        "variant": entry["variant"],
        "topology": topology,
        "model_seed": int(entry["model_seed"]),
        "architecture_signature": entry["architecture_identity"],
    }
    mismatches = {
        key: (identity.get(key), value)
        for key, value in expected_identity.items()
        if identity.get(key) != value
    }
    observed_subset = identity.get("data_subset")
    observed_subset_seed = (
        observed_subset.get("subset_seed") if isinstance(observed_subset, dict) else None
    )
    if observed_subset_seed != int(entry["data_subset_seed"]):
        mismatches["data_subset.subset_seed"] = (
            observed_subset_seed,
            int(entry["data_subset_seed"]),
        )
    if mismatches:
        raise RuntimeError(
            f"Selected gate artifact identity mismatch for {gate_name}:{topology}:seed_{seed}: "
            f"{mismatches}"
        )
    return checkpoint


def _initial_checkpoint(
    study: StudySpec,
    variant: VariantSpec,
    topology: str,
    seed: int,
    root: Path,
) -> Path | None:
    if variant.initialise_from_gate is not None:
        return _gate_checkpoint(study, variant.initialise_from_gate, topology, seed)
    if variant.initialise_from is not None:
        return _checkpoint(root, study, topology, variant.initialise_from, seed)
    return None


def _prediction_checkpoint(
    study: StudySpec,
    variant: VariantSpec,
    topology: str,
    seed: int,
    root: Path,
) -> Path:
    if variant.checkpoint_from_gate is not None:
        return _gate_checkpoint(study, variant.checkpoint_from_gate, topology, seed)
    if variant.checkpoint_from is None:
        return _checkpoint(root, study, topology, variant.name, seed)
    source_study_name, source_variant = variant.checkpoint_from
    source_study = load_study(_maintained_study_path(source_study_name))
    if seed not in source_study.seeds:
        raise ValueError(
            f"Checkpoint source {source_study_name}:{source_variant} does not declare "
            f"model seed {seed}"
        )
    return _checkpoint(root, source_study, topology, source_variant, seed)


@lru_cache(maxsize=None)
def _data_subset_identity_cached(
    source_path_text: str,
    subset_seed: int,
    train_count: int | None,
    validation_count: int | None,
    test_count: int | None,
) -> dict:
    source_path = _repo_path(source_path_text)
    requested = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    split_identity = {}
    with h5py.File(source_path, "r") as source:
        if "SPLIT" not in source or "EVENT_ID" not in source:
            raise KeyError(f"Canonical dataset requires EVENT_ID and SPLIT: {source_path}")
        membership = np.asarray(source["SPLIT"][:], dtype=np.uint8)
        for split, count in requested.items():
            split_value = {"train": 0, "validation": 1, "test": 2}[split]
            rows = np.flatnonzero(membership == split_value)
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
        "split_source": f"{source_path}:SPLIT",
        "invisible_truth": _file_identity(source_path),
        "splits": split_identity,
    }


def _data_subset_identity(resolved) -> dict:
    limits = resolved.data.max_events
    return _data_subset_identity_cached(
        str(Path(resolved.data.input).resolve()),
        int(resolved.data.subset_seed),
        None if limits.train is None else int(limits.train),
        None if limits.val is None else int(limits.val),
        None if limits.test is None else int(limits.test),
    )


def _require_clean_git() -> str:
    if ROOT != PROJECT_ROOT:
        verify_source_snapshot(
            RUNTIME_PATHS.source_sha,
            project_root=PROJECT_ROOT,
            source_root=ROOT,
        )
        return RUNTIME_PATHS.source_sha
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
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
    training_config = _training_config_identity(resolved)
    data_subset = _data_subset_identity(resolved)
    architecture_payload = training_config.get("model")
    return {
        "schema": ARTIFACT_IDENTITY_SCHEMA,
        "kind": "training",
        "git_commit": git_commit,
        "study": study.name,
        "study_manifest_sha256": _sha256_file(study.path),
        "topology": topology,
        "variant": variant.name,
        "variant_label": variant.label,
        "variant_patch": variant.patch,
        "model_seed": int(seed),
        "architecture_signature": _identity_digest(architecture_payload),
        "topology_config": _file_identity(resolved.topology_config),
        "data_subset": data_subset,
        "training_event_fingerprint": data_subset["splits"]["train"][
            "ordered_event_id_sha256"
        ],
        "base_experiment": str(experiment),
        "base_experiment_sha256": _sha256_file(experiment),
        "resolved_config": training_config,
        "resolved_gate_digest": study.gates_digest,
        "precision": str(resolved.training.precision),
        "trainable_components": sorted(str(value) for value in resolved.training.trainable),
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
    checkpoint_training_sha,
    recurrent_steps,
    git_commit,
) -> dict:
    resolved_inference = OmegaConf.to_container(resolved.inference, resolve=True)
    return {
        "schema": ARTIFACT_IDENTITY_SCHEMA,
        "kind": "prediction",
        "git_commit": git_commit,
        "study": study.name,
        "study_manifest_sha256": _sha256_file(study.path),
        "topology": topology,
        "variant": variant.name,
        "prediction": prediction.name,
        "model_seed": int(seed),
        "topology_config": _file_identity(resolved.topology_config),
        "data_subset": _data_subset_identity(resolved),
        "split": study.split,
        "checkpoint": _file_identity(checkpoint),
        "resolved_inference": resolved_inference,
        "resolved_gate_digest": study.gates_digest,
        "prediction_seed": int(resolved.inference.seed),
        "recurrent_steps": recurrent_steps,
        # A recovered prediction may consume a checkpoint trained at an older
        # source revision.  Keep that training provenance separate from the
        # revision that produced this prediction artifact.
        "checkpoint_training_sha": checkpoint_training_sha,
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
    analysis = study_root / topology / "analysis" / f"seed_{seed}"
    if analysis.exists():
        shutil.rmtree(analysis)
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


def _predictions_for_variant(variant: VariantSpec, selection: str) -> tuple[PredictionSpec, ...]:
    """Resolve a strict prediction-name filter for one variant."""
    names = tuple(prediction.name for prediction in variant.predictions)
    if selection == "all":
        return variant.predictions
    selected = tuple(part.strip() for part in selection.split(",") if part.strip())
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise ValueError(
            f"Unknown prediction name for variant {variant.name!r}: {unknown}; "
            f"allowed={list(names)}"
        )
    return tuple(
        prediction for prediction in variant.predictions if prediction.name in selected
    )


def _evaluation_complete(path: Path) -> bool:
    """Recognise a non-empty maintained evaluation JSON without rewriting it."""
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, dict)
        and isinstance(value.get("prediction_file"), str)
        and bool(value.get("prediction_file"))
        and isinstance(value.get("events"), int)
        and value["events"] > 0
    )


def _plot_complete(path: Path) -> bool:
    """Require actual non-empty plot files, not merely an existing directory."""
    return path.is_dir() and any(
        item.is_file() and item.stat().st_size > 0
        for item in path.iterdir()
    )


def _git_commit() -> str | None:
    if ROOT != PROJECT_ROOT:
        return RUNTIME_PATHS.source_sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
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
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _train_one(study, topology, variant, seed, *, root, global_overrides, device, force, dry_run):
    run_dir = _run_dir(root, study, topology, variant.name, seed)
    checkpoint = _checkpoint(root, study, topology, variant.name, seed)
    experiment, public, resolved = _public_run_config(
        study, topology, variant, seed, root, global_overrides
    )
    initial = _initial_checkpoint(study, variant, topology, seed, root)
    if initial is not None:
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
                    "initialise_from_gate": variant.initialise_from_gate,
                    "initial_components": list(variant.initialise_components),
                    "resolved_gate_digest": study.gates_digest,
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
    checkpoint = _prediction_checkpoint(study, variant, topology, seed, root)
    if not checkpoint.is_file() and not dry_run:
        raise FileNotFoundError(f"Prediction requires checkpoint {checkpoint}")
    target = _prediction_path(root, study, topology, variant.name, seed, prediction.name)
    _, public, resolved = _public_run_config(
        study, topology, variant, seed, root, global_overrides, prediction
    )
    recurrent_steps = prediction.recurrent_steps
    identity = None
    identity_path = target.with_suffix(".identity.json")
    checkpoint_training_sha = None
    if not dry_run:
        git_commit = _require_clean_git()
        if variant.checkpoint_from is None and variant.checkpoint_from_gate is None:
            training_identity_path = run_dir / "train/artifact_identity.json"
            if not training_identity_path.is_file():
                raise FileNotFoundError(
                    f"Prediction checkpoint has no training identity: "
                    f"{training_identity_path}"
                )
            source_payload = json.loads(training_identity_path.read_text())
            source_identity = source_payload.get("identity", {})
            if not isinstance(source_identity, dict):
                raise ValueError(
                    f"Prediction checkpoint training identity is malformed: "
                    f"{training_identity_path}"
                )
            _verify_artifact_record(
                training_identity_path,
                checkpoint,
                source_identity,
                "prediction checkpoint source",
            )
            if source_identity.get("kind") != "training":
                raise ValueError(
                    f"Prediction checkpoint source is not a training artifact: "
                    f"{training_identity_path}"
                )
            checkpoint_training_sha = source_identity.get("git_commit")
            training_experiment, _, training_resolved = _public_run_config(
                study, topology, variant, seed, root, global_overrides
            )
            initial = _initial_checkpoint(study, variant, topology, seed, root)
            if initial is not None and not initial.is_file():
                raise FileNotFoundError(
                    f"Variant {variant.name!r} requires checkpoint {initial}"
                )
            expected_training = _training_identity(
                study,
                topology,
                variant,
                seed,
                training_experiment,
                training_resolved,
                initial,
                checkpoint_training_sha,
            )
            # A gate can be locked after a model has been trained.  The
            # selected patch must still resolve identically, while the gate
            # digest itself is expected to change.  Compare the historical
            # identity with all other training fields unchanged.
            expected_training["resolved_gate_digest"] = source_identity.get(
                "resolved_gate_digest"
            )
            _verify_artifact_record(
                training_identity_path,
                checkpoint,
                expected_training,
                "training checkpoint",
            )
        else:
            if variant.checkpoint_from_gate is not None:
                gate_artifact = study.gate_artifact(
                    variant.checkpoint_from_gate, topology, seed
                )
                source_study_name = gate_artifact["source_study"]
                source_variant = gate_artifact["variant"]
            else:
                source_study_name, source_variant = variant.checkpoint_from
            source_record = checkpoint.parent.parent / "artifact_identity.json"
            if not source_record.is_file():
                raise FileNotFoundError(
                    f"Prediction checkpoint source {source_study_name}:{source_variant} "
                    f"has no artifact identity: {source_record}"
                )
            source_payload = json.loads(source_record.read_text())
            source_identity = source_payload.get("identity", {})
            source_artifact = source_payload.get("artifact", {})
            if not isinstance(source_identity, dict) or not isinstance(source_artifact, dict):
                raise ValueError(
                    f"Prediction checkpoint source identity is malformed: {source_record}"
                )
            _verify_artifact_record(
                source_record,
                checkpoint,
                source_identity,
                "prediction checkpoint source",
            )
            if source_identity.get("kind") != "training":
                raise ValueError(
                    f"Prediction checkpoint source is not a training artifact: {source_record}"
                )
            checkpoint_training_sha = source_identity.get("git_commit")
            current_model = OmegaConf.to_container(resolved.model, resolve=True)
            source_config = source_identity.get("resolved_config")
            if not isinstance(source_config, dict) or not isinstance(source_config.get("model"), dict):
                raise ValueError(
                    f"Prediction checkpoint source has no resolved model identity: {source_record}"
                )
            source_model = source_config["model"]
            if (
                source_identity.get("study") != source_study_name
                or source_identity.get("variant") != source_variant
                or source_identity.get("topology") != topology
                or int(source_identity.get("model_seed", -1)) != int(seed)
                or source_artifact.get("sha256") != _sha256_file(checkpoint)
                or source_model != current_model
            ):
                raise RuntimeError(
                    f"Prediction checkpoint identity does not match declared source "
                    f"{source_study_name}:{source_variant}:seed_{seed}: {source_record}"
                )
        identity = _prediction_identity(
            study,
            topology,
            variant,
            prediction,
            seed,
            resolved,
            checkpoint,
            checkpoint_training_sha,
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
    if target.is_file() and not force:
        if _evaluation_complete(target):
            print(f"SKIP evaluate (valid output exists): {target}")
            return
        print(f"REPAIR evaluate (invalid output): {target}")
    if dry_run:
        print(f"evaluate {source} -> {target}")
        return
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


def _plot_one(
    study, topology, variant, prediction, seed, *, root, formats, force, dry_run
):
    source = _prediction_path(root, study, topology, variant.name, seed, prediction.name)
    target = _run_dir(root, study, topology, variant.name, seed) / "plots" / prediction.name
    if target.exists() and not force:
        if _plot_complete(target):
            print(f"SKIP plot (valid output exists): {target}")
            return
        print(f"REPAIR plot (invalid output): {target}")
    print(f"plot {source} -> {target}")
    if not dry_run:
        if force and target.exists():
            shutil.rmtree(target)
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
            if not dry_run and not path.is_file():
                raise FileNotFoundError(
                    f"Comparison requires prediction artifact {path}; "
                    "the requested comparison set is incomplete"
                )
            paths.append(path)
            labels.append(f"{variant.label}: {prediction.label}")
    if len(paths) < 2:
        return
    if not dry_run:
        assert_same_events(paths)
        if study.name == "generator_inference":
            # Study 08 already has one compact evaluation JSON per prediction.
            # Calling plot_comparison here would re-read every large posterior
            # HDF5 product and recompute all metrics during aggregate status.
            # Keep the structural event-order contract, then leave quality
            # numbers to review_generator_inference.py.
            comparison_dir = root / study.name / topology / "comparisons" / f"seed_{seed}"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            (comparison_dir / "comparison_metrics.json").write_text(
                json.dumps(
                    {
                        "study": study.name,
                        "topology": topology,
                        "seed": seed,
                        "inputs": [
                            {"label": label, "prediction_file": str(path)}
                            for label, path in zip(labels, paths)
                        ],
                        "metrics_source": "per-prediction evaluation JSON",
                        "posterior_hdf5_reloaded": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            return
        plot_comparison(
            paths,
            root / study.name / topology / "comparisons" / f"seed_{seed}",
            labels=labels,
            formats=formats,
        )


def _training_event_count(
    study,
    topology: str,
    variant,
    seed: int,
    *,
    root: Path,
    global_overrides: tuple[str, ...],
) -> int:
    """Return the authoritative training-event count from semantic provenance."""
    run_dir = _run_dir(root, study, topology, variant.name, seed)
    sources: dict[str, int] = {}
    identity_path = run_dir / "train/artifact_identity.json"
    if identity_path.is_file():
        try:
            payload = json.loads(identity_path.read_text())
            selected = payload["identity"]["data_subset"]["splits"]["train"][
                "selected_events"
            ]
            sources["training artifact identity"] = int(selected)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Malformed training-event provenance for {study.name}:{topology}:"
                f"{variant.name}:seed_{seed}: {identity_path}"
            ) from error

    _, _, resolved = _public_run_config(
        study, topology, variant, seed, root, global_overrides
    )
    resolved_count = resolved.data.max_events.train
    if resolved_count is not None:
        sources["resolved variant configuration"] = int(resolved_count)

    patch_data = variant.patch.get("data", {})
    patch_events = patch_data.get("events", {}) if isinstance(patch_data, dict) else {}
    patch_count = patch_events.get("train") if isinstance(patch_events, dict) else None
    if patch_count is not None:
        sources["variant patch"] = int(patch_count)

    if not sources:
        raise ValueError(
            f"No semantic training-event count for {study.name}:{topology}:"
            f"{variant.name}:seed_{seed}"
        )
    if any(value <= 0 for value in sources.values()):
        raise ValueError(
            f"Training-event count must be positive for {study.name}:{topology}:"
            f"{variant.name}:seed_{seed}: {sources}"
        )
    if len(set(sources.values())) != 1:
        raise ValueError(
            f"Conflicting training-event provenance for {study.name}:{topology}:"
            f"{variant.name}:seed_{seed}: {sources}"
        )
    return next(iter(sources.values()))


def _study_specific_analysis(
    study, topologies, variants, seeds, *, root, global_overrides, device, dry_run
):
    if study.name in {"generator_formulations", "generator_inference"}:
        selected_names = {variant.name for variant in variants}
        required_names = (
            {"edm", "cfm", "ot_cfm", "conditional_ot_cfm"}
            if study.name == "generator_formulations"
            else {"edm", "cfm"}
        )
        if not required_names <= selected_names:
            print(
                f"Skip {study.name} human review for partial variant selection; "
                f"missing={sorted(required_names - selected_names)}"
            )
            return
        script = (
            "review_generator_formulations.py"
            if study.name == "generator_formulations"
            else "review_generator_inference.py"
        )
        command = [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--output-root",
            str(root),
        ]
        print(shlex.join(command))
        if not dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return
    if study.name == "encoder_backbone":
        required = {
            "transformer_baseline",
            "fourvector_transformer",
            "relational_transformer",
            "lgatr",
            "lgatr_slim",
        }
        selected = {variant.name for variant in variants}
        if not required <= selected:
            print(
                "Skip Study-12 complementarity for partial variant selection; "
                f"missing={sorted(required - selected)}"
            )
            return
        arguments = []
        for topology in topologies:
            for seed in seeds:
                for variant in variants:
                    if variant.name not in required:
                        continue
                    if len(variant.predictions) != 1:
                        raise ValueError(
                            f"Study-12 variant {variant.name!r} must have one prediction "
                            "for complementarity analysis"
                        )
                    prediction = variant.predictions[0]
                    path = _prediction_path(
                        root, study, topology, variant.name, seed, prediction.name
                    )
                    arguments.extend(
                        ["--prediction", f"seed_{seed}/{variant.name}={path}"]
                    )
        output = root / study.name
        command = [
            sys.executable,
            str(ROOT / "scripts/analyze_pairwise_complementarity.py"),
            "--output",
            str(output),
            *arguments,
        ]
        print(shlex.join(command))
        if not dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        uncertainty_arguments = []
        for topology in topologies:
            for seed in seeds:
                for variant in variants:
                    if variant.name not in required:
                        continue
                    prediction = variant.predictions[0]
                    path = _prediction_path(
                        root, study, topology, variant.name, seed, prediction.name
                    )
                    uncertainty_arguments.extend(
                        ["--prediction", f"{variant.name}/seed_{seed}={path}"]
                    )
        uncertainty_command = [
            sys.executable,
            str(ROOT / "scripts/analyze_study14.py"),
            "--output",
            str(output),
            *uncertainty_arguments,
        ]
        print(shlex.join(uncertainty_command))
        if not dry_run:
            subprocess.run(uncertainty_command, cwd=PROJECT_ROOT, check=True)
        return
    if study.name == "assignment_conditioning":
        arguments = []
        for topology in topologies:
            for seed in seeds:
                for variant in variants:
                    for prediction in variant.predictions:
                        path = _prediction_path(
                            root, study, topology, variant.name, seed, prediction.name
                        )
                        arguments += [
                            "--prediction",
                            f"{variant.name}/{prediction.name}/seed_{seed}={path}",
                        ]
        command = [
            sys.executable,
            str(ROOT / "scripts/analyze_assignment_conditioning.py"),
            "--output",
            str(root / study.name),
            *arguments,
        ]
        print(shlex.join(command))
        if not dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return
    if study.name == "c2_feedback":
        selected = {variant.name: variant for variant in variants}
        if not {"continued_no_feedback", "posterior_feedback"} <= set(selected):
            print("Skip Study-04 transition analysis for a partial variant selection.")
            return
        for topology in topologies:
            for seed in seeds:
                base_variant = selected["continued_no_feedback"]
                feedback_variant = selected["posterior_feedback"]
                if len(base_variant.predictions) != 1 or len(feedback_variant.predictions) != 1:
                    raise ValueError("Study-04 transition diagnostics require one prediction per variant")
                base_prediction = base_variant.predictions[0]
                feedback_prediction = feedback_variant.predictions[0]
                base = _prediction_path(
                    root, study, topology, base_variant.name, seed, base_prediction.name
                )
                feedback = _prediction_path(
                    root, study, topology, feedback_variant.name, seed, feedback_prediction.name
                )
                output = root / study.name / topology / "analysis" / f"seed_{seed}"
                command = [
                    sys.executable,
                    str(ROOT / "scripts/analyze_feedback_transitions.py"),
                    "--base", str(base),
                    "--feedback", str(feedback),
                    "--output", str(output),
                ]
                print(shlex.join(command))
                if not dry_run:
                    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return
    if study.name == "data_scaling":
        arguments_m1 = []
        arguments_m3 = []
        for topology in topologies:
            for seed in seeds:
                for variant in variants:
                    if len(variant.predictions) != 1:
                        raise ValueError("Study-11 scaling diagnostics require one prediction per variant")
                    prediction = variant.predictions[0]
                    path = _prediction_path(
                        root, study, topology, variant.name, seed, prediction.name
                    )
                    count = _training_event_count(
                        study,
                        topology,
                        variant,
                        seed,
                        root=root,
                        global_overrides=global_overrides,
                    )
                    if variant.name == "n500k":
                        arguments_m1 += ["--m1", f"M1:{count}:{seed}={path}"]
                    elif variant.name == "m3_n500k":
                        arguments_m3 += ["--m3", f"M3:{count}:{seed}={path}"]
        if not arguments_m1:
            print("Skip Study-11 scaling analysis without an M1 primary curve.")
            return
        command = [
            sys.executable,
            str(ROOT / "scripts/analyze_data_scaling.py"),
            "--output",
            str(root / study.name),
            *arguments_m1,
            *arguments_m3,
        ]
        print(shlex.join(command))
        if not dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return
    if study.name == "posterior_size":
        required = {"factorized", "joint"}
        selected = {variant.name for variant in variants}
        if not required <= selected:
            print(
                "Skip posterior-size analysis for partial variant selection; "
                f"missing={sorted(required - selected)}"
            )
            return
        for topology in topologies:
            for seed in seeds:
                factorized = _prediction_path(
                    root, study, topology, "factorized", seed, "k10_m64"
                )
                joint = _prediction_path(
                    root, study, topology, "joint", seed, "s64"
                )
                output = (
                    root / study.name / topology / "analysis" / f"seed_{seed}"
                    / "posterior_convergence.json"
                )
                command = [
                    sys.executable,
                    str(ROOT / "scripts/analyze_posterior_convergence.py"),
                    "--factorized", str(factorized),
                    "--joint", str(joint),
                    "--output", str(output),
                ]
                print(shlex.join(command))
                if not dry_run:
                    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return
    if study.name == "compatibility_reranking":
        required = {"compatibility"}
        selected = {variant.name for variant in variants}
        if not required <= selected:
            print(
                "Skip compatibility analysis for partial variant selection; "
                f"missing={sorted(required - selected)}"
            )
            return
        arguments = []
        for topology in topologies:
            for seed in seeds:
                prediction = _prediction_path(
                    root, study, topology, "compatibility", seed, "posterior"
                )
                experiment = _run_dir(
                    root, study, topology, "compatibility", seed
                ) / "predictions/posterior.yaml"
                checkpoint = _prediction_checkpoint(
                    study, study.variant_map["compatibility"], topology, seed, root
                )
                arguments.extend(
                    [
                        "--prediction", str(prediction),
                        "--experiment", str(experiment),
                        "--checkpoint", str(checkpoint),
                    ]
                )
        output = root / study.name / "study10_human_review"
        command = [
            sys.executable,
            str(ROOT / "scripts/run_study10_controls.py"),
            "--output", str(output),
            "--study-manifest-sha256", _sha256_file(study.path),
            "--gate-digest", study.gates_digest or "",
            "--device", device,
            *arguments,
        ]
        print(shlex.join(command))
        if not dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return
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
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


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
        "--predictions",
        default="all",
        help="Comma-separated prediction names for predict/evaluate/plot stages.",
    )
    parser.add_argument(
        "--exact-variants",
        action="store_true",
        help="Do not expand training dependencies; intended for dependency-ordered batch workers.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--expected-gates-digest",
        help="Fail if the locked gate selections changed after Slurm submission.",
    )
    args = parser.parse_args()

    study = resolve_study(args.study)
    if args.expected_gates_digest is not None and study.gates_digest != args.expected_gates_digest:
        raise RuntimeError(
            "Scientific gate selections changed after submission: "
            f"expected={args.expected_gates_digest}, found={study.gates_digest}"
        )
    topologies = _selected(args.topologies, study.topologies, "topologies")
    names = tuple(variant.name for variant in study.variants)
    requested_names = _selected(args.variants, names, "variants")
    requested = tuple(study.variant_map[name] for name in requested_names)
    if args.variants == "all":
        training = study.training_order() if study.training else ()
    elif not study.training:
        training = ()
    elif args.exact_variants:
        training = tuple(
            variant
            for variant in requested
            if variant.checkpoint_from is None
            and variant.checkpoint_from_gate is None
        )
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
    root = args.output_root.expanduser()
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    root = root.resolve()
    global_overrides = tuple(args.global_overrides)
    formats = tuple(args.formats)
    _plan(study, topologies, requested, training, seeds, root)

    if args.stage == "plan":
        for topology in topologies:
            for variant in training:
                for seed in seeds:
                    _public_run_config(
                        study, topology, variant, seed, root, global_overrides
                    )
            for variant in requested:
                for seed in seeds:
                    predictions = _predictions_for_variant(variant, args.predictions)
                    for prediction in predictions:
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
            if not study.training:
                print(f"No training stage for inference-only study {study.name}.")
                continue
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
                        predictions = _predictions_for_variant(variant, args.predictions)
                        for prediction in predictions:
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
                                    root=root, formats=formats, force=args.force,
                                    dry_run=args.dry_run,
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
                global_overrides=global_overrides,
                device=args.device,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
