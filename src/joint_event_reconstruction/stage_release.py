"""Execution-backed validation shared by the Paper-1 downstream gates.

The campaign controller owns Slurm observations.  This module validates the
scientific contents of a release artifact independently of the scheduler so a
hand-written ``PASS`` file cannot release a study.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


class StageReleaseError(RuntimeError):
    """A stage release artifact does not satisfy its declared contract."""


class StageReleaseReviewRequired(StageReleaseError):
    """The executed evidence exposes a scientific choice requiring review."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise StageReleaseError(f"Cannot read release JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageReleaseError(f"Release JSON must be an object: {path}")
    return value


def gate_checkpoint_hashes(
    gates: dict[str, dict],
    required_gates: Iterable[str],
    *,
    seeds: Iterable[int] | None = None,
) -> dict[str, str]:
    """Return the exact selected checkpoint map for the gates used by a stage."""
    result: dict[str, str] = {}
    for gate_name in required_gates:
        gate = gates.get(gate_name)
        if not isinstance(gate, dict) or gate.get("status") != "locked":
            raise StageReleaseError(f"Required upstream gate is not locked: {gate_name}")
        artifacts = gate.get("artifacts")
        if artifacts == {}:
            continue
        if not isinstance(artifacts, dict):
            raise StageReleaseError(f"Locked gate has no artifact registry: {gate_name}")
        for topology, by_seed in artifacts.items():
            if not isinstance(by_seed, dict):
                raise StageReleaseError(f"Invalid artifact registry for {gate_name}:{topology}")
            for seed, artifact in by_seed.items():
                if seeds is not None and int(str(seed).removeprefix("seed_")) not in set(seeds):
                    continue
                if not isinstance(artifact, dict) or not artifact.get("checkpoint"):
                    raise StageReleaseError(f"Missing checkpoint for {gate_name}:{topology}:{seed}")
                path = str(Path(str(artifact["checkpoint"])).expanduser().resolve())
                result[path] = str(artifact.get("checkpoint_sha256", ""))
    return result


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _subresult_paths(payload: dict) -> list[tuple[Path, str | None]]:
    subresults = payload.get("subresults")
    if not isinstance(subresults, dict) or not subresults:
        raise StageReleaseError("Release artifact has no subresults")
    result: list[tuple[Path, str | None]] = []
    for name, value in subresults.items():
        if isinstance(value, str):
            result.append((Path(value), None))
            continue
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise StageReleaseError(f"Invalid subresult declaration: {name}")
        result.append((Path(value["path"]), value.get("schema")))
    return result


def _require_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise StageReleaseError(f"{label} is missing or is not an object")
    return value


def _require_true(value: object, label: str) -> None:
    if value is not True:
        raise StageReleaseError(f"{label} is not established by the executed preflight")


def _require_finite(value: object, label: str) -> None:
    if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
        raise StageReleaseError(f"{label} is not finite")


def _validate_prediction_evidence(
    value: object,
    label: str,
    *,
    expected_posterior_samples: int,
) -> None:
    report = _require_mapping(value, label)
    _require_true(report.get("physical_four_vectors"), f"{label}.physical_four_vectors")
    _require_true(report.get("event_id_unique"), f"{label}.event_id_unique")
    if report.get("events") != 16:
        raise StageReleaseError(f"{label}.events is not the 16-event integration contract")
    if report.get("four_vector_convention") != "E_px_py_pz_GeV_v1":
        raise StageReleaseError(f"{label} has the wrong physical four-vector convention")
    if report.get("top_k") != 10 or report.get("posterior_samples") != expected_posterior_samples:
        raise StageReleaseError(f"{label} has the wrong bounded posterior dimensions")
    if report.get("mass_shell") != "fixed massless exact":
        raise StageReleaseError(f"{label} has no mass-shell evidence")
    evidence = _require_mapping(report.get("mass_shell_evidence"), f"{label}.mass_shell_evidence")
    _require_finite(evidence.get("max_energy_residual"), f"{label}.mass_shell_evidence.max_energy_residual")
    _require_finite(
        evidence.get("max_relative_energy_residual"),
        f"{label}.mass_shell_evidence.max_relative_energy_residual",
    )
    if evidence.get("storage_dtype") != "float32" or evidence.get("ulp_tolerance") != 8.0:
        raise StageReleaseError(f"{label} has incomplete storage-precision mass-shell evidence")
    if not report.get("event_id_sha256"):
        raise StageReleaseError(f"{label} has no event-order identity")


def _validate_evaluation_evidence(value: object, label: str) -> None:
    report = _require_mapping(value, label)
    if report.get("primary_metric_path") != "invisible.top1_assignment_conditioned.energy_score_joint":
        raise StageReleaseError(f"{label} does not validate the Study-13/15 primary metric")
    _require_finite(report.get("primary_metric"), f"{label}.primary_metric")
    _require_true(report.get("conditional_calibration"), f"{label}.conditional_calibration")


def _validate_training_split_evidence(value: object, label: str) -> None:
    """Validate the train/validation separation emitted by the mini-run probe."""
    training = _require_mapping(value, label)
    data_subset = _require_mapping(training.get("data_subset"), f"{label}.data_subset")
    _require_true(
        data_subset.get("train_validation_batch_disjoint"),
        f"{label}.data_subset.train_validation_batch_disjoint",
    )
    _require_true(data_subset.get("test_accessed") is False, f"{label}.data_subset.test_accessed")


def _validate_nested_subresults(
    report: dict,
    *,
    expected_source_sha: str,
    validation_root: Path,
) -> None:
    if not report.get("subresults"):
        return
    for nested_path, schema in _subresult_paths(report):
        nested_path = Path(nested_path).expanduser().resolve()
        if not _under(nested_path, validation_root) or not nested_path.is_file():
            raise StageReleaseError(f"nested integration subresult is missing/outside validation root: {nested_path}")
        nested = _read_json(nested_path)
        if nested.get("status") != "PASS" or nested.get("source_sha") != expected_source_sha:
            raise StageReleaseError(f"nested integration subresult is not an executed PASS: {nested_path}")
        if schema is not None and nested.get("schema") != schema:
            raise StageReleaseError(f"nested integration subresult schema mismatch: {nested_path}")
        if nested.get("test_accessed") is True:
            raise StageReleaseError(f"nested integration subresult accessed the sealed test split: {nested_path}")


def _checkpoint_from_report(
    value: object,
    *,
    label: str,
    expected_checkpoint_hashes: dict[str, str],
    validation_root: Path | None = None,
) -> tuple[Path, str]:
    report = _require_mapping(value, label)
    path = Path(str(report.get("path", ""))).expanduser().resolve()
    digest = str(report.get("sha256", ""))
    if not path.is_file() or not digest or sha256_file(path) != digest:
        raise StageReleaseError(f"{label} does not identify an existing content-addressed checkpoint")
    if expected_checkpoint_hashes.get(str(path)) != digest:
        raise StageReleaseError(f"{label} does not match the locked upstream checkpoint")
    if validation_root is not None and _under(path, validation_root):
        return path, digest
    return path, digest


def _validate_study15_evidence(
    report: dict,
    *,
    expected_source_sha: str,
    expected_checkpoint_hashes: dict[str, str],
    validation_root: Path,
) -> None:
    if report.get("schema") != "jer_study15_integration_v1":
        raise StageReleaseError("Study-15 integration evidence has the wrong schema")
    if report.get("study_manifest_sha256") and report.get("test_accessed") is True:
        raise StageReleaseError("Study-15 integration evidence accessed the sealed test split")
    selected_records = _require_mapping(report.get("selected_checkpoint_records"), "Study-15 selected checkpoint records")
    for gate, record in selected_records.items():
        if gate not in {"visible_assignment", "invisible_generator", "invisible_coordinates", "assignment_formulation"}:
            continue
        _checkpoint_from_report(
            record,
            label=f"Study-15 selected checkpoint {gate}",
            expected_checkpoint_hashes=expected_checkpoint_hashes,
        )
    branches = _require_mapping(report.get("branches"), "Study-15 branches")
    conventional = _require_mapping(branches.get("conventional"), "Study-15 conventional branch")
    if conventional.get("training_performed") is not False or conventional.get("checkpoint_reused") is not True:
        raise StageReleaseError("Study-15 conventional branch does not prove locked-checkpoint reuse")
    conventional_checkpoint, _ = _checkpoint_from_report(
        conventional.get("checkpoint"),
        label="Study-15 conventional checkpoint",
        expected_checkpoint_hashes=expected_checkpoint_hashes,
    )
    coordinate_record = _require_mapping(selected_records.get("invisible_coordinates"), "Study-15 coordinate checkpoint")
    if conventional_checkpoint != Path(str(coordinate_record.get("path"))).expanduser().resolve():
        raise StageReleaseError("Study-15 conventional branch is not seed-matched to the Study-13 coordinate checkpoint")
    _validate_prediction_evidence(
        conventional.get("prediction"),
        "Study-15 conventional prediction",
        expected_posterior_samples=32,
    )
    _validate_evaluation_evidence(conventional.get("evaluation"), "Study-15 conventional evaluation")

    conditional = _require_mapping(branches.get("conditional_lgatr"), "Study-15 Conditional L-GATr branch")
    training = _require_mapping(conditional.get("training"), "Study-15 Conditional L-GATr training")
    for key in ("finite_initial_loss", "finite_gradients", "nonzero_invisible_update", "finite_validation_loss", "coordinate_inverse_check"):
        _require_true(training.get(key), f"Study-15 conditional training.{key}")
    _validate_training_split_evidence(training, "Study-15 conditional training")
    reload = _require_mapping(conditional.get("checkpoint_reload"), "Study-15 Conditional L-GATr reload")
    _require_true(reload.get("strict_reload"), "Study-15 conditional strict reload")
    _require_true(reload.get("frozen_visible_equal", {}).get("equal"), "Study-15 conditional frozen visible weights")
    _require_true(reload.get("frozen_assignment_equal", {}).get("equal"), "Study-15 conditional frozen assignment weights")
    if reload.get("backbone") != "lgatr":
        raise StageReleaseError("Study-15 reloaded conditional backbone is not L-GATr")
    checkpoint, _ = _checkpoint_from_report(
        conditional.get("checkpoint"),
        label="Study-15 Conditional L-GATr checkpoint",
        expected_checkpoint_hashes={str(Path(str(conditional["checkpoint"].get("path"))).expanduser().resolve()): str(conditional["checkpoint"].get("sha256"))},
        validation_root=validation_root,
    )
    if not _under(checkpoint, validation_root):
        raise StageReleaseError("Study-15 Conditional L-GATr checkpoint escaped the validation root")
    identity_path = Path(str(conditional["checkpoint"].get("identity_file", ""))).expanduser().resolve()
    if not _under(identity_path, validation_root) or not identity_path.is_file():
        raise StageReleaseError("Study-15 conditional checkpoint identity is missing/outside validation root")
    if sha256_file(identity_path) != str(conditional["checkpoint"].get("identity_sha256", "")):
        raise StageReleaseError("Study-15 conditional checkpoint identity hash is incorrect")
    if conditional["checkpoint"].get("training_sha") != expected_source_sha:
        raise StageReleaseError("Study-15 conditional checkpoint training SHA is not the release SHA")
    _validate_prediction_evidence(
        conditional.get("prediction"),
        "Study-15 conditional prediction",
        expected_posterior_samples=32,
    )
    _validate_evaluation_evidence(conditional.get("evaluation"), "Study-15 conditional evaluation")
    config_diff = _require_mapping(report.get("config_diff"), "Study-15 config difference")
    if config_diff.get("status") != "PASS" or config_diff.get("unexpected_differences") != []:
        raise StageReleaseError("Study-15 config comparison has unexpected differences")
    if report.get("conditional_lgatr_backbone", {}).get("type") != "lgatr":
        raise StageReleaseError("Study-15 report does not record a Conditional L-GATr backbone")
    _validate_nested_subresults(report, expected_source_sha=expected_source_sha, validation_root=validation_root)


def _validate_study10_evidence(
    report: dict,
    *,
    expected_source_sha: str,
    expected_checkpoint_hashes: dict[str, str],
    validation_root: Path,
) -> None:
    if report.get("schema") != "jer_study10_preflight_v1":
        raise StageReleaseError("Study-10 integration evidence has the wrong schema")
    selected = _require_mapping(report.get("selected_generator_checkpoint"), "Study-10 selected generator checkpoint")
    path = Path(str(selected.get("path", ""))).expanduser().resolve()
    digest = str(selected.get("sha256", ""))
    if expected_checkpoint_hashes.get(str(path)) != digest or not path.is_file() or sha256_file(path) != digest:
        raise StageReleaseError("Study-10 selected generator checkpoint is not the locked upstream artifact")
    subset = _require_mapping(report.get("data_subset"), "Study-10 data subset")
    for key in ("train_event_id_sha256", "validation_event_id_sha256"):
        if not subset.get(key):
            raise StageReleaseError(f"Study-10 data subset is missing {key}")
    _require_true(subset.get("zero_overlap"), "Study-10 train/validation event separation")
    _require_true(subset.get("test_split_sealed"), "Study-10 test split seal")
    shared = _require_mapping(report.get("shared_posterior"), "Study-10 shared posterior")
    if shared != {
        "branches_per_prediction": 1,
        "Kmax": 10,
        "Mmax": 64,
        "nested_prefixes": True,
        "generation_count_by_split": {"train": 1, "validation": 1},
        "reused_across_rerankers": True,
        "reused_across_K_M_prefixes": True,
    }:
        raise StageReleaseError("Study-10 shared-posterior evidence is incomplete or misleading")
    split = _require_mapping(report.get("scorer_split_contract"), "Study-10 scorer split contract")
    feature_producer_source_sha = str(
        report.get("feature_producer_source_sha") or expected_source_sha
    )
    if report.get("analysis_source_sha", report.get("source_sha")) != expected_source_sha:
        raise StageReleaseError("Study-10 analysis source identity is not the release SHA")
    if split.get("scaler_fit_source_split") != "train" or split.get("scorer_fit_source_split") != "train" or split.get("alpha_selection_source_split") != "train":
        raise StageReleaseError("Study-10 scorer/scaler fit source is not the training split")
    if not split.get("scaler_identity_sha256"):
        raise StageReleaseError("Study-10 scorer split contract has no scaler identity")
    _require_true(split.get("zero_event_overlap"), "Study-10 scorer event separation")
    _require_true(split.get("validation_used_after_fit"), "Study-10 validation-after-fit ordering")
    visible_contract = _require_mapping(split.get("visible_only_input_contract"), "Study-10 visible-only input contract")
    if visible_contract.get("source") != "visible_only_scorer_input" or visible_contract.get("visible_feature_columns") != 4:
        raise StageReleaseError("Study-10 visible-only scorer input source is not explicit")
    _require_true(visible_contract.get("physics_padding_zero"), "Study-10 visible-only physics padding")
    for key in ("training_feature_artifact", "validation_feature_artifact"):
        feature_path = Path(str(split.get(key, ""))).expanduser().resolve()
        if not _under(feature_path, validation_root) or not feature_path.is_file():
            raise StageReleaseError(f"Study-10 {key} is missing/outside validation root")
        if sha256_file(feature_path) != str(split.get(f"{key}_sha256", "")):
            raise StageReleaseError(f"Study-10 {key} hash is incorrect")
        with h5py.File(feature_path, "r") as source:
            split_name = str(source.attrs.get("split", ""))
            expected_split = "train" if key == "training_feature_artifact" else "validation"
            if split_name != expected_split:
                raise StageReleaseError(f"Study-10 feature artifact split is wrong: {feature_path}")
            if str(source.attrs.get("source_sha", "")) != feature_producer_source_sha or str(source.attrs.get("resolved_gate_digest", "")) != str(report.get("resolved_gate_digest")):
                raise StageReleaseError(f"Study-10 feature artifact source/gate identity is wrong: {feature_path}")
            if str(source.attrs.get("checkpoint_sha256", "")) != digest:
                raise StageReleaseError(f"Study-10 feature artifact checkpoint identity is wrong: {feature_path}")
            if bool(source.attrs.get("test_split_used", True)) or str(source.attrs.get("physics_scaler_fit_split", "")) != "train":
                raise StageReleaseError(f"Study-10 feature artifact has invalid split/scaler provenance: {feature_path}")
    with h5py.File(Path(str(split["training_feature_artifact"])), "r") as train_file, h5py.File(Path(str(split["validation_feature_artifact"])), "r") as validation_file:
        train_array = np.asarray(train_file["EVENT_ID"][:])
        validation_array = np.asarray(validation_file["EVENT_ID"][:])
        train_ids = set(train_array.tolist())
        validation_ids = set(validation_array.tolist())
        if train_ids & validation_ids:
            raise StageReleaseError("Study-10 feature artifacts contain overlapping event IDs")
        train_id_sha = hashlib.sha256(np.ascontiguousarray(train_array).tobytes()).hexdigest()
        validation_id_sha = hashlib.sha256(np.ascontiguousarray(validation_array).tobytes()).hexdigest()
        if train_id_sha != subset.get("train_event_id_sha256") or validation_id_sha != subset.get("validation_event_id_sha256"):
            raise StageReleaseError("Study-10 feature artifact EVENT_ID hashes disagree with the release report")
    scorer_artifacts = _require_mapping(report.get("scorer_artifacts"), "Study-10 scorer artifacts")
    for name in ("physics", "visible_only"):
        scorer = _require_mapping(scorer_artifacts.get(name), f"Study-10 {name} scorer")
        scorer_path = Path(str(scorer.get("path", ""))).expanduser().resolve()
        if not _under(scorer_path, validation_root) or not scorer_path.is_file() or sha256_file(scorer_path) != str(scorer.get("sha256", "")):
            raise StageReleaseError(f"Study-10 {name} scorer artifact is invalid")
    if report.get("controls") != sorted({"visible_baseline", "matched_visible_only_control", "analytic_physics", "learned_physics", "truth_neutrino_oracle"}):
        raise StageReleaseError("Study-10 required controls are incomplete")
    nested = _require_mapping(report.get("nested_prefix_contract"), "Study-10 nested prefix contract")
    if nested.get("K_prefixes") != [1, 2, 3, 5, 10] or nested.get("M_prefixes") != [4, 8, 16, 32, 64] or nested.get("prefixes_from_single_posterior") is not True:
        raise StageReleaseError("Study-10 nested K/M prefix contract is incomplete")
    analysis = _require_mapping(report.get("analysis"), "Study-10 analysis")
    if not {"visible_baseline", "matched_visible_only_control", "analytic_physics", "learned_physics", "truth_neutrino_oracle"}.issubset(_require_mapping(analysis.get("comparisons"), "Study-10 comparisons")):
        raise StageReleaseError("Study-10 analysis is missing a required control")
    if report.get("test_accessed") is not False or report.get("source_sha") != expected_source_sha:
        raise StageReleaseError("Study-10 integration evidence has invalid source/test identity")


def _validate_study03_evidence(
    report: dict,
    *,
    expected_source_sha: str,
    expected_checkpoint_hashes: dict[str, str],
    validation_root: Path,
) -> None:
    if report.get("schema") != "jer_study03_integration_v1":
        raise StageReleaseError("Study-03 integration evidence has the wrong schema")
    if report.get("selected_coupling") in {None, "none"}:
        raise StageReleaseError("Study-03 release was produced without a deployable coupling")
    anti = _require_mapping(report.get("anti_leakage"), "Study-03 anti-leakage evidence")
    if anti.get("refined_conditioning") != "hypothesis" or anti.get("truth_assignment_in_production_path") is not False or anti.get("truth_oracle_mode") is not False:
        raise StageReleaseError("Study-03 truth-assignment leakage contract is incomplete")
    semantics = _require_mapping(
        report.get("conditioning_semantics"), "Study-03 conditioning semantics"
    )
    if semantics.get("deployable_inference_assignment_source") != "decoded_predicted_assignment":
        raise StageReleaseError("Study-03 deployable inference does not use decoded predicted assignments")
    if semantics.get("truth_assignment_in_deployable_inference") is not False:
        raise StageReleaseError("Study-03 deployable inference exposes the truth assignment")
    if (
        semantics.get("training_truth_teacher_forcing") is True
        or semantics.get("training_assignment_source") == "structured.pools[*].truth_index"
        or semantics.get("incorrect_predicted_assignments_in_training") is not True
        or semantics.get("scientific_review_required") is True
    ):
        raise StageReleaseReviewRequired(
            "Study-03 requires scientific review: refined training is truth-teacher-forced "
            "and does not expose realistic incorrect predicted assignments"
        )
    context = _require_mapping(report.get("conditioning_context"), "Study-03 conditioning context evidence")
    for key in ("distinct_predicted_hypotheses", "context_changes_with_predicted_assignment", "context_independent_of_truth_assignment"):
        _require_true(context.get(key), f"Study-03 conditioning context.{key}")
    branches = _require_mapping(report.get("branches"), "Study-03 branches")
    for name in ("none", "refined"):
        branch = _require_mapping(branches.get(name), f"Study-03 {name} branch")
        _validate_prediction_evidence(
            branch.get("prediction"),
            f"Study-03 {name} prediction",
            expected_posterior_samples=32,
        )
        _validate_evaluation_evidence(branch.get("evaluation"), f"Study-03 {name} evaluation")
    refined_training = _require_mapping(branches["refined"].get("training"), "Study-03 refined training")
    for key in ("finite_initial_loss", "finite_gradients", "nonzero_invisible_update", "coordinate_inverse_check"):
        _require_true(refined_training.get(key), f"Study-03 refined training.{key}")
    _validate_training_split_evidence(refined_training, "Study-03 refined training")
    reload = _require_mapping(branches["refined"].get("checkpoint_reload"), "Study-03 refined reload")
    _require_true(reload.get("strict_reload"), "Study-03 strict reload")
    _require_true(reload.get("frozen_visible_equal", {}).get("equal"), "Study-03 frozen visible weights")
    _require_true(reload.get("frozen_assignment_equal", {}).get("equal"), "Study-03 frozen assignment weights")
    generator = _require_mapping(report.get("selected_generator_checkpoint"), "Study-03 selected generator checkpoint")
    path = Path(str(generator.get("path", ""))).expanduser().resolve()
    digest = str(generator.get("sha256", ""))
    if expected_checkpoint_hashes.get(str(path)) != digest or not path.is_file() or sha256_file(path) != digest:
        raise StageReleaseError("Study-03 selected generator checkpoint is not the locked upstream artifact")
    if report.get("test_accessed") is not False or report.get("source_sha") != expected_source_sha:
        raise StageReleaseError("Study-03 integration evidence has invalid source/test identity")
    _validate_nested_subresults(report, expected_source_sha=expected_source_sha, validation_root=validation_root)


def _validate_stage_evidence(
    report: dict,
    *,
    expected_schema: str,
    expected_source_sha: str,
    expected_checkpoint_hashes: dict[str, str],
    validation_root: Path,
) -> None:
    validators = {
        "study15_release_v1": _validate_study15_evidence,
        "study10_release_v1": _validate_study10_evidence,
        "study03_release_v1": _validate_study03_evidence,
    }
    validator = validators.get(expected_schema)
    if validator is None:
        raise StageReleaseError(f"No explicit evidence validator is registered for {expected_schema}")
    subresults = _require_mapping(report.get("subresults"), "release subresults")
    integration_declaration = _require_mapping(subresults.get("integration"), "release integration subresult")
    integration_path = Path(str(integration_declaration.get("path", "")))
    if integration_declaration.get("schema") not in {
        "jer_study15_integration_v1",
        "jer_study10_preflight_v1",
        "jer_study03_integration_v1",
    }:
        raise StageReleaseError("release integration subresult does not use a maintained stage schema")
    integration = _read_json(integration_path)
    validator(
        integration,
        expected_source_sha=expected_source_sha,
        expected_checkpoint_hashes=expected_checkpoint_hashes,
        validation_root=validation_root,
    )


def validate_release_payload(
    payload: dict,
    *,
    artifact_path: Path,
    expected_schema: str,
    expected_source_sha: str,
    expected_campaign_id: str,
    expected_gate_digest: str,
    expected_manifest_sha256: str,
    expected_checkpoint_hashes: dict[str, str],
    validation_root: Path,
) -> dict:
    """Validate an executed stage artifact and all declared sub-results."""
    if payload.get("schema") != expected_schema:
        raise StageReleaseError(
            f"unexpected release schema: {payload.get('schema')!r}; expected {expected_schema!r}"
        )
    if payload.get("status") != "PASS":
        raise StageReleaseError("release artifact is not PASS")
    if payload.get("software_integration_evidence_only") is not True:
        raise StageReleaseError("release artifact is not marked software-integration evidence")
    for key, expected in (
        ("source_sha", expected_source_sha),
        ("campaign_id", expected_campaign_id),
        ("upstream_gate_digest", expected_gate_digest),
        ("manifest_sha256", expected_manifest_sha256),
    ):
        if payload.get(key) != expected:
            raise StageReleaseError(
                f"release {key} mismatch: expected={expected!r}, found={payload.get(key)!r}"
            )
    job_id = str(payload.get("slurm_job_id", ""))
    if not job_id.isdigit():
        raise StageReleaseError("release artifact has no numeric Slurm job ID")
    if payload.get("slurm_terminal_state") != "COMPLETED":
        raise StageReleaseError("release artifact does not record COMPLETED terminal state")
    if payload.get("slurm_exit_code") != "0:0":
        raise StageReleaseError("release artifact does not record exit code 0:0")

    required = payload.get("required_components")
    if not isinstance(required, dict) or not required:
        raise StageReleaseError("release required_components is missing")
    failed = sorted(name for name, value in required.items() if value is not True)
    if failed:
        raise StageReleaseError(f"release required components failed: {failed}")

    declared = payload.get("checkpoint_hashes")
    if not isinstance(declared, dict):
        raise StageReleaseError("release checkpoint_hashes is missing")
    for path_text, expected_hash in expected_checkpoint_hashes.items():
        if declared.get(path_text) != expected_hash:
            raise StageReleaseError(f"selected checkpoint is absent or mismatched: {path_text}")
    for path_text, expected_hash in declared.items():
        path = Path(str(path_text)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise StageReleaseError(f"release checkpoint hash does not match file: {path}")

    if not _under(artifact_path, validation_root):
        raise StageReleaseError(f"release artifact is outside the validation root: {artifact_path}")
    for path, schema in _subresult_paths(payload):
        path = path.expanduser().resolve()
        if not _under(path, validation_root):
            raise StageReleaseError(f"subresult is outside the validation root: {path}")
        if not path.is_file():
            raise StageReleaseError(f"declared subresult is missing: {path}")
        subresult = _read_json(path)
        if subresult.get("status") != "PASS":
            raise StageReleaseError(f"subresult is not PASS: {path}")
        if schema is not None and subresult.get("schema") != schema:
            raise StageReleaseError(
                f"subresult schema mismatch for {path}: {subresult.get('schema')!r}"
            )
        if subresult.get("source_sha") != expected_source_sha:
            raise StageReleaseError(f"subresult source SHA mismatch: {path}")
        if subresult.get("test_accessed") is True:
            raise StageReleaseError(f"subresult accessed the sealed test split: {path}")

    _validate_stage_evidence(
        payload,
        expected_schema=expected_schema,
        expected_source_sha=expected_source_sha,
        expected_checkpoint_hashes=expected_checkpoint_hashes,
        validation_root=validation_root,
    )

    return {
        "schema": expected_schema,
        "status": "PASS",
        "slurm_job_id": job_id,
        "validated_checkpoint_count": len(declared),
        "validated_subresult_count": len(payload["subresults"]),
    }
