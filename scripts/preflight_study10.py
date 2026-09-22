#!/usr/bin/env python3
"""Run the bounded Study-10 scorer/control preflight.

The expensive generator is predicted once at K=10,M=64.  The maintained
Study-10 producer then creates training/validation feature artifacts, fits the
two scorers only on training events, reloads them, and evaluates all controls
and nested prefixes on the independent validation events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.runtime_paths import runtime_paths  # noqa: E402
from joint_event_reconstruction.studies import resolve_study  # noqa: E402
from scripts.run_study import _gate_checkpoint, _public_run_config, _write_yaml  # noqa: E402
from scripts.run_study10_controls import (  # noqa: E402
    K_PREFIXES,
    M_PREFIXES,
    _analyse_seed,
    _build_one_seed,
    _generate_prediction,
    _input_tensors,
    _load_features,
    _read_prediction,
    visible_only_scorer_input,
)


STUDY_PATH = ROOT / "configs/studies/10_compatibility_reranking.yaml"
SMALL_OVERRIDES = (
    "data.events.train=32",
    "data.events.validation=16",
    "data.events.test=1",
    "training.workers=0",
    "inference.seed=424242",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _feature_event_ids(path: Path) -> tuple[np.ndarray, bool, str]:
    with h5py.File(path, "r") as source:
        event_ids = source["EVENT_ID"][:]
        test_used = bool(source.attrs.get("test_split_used", True))
        split = str(source.attrs.get("split", ""))
    return event_ids, test_used, split


def run(*, output_dir: Path, device: str, gates_path: Path, expected_source_sha: str, campaign_id: str) -> dict:
    paths = runtime_paths()
    if paths.source_sha != expected_source_sha:
        raise RuntimeError(f"SOURCE SHA MISMATCH: expected {expected_source_sha}, found {paths.source_sha}")
    study = resolve_study(STUDY_PATH, gates_path)
    if study.gates_digest is None:
        raise RuntimeError("Study-10 has no resolved gate digest")
    output_dir.mkdir(parents=True, exist_ok=True)
    variant = study.variant_map["compatibility"]
    checkpoint = _gate_checkpoint(study, "invisible_backbone", "ttbar2L", 42)
    _, public, resolved = _public_run_config(
        study, "ttbar2L", variant, 42, output_dir, SMALL_OVERRIDES
    )
    experiment = output_dir / "compatibility_experiment.yaml"
    _write_yaml(experiment, public)
    if str(resolved.model.invisible.coordinates.type) != "met_residual":
        raise RuntimeError("Study-10 did not inherit the selected Study-13 coordinates")

    validation_prediction = _generate_prediction(
        experiment, checkpoint, output_dir / "validation_prediction.h5", "validation", device
    )
    training_prediction = _generate_prediction(
        experiment, checkpoint, output_dir / "training_prediction.h5", "train", device
    )
    validation_record = _read_prediction(validation_prediction)
    training_record = _read_prediction(training_prediction)
    if validation_record.branch_count != 1 or training_record.branch_count != 1:
        raise RuntimeError("Study-10 preflight generated more than one expensive posterior branch")
    if validation_record.posterior.shape[1:3] != (10, 64):
        raise RuntimeError("Study-10 preflight did not produce K=10,M=64 posterior")
    if training_record.posterior.shape[1:3] != (10, 64):
        raise RuntimeError("Study-10 training features did not use K=10,M=64 posterior")
    train_ids = np.asarray(training_record.event_id)
    validation_ids = np.asarray(validation_record.event_id)
    if set(train_ids.tolist()) & set(validation_ids.tolist()):
        raise RuntimeError("Study-10 scorer train/validation event IDs overlap")

    feature_dir = output_dir / "feature_artifacts"
    train_features, validation_features = _build_one_seed(
        validation_prediction,
        training_prediction,
        experiment,
        checkpoint,
        feature_dir / "seed_42_placeholder.h5",
        device,
        64,
        42,
        _sha256(STUDY_PATH),
        study.gates_digest,
    )
    train_feature_ids, train_test, train_split = _feature_event_ids(train_features)
    validation_feature_ids, validation_test, validation_split = _feature_event_ids(validation_features)
    if train_test or validation_test or train_split != "train" or validation_split != "validation":
        raise RuntimeError("Study-10 feature artifacts have an invalid split/test contract")
    if set(train_feature_ids.tolist()) & set(validation_feature_ids.tolist()):
        raise RuntimeError("Study-10 feature artifacts overlap between scorer train and validation")
    analysis = _analyse_seed(
        train_features,
        validation_features,
        output_dir,
        epochs=1,
        seed=42,
        experiment=experiment,
    )
    with h5py.File(validation_features, "r") as source:
        nested = source["PHYSICS/nested_analytic_scores"][:]
        valid_hypotheses = source["valid_hypotheses"][:].astype(bool, copy=False)
        k_prefixes = json.loads(source["PHYSICS"].attrs["K_prefixes"])
        m_prefixes = json.loads(source["PHYSICS"].attrs["M_prefixes"])
        if nested.shape[:2] != (len(K_PREFIXES), len(M_PREFIXES)):
            raise RuntimeError(f"Study-10 nested K/M product has the wrong shape: {nested.shape}")
        if np.isnan(nested).any() or np.isposinf(nested).any():
            raise RuntimeError("Study-10 nested K/M product contains NaN or positive infinity")
        for k_index, k in enumerate(K_PREFIXES):
            active = valid_hypotheses[:, :k]
            for m_index in range(len(M_PREFIXES)):
                if not np.isfinite(nested[k_index, m_index, :, :k][active]).all():
                    raise RuntimeError(
                        f"Study-10 nested K/M product is non-finite for valid hypotheses: K={k}"
                    )
        if k_prefixes != list(K_PREFIXES) or m_prefixes != list(M_PREFIXES):
            raise RuntimeError(
                f"Study-10 nested K/M prefixes are not the declared prefixes: {k_prefixes}, {m_prefixes}"
            )
        feature_names = json.loads(source.attrs["feature_names"])
    if not feature_names:
        raise RuntimeError("Study-10 physics feature schema is empty")
    train_features_payload = _load_features(train_features)
    validation_features_payload = _load_features(validation_features)
    train_physics, train_visible = _input_tensors(train_features_payload)
    validation_physics, validation_visible = _input_tensors(validation_features_payload)
    if train_physics.shape[-1] <= 4 or validation_physics.shape[-1] != train_physics.shape[-1]:
        raise RuntimeError("Study-10 scorer feature widths are not comparable")
    expected_visible = visible_only_scorer_input(
        train_features_payload["visible_log_score"].float(),
        train_features_payload["visible_probability"].float(),
        train_features_payload["entropy"].float(),
        train_features_payload["margin"].float(),
        target_features=train_physics.shape[-1],
    )
    if not torch.equal(train_visible, expected_visible) or not torch.equal(
        train_visible[..., 4:], torch.zeros_like(train_visible[..., 4:])
    ):
        raise RuntimeError("Study-10 visible-only control input is not physics-blind")
    comparisons = analysis["comparisons"]
    required_controls = {"visible_baseline", "matched_visible_only_control", "analytic_physics", "learned_physics", "truth_neutrino_oracle"}
    if not required_controls <= set(comparisons):
        raise RuntimeError("Study-10 preflight is missing one or more scientific controls")
    if not Path(analysis["train_feature_artifact"]).is_file() or not Path(analysis["validation_feature_artifact"]).is_file():
        raise RuntimeError("Study-10 scorer feature artifacts were not retained")

    review = {
        "schema": "jer_study10_preflight_v1",
        "status": "PASS",
        "software_integration_evidence_only": True,
        "source_sha": expected_source_sha,
        "analysis_source_sha": expected_source_sha,
        "feature_producer_source_sha": train_features_payload["feature_producer_source_sha"],
        "campaign_id": campaign_id,
        "study": "compatibility_reranking",
        "study_manifest_sha256": _sha256(STUDY_PATH),
        "resolved_gate_digest": study.gates_digest,
        "selected_generator_checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256(checkpoint),
        },
        "data_subset": {
            "train_events": int(len(train_ids)),
            "validation_events": int(len(validation_ids)),
            "train_event_id_sha256": hashlib.sha256(np.ascontiguousarray(train_ids).tobytes()).hexdigest(),
            "validation_event_id_sha256": hashlib.sha256(np.ascontiguousarray(validation_ids).tobytes()).hexdigest(),
            "zero_overlap": True,
            "test_split_sealed": True,
        },
        "shared_posterior": {
            "branches_per_prediction": 1,
            "Kmax": 10,
            "Mmax": 64,
            "nested_prefixes": True,
            "generation_count_by_split": {"train": 1, "validation": 1},
            "reused_across_rerankers": True,
            "reused_across_K_M_prefixes": True,
        },
        "scorer_split_contract": {
            "training_feature_artifact": str(train_features),
            "validation_feature_artifact": str(validation_features),
            "training_feature_artifact_sha256": _sha256(train_features),
            "validation_feature_artifact_sha256": _sha256(validation_features),
            "scaler_fit_source_split": "train",
            "scaler_identity_sha256": str(
                train_features_payload.get("physics_scaler_sha256", "")
            ),
            "fit_event_id_sha256": hashlib.sha256(np.ascontiguousarray(train_feature_ids).tobytes()).hexdigest(),
            "validation_event_id_sha256": hashlib.sha256(np.ascontiguousarray(validation_feature_ids).tobytes()).hexdigest(),
            "zero_event_overlap": True,
            "scorer_fit_source_split": "train",
            "alpha_selection_source_split": "train",
            "validation_used_after_fit": True,
            "visible_only_input_contract": {
                "source": "visible_only_scorer_input",
                "visible_feature_columns": 4,
                "physics_padding_zero": True,
            },
            "feature_names": feature_names,
        },
        "scorer_artifacts": {
            "physics": {
                "path": str(output_dir / "scorers/seed_42_physics.pt"),
                "sha256": _sha256(output_dir / "scorers/seed_42_physics.pt"),
            },
            "visible_only": {
                "path": str(output_dir / "scorers/seed_42_visible_only.pt"),
                "sha256": _sha256(output_dir / "scorers/seed_42_visible_only.pt"),
            },
        },
        "controls": sorted(required_controls),
        "nested_prefix_contract": {
            "K_prefixes": list(K_PREFIXES),
            "M_prefixes": list(M_PREFIXES),
            "underlying_Kmax": 10,
            "underlying_Mmax": 64,
            "prefixes_from_single_posterior": True,
        },
        "analysis": analysis,
        "required_components": {
            "selected_generator_loaded": True,
            "shared_posterior": True,
            "scorer_train_validation_separation": True,
            "visible_only_control": True,
            "analytic_control": True,
            "learned_scorer_train_reload": True,
            "truth_neutrino_oracle": True,
            "nested_K_M_prefixes": True,
            "test_split_sealed": True,
        },
        "test_accessed": False,
    }
    _write_json(output_dir / "result.json", review)
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gates-path", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(run(output_dir=args.output_dir.expanduser().resolve(), device=args.device, gates_path=args.gates_path.expanduser().resolve(), expected_source_sha=args.expected_source_sha, campaign_id=args.campaign_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
