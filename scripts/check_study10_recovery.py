#!/usr/bin/env python3
"""Focused checks for resumable Study-10 postprocessing."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.studies import resolve_study  # noqa: E402
import scripts.run_study10_controls as controls  # noqa: E402
from scripts.run_study10_controls import (  # noqa: E402
    _feature_artifact_contract,
    _seed_record_contract,
    _write_review,
)
from scripts.submit_campaign import _outputs_complete  # noqa: E402


def main() -> None:
    base = ROOT / "results/validation/stage-release/32d3b7d2ad86fb3ec22c6adc3a1ead96ede8b36b/assignment_coupling/attempt-001"
    result = json.loads((base / "result.json").read_text())
    analysis = result["analysis"]
    checkpoint = Path(result["selected_generator_checkpoint"]["path"])
    train_contract = _feature_artifact_contract(
        Path(analysis["train_feature_artifact"]),
        expected_split="train",
        expected_feature_producer_sha=result["source_sha"],
    )
    validation_contract = _feature_artifact_contract(
        Path(analysis["validation_feature_artifact"]),
        expected_split="validation",
        expected_feature_producer_sha=result["source_sha"],
    )
    record = copy.deepcopy(analysis)
    record.update({
        "seed": 42,
        "scorer_training_epochs": int(record.get("scorer_training_epochs", 1)),
        "source_sha": result["source_sha"],
        "analysis_source_sha": result["source_sha"],
        "feature_producer_source_sha": result["source_sha"],
        "study_manifest_sha256": result["study_manifest_sha256"],
        "resolved_gate_digest": result["resolved_gate_digest"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": result["selected_generator_checkpoint"]["sha256"],
        "checkpoint_training_sha": "f2084c8b2308a7567cec8a269520c32dff43638f",
        "scorer_artifacts": result["scorer_artifacts"],
        "training_events_total": train_contract["events_total"],
        "training_events_supported": train_contract["events_supported"],
        "training_events_unsupported": train_contract["events_unsupported"],
        "training_supported_fraction": train_contract["events_supported"] / train_contract["events_total"],
        "validation_events_total": validation_contract["events_total"],
        "validation_events_supported": validation_contract["events_supported"],
        "validation_events_unsupported": validation_contract["events_unsupported"],
        "validation_supported_fraction": validation_contract["events_supported"] / validation_contract["events_total"],
    })
    with tempfile.TemporaryDirectory(prefix="jer-study10-recovery-contract-") as directory:
        root = Path(directory)
        record_path = root / "seed_42.json"
        record_path.write_text(json.dumps(record))
        checked = _seed_record_contract(
            record_path,
            expected_seed=42,
            expected_source_sha=result["source_sha"],
            expected_manifest_sha256=result["study_manifest_sha256"],
            expected_gate_digest=result["resolved_gate_digest"],
            expected_feature_producer_sha=result["source_sha"],
            expected_checkpoint=checkpoint,
            expected_checkpoint_sha256=result["selected_generator_checkpoint"]["sha256"],
        )
        if checked["seed"] != 42:
            raise AssertionError("valid bounded Study-10 seed record was not reused")
        try:
            _write_review(
                root / "study10_human_review",
                [checked],
                result["study_manifest_sha256"],
                result["resolved_gate_digest"],
            )
        except ValueError:
            pass
        else:
            raise AssertionError("finalisation accepted an incomplete seed set")

        truncated = root / "truncated.h5"
        source_feature = Path(analysis["train_feature_artifact"])
        truncated.write_bytes(source_feature.read_bytes()[:1024])
        try:
            _feature_artifact_contract(
                truncated,
                expected_split="train",
                expected_source_sha=result["source_sha"],
                expected_feature_producer_sha=result["source_sha"],
            )
        except (OSError, ValueError):
            pass
        else:
            raise AssertionError("truncated feature artifact was accepted")

        mismatched = copy.deepcopy(record)
        mismatched["source_sha"] = "0" * 40
        mismatch_path = root / "mismatched.json"
        mismatch_path.write_text(json.dumps(mismatched))
        try:
            _seed_record_contract(
                mismatch_path,
                expected_seed=42,
                expected_source_sha=result["source_sha"],
                expected_manifest_sha256=result["study_manifest_sha256"],
                expected_gate_digest=result["resolved_gate_digest"],
                expected_feature_producer_sha=result["source_sha"],
            )
        except ValueError:
            pass
        else:
            raise AssertionError("source-mismatched seed record was accepted")

        feature_root = root / "feature_artifacts"
        feature_root.mkdir()
        shutil.copy2(source_feature, feature_root / "seed_42_train_features.h5")
        shutil.copy2(
            Path(analysis["validation_feature_artifact"]),
            feature_root / "seed_42_validation_features.h5",
        )
        reuse_output = root / "study10_human_review"
        reuse_record = root / "reuse_seed_42.json"
        with patch.object(controls, "_build_one_seed", side_effect=AssertionError("feature rebuild")), patch.object(
            controls, "_generate_prediction", side_effect=AssertionError("prediction regeneration")
        ):
            reused = controls._run_one_seed(
                prediction=root / "unused-prediction.h5",
                training_prediction=None,
                experiment=base / "compatibility_experiment.yaml",
                checkpoint=checkpoint,
                output=reuse_output,
                device="cpu",
                chunk_size=64,
                epochs=1,
                seed=42,
                study_manifest_sha256=result["study_manifest_sha256"],
                gate_digest=result["resolved_gate_digest"],
                feature_producer_source_sha=result["source_sha"],
                seed_record=reuse_record,
            )
        if reused["feature_producer_source_sha"] != result["source_sha"]:
            raise AssertionError("Existing feature producer identity was not retained")
        with patch.object(controls, "_analyse_seed", side_effect=AssertionError("reanalysis")), patch.object(
            controls, "_build_one_seed", side_effect=AssertionError("feature rebuild")
        ):
            resumed = controls._run_one_seed(
                prediction=root / "unused-prediction.h5",
                training_prediction=None,
                experiment=base / "compatibility_experiment.yaml",
                checkpoint=checkpoint,
                output=reuse_output,
                device="cpu",
                chunk_size=64,
                epochs=1,
                seed=42,
                study_manifest_sha256=result["study_manifest_sha256"],
                gate_digest=result["resolved_gate_digest"],
                feature_producer_source_sha=result["source_sha"],
                seed_record=reuse_record,
            )
        if resumed["seed"] != 42:
            raise AssertionError("Complete Study-10 seed record was not skipped")

    submission = json.loads(
        (ROOT / "results/studies/compatibility_reranking/_slurm/paper1-assignment_coupling-a2/submission.json").read_text()
    )
    study = resolve_study(submission["study_manifest_snapshot"], submission["gates_path"])
    complete, reason = _outputs_complete(study, submission)
    if complete or "study10_human_review" not in reason:
        raise AssertionError("incomplete Study-10 dedicated analysis was not reported")
    print("PASS: resumable Study-10 seed and completion contracts")


if __name__ == "__main__":
    main()
