#!/usr/bin/env python3
"""Scan compatibility reranking weights from one stored posterior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_recurrent_depth import _paired
from joint_event_reconstruction.prediction_schema import (
    PREDICTION_SCHEMA,
    validate_prediction,
)


WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _ranking(assignment: np.ndarray, compatibility: np.ndarray, weight: float) -> np.ndarray:
    """Match prediction.py's stable descending reranking exactly."""
    # Prediction HDF5 stores both terms as float32 and production adds them
    # before calling stable argsort.  Keep that arithmetic, including near
    # ties, identical in the offline scan.
    score = assignment.astype(np.float32) + np.float32(weight) * compatibility.astype(
        np.float32
    )
    return np.argsort(
        -score, axis=1, kind="stable"
    )


def analyze(path: Path) -> dict:
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema_version") != PREDICTION_SCHEMA:
            raise ValueError("Compatibility scan requires a factorized prediction")
        if source.attrs.get("split") != "validation":
            raise ValueError(
                "Compatibility scan is a validation-selection diagnostic; "
                f"expected split='validation' in {path}, found {source.attrs.get('split')!r}"
            )
        if "JOINT" not in source:
            raise ValueError("Compatibility scan requires a JOINT compatibility group")
        assignment = source["VISIBLE/assignment_log_scores"][:].astype(np.float64)
        compatibility = source["JOINT/compatibility_logit"][:].astype(np.float64)
        truth = source["TRUTH/topk_assignment"][:].astype(bool)
        events = source["EVENT_ID"][:]
        if assignment.shape[1] < 2:
            raise ValueError(
                "Compatibility scan requires at least two stored assignment hypotheses"
            )
        stored_weight = float(source["JOINT"].attrs["compatibility_weight"])
        stored_ranking = source["JOINT/rank_indices"][:]
    if assignment.shape != compatibility.shape or truth.shape != assignment.shape:
        raise ValueError("Stored assignment, compatibility and truth arrays must match")
    if not np.isfinite(assignment).all() or not np.isfinite(compatibility).all():
        raise ValueError("Compatibility scan inputs must be finite")
    if len(events) == 0 or len(np.unique(events)) != len(events):
        raise ValueError("Compatibility scan requires non-empty unique EVENT_ID values")
    if stored_weight != 0.0:
        raise ValueError(
            "Offline compatibility scan requires a posterior generated with weight=0"
        )
    baseline_ranking = _ranking(assignment, compatibility, 0.0)
    if not np.array_equal(stored_ranking, baseline_ranking):
        raise ValueError(
            "Stored weight=0 ranking does not match the maintained production reranker"
        )
    baseline = truth[np.arange(len(truth)), baseline_ranking[:, 0]]
    scan = {}
    for index, weight in enumerate(WEIGHTS):
        ranking = _ranking(assignment, compatibility, weight)
        selected = ranking[:, 0]
        correct = truth[np.arange(len(truth)), selected]
        scan[str(weight)] = {
            "accuracy": float(correct.mean()),
            "selected_hypothesis_index_mean": float(selected.mean()),
            **_paired(baseline, correct, seed=13000 + index),
        }
    return {
        "prediction": str(path),
        "events": len(events),
        "event_id_unique": True,
        "selection_split": "validation",
        "stored_generation_weight": stored_weight,
        "production_weight_zero_ranking_matches": True,
        "compatibility_is_not_a_normalized_likelihood": True,
        "assignment_only_is_always_reported": True,
        "scan": scan,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.prediction.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
