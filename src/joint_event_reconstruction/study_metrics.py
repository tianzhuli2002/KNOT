"""Metrics used specifically for controlled cross-formulation studies."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.prediction import PREDICTION_SCHEMA
from joint_event_reconstruction.prediction_discrete import DISCRETE_PREDICTION_SCHEMA
from joint_event_reconstruction.prediction_joint import JOINT_PREDICTION_SCHEMA


def _mode(states: np.ndarray) -> np.ndarray:
    result = np.empty(len(states), dtype=states.dtype)
    for event, samples in enumerate(states):
        values, counts = np.unique(samples, return_counts=True)
        result[event] = values[np.argmax(counts)]
    return result


def _binomial_error(value: float, count: int) -> float:
    return float(np.sqrt(value * (1.0 - value) / count)) if count else float("nan")


def common_non_null_truth_assignment(path: str | Path) -> dict:
    """Evaluate assignment on events whose truth is non-NULL for every root.

    This is the one event subset shared by the discriminative non-NULL decoder and the
    NULL-including DFM/joint clean support. It complements rather than replaces each
    formulation's native-support metric.
    """
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as source:
        schema = str(source.attrs["schema_version"])
        if schema == PREDICTION_SCHEMA:
            roots = json.loads(source.attrs["root_order"])
            eligible = np.ones(len(source["EVENT_ID"]), dtype=bool)
            for root in roots:
                eligible &= source[f"VISIBLE/nodes/{root}/reconstructable"][:].astype(bool)
            correct = source["TRUTH/topk_assignment"][:, 0].astype(bool)
            estimator = "exact_constrained_top1"
        elif schema in {DISCRETE_PREDICTION_SCHEMA, JOINT_PREDICTION_SCHEMA}:
            truth_roots = source["TRUTH/root_candidate_indices"][:]
            eligible = (truth_roots >= 0).all(axis=1)
            states = source["HYPOTHESES/assignment_state"][:]
            truth_state = source["TRUTH/assignment_state"][:]
            correct = _mode(states) == truth_state
            estimator = "empirical_posterior_mode"
        else:
            raise ValueError(f"Unsupported prediction schema {schema!r}")

        count = int(eligible.sum())
        accuracy = float(correct[eligible].mean()) if count else None
        return {
            "definition": "truth_complete_non_null_for_every_root",
            "events": count,
            "total_events": int(len(eligible)),
            "eligible_fraction": float(eligible.mean()),
            "accuracy": accuracy,
            "standard_error": (
                _binomial_error(accuracy, count) if accuracy is not None else None
            ),
            "estimator": estimator,
            "native_support_is_not_replaced": True,
        }


def assert_same_events(paths: list[str | Path]) -> None:
    """Require identical ordered EVENT_ID arrays before cross-model event comparisons."""
    if len(paths) < 2:
        return
    reference_path = Path(paths[0]).expanduser().resolve()
    with h5py.File(reference_path, "r") as source:
        reference = source["EVENT_ID"][:]
    for value in paths[1:]:
        path = Path(value).expanduser().resolve()
        with h5py.File(path, "r") as source:
            observed = source["EVENT_ID"][:]
        if not np.array_equal(observed, reference):
            raise ValueError(
                "Cross-model study comparison requires identical ordered EVENT_ID arrays: "
                f"reference={reference_path}, mismatch={path}"
            )
