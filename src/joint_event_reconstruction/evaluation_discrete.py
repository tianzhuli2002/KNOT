"""Metrics for assignment-only Discrete Flow Matching posterior samples."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.assignment_semantics import read_hdf5_support
from joint_event_reconstruction.prediction_schema import (
    DISCRETE_PREDICTION_SCHEMA,
    validate_discrete_prediction,
)


def _empirical_mode(values: np.ndarray) -> np.ndarray:
    modes = np.empty(values.shape[0], dtype=values.dtype)
    for event, samples in enumerate(values):
        unique, counts = np.unique(samples, return_counts=True)
        modes[event] = unique[np.argmax(counts)]
    return modes


def _binomial_error(value: float, count: int) -> float:
    return float(np.sqrt(value * (1.0 - value) / count)) if count else float("nan")


def _categorical_metrics(samples: np.ndarray, truth: np.ndarray) -> dict:
    mode = _empirical_mode(samples)
    hit = samples == truth[:, None]
    accuracy = float((mode == truth).mean())
    return {
        "events": int(len(samples)),
        "empirical_mode_accuracy": accuracy,
        "empirical_mode_accuracy_standard_error": _binomial_error(
            accuracy, len(samples)
        ),
        "truth_sample_frequency": float(hit.mean()),
        "truth_sample_coverage": float(hit.any(axis=1).mean()),
        "unique_values_mean": float(
            np.mean([len(np.unique(event)) for event in samples])
        ),
        "probabilities_are_monte_carlo_estimates": True,
    }


def evaluate_discrete_predictions(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    validate_discrete_prediction(path)
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema_version") != DISCRETE_PREDICTION_SCHEMA:
            raise ValueError(f"Not a discrete-flow prediction file: {path}")
        support = read_hdf5_support(source)
        states = source["HYPOTHESES/assignment_state"][:]
        truth_state = source["TRUTH/assignment_state"][:]
        node_samples = source["HYPOTHESES/node_candidate_indices"][:]
        truth_nodes = source["TRUTH/node_candidate_indices"][:]
        njet = source["OBSERVED/njet"][:]
        nodes = json.loads(source.attrs["node_order"])

        assignment = _categorical_metrics(states, truth_state)
        state_mode = _empirical_mode(states)
        by_njet = {}
        for value in np.unique(njet):
            mask = njet == value
            count = int(mask.sum())
            accuracy = float((state_mode[mask] == truth_state[mask]).mean())
            by_njet[str(int(value))] = {
                "events": count,
                "accuracy": accuracy,
                "standard_error": _binomial_error(accuracy, count),
            }
        assignment["mode_accuracy_vs_njet"] = by_njet
        assignment["nodes"] = {
            node: _categorical_metrics(
                node_samples[:, :, index], truth_nodes[:, index]
            )
            for index, node in enumerate(nodes)
        }

        return {
            "prediction_file": str(path),
            "events": len(states),
            "topology_signature": str(source.attrs["topology_signature"]),
            "architecture_signature_version": int(
                source.attrs["architecture_signature_version"]
            ),
            "architecture_signature": str(source.attrs["architecture_signature"]),
            "model_mode": "factorized",
            "assignment_formulation": "flow_matching",
            "assignment_support": support,
            "posterior_samples": int(source.attrs["posterior_samples"]),
            "flow_steps": int(source.attrs["flow_steps"]),
            "nfe_per_sample": int(source.attrs["nfe_per_sample"]),
            "network_calls_per_event": int(source.attrs["network_calls_per_event"]),
            "assignment_state_count": int(source.attrs["assignment_state_count"]),
            "assignment": assignment,
        }
