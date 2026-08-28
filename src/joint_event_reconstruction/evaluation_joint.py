"""Metrics for sampled joint discrete-continuous Flow Matching predictions."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.prediction_joint import (
    JOINT_PREDICTION_SCHEMA,
    validate_joint_prediction,
)


def _energy_score(samples: np.ndarray, truth: np.ndarray) -> float:
    samples = samples.reshape(len(samples), samples.shape[1], -1)
    truth = truth.reshape(len(truth), -1)
    data_term = np.linalg.norm(samples - truth[:, None], axis=-1).mean(axis=1)
    pair = samples[:, :, None] - samples[:, None, :]
    sample_term = np.linalg.norm(pair, axis=-1).mean(axis=(1, 2))
    return float(np.mean(data_term - 0.5 * sample_term))


def _empirical_mode(states: np.ndarray) -> np.ndarray:
    modes = np.empty(len(states), dtype=states.dtype)
    for event, values in enumerate(states):
        unique, counts = np.unique(values, return_counts=True)
        modes[event] = unique[np.argmax(counts)]
    return modes


def evaluate_joint_predictions(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    validate_joint_prediction(path)
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema_version") != JOINT_PREDICTION_SCHEMA:
            raise ValueError(f"Not a joint-flow prediction file: {path}")
        states = source["HYPOTHESES/assignment_state"][:]
        truth_state = source["TRUTH/assignment_state"][:]
        truth_hit = states == truth_state[:, None]
        mode = _empirical_mode(states)
        njet = source["OBSERVED/njet"][:]
        samples = source["HYPOTHESES/invisible"][:]
        truth = source["TRUTH/NEUTRINO"][:]
        roles = json.loads(source.attrs["invisible_roles"])

        result = {
            "prediction_file": str(path),
            "events": len(states),
            "topology_signature": str(source.attrs["topology_signature"]),
            "model_mode": "joint_flow",
            "flow_steps": int(source.attrs["flow_steps"]),
            "posterior_samples": int(source.attrs["posterior_samples"]),
            "recurrent_state": bool(source.attrs["recurrent_state"]),
            "assignment": {
                "empirical_mode_accuracy": float((mode == truth_state).mean()),
                "truth_sample_frequency": float(truth_hit.mean()),
                "truth_sample_coverage": float(truth_hit.any(axis=1).mean()),
                "unique_states_mean": float(np.mean([
                    len(np.unique(event)) for event in states
                ])),
                "probabilities_are_monte_carlo_estimates": True,
                "mode_accuracy_vs_njet": {
                    str(int(value)): {
                        "events": int((njet == value).sum()),
                        "accuracy": float((mode[njet == value] == truth_state[njet == value]).mean()),
                    }
                    for value in np.unique(njet)
                },
            },
            "invisible": {
                "energy_score_joint": _energy_score(samples, truth),
                "roles": {},
            },
        }

        mean = samples.mean(axis=1)
        levels = (0.50, 0.68, 0.90, 0.95)
        for role_index, role_name in enumerate(roles):
            role_result = {}
            for component, component_index in zip(("px", "py", "pz"), range(3)):
                mean_residual = (
                    mean[:, role_index, component_index]
                    - truth[:, role_index, component_index]
                )
                component_result = {
                    "bias": float(mean_residual.mean()),
                    "rmse": float(np.sqrt(np.mean(mean_residual ** 2))),
                    "median_residual": float(np.median(mean_residual)),
                    "central_68_width": float(
                        np.quantile(mean_residual, 0.84)
                        - np.quantile(mean_residual, 0.16)
                    ),
                }
                values = samples[:, :, role_index, component_index]
                target = truth[:, role_index, component_index]
                for level in levels:
                    low = np.quantile(values, (1 - level) / 2, axis=1)
                    high = np.quantile(values, 1 - (1 - level) / 2, axis=1)
                    component_result[f"coverage_{int(level * 100)}"] = float(
                        ((target >= low) & (target <= high)).mean()
                    )
                role_result[component] = component_result
            role_result["energy_score"] = _energy_score(
                samples[:, :, role_index], truth[:, role_index]
            )
            result["invisible"]["roles"][role_name] = role_result
        return result


def evaluate_any_predictions(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as source:
        schema = source.attrs.get("schema_version")
    if schema == JOINT_PREDICTION_SCHEMA:
        return evaluate_joint_predictions(path)
    from joint_event_reconstruction.evaluation import evaluate_predictions
    return evaluate_predictions(path)
