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


def _kinematics(vectors: np.ndarray) -> dict[str, np.ndarray]:
    momentum = vectors[..., 1:]
    pt = np.linalg.norm(momentum[..., :2], axis=-1)
    mass = np.sqrt(
        np.maximum(vectors[..., 0] ** 2 - np.sum(momentum ** 2, axis=-1), 0)
    )
    eta = np.arcsinh(
        np.divide(momentum[..., 2], np.maximum(pt, 1.0e-12))
    )
    phi = np.arctan2(momentum[..., 1], momentum[..., 0])
    return {"mass": mass, "pt": pt, "eta": eta, "phi": phi}


def _wrapped_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(first - second), np.cos(first - second))


def _composite_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    """Separate reconstruction support from conditional kinematic quality."""
    sample_finite = np.isfinite(predicted).all(axis=-1)
    truth_finite = np.isfinite(truth).all(axis=-1)
    any_sample = sample_finite.any(axis=1)
    eligible = truth_finite & any_sample

    result = {
        "truth_reconstructable_events": int(truth_finite.sum()),
        "sample_reconstructable_rate": float(sample_finite.mean()),
        "event_any_reconstructable_rate": float(any_sample.mean()),
        "truth_support_frequency": (
            float(sample_finite[truth_finite].mean()) if truth_finite.any() else None
        ),
        "false_support_frequency": (
            float(sample_finite[~truth_finite].mean()) if (~truth_finite).any() else None
        ),
        "kinematic_events": int(eligible.sum()),
    }
    if not eligible.any():
        return result

    safe = np.where(sample_finite[..., None], predicted, np.nan)
    mean = np.nanmean(safe[eligible], axis=1)
    predicted_kinematics = _kinematics(mean)
    truth_kinematics = _kinematics(truth[eligible])
    for observable in ("mass", "pt", "eta", "phi"):
        residual = (
            _wrapped_delta(
                predicted_kinematics[observable], truth_kinematics[observable]
            )
            if observable == "phi"
            else predicted_kinematics[observable] - truth_kinematics[observable]
        )
        result[observable] = {
            "bias": float(residual.mean()),
            "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "central_68_width": float(
                np.quantile(residual, 0.84) - np.quantile(residual, 0.16)
            ),
        }
    return result


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
            "architecture_signature_version": int(
                source.attrs["architecture_signature_version"]
            ),
            "architecture_signature": str(source.attrs["architecture_signature"]),
            "model_mode": "joint_flow",
            "assignment_formulation": "flow_matching",
            "invisible_formulation": "flow_matching",
            "flow_steps": int(source.attrs["flow_steps"]),
            "actual_nfe": int(source.attrs["actual_nfe"]),
            "nfe_per_sample": int(source.attrs["nfe_per_sample"]),
            "network_calls_per_event": int(source.attrs["network_calls_per_event"]),
            "posterior_samples": int(source.attrs["posterior_samples"]),
            "recurrent_state": bool(source.attrs["recurrent_state"]),
            "assignment_state_count": int(source.attrs["assignment_state_count"]),
            "assignment": {
                "empirical_mode_accuracy": float((mode == truth_state).mean()),
                "truth_sample_frequency": float(truth_hit.mean()),
                "truth_sample_coverage": float(truth_hit.any(axis=1).mean()),
                "unique_states_mean": float(
                    np.mean([len(np.unique(event)) for event in states])
                ),
                "probabilities_are_monte_carlo_estimates": True,
                "mode_accuracy_vs_njet": {
                    str(int(value)): {
                        "events": int((njet == value).sum()),
                        "accuracy": float(
                            (mode[njet == value] == truth_state[njet == value]).mean()
                        ),
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

        if "COMPOSITES" in source and "TRUTH/COMPOSITES" in source:
            result["physics"] = {
                node: _composite_metrics(
                    source[f"COMPOSITES/{node}"][:],
                    source[f"TRUTH/COMPOSITES/{node}"][:],
                )
                for node in source["COMPOSITES"]
            }
            if "OBSERVED/MET" in source:
                predicted_px = samples[..., 0].sum(axis=2).mean(axis=1)
                predicted_py = samples[..., 1].sum(axis=2).mean(axis=1)
                met = source["OBSERVED/MET"][:]
                dx = predicted_px - met[:, 1]
                dy = predicted_py - met[:, 2]
                result["physics"]["met_closure"] = {
                    "px_bias": float(dx.mean()),
                    "py_bias": float(dy.mean()),
                    "transverse_rmse": float(
                        np.sqrt(np.mean(dx ** 2 + dy ** 2))
                    ),
                }
        return result


def evaluate_any_predictions(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as source:
        schema = source.attrs.get("schema_version")
    if schema == JOINT_PREDICTION_SCHEMA:
        return evaluate_joint_predictions(path)

    from joint_event_reconstruction.prediction_discrete import DISCRETE_PREDICTION_SCHEMA
    if schema == DISCRETE_PREDICTION_SCHEMA:
        from joint_event_reconstruction.evaluation_discrete import evaluate_discrete_predictions
        return evaluate_discrete_predictions(path)

    from joint_event_reconstruction.evaluation import evaluate_predictions
    return evaluate_predictions(path)
