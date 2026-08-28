"""Metrics for sampled joint discrete-continuous Flow Matching predictions."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.assignment_semantics import read_hdf5_support
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


def _single_event_energy(samples: np.ndarray, truth: np.ndarray) -> float:
    flat = samples.reshape(len(samples), -1)
    target = truth.reshape(-1)
    data_term = np.linalg.norm(flat - target[None], axis=-1).mean()
    pair = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1).mean()
    return float(data_term - 0.5 * pair)


def _empirical_mode(states: np.ndarray) -> np.ndarray:
    modes = np.empty(len(states), dtype=states.dtype)
    for event, values in enumerate(states):
        unique, counts = np.unique(values, return_counts=True)
        modes[event] = unique[np.argmax(counts)]
    return modes


def _empirical_entropy(states: np.ndarray) -> np.ndarray:
    result = np.empty(len(states), dtype=np.float64)
    for event, values in enumerate(states):
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts / counts.sum()
        result[event] = -np.sum(probabilities * np.log(probabilities))
    return result


def _binomial_error(value: float, count: int) -> float:
    return float(np.sqrt(value * (1.0 - value) / count)) if count else float("nan")


def _kinematics(vectors: np.ndarray) -> dict[str, np.ndarray]:
    momentum = vectors[..., 1:]
    pt = np.linalg.norm(momentum[..., :2], axis=-1)
    mass = np.sqrt(
        np.maximum(vectors[..., 0] ** 2 - np.sum(momentum**2, axis=-1), 0)
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
            float(sample_finite[~truth_finite].mean())
            if (~truth_finite).any()
            else None
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
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "central_68_width": float(
                np.quantile(residual, 0.84) - np.quantile(residual, 0.16)
            ),
        }
    return result


def _truth_assignment_continuous_diagnostic(
    samples: np.ndarray,
    truth: np.ndarray,
    truth_hit: np.ndarray,
    roles: list[str],
) -> dict | None:
    eligible = truth_hit.any(axis=1)
    if not eligible.any():
        return None
    energy = []
    role_energy = {role: [] for role in roles}
    event_means = []
    targets = []
    sample_counts = []
    for event in np.flatnonzero(eligible):
        selected = samples[event, truth_hit[event]]
        sample_counts.append(len(selected))
        energy.append(_single_event_energy(selected, truth[event]))
        event_means.append(selected.mean(axis=0))
        targets.append(truth[event])
        for role_index, role in enumerate(roles):
            role_energy[role].append(
                _single_event_energy(
                    selected[:, role_index], truth[event, role_index]
                )
            )
    means = np.stack(event_means)
    targets = np.stack(targets)
    result = {
        "diagnostic_not_end_to_end": True,
        "events_with_at_least_one_truth_assignment_sample": int(eligible.sum()),
        "mean_truth_assignment_samples_for_these_events": float(
            np.mean(sample_counts)
        ),
        "energy_score_joint": float(np.mean(energy)),
        "roles": {},
    }
    for role_index, role in enumerate(roles):
        residual = means[:, role_index] - targets[:, role_index]
        result["roles"][role] = {
            "energy_score": float(np.mean(role_energy[role])),
            "px_rmse": float(np.sqrt(np.mean(residual[:, 0] ** 2))),
            "py_rmse": float(np.sqrt(np.mean(residual[:, 1] ** 2))),
            "pz_rmse": float(np.sqrt(np.mean(residual[:, 2] ** 2))),
        }
    return result


def evaluate_joint_predictions(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    validate_joint_prediction(path)
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema_version") != JOINT_PREDICTION_SCHEMA:
            raise ValueError(f"Not a joint-flow prediction file: {path}")
        support = read_hdf5_support(source)
        states = source["HYPOTHESES/assignment_state"][:]
        truth_state = source["TRUTH/assignment_state"][:]
        truth_hit = states == truth_state[:, None]
        mode = _empirical_mode(states)
        entropy = _empirical_entropy(states)
        njet = source["OBSERVED/njet"][:]
        samples = source["HYPOTHESES/invisible"][:]
        truth = source["TRUTH/NEUTRINO"][:]
        roles = json.loads(source.attrs["invisible_roles"])

        accuracy = float((mode == truth_state).mean())
        by_njet = {}
        for value in np.unique(njet):
            mask = njet == value
            count = int(mask.sum())
            local_accuracy = float((mode[mask] == truth_state[mask]).mean())
            by_njet[str(int(value))] = {
                "events": count,
                "accuracy": local_accuracy,
                "standard_error": _binomial_error(local_accuracy, count),
            }

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
            "assignment_support": support,
            "invisible_formulation": "flow_matching",
            "flow_steps": int(source.attrs["flow_steps"]),
            "nfe_per_sample": int(source.attrs["nfe_per_sample"]),
            "sample_chunk_size": int(source.attrs.get("sample_chunk_size", 1)),
            "posterior_chunk_count": int(
                source.attrs.get("posterior_chunk_count", len(states[0]))
            ),
            "batched_network_evaluations_per_event": int(
                source.attrs.get(
                    "batched_network_evaluations_per_event",
                    int(source.attrs["flow_steps"]) * len(states[0]),
                )
            ),
            "sample_equivalent_nfe_per_event": int(
                source.attrs.get(
                    "sample_equivalent_nfe_per_event",
                    int(source.attrs["flow_steps"]) * len(states[0]),
                )
            ),
            "posterior_samples": int(source.attrs["posterior_samples"]),
            "recurrent_state": bool(source.attrs["recurrent_state"]),
            "assignment_state_count": int(source.attrs["assignment_state_count"]),
            "assignment": {
                "empirical_mode_accuracy": accuracy,
                "empirical_mode_accuracy_standard_error": _binomial_error(
                    accuracy, len(states)
                ),
                "truth_sample_frequency": float(truth_hit.mean()),
                "truth_sample_coverage": float(truth_hit.any(axis=1).mean()),
                "unique_states_mean": float(
                    np.mean([len(np.unique(event)) for event in states])
                ),
                "empirical_entropy_mean": float(entropy.mean()),
                "effective_state_count_mean": float(np.exp(entropy).mean()),
                "probabilities_are_monte_carlo_estimates": True,
                "mode_accuracy_vs_njet": by_njet,
            },
            "invisible": {
                "interpretation": "native joint posterior samples p(A,nu|X)",
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
                    "rmse": float(np.sqrt(np.mean(mean_residual**2))),
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

        diagnostic = _truth_assignment_continuous_diagnostic(
            samples, truth, truth_hit, roles
        )
        if diagnostic is not None:
            result["invisible"][
                "conditional_on_sampled_truth_assignment"
            ] = diagnostic

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
                    "transverse_rmse": float(np.sqrt(np.mean(dx**2 + dy**2))),
                }
        return result


def evaluate_any_predictions(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as source:
        schema = source.attrs.get("schema_version")
    if schema == JOINT_PREDICTION_SCHEMA:
        return evaluate_joint_predictions(path)

    from joint_event_reconstruction.prediction_discrete import (
        DISCRETE_PREDICTION_SCHEMA,
    )

    if schema == DISCRETE_PREDICTION_SCHEMA:
        from joint_event_reconstruction.evaluation_discrete import (
            evaluate_discrete_predictions,
        )

        return evaluate_discrete_predictions(path)

    from joint_event_reconstruction.evaluation import evaluate_predictions

    return evaluate_predictions(path)
