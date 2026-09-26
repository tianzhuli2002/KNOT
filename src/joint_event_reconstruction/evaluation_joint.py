"""Metrics for sampled joint discrete-continuous Flow Matching predictions."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.assignment_semantics import read_hdf5_support
from joint_event_reconstruction.evaluation import (
    _posterior_kinematic_metrics,
    _posterior_metrics,
    _scalar_posterior_metrics,
    _conditional_posterior_metrics,
    _weighted_quantile_rows,
)
from joint_event_reconstruction.prediction_schema import (
    DISCRETE_PREDICTION_SCHEMA,
    JOINT_PREDICTION_SCHEMA,
    PREDICTION_SCHEMA,
    validate_prediction,
    validate_joint_prediction,
)
from joint_event_reconstruction.physics_observables import (
    four_vector_observables,
    met_closure,
    root_system_observables,
    wrapped_delta,
)
from joint_event_reconstruction.reliability import (
    reliability_scores,
    selective_risk_curve,
)


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
    return four_vector_observables(vectors)


def _wrapped_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return wrapped_delta(first, second)


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
        "posterior_distribution": _posterior_kinematic_metrics(predicted, truth),
    }
    if not eligible.any():
        return result

    safe = np.where(sample_finite[..., None], predicted, np.nan)
    mean = np.nanmean(safe[eligible], axis=1)
    predicted_kinematics = _kinematics(mean)
    truth_kinematics = _kinematics(truth[eligible])
    for observable in ("mass", "pt", "eta", "rapidity", "phi"):
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


def _met_exact_projection_metrics(source: h5py.File) -> dict | None:
    """Summarise detector-MET projection without replacing the physical truth target."""
    path = "TRUTH/MET_EXACT_PROJECTION"
    if path not in source:
        return None
    group = source[path]
    px = group["px_shift"][:].astype(np.float64)
    py = group["py_shift"][:].astype(np.float64)
    magnitude = group["per_role_transverse_magnitude"][:].astype(np.float64)

    def summary(values: np.ndarray) -> dict:
        return {
            "mean": float(values.mean()),
            "rmse": float(np.sqrt(np.mean(values**2))),
            "maximum_absolute": float(np.max(np.abs(values))),
        }

    return {
        "definition": str(group.attrs.get("definition", "")),
        "events": int(len(px)),
        "px_shift": summary(px),
        "py_shift": summary(py),
        "per_role_transverse_magnitude": {
            "mean": float(magnitude.mean()),
            "rmse": float(np.sqrt(np.mean(magnitude**2))),
            "maximum": float(magnitude.max()),
        },
        "projected_target": "truth neutrino momenta after equal transverse MET projection",
    }


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
                **_posterior_metrics(samples, truth, roles),
            },
        }
        projection_metrics = _met_exact_projection_metrics(source)
        if projection_metrics is not None:
            projected_truth = source[
                "TRUTH/MET_EXACT_PROJECTION/NEUTRINO_PROJECTED"
            ][:]
            result["invisible"]["truth_projection"] = projection_metrics
            result["invisible"][
                "posterior_vs_projected_target"
            ] = _posterior_metrics(samples, projected_truth, roles)
        result["invisible"]["conditional_calibration"] = (
            _conditional_posterior_metrics(
                samples,
                truth,
                roles,
                {"njet": njet, "assignment_entropy": entropy},
            )
        )
        posterior_width = samples.std(axis=1).mean(axis=(1, 2))
        first_values = samples[:, :, 0, 0]
        unit_weights = np.full(first_values.shape, 1.0 / first_values.shape[1])
        low = _weighted_quantile_rows(first_values, unit_weights, 0.16)
        high = _weighted_quantile_rows(first_values, unit_weights, 0.84)
        covered = (truth[:, 0, 0] >= low) & (truth[:, 0, 0] <= high)
        result["reliability"] = {
            name: selective_risk_curve(
                score,
                fractions=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00),
                assignment_correct=mode == truth_state,
                covered=covered,
            )
            for name, score in reliability_scores(
                entropy=entropy, posterior_width=posterior_width
            ).items()
        }

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
            root_order = json.loads(source.attrs.get("root_order", "[]"))
            if root_order and all(root in source["COMPOSITES"] for root in root_order):
                root_samples = {
                    root: source[f"COMPOSITES/{root}"][:] for root in root_order
                }
                root_truth = {
                    root: source[f"TRUTH/COMPOSITES/{root}"][:] for root in root_order
                }
                sample_observables = root_system_observables(root_samples, root_order)
                truth_observables = root_system_observables(root_truth, root_order)
                weights = np.full(samples.shape[:2], 1.0 / samples.shape[1])
                root_result = {}
                for observable, values in sample_observables.items():
                    target = truth_observables[observable]
                    valid = np.isfinite(target) & np.isfinite(values).all(axis=1)
                    if observable in {"system_phi", "delta_phi"}:
                        values = _wrapped_delta(values, target[:, None])
                        target = np.zeros_like(target)
                    root_result[observable] = (
                        {"events": 0}
                        if not valid.any()
                        else {
                            "events": int(valid.sum()),
                            **_scalar_posterior_metrics(
                                values[valid], target[valid], weights[valid]
                            ),
                        }
                    )
                result["physics"]["root_system"] = root_result
            if "OBSERVED/MET" in source:
                met = source["OBSERVED/MET"][:]
                result["physics"]["met_closure"] = met_closure(
                    samples, met[:, 1:3]
                )
        return result


def evaluate_any_predictions(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as source:
        schema = source.attrs.get("schema_version")
    if schema == PREDICTION_SCHEMA:
        with h5py.File(path, "r") as source:
            visible_truth = bool(source.attrs.get("visible_truth_available", True))
            invisible_truth = bool(source.attrs.get("invisible_truth_available", True))
        if not visible_truth or not invisible_truth:
            validate_prediction(path)
            with h5py.File(path, "r") as source:
                event_count = len(source["EVENT_ID"])
                probabilities = source["VISIBLE/constrained_probabilities"][:]
                result = {
                    "prediction_file": str(path),
                    "events": event_count,
                    "visible_truth_available": visible_truth,
                    "invisible_truth_available": invisible_truth,
                    "assignment_hypotheses": int(source.attrs["K"]),
                    "retained_topk_probability_mass": {
                        "mean": float(probabilities.sum(axis=1).mean()),
                        "minimum": float(probabilities.sum(axis=1).min()),
                    },
                    "physics_metrics": "not computed: the prediction file has no truth sidecar",
                    "schema_validated": True,
                }
                if "INVISIBLE/samples" in source:
                    raw_roles = source.attrs["invisible_roles"]
                    if isinstance(raw_roles, bytes):
                        raw_roles = raw_roles.decode("utf-8")
                    roles = json.loads(str(raw_roles))
                    result["invisible_roles"] = roles
                    if roles == ["top.nu", "antitop.nu"]:
                        result["invisible_truth_contract"] = {
                            "convention": str(source.attrs["invisible_role_convention"]),
                            "roles": roles,
                            "truth_definition": str(source.attrs["neutrino_truth_definition"]),
                            "units": str(source.attrs["invisible_units"]),
                            "source_file_sha256": str(source.attrs["source_file_sha256"]),
                            "source_manifest_sha256": str(source.attrs["source_manifest_sha256"]),
                            "split_membership_sha256": str(source.attrs["split_membership_sha256"]),
                        }
                if "INVISIBLE/samples" in source:
                    result["invisible_posterior"] = {
                        "samples_per_event_hypothesis": int(source.attrs["M"]),
                        "actual_nfe": int(source["INVISIBLE"].attrs["actual_nfe"]),
                        "finite": bool(np.isfinite(source["INVISIBLE/samples"][:]).all()),
                    }
                return result
    if schema == JOINT_PREDICTION_SCHEMA:
        return evaluate_joint_predictions(path)

    if schema == DISCRETE_PREDICTION_SCHEMA:
        from joint_event_reconstruction.evaluation_discrete import (
            evaluate_discrete_predictions,
        )

        return evaluate_discrete_predictions(path)

    from joint_event_reconstruction.evaluation import evaluate_predictions

    return evaluate_predictions(path)
