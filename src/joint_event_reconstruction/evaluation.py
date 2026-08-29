"""Metrics for discriminative assignments and factorized generative posteriors."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.assignment_semantics import read_hdf5_support
from joint_event_reconstruction.prediction_schema import validate_prediction


COVERAGE_LEVELS = (0.50, 0.68, 0.90, 0.95)


def _normalised_weights(samples: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.full(samples.shape[:2], 1.0 / samples.shape[1], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != samples.shape[:2]:
        raise ValueError(
            f"Posterior weights {weights.shape} do not match samples {samples.shape[:2]}"
        )
    total = weights.sum(axis=1, keepdims=True)
    if (total <= 0).any() or not np.isfinite(total).all():
        raise ValueError("Posterior weights must have positive finite event sums")
    return weights / total


def _weighted_quantile_rows(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for event in range(len(values)):
        order = np.argsort(values[event], kind="stable")
        cumulative = np.cumsum(weights[event, order])
        index = min(
            int(np.searchsorted(cumulative, quantile, side="left")), len(order) - 1
        )
        result[event] = values[event, order[index]]
    return result


def _weighted_energy_score(
    samples: np.ndarray, truth: np.ndarray, weights: np.ndarray | None = None
) -> float:
    flat = samples.reshape(len(samples), samples.shape[1], -1).astype(np.float64)
    target = truth.reshape(len(truth), -1).astype(np.float64)
    normalised = _normalised_weights(flat, weights)
    scores = []
    for event in range(len(flat)):
        distances = np.linalg.norm(flat[event] - target[event], axis=-1)
        data_term = np.sum(normalised[event] * distances)
        pair = np.linalg.norm(flat[event, :, None] - flat[event, None, :], axis=-1)
        sample_term = np.sum(
            normalised[event, :, None] * normalised[event, None, :] * pair
        )
        scores.append(data_term - 0.5 * sample_term)
    return float(np.mean(scores))


def _posterior_metrics(
    samples: np.ndarray,
    truth: np.ndarray,
    role_names: list[str],
    *,
    weights: np.ndarray | None = None,
) -> dict:
    if samples.ndim != 4 or samples.shape[2:] != truth.shape[1:]:
        raise ValueError(
            "Invisible posterior must be [N,S,Rnu,3] matching truth [N,Rnu,3]"
        )
    normalised = _normalised_weights(samples, weights)
    mean = np.sum(samples * normalised[:, :, None, None], axis=1)
    result = {
        "energy_score_joint": _weighted_energy_score(samples, truth, normalised),
        "samples_per_event": int(samples.shape[1]),
        "roles": {},
    }
    for role_index, role_name in enumerate(role_names):
        role_result = {
            "energy_score": _weighted_energy_score(
                samples[:, :, role_index], truth[:, role_index], normalised
            )
        }
        for component, component_index in zip(("px", "py", "pz"), range(3)):
            mean_residual = (
                mean[:, role_index, component_index]
                - truth[:, role_index, component_index]
            )
            values = samples[:, :, role_index, component_index]
            target = truth[:, role_index, component_index]
            component_result = {
                "bias": float(mean_residual.mean()),
                "rmse": float(np.sqrt(np.mean(mean_residual**2))),
                "median_residual": float(np.median(mean_residual)),
                "central_68_width": float(
                    np.quantile(mean_residual, 0.84) - np.quantile(mean_residual, 0.16)
                ),
            }
            for level in COVERAGE_LEVELS:
                low = _weighted_quantile_rows(values, normalised, (1 - level) / 2)
                high = _weighted_quantile_rows(
                    values, normalised, 1 - (1 - level) / 2
                )
                component_result[f"coverage_{int(level * 100)}"] = float(
                    ((target >= low) & (target <= high)).mean()
                )
            role_result[component] = component_result
        result["roles"][role_name] = role_result
    return result


def _kinematics(vectors: np.ndarray) -> dict[str, np.ndarray]:
    momentum = vectors[..., 1:]
    pt = np.linalg.norm(momentum[..., :2], axis=-1)
    mass = np.sqrt(
        np.maximum(vectors[..., 0] ** 2 - np.sum(momentum**2, axis=-1), 0)
    )
    eta = np.arcsinh(np.divide(momentum[..., 2], np.maximum(pt, 1.0e-12)))
    phi = np.arctan2(momentum[..., 1], momentum[..., 0])
    return {"mass": mass, "pt": pt, "eta": eta, "phi": phi}


def _wrapped_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(first - second), np.cos(first - second))


def _point_kinematic_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    finite = np.isfinite(predicted).all(axis=-1) & np.isfinite(truth).all(axis=-1)
    result = {"events": int(finite.sum())}
    if not finite.any():
        return result
    predicted_kinematics = _kinematics(predicted[finite])
    truth_kinematics = _kinematics(truth[finite])
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


def bootstrap_mean_interval(
    values: np.ndarray, *, replicas: int = 1000, seed: int = 42
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap input must be a non-empty event-level vector")
    rng = np.random.default_rng(seed)
    unique = np.unique(values)
    if np.isin(unique, (0.0, 1.0)).all():
        means = rng.binomial(len(values), values.mean(), size=replicas) / len(values)
    else:
        means = np.asarray(
            [
                values[rng.integers(0, len(values), size=len(values))].mean()
                for _ in range(replicas)
            ]
        )
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def _binomial_error(value: float, count: int) -> float:
    return float(np.sqrt(value * (1.0 - value) / count)) if count else float("nan")


def evaluate_predictions(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        version = source.attrs.get("architecture_signature_version")
        conditioning = str(
            source.attrs.get("invisible_assignment_conditioning", "unknown")
        )
        support = read_hdf5_support(source)
        result = {
            "prediction_file": str(path),
            "events": len(source["EVENT_ID"]),
            "topology_signature": str(source.attrs["topology_signature"]),
            "architecture_signature_version": None if version is None else int(version),
            "architecture_signature": str(source.attrs["architecture_signature"]),
            "model_mode": "factorized",
            "assignment_formulation": "discriminative",
            "assignment_support": support,
            "invisible_formulation": str(
                source.attrs.get("invisible_formulation", "disabled")
            ),
            "invisible_assignment_conditioning": conditioning,
            "top_k_assignments": int(source.attrs["K"]),
            "posterior_samples": int(source.attrs["M"]),
            "recurrent_steps": int(source.attrs["recurrent_steps"]),
            "visible": {},
        }
        if "INVISIBLE/samples" in source:
            result["nfe_per_invisible_sample"] = int(
                source["INVISIBLE"].attrs["actual_nfe"]
            )
            result["unique_assignment_conditioning_branches"] = int(
                source["INVISIBLE"].attrs["unique_assignment_conditioning_branches"]
            )
            result["sample_equivalent_nfe_per_event"] = int(
                source["INVISIBLE"].attrs["sample_equivalent_nfe_per_event"]
            )

        nodes = source["VISIBLE/nodes"]
        for name in nodes:
            node = nodes[name]
            reconstructable = node["reconstructable"][:].astype(bool)
            selected = node["selected_candidate"][:]
            truth = node["truth_candidate"][:]
            correct = selected == truth
            raw = float(correct.mean())
            conditional = (
                float(correct[reconstructable].mean()) if reconstructable.any() else None
            )
            result["visible"][name] = {
                "events": int(len(correct)),
                "reconstructable": int(reconstructable.sum()),
                "raw_efficiency": raw,
                "raw_efficiency_standard_error": _binomial_error(raw, len(correct)),
                "conditional_efficiency": conditional,
                "conditional_efficiency_standard_error": (
                    _binomial_error(conditional, int(reconstructable.sum()))
                    if conditional is not None
                    else None
                ),
                "mean_truth_probability": (
                    float(node["truth_probability"][:][reconstructable].mean())
                    if reconstructable.any()
                    else None
                ),
                "mean_entropy": float(node["entropy"][:].mean()),
                "mean_margin": float(node["margin"][:].mean()),
                "null_rate": float((selected < 0).mean()),
                "conditional_efficiency_bootstrap_95": (
                    list(bootstrap_mean_interval(correct[reconstructable]))
                    if reconstructable.any()
                    else None
                ),
            }

        truth_topk = source["TRUTH/topk_assignment"][:]
        assignment_probabilities = source["VISIBLE/constrained_probabilities"][:]
        retained_mass = assignment_probabilities.sum(axis=1)
        top1 = float(truth_topk[:, 0].mean())
        topk = float(truth_topk.any(axis=1).mean())
        result["visible"]["complete"] = {
            "support": support["name"],
            "top1_efficiency": top1,
            "top1_efficiency_standard_error": _binomial_error(top1, len(truth_topk)),
            "topk_coverage": topk,
            "topk_coverage_standard_error": _binomial_error(topk, len(truth_topk)),
            "truth_rank_mean_reciprocal": float(
                np.mean(
                    [
                        1.0 / (np.flatnonzero(row)[0] + 1) if row.any() else 0.0
                        for row in truth_topk
                    ]
                )
            ),
            "mean_exact_logZ": float(source["VISIBLE/exact_logZ"][:].mean()),
            "probabilities_are_exact_constrained_within_support": True,
            "retained_topk_probability_mass": {
                "mean": float(retained_mass.mean()),
                "median": float(np.median(retained_mass)),
                "minimum": float(retained_mass.min()),
            },
        }
        njet = source["OBSERVED/njet"][:]
        by_njet = {}
        for value in np.unique(njet):
            mask = njet == value
            count = int(mask.sum())
            efficiency = float(truth_topk[mask, 0].mean())
            by_njet[str(int(value))] = {
                "events": count,
                "efficiency": efficiency,
                "standard_error": _binomial_error(efficiency, count),
            }
        result["visible"]["complete"]["top1_vs_njet"] = by_njet

        if "INVISIBLE/samples" in source:
            samples = source["INVISIBLE/samples"][:]
            truth = source["TRUTH/NEUTRINO"][:]
            roles = json.loads(source.attrs["invisible_roles"])
            invisible = {
                "interpretation": (
                    "assignment-weighted truncated mixture p(A|X)p(nu|X,A)"
                    if conditioning == "hypothesis"
                    else (
                        f"single p(nu|X, conditioning={conditioning}) posterior; "
                        "K does not add generative branches"
                    )
                ),
                "top1_assignment_conditioned": _posterior_metrics(
                    samples[:, 0], truth, roles
                ),
            }

            if conditioning == "hypothesis":
                batch_size, top_k, posterior_samples = samples.shape[:3]
                flat = samples.reshape(
                    batch_size, top_k * posterior_samples, *samples.shape[3:]
                )
                branch_weights = assignment_probabilities / retained_mass[:, None]
                weights = np.repeat(
                    branch_weights[:, :, None] / posterior_samples,
                    posterior_samples,
                    axis=2,
                ).reshape(batch_size, top_k * posterior_samples)
                invisible["truncated_assignment_weighted_mixture"] = _posterior_metrics(
                    flat, truth, roles, weights=weights
                )
                invisible["truncated_assignment_weighted_mixture"][
                    "mean_retained_assignment_probability_mass"
                ] = float(retained_mass.mean())

                correct_top1 = truth_topk[:, 0]
                if correct_top1.any():
                    invisible["conditional_on_correct_top1_assignment"] = {
                        "selection_biased_diagnostic": True,
                        "events": int(correct_top1.sum()),
                        **_posterior_metrics(
                            samples[correct_top1, 0], truth[correct_top1], roles
                        ),
                    }
                truth_in_topk = truth_topk.any(axis=1)
                if truth_in_topk.any():
                    selected = truth_topk[truth_in_topk].argmax(axis=1)
                    rows = np.arange(int(truth_in_topk.sum()))
                    truth_branch_samples = samples[truth_in_topk][rows, selected]
                    invisible["truth_assignment_branch_when_present_in_topk"] = {
                        "diagnostic_not_end_to_end": True,
                        "events": int(truth_in_topk.sum()),
                        **_posterior_metrics(
                            truth_branch_samples, truth[truth_in_topk], roles
                        ),
                    }
            else:
                invisible["truncated_assignment_weighted_mixture"] = {
                    "same_as_single_generator_posterior": True,
                    **_posterior_metrics(samples[:, 0], truth, roles),
                }
            result["invisible"] = invisible

        if "COMPOSITES" in source and "TRUTH/COMPOSITES" in source:
            physics = {}
            for node in source["COMPOSITES"]:
                predicted_samples = source[f"COMPOSITES/{node}"][:]
                truth_vector = source[f"TRUTH/COMPOSITES/{node}"][:]
                top1_point = np.nanmean(predicted_samples[:, 0], axis=1)
                node_result = {
                    "top1_assignment_posterior_mean": _point_kinematic_metrics(
                        top1_point, truth_vector
                    )
                }
                if conditioning == "hypothesis":
                    probabilities = assignment_probabilities / retained_mass[:, None]
                    branch_means = np.nanmean(predicted_samples, axis=2)
                    finite = np.isfinite(branch_means).all(axis=-1)
                    weighted = np.where(
                        finite[..., None],
                        branch_means * probabilities[..., None],
                        0.0,
                    ).sum(axis=1)
                    support_weight = np.where(finite, probabilities, 0.0).sum(axis=1)
                    weighted = np.divide(
                        weighted,
                        support_weight[:, None],
                        out=np.full_like(weighted, np.nan),
                        where=support_weight[:, None] > 0,
                    )
                    node_result["assignment_weighted_posterior_mean"] = (
                        _point_kinematic_metrics(weighted, truth_vector)
                    )
                physics[node] = node_result

            if "OBSERVED/MET" in source and "INVISIBLE/samples" in source:
                invisible_samples = source["INVISIBLE/samples"][:]
                if conditioning == "hypothesis":
                    branch = assignment_probabilities / retained_mass[:, None]
                    predicted_px = np.sum(
                        branch * invisible_samples[..., 0].sum(axis=3).mean(axis=2),
                        axis=1,
                    )
                    predicted_py = np.sum(
                        branch * invisible_samples[..., 1].sum(axis=3).mean(axis=2),
                        axis=1,
                    )
                else:
                    predicted_px = invisible_samples[:, 0, ..., 0].sum(axis=2).mean(axis=1)
                    predicted_py = invisible_samples[:, 0, ..., 1].sum(axis=2).mean(axis=1)
                met = source["OBSERVED/MET"][:]
                dx = predicted_px - met[:, 1]
                dy = predicted_py - met[:, 2]
                physics["met_closure"] = {
                    "px_bias": float(dx.mean()),
                    "py_bias": float(dy.mean()),
                    "transverse_rmse": float(np.sqrt(np.mean(dx**2 + dy**2))),
                }
            result["physics"] = physics

        if "JOINT/final_score" in source:
            ranking = source["JOINT/rank_indices"][:]
            reranked_truth = np.take_along_axis(truth_topk, ranking, axis=1)
            assignment_only = float(truth_topk[:, 0].mean())
            combined = float(reranked_truth[:, 0].mean())
            result["compatibility"] = {
                "weight": float(source["JOINT"].attrs["compatibility_weight"]),
                "assignment_only_top1": assignment_only,
                "assignment_only_top1_standard_error": _binomial_error(
                    assignment_only, len(truth_topk)
                ),
                "combined_top1": combined,
                "combined_top1_standard_error": _binomial_error(
                    combined, len(truth_topk)
                ),
                "mean_logit": float(source["JOINT/compatibility_logit"][:].mean()),
                "normalised_likelihood": False,
            }
        return result
