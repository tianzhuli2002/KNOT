"""Metrics for discriminative assignments and factorized generative posteriors."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.assignment_semantics import read_hdf5_support
from joint_event_reconstruction.physics_observables import (
    four_vector_observables,
    met_closure,
    root_system_observables,
    wrapped_delta,
)
from joint_event_reconstruction.prediction_schema import validate_prediction
from joint_event_reconstruction.reliability import (
    reliability_scores,
    selective_risk_curve,
)


COVERAGE_LEVELS = (0.50, 0.68, 0.90, 0.95)
COVERAGE_CURVE_LEVELS = tuple(
    sorted(set(np.round(np.arange(0.1, 1.0, 0.1), 2)) | set(COVERAGE_LEVELS))
)
RELIABILITY_FRACTIONS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00)


def _normalised_weights(samples: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.full(samples.shape[:2], 1.0 / samples.shape[1], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != samples.shape[:2]:
        raise ValueError(
            f"Posterior weights {weights.shape} do not match samples {samples.shape[:2]}"
        )
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Posterior weights must be finite and non-negative")
    total = weights.sum(axis=1, keepdims=True)
    if (total <= 0).any() or not np.isfinite(total).all():
        raise ValueError("Posterior weights must have positive finite event sums")
    return weights / total


def _weighted_quantile_rows(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> np.ndarray:
    if values.ndim != 2 or weights.shape != values.shape:
        raise ValueError("Weighted quantiles require [N,S] values and matching weights")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Weighted quantile must be between zero and one")
    result = np.empty(len(values), dtype=np.float64)
    for event in range(len(values)):
        order = np.argsort(values[event], kind="stable")
        ordered_values = values[event, order]
        ordered_weights = weights[event, order]
        positive = ordered_weights > 0.0
        if not positive.any():
            raise ValueError("Weighted quantiles require positive event weight")
        ordered_values = ordered_values[positive]
        ordered_weights = ordered_weights[positive]
        cumulative = np.cumsum(ordered_weights)
        index = min(
            int(np.searchsorted(cumulative, quantile, side="left")),
            len(ordered_values) - 1,
        )
        result[event] = ordered_values[index]
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


def _weighted_crps_rows(
    values: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Exact CRPS for finite weighted samples, one value per event."""
    if values.ndim != 2 or weights.shape != values.shape or truth.shape != (len(values),):
        raise ValueError("Weighted CRPS requires matching [N,S] values, weights and [N] truth")
    result = np.empty(len(values), dtype=np.float64)
    for event in range(len(values)):
        order = np.argsort(values[event], kind="stable")
        ordered = values[event, order].astype(np.float64)
        local_weights = weights[event, order]
        data_term = np.sum(local_weights * np.abs(ordered - truth[event]))
        cumulative_before = np.cumsum(local_weights) - local_weights
        pair_term = 2.0 * np.sum(
            local_weights
            * ordered
            * (2.0 * cumulative_before + local_weights - 1.0)
        )
        result[event] = data_term - 0.5 * pair_term
    return result


def _finite_sample_rank(
    values: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Weighted mid-rank CDF values; discrete for a finite posterior sample."""
    if values.ndim != 2 or weights.shape != values.shape or truth.shape != (len(values),):
        raise ValueError("Finite-sample ranks require matching [N,S] values, weights and [N] truth")
    below = np.sum(weights * (values < truth[:, None]), axis=1)
    equal = np.sum(weights * (values == truth[:, None]), axis=1)
    return below + 0.5 * equal


def _rank_summary(rank: np.ndarray) -> dict:
    rank = np.asarray(rank, dtype=np.float64)
    if rank.ndim != 1 or not len(rank) or not np.isfinite(rank).all() or (
        (rank < 0.0).any() or (rank > 1.0).any()
    ):
        raise ValueError("Finite-sample ranks must be a non-empty finite vector in [0, 1]")
    edges = np.linspace(0.0, 1.0, 11)
    histogram, _ = np.histogram(rank, bins=edges)
    ordered = np.sort(rank)
    count = len(rank)
    expected = (np.arange(count) + 0.5) / count
    upper = np.arange(1, count + 1) / count - ordered
    lower = ordered - np.arange(count) / count
    return {
        "method": "weighted finite-sample mid-rank; discrete, not continuous PIT",
        "histogram_counts": histogram.tolist(),
        "histogram_edges": edges.tolist(),
        "mean": float(rank.mean()),
        "variance": float(rank.var()),
        "kolmogorov_smirnov_distance_to_uniform": float(
            max(np.max(upper), np.max(lower))
        ),
        "cramer_von_mises": float(
            1.0 / (12.0 * count) + np.sum((ordered - expected) ** 2)
        ),
    }


def _scalar_posterior_metrics(
    values: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> dict:
    mean = np.sum(values * weights, axis=1)
    residual = mean - truth
    rank = _finite_sample_rank(values, truth, weights)
    coverage = {}
    for level in COVERAGE_CURVE_LEVELS:
        low = _weighted_quantile_rows(values, weights, (1 - level) / 2)
        high = _weighted_quantile_rows(values, weights, 1 - (1 - level) / 2)
        coverage[f"{level:.2f}"] = float(((truth >= low) & (truth <= high)).mean())
    return {
        "bias": float(residual.mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "median_residual": float(np.median(residual)),
        "central_68_width": float(
            np.quantile(residual, 0.84) - np.quantile(residual, 0.16)
        ),
        "crps": float(_weighted_crps_rows(values, truth, weights).mean()),
        "finite_sample_rank": _rank_summary(rank),
        "coverage_curve": coverage,
    }


def _posterior_metrics(
    samples: np.ndarray,
    truth: np.ndarray,
    role_names: list[str],
    *,
    weights: np.ndarray | None = None,
) -> dict:
    if (
        samples.ndim != 4
        or truth.ndim != 3
        or samples.shape[0] != truth.shape[0]
        or samples.shape[2:] != truth.shape[1:]
        or samples.shape[1] <= 0
        or len(role_names) != truth.shape[1]
        or truth.shape[2] != 3
        or not np.isfinite(samples).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError(
            "Invisible posterior must be [N,S,Rnu,3] matching truth [N,Rnu,3]"
        )
    normalised = _normalised_weights(samples, weights)
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
            values = samples[:, :, role_index, component_index]
            target = truth[:, role_index, component_index]
            component_result = _scalar_posterior_metrics(
                values, target, normalised
            )
            for level in COVERAGE_LEVELS:
                component_result[f"coverage_{int(level * 100)}"] = component_result[
                    "coverage_curve"
                ][f"{level:.2f}"]
            role_result[component] = component_result
        result["roles"][role_name] = role_result
    return result


def _conditional_posterior_metrics(
    samples: np.ndarray,
    truth: np.ndarray,
    role_names: list[str],
    variables: dict[str, np.ndarray],
    *,
    weights: np.ndarray | None = None,
    quantile_bins: int = 5,
) -> dict:
    result = {}
    for name, raw in variables.items():
        values = np.asarray(raw)
        if values.shape != (len(samples),):
            raise ValueError(f"Conditional variable {name} must have shape [N]")
        if values.dtype == np.bool_:
            edges = [False, True]
            masks = [(str(value).lower(), values == value) for value in edges]
            recorded_edges = edges
        elif name == "njet":
            unique = np.unique(values)
            masks = [(str(int(value)), values == value) for value in unique]
            recorded_edges = [int(value) for value in unique]
        else:
            if not np.isfinite(values).all():
                raise ValueError(f"Conditional variable {name} contains non-finite values")
            recorded_edges = np.unique(
                np.quantile(values, np.linspace(0, 1, quantile_bins + 1))
            ).tolist()
            masks = []
            for index, (low, high) in enumerate(
                zip(recorded_edges[:-1], recorded_edges[1:])
            ):
                mask = (values >= low) & (
                    values <= high if index == len(recorded_edges) - 2 else values < high
                )
                masks.append((str(index), mask))
        bins = []
        for label, selection in masks:
            count = int(selection.sum())
            row = {
                "bin": label,
                "events": count,
                "small_sample_warning": count < 30,
            }
            if count:
                row["posterior"] = _posterior_metrics(
                    samples[selection],
                    truth[selection],
                    role_names,
                    weights=None if weights is None else weights[selection],
                )
            bins.append(row)
        result[name] = {"bin_edges": recorded_edges, "bins": bins}
    return result


def _kinematics(vectors: np.ndarray) -> dict[str, np.ndarray]:
    return four_vector_observables(vectors)


def _wrapped_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return wrapped_delta(first, second)


def _point_kinematic_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict:
    finite = np.isfinite(predicted).all(axis=-1) & np.isfinite(truth).all(axis=-1)
    result = {"events": int(finite.sum())}
    if not finite.any():
        return result
    predicted_kinematics = _kinematics(predicted[finite])
    truth_kinematics = _kinematics(truth[finite])
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


def truth_oracle_support(source: h5py.File) -> np.ndarray:
    """Return events for which every truth-conditioned parent is reconstructable."""
    raw_roles = source.attrs.get("invisible_roles")
    if isinstance(raw_roles, bytes):
        raw_roles = raw_roles.decode("utf-8")
    try:
        roles = json.loads(raw_roles)
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise ValueError("Prediction has invalid invisible_roles metadata") from error
    if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
        raise ValueError("Prediction has invalid invisible_roles metadata")

    count = len(source["EVENT_ID"])
    support = np.ones(count, dtype=bool)
    for role in roles:
        if "." not in role:
            raise ValueError(f"Invisible role {role!r} has no parent node")
        node = role.rsplit(".", 1)[0]
        name = f"VISIBLE/nodes/{node}/reconstructable"
        if name not in source:
            raise ValueError(
                f"Truth-oracle support requires missing prediction field {name}"
            )
        values = source[name][:]
        if values.shape != (count,) or values.dtype != np.bool_:
            raise ValueError(f"Prediction field {name} has invalid support values")
        support &= values
    return support


def _posterior_kinematic_metrics(
    predicted: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    """Posterior calibration for composite mass and pT on valid support."""
    if predicted.ndim != 3 or predicted.shape[2] != 4 or truth.shape != predicted.shape[::2]:
        raise ValueError("Composite posterior must be [N,S,4] with truth [N,4]")
    sample_finite = np.isfinite(predicted).all(axis=2)
    truth_finite = np.isfinite(truth).all(axis=1)
    eligible = truth_finite & sample_finite.any(axis=1)
    result = {
        "truth_reconstructable_events": int(truth_finite.sum()),
        "posterior_support_fraction": float(sample_finite.mean()),
        "events_with_posterior_support": int(eligible.sum()),
    }
    if not eligible.any():
        return result
    local_weights = _normalised_weights(
        predicted,
        None if weights is None else weights,
    )[eligible]
    local_valid = sample_finite[eligible]
    local_weights = np.where(local_valid, local_weights, 0.0)
    totals = local_weights.sum(axis=1, keepdims=True)
    local_weights = local_weights / totals
    safe = np.where(local_valid[..., None], predicted[eligible], 0.0)
    predicted_kinematics = _kinematics(safe)
    truth_kinematics = _kinematics(truth[eligible])
    for observable in ("mass", "pt", "eta", "rapidity", "phi"):
        values = predicted_kinematics[observable]
        target = truth_kinematics[observable]
        valid = np.isfinite(target) & np.isfinite(values).all(axis=1)
        if not valid.any():
            result[observable] = {"events": 0}
            continue
        if observable == "phi":
            values = _wrapped_delta(values, target[:, None])
            target = np.zeros_like(target)
        result[observable] = {
            "events": int(valid.sum()),
            **_scalar_posterior_metrics(
                values[valid], target[valid], local_weights[valid]
            ),
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
    projected = group["NEUTRINO_PROJECTED"][:]

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
        "projected_sum_px": float(projected[..., 1].sum(axis=1).mean()),
        "projected_sum_py": float(projected[..., 2].sum(axis=1).mean()),
    }


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
            oracle_support = (
                truth_oracle_support(source) if conditioning == "truth" else None
            )
            if oracle_support is not None:
                if not oracle_support.any():
                    raise ValueError(
                        "Truth-oracle prediction has no reconstructable evaluation events"
                    )
                result["truth_oracle_support"] = {
                    "definition": "all invisible parent nodes reconstructable",
                    "events": int(oracle_support.sum()),
                    "fraction": float(oracle_support.mean()),
                }

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
            metric_samples = samples[:, 0]
            metric_truth = truth
            if conditioning == "truth":
                metric_samples = metric_samples[oracle_support]
                metric_truth = metric_truth[oracle_support]
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
                    metric_samples, metric_truth, roles
                ),
            }
            projection_metrics = _met_exact_projection_metrics(source)
            if projection_metrics is not None:
                projected_truth = source[
                    "TRUTH/MET_EXACT_PROJECTION/NEUTRINO_PROJECTED"
                ][:]
                if conditioning == "truth":
                    projected_truth = projected_truth[oracle_support]
                invisible["truth_projection"] = projection_metrics
                invisible["top1_assignment_conditioned_vs_projected_target"] = (
                    _posterior_metrics(metric_samples, projected_truth, roles)
                )

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
                diagnostic_samples = flat
                diagnostic_truth = truth
                diagnostic_weights = weights
                diagnostic_selection = np.ones(len(truth), dtype=bool)
            else:
                invisible["truncated_assignment_weighted_mixture"] = {
                    "same_as_single_generator_posterior": True,
                    **_posterior_metrics(metric_samples, metric_truth, roles),
                }
                diagnostic_samples = metric_samples
                diagnostic_truth = metric_truth
                diagnostic_weights = None
                diagnostic_selection = (
                    oracle_support
                    if conditioning == "truth"
                    else np.ones(len(truth), dtype=bool)
                )
            node_entropy = np.stack(
                [nodes[name]["entropy"][:] for name in nodes], axis=1
            ).mean(axis=1)
            node_margin = np.stack(
                [nodes[name]["margin"][:] for name in nodes], axis=1
            ).mean(axis=1)
            reconstructable = np.stack(
                [nodes[name]["reconstructable"][:] for name in nodes], axis=1
            ).all(axis=1)
            conditional_variables = {
                "njet": njet[diagnostic_selection],
                "assignment_entropy": node_entropy[diagnostic_selection],
                "assignment_margin": node_margin[diagnostic_selection],
                "retained_assignment_mass": retained_mass[diagnostic_selection],
                "reconstructable": reconstructable[diagnostic_selection],
            }
            invisible["conditional_calibration"] = _conditional_posterior_metrics(
                diagnostic_samples,
                diagnostic_truth,
                roles,
                conditional_variables,
                weights=diagnostic_weights,
            )

            normalised_weights = _normalised_weights(
                diagnostic_samples, diagnostic_weights
            )
            posterior_mean = np.sum(
                diagnostic_samples * normalised_weights[..., None, None], axis=1
            )
            posterior_width = np.sqrt(
                np.sum(
                    normalised_weights[..., None, None]
                    * (diagnostic_samples - posterior_mean[:, None]) ** 2,
                    axis=1,
                )
            ).mean(axis=(1, 2))
            first_values = diagnostic_samples[:, :, 0, 0]
            first_truth = diagnostic_truth[:, 0, 0]
            low = _weighted_quantile_rows(first_values, normalised_weights, 0.16)
            high = _weighted_quantile_rows(first_values, normalised_weights, 0.84)
            covered = (first_truth >= low) & (first_truth <= high)
            score_inputs = reliability_scores(
                entropy=node_entropy[diagnostic_selection],
                margin=node_margin[diagnostic_selection],
                retained_mass=retained_mass[diagnostic_selection],
                posterior_width=posterior_width,
            )
            assignment_correct = truth_topk[:, 0][diagnostic_selection]
            result["reliability"] = {
                name: selective_risk_curve(
                    score,
                    fractions=RELIABILITY_FRACTIONS,
                    assignment_correct=assignment_correct,
                    covered=covered,
                )
                for name, score in score_inputs.items()
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
                    ),
                    "top1_assignment_posterior_distribution": (
                        _posterior_kinematic_metrics(
                            predicted_samples[:, 0], truth_vector
                        )
                    ),
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
                    batch, top_k, posterior, dimensions = predicted_samples.shape
                    mixture = predicted_samples.reshape(
                        batch, top_k * posterior, dimensions
                    )
                    mixture_weights = np.repeat(
                        probabilities[:, :, None] / posterior,
                        posterior,
                        axis=2,
                    ).reshape(batch, top_k * posterior)
                    node_result["assignment_weighted_posterior_distribution"] = (
                        _posterior_kinematic_metrics(
                            mixture, truth_vector, mixture_weights
                        )
                    )
                physics[node] = node_result

            root_order = json.loads(source.attrs.get("root_order", "[]"))
            if root_order and all(root in source["COMPOSITES"] for root in root_order):
                root_samples = {
                    root: source[f"COMPOSITES/{root}"][:, 0] for root in root_order
                }
                root_truth = {
                    root: source[f"TRUTH/COMPOSITES/{root}"][:] for root in root_order
                }
                sample_observables = root_system_observables(root_samples, root_order)
                truth_observables = root_system_observables(root_truth, root_order)
                root_result = {}
                unit_weights = np.full(
                    next(iter(root_samples.values())).shape[:2],
                    1.0 / next(iter(root_samples.values())).shape[1],
                )
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
                                values[valid], target[valid], unit_weights[valid]
                            ),
                        }
                    )
                physics["root_system"] = root_result

            if "OBSERVED/MET" in source and "INVISIBLE/samples" in source:
                invisible_samples = source["INVISIBLE/samples"][:]
                if conditioning == "hypothesis":
                    branch = assignment_probabilities / retained_mass[:, None]
                    batch_size, branches, posterior = invisible_samples.shape[:3]
                    closure_samples = invisible_samples.reshape(
                        batch_size, branches * posterior, *invisible_samples.shape[3:]
                    )
                    closure_weights = np.repeat(
                        branch[:, :, None] / posterior, posterior, axis=2
                    ).reshape(batch_size, branches * posterior)
                else:
                    closure_samples = invisible_samples[:, 0]
                    closure_weights = None
                met = source["OBSERVED/MET"][:]
                physics["met_closure"] = met_closure(
                    closure_samples, met[:, 1:3], closure_weights
                )
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
