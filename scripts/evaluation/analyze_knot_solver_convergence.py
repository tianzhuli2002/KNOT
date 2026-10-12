#!/usr/bin/env python3
"""Summarize stored KNOT posterior solver-convergence runs on paired events."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

RUNS = ("K32-Heun8", "K128-Heun8", "K128-Heun16", "K128-Heun32")
PAIR_COMPARISONS = (
    ("K32-Heun8", "K128-Heun8"),
    ("K128-Heun8", "K128-Heun16"),
    ("K128-Heun16", "K128-Heun32"),
)
SIDES = ("top", "antitop")
COMPONENTS = ("E", "px", "py", "pz")
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 42011
JOINT_SHUFFLE_SEED = 42012
DECAY_SHUFFLE_SEED = 42012
CALIBRATION_LEVELS = (0.50, 0.68, 0.90)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def invariant_mass(p4: np.ndarray, name: str) -> np.ndarray:
    mass2 = p4[..., 0] ** 2 - np.square(p4[..., 1:]).sum(axis=-1)
    if not np.isfinite(mass2).all():
        raise FloatingPointError(f"{name} has non-finite invariant mass squared")
    if np.any(mass2 < -1.0e-5):
        first = np.argwhere(mass2 < -1.0e-5)[0].tolist()
        raise ValueError(f"{name} has a spacelike four-vector at index {first}")
    return np.sqrt(np.maximum(mass2, 0.0))


def eta(p4: np.ndarray, name: str) -> np.ndarray:
    pt = np.hypot(p4[..., 1], p4[..., 2])
    if not np.isfinite(pt).all() or np.any(pt <= 0):
        raise ValueError(f"{name} has non-finite or zero pT")
    return np.arcsinh(p4[..., 3] / pt)


def transverse_momentum(p4: np.ndarray) -> np.ndarray:
    return np.hypot(p4[..., 1], p4[..., 2])


def wrapped_delta_phi(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (left - right)))


def load_run(root: Path, run: str) -> dict:
    run_dir = root / run
    posterior_path = run_dir / "posterior.h5"
    metrics_path = run_dir / "metrics.json"
    for path in (posterior_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required convergence input is absent: {path}")
    required = (
        "EVENT_ID",
        "TRUTH/neutrinos",
        "TRUTH/parents",
        "TRUTH/assignment_support",
        "invisible",
        "parents",
        "measured_b/sample_four_vectors",
        "measured_b/truth_four_vectors",
        "measured/leptons",
    )
    with h5py.File(posterior_path, "r") as source:
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"{posterior_path} lacks required fields: {missing}")
        if source.attrs.get("sealed_test_used") not in (False, 0):
            raise ValueError(f"{posterior_path} is not marked sealed_test_used=false")
        arrays = {name: source[name][:] for name in required}
    metrics = json.loads(metrics_path.read_text())
    ids = arrays["EVENT_ID"]
    truth_n = arrays["TRUTH/neutrinos"]
    truth_p = arrays["TRUTH/parents"]
    nu = arrays["invisible"]
    parent = arrays["parents"]
    b_draws = arrays["measured_b/sample_four_vectors"]
    truth_b = arrays["measured_b/truth_four_vectors"]
    leptons = arrays["measured/leptons"]
    n_events = len(ids)
    if ids.ndim != 1 or len(np.unique(ids)) != n_events:
        raise ValueError(f"{posterior_path} EVENT_ID must be one-dimensional and unique")
    if truth_n.shape != (n_events, 2, 4) or truth_p.shape != (n_events, 2, 4):
        raise ValueError(f"{posterior_path} violates the top/antitop truth-vector contract")
    if nu.ndim != 4 or nu.shape[0] != n_events or nu.shape[2:] != (2, 4):
        raise ValueError(f"{posterior_path} invisible samples have invalid shape {nu.shape}")
    if parent.shape != nu.shape or b_draws.shape != nu.shape:
        raise ValueError(f"{posterior_path} sampled parent, b, and neutrino shapes differ")
    if truth_b.shape != (n_events, 2, 4) or leptons.shape != (n_events, 2, 4):
        raise ValueError(f"{posterior_path} measured vectors have invalid shapes")
    if arrays["TRUTH/assignment_support"].shape != (n_events,):
        raise ValueError(f"{posterior_path} assignment support has invalid shape")
    if not arrays["TRUTH/assignment_support"].astype(bool).all():
        raise ValueError(f"{posterior_path} contains unsupported truth assignments")
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f"{posterior_path} contains non-finite values in {name}")
    return {
        "run": run,
        "path": posterior_path,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "arrays": arrays,
        "ids": ids,
        "truth_n": truth_n.astype(np.float64),
        "truth_p": truth_p.astype(np.float64),
        "nu": nu.astype(np.float64),
        "parent": parent.astype(np.float64),
        "b_draws": b_draws.astype(np.float64),
        "truth_b": truth_b.astype(np.float64),
        "leptons": leptons.astype(np.float64),
    }


def error_bootstrap(error: np.ndarray, indices: np.ndarray) -> dict:
    sampled = error[indices]
    bias = sampled.mean(axis=1)
    rmse = np.sqrt(np.square(sampled).mean(axis=1))
    return {
        "bias": [float(x) for x in np.quantile(bias, [0.025, 0.975])],
        "RMSE": [float(x) for x in np.quantile(rmse, [0.025, 0.975])],
    }


def mean_bootstrap(values: np.ndarray, indices: np.ndarray) -> list[float]:
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_error_comparison(
    left: np.ndarray,
    right: np.ndarray,
    indices: np.ndarray,
    *,
    include_bias: bool = True,
) -> dict:
    a, b = left[indices], right[indices]
    rmse_difference = np.sqrt(np.square(a).mean(axis=1)) - np.sqrt(
        np.square(b).mean(axis=1)
    )
    result = {
        "RMSE_difference_left_minus_right": float(
            np.sqrt(np.square(left).mean()) - np.sqrt(np.square(right).mean())
        ),
        "event_bootstrap_95pct_interval_for_RMSE_difference": [
            float(x) for x in np.quantile(rmse_difference, [0.025, 0.975])
        ],
    }
    if include_bias:
        bias_difference = (a - b).mean(axis=1)
        result.update(
            {
                "bias_difference_left_minus_right": float(left.mean() - right.mean()),
                "event_bootstrap_95pct_interval_for_bias_difference": [
                    float(x) for x in np.quantile(bias_difference, [0.025, 0.975])
                ],
            }
        )
    return result


def per_event_joint_diagnostics(nu: np.ndarray, parents: np.ndarray) -> dict:
    n = nu[:, :, 0, 3]
    p = parents[:, :, 0, 3]
    permutation = np.argsort(
        np.random.default_rng(JOINT_SHUFFLE_SEED).random(p.shape), axis=1, kind="stable"
    )
    shuffled = np.take_along_axis(p, permutation, axis=1)
    n_centered = n - n.mean(axis=1, keepdims=True)

    def correlation(right: np.ndarray) -> np.ndarray:
        right_centered = right - right.mean(axis=1, keepdims=True)
        denominator = np.sqrt(
            np.maximum(
                np.mean(n_centered**2, axis=1) * np.mean(right_centered**2, axis=1),
                1.0e-20,
            )
        )
        return np.mean(n_centered * right_centered, axis=1) / denominator

    return {
        "coherent_correlation": correlation(p),
        "shuffled_correlation": correlation(shuffled),
    }


def per_event_energy_scores(
    nu: np.ndarray,
    parents: np.ndarray,
    truth_n: np.ndarray,
    truth_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    samples = np.stack((nu[:, :, 0, 3], parents[:, :, 0, 3]), axis=-1)
    truth = np.stack((truth_n[:, 0, 3], truth_p[:, 0, 3]), axis=-1)
    permutation = np.argsort(
        np.random.default_rng(JOINT_SHUFFLE_SEED).random(samples.shape[:2]),
        axis=1,
        kind="stable",
    )
    shuffled = samples.copy()
    shuffled[..., 1] = np.take_along_axis(samples[..., 1], permutation, axis=1)
    count = samples.shape[1]
    coherent_scores = np.empty(len(samples), dtype=np.float64)
    shuffled_scores = np.empty(len(samples), dtype=np.float64)
    for start in range(0, len(samples), 64):
        stop = min(start + 64, len(samples))
        for values, output in (
            (samples[start:stop], coherent_scores),
            (shuffled[start:stop], shuffled_scores),
        ):
            distance_truth = np.linalg.norm(values - truth[start:stop, None], axis=-1).mean(axis=1)
            pair_distance = np.linalg.norm(
                values[:, :, None, :] - values[:, None, :, :], axis=-1
            )
            output[start:stop] = distance_truth - 0.5 * pair_distance.sum(
                axis=(1, 2)
            ) / (count * (count - 1))
    return coherent_scores, shuffled_scores


def summarize_run(run: dict, truth_mtt: np.ndarray, indices: np.ndarray) -> tuple[dict, dict]:
    nu, parent = run["nu"], run["parent"]
    truth_n, truth_p = run["truth_n"], run["truth_p"]
    draw_pair = parent.sum(axis=2)
    draw_mtt = invariant_mass(draw_pair, f"{run['run']} posterior mtt")
    event_mtt = draw_mtt.mean(axis=1)
    mtt_error = event_mtt - truth_mtt
    response_slope = float(np.polyfit(truth_mtt, event_mtt, 1)[0])
    sampled_x, sampled_y = truth_mtt[indices], event_mtt[indices]
    sampled_x_centered = sampled_x - sampled_x.mean(axis=1, keepdims=True)
    sampled_y_centered = sampled_y - sampled_y.mean(axis=1, keepdims=True)
    slope_draws = np.sum(sampled_x_centered * sampled_y_centered, axis=1) / np.sum(
        sampled_x_centered**2, axis=1
    )
    errors: dict[str, np.ndarray] = {"mtt_posterior_mean": mtt_error}
    coverage: dict[str, np.ndarray] = {}
    widths: dict[str, np.ndarray] = {}
    neutrino_summary = {}
    parent_summary = {}
    for side_index, side in enumerate(SIDES):
        truth_xyz = truth_n[:, side_index, 1:]
        mean_xyz = nu[:, :, side_index, 1:].mean(axis=1)
        xyz_error = mean_xyz - truth_xyz
        xyz_mse = np.square(xyz_error).sum(axis=1)
        errors[f"neutrino_3vector_{side}"] = np.sqrt(xyz_mse)

        draw_pt = transverse_momentum(nu[:, :, side_index])
        truth_pt = transverse_momentum(truth_n[:, side_index])
        errors[f"neutrino_pT_{side}"] = draw_pt.mean(axis=1) - truth_pt
        errors[f"neutrino_pz_{side}"] = nu[:, :, side_index, 3].mean(axis=1) - truth_n[:, side_index, 3]
        pz_draws = nu[:, :, side_index, 3]
        pz_calibration = {}
        for level in CALIBRATION_LEVELS:
            interval_tail = (1.0 - level) / 2.0
            low, high = np.quantile(
                pz_draws, [interval_tail, 1.0 - interval_tail], axis=1
            )
            coverage_name = f"neutrino_pz_{level}_{side}"
            width_name = f"neutrino_pz_{level}_width_{side}"
            coverage[coverage_name] = (
                (truth_n[:, side_index, 3] >= low)
                & (truth_n[:, side_index, 3] <= high)
            ).astype(np.float64)
            widths[width_name] = high - low
            pz_calibration[str(level)] = {
                "empirical_coverage": float(coverage[coverage_name].mean()),
                "event_bootstrap_95pct_interval_for_coverage": mean_bootstrap(
                    coverage[coverage_name], indices
                ),
                "mean_central_interval_width_GeV": float(widths[width_name].mean()),
                "event_bootstrap_95pct_interval_for_mean_width_GeV": mean_bootstrap(
                    widths[width_name], indices
                ),
            }

        parent_pt_draws = transverse_momentum(parent[:, :, side_index])
        truth_parent_pt = transverse_momentum(truth_p[:, side_index])
        errors[f"parent_pT_{side}"] = parent_pt_draws.mean(axis=1) - truth_parent_pt
        parent_eta_draws = eta(parent[:, :, side_index], f"{run['run']} {side} parent draws")
        truth_parent_eta = eta(truth_p[:, side_index], f"{run['run']} {side} truth parents")
        errors[f"parent_eta_{side}"] = parent_eta_draws.mean(axis=1) - truth_parent_eta
        neutrino_summary[side] = {
            "3vector_RMSE_GeV": float(np.sqrt(xyz_mse.mean())),
            "3vector_RMSE_event_bootstrap_95pct_interval_GeV": [
                float(x)
                for x in np.quantile(
                    np.sqrt(xyz_mse[indices].mean(axis=1)), [0.025, 0.975]
                )
            ],
            "pT": {
                "bias_GeV": float(errors[f"neutrino_pT_{side}"].mean()),
                "RMSE_GeV": float(np.sqrt(np.square(errors[f"neutrino_pT_{side}"]).mean())),
                "event_bootstrap_95pct_interval": error_bootstrap(
                    errors[f"neutrino_pT_{side}"], indices
                ),
            },
            "pz": {
                "bias_GeV": float(errors[f"neutrino_pz_{side}"].mean()),
                "RMSE_GeV": float(np.sqrt(np.square(errors[f"neutrino_pz_{side}"]).mean())),
                "event_bootstrap_95pct_interval": error_bootstrap(
                    errors[f"neutrino_pz_{side}"], indices
                ),
            },
            "pz_central_interval_calibration": pz_calibration,
        }
        parent_summary[side] = {
            "pT": {
                "bias_GeV": float(errors[f"parent_pT_{side}"].mean()),
                "RMSE_GeV": float(np.sqrt(np.square(errors[f"parent_pT_{side}"]).mean())),
                "event_bootstrap_95pct_interval": error_bootstrap(
                    errors[f"parent_pT_{side}"], indices
                ),
            },
            "eta": {
                "bias": float(errors[f"parent_eta_{side}"].mean()),
                "RMSE": float(np.sqrt(np.square(errors[f"parent_eta_{side}"]).mean())),
                "event_bootstrap_95pct_interval": error_bootstrap(
                    errors[f"parent_eta_{side}"], indices
                ),
            },
        }

    errors["neutrino_3vector_combined"] = np.sqrt(
        0.5
        * (
            np.square(errors["neutrino_3vector_top"])
            + np.square(errors["neutrino_3vector_antitop"])
        )
    )

    q90, q99 = np.quantile(truth_mtt, [0.90, 0.99])
    tail_summary = {}
    for label, threshold in (("q90", q90), ("q99", q99)):
        truth_tail = (truth_mtt > threshold).astype(np.float64)
        predicted_probability = (draw_mtt > threshold).mean(axis=1)
        denominator = truth_tail[indices].mean(axis=1)
        numerator = predicted_probability[indices].mean(axis=1)
        valid = denominator > 0
        ratios = numerator[valid] / denominator[valid]
        tail_summary[label] = {
            "truth_threshold_GeV": float(threshold),
            "truth_fraction": float(truth_tail.mean()),
            "predictive_fraction": float(predicted_probability.mean()),
            "predictive_to_truth_fraction_ratio": float(
                predicted_probability.mean() / truth_tail.mean()
            ),
            "event_bootstrap_95pct_interval_for_ratio": [
                float(x) for x in np.quantile(ratios, [0.025, 0.975])
            ],
            "predictive_quantile_GeV": float(np.quantile(draw_mtt.reshape(-1), float(label[1:]) / 100.0)),
        }
        errors[f"predictive_tail_{label}"] = predicted_probability

    mtt_coverage = {}
    mtt_widths = {}
    for level in CALIBRATION_LEVELS:
        interval_tail = (1.0 - level) / 2.0
        low, high = np.quantile(
            draw_mtt, [interval_tail, 1.0 - interval_tail], axis=1
        )
        mtt_coverage[str(level)] = (
            (truth_mtt >= low) & (truth_mtt <= high)
        ).astype(np.float64)
        mtt_widths[str(level)] = high - low
    mtt_calibration_summary = {
        level: {
            "empirical_coverage": float(values.mean()),
            "event_bootstrap_95pct_interval_for_coverage": mean_bootstrap(
                values, indices
            ),
            "mean_central_interval_width_GeV": float(mtt_widths[level].mean()),
            "event_bootstrap_95pct_interval_for_mean_width_GeV": mean_bootstrap(
                mtt_widths[level], indices
            ),
        }
        for level, values in mtt_coverage.items()
    }

    order = np.argsort(truth_mtt, kind="stable")
    deciles = []
    for decile, rows in enumerate(np.array_split(order, 10)):
        error = mtt_error[rows]
        deciles.append(
            {
                "truth_decile": decile + 1,
                "events": len(rows),
                "truth_mtt_mean_GeV": float(truth_mtt[rows].mean()),
                "posterior_mean_mtt_bias_GeV": float(error.mean()),
                "posterior_mean_mtt_RMSE_GeV": float(np.sqrt(np.square(error).mean())),
            }
        )

    joint = per_event_joint_diagnostics(nu, parent)
    energy, shuffled_energy = per_event_energy_scores(
        nu, parent, truth_n, truth_p
    )
    errors["joint_energy_coherent"] = energy
    errors["joint_energy_shuffle_penalty"] = shuffled_energy - energy
    joint_summary = {
        "coherent_mean_conditional_correlation": float(joint["coherent_correlation"].mean()),
        "shuffled_mean_conditional_correlation": float(joint["shuffled_correlation"].mean()),
        "coherent_correlation_event_bootstrap_95pct_interval": mean_bootstrap(
            joint["coherent_correlation"], indices
        ),
        "shuffled_correlation_event_bootstrap_95pct_interval": mean_bootstrap(
            joint["shuffled_correlation"], indices
        ),
        "energy_score_coherent_GeV": float(energy.mean()),
        "energy_score_shuffled_GeV": float(shuffled_energy.mean()),
        "energy_score_shuffle_minus_coherent_GeV": float((shuffled_energy - energy).mean()),
        "energy_score_shuffle_penalty_event_bootstrap_95pct_interval_GeV": mean_bootstrap(
            shuffled_energy - energy, indices
        ),
    }

    permutation = np.argsort(
        np.random.default_rng(DECAY_SHUFFLE_SEED).random(nu.shape[:2]),
        axis=1,
        kind="stable",
    )
    b_shuffled = np.take_along_axis(
        run["b_draws"], permutation[:, :, None, None], axis=1
    )
    nu_shuffled = np.take_along_axis(
        nu, permutation[:, :, None, None], axis=1
    )
    lepton = run["leptons"][:, None]

    def decay_residuals(
        b_draw: np.ndarray, n_draw: np.ndarray, label: str
    ) -> dict[str, np.ndarray]:
        chain = b_draw + lepton + n_draw
        return {
            "delta_pT": transverse_momentum(chain) - transverse_momentum(parent),
            "delta_mass": invariant_mass(chain, f"{label} daughter-chain vectors")
            - invariant_mass(parent, f"{label} parent vectors"),
            "delta_R": np.hypot(
                eta(chain, f"{label} daughter-chain vectors")
                - eta(parent, f"{label} parent vectors"),
                wrapped_delta_phi(
                    np.arctan2(chain[..., 2], chain[..., 1]),
                    np.arctan2(parent[..., 2], parent[..., 1]),
                ),
            ),
        }

    truth_chain = run["truth_b"] + run["leptons"] + truth_n
    residual_sets = {
        "truth_assignment_truth_neutrino_truth_parent": {
            "delta_pT": transverse_momentum(truth_chain)
            - transverse_momentum(truth_p),
            "delta_mass": invariant_mass(truth_chain, "truth daughter-chain vectors")
            - invariant_mass(truth_p, "after-FSR truth parents"),
            "delta_R": np.hypot(
                eta(truth_chain, "truth daughter-chain vectors")
                - eta(truth_p, "after-FSR truth parents"),
                wrapped_delta_phi(
                    np.arctan2(truth_chain[..., 2], truth_chain[..., 1]),
                    np.arctan2(truth_p[..., 2], truth_p[..., 1]),
                ),
            ),
        },
        "coherent": decay_residuals(run["b_draws"], nu, run["run"]),
        "assignment_b_shuffled": decay_residuals(b_shuffled, nu, run["run"]),
        "neutrino_parent_pair_shuffled": decay_residuals(
            run["b_draws"], nu_shuffled, run["run"]
        ),
    }
    decay_summary = {
        label: {
            name: {
                "mean_absolute": float(np.abs(value).mean()),
                "RMSE": float(np.sqrt(np.square(value).mean())),
            }
            for name, value in residuals.items()
        }
        for label, residuals in residual_sets.items()
    }
    decay_event_rmse = {
        label: np.sqrt(np.square(residuals["delta_pT"]).mean(axis=(1, 2)))
        for label, residuals in residual_sets.items()
        if label != "truth_assignment_truth_neutrino_truth_parent"
    }
    errors["decay_chain_delta_pT_RMSE_coherent"] = decay_event_rmse["coherent"]
    errors["decay_chain_assignment_b_shuffle_penalty"] = (
        decay_event_rmse["assignment_b_shuffled"] - decay_event_rmse["coherent"]
    )
    errors["decay_chain_neutrino_shuffle_penalty"] = (
        decay_event_rmse["neutrino_parent_pair_shuffled"] - decay_event_rmse["coherent"]
    )
    decay_rmse_bootstrap = np.sqrt(
        np.square(decay_event_rmse["coherent"][indices]).mean(axis=1)
    )
    decay_assignment_shuffle_bootstrap = (
        errors["decay_chain_assignment_b_shuffle_penalty"][indices].mean(axis=1)
    )
    decay_neutrino_shuffle_bootstrap = (
        errors["decay_chain_neutrino_shuffle_penalty"][indices].mean(axis=1)
    )

    mtt_error_boot = error_bootstrap(mtt_error, indices)
    summary = {
        "posterior_draws_per_event": int(nu.shape[1]),
        "assignment": run["metrics"]["assignment"],
        "posterior_mean_mtt": {
            "bias_GeV": float(mtt_error.mean()),
            "RMSE_GeV": float(np.sqrt(np.square(mtt_error).mean())),
            "event_bootstrap_95pct_interval": mtt_error_boot,
            "truth_to_posterior_mean_response_slope": response_slope,
            "event_bootstrap_95pct_interval_for_response_slope": [
                float(x) for x in np.quantile(slope_draws, [0.025, 0.975])
            ],
            "bias_and_RMSE_by_truth_mtt_decile": deciles,
        },
        "mtt_central_interval_calibration": mtt_calibration_summary,
        "stored_14d_marginal_calibration": run["metrics"]["posterior_calibration_14d"],
        "posterior_predictive_mtt_tail": tail_summary,
        "neutrino": {
            **neutrino_summary,
            "combined_3vector_RMSE_GeV": float(
                np.sqrt(
                    0.5
                    * (
                        np.square(errors["neutrino_3vector_top"])
                        + np.square(errors["neutrino_3vector_antitop"])
                    ).mean()
                )
            ),
        },
        "parents": parent_summary,
        "joint_neutrino_parent_pz": joint_summary,
        "decay_chain_consistency": {
            "definition": "delta-pT, delta-mass, and delta-R between each posterior parent and its paired sampled measured-b candidate, charge-side lepton, and neutrino; shuffled controls independently permute assignment b or neutrino draws within event",
            "summary": decay_summary,
            "coherent_delta_pT_RMSE_GeV": decay_summary["coherent"]["delta_pT"]["RMSE"],
            "coherent_delta_pT_RMSE_event_bootstrap_95pct_interval_GeV": [
                float(x) for x in np.quantile(decay_rmse_bootstrap, [0.025, 0.975])
            ],
            "assignment_b_shuffle_minus_coherent_event_mean_RMSE_GeV": float(
                errors["decay_chain_assignment_b_shuffle_penalty"].mean()
            ),
            "assignment_b_shuffle_penalty_event_bootstrap_95pct_interval_GeV": [
                float(x)
                for x in np.quantile(
                    decay_assignment_shuffle_bootstrap, [0.025, 0.975]
                )
            ],
            "neutrino_shuffle_minus_coherent_event_mean_RMSE_GeV": float(
                errors["decay_chain_neutrino_shuffle_penalty"].mean()
            ),
            "neutrino_shuffle_penalty_event_bootstrap_95pct_interval_GeV": [
                float(x)
                for x in np.quantile(
                    decay_neutrino_shuffle_bootstrap, [0.025, 0.975]
                )
            ],
        },
        "runtime_seconds": run["metrics"]["runtime_seconds"],
        "sealed_test_used": False,
    }
    return summary, {
        **errors,
        **{f"coverage:{name}": value for name, value in coverage.items()},
        **{f"width:{name}": value for name, value in widths.items()},
        **{f"mtt_coverage:{name}": value for name, value in mtt_coverage.items()},
        **{f"mtt_width:{name}": value for name, value in mtt_widths.items()},
        "joint_coherent_correlation": joint["coherent_correlation"],
        "joint_shuffle_correlation": joint["shuffled_correlation"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAPS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    root, output = args.input_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite convergence summary: {output}")
    runs = {name: load_run(root, name) for name in RUNS}
    ids = runs[RUNS[0]]["ids"]
    for name in RUNS[1:]:
        run = runs[name]
        if not np.array_equal(ids, run["ids"]):
            raise ValueError(f"{name} ordered EVENT_ID list differs from {RUNS[0]}")
        for truth_name in ("truth_n", "truth_p"):
            if not np.array_equal(runs[RUNS[0]][truth_name], run[truth_name]):
                raise ValueError(f"{name} {truth_name} differs from the common truth")
    truth_parent = runs[RUNS[0]]["truth_p"]
    truth_mtt = invariant_mass(truth_parent.sum(axis=1), "common after-FSR truth mtt")
    indices = np.random.default_rng(args.bootstrap_seed).integers(
        0, len(ids), size=(args.bootstrap_resamples, len(ids))
    )
    metrics, event_metrics = {}, {}
    for name in RUNS:
        metrics[name], event_metrics[name] = summarize_run(
            runs[name], truth_mtt, indices
        )

    comparisons = {}
    paired_metric_names = (
        "mtt_posterior_mean",
        "neutrino_3vector_top",
        "neutrino_3vector_antitop",
        "neutrino_3vector_combined",
        "neutrino_pT_top",
        "neutrino_pT_antitop",
        "neutrino_pz_top",
        "neutrino_pz_antitop",
        "parent_pT_top",
        "parent_pT_antitop",
        "parent_eta_top",
        "parent_eta_antitop",
        "decay_chain_delta_pT_RMSE_coherent",
    )
    paired_mean_names = (
        "predictive_tail_q90",
        "predictive_tail_q99",
        "joint_energy_coherent",
        "joint_energy_shuffle_penalty",
        "joint_coherent_correlation",
        "joint_shuffle_correlation",
        "decay_chain_assignment_b_shuffle_penalty",
        "decay_chain_neutrino_shuffle_penalty",
        *(f"coverage:neutrino_pz_{level}_{side}" for level in (0.50, 0.68, 0.90) for side in SIDES),
        *(f"width:neutrino_pz_{level}_width_{side}" for level in (0.50, 0.68, 0.90) for side in SIDES),
        *(f"mtt_coverage:{level}" for level in ("0.5", "0.68", "0.9")),
        *(f"mtt_width:{level}" for level in ("0.5", "0.68", "0.9")),
    )
    for left, right in PAIR_COMPARISONS:
        comparisons[f"{left}_minus_{right}"] = {
            "unit": "paired EVENT_ID bootstrap; posterior draws remain fixed within each event",
            "point_error_metrics": {
                name: paired_error_comparison(
                    event_metrics[left][name],
                    event_metrics[right][name],
                    indices,
                    include_bias=not name.startswith(
                        ("neutrino_3vector_", "decay_chain_")
                    ),
                )
                for name in paired_metric_names
            },
            "mean_metrics": {
                name: {
                    "difference_left_minus_right": float(
                        event_metrics[left][name].mean()
                        - event_metrics[right][name].mean()
                    ),
                    "event_bootstrap_95pct_interval": [
                        float(x)
                        for x in np.quantile(
                            (
                                event_metrics[left][name][indices]
                                - event_metrics[right][name][indices]
                            ).mean(axis=1),
                            [0.025, 0.975],
                        )
                    ],
                }
                for name in paired_mean_names
            },
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "completed",
        "analysis": "stored KNOT posterior solver convergence; no inference recomputed",
        "inputs": {
            name: {
                "posterior": str(runs[name]["path"]),
                "posterior_sha256": sha256(runs[name]["path"]),
                "metrics": str(runs[name]["metrics_path"]),
                "metrics_sha256": sha256(runs[name]["metrics_path"]),
                "events": len(ids),
                "posterior_draws_per_event": int(runs[name]["nu"].shape[1]),
                "ordered_EVENT_ID_sha256": ids_digest(runs[name]["ids"]),
            }
            for name in RUNS
        },
        "sealed_test_used": False,
        "truth_contract": "top/antitop after-FSR parents and direct neutrinos; role order top then antitop",
        "bootstrap": {
            "unit": "event",
            "replicates": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "paired_comparison_seed": args.bootstrap_seed,
        },
        "metrics": metrics,
        "paired_comparisons": comparisons,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": payload["status"], "events": len(ids), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
