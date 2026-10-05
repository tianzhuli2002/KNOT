"""Compare frozen KNOT M/C/J validation posteriors on a paired event sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FREEZE = ROOT / "configs/experiments/knot_scientific_closure/matched_production_freeze.json"
DEFAULT_EVALUATION = ROOT / "results/experiments/knot/matched-production-v1/evaluation-v1"
COLORS = {"M": "#0072B2", "C": "#D55E00", "J": "#009E73"}
VARIANTS = ("M", "C", "J")
SPIN_NAMES = (
    "delta_phi_ll",
    "cos_theta_plus",
    "cos_theta_minus",
    "helicity_product",
    "cos_opening",
)


def sha256_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def four_vector_mass(vector):
    vector = np.asarray(vector)
    return np.sqrt(np.maximum(vector[..., 0] ** 2 - np.sum(vector[..., 1:] ** 2, axis=-1), 0.0))


def pt(vector):
    return np.hypot(vector[..., 1], vector[..., 2])


def eta(vector):
    return np.arcsinh(vector[..., 3] / np.maximum(pt(vector), 1e-12))


def phi(vector):
    return np.arctan2(vector[..., 2], vector[..., 1])


def wrapped_delta(value):
    return np.angle(np.exp(1j * value))


def rapidity(vector):
    plus, minus = vector[..., 0] + vector[..., 3], vector[..., 0] - vector[..., 3]
    if np.any(plus <= 0) or np.any(minus <= 0):
        raise ValueError("Nonpositive E±pz encountered in a KNOT posterior")
    return 0.5 * np.log(plus / minus)


def read_variant(path: Path, variant: str):
    posterior_path = path / variant / "posterior.h5"
    metrics_path = path / variant / "metrics.json"
    if not posterior_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Missing completed {variant} evaluation under {path}")
    with h5py.File(posterior_path, "r") as source:
        if source.attrs.get("sealed_test_used") not in (False, 0):
            raise ValueError(f"{variant} posterior is not marked sealed_test_used=false")
        required = (
            "EVENT_ID",
            "sample_index",
            "assignment_state",
            "assignment_candidates",
            "assignment_support",
            "assignment_log_probability",
            "invisible",
            "parents",
            "TRUTH/neutrinos",
            "TRUTH/parents",
            "TRUTH/assignment_state",
            "TRUTH/assignment_support",
            "measured/leptons",
            "measured_b/sample_four_vectors",
            "measured_b/truth_four_vectors",
            "diagnostics/assignment_entropy_nats",
            "diagnostics/assignment_MAP_state",
            "diagnostics/assignment_MAP_node_correct",
            "diagnostics/assignment_truth_NLL",
        )
        absent = [name for name in required if name not in source]
        if absent:
            raise ValueError(f"{variant} posterior lacks required fields: {absent}")
        values = {name: source[name][:] for name in required}
        for name in (
            "diagnostics/assignment_MAP_probability",
            "diagnostics/assignment_MAP_leaf_correct/top_b",
            "diagnostics/assignment_MAP_leaf_correct/antitop_b",
            "oracle/truth_assignment_truth_neutrino/parents",
            "oracle/predicted_assignment_truth_neutrino/parents",
            "oracle/neutrino_truth_assignment/invisible",
            "oracle/neutrino_predicted_assignment/invisible",
        ):
            if name in source:
                values[name] = source[name][:]
        values["attrs"] = {
            key: source.attrs[key]
            for key in ("factorisation", "posterior_semantics", "assignment_nodes", "spin_definition")
            if key in source.attrs
        }
        if "spin" in source:
            for kind in ("truth", "paired", "shuffled"):
                for name in SPIN_NAMES:
                    key = f"spin/{kind}/{name}"
                    if key not in source:
                        raise ValueError(f"{variant} posterior lacks spin observable {key}")
                    values[key] = source[key][:]
    metrics = json.loads(metrics_path.read_text())
    if metrics["variant"] != variant or metrics["sealed_test_used"] is not False:
        raise ValueError(f"{variant} metric manifest has inconsistent variant/test status")
    return values, metrics


def join_measured_categories(event_ids: np.ndarray, source_path: Path):
    """Read only ID, njet and two-lepton channel fields from the train+validation file."""
    with h5py.File(source_path, "r") as source:
        if "SPLIT" not in source or not np.isin(source["SPLIT"][:], (0, 1)).all():
            raise ValueError("Measured input includes rows outside train+validation")
        ids = source["EVENT_ID"][:]
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        index = np.searchsorted(sorted_ids, event_ids)
        if np.any(index >= len(sorted_ids)) or not np.array_equal(sorted_ids[index], event_ids):
            raise ValueError("A validation EVENT_ID is absent from train+validation measured input")
        rows = order[index]
        if len(np.unique(rows)) != len(rows):
            raise ValueError("Selected validation EVENT_ID values are not unique")
        read_order = np.argsort(rows, kind="stable")
        sorted_rows = rows[read_order]
        globals_sorted = source["INPUTS/GLOBAL"][sorted_rows]
        leptons_sorted = source["INPUTS/LEPTON"][sorted_rows]
        inverse = np.argsort(read_order)
        globals_selected = globals_sorted[inverse]
        leptons_selected = leptons_sorted[inverse]
    njet = np.rint(globals_selected["njet"]).astype(np.int16)
    is_el = leptons_selected["is_el"] > 0.5
    is_mu = leptons_selected["is_mu"] > 0.5
    if not np.all(is_el | is_mu) or not np.all((is_el ^ is_mu).sum(axis=1) == 2):
        raise ValueError("Selected events contain an invalid dilepton channel assignment")
    channel = np.where(is_el.all(axis=1), "ee", np.where(is_mu.all(axis=1), "mumu", "emu"))
    return njet, channel


def kinematic_features(posterior, truth_n, truth_p):
    n_draw = posterior["invisible"]
    p_draw = posterior["parents"]
    pair_draw = p_draw.sum(axis=2)
    truth_pair = truth_p.sum(axis=1)
    feature_draws = {
        "top_pT": pt(p_draw[:, :, 0]),
        "antitop_pT": pt(p_draw[:, :, 1]),
        "top_mass": four_vector_mass(p_draw[:, :, 0]),
        "antitop_mass": four_vector_mass(p_draw[:, :, 1]),
        "mttbar": four_vector_mass(pair_draw),
        "pTttbar": pt(pair_draw),
        "yttbar": rapidity(pair_draw),
        "abs_delta_phi_tt": np.abs(wrapped_delta(phi(p_draw[:, :, 0]) - phi(p_draw[:, :, 1]))),
        "delta_y_tt": rapidity(p_draw[:, :, 0]) - rapidity(p_draw[:, :, 1]),
        "nu_top_pz": n_draw[:, :, 0, 3],
        "nu_antitop_pz": n_draw[:, :, 1, 3],
        "nu_top_pT": pt(n_draw[:, :, 0]),
        "nu_antitop_pT": pt(n_draw[:, :, 1]),
    }
    truth_features = {
        "top_pT": pt(truth_p[:, 0]),
        "antitop_pT": pt(truth_p[:, 1]),
        "top_mass": four_vector_mass(truth_p[:, 0]),
        "antitop_mass": four_vector_mass(truth_p[:, 1]),
        "mttbar": four_vector_mass(truth_pair),
        "pTttbar": pt(truth_pair),
        "yttbar": rapidity(truth_pair),
        "abs_delta_phi_tt": np.abs(wrapped_delta(phi(truth_p[:, 0]) - phi(truth_p[:, 1]))),
        "delta_y_tt": rapidity(truth_p[:, 0]) - rapidity(truth_p[:, 1]),
        "nu_top_pz": truth_n[:, 0, 3],
        "nu_antitop_pz": truth_n[:, 1, 3],
        "nu_top_pT": pt(truth_n[:, 0]),
        "nu_antitop_pT": pt(truth_n[:, 1]),
    }
    point_features = {name: values.mean(axis=1) for name, values in feature_draws.items()}
    for side, index in (("top", 0), ("antitop", 1)):
        de = eta(n_draw[:, :, index]) - eta(truth_n[:, None, index])
        dp = wrapped_delta(phi(n_draw[:, :, index]) - phi(truth_n[:, None, index]))
        feature_draws[f"nu_{side}_deltaR"] = np.hypot(de, dp)
        point_features[f"nu_{side}_deltaR"] = feature_draws[f"nu_{side}_deltaR"].mean(axis=1)
        truth_features[f"nu_{side}_deltaR"] = np.zeros(len(truth_n))
    return feature_draws, point_features, truth_features


def point_metric(prediction, truth, *, periodic=False):
    delta = wrapped_delta(prediction - truth) if periodic else prediction - truth
    return {
        "bias": float(delta.mean()),
        "RMSE": float(np.sqrt(np.mean(delta**2))),
        "residual_std": float(delta.std()),
    }


def calibration(samples, truth):
    if len(truth) == 0:
        return {
            "coverage": {"0.5": None, "0.68": None, "0.9": None},
            "rank_PIT_histogram_10_bins": [0] * 10,
            "width_vs_median_absolute_error": [],
        }
    coverage = {}
    for level in (0.50, 0.68, 0.90):
        tail = (1.0 - level) / 2.0
        low, high = np.quantile(samples, [tail, 1.0 - tail], axis=1)
        coverage[str(level)] = float(((truth >= low) & (truth <= high)).mean())
    rank = (samples < truth[:, None]).sum(axis=1) + 0.5 * (samples == truth[:, None]).sum(axis=1)
    pit = (rank + 0.5) / (samples.shape[1] + 1.0)
    hist = np.histogram(pit, bins=np.linspace(0, 1, 11))[0]
    width = np.quantile(samples, 0.84, axis=1) - np.quantile(samples, 0.16, axis=1)
    center = np.median(samples, axis=1)
    order = np.argsort(width, kind="stable")
    groups = np.array_split(order, min(5, len(order))) if len(order) else []
    return {
        "coverage": coverage,
        "rank_PIT_histogram_10_bins": hist.astype(int).tolist(),
        "width_vs_median_absolute_error": [
            {
                "events": len(group),
                "mean_68_width": float(width[group].mean()),
                "mean_absolute_error": float(np.abs(center[group] - truth[group]).mean()),
            }
            for group in groups
            if len(group)
        ],
    }


def energy_score_by_event(samples, truth, chunk=256):
    output = np.empty(len(samples), dtype=np.float64)
    count = samples.shape[1]
    for start in range(0, len(samples), chunk):
        stop = min(start + chunk, len(samples))
        block = samples[start:stop]
        distance_truth = np.linalg.norm(block - truth[start:stop, None], axis=-1).mean(axis=1)
        distance_pairs = np.linalg.norm(block[:, :, None, :] - block[:, None, :, :], axis=-1)
        output[start:stop] = distance_truth - 0.5 * distance_pairs.sum(axis=(1, 2)) / (count * (count - 1))
    return output


def joint_diagnostics(posterior, truth_n, truth_p, seed):
    n, p = posterior["invisible"], posterior["parents"]
    values = np.stack((n[:, :, 0, 3], p[:, :, 0, 3], n[:, :, 1, 3], p[:, :, 1, 3]), axis=-1)
    truth = np.stack((truth_n[:, 0, 3], truth_p[:, 0, 3], truth_n[:, 1, 3], truth_p[:, 1, 3]), axis=-1)
    permutation = np.argsort(np.random.default_rng(seed).random(values.shape[:2]), axis=1, kind="stable")
    shuffled = values.copy()
    shuffled[:, :, 1] = np.take_along_axis(values[:, :, 1], permutation, axis=1)
    shuffled[:, :, 3] = np.take_along_axis(values[:, :, 3], permutation, axis=1)
    coherent_score = energy_score_by_event(values, truth)
    shuffled_score = energy_score_by_event(shuffled, truth)
    correlation = {}
    for side, n_index, p_index in (("top", 0, 1), ("antitop", 2, 3)):
        left, right = values[:, :, n_index], values[:, :, p_index]
        left_centered = left - left.mean(axis=1, keepdims=True)
        right_centered = right - right.mean(axis=1, keepdims=True)
        corr = np.mean(left_centered * right_centered, axis=1) / np.sqrt(
            np.maximum(np.mean(left_centered**2, axis=1) * np.mean(right_centered**2, axis=1), 1e-20)
        )
        correlation[side] = corr
    marginal_change = {
        "top_neutrino_pz_mean": float(abs(values[:, :, 0].mean() - shuffled[:, :, 0].mean())),
        "top_parent_pz_mean": float(abs(values[:, :, 1].mean() - shuffled[:, :, 1].mean())),
        "antitop_neutrino_pz_variance": float(abs(values[:, :, 2].var() - shuffled[:, :, 2].var())),
        "antitop_parent_pz_variance": float(abs(values[:, :, 3].var() - shuffled[:, :, 3].var())),
    }
    return {
        "coherent_energy_score_by_event": coherent_score,
        "shuffled_energy_score_by_event": shuffled_score,
        "shuffle_minus_coherent_by_event": shuffled_score - coherent_score,
        "correlation_by_side": correlation,
        "coherent_energy_score_GeV": float(coherent_score.mean()),
        "shuffled_energy_score_GeV": float(shuffled_score.mean()),
        "shuffle_minus_coherent_GeV": float((shuffled_score - coherent_score).mean()),
        "correlation_mean_by_side": {key: float(value.mean()) for key, value in correlation.items()},
        "marginal_change_after_shuffle": marginal_change,
    }


def decay_chain_diagnostics(posterior, truth_n, truth_p, leptons, seed):
    b = posterior["measured_b/sample_four_vectors"]
    n, p = posterior["invisible"], posterior["parents"]
    lepton = leptons[:, None]
    permutation = np.argsort(np.random.default_rng(seed).random(n.shape[:2]), axis=1, kind="stable")
    b_shuffled = np.take_along_axis(b, permutation[:, :, None, None], axis=1)
    n_shuffled = np.take_along_axis(n, permutation[:, :, None, None], axis=1)

    def residuals(b_draw, n_draw, p_draw):
        chain = b_draw + lepton + n_draw
        return {
            "delta_pt": pt(chain) - pt(p_draw),
            "delta_mass": four_vector_mass(chain) - four_vector_mass(p_draw),
            "delta_R": np.hypot(
                eta(chain) - eta(p_draw), wrapped_delta(phi(chain) - phi(p_draw))
            ),
        }

    coherent = residuals(b, n, p)
    assignment_shuffled = residuals(b_shuffled, n, p)
    np_shuffled = residuals(b, n_shuffled, p)
    truth_support = posterior["TRUTH/assignment_support"].astype(bool)
    true_chain = posterior["measured_b/truth_four_vectors"][truth_support] + leptons[truth_support] + truth_n[truth_support]
    truth_baseline = {
        "delta_pt": pt(true_chain) - pt(truth_p[truth_support]),
        "delta_mass": four_vector_mass(true_chain) - four_vector_mass(truth_p[truth_support]),
        "delta_R": np.hypot(
            eta(true_chain) - eta(truth_p[truth_support]),
            wrapped_delta(phi(true_chain) - phi(truth_p[truth_support])),
        ),
    }
    summaries = {}
    for label, data in (
        ("truth_assignment_truth_neutrino_truth_parent", truth_baseline),
        ("coherent", coherent),
        ("assignment_b_shuffled", assignment_shuffled),
        ("neutrino_parent_pair_shuffled", np_shuffled),
    ):
        summaries[label] = {
            name: {
                "mean_absolute": float(np.abs(value).mean()),
                "RMSE": float(np.sqrt(np.mean(value**2))),
            }
            for name, value in data.items()
        }
    return {
        "summary": summaries,
        "pt_RMSE_by_event": {"coherent": np.sqrt(np.mean(coherent["delta_pt"] ** 2, axis=(1, 2))),
                              "assignment_b_shuffled": np.sqrt(np.mean(assignment_shuffled["delta_pt"] ** 2, axis=(1, 2))),
                              "neutrino_parent_pair_shuffled": np.sqrt(np.mean(np_shuffled["delta_pt"] ** 2, axis=(1, 2)))},
    }


def metric_pack(posterior, metrics, truth_n, truth_p, seed):
    draws, points, truth = kinematic_features(posterior, truth_n, truth_p)
    point_metrics = {}
    for name, value in points.items():
        point_metrics[name] = point_metric(value, truth[name], periodic=name in ())
    calibration_names = ("nu_top_pz", "top_pT", "mttbar")
    calibration_results = {
        name: calibration(draws[name], truth[name]) for name in calibration_names
    }
    support = posterior["TRUTH/assignment_support"].astype(bool)
    map_correct = posterior["diagnostics/assignment_MAP_state"] == posterior["TRUTH/assignment_state"]
    node_correct = posterior["diagnostics/assignment_MAP_node_correct"].astype(bool)
    truth_nll = posterior["diagnostics/assignment_truth_NLL"]
    assignment = {
        "truth_supported_events": int(support.sum()),
        "truth_unsupported_events": int((~support).sum()),
        "complete_MAP_accuracy_supported": float(map_correct[support].mean()),
        "truth_assignment_NLL_supported": float(truth_nll[support].mean()),
        "MAP_probability_supported": float(posterior.get("diagnostics/assignment_MAP_probability", np.full(len(support), np.nan))[support].mean()),
        "per_node_MAP_accuracy_supported": {
            name: float(node_correct[support, index].mean())
            for index, name in enumerate(json.loads(posterior["attrs"]["assignment_nodes"]))
        },
    }
    for key in ("diagnostics/assignment_MAP_leaf_correct/top_b", "diagnostics/assignment_MAP_leaf_correct/antitop_b"):
        if key in posterior:
            assignment[key.rsplit("/", 1)[-1]] = float(posterior[key][support].mean())
    joint = joint_diagnostics(posterior, truth_n, truth_p, seed)
    decay = decay_chain_diagnostics(
        posterior, truth_n, truth_p, posterior["measured/leptons"], seed + 1
    )
    return {
        "reconstruction": point_metrics,
        "posterior_calibration": calibration_results,
        "assignment": assignment,
        "joint": {key: value for key, value in joint.items() if not key.endswith("_by_event") and key != "correlation_by_side"},
        "decay_chain": decay["summary"],
        "diagnostic_arrays": {
            "draws": draws,
            "points": points,
            "truth": truth,
            "support": support,
            "map_correct": map_correct,
            "assignment_nll": truth_nll,
            "joint": joint,
            "decay": decay,
        },
    }


def wilson(successes, total, z=1.0):
    if total <= 0:
        return None
    fraction = successes / total
    denominator = 1.0 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    half = z * np.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def bootstrap_comparison(arrays, replicates, seed):
    n_events = len(next(iter(arrays.values()))["mtt_squared_error"])
    random = np.random.default_rng(seed)
    draws = random.integers(0, n_events, size=(replicates, n_events), dtype=np.int32)

    def statistic(values, index):
        if values.ndim == 1:
            return values[index].mean(axis=1)
        return values[index].mean(axis=1)

    definitions = {
        "mtt_RMSE_GeV": ("mtt_squared_error", "sqrt"),
        "mtt_bias_GeV": ("mtt_residual", "mean"),
        "pTttbar_RMSE_GeV": ("ttbar_pt_squared_error", "sqrt"),
        "top_pT_RMSE_GeV": ("top_pt_squared_error", "sqrt"),
        "nu_top_pz_RMSE_GeV": ("nu_top_pz_squared_error", "sqrt"),
        "joint_energy_score_GeV": ("joint_energy", "mean"),
        "joint_shuffle_penalty_GeV": ("joint_shuffle_penalty", "mean"),
        "decay_chain_delta_pt_RMSE_GeV": ("decay_pt_squared", "sqrt"),
        "mtt_68_coverage": ("mtt_covered_68", "mean"),
        "assignment_accuracy_supported": ("assignment_correct", "masked_mean"),
        "high_mtt_RMSE_GeV": ("high_mtt_squared_error", "masked_sqrt"),
    }
    output = {}
    for label, (field, transform) in definitions.items():
        per_variant = {}
        if transform in ("masked_mean", "masked_sqrt"):
            valid = np.flatnonzero(np.isfinite(arrays["M"][field]))
            for variant in VARIANTS[1:]:
                if not np.array_equal(np.isfinite(arrays[variant][field]), np.isfinite(arrays["M"][field])):
                    raise ValueError(f"Paired bootstrap population differs for {label}/{variant}")
            conditional_draws = valid[random.integers(0, len(valid), size=(replicates, len(valid)))]
        for variant in VARIANTS:
            values = arrays[variant][field]
            if transform == "masked_mean":
                replica = values[conditional_draws].mean(axis=1)
            elif transform == "masked_sqrt":
                replica = np.sqrt(values[conditional_draws].mean(axis=1))
            else:
                replica = statistic(values, draws)
                if transform == "sqrt":
                    replica = np.sqrt(replica)
            per_variant[variant] = replica
        output[label] = {
            "estimate": {
                variant: float(np.nanmean(arrays[variant][field]) ** (0.5 if transform in ("sqrt", "masked_sqrt") else 1.0))
                for variant in VARIANTS
            },
            "paired_bootstrap_68pct_interval": {
                variant: np.nanquantile(values, [0.16, 0.84]).astype(float).tolist()
                for variant, values in per_variant.items()
            },
            "paired_bootstrap_95pct_interval": {
                variant: np.nanquantile(values, [0.025, 0.975]).astype(float).tolist()
                for variant, values in per_variant.items()
            },
            "paired_differences_vs_M_68pct": {
                variant: None if variant == "M" else np.nanquantile(per_variant[variant] - per_variant["M"], [0.16, 0.84]).astype(float).tolist()
                for variant in VARIANTS
            },
            "paired_differences_vs_M_95pct": {
                variant: None if variant == "M" else np.nanquantile(per_variant[variant] - per_variant["M"], [0.025, 0.975]).astype(float).tolist()
                for variant in VARIANTS
            },
            "paired_differences_vs_C_68pct": {
                variant: None if variant == "C" else np.nanquantile(per_variant[variant] - per_variant["C"], [0.16, 0.84]).astype(float).tolist()
                for variant in VARIANTS
            },
            "paired_differences_vs_C_95pct": {
                variant: None if variant == "C" else np.nanquantile(per_variant[variant] - per_variant["C"], [0.025, 0.975]).astype(float).tolist()
                for variant in VARIANTS
            },
        }
    return output


def save_spectrum_figure(output, packs):
    names = ("mttbar", "pTttbar", "yttbar", "abs_delta_phi_tt", "delta_y_tt", "top_pT")
    labels = (r"$m_{t\bar t}$ [GeV]", r"$p_{T,t\bar t}$ [GeV]", r"$y_{t\bar t}$", r"$|\Delta\phi_{t\bar t}|$", r"$\Delta y_{t\bar t}$", r"$p_{T,t}$ [GeV]")
    fig, axes = plt.subplots(2, 6, figsize=(20, 7), gridspec_kw={"height_ratios": [3, 1]})
    for column, (name, label) in enumerate(zip(names, labels)):
        truth = packs["M"]["diagnostic_arrays"]["truth"][name]
        all_values = [packs[v]["diagnostic_arrays"]["draws"][name].ravel() for v in VARIANTS]
        low = float(np.quantile(truth, 0.002))
        high = float(np.quantile(truth, 0.998))
        if not high > low:
            padding = max(abs(low) * 0.05, 1.0)
            low, high = low - padding, high + padding
        if name in ("mttbar", "pTttbar", "top_pT"):
            low = max(0.0, low)
        edges = np.linspace(low, high, 41)
        truth_counts, _ = np.histogram(truth, edges)
        bin_width = np.diff(edges)
        truth_hist = np.divide(truth_counts, truth_counts.sum() * bin_width, out=np.zeros_like(bin_width), where=truth_counts.sum() > 0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        axes[0, column].step(centers, truth_hist, where="mid", color="#222222", lw=1.6, label="truth")
        for variant, samples in zip(VARIANTS, all_values):
            counts, _ = np.histogram(samples, edges)
            hist = np.divide(counts, counts.sum() * bin_width, out=np.zeros_like(bin_width), where=counts.sum() > 0)
            axes[0, column].step(centers, hist, where="mid", color=COLORS[variant], label=variant)
            ratio = np.divide(hist, truth_hist, out=np.full_like(hist, np.nan), where=truth_hist > 0)
            axes[1, column].plot(centers, ratio, color=COLORS[variant], lw=1.0)
        axes[1, column].axhline(1, color="#222222", lw=0.8, ls="--")
        axes[0, column].set_xlabel(label)
        axes[1, column].set_ylim(0.4, 1.6)
        axes[1, column].set_xlabel(label)
        if column == 0:
            axes[0, column].set_ylabel("density")
            axes[1, column].set_ylabel("model / truth")
            axes[0, column].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "spectra_with_truth_ratio.png", dpi=160)
    plt.close(fig)


def save_core_plots(output, posterior, packs, njet, channels, common_support, entropy_edges, mtt_edges):
    save_spectrum_figure(output, packs)
    colors = COLORS

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    jet_bins = np.unique(np.quantile(njet, np.linspace(0, 1, 9)).astype(int))
    if len(jet_bins) < 2:
        jet_bins = np.array([int(njet.min()), int(njet.max()) + 1])
    jet_pairs = list(pairwise(jet_bins))
    groups = [(int(low), int(high), (njet >= low) & (njet <= high if i == len(jet_pairs)-1 else njet < high)) for i, (low, high) in enumerate(jet_pairs)]
    for variant, axis in zip(VARIANTS, axes):
        support = posterior[variant]["TRUTH/assignment_support"].astype(bool)
        node_correct = posterior[variant]["diagnostics/assignment_MAP_node_correct"].astype(bool).all(axis=1)
        xs, accuracy, low, high, support_fraction = [], [], [], [], []
        for low_edge, high_edge, mask in groups:
            xs.append(0.5 * (low_edge + high_edge))
            total = int(mask.sum())
            supported = mask & support
            count = int(supported.sum())
            fraction = count / total if total else np.nan
            support_fraction.append(fraction)
            accuracy.append(float(node_correct[supported].mean()) if count else np.nan)
            interval = wilson(int(node_correct[supported].sum()), count)
            low.append(accuracy[-1] - interval[0] if interval else np.nan)
            high.append(interval[1] - accuracy[-1] if interval else np.nan)
        axis.errorbar(xs, accuracy, yerr=[low, high], fmt="o-", color=colors[variant], label="MAP accuracy, supported")
        axis.plot(xs, support_fraction, "s--", color="#555555", label="truth support")
        axis.set_title(f"KNOT-{variant}")
        axis.set_xlabel("measured jet multiplicity")
        axis.set_ylim(0, 1.03)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("fraction")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "assignment_by_jet_multiplicity.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for variant, axis in zip(VARIANTS, axes):
        entropy = posterior[variant]["diagnostics/assignment_entropy_nats"]
        supported = posterior[variant]["TRUTH/assignment_support"].astype(bool) & common_support
        rows = {"assignment_accuracy": [], "mtt_RMSE_GeV": [], "joint_energy_score_GeV": []}
        for index, (left, right) in enumerate(pairwise(entropy_edges)):
            mask = supported & (entropy >= left) & ((entropy < right) if index < len(entropy_edges)-2 else (entropy <= right))
            if mask.any():
                rows["assignment_accuracy"].append((0.5 * (left + right), float((posterior[variant]["diagnostics/assignment_MAP_state"][mask] == posterior[variant]["TRUTH/assignment_state"][mask]).mean())))
                residual = packs[variant]["diagnostic_arrays"]["points"]["mttbar"][mask] - packs[variant]["diagnostic_arrays"]["truth"]["mttbar"][mask]
                rows["mtt_RMSE_GeV"].append((0.5 * (left + right), float(np.sqrt(np.mean(residual**2)))))
                rows["joint_energy_score_GeV"].append((0.5 * (left + right), float(packs[variant]["diagnostic_arrays"]["joint"]["coherent_energy_score_by_event"][mask].mean())))
        if rows["assignment_accuracy"]:
            axis.plot(*np.asarray(rows["assignment_accuracy"]).T, "o-", color=colors[variant], label="MAP assignment accuracy")
        axis.set_xlabel("assignment entropy [nats]")
        axis.set_ylabel("supported MAP accuracy")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.set_title(f"KNOT-{variant}")
    fig.tight_layout()
    fig.savefig(output / "assignment_entropy_diagnostic.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for variant in VARIANTS:
        entropy = posterior[variant]["diagnostics/assignment_entropy_nats"]
        supported = posterior[variant]["TRUTH/assignment_support"].astype(bool) & common_support
        rows = {"mtt_RMSE_GeV": [], "joint_energy_score_GeV": []}
        for index, (left, right) in enumerate(pairwise(entropy_edges)):
            mask = supported & (entropy >= left) & ((entropy < right) if index < len(entropy_edges)-2 else (entropy <= right))
            if mask.any():
                center = 0.5 * (left + right)
                residual = packs[variant]["diagnostic_arrays"]["points"]["mttbar"][mask] - packs[variant]["diagnostic_arrays"]["truth"]["mttbar"][mask]
                rows["mtt_RMSE_GeV"].append((center, float(np.sqrt(np.mean(residual**2)))))
                rows["joint_energy_score_GeV"].append((center, float(packs[variant]["diagnostic_arrays"]["joint"]["coherent_energy_score_by_event"][mask].mean())))
        if rows["mtt_RMSE_GeV"]:
            axes[0].plot(*np.asarray(rows["mtt_RMSE_GeV"]).T, "o-", color=colors[variant], label=variant)
            axes[1].plot(*np.asarray(rows["joint_energy_score_GeV"]).T, "o-", color=colors[variant], label=variant)
    axes[0].set(xlabel="mean assignment entropy [nats]", ylabel="mtt RMSE [GeV]")
    axes[1].set(xlabel="mean assignment entropy [nats]", ylabel="joint energy score [GeV]")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / "entropy_reconstruction_and_joint.png", dpi=160)
    plt.close(fig)

    names = ("mttbar", "top_pT", "nu_top_pz", "top_mass", "pTttbar", "nu_antitop_pz")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, name in zip(axes.flat, names):
        for variant in VARIANTS:
            residual = packs[variant]["diagnostic_arrays"]["points"][name] - packs[variant]["diagnostic_arrays"]["truth"][name]
            axis.hist(residual, bins=45, density=True, histtype="step", lw=1.3, color=colors[variant], label=variant)
        axis.axvline(0, color="#222222", lw=0.8, ls="--")
        axis.set_xlabel(f"{name} residual")
        axis.set_ylabel("density")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output / "point_reconstruction_residuals.png", dpi=160)
    plt.close(fig)

    truth_mtt = packs["M"]["diagnostic_arrays"]["truth"]["mttbar"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant in VARIANTS:
        residual = packs[variant]["diagnostic_arrays"]["points"]["mttbar"] - truth_mtt
        means, errors, centers = [], [], []
        for index, (lo, hi) in enumerate(pairwise(mtt_edges)):
            mask = (truth_mtt >= lo) & (truth_mtt <= hi if index == len(mtt_edges) - 2 else truth_mtt < hi)
            means.append(float(residual[mask].mean()) if mask.any() else np.nan)
            errors.append(float(residual[mask].std(ddof=1) / np.sqrt(mask.sum())) if mask.sum() > 1 else 0)
            centers.append(float(0.5 * (lo + hi)))
        ax.errorbar(centers, means, yerr=errors, fmt="o-", color=colors[variant], label=variant)
    ax.axhline(0, color="#222222", ls="--", lw=0.8)
    ax.set(xlabel=r"truth $m_{t\bar t}$ [GeV]", ylabel="posterior-mean residual [GeV]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "mttbar_differential_bias.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for variant, axis in zip(VARIANTS, axes):
        pred = packs[variant]["diagnostic_arrays"]["draws"]["mttbar"].mean(axis=1)
        matrix, _, _ = np.histogram2d(truth_mtt, pred, bins=(mtt_edges, mtt_edges))
        matrix = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix), where=matrix.sum(axis=1, keepdims=True) > 0)
        image = axis.imshow(matrix, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
        axis.plot([0, len(mtt_edges)-2], [0, len(mtt_edges)-2], color="white", ls="--", lw=0.8)
        axis.set(title=f"KNOT-{variant}", xlabel="predicted mtt bin", ylabel="truth mtt bin")
    fig.colorbar(image, ax=axes, label="row-normalized fraction")
    fig.tight_layout()
    fig.savefig(output / "mttbar_migration.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    calibration_names = ("nu_top_pz", "top_pT", "mttbar")
    for axis, name in zip(axes, calibration_names):
        levels = np.array([0.5, 0.68, 0.9])
        for variant in VARIANTS:
            coverage = [packs[variant]["posterior_calibration"][name]["coverage"][str(float(level))] for level in levels]
            axis.plot(levels, coverage, "o-", color=colors[variant], label=variant)
        axis.plot([0, 1], [0, 1], color="#222222", ls="--", lw=0.8)
        axis.set(title=name, xlabel="nominal coverage", ylabel="empirical coverage", xlim=(0.4, 0.95), ylim=(0.35, 1.0))
        axis.grid(alpha=0.2)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "posterior_coverage.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for variant, axis in zip(VARIANTS, axes):
        joint = packs[variant]["diagnostic_arrays"]["joint"]
        coherent = joint["coherent_energy_score_by_event"]
        shuffled = joint["shuffled_energy_score_by_event"]
        axis.hist(coherent, bins=35, density=True, histtype="step", color=colors[variant], label="coherent")
        axis.hist(shuffled, bins=35, density=True, histtype="step", color="#555555", ls="--", label="parent samples shuffled")
        axis.set(title=f"KNOT-{variant}", xlabel="four-component energy score [GeV]", ylabel="density")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "joint_coherence_energy_score.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for variant, axis in zip(VARIANTS, axes):
        decay = packs[variant]["diagnostic_arrays"]["decay"]
        for label in ("coherent", "assignment_b_shuffled", "neutrino_parent_pair_shuffled"):
            axis.hist(decay["pt_RMSE_by_event"][label], bins=35, density=True, histtype="step", label=label.replace("_", " "))
        axis.set(title=f"KNOT-{variant}", xlabel=r"per-event decay-chain $\Delta p_T$ RMSE [GeV]", ylabel="density")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "decay_chain_consistency.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for axis, name in zip(axes.flat, SPIN_NAMES):
        truth = posterior["C"][f"spin/truth/{name}"]
        limits = (0, np.pi) if name == "delta_phi_ll" else (-1, 1)
        axis.hist(truth, bins=32, range=limits, density=True, histtype="step", color="#222222", label="hybrid truth")
        for variant in VARIANTS:
            axis.hist(posterior[variant][f"spin/paired/{name}"].ravel(), bins=32, range=limits, density=True, histtype="step", color=colors[variant], label=f"{variant} paired")
        for variant in ("C", "J"):
            axis.hist(posterior[variant][f"spin/shuffled/{name}"].ravel(), bins=32, range=limits, density=True, histtype="step", color=colors[variant], ls="--", label=f"{variant} shuffled")
        axis.set_xlabel(name)
        axis.set_ylabel("density")
    axes.flat[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "hybrid_spin_observables.png", dpi=160)
    plt.close(fig)

    truth_mtt = packs["M"]["diagnostic_arrays"]["truth"]["mttbar"]
    threshold = np.quantile(truth_mtt, 0.90)
    tail = truth_mtt >= threshold
    rows = []
    for variant in VARIANTS:
        arrays = packs[variant]["diagnostic_arrays"]
        for region, mask in (("bulk", ~tail), ("upper_truth_decile", tail)):
            residual = arrays["points"]["mttbar"][mask] - truth_mtt[mask]
            covered = calibration(arrays["draws"]["mttbar"][mask], truth_mtt[mask])["coverage"]["0.68"]
            rows.append({"variant": variant, "region": region, "events": int(mask.sum()), "bias_GeV": float(residual.mean()), "RMSE_GeV": float(np.sqrt(np.mean(residual**2))), "coverage68": covered})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for variant in VARIANTS:
        row = [x for x in rows if x["variant"] == variant]
        axes[0].plot([0, 1], [row[0]["RMSE_GeV"], row[1]["RMSE_GeV"]], "o-", color=colors[variant], label=variant)
        axes[1].plot([0, 1], [row[0]["coverage68"], row[1]["coverage68"]], "o-", color=colors[variant], label=variant)
    axes[0].set(xticks=[0, 1], xticklabels=["bulk", "upper 10%"], ylabel="mtt RMSE [GeV]")
    axes[1].set(xticks=[0, 1], xticklabels=["bulk", "upper 10%"], ylabel="mtt 68% coverage", ylim=(0, 1))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "high_mtt_tail.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, variant in zip(axes, VARIANTS):
        arrays = packs[variant]["diagnostic_arrays"]
        residual = arrays["points"]["mttbar"] - truth_mtt
        for channel in ("ee", "mumu", "emu"):
            mask = channels == channel
            if mask.any():
                axis.hist(residual[mask], bins=35, density=True, histtype="step", label=channel)
        axis.set(title=f"KNOT-{variant}", xlabel="mtt residual [GeV]", ylabel="density")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "channel_robustness.png", dpi=160)
    plt.close(fig)

    if "oracle/truth_assignment_truth_neutrino/parents" in posterior["C"]:
        oracle = posterior["C"]
        truth_support = oracle["TRUTH/assignment_support"].astype(bool)
        truth_parent = oracle["TRUTH/parents"][truth_support]
        labels = ("truth A, truth N", "pred A, truth N", "pred A, pred N")
        datasets = [
            oracle["oracle/truth_assignment_truth_neutrino/parents"],
            oracle["oracle/predicted_assignment_truth_neutrino/parents"],
            oracle["parents"][truth_support],
        ]
        rows = []
        for label, parents in zip(labels, datasets):
            rows.append({"endpoint": label, "top_pT_RMSE_GeV": float(np.sqrt(np.mean((pt(parents[:, :, 0]).mean(axis=1) - pt(truth_parent[:, 0]))**2))),
                         "antitop_pT_RMSE_GeV": float(np.sqrt(np.mean((pt(parents[:, :, 1]).mean(axis=1) - pt(truth_parent[:, 1]))**2))),
                         "top_mass_RMSE_GeV": float(np.sqrt(np.mean((four_vector_mass(parents[:, :, 0]).mean(axis=1) - four_vector_mass(truth_parent[:, 0]))**2))),
                         "antitop_mass_RMSE_GeV": float(np.sqrt(np.mean((four_vector_mass(parents[:, :, 1]).mean(axis=1) - four_vector_mass(truth_parent[:, 1]))**2)))})
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(labels))
        width = 0.18
        for index, metric_name in enumerate(("top_pT_RMSE_GeV", "antitop_pT_RMSE_GeV", "top_mass_RMSE_GeV", "antitop_mass_RMSE_GeV")):
            ax.bar(x + (index - 1.5) * width, [row[metric_name] for row in rows], width, label=metric_name)
        ax.set_xticks(x, labels)
        ax.set_ylabel("RMSE [GeV]")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "knot_c_oracle_ladder.png", dpi=160)
        plt.close(fig)
        (output / "oracle_summary.json").write_text(json.dumps({"events": int(truth_support.sum()), "endpoints": rows}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--measured", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42011)
    args = parser.parse_args()
    if args.bootstrap_replicates < 50:
        parser.error("Use at least 50 paired event-bootstrap replicates")
    freeze = json.loads(FREEZE.read_text())
    measured_path = args.measured or Path(freeze["data"]["measured_input"])
    output = args.output or args.evaluation / "physics-analysis-v1"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite KNOT physics analysis: {output}")
    output.mkdir(parents=True)

    posterior, training_metrics = {}, {}
    for variant in VARIANTS:
        posterior[variant], training_metrics[variant] = read_variant(args.evaluation, variant)
    event_ids = posterior["M"]["EVENT_ID"].astype(np.int64)
    for variant in VARIANTS[1:]:
        if not np.array_equal(event_ids, posterior[variant]["EVENT_ID"]):
            raise ValueError("M/C/J evaluation must use exactly the same ordered EVENT_ID values")
        if not np.array_equal(posterior["M"]["sample_index"], posterior[variant]["sample_index"]):
            raise ValueError("M/C/J posterior sample budgets or event-local indices differ")
    njet, channels = join_measured_categories(event_ids, measured_path)
    truth_n, truth_p = posterior["M"]["TRUTH/neutrinos"], posterior["M"]["TRUTH/parents"]
    for variant in VARIANTS[1:]:
        if not np.array_equal(truth_n, posterior[variant]["TRUTH/neutrinos"]) or not np.array_equal(truth_p, posterior[variant]["TRUTH/parents"]):
            raise ValueError("M/C/J event truth arrays do not match in the shared evaluation order")
    packs = {}
    for i, variant in enumerate(VARIANTS):
        pack = metric_pack(posterior[variant], training_metrics[variant], truth_n, truth_p, args.seed + i)
        pack["metrics"] = training_metrics[variant]
        packs[variant] = pack

    support = np.logical_and.reduce([posterior[v]["TRUTH/assignment_support"].astype(bool) for v in VARIANTS])
    mean_entropy = np.mean([posterior[v]["diagnostics/assignment_entropy_nats"] for v in VARIANTS], axis=0)
    entropy_edges = np.unique(np.quantile(mean_entropy[support], [0, 0.25, 0.5, 0.75, 1.0]))
    truth_mtt = four_vector_mass(truth_p.sum(axis=1))
    mtt_edges = np.unique(np.quantile(truth_mtt, np.linspace(0, 1, 9)))
    if len(mtt_edges) < 3:
        raise ValueError("Truth mttbar sample does not support migration bins")

    differential, channel_summary, entropy_summary = {}, {}, {}
    bootstrap_arrays = {}
    threshold = float(np.quantile(truth_mtt, 0.90))
    high_tail = truth_mtt >= threshold
    for variant in VARIANTS:
        arrays = packs[variant]["diagnostic_arrays"]
        support_v = arrays["support"]
        mtt_residual = arrays["points"]["mttbar"] - truth_mtt
        joint = arrays["joint"]
        decay = arrays["decay"]
        mtt_draws = arrays["draws"]["mttbar"]
        low68, high68 = np.quantile(mtt_draws, [0.16, 0.84], axis=1)
        high_sq = np.where(high_tail, mtt_residual**2, np.nan)
        accuracy = np.where(support_v, arrays["map_correct"], np.nan).astype(float)
        bootstrap_arrays[variant] = {
            "mtt_squared_error": mtt_residual**2,
            "mtt_residual": mtt_residual,
            "ttbar_pt_squared_error": (arrays["points"]["pTttbar"] - arrays["truth"]["pTttbar"])**2,
            "top_pt_squared_error": (arrays["points"]["top_pT"] - arrays["truth"]["top_pT"])**2,
            "nu_top_pz_squared_error": (arrays["points"]["nu_top_pz"] - arrays["truth"]["nu_top_pz"])**2,
            "joint_energy": joint["coherent_energy_score_by_event"],
            "joint_shuffle_penalty": joint["shuffle_minus_coherent_by_event"],
            "decay_pt_squared": decay["pt_RMSE_by_event"]["coherent"]**2,
            "mtt_covered_68": ((truth_mtt >= low68) & (truth_mtt <= high68)).astype(float),
            "assignment_correct": accuracy,
            "high_mtt_squared_error": high_sq,
        }
        differential[variant] = []
        for index, (lo, hi) in enumerate(pairwise(mtt_edges)):
            mask = (truth_mtt >= lo) & (truth_mtt <= hi if index == len(mtt_edges) - 2 else truth_mtt < hi)
            residual = mtt_residual[mask]
            differential[variant].append({"truth_mtt_low": float(lo), "truth_mtt_high": float(hi), "events": int(mask.sum()), "bias_GeV": float(residual.mean()) if len(residual) else None, "RMSE_GeV": float(np.sqrt(np.mean(residual**2))) if len(residual) else None})
        channel_summary[variant] = {}
        for channel in ("ee", "mumu", "emu"):
            mask = channels == channel
            residual = mtt_residual[mask]
            channel_summary[variant][channel] = {"events": int(mask.sum()), "mtt_bias_GeV": float(residual.mean()) if mask.any() else None, "mtt_RMSE_GeV": float(np.sqrt(np.mean(residual**2))) if mask.any() else None, "mtt_coverage68": calibration(mtt_draws[mask], truth_mtt[mask])["coverage"]["0.68"]}
        entropy_summary[variant] = []
        entropy = posterior[variant]["diagnostics/assignment_entropy_nats"]
        for i, (lo, hi) in enumerate(pairwise(entropy_edges)):
            mask = support & (entropy >= lo) & ((entropy < hi) if i < len(entropy_edges)-2 else (entropy <= hi))
            entropy_summary[variant].append({"entropy_low": float(lo), "entropy_high": float(hi), "events": int(mask.sum()), "MAP_accuracy": float(arrays["map_correct"][mask].mean()) if mask.any() else None, "assignment_NLL": float(arrays["assignment_nll"][mask].mean()) if mask.any() else None, "mtt_RMSE_GeV": float(np.sqrt(np.mean(mtt_residual[mask]**2))) if mask.any() else None, "joint_energy_score_GeV": float(joint["coherent_energy_score_by_event"][mask].mean()) if mask.any() else None, "decay_chain_delta_pt_RMSE_GeV": float(np.sqrt(np.mean(decay["pt_RMSE_by_event"]["coherent"][mask]**2))) if mask.any() else None})

    spin_summary = {}
    for variant in VARIANTS:
        spin_summary[variant] = {}
        for name in SPIN_NAMES:
            spin_summary[variant][name] = {
                "truth_hybrid_mean": float(posterior[variant][f"spin/truth/{name}"].mean()),
                "coherent_mean": float(posterior[variant][f"spin/paired/{name}"].mean()),
                "shuffled_mean": float(posterior[variant][f"spin/shuffled/{name}"].mean()),
                "coherent_vs_shuffled_RMS": float(np.sqrt(np.mean((posterior[variant][f"spin/paired/{name}"].mean(axis=1) - posterior[variant][f"spin/shuffled/{name}"].mean(axis=1))**2))),
                "coherent_vs_shuffled_event_SE": float(np.std(posterior[variant][f"spin/paired/{name}"].mean(axis=1) - posterior[variant][f"spin/shuffled/{name}"].mean(axis=1), ddof=1) / np.sqrt(len(event_ids))),
            }
    bootstrap = bootstrap_comparison(bootstrap_arrays, args.bootstrap_replicates, args.seed + 550)
    shuffle_intervals = bootstrap["joint_shuffle_penalty_GeV"]["paired_bootstrap_95pct_interval"]
    mtt_differences = bootstrap["mtt_RMSE_GeV"]["paired_differences_vs_M_95pct"]
    pt_differences = bootstrap["pTttbar_RMSE_GeV"]["paired_differences_vs_M_95pct"]
    c_shuffle = shuffle_intervals["C"]
    j_shuffle = shuffle_intervals["J"]
    m_shuffle = shuffle_intervals["M"]
    coherent_shuffle_effect = c_shuffle[0] > 0 and j_shuffle[0] > 0
    marginal_metrics_comparable = all(
        interval[0] <= 0 <= interval[1]
        for interval in (mtt_differences["C"], mtt_differences["J"], pt_differences["C"], pt_differences["J"])
    )
    if coherent_shuffle_effect:
        mechanism = "KNOT_MECHANISM_CONFIRMED"
    elif c_shuffle[1] < 0 and j_shuffle[1] < 0:
        mechanism = "KNOT_MECHANISM_NOT_SUPPORTED"
    else:
        mechanism = "KNOT_MECHANISM_NOT_RESOLVED"
    if coherent_shuffle_effect and marginal_metrics_comparable:
        pilot_outcome = "K1"
    elif coherent_shuffle_effect and bootstrap["joint_energy_score_GeV"]["paired_differences_vs_C_95pct"]["J"][1] < 0:
        pilot_outcome = "K2"
    elif all(interval[0] <= 0 <= interval[1] for interval in (mtt_differences["C"], mtt_differences["J"])) and not coherent_shuffle_effect:
        pilot_outcome = "K3"
    elif mtt_differences["C"][0] > 0 and mtt_differences["J"][0] > 0:
        pilot_outcome = "K4"
    else:
        pilot_outcome = "K3"
    spin_pairing_sensitive = {
        variant: {
            name: abs(spin_summary[variant][name]["coherent_mean"] - spin_summary[variant][name]["shuffled_mean"])
            > 2 * spin_summary[variant][name]["coherent_vs_shuffled_event_SE"]
            for name in SPIN_NAMES
        }
        for variant in VARIANTS
    }
    spin_pairing_sensitive["labframe_delta_phi_ll_control_insensitive"] = all(
        not spin_pairing_sensitive[variant]["delta_phi_ll"] for variant in VARIANTS
    )
    parameter_counts = freeze["training"]["parameter_counts"]
    c_total = parameter_counts["C"]["total"]
    j_total = parameter_counts["J"]["total"]
    c_vs_j = {
        "J_minus_C_joint_energy_score_95pct_GeV": bootstrap["joint_energy_score_GeV"]["paired_differences_vs_C_95pct"]["J"],
        "J_minus_C_mtt_RMSE_95pct_GeV": bootstrap["mtt_RMSE_GeV"]["paired_differences_vs_C_95pct"]["J"],
        "J_parameter_count_fraction_above_C": float((j_total - c_total) / c_total),
        "interpretation": "A measured J-C difference is a finite-capacity, optimization, inductive-bias or sequential-error-propagation effect; it is not evidence that J is more expressive in principle.",
    }
    summary = {
        "events": len(event_ids),
        "samples_per_event": int(posterior["M"]["sample_index"].shape[1]),
        "event_id_sha256_ordered_little_endian_int64": sha256_ids(event_ids),
        "sealed_test_used": False,
        "spin_truth_benchmark": "hybrid truth closure: after-FSR truth parents with measured charge-ordered leptons",
        "event_categories": {"njet_counts": {str(x): int((njet == x).sum()) for x in np.unique(njet)}, "channels": {ch: int((channels == ch).sum()) for ch in ("ee", "mumu", "emu")}},
        "common_supported_events": int(support.sum()),
        "common_entropy_edges_nats": entropy_edges.tolist(),
        "truth_mttbar_quantile_edges_GeV": mtt_edges.tolist(),
        "high_mtt_truth_decile_threshold_GeV": threshold,
        "variants": {variant: {key: value for key, value in packs[variant].items() if key != "diagnostic_arrays" and key != "metrics"} | {"training_evaluation_metrics": training_metrics[variant]} for variant in VARIANTS},
        "mttbar_differential": differential,
        "entropy_diagnostics": entropy_summary,
        "channel_robustness": channel_summary,
        "spin_hybrid": spin_summary,
        "spin_pairing_sensitivity": spin_pairing_sensitive,
        "paired_event_bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.seed + 550, "uncertainty_scope": "validation event sampling only; one fixed training seed", "metrics": bootstrap},
        "KNOT_MECHANISM": mechanism,
        "KNOT_pilot_outcome": pilot_outcome,
        "KNOT_C_vs_J": c_vs_j,
        "marginal_metrics_comparable_CJ_vs_M": marginal_metrics_comparable,
        "marginal_shuffle_effect_M_95pct_GeV": m_shuffle,
        "followup_training_decision": "Keep the matched M/C/J checkpoints and use the evaluator outputs for interpretation; make no architecture change from one training seed. If a future meaningful J-C difference appears, treat capacity/optimization as controls before an expressivity claim.",
        "production_decision": "KNOT_FULL_GPU_CAMPAIGN_WITH_EVALUATOR_UPDATE_ONLY" if pilot_outcome != "K4" else "KNOT_FULL_GPU_CAMPAIGN_PAUSE_FOR_SPECIFIC_ISSUE",
    }
    if "oracle/truth_assignment_truth_neutrino/parents" in posterior["C"]:
        c = posterior["C"]
        good = c["TRUTH/assignment_support"].astype(bool)
        truth_p_supported = c["TRUTH/parents"][good]
        truth_n_supported = c["TRUTH/neutrinos"][good]
        oracle_names = ("P|A_truth,N_truth", "P|A_pred,N_truth", "P|A_pred,N_pred")
        oracle_parent_values = (c["oracle/truth_assignment_truth_neutrino/parents"], c["oracle/predicted_assignment_truth_neutrino/parents"], c["parents"][good])
        c_oracle = []
        for name, parents in zip(oracle_names, oracle_parent_values):
            c_oracle.append({"endpoint": name, "top_pT_RMSE_GeV": float(np.sqrt(np.mean((pt(parents[:, :, 0]).mean(axis=1) - pt(truth_p_supported[:, 0]))**2))), "antitop_pT_RMSE_GeV": float(np.sqrt(np.mean((pt(parents[:, :, 1]).mean(axis=1) - pt(truth_p_supported[:, 1]))**2))), "top_mass_RMSE_GeV": float(np.sqrt(np.mean((four_vector_mass(parents[:, :, 0]).mean(axis=1) - four_vector_mass(truth_p_supported[:, 0]))**2))), "antitop_mass_RMSE_GeV": float(np.sqrt(np.mean((four_vector_mass(parents[:, :, 1]).mean(axis=1) - four_vector_mass(truth_p_supported[:, 1]))**2)))})
        for key, label in (("oracle/neutrino_truth_assignment/invisible", "N|A_truth"), ("oracle/neutrino_predicted_assignment/invisible", "N|A_pred")):
            if key not in c:
                continue
            n_sample = c[key]
            mean = n_sample.mean(axis=1)
            c_oracle.append({"endpoint": label, "top_nu_pT_RMSE_GeV": float(np.sqrt(np.mean((pt(mean[:, 0]) - pt(truth_n_supported[:, 0]))**2))), "antitop_nu_pT_RMSE_GeV": float(np.sqrt(np.mean((pt(mean[:, 1]) - pt(truth_n_supported[:, 1]))**2))), "top_nu_pz_RMSE_GeV": float(np.sqrt(np.mean((mean[:, 0, 3] - truth_n_supported[:, 0, 3])**2))), "antitop_nu_pz_RMSE_GeV": float(np.sqrt(np.mean((mean[:, 1, 3] - truth_n_supported[:, 1, 3])**2)))})
        summary["KNOT_C_oracle_ladder"] = {"events": int(good.sum()), "endpoints": c_oracle}

    provenance = {
        "freeze_record": str(FREEZE.relative_to(ROOT)),
        "source_commit": freeze["source_commit"],
        "closure_branch_commit": freeze["closure_branch_commit"],
        "evaluation_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "measured_input": str(measured_path),
        "evaluation_root": str(args.evaluation.resolve()),
    }
    summary["provenance"] = provenance
    (output / "physics_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    save_core_plots(output, posterior, packs, njet, channels, support, entropy_edges, mtt_edges)
    report_lines = ["# KNOT matched-production posterior evaluation", "", f"Events: {len(event_ids):,}; samples/event: {posterior['M']['sample_index'].shape[1]}; reserved test used: false.", "", "## Reconstruction summary", "", "| Variant | mtt bias [GeV] | mtt RMSE [GeV] | joint energy score [GeV] | N-P shuffle penalty [GeV] |", "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        mtt = packs[variant]["reconstruction"]["mttbar"]
        joint = packs[variant]["joint"]
        report_lines.append(f"| {variant} | {mtt['bias']:.3f} | {mtt['RMSE']:.3f} | {joint['coherent_energy_score_GeV']:.3f} | {joint['shuffle_minus_coherent_GeV']:.3f} |")
    report_lines += [
        "",
        "Derived-observable bias and RMSE use each event's posterior mean of the observable evaluated sample by sample. The joint energy score uses top and antitop neutrino/parent pz together; shuffling changes only the within-event neutrino-parent pairing.",
        "",
        "## Interpretation limits",
        "",
        "M sample indices are marginal and are not interpreted as coherent event hypotheses. C and J are evaluated with their saved event-local ancestry/joint samples. Spin truth is a hybrid closure sample because generator-level lepton four-vectors are unavailable. Bootstrap intervals resample validation events only and do not include training-seed uncertainty.",
        "",
        f"Pilot outcome: {summary['KNOT_pilot_outcome']}. A J-C joint-score difference does not imply that J is more expressive; parameter count, optimization and sequential error propagation remain possible explanations.",
        f"Mechanism classification: {summary['KNOT_MECHANISM']} (one matched production seed; differences require paired uncertainty and semantic interpretation).",
        f"Production recommendation: {summary['production_decision']}.",
        "",
    ]
    (output / "physics_summary.md").write_text("\n".join(report_lines))
    print(json.dumps({"output": str(output), "events": len(event_ids), "samples": posterior["M"]["sample_index"].shape[1], "event_id_sha256": sha256_ids(event_ids)}, indent=2))


if __name__ == "__main__":
    main()
