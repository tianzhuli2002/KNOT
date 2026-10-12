#!/usr/bin/env python3
"""Separate KNOT posterior-mean mtt shrinkage from predictive tail closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy.stats import wasserstein_distance

matplotlib.use("Agg")
from matplotlib import pyplot as plt

VARIANTS = ("M", "C", "J")
COLORS = {"M": "#0072B2", "C": "#D55E00", "J": "#009E73"}
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 42011
TAIL_QUANTILES = (0.90, 0.99)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def mass(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mass_squared = vector[..., 0] ** 2 - np.sum(vector[..., 1:] ** 2, axis=-1)
    return np.sqrt(np.maximum(mass_squared, 0.0)), mass_squared


def load_posterior(path: Path, variant: str) -> dict:
    with h5py.File(path, "r") as source:
        if source.attrs.get("sealed_test_used") not in (False, 0):
            raise ValueError(f"{path} is not marked sealed_test_used=false")
        required = ("EVENT_ID", "TRUTH/neutrinos", "TRUTH/parents", "invisible", "parents")
        missing = [name for name in required if name not in source]
        if missing:
            raise ValueError(f"{path} lacks required KNOT fields: {missing}")
        values = {name: source[name][:] for name in required}
        attrs = dict(source.attrs)
    ids = values["EVENT_ID"]
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path} EVENT_ID must be one-dimensional and unique")
    n_events = len(ids)
    truth_nu, truth_parent = values["TRUTH/neutrinos"], values["TRUTH/parents"]
    nu, parent = values["invisible"], values["parents"]
    if truth_nu.shape != (n_events, 2, 4) or truth_parent.shape != (n_events, 2, 4):
        raise ValueError(f"{path} truth vector shapes do not match the top/antitop contract")
    if nu.ndim != 4 or nu.shape[0] != n_events or nu.shape[2:] != (2, 4):
        raise ValueError(f"{path} invisible samples must have shape (events, K, 2, 4)")
    if parent.shape != nu.shape:
        raise ValueError(f"{path} parent and invisible posterior shapes differ")
    for name, array in values.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{path} contains non-finite values in {name}")
    return {"variant": variant, **values, "attrs": attrs}


def bootstrap_tail(
    truth_mtt: np.ndarray,
    predictive_event_probability: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
) -> dict:
    truth_tail = truth_mtt > threshold
    predicted_fraction = float(predictive_event_probability.mean())
    truth_fraction = float(truth_tail.mean())
    if truth_fraction <= 0:
        raise ValueError(f"No truth events exceed the fixed tail threshold {threshold}")
    ratios = np.empty(BOOTSTRAPS, dtype=np.float64)
    differences = np.empty(BOOTSTRAPS, dtype=np.float64)
    n_events = len(truth_mtt)
    for index in range(BOOTSTRAPS):
        event_index = rng.integers(0, n_events, size=n_events)
        truth_boot = float(truth_tail[event_index].mean())
        pred_boot = float(predictive_event_probability[event_index].mean())
        ratios[index] = pred_boot / truth_boot if truth_boot > 0 else np.nan
        differences[index] = pred_boot - truth_boot
    finite_ratios = ratios[np.isfinite(ratios)]
    return {
        "truth_fraction": truth_fraction,
        "predictive_fraction": predicted_fraction,
        "predictive_to_truth_ratio": predicted_fraction / truth_fraction,
        "paired_event_bootstrap_95pct_ratio": np.quantile(finite_ratios, [0.025, 0.975]).tolist(),
        "paired_event_bootstrap_95pct_fraction_difference": np.quantile(
            differences, [0.025, 0.975]
        ).tolist(),
    }


def analyze(posterior: dict, truth_mtt: np.ndarray, thresholds: dict, rng) -> dict:
    parents = posterior["parents"]
    neutrinos = posterior["invisible"]
    truth_parent = posterior["TRUTH/parents"]
    truth_neutrino = posterior["TRUTH/neutrinos"]
    pair_draws = parents.sum(axis=2)
    truth_pair = truth_parent.sum(axis=1)
    predictive_mtt, predictive_mtt_squared = mass(pair_draws)
    _truth_pair_mtt, truth_pair_mtt_squared = mass(truth_pair)
    point_mtt = predictive_mtt.mean(axis=1)
    top_decile = truth_mtt >= thresholds[0.90]
    if not top_decile.any():
        raise ValueError("The truth-defined top-decile population is empty")

    predicted_nu_xyz = neutrinos[..., 1:].mean(axis=1)
    truth_nu_xyz = truth_neutrino[..., 1:]
    nu_error_squared = np.sum((predicted_nu_xyz - truth_nu_xyz) ** 2, axis=-1)
    nu_rmse = np.sqrt(nu_error_squared.mean(axis=0))
    nu_combined_rmse = float(np.sqrt(nu_error_squared.mean()))
    nu_pt_rmse = np.sqrt(
        ((np.hypot(predicted_nu_xyz[..., 0], predicted_nu_xyz[..., 1])
          - np.hypot(truth_nu_xyz[..., 0], truth_nu_xyz[..., 1])) ** 2).mean(axis=0)
    )
    nu_pz_rmse = np.sqrt(((predicted_nu_xyz[..., 2] - truth_nu_xyz[..., 2]) ** 2).mean(axis=0))

    quantiles = (0.50, 0.68, 0.90, 0.95, 0.99)
    predicted_flat = predictive_mtt.reshape(-1)
    point_error = point_mtt - truth_mtt
    lower68, upper68 = np.quantile(predictive_mtt, [0.16, 0.84], axis=1)
    lower90, upper90 = np.quantile(predictive_mtt, [0.05, 0.95], axis=1)
    tail = {}
    for quantile, threshold in thresholds.items():
        probability_by_event = np.mean(predictive_mtt > threshold, axis=1)
        tail[f"truth_q{int(quantile * 100)}"] = {
            "truth_threshold_GeV": threshold,
            **bootstrap_tail(truth_mtt, probability_by_event, threshold, rng),
        }

    response_bins = []
    order = np.argsort(truth_mtt, kind="stable")
    for bin_index, rows in enumerate(np.array_split(order, 10)):
        response_bins.append(
            {
                "bin": bin_index,
                "events": len(rows),
                "truth_mtt_mean_GeV": float(truth_mtt[rows].mean()),
                "posterior_mean_mtt_bias_GeV": float(point_error[rows].mean()),
                "posterior_mean_mtt_rmse_GeV": float(np.sqrt(np.mean(point_error[rows] ** 2))),
            }
        )

    return {
        "events": len(truth_mtt),
        "samples_per_event": int(predictive_mtt.shape[1]),
        "truth_mtt_quantiles_GeV": {
            str(q): float(np.quantile(truth_mtt, q)) for q in quantiles
        },
        "posterior_predictive_mtt_quantiles_GeV": {
            str(q): float(np.quantile(predicted_flat, q)) for q in quantiles
        },
        "posterior_predictive_wasserstein1_GeV": float(
            wasserstein_distance(truth_mtt, predicted_flat)
        ),
        "posterior_predictive_tail_closure": tail,
        "posterior_mean_mtt": {
            "bias_GeV": float(point_error.mean()),
            "rmse_GeV": float(np.sqrt(np.mean(point_error ** 2))),
            "truth_to_estimate_response_slope": float(np.polyfit(truth_mtt, point_mtt, 1)[0]),
            "truth_top_decile_bias_GeV": float(point_error[top_decile].mean()),
            "truth_top_decile_rmse_GeV": float(np.sqrt(np.mean(point_error[top_decile] ** 2))),
            "bias_by_truth_decile": response_bins,
        },
        "mtt_central_interval_coverage": {
            "68pct": float(np.mean((truth_mtt >= lower68) & (truth_mtt <= upper68))),
            "90pct": float(np.mean((truth_mtt >= lower90) & (truth_mtt <= upper90))),
            "truth_top_decile_68pct": float(
                np.mean((truth_mtt[top_decile] >= lower68[top_decile])
                        & (truth_mtt[top_decile] <= upper68[top_decile]))
            ),
        },
        "neutrino_point_mean_rmse_GeV": {
            "full_3vector": {
                "top": float(nu_rmse[0]),
                "antitop": float(nu_rmse[1]),
                "combined_per_neutrino": nu_combined_rmse,
            },
            "pt": {"top": float(nu_pt_rmse[0]), "antitop": float(nu_pt_rmse[1])},
            "pz": {"top": float(nu_pz_rmse[0]), "antitop": float(nu_pz_rmse[1])},
        },
        "spacelike_mtt_count_clamped_by_existing_mass_convention": {
            "truth": int(np.count_nonzero(truth_pair_mtt_squared < 0)),
            "posterior_draws": int(np.count_nonzero(predictive_mtt_squared < 0)),
        },
    }


def make_figure(truth_mtt: np.ndarray, posteriors: dict, metrics: dict, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 10.5), constrained_layout=True)
    bins = np.linspace(300.0, 3000.0, 55)
    axes[0].hist(truth_mtt, bins=bins, density=True, histtype="stepfilled", color="#777777", alpha=0.25, label="After-FSR truth")
    axes[0].hist(truth_mtt, bins=bins, density=True, histtype="step", color="#555555", linewidth=1.6)
    for variant in VARIANTS:
        pair = posteriors[variant]["parents"].sum(axis=2)
        pred_mtt, _ = mass(pair)
        axes[0].hist(
            pred_mtt.reshape(-1), bins=bins, density=True, histtype="step",
            color=COLORS[variant], linewidth=1.6, label=f"KNOT-{variant} posterior predictive",
        )
    axes[0].set_yscale("log")
    axes[0].set_xlim(300, 3000)
    axes[0].set_ylabel("Normalized density")
    axes[0].set_title("Point-estimate shrinkage and posterior-predictive tail closure")
    axes[0].legend(frameon=False, ncol=2)

    for variant in VARIANTS:
        rows = metrics[variant]["posterior_mean_mtt"]["bias_by_truth_decile"]
        axes[1].plot(
            [row["truth_mtt_mean_GeV"] for row in rows],
            [row["posterior_mean_mtt_bias_GeV"] for row in rows],
            marker="o", markersize=3.5, color=COLORS[variant], label=f"KNOT-{variant}",
        )
    axes[1].axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("Posterior-mean mtt bias [GeV]")
    axes[1].set_xlabel("Truth mtt [GeV], truth-defined decile mean")
    axes[1].legend(frameon=False, ncol=3)

    positions = np.arange(len(TAIL_QUANTILES))
    for variant in VARIANTS:
        values, lower, upper = [], [], []
        for quantile in TAIL_QUANTILES:
            result = metrics[variant]["posterior_predictive_tail_closure"][f"truth_q{int(quantile * 100)}"]
            ratio = result["predictive_to_truth_ratio"]
            interval = result["paired_event_bootstrap_95pct_ratio"]
            values.append(ratio)
            lower.append(ratio - interval[0])
            upper.append(interval[1] - ratio)
        axes[2].errorbar(
            positions, values, yerr=np.asarray([lower, upper]), marker="o", capsize=3,
            color=COLORS[variant], label=f"KNOT-{variant}",
        )
    axes[2].axhline(1.0, color="#555555", linewidth=1.0, linestyle="--")
    axes[2].set_xticks(positions, ["Truth top 10%", "Truth top 1%"])
    axes[2].set_ylabel("Predictive / truth tail fraction")
    axes[2].set_ylim(0.0, 2.0)
    axes[2].legend(frameon=False, ncol=3)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluation_root = args.evaluation_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing tail analysis output: {output}")
    loaded = {
        variant: load_posterior(evaluation_root / variant / "posterior.h5", variant)
        for variant in VARIANTS
    }
    ids = loaded["M"]["EVENT_ID"]
    for variant in VARIANTS[1:]:
        if not np.array_equal(ids, loaded[variant]["EVENT_ID"]):
            raise ValueError(f"KNOT-{variant} EVENT_ID order differs from KNOT-M")
    truth_parent = loaded["J"]["TRUTH/parents"]
    truth_pair = truth_parent.sum(axis=1)
    truth_mtt, _ = mass(truth_pair)
    if not np.isfinite(truth_mtt).all():
        raise ValueError("After-FSR truth mtt contains non-finite values")
    thresholds = {q: float(np.quantile(truth_mtt, q)) for q in TAIL_QUANTILES}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metrics = {
        variant: analyze(loaded[variant], truth_mtt, thresholds, rng)
        for variant in VARIANTS
    }
    output.mkdir(parents=True, exist_ok=False)
    plot_path = output / "high_mtt_posterior_predictive_closure.png"
    make_figure(truth_mtt, loaded, metrics, plot_path)
    input_records = {}
    for variant in VARIANTS:
        path = evaluation_root / variant / "posterior.h5"
        input_records[variant] = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "status": "completed",
        "diagnostic": "stored 10k KNOT posterior tail closure; no inference recomputed",
        "evaluation_root": str(evaluation_root),
        "inputs": input_records,
        "events": len(ids),
        "samples_per_event": int(loaded["J"]["invisible"].shape[1]),
        "ordered_EVENT_ID_sha256": sha256_ids(ids),
        "sealed_test_used": False,
        "truth_contract": "top/antitop partonic after-FSR four-vectors; neutrino slots top then antitop",
        "tail_thresholds_from_truth_GeV": {str(q): value for q, value in thresholds.items()},
        "bootstrap": {"unit": "event", "replicates": BOOTSTRAPS, "seed": BOOTSTRAP_SEED},
        "posterior_draw_weighting": "each event total weight 1; each of its K stored draws weight 1/K",
        "mass_convention": "sqrt(max(E^2-|p|^2,0)); spacelike counts are recorded in metrics",
        "outputs": {"metrics": "metrics.json", "summary": "summary.md", "figure": plot_path.name},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    j = metrics["J"]
    lines = [
        "# KNOT posterior mean and high-mtt tail closure",
        "",
        f"Stored posterior input: `{evaluation_root}` ({len(ids):,} common events; K={loaded['J']['invisible'].shape[1]}). No new inference was run.",
        "",
        "The truth-defined top-decile posterior-mean bias is shown separately from the full posterior-predictive tail fraction. Each event contributes total predictive weight one.",
        "",
        "| Variant | mtt mean bias [GeV] | Response slope | Top-decile point bias [GeV] | Predictive/truth top-10% | Predictive/truth top-1% | W1 [GeV] | 68% coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        value = metrics[variant]
        tail90 = value["posterior_predictive_tail_closure"]["truth_q90"]["predictive_to_truth_ratio"]
        tail99 = value["posterior_predictive_tail_closure"]["truth_q99"]["predictive_to_truth_ratio"]
        lines.append(
            f"| KNOT-{variant} | {value['posterior_mean_mtt']['bias_GeV']:.2f} | {value['posterior_mean_mtt']['truth_to_estimate_response_slope']:.3f} | {value['posterior_mean_mtt']['truth_top_decile_bias_GeV']:.2f} | {tail90:.3f} | {tail99:.3f} | {value['posterior_predictive_wasserstein1_GeV']:.2f} | {value['mtt_central_interval_coverage']['68pct']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"KNOT-J posterior-mean mtt has a truth-top-decile bias of {j['posterior_mean_mtt']['truth_top_decile_bias_GeV']:.1f} GeV. Its posterior-predictive tail fractions are {j['posterior_predictive_tail_closure']['truth_q90']['predictive_to_truth_ratio']:.3f} and {j['posterior_predictive_tail_closure']['truth_q99']['predictive_to_truth_ratio']:.3f} of truth at the truth q90 and q99 thresholds, respectively. The event-bootstrap intervals are in `metrics.json`.",
            "",
            "Interpretation: the stored K=32 KNOT-J posterior retains high-mtt population support on this validation sample; the posterior-mean shrinkage is not evidence of a missing generated tail. This result does not resolve the separate mtt coverage deficit.",
            "",
            f"KNOT-J old-style neutrino full 3-vector RMSE: top {j['neutrino_point_mean_rmse_GeV']['full_3vector']['top']:.2f} GeV, antitop {j['neutrino_point_mean_rmse_GeV']['full_3vector']['antitop']:.2f} GeV; combined per-neutrino {j['neutrino_point_mean_rmse_GeV']['full_3vector']['combined_per_neutrino']:.2f} GeV.",
            "",
            f"Figure: `{plot_path.name}`.",
        ]
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output), "events": len(ids), "KNOT_J": j}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
