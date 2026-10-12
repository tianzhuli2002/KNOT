"""Create compact figures for validated KNOT diagnostics and KK comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: I001


ROOT = Path(__file__).resolve().parents[2]
SETTING_ORDER = ("K32-Heun8", "K128-Heun8", "K128-Heun16", "K128-Heun32")
SETTING_LABELS = ("K32 / H8", "K128 / H8", "K128 / H16", "K128 / H32")
PROBE_ORDER = ("h", "h_A_truth", "h_A_map", "h_A_map_N_truth", "h_A_map_N_pred")
PROBE_LABELS = (
    r"$h$",
    r"$h + A_{truth}$",
    r"$h + A_{MAP}$",
    r"$h + A_{MAP} + N_{truth}$",
    r"$h + A_{MAP} + N_{pred}$",
)


def _read_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _finite_values(values: list[float], label: str) -> None:
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ValueError(f"Non-finite values in {label}: {values}")


def _interval_errors(values: list[float], intervals: list[list[float]]) -> np.ndarray:
    lower = [value - interval[0] for value, interval in zip(values, intervals, strict=True)]
    upper = [interval[1] - value for value, interval in zip(values, intervals, strict=True)]
    errors = np.asarray([lower, upper])
    if np.any(errors < -1e-8):
        raise ValueError("A point estimate falls outside its event-bootstrap interval")
    return np.maximum(errors, 0.0)


def _style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out")
    axis.grid(axis="y", color="#E6E8EB", linewidth=0.7)
    axis.set_axisbelow(True)


def _save_figure(fig, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return base.with_suffix(".png"), base.with_suffix(".pdf")


def plot_posterior_convergence(source: Path, output_base: Path) -> tuple[Path, Path]:
    data = _read_json(source)
    if data.get("sealed_test_used") is not False:
        raise ValueError("Posterior convergence figure requires sealed_test_used=false")
    metrics = data["metrics"]
    if tuple(metrics) != SETTING_ORDER:
        raise ValueError(f"Unexpected posterior settings/order: {tuple(metrics)}")

    coverage = [metrics[name]["mtt_central_interval_calibration"]["0.68"] for name in SETTING_ORDER]
    mtt_rmse = [metrics[name]["posterior_mean_mtt"]["RMSE_GeV"] for name in SETTING_ORDER]
    mtt_rmse_ci = [
        metrics[name]["posterior_mean_mtt"]["event_bootstrap_95pct_interval"]["RMSE"]
        for name in SETTING_ORDER
    ]
    top_pz = [
        metrics[name]["stored_14d_marginal_calibration"]["neutrino_top_pz"]
        ["nominal_vs_empirical_coverage"]["0.68"]
        for name in SETTING_ORDER
    ]
    antitop_pz = [
        metrics[name]["stored_14d_marginal_calibration"]["neutrino_antitop_pz"]
        ["nominal_vs_empirical_coverage"]["0.68"]
        for name in SETTING_ORDER
    ]
    nu3_rmse = [metrics[name]["neutrino"]["combined_3vector_RMSE_GeV"] for name in SETTING_ORDER]
    energy = [
        metrics[name]["joint_neutrino_parent_pz"]["energy_score_coherent_GeV"]
        for name in SETTING_ORDER
    ]
    shuffle_penalty = [
        metrics[name]["joint_neutrino_parent_pz"]["energy_score_shuffle_minus_coherent_GeV"]
        for name in SETTING_ORDER
    ]
    shuffle_ci = [
        metrics[name]["joint_neutrino_parent_pz"]
        ["energy_score_shuffle_penalty_event_bootstrap_95pct_interval_GeV"]
        for name in SETTING_ORDER
    ]
    decay_rmse = [
        metrics[name]["decay_chain_consistency"]["coherent_delta_pT_RMSE_GeV"]
        for name in SETTING_ORDER
    ]
    decay_ci = [
        metrics[name]["decay_chain_consistency"]
        ["coherent_delta_pT_RMSE_event_bootstrap_95pct_interval_GeV"]
        for name in SETTING_ORDER
    ]
    response_slope = [
        metrics[name]["posterior_mean_mtt"]["truth_to_posterior_mean_response_slope"]
        for name in SETTING_ORDER
    ]
    response_ci = [
        metrics[name]["posterior_mean_mtt"]["event_bootstrap_95pct_interval_for_response_slope"]
        for name in SETTING_ORDER
    ]
    runtime = [metrics[name]["runtime_seconds"] for name in SETTING_ORDER]

    fig, axes = plt.subplots(3, 3, figsize=(13.2, 9.2), sharex=True)
    x = np.arange(len(SETTING_ORDER))
    blue, orange, green = "#0072B2", "#D55E00", "#009E73"

    axes[0, 0].errorbar(
        x,
        [row["empirical_coverage"] for row in coverage],
        yerr=np.asarray(
            [
                [row["empirical_coverage"] - row["event_bootstrap_95pct_interval_for_coverage"][0]
                 for row in coverage],
                [row["event_bootstrap_95pct_interval_for_coverage"][1] - row["empirical_coverage"]
                 for row in coverage],
            ]
        ),
        color=blue,
        marker="o",
        capsize=3,
    )
    axes[0, 0].axhline(0.68, color="#555555", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("Empirical coverage")
    axes[0, 0].set_title(r"$m_{t\bar{t}}$ 68% interval")

    axes[0, 1].plot(x, top_pz, marker="o", color=blue, label="top")
    axes[0, 1].plot(x, antitop_pz, marker="s", color=orange, label="antitop")
    axes[0, 1].axhline(0.68, color="#555555", linestyle="--", linewidth=1)
    axes[0, 1].set_ylabel("Empirical coverage")
    axes[0, 1].set_title(r"Neutrino $p_z$ 68% intervals")
    axes[0, 1].legend(frameon=False, fontsize=8)

    axes[0, 2].errorbar(
        x,
        mtt_rmse,
        yerr=_interval_errors(mtt_rmse, mtt_rmse_ci),
        color=green,
        marker="o",
        capsize=3,
    )
    axes[0, 2].set_ylabel("RMSE [GeV]")
    axes[0, 2].set_title(r"Posterior-mean $m_{t\bar{t}}$")

    axes[1, 0].plot(x, nu3_rmse, marker="o", color=blue)
    axes[1, 0].set_ylabel("RMSE [GeV]")
    axes[1, 0].set_title(r"Combined neutrino 3-vector")

    axes[1, 1].plot(x, energy, marker="o", color=green)
    axes[1, 1].set_ylabel("Energy score [GeV]")
    axes[1, 1].set_title("Coherent joint score")

    axes[1, 2].errorbar(
        x,
        shuffle_penalty,
        yerr=_interval_errors(shuffle_penalty, shuffle_ci),
        color=orange,
        marker="o",
        capsize=3,
    )
    axes[1, 2].set_ylabel("Penalty [GeV]")
    axes[1, 2].set_title("N/P shuffle penalty")

    axes[2, 0].errorbar(
        x,
        decay_rmse,
        yerr=_interval_errors(decay_rmse, decay_ci),
        color=blue,
        marker="o",
        capsize=3,
    )
    axes[2, 0].set_ylabel(r"$\Delta p_T$ RMSE [GeV]")
    axes[2, 0].set_title("Decay-chain consistency")

    axes[2, 1].errorbar(
        x,
        response_slope,
        yerr=_interval_errors(response_slope, response_ci),
        color=green,
        marker="o",
        capsize=3,
    )
    axes[2, 1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[2, 1].set_ylabel("Response slope")
    axes[2, 1].set_title("Truth-to-posterior-mean response")

    axes[2, 2].plot(x, runtime, marker="o", color=orange)
    axes[2, 2].set_ylabel("Runtime [s]")
    axes[2, 2].set_title("Inference on 2,000 events")

    for axis in axes.flat:
        _style_axis(axis)
    for axis in axes[2, :]:
        axis.set_xticks(x, SETTING_LABELS, rotation=25, ha="right")
        axis.set_xlabel("Posterior draws / Heun steps")
    fig.suptitle("KNOT-J posterior numerical convergence", fontsize=15)
    fig.text(
        0.01,
        0.01,
        "2,000 paired validation events; error bars show 95% event-bootstrap intervals where stored. "
        "Dashed lines mark nominal coverage or unit response.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    return _save_figure(fig, output_base)


def plot_frozen_probe_comparison(source: Path, output_base: Path) -> tuple[Path, Path]:
    data = _read_json(source)
    if data.get("status") != "validated" or data.get("sealed_test_used") is not False:
        raise ValueError("Frozen-probe figure requires a validated run without sealed-test use")
    if data.get("validation_events_selected") != 2000:
        raise ValueError("Frozen-probe validation population is not the expected 2,000 events")

    knots = data["KNOT_J_reference_metrics"]["mtt_posterior_mean"]
    knot_rmse = knots["RMSE_GeV"]
    knot_ci = knots["event_bootstrap_95pct_interval"]["RMSE"]
    _finite_values([knot_rmse, *knot_ci], "KNOT-J mtt RMSE")

    fig, axis = plt.subplots(figsize=(10.2, 5.7))
    y = np.arange(len(PROBE_ORDER))
    colors = {"linear": "#0072B2", "mlp": "#D55E00"}
    markers = {"linear": "o", "mlp": "s"}
    offsets = {"linear": -0.11, "mlp": 0.11}

    axis.axvspan(knot_ci[0], knot_ci[1], color="#B9BEC5", alpha=0.25, linewidth=0)
    axis.axvline(knot_rmse, color="#343A40", linewidth=2, label="KNOT-J posterior mean")
    for kind in ("linear", "mlp"):
        values = []
        intervals = []
        for condition in PROBE_ORDER:
            metrics = data["metrics"][kind][condition]["validation_metrics"]
            values.append(metrics["mtt_posterior_probe_RMSE_GeV"])
            intervals.append(metrics["event_bootstrap_95pct_interval"]["RMSE_GeV"])
        _finite_values(values, f"{kind} probe mtt RMSE")
        axis.errorbar(
            values,
            y + offsets[kind],
            xerr=_interval_errors(values, intervals),
            marker=markers[kind],
            linestyle="none",
            color=colors[kind],
            capsize=3,
            label="Linear ridge" if kind == "linear" else "Small MLP",
        )

    axis.set_yticks(y, PROBE_LABELS)
    axis.invert_yaxis()
    axis.set_xlim(65, 195)
    axis.set_xlabel(r"$m_{t\bar{t}}$ RMSE [GeV]")
    axis.set_title("Frozen KNOT-J encoder probes on common validation events")
    handles, labels = axis.get_legend_handles_labels()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="x", color="#E6E8EB", linewidth=0.7)
    axis.set_axisbelow(True)
    fig.legend(handles, labels, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Frozen KNOT-J encoder probes on common validation events", fontsize=15, y=0.99)
    fig.text(
        0.01,
        0.01,
        "2,000 validation events; horizontal bars are 95% event-bootstrap intervals. "
        "$A_{truth}$ and $N_{truth}$ rows use oracle truth context and are diagnostic only. "
        "The shaded band is the KNOT-J 95% interval.",
        fontsize=8,
    )
    axis.set_title("")
    fig.tight_layout(rect=(0, 0.055, 1, 0.88))
    return _save_figure(fig, output_base)


def plot_kk_vyper_comparison(source: Path, output_base: Path) -> tuple[Path, Path]:
    data = _read_json(source)
    if (
        data.get("status") != "completed"
        or data.get("dsid") != 574801
        or data.get("same_exact_event_join") is not True
        or data.get("events") != 8721
    ):
        raise ValueError("KK comparison requires the validated exact 8,721-event DSID 574801 join")

    model_order = ("M", "C", "J", "VyPER")
    colors = {"M": "#CC79A7", "C": "#009E73", "J": "#0072B2", "VyPER": "#6B7280"}
    x = np.arange(len(model_order))

    mtt_rmse = [data["metrics"][name]["mtt"]["RMSE_GeV"] for name in model_order]
    mtt_ci = [
        data["metrics"][name]["mtt"]["event_bootstrap_95pct_interval"]["RMSE"]
        for name in model_order
    ]
    _finite_values(mtt_rmse, "KK mtt RMSE")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.2, 5.8),
        gridspec_kw={"width_ratios": (1.0, 1.35, 1.4)},
    )
    axes[0].set_title(r"$m_{t\bar{t}}$ RMSE")
    for i, model in enumerate(model_order):
        axes[0].errorbar(
            i,
            mtt_rmse[i],
            yerr=_interval_errors([mtt_rmse[i]], [mtt_ci[i]]),
            color=colors[model],
            marker="o",
            capsize=3,
        )
    axes[0].set_xticks(x, model_order)
    axes[0].set_ylabel("RMSE [GeV]")

    role_offsets = {"top": -0.12, "antitop": 0.12}
    role_markers = {"top": "o", "antitop": "s"}
    axes[1].set_title("Parent $p_T$ RMSE")
    for i, model in enumerate(model_order):
        for role in ("top", "antitop"):
            metric = data["metrics"][model]["parents"][role]
            value = metric["pT_RMSE_GeV"]
            interval = metric["pT_event_bootstrap_95pct_interval"]["RMSE"]
            _finite_values([value, *interval], f"{model} {role} parent pT RMSE")
            axes[1].errorbar(
                i + role_offsets[role],
                value,
                yerr=_interval_errors([value], [interval]),
                color=colors[model],
                marker=role_markers[role],
                linestyle="none",
                capsize=3,
            )
    axes[1].set_xticks(x, model_order)
    axes[1].set_ylabel("RMSE [GeV]")

    difference_metrics = (
        ("mtt", "mtt_RMSE_vs_VyPER"),
        ("top parent $p_T$", "top"),
        ("antitop parent $p_T$", "antitop"),
    )
    y_centers = np.arange(len(difference_metrics))
    model_offsets = {"M": -0.18, "C": 0.0, "J": 0.18}
    axes[2].set_title("Paired KNOT − VyPER RMSE")
    for y_center, (label, key) in zip(y_centers, difference_metrics, strict=True):
        for model in ("M", "C", "J"):
            paired = data["paired_KNOT_minus_VyPER"][model]
            difference = (
                paired[key]["first_minus_second_RMSE_GeV"]
                if key == "mtt_RMSE_vs_VyPER"
                else paired["parent_pT_RMSE_vs_VyPER"][key]["first_minus_second_RMSE_GeV"]
            )
            interval = (
                paired[key]["event_bootstrap_95pct_interval_GeV"]
                if key == "mtt_RMSE_vs_VyPER"
                else paired["parent_pT_RMSE_vs_VyPER"][key]["event_bootstrap_95pct_interval_GeV"]
            )
            _finite_values([difference, *interval], f"{model} paired {label} difference")
            axes[2].errorbar(
                difference,
                y_center + model_offsets[model],
                xerr=_interval_errors([difference], [interval]),
                color=colors[model],
                marker="o",
                linestyle="none",
                capsize=3,
            )
    axes[2].axvline(0.0, color="#555555", linestyle="--", linewidth=1)
    axes[2].set_yticks(y_centers, [label for label, _ in difference_metrics])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("RMSE difference [GeV]")

    for axis in axes:
        _style_axis(axis)
    axes[0].set_xlabel("Model")
    axes[1].set_xlabel("Model")
    from matplotlib.lines import Line2D

    fig.legend(
        [Line2D([], [], color=colors[model], marker="o", linestyle="none") for model in model_order],
        model_order,
        frameon=False,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle("Frozen KNOT and VyPER on the same 2 TeV candidate", fontsize=15, y=0.99)
    fig.text(
        0.01,
        0.01,
        "DSID 574801; 8,721 exact events; truth is partonic after-FSR. "
        "95% event-bootstrap intervals; circles/squares show top/antitop parent $p_T$. "
        "Negative paired difference favors KNOT. Assignment and neutrino results are in the "
        "companion charge-side contract figure. e8588-to-JO lineage is unverified; "
        "this is one candidate mass, not a mass trend.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.87))
    return _save_figure(fig, output_base)


def plot_kk_vyper_assignment_neutrino(
    source: Path, output_base: Path
) -> tuple[Path, Path]:
    data = _read_json(source)
    if (
        data.get("status") != "completed"
        or data.get("dsid") != 574801
        or data.get("same_exact_ROOT_event_join") is not True
        or data.get("event_count") != 8721
        or data.get("truth_assignment_supported_events") != 4618
    ):
        raise ValueError("Charge-side comparison requires the validated DSID 574801 event contract")

    method_order = ("VyPER", "M", "C", "J")
    labels = ("VyPER", "KNOT-M", "KNOT-C", "KNOT-J")
    colors = {"VyPER": "#6B7280", "M": "#CC79A7", "C": "#009E73", "J": "#0072B2"}
    assignment_values = []
    assignment_intervals = []
    neutrino_values = []
    neutrino_intervals = []
    for method in method_order:
        if method == "VyPER":
            assignment = data["VyPER"]["assignment"]["complete"]
            neutrino = data["VyPER"]
        else:
            knot = data["KNOT"][method]
            assignment = knot["assignment_MAP"]
            neutrino = knot
        assignment_values.append(float(assignment["accuracy"] if method == "VyPER" else assignment["complete_accuracy"]))
        assignment_intervals.append(assignment["event_bootstrap_95pct_interval"])
        neutrino_values.append(float(neutrino["neutrino_3vector_RMSE_GeV"]))
        neutrino_intervals.append(
            neutrino["neutrino_3vector_event_bootstrap_95pct_interval_GeV"]
        )
    _finite_values(assignment_values, "KK exact-MAP assignment accuracy")
    _finite_values(neutrino_values, "KK neutrino 3-vector RMSE")

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.6), gridspec_kw={"wspace": 0.32})
    y = np.arange(len(method_order))
    for index, method in enumerate(method_order):
        accuracy = assignment_values[index] * 100.0
        accuracy_interval = np.asarray(assignment_intervals[index], dtype=float) * 100.0
        axes[0].errorbar(
            accuracy,
            index,
            xerr=np.asarray([[accuracy - accuracy_interval[0]], [accuracy_interval[1] - accuracy]]),
            marker="o",
            linestyle="none",
            color=colors[method],
            capsize=3,
        )
        rmse = neutrino_values[index]
        rmse_interval = neutrino_intervals[index]
        axes[1].errorbar(
            rmse,
            index,
            xerr=_interval_errors([rmse], [rmse_interval]),
            marker="o",
            linestyle="none",
            color=colors[method],
            capsize=3,
        )

    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(90.0, 100.0)
    axes[0].set_xlabel("Complete assignment accuracy [%]")
    axes[0].set_title("Exact MAP on supported events")
    axes[0].text(
        0.02,
        0.03,
        "4,618 supported events",
        transform=axes[0].transAxes,
        fontsize=8,
    )

    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Neutrino 3-vector RMSE [GeV]")
    axes[1].set_title("Charge-side neutrinos on all events")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(direction="out")
        axis.grid(axis="x", color="#E6E8EB", linewidth=0.7)
        axis.set_axisbelow(True)
    fig.suptitle("KNOT and VyPER charge-side comparison on the 2 TeV candidate", fontsize=14)
    fig.text(
        0.01,
        0.015,
        "Error bars: 95% event-bootstrap intervals. Assignment compares KNOT model MAP with "
        "deterministic VyPER on 4,618 truth-supported events. Neutrinos use all 8,721 events; "
        "KNOT is the mean of 32 sampled (px, py, pz) vectors and VyPER uses stored charge-side "
        "branches. One DSID 574801 candidate only; e8588-to-JO lineage is unverified.",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.92))
    return _save_figure(fig, output_base)


def main() -> None:
    posterior_source = ROOT / "results/experiments/knot/posterior-convergence-v1/event-paired-summary-v2.json"
    probe_source = ROOT / "results/experiments/knot/encoder-probes-v3/summary.json"
    kk_vyper_source = ROOT / "results/experiments/knot/kk-vyper-parent-v1/all-models.json"
    kk_contract_source = ROOT / "results/experiments/knot/kk-vyper-contract-v2/metrics.json"
    posterior_output = ROOT / "results/experiments/knot/posterior-convergence-v1/convergence-summary"
    probe_output = ROOT / "results/experiments/knot/encoder-probes-v3/frozen-probe-mtt-rmse"
    kk_vyper_output = ROOT / "results/experiments/knot/kk-vyper-parent-v1/frozen-knot-vyper-candidate-comparison"
    kk_contract_output = ROOT / "results/experiments/knot/kk-vyper-contract-v2/assignment-neutrino-comparison"
    for path in (posterior_source, probe_source, kk_vyper_source, kk_contract_source):
        if not path.is_file():
            raise FileNotFoundError(f"Required plotting input is missing: {path}")
    posterior_paths = plot_posterior_convergence(posterior_source, posterior_output)
    probe_paths = plot_frozen_probe_comparison(probe_source, probe_output)
    kk_vyper_paths = plot_kk_vyper_comparison(kk_vyper_source, kk_vyper_output)
    kk_contract_paths = plot_kk_vyper_assignment_neutrino(
        kk_contract_source, kk_contract_output
    )
    for path in (*posterior_paths, *probe_paths, *kk_vyper_paths, *kk_contract_paths):
        print(path)


if __name__ == "__main__":
    main()
