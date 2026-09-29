#!/usr/bin/env python3
"""Plot full-range validation diagnostics from the immutable REALNU R2 event table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARENT_ROLES = ("top", "antitop")
NU_ROLES = ("top.nu", "antitop.nu")
PARENT_MODELS = ("PCFM_Q", "J0", "J_A", "J_AN")
CONSTITUENT_MODELS = ("J0", "J_A", "J_AN")
STANDARD_MODELS = ("Q0", "Q1", "J0", "J_A", "J_AN")
COLORS = {
    "Perfect reco": "#202124",
    "PCFM_Q": "#65727E",
    "J0": "#E18A21",
    "J_A": "#27956D",
    "J_AN": "#8053A2",
    "Q0": "#2878B5",
    "Q1": "#C43C39",
}
ORACLE_PATHS = (
    "perfect_reco",
    "predicted_assignment_truth_neutrinos",
    "truth_assignment_predicted_neutrinos",
    "predicted_assignment_predicted_neutrinos",
)
ORACLE_LABELS = {
    "perfect_reco": "Perfect reco",
    "predicted_assignment_truth_neutrinos": "Wrong assignment, truth neutrinos",
    "truth_assignment_predicted_neutrinos": "Truth assignment, predicted neutrinos",
    "predicted_assignment_predicted_neutrinos": "Full constituent route",
}


def _p4_observables(p4: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(p4, dtype=np.float64)
    energy, px, py, pz = np.moveaxis(values, -1, 0)
    pt = np.hypot(px, py)
    mass_squared = energy * energy - px * px - py * py - pz * pz
    scale = np.maximum(energy * energy + px * px + py * py + pz * pz, 1.0)
    mass = np.where(mass_squared >= -1e-10 * scale, np.sqrt(np.maximum(mass_squared, 0.0)), np.nan)
    rapidity = np.full_like(energy, np.nan)
    valid = (energy + pz > 0) & (energy - pz > 0)
    rapidity[valid] = 0.5 * np.log((energy[valid] + pz[valid]) / (energy[valid] - pz[valid]))
    return {
        "pT": pt,
        "y": rapidity,
        "phi": np.arctan2(py, px),
        "mass": mass,
        "p4": values,
    }


def _wrap(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def _metrics(truth: np.ndarray, predicted: np.ndarray, *, angular: bool) -> dict[str, float | int]:
    valid = np.isfinite(truth) & np.isfinite(predicted)
    truth = np.asarray(truth[valid], dtype=np.float64)
    predicted = np.asarray(predicted[valid], dtype=np.float64)
    residual = _wrap(predicted - truth) if angular else predicted - truth
    if not len(residual):
        return {"N": 0}
    result: dict[str, float | int] = {
        "N": int(len(residual)),
        "bias": float(residual.mean()),
        "std": float(residual.std()),
        "sigma68": float((np.quantile(residual, 0.84) - np.quantile(residual, 0.16)) / 2.0),
        "MAE": float(np.abs(residual).mean()),
        "RMSE": float(np.sqrt(np.mean(residual * residual))),
    }
    finite_positive = truth > 0
    if not angular and finite_positive.any():
        fractional = residual[finite_positive] / truth[finite_positive]
        result["fractional_bias"] = float(fractional.mean())
        result["fractional_RMSE"] = float(np.sqrt(np.mean(fractional * fractional)))
    return result


def _read_inputs(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Accepted R2 event table is missing: {path}")
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema") != "realnu_event_metrics_v1":
            raise ValueError(f"Unexpected R2 event-table schema: {source.attrs.get('schema')!r}")
        if str(source.attrs.get("sealed_test_used", "")).lower() != "false":
            raise ValueError("R2 event table does not establish sealed_test_used=false")
        if source.attrs.get("parent_role_order") != '["top", "antitop"]':
            raise ValueError("R2 parent truth roles are not [top, antitop]")
        if source.attrs.get("neutrino_role_order") != '["top.nu", "antitop.nu"]':
            raise ValueError("R2 neutrino roles are not [top.nu, antitop.nu]")
        ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        truth_mtt = np.asarray(source["truth/mttbar"][:], dtype=np.float64)
        if len(ids) != 100_342 or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Expected 100342 unique validation events, found {len(ids)}")
        if not np.isfinite(truth_mtt).all():
            raise ValueError("Validation truth mttbar contains non-finite values")
        result = {
            "ids": ids,
            "truth_mttbar": truth_mtt,
            "truth_parents": np.asarray(source["truth/parents"][:], dtype=np.float64),
            "truth_neutrinos": np.asarray(source["truth/neutrinos"][:], dtype=np.float64),
            "truth_parent_support": np.asarray(source["support/parent_roles"][:], dtype=bool),
            "truth_neutrino_support": np.asarray(source["support/neutrino_roles"][:], dtype=bool),
            "models": {},
        }
        for name in PARENT_MODELS:
            if name not in source["models"]:
                raise KeyError(f"Missing required direct-parent model in R2 table: {name}")
            group = source[f"models/{name}"]
            result["models"][name] = {
                "parents": np.asarray(group["parent_joint_medoid"][:], dtype=np.float64),
                "mttbar": np.asarray(group["parent_mttbar_joint_medoid"][:], dtype=np.float64),
            }
        for name in STANDARD_MODELS:
            group = source[f"models/{name}"]
            result["models"].setdefault(name, {})["neutrinos"] = np.asarray(
                group["neutrino_joint_medoid"][:], dtype=np.float64
            )
            if name in CONSTITUENT_MODELS:
                for path_name in ORACLE_PATHS:
                    oracle = group[f"constituent_oracles/{path_name}"]
                    result["models"][name].setdefault("constituent", {})[path_name] = {
                        "parents": np.asarray(oracle["parents"][:], dtype=np.float64),
                        "mttbar": np.asarray(oracle["mttbar"][:], dtype=np.float64),
                    }
        return result


def _series(data: dict, observable: str) -> tuple[np.ndarray, dict[str, np.ndarray], bool, str]:
    truth = data["truth_mttbar"] if observable == "mttbar" else None
    angular = observable.endswith("phi")
    unit = "GeV" if observable in {"mttbar", "top_pT", "antitop_pT", "ttbar_pT", "nu_top_pT", "nu_antitop_pT"} else ("rad" if angular else "")
    if observable == "mttbar":
        predictions: dict[str, np.ndarray] = {"Perfect reco": data["models"]["J0"]["constituent"]["perfect_reco"]["mttbar"]}
        for mode in PARENT_MODELS:
            predictions[mode] = data["models"][mode]["mttbar"]
        for mode in CONSTITUENT_MODELS:
            predictions[f"{mode} full constituent"] = data["models"][mode]["constituent"]["predicted_assignment_predicted_neutrinos"]["mttbar"]
        return truth, predictions, angular, unit

    truth_parent_obs = _p4_observables(data["truth_parents"])
    truth_pair = data["truth_parents"][:, 0] + data["truth_parents"][:, 1]
    truth_pair_obs = _p4_observables(truth_pair)
    if observable.startswith("nu_"):
        role = 0 if observable.startswith("nu_top") else 1
        component = "pT" if observable.endswith("pT") else "pz"
        truth = np.hypot(data["truth_neutrinos"][:, role, 0], data["truth_neutrinos"][:, role, 1]) if component == "pT" else data["truth_neutrinos"][:, role, 2]
        predictions = {}
        for mode in STANDARD_MODELS:
            values = data["models"][mode]["neutrinos"][:, role]
            predictions[mode] = np.hypot(values[:, 0], values[:, 1]) if component == "pT" else values[:, 2]
        return truth, predictions, angular, unit

    role = 0 if observable.startswith("top_") else 1
    object_obs = observable.removeprefix("top_").removeprefix("antitop_")
    if observable.startswith("ttbar_"):
        key = observable.removeprefix("ttbar_")
        truth = truth_pair_obs[key]
    else:
        key = object_obs
        truth = truth_parent_obs[key][:, role]
    predictions = {"Perfect reco": np.full_like(truth, np.nan)}
    for mode in CONSTITUENT_MODELS:
        p4 = data["models"][mode]["constituent"]["perfect_reco"]["parents"]
        if observable.startswith("ttbar_"):
            p4 = p4[:, 0] + p4[:, 1]
            predictions["Perfect reco"] = _p4_observables(p4)[key]
        else:
            predictions["Perfect reco"] = _p4_observables(p4)[key][:, role]
        break
    for mode in PARENT_MODELS:
        p4 = data["models"][mode]["parents"]
        if observable.startswith("ttbar_"):
            p4 = p4[:, 0] + p4[:, 1]
            predictions[mode] = _p4_observables(p4)[key]
        else:
            predictions[mode] = _p4_observables(p4)[key][:, role]
    return truth, predictions, angular, unit


def _plot_observable(name: str, truth: np.ndarray, predictions: dict[str, np.ndarray], angular: bool, unit: str, output: Path) -> list[dict]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 7, "xtick.direction": "in",
        "ytick.direction": "in", "savefig.bbox": "tight",
    })
    records = []
    for label, prediction in predictions.items():
        record = {"observable": name, "path": label, **_metrics(truth, prediction, angular=angular)}
        records.append(record)
    finite_truth = truth[np.isfinite(truth)]
    if not len(finite_truth):
        raise ValueError(f"No finite truth values for {name}")
    finite_reco = np.concatenate(
        [np.asarray(values, dtype=np.float64)[np.isfinite(values)] for values in predictions.values()]
    )
    low = min(float(finite_truth.min()), float(finite_reco.min()))
    high = max(float(finite_truth.max()), float(finite_reco.max()))
    if high <= low:
        low, high = low - 1.0, high + 1.0
    margin = max((high - low) * 0.015, 1e-6)
    full_range = (low - margin, high + margin)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
    ax = axes[0, 0]
    bins = np.linspace(*full_range, 81)
    ax.hist(finite_truth, bins=bins, histtype="step", color=COLORS["Perfect reco"], linewidth=2.0, label="Parton truth")
    for label, prediction in predictions.items():
        values = prediction[np.isfinite(prediction)]
        if len(values):
            ax.hist(values, bins=bins, histtype="step", linewidth=1.2, color=COLORS.get(label.split()[0], "#777777"), label=label)
    ax.set(xlabel=f"{name} [{unit}]" if unit else name, ylabel="Events", title="Full finite-range spectrum", xlim=full_range)
    ax.set_yscale("log")
    ax.legend(ncol=1, loc="best")

    ax = axes[0, 1]
    residuals = []
    for label, prediction in predictions.items():
        valid = np.isfinite(truth) & np.isfinite(prediction)
        residual = _wrap(prediction[valid] - truth[valid]) if angular else prediction[valid] - truth[valid]
        if len(residual):
            residuals.append((label, residual))
    residual_low = min(float(values.min()) for _, values in residuals)
    residual_high = max(float(values.max()) for _, values in residuals)
    if residual_high <= residual_low:
        residual_low, residual_high = residual_low - 1.0, residual_high + 1.0
    residual_bins = np.linspace(residual_low, residual_high, 81)
    for label, values in residuals:
        ax.hist(values, bins=residual_bins, histtype="step", linewidth=1.2, color=COLORS.get(label.split()[0], "#777777"), label=label)
    ax.axvline(0, color="black", linewidth=1)
    ax.set(xlabel=f"Reco − truth [{unit}]" if unit else "Reco − truth", ylabel="Events", title="Residuals over full finite range")
    ax.set_yscale("log")
    ax.legend(fontsize=6, loc="best")

    ax = axes[1, 0]
    label, prediction = next(iter(predictions.items()))
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if valid.any():
        h = ax.hist2d(truth[valid], prediction[valid], bins=(80, 80), range=(full_range, full_range), cmap="Blues", cmin=1, norm=matplotlib.colors.LogNorm())
    ax.plot(full_range, full_range, color="#C43C39", linestyle="--", linewidth=1)
    ax.set(xlabel=f"Parton truth [{unit}]" if unit else "Parton truth", ylabel=f"{label} [{unit}]" if unit else label, title=f"{label}: truth vs reco (log density)", xlim=full_range, ylim=full_range)
    if "h" in locals():
        fig.colorbar(h[3], ax=ax, label="Events / bin")

    ax = axes[1, 1]
    differential_edges = np.linspace(low, high + 1e-12, 13)
    centers = 0.5 * (differential_edges[:-1] + differential_edges[1:])
    for label, prediction in predictions.items():
        bias, rmse = [], []
        for lo, hi in zip(differential_edges[:-1], differential_edges[1:]):
            valid = np.isfinite(truth) & np.isfinite(prediction) & (truth >= lo) & (truth < hi if hi < differential_edges[-1] else truth <= hi)
            residual = _wrap(prediction[valid] - truth[valid]) if angular else prediction[valid] - truth[valid]
            bias.append(float(residual.mean()) if len(residual) else np.nan)
            rmse.append(float(np.sqrt(np.mean(residual * residual))) if len(residual) else np.nan)
        ax.plot(centers, bias, marker="o", markersize=2.8, linewidth=1.0, color=COLORS.get(label.split()[0], "#777777"), label=f"{label} bias")
        ax.plot(centers, rmse, linestyle="--", marker=".", markersize=2.3, linewidth=0.9, color=COLORS.get(label.split()[0], "#777777"), label=f"{label} RMSE")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel=f"Parton truth [{unit}]" if unit else "Parton truth", ylabel=f"Residual metric [{unit}]" if unit else "Residual metric", title="Differential bias and RMSE")
    ax.legend(ncol=2, fontsize=6, loc="best")
    fig.suptitle(name.replace("_", " "))
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.pdf")
    fig.savefig(output / f"{name}.png", dpi=180)
    plt.close(fig)

    correlation_labels = list(predictions)
    columns = 3
    rows = (len(correlation_labels) + columns - 1) // columns
    fig, correlation_axes = plt.subplots(rows, columns, figsize=(13.0, 4.0 * rows), constrained_layout=True, squeeze=False)
    for ax, label in zip(correlation_axes.flat, correlation_labels):
        prediction = predictions[label]
        valid = np.isfinite(truth) & np.isfinite(prediction)
        if valid.any():
            histogram, xedges, yedges = np.histogram2d(
                truth[valid], prediction[valid], bins=(80, 80), range=(full_range, full_range)
            )
            positive = histogram[histogram > 0]
            if positive.size:
                mesh = ax.pcolormesh(
                    xedges, yedges, histogram.T, shading="auto", cmap="Blues",
                    norm=matplotlib.colors.LogNorm(vmin=max(1.0, float(positive.min())), vmax=float(positive.max())),
                )
                fig.colorbar(mesh, ax=ax, label="Events / bin")
        ax.plot(full_range, full_range, color="#C43C39", linestyle="--", linewidth=1)
        ax.set(xlabel=f"Parton truth [{unit}]" if unit else "Parton truth", ylabel=f"{label} [{unit}]" if unit else label, title=label, xlim=full_range, ylim=full_range)
    for ax in correlation_axes.flat[len(correlation_labels):]:
        ax.remove()
    fig.suptitle(f"{name.replace('_', ' ')}: truth vs reconstruction, full range")
    fig.savefig(output / f"{name}_truth_vs_reco.pdf")
    fig.savefig(output / f"{name}_truth_vs_reco.png", dpi=180)
    plt.close(fig)

    if name == "mttbar":
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        for ax, xlim, title in (
            (axes[0], (0.0, high), "Full finite range"),
            (axes[1], (345.0, 500.0), "345–500 GeV threshold"),
        ):
            bins = np.linspace(xlim[0], xlim[1], 61)
            ax.hist(finite_truth, bins=bins, histtype="step", color="black", linewidth=1.8, label="Parton truth")
            for label, prediction in predictions.items():
                values = prediction[np.isfinite(prediction)]
                if len(values):
                    ax.hist(values, bins=bins, histtype="step", linewidth=1.0, color=COLORS.get(label.split()[0], "#777777"), label=label)
            ax.set(xlim=xlim, xlabel="mttbar [GeV]", ylabel="Events", title=title)
            ax.set_yscale("log")
            ax.axvline(345, color="#444444", linestyle=":", linewidth=0.8)
            ax.axvline(400, color="#444444", linestyle=":", linewidth=0.8)
        axes[1].legend(fontsize=6, loc="best")
        fig.savefig(output / "mttbar_full_and_threshold.pdf")
        fig.savefig(output / "mttbar_full_and_threshold.png", dpi=180)
        plt.close(fig)
    return records


def _write_migration(truth: np.ndarray, predictions: dict[str, np.ndarray], output: Path) -> None:
    finite_max = max(float(np.nanmax(truth)), *(float(np.nanmax(x[np.isfinite(x)])) for x in predictions.values() if np.isfinite(x).any()))
    edges = np.asarray([0.0, 345.0, 400.0, 500.0, 700.0, 1000.0, max(finite_max + 1.0, 1000.01)])
    names = ("<345", "345–400", "400–500", "500–700", "700–1000", ">=1000")
    tables = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    models = {"Perfect reco": predictions["Perfect reco"], **{name: values for name, values in predictions.items() if name in PARENT_MODELS}}
    active_axes = []
    for ax, (label, predicted) in zip(axes.flat, models.items()):
        active_axes.append(ax)
        valid = np.isfinite(truth) & np.isfinite(predicted) & (truth >= 0) & (predicted >= 0)
        matrix, _, _ = np.histogram2d(truth[valid], predicted[valid], bins=(edges, edges))
        denominator = matrix.sum(axis=1, keepdims=True)
        percent = np.divide(100.0 * matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0)
        image = ax.imshow(percent, origin="lower", vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.plot(np.arange(len(names)), np.arange(len(names)), color="#C43C39", linewidth=1.2)
        for i in range(len(names)):
            ax.add_patch(matplotlib.patches.Rectangle(
                (i - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                edgecolor="#C43C39", linewidth=1.2,
            ))
        ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)), xticklabels=names, yticklabels=names, xlabel="Reco mttbar bin", ylabel="Truth mttbar bin", title=label)
        ax.tick_params(axis="x", rotation=35)
        for i in range(len(names)):
            for j in range(len(names)):
                if denominator[i, 0] > 0:
                    ax.text(j, i, f"{percent[i, j]:.0f}", ha="center", va="center", fontsize=6, color="black")
        for i in range(len(names)):
            for j in range(len(names)):
                tables.append({"path": label, "truth_bin": names[i], "reco_bin": names[j], "percent": float(percent[i, j]), "N_truth": int(denominator[i, 0])})
    for ax in axes.flat[len(models):]:
        ax.remove()
    fig.colorbar(image, ax=active_axes, label="Row-normalised events [%]")
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "mttbar_migration.pdf")
    fig.savefig(output / "mttbar_migration.png", dpi=180)
    plt.close(fig)
    with (output.parent / "tables" / "mttbar_migration.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)


def run(production_root: Path) -> None:
    production_root = production_root.expanduser().resolve()
    diagnostic_root = production_root / "evaluation_v3_diagnostics"
    output = diagnostic_root / "plots"
    if output == production_root / "evaluation_r2" or production_root / "evaluation_r2" in output.parents:
        raise ValueError("New plots must not be written under evaluation_r2")
    data = _read_inputs(production_root / "evaluation_r2/core/event_metrics.h5")
    observables = (
        "mttbar", "top_pT", "antitop_pT", "top_y", "antitop_y", "ttbar_pT", "ttbar_y",
        "top_phi", "antitop_phi", "nu_top_pz", "nu_antitop_pz", "nu_top_pT", "nu_antitop_pT",
    )
    rows = []
    for name in observables:
        truth, predictions, angular, unit = _series(data, name)
        rows.extend(_plot_observable(name, truth, predictions, angular, unit, output))
        if name == "mttbar":
            _write_migration(truth, predictions, output)
    table_dir = diagnostic_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (table_dir / "observable_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"events={len(data['ids'])} sealed_test_used=false plots={output} metrics={table_dir / 'observable_metrics.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.production_root)


if __name__ == "__main__":
    main()
