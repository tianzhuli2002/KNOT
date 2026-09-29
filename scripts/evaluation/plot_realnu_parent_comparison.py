#!/usr/bin/env python3
"""Create standalone figures and a concise report from parent reductions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import h5py

PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
PRODUCTION_ROOT = PROJECT_ROOT / (
    "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/"
    "realnu_top_antitop_20260927_release2"
)
DEFAULT_EVENT_TABLE = PRODUCTION_ROOT / "evaluation_r2/core/event_metrics.h5"


CORE_OBSERVABLES = ("mttbar", "ptttbar", "top_mass")
ESTIMATORS = ("single_draw", "observable_mean", "observable_q50", "joint_medoid")
LABELS = {
    "single_draw": "Single draw",
    "observable_mean": "Observable mean",
    "observable_q50": "Observable q50",
    "joint_medoid": "Joint medoid",
}
OBSERVABLE_LABELS = {
    "mttbar": r"$m_{t\bar{t}}$ [GeV]",
    "ptttbar": r"$p_{T,t\bar{t}}$ [GeV]",
    "top_mass": r"$m_t$ [GeV]",
}
MODEL_COLORS = {
    "PCFM_noQ": "#0072B2",
    "PCFM_Q": "#D55E00",
    "PCFM_geom": "#009E73",
}
MASS_REGION_LABELS = ("<345", "345–400", "400–500", "500–700", "700–1000", ">1000")


def _color(label: str, index: int) -> str:
    for key, value in MODEL_COLORS.items():
        if key.lower() in label.lower():
            return value
    return plt.get_cmap("tab10")(index)


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


def _mass(parents: np.ndarray) -> np.ndarray:
    parents = np.asarray(parents, dtype=np.float64)
    summed = parents.sum(axis=-2)
    mass2 = summed[..., 0] ** 2 - np.sum(summed[..., 1:] ** 2, axis=-1)
    if np.any(mass2 < -1.0e-3):
        raise ValueError("A plotted parent pair is spacelike")
    return np.sqrt(np.maximum(mass2, 0.0))


def _save(figure, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(target.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return target.with_suffix(".png")


def _load(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if value.get("schema") != "realnu_parent_posterior_reduction_v1":
        raise ValueError(f"Unsupported parent-reduction summary: {path}")
    if value.get("sealed_test_used") is not False:
        raise ValueError(f"Parent summary must record sealed_test_used=false: {path}")
    arrays_path = Path(value["arrays"]).expanduser().resolve()
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return value, arrays


def _align_records(records: list[tuple[str, dict, dict[str, np.ndarray]]]) -> list[tuple[str, dict, dict[str, np.ndarray]]]:
    if not records:
        raise ValueError("At least one parent reduction summary is required")
    label, first, first_arrays = records[0]
    reference_ids = np.asarray(first_arrays["EVENT_ID"], dtype=np.int64)
    if len(np.unique(reference_ids)) != len(reference_ids):
        raise ValueError(f"{label} contains duplicate EVENT_IDs")
    aligned = [(label, first, first_arrays)]
    for label, record, arrays in records[1:]:
        for key in ("source_bank_sha256", "case", "solver", "steps", "samples"):
            if record.get(key) != first.get(key):
                raise ValueError(f"{label} is not paired with the first model: {key} differs")
        candidate_ids = np.asarray(arrays["EVENT_ID"], dtype=np.int64)
        if len(np.unique(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"{label} contains duplicate EVENT_IDs")
        if set(map(int, reference_ids)) != set(map(int, candidate_ids)):
            raise ValueError(f"{label} does not cover the exact same EVENT_ID set")
        positions = {int(event_id): i for i, event_id in enumerate(candidate_ids)}
        reorder = np.asarray([positions[int(event_id)] for event_id in reference_ids])
        arrays = {name: value[reorder] for name, value in arrays.items()}
        for name in arrays:
            if name.startswith("truth__") and not np.allclose(
                first_arrays[name], arrays[name], rtol=1.0e-6, atol=1.0e-5,
                equal_nan=True,
            ):
                raise ValueError(f"{label} has different exact-EVENT_ID truth for {name}")
        for name in ("channel", "source_njet", "model_visible_njet", "support_parent_roles"):
            if not np.array_equal(first_arrays[name], arrays[name]):
                raise ValueError(f"{label} has different validation region/support data: {name}")
        aligned.append((label, record, arrays))
    return aligned


def _write_rmse_plot(records: list[tuple[str, dict]], output_dir: Path) -> Path:
    figure, axes = plt.subplots(1, len(CORE_OBSERVABLES), figsize=(15, 4.8), constrained_layout=True)
    if len(CORE_OBSERVABLES) == 1:
        axes = [axes]
    width = 0.8 / max(1, len(records))
    for axis, observable in zip(axes, CORE_OBSERVABLES, strict=True):
        centers = np.arange(len(ESTIMATORS), dtype=np.float64)
        for model_index, (label, record) in enumerate(records):
            values = [
                record["metrics"][estimator][observable]["inclusive"]["rmse"]
                for estimator in ESTIMATORS
            ]
            offsets = (model_index - (len(records) - 1) / 2.0) * width
            axis.bar(centers + offsets, values, width=width, label=label)
        axis.set_xticks(centers, [LABELS[name] for name in ESTIMATORS], rotation=22, ha="right")
        axis.set_ylabel("RMSE")
        axis.set_title(OBSERVABLE_LABELS[observable])
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    figure.suptitle("Real-neutrino parent validation: inclusive point-estimator RMSE")
    target = output_dir / "inclusive_rmse.png"
    figure.savefig(target, dpi=240)
    figure.savefig(target.with_suffix(".pdf"))
    plt.close(figure)
    return target


def _write_njet_plot(records: list[tuple[str, dict]], output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for label, record in records:
        values = record["metrics"]["joint_medoid"]["mttbar"]
        jet_bins = sorted(
            (name for name in values if name.startswith("model_visible_njet/")),
            key=lambda name: int(name.rsplit("/", 1)[1]),
        )
        x = [int(name.rsplit("/", 1)[1]) for name in jet_bins]
        y = [values[name]["rmse"] for name in jet_bins]
        axis.plot(x, y, marker="o", linewidth=1.8, label=label)
    axis.set_xlabel("Measured visible jet multiplicity")
    axis.set_ylabel(r"Joint medoid $m_{t\bar{t}}$ RMSE [GeV]")
    axis.set_title("Parent-mass resolution by measured jet multiplicity")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    target = output_dir / "mttbar_rmse_by_visible_njet.png"
    figure.savefig(target, dpi=240)
    figure.savefig(target.with_suffix(".pdf"))
    plt.close(figure)
    return target


def _write_coverage_plot(records: list[tuple[str, dict]], output_dir: Path) -> Path:
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    keys = [(estimator, observable) for estimator in ESTIMATORS for observable in CORE_OBSERVABLES]
    centers = np.arange(len(keys), dtype=np.float64)
    width = 0.8 / max(1, len(records))
    for model_index, (label, record) in enumerate(records):
        values = [
            record["metrics"][estimator][observable]["inclusive"]["truth_coverage_68"]
            for estimator, observable in keys
        ]
        offsets = (model_index - (len(records) - 1) / 2.0) * width
        axis.bar(centers + offsets, values, width=width, label=label)
    axis.axhline(0.68, color="black", linestyle="--", linewidth=1, label="Nominal 68%")
    axis.set_xticks(
        centers,
        [f"{LABELS[estimator]}\n{observable}" for estimator, observable in keys],
        rotation=35,
        ha="right",
    )
    axis.set_ylabel("Truth coverage of posterior 16–84% interval")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Posterior interval coverage")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    target = output_dir / "posterior_coverage68.png"
    figure.savefig(target, dpi=240)
    figure.savefig(target.with_suffix(".pdf"))
    plt.close(figure)
    return target


def _write_report(records: list[tuple[str, dict]], output_dir: Path) -> Path:
    lines = [
        "# Real-neutrino parent comparison",
        "",
        "Validation-only parent CFM comparison. Parent truth is joined by exact `EVENT_ID`;",
        "prediction files contain no truth arrays and the sealed test split is unopened.",
        "",
        "## Model and sampling provenance",
        "",
        "| Model | Velocity backbone | Conditioning | Checkpoint SHA256 | Source bank SHA256 | Events | Case / samples |",
        "|---|---|---|---|---|---:|---|",
    ]
    for label, record in records:
        lines.append(
            f"| {label} | {record['velocity_backbone']} | {record['conditioning']} | "
            f"`{record['checkpoint_sha256']}` | `{record['source_bank_sha256']}` | "
            f"{record['events']:,} | {record['case']} / {record['samples']} |"
        )
    lines.extend(
        [
            "",
            "The coherent joint medoid is the sampled top/antitop pair minimizing summed "
            "Euclidean distance in the train-scaled joint 8D parent space. Observable means "
            "and q50 values are reduced per observable; angles use circular statistics.",
            "",
            "## Inclusive RMSE",
            "",
            "| Model | Estimator | $m_{t\\bar{t}}$ [GeV] | $p_{T,t\\bar{t}}$ [GeV] | top mass [GeV] |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for label, record in records:
        for estimator in ESTIMATORS:
            values = [
                record["metrics"][estimator][observable]["inclusive"]["rmse"]
                for observable in CORE_OBSERVABLES
            ]
            lines.append(
                f"| {label} | {LABELS[estimator]} | "
                + " | ".join(f"{value:.4g}" for value in values)
                + " |"
            )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `inclusive_rmse.png` and `.pdf`: core observable RMSE by point estimator.",
            "- `mttbar_rmse_by_visible_njet.png` and `.pdf`: joint-medoid parent-mass resolution by jet count.",
            "- `posterior_coverage68.png` and `.pdf`: posterior 16–84% truth coverage.",
            "",
            "Complete observables, channels, jet bins, posterior widths, and coverage are in the input JSON reduction files.",
            "",
        ]
    )
    target = output_dir / "comparison_report.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _perfect_reco(
    event_table: Path, event_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    with h5py.File(event_table, "r") as table:
        if table.attrs.get("schema") != "realnu_event_metrics_v1":
            raise ValueError("Perfect-reco plotting requires the accepted event metrics table")
        if bool(table.attrs.get("sealed_test_used", True)):
            raise ValueError("Event metrics table does not establish sealed_test_used=false")
        table_ids = np.asarray(table["EVENT_ID"][:], dtype=np.int64)
        if len(np.unique(table_ids)) != len(table_ids):
            raise ValueError("Accepted event metrics table contains duplicate EVENT_IDs")
        order = np.argsort(table_ids, kind="stable")
        positions = np.searchsorted(table_ids[order], event_ids)
        if np.any(positions >= len(order)) or not np.array_equal(
            table_ids[order][positions], event_ids
        ):
            raise ValueError("Plot EVENT_IDs do not exactly join to accepted validation data")
        rows = order[positions]
        truth = np.asarray(table["truth/mttbar"][:][rows], dtype=np.float64)
        perfect_parents = np.asarray(
            table["models/Q1/constituent_oracles/perfect_reco/parents"][:][rows],
            dtype=np.float64,
        )
    finite_rows = np.isfinite(perfect_parents).all(axis=(-1, -2))
    if not finite_rows.any():
        raise ValueError("Perfect-reco mttbar reference has no finite parent pairs")
    return truth, _mass(perfect_parents[finite_rows]), int(finite_rows.sum())


def _validate_population(event_table: Path, event_ids: np.ndarray, *, allow_subset: bool) -> None:
    with h5py.File(event_table, "r") as table:
        if table.attrs.get("schema") != "realnu_event_metrics_v1":
            raise ValueError("Plot event table has an unexpected schema")
        if bool(table.attrs.get("sealed_test_used", True)):
            raise ValueError("Plot event table does not establish sealed_test_used=false")
        expected = np.asarray(table["EVENT_ID"][:], dtype=np.int64)
    if not allow_subset and set(map(int, event_ids)) != set(map(int, expected)):
        raise ValueError("Headline comparison plots require the complete validation population")


def _plot_spectrum(
    records: list[tuple[str, dict, dict[str, np.ndarray]]],
    output_dir: Path,
    estimator: str,
    event_table: Path,
) -> list[Path]:
    event_ids = np.asarray(records[0][2]["EVENT_ID"], dtype=np.int64)
    truth, perfect, perfect_support = _perfect_reco(event_table, event_ids)
    reconstructed = [
        np.asarray(arrays[f"pred__{estimator}__mttbar"], dtype=np.float64)
        for _, _, arrays in records
    ]
    if not np.isfinite(truth).all() or not np.isfinite(perfect).all() or any(
        not np.isfinite(values).all() for values in reconstructed
    ):
        raise ValueError("mttbar spectrum contains non-finite values")
    upper = max(float(np.max(truth)), float(np.max(perfect)), *(float(np.max(v)) for v in reconstructed))
    upper = max(1000.0, np.ceil(upper / 100.0) * 100.0)
    edges = np.linspace(0.0, upper, 101)
    counts = {"Parton truth": np.histogram(truth, bins=edges)[0]}
    perfect_label = f"Perfect reco ({perfect_support:,}/{len(event_ids):,} finite)"
    counts[perfect_label] = np.histogram(perfect, bins=edges)[0]
    for (label, _, _), values in zip(records, reconstructed, strict=True):
        counts[label] = np.histogram(values, bins=edges)[0]
    centers = 0.5 * (edges[:-1] + edges[1:])
    figure, (axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(9.0, 7.2), sharex=True,
        gridspec_kw={"height_ratios": (3.1, 1.0)}, constrained_layout=True,
    )
    axis.stairs(counts["Parton truth"], edges, color="black", linewidth=2.0, label="Parton truth")
    axis.stairs(counts[perfect_label], edges, color="#666666", linestyle="--", label=perfect_label)
    perfect_ratio = np.divide(
        counts[perfect_label], counts["Parton truth"],
        out=np.full(len(centers), np.nan, dtype=np.float64),
        where=counts["Parton truth"] > 0,
    )
    ratio_axis.plot(centers, perfect_ratio, color="#666666", linestyle="--", linewidth=1.3, label=perfect_label)
    for index, (label, _, _) in enumerate(records):
        axis.stairs(counts[label], edges, color=_color(label, index), linewidth=1.6, label=label)
        ratio = np.divide(
            counts[label], counts["Parton truth"],
            out=np.full(len(centers), np.nan, dtype=np.float64),
            where=counts["Parton truth"] > 0,
        )
        ratio_axis.plot(centers, ratio, color=_color(label, index), marker=".", markersize=3, label=label)
    ratio_axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ratio_axis.set_ylim(0.0, max(2.0, np.nanpercentile(
        np.concatenate([np.asarray(counts[label]) / np.maximum(counts["Parton truth"], 1) for label, _, _ in records]), 98
    ) * 1.15))
    axis.set_yscale("log")
    axis.set_ylabel("Events / bin")
    ratio_axis.set_ylabel("Reco / truth")
    ratio_axis.set_xlabel(r"$m_{t\bar{t}}$ [GeV]")
    axis.set_xlim(0.0, upper)
    axis.legend(frameon=False, ncol=2)
    ratio_axis.legend(frameon=False, ncol=2)
    axis.set_title(
        f"Validation $m_{{t\\bar{{t}}}}$ spectrum — {LABELS[estimator]}\n"
        f"Perfect-reco reference uses {perfect_support:,}/{len(event_ids):,} finite events"
    )
    ratio_axis.grid(axis="y", alpha=0.25)
    full_path = _save(figure, output_dir / "mttbar_spectrum_full")

    threshold_edges = np.arange(300.0, 555.0, 5.0)
    threshold_counts = {
        "Parton truth": np.histogram(truth, bins=threshold_edges)[0],
        perfect_label: np.histogram(perfect, bins=threshold_edges)[0],
    }
    for (label, _, _), values in zip(records, reconstructed, strict=True):
        threshold_counts[label] = np.histogram(values, bins=threshold_edges)[0]
    threshold_centers = 0.5 * (threshold_edges[:-1] + threshold_edges[1:])
    figure, (axis, ratio_axis) = plt.subplots(
        2, 1, figsize=(9.0, 6.6), sharex=True,
        gridspec_kw={"height_ratios": (3.1, 1.0)}, constrained_layout=True,
    )
    axis.stairs(threshold_counts["Parton truth"], threshold_edges, color="black", linewidth=2.0, label="Parton truth")
    axis.stairs(
        threshold_counts[perfect_label], threshold_edges, color="#666666",
        linestyle="--", label=perfect_label,
    )
    perfect_ratio = np.divide(
        threshold_counts[perfect_label], threshold_counts["Parton truth"],
        out=np.full(len(threshold_centers), np.nan),
        where=threshold_counts["Parton truth"] > 0,
    )
    ratio_axis.plot(
        threshold_centers, perfect_ratio, color="#666666", linestyle="--",
        linewidth=1.3, label=perfect_label,
    )
    for index, (label, _, _) in enumerate(records):
        axis.stairs(threshold_counts[label], threshold_edges, color=_color(label, index), linewidth=1.6, label=label)
        ratio = np.divide(
            threshold_counts[label], threshold_counts["Parton truth"],
            out=np.full(len(threshold_centers), np.nan),
            where=threshold_counts["Parton truth"] > 0,
        )
        ratio_axis.plot(threshold_centers, ratio, color=_color(label, index), marker=".", markersize=3, label=label)
    ratio_axis.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    ratio_axis.set_ylabel("Reco / truth")
    ratio_axis.set_xlabel(r"$m_{t\bar{t}}$ [GeV]")
    axis.set_ylabel("Events / 5 GeV")
    axis.set_xlim(300.0, 550.0)
    axis.set_title(
        f"Threshold spectrum, 300–550 GeV — {LABELS[estimator]}\n"
        f"Perfect-reco reference uses {perfect_support:,}/{len(event_ids):,} finite events"
    )
    axis.legend(frameon=False, ncol=2)
    ratio_axis.legend(frameon=False, ncol=2)
    ratio_axis.grid(axis="y", alpha=0.25)
    threshold_path = _save(figure, output_dir / "mttbar_spectrum_threshold")
    return [full_path, threshold_path]


def _residual_summary(residual: np.ndarray) -> str:
    q16, q84 = np.quantile(residual, [0.16, 0.84])
    return (
        f"bias {np.mean(residual):.1f} GeV\n"
        f"std {np.std(residual):.1f} GeV\n"
        f"$\\sigma_{{68}}$ {(q84 - q16) / 2.0:.1f} GeV\n"
        f"MAE {np.mean(np.abs(residual)):.1f} GeV\n"
        f"RMSE {np.sqrt(np.mean(residual * residual)):.1f} GeV"
    )


def _plot_mttbar_residuals(
    records: list[tuple[str, dict, dict[str, np.ndarray]]],
    output_dir: Path,
    estimator: str,
) -> list[Path]:
    residuals = {
        label: np.asarray(arrays[f"pred__{estimator}__mttbar"], dtype=np.float64)
        - np.asarray(arrays["truth__mttbar"], dtype=np.float64)
        for label, _, arrays in records
    }
    all_residuals = np.concatenate(tuple(residuals.values()))
    low = float(np.floor(np.min(all_residuals) / 10.0) * 10.0)
    high = float(np.ceil(np.max(all_residuals) / 10.0) * 10.0)
    if high <= low:
        high = low + 10.0
    paths = []
    for name, limits, logarithmic in (
        ("mttbar_residual_full", (low, high), True),
        ("mttbar_residual_central", (-200.0, 200.0), False),
    ):
        figure, axis = plt.subplots(figsize=(9.0, 5.7), constrained_layout=True)
        edges = np.linspace(limits[0], limits[1], 121)
        for index, (label, _, _) in enumerate(records):
            residual = residuals[label]
            axis.hist(
                residual, bins=edges, histtype="step", linewidth=1.6,
                color=_color(label, index), label=label,
            )
        if logarithmic:
            axis.set_yscale("log")
        axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.set_xlim(*limits)
        axis.set_xlabel(r"$m_{t\bar{t}}^{\mathrm{reco}}-m_{t\bar{t}}^{\mathrm{truth}}$ [GeV]")
        axis.set_ylabel("Events / bin")
        axis.set_title(f"$m_{{t\\bar{{t}}}}$ residual — {LABELS[estimator]}")
        axis.legend(frameon=False)
        if name.endswith("central"):
            stats = "\n\n".join(
                f"{label}:\n{_residual_summary(residuals[label])}"
                for label, _, _ in records
            )
            axis.text(
                0.99, 0.98, stats, transform=axis.transAxes,
                ha="right", va="top", fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
            )
        paths.append(_save(figure, output_dir / name))
    return paths


def _plot_mttbar_truth_reco(
    records: list[tuple[str, dict, dict[str, np.ndarray]]],
    output_dir: Path,
    estimator: str,
) -> Path:
    truth = np.asarray(records[0][2]["truth__mttbar"], dtype=np.float64)
    maximum = max(
        float(np.max(truth)),
        *(float(np.max(np.asarray(arrays[f"pred__{estimator}__mttbar"], dtype=np.float64))) for _, _, arrays in records),
    )
    maximum = max(500.0, np.ceil(maximum / 100.0) * 100.0)
    edges = np.linspace(0.0, maximum, 81)
    histograms = []
    response_lines = []
    for _, _, arrays in records:
        prediction = np.asarray(arrays[f"pred__{estimator}__mttbar"], dtype=np.float64)
        histogram, _, _ = np.histogram2d(truth, prediction, bins=(edges, edges))
        histograms.append(histogram)
        centers, median_response, counts = [], [], []
        indices = np.digitize(truth, edges) - 1
        indices[truth == edges[-1]] = len(edges) - 2
        for index in range(len(edges) - 1):
            selected = indices == index
            if selected.sum() < 20:
                continue
            centers.append(0.5 * (edges[index] + edges[index + 1]))
            median_response.append(float(np.median(prediction[selected])))
            counts.append(int(selected.sum()))
        response_lines.append((np.asarray(centers), np.asarray(median_response), np.asarray(counts)))
    vmax = max(2.0, max(float(np.max(hist)) for hist in histograms))
    figure, axes = plt.subplots(
        1, len(records), figsize=(5.6 * len(records), 5.4),
        constrained_layout=True, squeeze=False,
    )
    for index, ((label, _, _), axis) in enumerate(zip(records, axes[0], strict=True)):
        mesh = axis.pcolormesh(
            edges, edges, histograms[index].T,
            norm=LogNorm(vmin=1.0, vmax=vmax), cmap="viridis", shading="auto",
        )
        axis.plot([0.0, maximum], [0.0, maximum], color="white", linestyle="--", linewidth=1.2, label="$y=x$")
        x, y, _ = response_lines[index]
        axis.plot(x, y, color="#CC3311", linewidth=1.8, label="Binned median response")
        axis.set_xlim(0.0, maximum)
        axis.set_ylim(0.0, maximum)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(r"Parton truth $m_{t\bar{t}}$ [GeV]")
        if index == 0:
            axis.set_ylabel(r"Reconstructed $m_{t\bar{t}}$ [GeV]")
        axis.set_title(label)
        axis.legend(frameon=False, loc="upper left")
    figure.colorbar(mesh, ax=list(axes[0]), shrink=0.8, label="Events / bin")
    figure.suptitle(f"Truth versus reconstructed parent mass — {LABELS[estimator]}")
    return _save(figure, output_dir / "mttbar_truth_vs_reco")


def _truth_bins(maximum: float, count: int = 30) -> np.ndarray:
    return np.linspace(0.0, maximum, count + 1)


def _conditional_metrics(
    truth: np.ndarray, prediction: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    centers = 0.5 * (edges[:-1] + edges[1:])
    residual = prediction - truth
    result = {name: np.full(len(centers), np.nan) for name in ("bias", "std", "sigma68", "rmse", "fractional_sigma68", "fractional_rmse")}
    counts = np.zeros(len(centers), dtype=np.int64)
    indices = np.digitize(truth, edges) - 1
    indices[truth == edges[-1]] = len(centers) - 1
    for index in range(len(centers)):
        selected = indices == index
        values = residual[selected]
        if not selected.any():
            continue
        counts[index] = int(selected.sum())
        q16, q84 = np.quantile(values, [0.16, 0.84])
        result["bias"][index] = float(np.mean(values))
        result["std"][index] = float(np.std(values))
        result["sigma68"][index] = float((q84 - q16) / 2.0)
        result["rmse"][index] = float(np.sqrt(np.mean(values * values)))
        scale = float(np.mean(truth[selected]))
        if scale > 0.0:
            result["fractional_sigma68"][index] = result["sigma68"][index] / scale
            result["fractional_rmse"][index] = result["rmse"][index] / scale
    return centers, result, counts


def _plot_differential_mttbar(
    records: list[tuple[str, dict, dict[str, np.ndarray]]],
    output_dir: Path,
    estimator: str,
) -> list[Path]:
    truth = np.asarray(records[0][2]["truth__mttbar"], dtype=np.float64)
    maximum = float(max(
        np.max(truth),
        *(np.max(np.asarray(arrays[f"pred__{estimator}__mttbar"], dtype=np.float64)) for _, _, arrays in records),
    ))
    edges = _truth_bins(max(500.0, np.ceil(maximum / 100.0) * 100.0), count=30)
    products = {
        label: _conditional_metrics(
            truth,
            np.asarray(arrays[f"pred__{estimator}__mttbar"], dtype=np.float64),
            edges,
        )
        for label, _, arrays in records
    }
    definitions = (
        ("bias", r"Conditional bias $E[\Delta m_{t\bar{t}}|m_{t\bar{t}}^{truth}]$ [GeV]", "mttbar_bias_vs_truth"),
        ("std", r"Conditional residual std [GeV]", "mttbar_std_vs_truth"),
        ("sigma68", r"Conditional $\sigma_{68}$ [GeV]", "mttbar_sigma68_vs_truth"),
        ("rmse", r"Conditional RMSE [GeV]", "mttbar_rmse_vs_truth"),
        ("fractional_sigma68", r"Conditional $\sigma_{68}/m_{t\bar{t}}^{truth}$", "mttbar_fractional_sigma68_vs_truth"),
        ("fractional_rmse", r"Conditional RMSE$/m_{t\bar{t}}^{truth}$", "mttbar_fractional_rmse_vs_truth"),
    )
    paths = []
    for key, ylabel, filename in definitions:
        figure, axis = plt.subplots(figsize=(9.0, 5.7), constrained_layout=True)
        for index, (label, _, _) in enumerate(records):
            centers, metrics, counts = products[label]
            good = np.isfinite(metrics[key]) & (counts > 0)
            axis.plot(centers[good], metrics[key][good], marker="o", markersize=3.2, linewidth=1.5, color=_color(label, index), label=label)
        if key == "bias":
            axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.set_xlabel(r"Parton truth $m_{t\bar{t}}$ [GeV]")
        axis.set_ylabel(ylabel)
        axis.set_title(f"Differential parent-mass response — {LABELS[estimator]}")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        paths.append(_save(figure, output_dir / filename))
    return paths


def _mass_category(values: np.ndarray) -> np.ndarray:
    edges = (345.0, 400.0, 500.0, 700.0, 1000.0)
    values = np.asarray(values, dtype=np.float64)
    category = np.empty(len(values), dtype=np.int64)
    category[values < edges[0]] = 0
    category[(values >= 345.0) & (values < 400.0)] = 1
    category[(values >= 400.0) & (values < 500.0)] = 2
    category[(values >= 500.0) & (values < 700.0)] = 3
    category[(values >= 700.0) & (values <= 1000.0)] = 4
    category[values > 1000.0] = 5
    return category


def _plot_migration(
    records: list[tuple[str, dict, dict[str, np.ndarray]]],
    output_dir: Path,
    estimator: str,
) -> Path:
    labels = [label for label, _, _ in records]
    figure, axes = plt.subplots(1, len(records), figsize=(5.4 * len(records), 5.8), constrained_layout=True, squeeze=False)
    axes = axes[0]
    matrices = []
    counts = []
    for _, _, arrays in records:
        truth = np.asarray(arrays["truth__mttbar"], dtype=np.float64)
        prediction = np.asarray(arrays[f"pred__{estimator}__mttbar"], dtype=np.float64)
        true_bin, reco_bin = _mass_category(truth), _mass_category(prediction)
        matrix = np.zeros((6, 6), dtype=np.int64)
        np.add.at(matrix, (true_bin, reco_bin), 1)
        row_counts = matrix.sum(axis=1)
        percentage = np.divide(
            matrix * 100.0, row_counts[:, None],
            out=np.zeros_like(matrix, dtype=np.float64), where=row_counts[:, None] > 0,
        )
        matrices.append(percentage)
        counts.append(row_counts)
    for index, ((label, _, _), axis) in enumerate(zip(records, axes, strict=True)):
        image = axis.imshow(matrices[index], origin="upper", vmin=0.0, vmax=100.0, cmap="Blues", aspect="equal")
        axis.set_xticks(range(6), MASS_REGION_LABELS, rotation=30, ha="right")
        axis.set_yticks(range(6), [f"{region}\nN={int(n):,}" for region, n in zip(MASS_REGION_LABELS, counts[index], strict=True)])
        axis.set_xlabel("Reconstructed $m_{t\\bar{t}}$ region")
        if index == 0:
            axis.set_ylabel("Truth $m_{t\\bar{t}}$ region (row-normalised)")
        axis.set_title(label)
        for row in range(6):
            for col in range(6):
                axis.text(col, row, f"{matrices[index][row, col]:.1f}%", ha="center", va="center", fontsize=8, color="black")
        for diagonal in range(6):
            axis.add_patch(plt.Rectangle((diagonal - 0.5, diagonal - 0.5), 1, 1, fill=False, edgecolor="#CC3311", linewidth=1.8))
    colorbar = figure.colorbar(image, ax=list(axes), shrink=0.8, pad=0.02)
    colorbar.set_label("Truth-row percentage")
    figure.suptitle(f"$m_{{t\\bar{{t}}}}$ migration — {LABELS[estimator]}")
    return _save(figure, output_dir / "mttbar_migration_matrix")


def _plot_phi(
    records: list[tuple[str, dict, dict[str, np.ndarray]]],
    output_dir: Path,
    estimator: str,
) -> list[Path]:
    paths = []
    for observable, role in (("top_phi", "top"), ("antitop_phi", "antitop")):
        bins = np.linspace(-np.pi, np.pi, 49)
        histograms = []
        for _, _, arrays in records:
            truth = np.asarray(arrays[f"truth__{observable}"], dtype=np.float64)
            prediction = np.asarray(arrays[f"pred__{estimator}__{observable}"], dtype=np.float64)
            hist, _, _ = np.histogram2d(truth, prediction, bins=(bins, bins))
            histograms.append(hist)
        vmax = max(2.0, max(float(np.max(hist)) for hist in histograms))
        figure, axes = plt.subplots(1, len(records), figsize=(5.4 * len(records), 5.2), constrained_layout=True, squeeze=False)
        for index, ((label, _, arrays), axis) in enumerate(zip(records, axes[0], strict=True)):
            truth = np.asarray(arrays[f"truth__{observable}"], dtype=np.float64)
            prediction = np.asarray(arrays[f"pred__{estimator}__{observable}"], dtype=np.float64)
            mesh = axis.pcolormesh(
                bins, bins, histograms[index].T,
                norm=LogNorm(vmin=1.0, vmax=vmax), cmap="viridis", shading="auto",
            )
            axis.plot([-np.pi, np.pi], [-np.pi, np.pi], color="white", linestyle="--", linewidth=1.2)
            axis.set_xlim(-np.pi, np.pi)
            axis.set_ylim(-np.pi, np.pi)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(f"Truth $\\phi_{{{role}}}$ [rad]")
            if index == 0:
                axis.set_ylabel(f"Reco $\\phi_{{{role}}}$ [rad]")
            axis.set_title(label)
        figure.colorbar(mesh, ax=list(axes[0]), shrink=0.8, label="Events / bin")
        figure.suptitle(f"{role.capitalize()} azimuth: truth versus reconstruction — {LABELS[estimator]}")
        paths.append(_save(figure, output_dir / f"{role}_phi_truth_vs_reco"))

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    for axis, (observable, role) in zip(axes, (("top_phi", "top"), ("antitop_phi", "antitop")), strict=True):
        for index, (label, _, arrays) in enumerate(records):
            truth = np.asarray(arrays[f"truth__{observable}"], dtype=np.float64)
            prediction = np.asarray(arrays[f"pred__{estimator}__{observable}"], dtype=np.float64)
            residual = _wrap(prediction - truth)
            axis.hist(residual, bins=np.linspace(-np.pi, np.pi, 61), histtype="step", linewidth=1.5, color=_color(label, index), label=label)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        axis.set_xlim(-np.pi, np.pi)
        axis.set_xlabel(r"Wrapped $\Delta\phi$ [rad]")
        axis.set_ylabel("Events / bin")
        axis.set_title(role.capitalize())
        axis.legend(frameon=False)
    figure.suptitle(f"Wrapped parent azimuth residuals — {LABELS[estimator]}")
    paths.append(_save(figure, output_dir / "parent_phi_wrapped_residuals"))
    return paths


def _angle_record(record: dict, angle: float) -> dict:
    key = min(record["per_angle"], key=lambda value: abs(float(value) - angle))
    return record["per_angle"][key]


def _plot_rotation_diagnostics(
    rotations: list[tuple[str, dict]], output_dir: Path
) -> list[Path]:
    if not rotations:
        return []
    reference = rotations[0][1]
    for label, record in rotations:
        for key in ("source_bank_sha256", "sample_event_id_sha256", "angles_rad", "steps"):
            if record.get(key) != reference.get(key):
                raise ValueError(f"Rotation diagnostic {label} is not paired: {key} differs")
        if record.get("sealed_test_used") is not False:
            raise ValueError(f"Rotation diagnostic {label} does not establish sealed_test_used=false")
    angles = np.asarray(reference["angles_rad"], dtype=np.float64)
    roles = (("top", "top_phi_shift_rad", "top_phi_residual_rad"), ("antitop", "antitop_phi_shift_rad", "antitop_phi_residual_rad"))
    paths = []

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    for axis, (role, sensitivity_key, _) in zip(axes, roles, strict=True):
        axis.plot(angles, angles, color="black", linestyle="--", linewidth=1.2, label="Expected response $\\Delta\\phi=\\alpha$")
        for index, (label, record) in enumerate(rotations):
            stats = [
                _angle_record(record, angle)["input_rotation_same_source"][sensitivity_key]
                for angle in angles
            ]
            means = np.asarray([item["mean"] for item in stats])
            low = means - np.asarray([item["q05"] for item in stats])
            high = np.asarray([item["q95"] for item in stats]) - means
            axis.errorbar(
                angles, means, yerr=np.vstack((np.maximum(low, 0.0), np.maximum(high, 0.0))),
                color=_color(label, index), marker="o", capsize=2.5, linewidth=1.5, label=label,
            )
        axis.set_xlabel("Applied measured-event rotation $\\alpha$ [rad]")
        axis.set_ylabel(f"Observed {role} $\\Delta\\phi$ [rad]")
        axis.set_title(role.capitalize())
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Fixed-source orientation sensitivity (central 90% event range)")
    paths.append(_save(figure, output_dir / "fixed_source_orientation_response"))

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)
    for axis, (role, _, covariance_key) in zip(axes, roles, strict=True):
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        for index, (label, record) in enumerate(rotations):
            stats = [
                _angle_record(record, angle)["formal_co_rotated_source_covariance"][covariance_key]
                for angle in angles
            ]
            means = np.asarray([item["mean"] for item in stats])
            low = means - np.asarray([item["q05"] for item in stats])
            high = np.asarray([item["q95"] for item in stats]) - means
            axis.errorbar(
                angles, means, yerr=np.vstack((np.maximum(low, 0.0), np.maximum(high, 0.0))),
                color=_color(label, index), marker="o", capsize=2.5, linewidth=1.5, label=label,
            )
        axis.set_xlabel("Applied co-rotation $\\alpha$ [rad]")
        axis.set_ylabel(f"Formal {role} azimuth residual [rad]")
        axis.set_title(role.capitalize())
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Co-rotated-source covariance residual (central 90% event range)")
    paths.append(_save(figure, output_dir / "co_rotated_source_covariance_residual"))

    if all("geometric_context_rotation" in record for _, record in rotations):
        figure, axis = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
        for index, (label, record) in enumerate(rotations):
            residuals = [
                record["geometric_context_rotation"][min(
                    record["geometric_context_rotation"],
                    key=lambda value: abs(float(value) - angle),
                )]["rms"]
                for angle in angles
            ]
            axis.plot(angles, residuals, color=_color(label, index), marker="o", label=label)
        axis.set_xlabel("Applied measured-event rotation $\\alpha$ [rad]")
        axis.set_ylabel(r"$H_{geom}$ rotation residual RMS [multivector units]")
        axis.set_title("L-GATr geometric-token covariance")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        paths.append(_save(figure, output_dir / "lgatr_geometric_token_covariance"))
    return paths


def _plot_convergence(summary_path: Path, output_dir: Path) -> list[Path]:
    summary = json.loads(summary_path.expanduser().resolve().read_text(encoding="utf-8"))
    if summary.get("schema") != "realnu_parent_solver_scan_analysis_v1":
        raise ValueError("Unsupported PCFM_Q solver scan analysis summary")
    if summary.get("sealed_test_used") is not False:
        raise ValueError("Solver scan analysis does not establish sealed_test_used=false")
    cases = summary["cases"]
    sample_counts = sorted(
        {int(sample_count) for case in cases.values() for sample_count in case["sample_counts"]}
    )
    estimators = ("observable_mean", "observable_q50", "joint_medoid")
    step_counts = sorted(int(case["steps"]) for case in cases.values())
    paths = []
    for metric, ylabel, filename in (
        ("rmse", "Inclusive $m_{t\\bar{t}}$ RMSE [GeV]", "mttbar_rmse_vs_samples"),
        ("bias", "Inclusive $m_{t\\bar{t}}$ bias [GeV]", "mttbar_bias_vs_samples"),
        ("std", "Inclusive $m_{t\\bar{t}}$ residual std [GeV]", "mttbar_std_vs_samples"),
        ("sigma68", "Inclusive $m_{t\\bar{t}}$ $\\sigma_{68}$ [GeV]", "mttbar_sigma68_vs_samples"),
        ("mae", "Inclusive $m_{t\\bar{t}}$ MAE [GeV]", "mttbar_mae_vs_samples"),
    ):
        figure, axes = plt.subplots(1, len(step_counts), figsize=(5.0 * len(step_counts), 4.7), constrained_layout=True, squeeze=False)
        for step_index, steps in enumerate(step_counts):
            axis = axes[0, step_index]
            case_name = f"heun_steps_{steps}"
            for estimator in estimators:
                values = []
                valid_m = []
                for sample_count in sample_counts:
                    entry = cases[case_name]["sample_counts"].get(str(sample_count))
                    if entry is None:
                        continue
                    stat = entry["metrics"][estimator]["mttbar"]["inclusive"][metric]
                    if stat is not None:
                        values.append(stat)
                        valid_m.append(sample_count)
                axis.plot(valid_m, values, marker="o", linewidth=1.5, label=LABELS[estimator])
            axis.set_xscale("log", base=2)
            axis.set_xticks(sample_counts, [str(value) for value in sample_counts])
            axis.set_xlabel("Nested source samples $M$")
            if step_index == 0:
                axis.set_ylabel(ylabel)
            axis.set_title(f"Heun {steps} steps ({2 * steps} NFE / sample)")
            axis.grid(alpha=0.25)
        axes[0, 0].legend(frameon=False, fontsize=8)
        figure.suptitle("PCFM_Q posterior-estimator convergence")
        paths.append(_save(figure, output_dir / filename))

    figure, axis = plt.subplots(figsize=(7.8, 5.0), constrained_layout=True)
    for estimator in estimators:
        x, y = [], []
        for steps in step_counts:
            case_name = f"heun_steps_{steps}"
            for sample_count in sample_counts:
                entry = cases[case_name]["sample_counts"].get(str(sample_count))
                if entry is None:
                    continue
                value = entry["metrics"][estimator]["mttbar"]["inclusive"]["rmse"]
                if value is not None:
                    x.append(entry["aggregate_nfe_per_event"])
                    y.append(value)
        axis.scatter(x, y, marker="o", label=LABELS[estimator])
    axis.set_xscale("log", base=2)
    axis.set_xlabel("ODE NFE × posterior samples per event")
    axis.set_ylabel("Inclusive $m_{t\\bar{t}}$ RMSE [GeV]")
    axis.set_title("Accuracy versus inference compute (NFE work units)")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    paths.append(_save(figure, output_dir / "mttbar_rmse_vs_nfe_work"))
    convergence = summary.get("solver_convergence_vs_heun32", {})
    if convergence:
        for observable in ("mttbar", "top_phi", "antitop_phi"):
            figure, axes = plt.subplots(
                1, 3, figsize=(15.0, 4.8), constrained_layout=True, squeeze=False
            )
            for axis, sample_count in zip(axes[0], ("1", "8", "32"), strict=True):
                sample_record = convergence.get(sample_count, {})
                for estimator in estimators:
                    x, y = [], []
                    for case_name, case_record in sample_record.items():
                        steps = int(case_name.rsplit("_", 1)[1])
                        value = case_record[estimator]["observables"][observable][
                            "solver_delta_over_reconstruction_rmse"
                        ]
                        x.append(steps)
                        y.append(value)
                    order = np.argsort(x)
                    axis.plot(
                        np.asarray(x)[order], np.asarray(y)[order], marker="o",
                        linewidth=1.5, label=LABELS[estimator],
                    )
                axis.set_xticks((4, 8, 16), ("4", "8", "16"))
                axis.set_xlabel("Heun step count; reference is Heun32")
                axis.set_title(f"M={sample_count}")
                axis.grid(alpha=0.25)
                if sample_count == "1":
                    axis.legend(frameon=False, fontsize=8)
            axes[0, 0].set_ylabel(
                "RMS prediction change / Heun32 reconstruction RMSE"
            )
            figure.suptitle(
                f"PCFM_Q solver discretisation relative to reconstruction uncertainty: {observable}"
            )
            paths.append(
                _save(
                    figure,
                    output_dir / f"{observable}_solver_delta_vs_reconstruction_uncertainty",
                )
            )
    return paths


def _plot_runtime_profile(summary_path: Path, output_dir: Path) -> Path:
    profile = json.loads(summary_path.expanduser().resolve().read_text(encoding="utf-8"))
    if profile.get("schema") != "realnu_parent_runtime_profile_v1":
        raise ValueError("Unsupported parent runtime-profile summary")
    if profile.get("truth_free_input") is not True or profile.get("sealed_test_used") is not False:
        raise ValueError("Runtime profile must establish truth-free validation use")
    counts = sorted({int(case["samples"]) for case in profile["cases"].values()})
    steps = sorted({int(case["steps"]) for case in profile["cases"].values()})
    figure, axis = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    for step_count in steps:
        seconds_per_event = [
            profile["cases"][f"heun_steps_{step_count}_samples_{sample_count}"][
                "median_seconds_per_event"
            ]
            for sample_count in counts
        ]
        axis.plot(
            counts, seconds_per_event, marker="o", linewidth=1.6,
            label=f"Heun {step_count} steps ({2 * step_count} NFE/sample)",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks(counts, [str(value) for value in counts])
    axis.set_xlabel("Posterior source samples $M$")
    axis.set_ylabel("Measured PCFM_Q CPU sampling time [s / event]")
    axis.set_title(
        f"CPU sampling cost versus posterior size ({profile['events']:,} validation events)"
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    return _save(figure, output_dir / "runtime_vs_samples_cpu")


def build(records: list[tuple[str, dict]], output_dir: Path) -> dict:
    if not records:
        raise ValueError("At least one parent reduction summary is required")
    first = records[0][1]
    for label, record in records[1:]:
        for key in ("source_bank_sha256", "event_id_sha256", "case", "solver", "steps", "samples"):
            if record.get(key) != first.get(key):
                raise ValueError(f"{label} is not paired with the first model: {key} differs")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        _write_rmse_plot(records, output_dir),
        _write_njet_plot(records, output_dir),
        _write_coverage_plot(records, output_dir),
    ]
    report = _write_report(records, output_dir)
    return {
        "status": "completed",
        "models": [label for label, _ in records],
        "figures": [str(path.resolve()) for path in plots],
        "report": str(report.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, action="append", required=True,
        help="A model's reducer JSON; repeat once per model in the desired display order.",
    )
    parser.add_argument(
        "--label", action="append", default=None,
        help="Optional label per --summary; repeat in the same order.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-table", type=Path, default=DEFAULT_EVENT_TABLE)
    parser.add_argument("--estimator", choices=ESTIMATORS, default="joint_medoid")
    parser.add_argument("--allow-subset", action="store_true", help="Smoke plots only; headlines require full validation.")
    parser.add_argument(
        "--rotation-summary", action="append", default=[], metavar="MODEL=JSON",
        help="Formal rotation diagnostic; repeat once per model for paired rotation figures.",
    )
    parser.add_argument("--convergence-summary", type=Path)
    parser.add_argument("--runtime-summary", type=Path)
    args = parser.parse_args()
    if args.label is not None and len(args.label) != len(args.summary):
        raise ValueError("Supply one --label per --summary")
    labels = args.label or [path.stem for path in args.summary]
    records = [
        (label, *(_load(path)))
        for label, path in zip(labels, args.summary, strict=True)
    ]
    records = _align_records(records)
    event_ids = np.asarray(records[0][2]["EVENT_ID"], dtype=np.int64)
    _validate_population(args.event_table, event_ids, allow_subset=args.allow_subset)
    if args.estimator not in ESTIMATORS:
        raise ValueError(f"Unsupported plot estimator: {args.estimator}")
    output_dir = args.output_dir.expanduser().resolve()
    headline_dir = output_dir / "headline"
    geometry_dir = output_dir / "geometry"
    headline_dir.mkdir(parents=True, exist_ok=True)
    geometry_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    figures.extend(_plot_spectrum(records, headline_dir, args.estimator, args.event_table))
    figures.extend(_plot_mttbar_residuals(records, headline_dir, args.estimator))
    figures.append(_plot_mttbar_truth_reco(records, headline_dir, args.estimator))
    figures.extend(_plot_differential_mttbar(records, headline_dir, args.estimator))
    figures.append(_plot_migration(records, headline_dir, args.estimator))
    figures.extend(_plot_phi(records, geometry_dir, args.estimator))
    rotation_records = []
    for item in args.rotation_summary:
        if "=" not in item:
            raise ValueError("Rotation summaries must use MODEL=JSON syntax")
        label, filename = item.split("=", 1)
        rotation = json.loads(Path(filename).expanduser().resolve().read_text(encoding="utf-8"))
        rotation_records.append((label, rotation))
    figures.extend(_plot_rotation_diagnostics(rotation_records, geometry_dir))
    convergence_dir = output_dir / "convergence"
    if args.convergence_summary is not None:
        figures.extend(_plot_convergence(args.convergence_summary, convergence_dir))
    if args.runtime_summary is not None:
        figures.append(_plot_runtime_profile(args.runtime_summary, convergence_dir))
    if len(records) <= 3:
        # Keep compact smoke diagnostics alongside the publication-style figures.
        figures.extend(
            [
                _write_rmse_plot([(label, meta) for label, meta, _ in records], output_dir),
                _write_njet_plot([(label, meta) for label, meta, _ in records], output_dir),
                _write_coverage_plot([(label, meta) for label, meta, _ in records], output_dir),
            ]
        )
        report = _write_report([(label, meta) for label, meta, _ in records], output_dir)
    else:
        report = _write_report([(label, meta) for label, meta, _ in records], output_dir)
    print(
        json.dumps(
            {
                "status": "completed",
                "models": [label for label, _, _ in records],
                "estimator": args.estimator,
                "event_count": int(len(event_ids)),
                "allow_subset": args.allow_subset,
                "figures": [str(path.resolve()) for path in figures],
                "rotation_models": [label for label, _ in rotation_records],
                "convergence_summary": (
                    str(args.convergence_summary.expanduser().resolve())
                    if args.convergence_summary is not None else None
                ),
                "report": str(report.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
