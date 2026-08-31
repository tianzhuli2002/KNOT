"""Clear, support-aware plots for maintained reconstruction prediction products."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from joint_event_reconstruction.assignment_semantics import read_hdf5_support
from joint_event_reconstruction.evaluation import (
    COVERAGE_CURVE_LEVELS,
    _finite_sample_rank,
    _weighted_quantile_rows,
    truth_oracle_support,
)
from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions
from joint_event_reconstruction.prediction_schema import (
    DISCRETE_PREDICTION_SCHEMA,
    JOINT_PREDICTION_SCHEMA,
    PREDICTION_SCHEMA,
)


COMPONENTS = ("px", "py", "pz")
CALIBRATION_LEVELS = np.asarray(COVERAGE_CURVE_LEVELS)
FIGURE_SIZE = (7.0, 5.0)
PLOT_STYLE = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 120,
    "savefig.dpi": 220,
}
SUPPORT_LABELS = {
    "complete_non_null": "complete non-NULL",
    "complete_null_including": "complete, NULL-including",
}


def _figure():
    with matplotlib.rc_context(PLOT_STYLE):
        return plt.subplots(figsize=FIGURE_SIZE)


def _save(fig, base: Path, formats: tuple[str, ...]) -> list[Path]:
    fig.tight_layout()
    outputs = []
    for extension in formats:
        path = base.with_suffix(f".{extension}")
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 220
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _finish(ax, *, legend: bool = False) -> None:
    ax.tick_params(direction="out")
    if legend:
        ax.legend(frameon=False)


def _stats_box(ax, lines: list[str]) -> None:
    ax.text(
        0.98,
        0.96,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )


def _support_label(name: str) -> str:
    return SUPPORT_LABELS.get(name, name.replace("_", " "))


def _support_note(ax, source: h5py.File) -> None:
    support = read_hdf5_support(source)
    ax.text(
        0.02,
        0.96,
        f"support: {_support_label(support['name'])}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )


def _binomial_error(value: float, count: int) -> float:
    return float(np.sqrt(value * (1.0 - value) / count)) if count else float("nan")


def _mode(states: np.ndarray) -> np.ndarray:
    result = np.empty(len(states), dtype=states.dtype)
    for event, row in enumerate(states):
        values, counts = np.unique(row, return_counts=True)
        result[event] = values[np.argmax(counts)]
    return result


def _factorized_posterior(source: h5py.File):
    samples = source["INVISIBLE/samples"][:]
    conditioning = str(
        source.attrs.get("invisible_assignment_conditioning", "unknown")
    )
    if conditioning != "hypothesis":
        selected = samples[:, 0]
        weights = np.full(
            selected.shape[:2], 1.0 / selected.shape[1], dtype=np.float64
        )
        return selected, weights, conditioning

    probabilities = source["VISIBLE/constrained_probabilities"][:].astype(np.float64)
    retained = probabilities.sum(axis=1, keepdims=True)
    if (retained <= 0).any():
        raise ValueError("Top-K assignment probability mass must be positive")
    probabilities = probabilities / retained
    batch, top_k, posterior = samples.shape[:3]
    flat = samples.reshape(batch, top_k * posterior, *samples.shape[3:])
    weights = np.repeat(
        probabilities[:, :, None] / posterior, posterior, axis=2
    ).reshape(batch, top_k * posterior)
    return flat, weights, conditioning


def _joint_posterior(source: h5py.File):
    samples = source["HYPOTHESES/invisible"][:]
    weights = np.full(
        samples.shape[:2], 1.0 / samples.shape[1], dtype=np.float64
    )
    return samples, weights, "joint_flow"


def _posterior_mean(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(samples * weights[:, :, None, None], axis=1)


def _coverage(
    values: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    result = []
    for level in CALIBRATION_LEVELS:
        low = _weighted_quantile_rows(values, weights, (1 - level) / 2)
        high = _weighted_quantile_rows(values, weights, 1 - (1 - level) / 2)
        result.append(((target >= low) & (target <= high)).mean())
    return np.asarray(result)


def _pit(values: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _finite_sample_rank(values, target, weights)


def _mass(vectors: np.ndarray) -> np.ndarray:
    mass2 = vectors[..., 0] ** 2 - np.sum(vectors[..., 1:] ** 2, axis=-1)
    return np.sqrt(np.maximum(mass2, 0.0))


def _residual_plot(
    values: np.ndarray,
    *,
    xlabel: str,
    output: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return []
    fig, ax = _figure()
    ax.hist(values, bins=50, histtype="step", linewidth=1.5)
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Events")
    central = np.quantile(values, 0.84) - np.quantile(values, 0.16)
    _stats_box(
        ax,
        [
            f"N = {len(values)}",
            f"bias = {values.mean():.3g}",
            f"RMSE = {np.sqrt(np.mean(values**2)):.3g}",
            f"68% width = {central:.3g}",
        ],
    )
    _finish(ax)
    return _save(fig, output, formats)


def _assignment_vs_njet(
    njet: np.ndarray,
    correct: np.ndarray,
    *,
    ylabel: str,
    source: h5py.File,
    output: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    jet_counts = np.unique(njet)
    values = []
    errors = []
    events = []
    for jet_count in jet_counts:
        mask = njet == jet_count
        count = int(mask.sum())
        value = float(correct[mask].mean())
        events.append(count)
        values.append(value)
        errors.append(_binomial_error(value, count))
    fig, ax = _figure()
    ax.errorbar(
        jet_counts,
        values,
        yerr=errors,
        marker="o",
        linestyle="-",
        capsize=3,
    )
    ax.set_xlabel("Jet multiplicity")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(jet_counts)
    _support_note(ax, source)
    _stats_box(ax, [f"N = {sum(events)}", "error bars: binomial SE"])
    _finish(ax)
    return _save(fig, output, formats)


def _assignment_plots(
    source: h5py.File,
    schema: str,
    output: Path,
    formats: tuple[str, ...],
):
    written = []
    njet = source["OBSERVED/njet"][:]
    if schema == PREDICTION_SCHEMA:
        correct = source["TRUTH/topk_assignment"][:, 0].astype(bool)
        written += _assignment_vs_njet(
            njet,
            correct,
            ylabel="Top-1 complete-assignment efficiency",
            source=source,
            output=output / "assignment_vs_njet",
            formats=formats,
        )

        probabilities = source["VISIBLE/constrained_probabilities"][:]
        retained = probabilities.sum(axis=1)
        fig, ax = _figure()
        ax.hist(retained, bins=np.linspace(0.0, 1.0, 41), histtype="step", linewidth=1.5)
        ax.axvline(retained.mean(), linestyle="--", linewidth=1.0)
        ax.set_xlabel("Retained top-K assignment probability mass")
        ax.set_ylabel("Events")
        ax.set_xlim(0.0, 1.02)
        _support_note(ax, source)
        _stats_box(
            ax,
            [
                f"N = {len(retained)}",
                f"mean = {retained.mean():.3f}",
                f"median = {np.median(retained):.3f}",
                f"min = {retained.min():.3f}",
            ],
        )
        _finish(ax)
        written += _save(fig, output / "assignment_retained_mass", formats)

        node_group = source["VISIBLE/nodes"]
        fig, ax = _figure()
        plotted = 0
        for name in node_group:
            values = node_group[name]["truth_probability"][:]
            reconstructable = node_group[name]["reconstructable"][:].astype(bool)
            if reconstructable.any():
                ax.hist(
                    values[reconstructable],
                    bins=np.linspace(0.0, 1.0, 41),
                    histtype="step",
                    linewidth=1.4,
                    label=name,
                )
                plotted += 1
        ax.set_xlabel("Truth-candidate probability")
        ax.set_ylabel("Events")
        ax.set_xlim(0.0, 1.0)
        _support_note(ax, source)
        _finish(ax, legend=plotted > 0)
        written += _save(fig, output / "assignment_truth_probability", formats)
        return written

    states = source["HYPOTHESES/assignment_state"][:]
    truth_state = source["TRUTH/assignment_state"][:]
    mode = _mode(states)
    correct = mode == truth_state
    written += _assignment_vs_njet(
        njet,
        correct,
        ylabel="Empirical-mode assignment accuracy",
        source=source,
        output=output / "assignment_vs_njet",
        formats=formats,
    )

    truth_frequency = (states == truth_state[:, None]).mean(axis=1)
    fig, ax = _figure()
    ax.hist(
        truth_frequency,
        bins=np.linspace(0.0, 1.0, 31),
        histtype="step",
        linewidth=1.5,
    )
    ax.axvline(truth_frequency.mean(), linestyle="--", linewidth=1.0)
    ax.set_xlabel("Truth-assignment posterior sample frequency")
    ax.set_ylabel("Events")
    ax.set_xlim(0.0, 1.0)
    _support_note(ax, source)
    _stats_box(
        ax,
        [
            f"N = {len(truth_frequency)}",
            f"samples / event = {states.shape[1]}",
            f"mean = {truth_frequency.mean():.3f}",
        ],
    )
    _finish(ax)
    written += _save(fig, output / "assignment_truth_frequency", formats)

    unique = np.asarray([len(np.unique(row)) for row in states])
    fig, ax = _figure()
    bins = np.arange(unique.max() + 2) - 0.5
    ax.hist(unique, bins=bins, histtype="step", linewidth=1.5)
    ax.axvline(unique.mean(), linestyle="--", linewidth=1.0)
    ax.set_xlabel("Unique sampled assignment states per event")
    ax.set_ylabel("Events")
    _support_note(ax, source)
    _stats_box(ax, [f"N = {len(unique)}", f"mean = {unique.mean():.2f}"])
    _finish(ax)
    written += _save(fig, output / "assignment_unique_states", formats)
    return written


def _invisible_plots(
    source: h5py.File,
    schema: str,
    output: Path,
    formats: tuple[str, ...],
):
    if schema == DISCRETE_PREDICTION_SCHEMA:
        return []
    if schema == PREDICTION_SCHEMA:
        if "INVISIBLE/samples" not in source:
            return []
        samples, weights, conditioning = _factorized_posterior(source)
    elif schema == JOINT_PREDICTION_SCHEMA:
        samples, weights, _ = _joint_posterior(source)
    else:
        return []

    truth = source["TRUTH/NEUTRINO"][:]
    if schema == PREDICTION_SCHEMA and conditioning == "truth":
        support = truth_oracle_support(source)
        if not support.any():
            return []
        samples = samples[support]
        weights = weights[support]
        truth = truth[support]
    roles = json.loads(source.attrs["invisible_roles"])
    mean = _posterior_mean(samples, weights)
    written = []
    for role_index, role in enumerate(roles):
        safe_role = role.replace("/", "_").replace(".", "_")
        for component_index, component in enumerate(COMPONENTS):
            residual = (
                mean[:, role_index, component_index]
                - truth[:, role_index, component_index]
            )
            written += _residual_plot(
                residual,
                xlabel=f"{role} {component} posterior-mean residual [GeV]",
                output=output / f"neutrino_{safe_role}_{component}_residual",
                formats=formats,
            )

            values = samples[:, :, role_index, component_index]
            target = truth[:, role_index, component_index]
            coverage = _coverage(values, target, weights)
            errors = np.asarray(
                [_binomial_error(value, len(target)) for value in coverage]
            )
            fig, ax = _figure()
            ax.plot(
                (0.0, 1.0),
                (0.0, 1.0),
                linestyle="--",
                linewidth=1.0,
                label="ideal calibration",
            )
            ax.errorbar(
                CALIBRATION_LEVELS,
                coverage,
                yerr=errors,
                marker="o",
                linestyle="-",
                capsize=3,
                label="posterior",
            )
            ax.set_xlabel("Nominal central interval coverage")
            ax.set_ylabel("Empirical coverage")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            _stats_box(ax, [f"N = {len(target)}", "error bars: binomial SE"])
            _finish(ax, legend=True)
            written += _save(
                fig,
                output / f"neutrino_{safe_role}_{component}_calibration",
                formats,
            )

            pit = _pit(values, target, weights)
            fig, ax = _figure()
            ax.hist(
                pit,
                bins=np.linspace(0.0, 1.0, 21),
                density=True,
                histtype="step",
                linewidth=1.5,
            )
            ax.axhline(1.0, linestyle="--", linewidth=1.0, label="uniform reference")
            ax.set_xlabel(f"{role} {component} finite-sample mid-rank")
            ax.set_ylabel("Density")
            ax.set_xlim(0.0, 1.0)
            _stats_box(ax, [f"N = {len(pit)}"])
            _finish(ax, legend=True)
            written += _save(
                fig,
                output / f"neutrino_{safe_role}_{component}_pit",
                formats,
            )
    return written


def _factorized_composite_point(source: h5py.File, predicted: np.ndarray) -> np.ndarray:
    conditioning = str(source.attrs.get("invisible_assignment_conditioning", ""))
    if conditioning != "hypothesis":
        return np.nanmean(predicted[:, 0], axis=1)

    branch = source["VISIBLE/constrained_probabilities"][:].astype(np.float64)
    branch /= branch.sum(axis=1, keepdims=True)
    branch_mean = np.nanmean(predicted, axis=2)
    finite = np.isfinite(branch_mean).all(axis=-1)
    support = np.where(finite, branch, 0.0)
    denominator = support.sum(axis=1, keepdims=True)
    return np.divide(
        np.where(finite[..., None], branch_mean * branch[..., None], 0.0).sum(axis=1),
        denominator,
        out=np.full((len(predicted), 4), np.nan),
        where=denominator > 0,
    )


def _physics_plots(
    source: h5py.File,
    schema: str,
    output: Path,
    formats: tuple[str, ...],
):
    if "COMPOSITES" not in source or "TRUTH/COMPOSITES" not in source:
        return []
    written = []
    for node in source["COMPOSITES"]:
        predicted = source[f"COMPOSITES/{node}"][:]
        truth = source[f"TRUTH/COMPOSITES/{node}"][:]
        point = (
            _factorized_composite_point(source, predicted)
            if schema == PREDICTION_SCHEMA
            else np.nanmean(predicted, axis=1)
        )
        finite = np.isfinite(point).all(axis=-1) & np.isfinite(truth).all(axis=-1)
        if not finite.any():
            continue
        residual = _mass(point[finite]) - _mass(truth[finite])
        written += _residual_plot(
            residual,
            xlabel=f"{node} posterior-mean mass residual [GeV]",
            output=output / f"composite_{node}_mass_residual",
            formats=formats,
        )
    return written


def _met_plot(
    source: h5py.File,
    schema: str,
    output: Path,
    formats: tuple[str, ...],
):
    if "OBSERVED/MET" not in source:
        return []
    if schema == PREDICTION_SCHEMA and "INVISIBLE/samples" in source:
        samples, weights, _ = _factorized_posterior(source)
    elif schema == JOINT_PREDICTION_SCHEMA:
        samples, weights, _ = _joint_posterior(source)
    else:
        return []

    predicted_px = np.sum(np.sum(samples[..., 0], axis=2) * weights, axis=1)
    predicted_py = np.sum(np.sum(samples[..., 1], axis=2) * weights, axis=1)
    met = source["OBSERVED/MET"][:]
    dx = predicted_px - met[:, 1]
    dy = predicted_py - met[:, 2]
    finite = np.isfinite(dx) & np.isfinite(dy)
    if not finite.any():
        return []
    dx = dx[finite]
    dy = dy[finite]

    fig, ax = _figure()
    ax.hist(dx, bins=50, histtype="step", linewidth=1.4, label=r"$p_x$")
    ax.hist(dy, bins=50, histtype="step", linewidth=1.4, label=r"$p_y$")
    ax.axvline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("Generated invisible sum - observed MET component [GeV]")
    ax.set_ylabel("Events")
    _stats_box(
        ax,
        [
            f"N = {len(dx)}",
            f"px bias = {dx.mean():.3g} GeV",
            f"py bias = {dy.mean():.3g} GeV",
            f"2D RMSE = {np.sqrt(np.mean(dx**2 + dy**2)):.3g} GeV",
        ],
    )
    _finish(ax, legend=True)
    return _save(fig, output / "met_closure", formats)


def plot_prediction(
    prediction: str | Path,
    output_dir: str | Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[Path]:
    prediction = Path(prediction).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_any_predictions(prediction)
    metrics_path = output / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    written = [metrics_path]

    with h5py.File(prediction, "r") as source:
        schema = str(source.attrs["schema_version"])
        if schema not in {
            PREDICTION_SCHEMA,
            DISCRETE_PREDICTION_SCHEMA,
            JOINT_PREDICTION_SCHEMA,
        }:
            raise ValueError(f"Unsupported prediction schema {schema!r}")
        read_hdf5_support(source)
        written += _assignment_plots(source, schema, output, formats)
        written += _invisible_plots(source, schema, output, formats)
        written += _physics_plots(source, schema, output, formats)
        written += _met_plot(source, schema, output, formats)
    return written


def _assignment_summary(metrics: dict) -> tuple[float, float]:
    if metrics["assignment_formulation"] == "discriminative":
        complete = metrics["visible"]["complete"]
        return (
            complete["top1_efficiency"],
            complete["top1_efficiency_standard_error"],
        )
    assignment = metrics["assignment"]
    return (
        assignment["empirical_mode_accuracy"],
        assignment["empirical_mode_accuracy_standard_error"],
    )


def _energy_score(metrics: dict):
    invisible = metrics.get("invisible")
    if invisible is None:
        return None
    if metrics["model_mode"] == "factorized":
        return invisible["truncated_assignment_weighted_mixture"]["energy_score_joint"]
    return invisible["energy_score_joint"]


def _sample_equivalent_nfe(metrics: dict):
    if "sample_equivalent_nfe_per_event" in metrics:
        return metrics["sample_equivalent_nfe_per_event"]
    if (
        metrics["model_mode"] == "factorized"
        and metrics["assignment_formulation"] == "flow_matching"
    ):
        return metrics.get("network_calls_per_event")
    return None


def _bar_plot(
    labels: list[str],
    values: list[float],
    *,
    ylabel: str,
    output: Path,
    formats: tuple[str, ...],
    errors: list[float] | None = None,
    note: str | None = None,
) -> list[Path]:
    fig, ax = _figure()
    positions = np.arange(len(labels))
    ax.bar(positions, values, yerr=errors, capsize=3 if errors is not None else 0)
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    if note is not None:
        ax.text(0.02, 0.96, note, transform=ax.transAxes, ha="left", va="top", fontsize=9)
    if values:
        upper = max(values)
        offset = 0.02 * upper if upper > 0 else 0.02
        for position, value in zip(positions, values):
            ax.text(position, value + offset, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    if errors is not None and values and max(values) <= 1.0:
        ax.set_ylim(0.0, min(1.08, max(1.0, max(values) + max(errors, default=0.0) + 0.08)))
    _finish(ax)
    return _save(fig, output, formats)


def plot_comparison(
    predictions: list[str | Path],
    output_dir: str | Path,
    *,
    labels: list[str] | None = None,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[Path]:
    if len(predictions) < 2:
        return []
    paths = [Path(path).expanduser().resolve() for path in predictions]
    metrics = [evaluate_any_predictions(path) for path in paths]
    if labels is None:
        labels = [path.stem for path in paths]
    if len(labels) != len(paths):
        raise ValueError("Comparison labels must match prediction files")

    topology_signatures = {item["topology_signature"] for item in metrics}
    if len(topology_signatures) != 1:
        raise ValueError(
            "Comparison plots require one topology signature; do not compare physics "
            "metrics across different topology definitions"
        )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "topology_signature": next(iter(topology_signatures)),
        "inputs": [
            {
                "label": label,
                "prediction_file": str(path),
                "model_mode": item["model_mode"],
                "assignment_formulation": item["assignment_formulation"],
                "assignment_support": item["assignment_support"],
            }
            for label, path, item in zip(labels, paths, metrics)
        ],
        "assignment_accuracy_comparison_is_grouped_by_support": True,
    }
    manifest_path = output / "comparison_metrics.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    written = [manifest_path]

    by_support: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(metrics):
        by_support[item["assignment_support"]["name"]].append(index)
    for support, indices in sorted(by_support.items()):
        summaries = [_assignment_summary(metrics[index]) for index in indices]
        written += _bar_plot(
            [labels[index] for index in indices],
            [value for value, _ in summaries],
            errors=[error for _, error in summaries],
            ylabel="Complete-assignment accuracy",
            note=f"support: {_support_label(support)}; error bars: binomial SE",
            output=output / f"comparison_assignment_accuracy_{support}",
            formats=formats,
        )

    energy = [_energy_score(item) for item in metrics]
    available = [index for index, value in enumerate(energy) if value is not None]
    if available:
        written += _bar_plot(
            [labels[index] for index in available],
            [float(energy[index]) for index in available],
            ylabel="Joint invisible energy score",
            output=output / "comparison_energy_score",
            formats=formats,
        )

    nfe = [_sample_equivalent_nfe(item) for item in metrics]
    available = [index for index, value in enumerate(nfe) if value is not None]
    if available:
        written += _bar_plot(
            [labels[index] for index in available],
            [float(nfe[index]) for index in available],
            ylabel="Sample-equivalent network evaluations / event",
            output=output / "comparison_sample_equivalent_nfe",
            formats=formats,
        )
    return written
