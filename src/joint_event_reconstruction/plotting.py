"""Schema-aware plots for maintained reconstruction prediction products."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions
from joint_event_reconstruction.prediction import PREDICTION_SCHEMA
from joint_event_reconstruction.prediction_discrete import DISCRETE_PREDICTION_SCHEMA
from joint_event_reconstruction.prediction_joint import JOINT_PREDICTION_SCHEMA


COMPONENTS = ("px", "py", "pz")
CALIBRATION_LEVELS = np.asarray((0.50, 0.68, 0.90, 0.95))


def _save(fig, base: Path, formats: tuple[str, ...]) -> list[Path]:
    outputs = []
    for extension in formats:
        path = base.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight")
        outputs.append(path)
    plt.close(fig)
    return outputs


def _mode(states: np.ndarray) -> np.ndarray:
    result = np.empty(len(states), dtype=states.dtype)
    for event, row in enumerate(states):
        values, counts = np.unique(row, return_counts=True)
        result[event] = values[np.argmax(counts)]
    return result


def _weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> np.ndarray:
    result = np.empty(len(values), dtype=np.float64)
    for event in range(len(values)):
        order = np.argsort(values[event], kind="stable")
        cumulative = np.cumsum(weights[event, order])
        index = min(
            int(np.searchsorted(cumulative, quantile, side="left")),
            values.shape[1] - 1,
        )
        result[event] = values[event, order[index]]
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
        low = _weighted_quantile(values, weights, (1 - level) / 2)
        high = _weighted_quantile(values, weights, 1 - (1 - level) / 2)
        result.append(((target >= low) & (target <= high)).mean())
    return np.asarray(result)


def _pit(values: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.sum(weights * (values <= target[:, None]), axis=1)


def _mass(vectors: np.ndarray) -> np.ndarray:
    mass2 = vectors[..., 0] ** 2 - np.sum(vectors[..., 1:] ** 2, axis=-1)
    return np.sqrt(np.maximum(mass2, 0.0))


def _assignment_plots(source: h5py.File, schema: str, output: Path, formats):
    written = []
    njet = source["OBSERVED/njet"][:]
    if schema == PREDICTION_SCHEMA:
        truth = source["TRUTH/topk_assignment"][:, 0].astype(bool)
        values = []
        labels = []
        for jet_count in np.unique(njet):
            mask = njet == jet_count
            labels.append(str(int(jet_count)))
            values.append(float(truth[mask].mean()))
        fig, ax = plt.subplots()
        ax.plot(labels, values, marker="o")
        ax.set_xlabel("Jet multiplicity")
        ax.set_ylabel("Top-1 complete-assignment efficiency")
        ax.set_ylim(0, 1)
        written += _save(fig, output / "assignment_vs_njet", formats)

        probabilities = source["VISIBLE/constrained_probabilities"][:]
        retained = probabilities.sum(axis=1)
        fig, ax = plt.subplots()
        ax.hist(retained, bins=40)
        ax.set_xlabel("Retained top-K assignment probability mass")
        ax.set_ylabel("Events")
        written += _save(fig, output / "assignment_retained_mass", formats)

        node_group = source["VISIBLE/nodes"]
        fig, ax = plt.subplots()
        for name in node_group:
            values = node_group[name]["truth_probability"][:]
            reconstructable = node_group[name]["reconstructable"][:].astype(bool)
            if reconstructable.any():
                ax.hist(
                    values[reconstructable], bins=40, histtype="step", label=name
                )
        ax.set_xlabel("Truth candidate probability")
        ax.set_ylabel("Events")
        if len(node_group):
            ax.legend()
        written += _save(fig, output / "assignment_truth_probability", formats)
        return written

    states = source["HYPOTHESES/assignment_state"][:]
    truth_state = source["TRUTH/assignment_state"][:]
    mode = _mode(states)
    accuracy = mode == truth_state
    values = []
    labels = []
    for jet_count in np.unique(njet):
        mask = njet == jet_count
        labels.append(str(int(jet_count)))
        values.append(float(accuracy[mask].mean()))
    fig, ax = plt.subplots()
    ax.plot(labels, values, marker="o")
    ax.set_xlabel("Jet multiplicity")
    ax.set_ylabel("Empirical-mode assignment accuracy")
    ax.set_ylim(0, 1)
    written += _save(fig, output / "assignment_vs_njet", formats)

    truth_frequency = (states == truth_state[:, None]).mean(axis=1)
    fig, ax = plt.subplots()
    ax.hist(truth_frequency, bins=np.linspace(0, 1, 31))
    ax.set_xlabel("Truth-assignment posterior sample frequency")
    ax.set_ylabel("Events")
    written += _save(fig, output / "assignment_truth_frequency", formats)

    unique = np.asarray([len(np.unique(row)) for row in states])
    fig, ax = plt.subplots()
    ax.hist(unique, bins=np.arange(unique.max() + 2) - 0.5)
    ax.set_xlabel("Unique sampled assignment states per event")
    ax.set_ylabel("Events")
    written += _save(fig, output / "assignment_unique_states", formats)
    return written


def _invisible_plots(source: h5py.File, schema: str, output: Path, formats):
    if schema == DISCRETE_PREDICTION_SCHEMA:
        return []
    if schema == PREDICTION_SCHEMA:
        if "INVISIBLE/samples" not in source:
            return []
        samples, weights, _ = _factorized_posterior(source)
    elif schema == JOINT_PREDICTION_SCHEMA:
        samples, weights, _ = _joint_posterior(source)
    else:
        return []
    truth = source["TRUTH/NEUTRINO"][:]
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
            fig, ax = plt.subplots()
            ax.hist(residual, bins=50)
            ax.set_xlabel(f"{role} {component} posterior-mean residual [GeV]")
            ax.set_ylabel("Events")
            written += _save(
                fig,
                output / f"neutrino_{safe_role}_{component}_residual",
                formats,
            )

            values = samples[:, :, role_index, component_index]
            target = truth[:, role_index, component_index]
            coverage = _coverage(values, target, weights)
            fig, ax = plt.subplots()
            ax.plot(CALIBRATION_LEVELS, CALIBRATION_LEVELS, linestyle="--")
            ax.plot(CALIBRATION_LEVELS, coverage, marker="o")
            ax.set_xlabel("Nominal central interval coverage")
            ax.set_ylabel("Empirical coverage")
            ax.set_xlim(0.45, 1.0)
            ax.set_ylim(0.45, 1.0)
            written += _save(
                fig,
                output / f"neutrino_{safe_role}_{component}_calibration",
                formats,
            )

            pit = _pit(values, target, weights)
            fig, ax = plt.subplots()
            ax.hist(pit, bins=np.linspace(0, 1, 21))
            ax.set_xlabel(f"{role} {component} posterior rank / PIT")
            ax.set_ylabel("Events")
            written += _save(
                fig,
                output / f"neutrino_{safe_role}_{component}_pit",
                formats,
            )
    return written


def _physics_plots(source: h5py.File, schema: str, output: Path, formats):
    if "COMPOSITES" not in source or "TRUTH/COMPOSITES" not in source:
        return []
    written = []
    for node in source["COMPOSITES"]:
        predicted = source[f"COMPOSITES/{node}"][:]
        truth = source[f"TRUTH/COMPOSITES/{node}"][:]
        if schema == PREDICTION_SCHEMA:
            if str(source.attrs.get("invisible_assignment_conditioning", "")) == "hypothesis":
                branch = source["VISIBLE/constrained_probabilities"][:].astype(np.float64)
                branch /= branch.sum(axis=1, keepdims=True)
                branch_mean = np.nanmean(predicted, axis=2)
                finite = np.isfinite(branch_mean).all(axis=-1)
                support = np.where(finite, branch, 0.0)
                denominator = support.sum(axis=1, keepdims=True)
                point = np.divide(
                    np.where(finite[..., None], branch_mean * branch[..., None], 0.0).sum(axis=1),
                    denominator,
                    out=np.full((len(predicted), 4), np.nan),
                    where=denominator > 0,
                )
            else:
                point = np.nanmean(predicted[:, 0], axis=1)
        else:
            point = np.nanmean(predicted, axis=1)
        finite = np.isfinite(point).all(axis=-1) & np.isfinite(truth).all(axis=-1)
        if not finite.any():
            continue
        residual = _mass(point[finite]) - _mass(truth[finite])
        fig, ax = plt.subplots()
        ax.hist(residual, bins=50)
        ax.set_xlabel(f"{node} posterior-mean mass residual [GeV]")
        ax.set_ylabel("Events")
        written += _save(
            fig,
            output / f"composite_{node}_mass_residual",
            formats,
        )
    return written


def _met_plot(source: h5py.File, schema: str, output: Path, formats):
    if "OBSERVED/MET" not in source:
        return []
    if schema == PREDICTION_SCHEMA and "INVISIBLE/samples" in source:
        samples, weights, _ = _factorized_posterior(source)
    elif schema == JOINT_PREDICTION_SCHEMA:
        samples, weights, _ = _joint_posterior(source)
    else:
        return []
    predicted_px = np.sum(
        np.sum(samples[..., 0], axis=2) * weights, axis=1
    )
    predicted_py = np.sum(
        np.sum(samples[..., 1], axis=2) * weights, axis=1
    )
    met = source["OBSERVED/MET"][:]
    fig, ax = plt.subplots()
    ax.hist(predicted_px - met[:, 1], bins=50, histtype="step", label="px")
    ax.hist(predicted_py - met[:, 2], bins=50, histtype="step", label="py")
    ax.set_xlabel("Generated invisible sum - observed MET component [GeV]")
    ax.set_ylabel("Events")
    ax.legend()
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
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    written = [output / "metrics.json"]
    with h5py.File(prediction, "r") as source:
        schema = str(source.attrs["schema_version"])
        if schema not in {
            PREDICTION_SCHEMA,
            DISCRETE_PREDICTION_SCHEMA,
            JOINT_PREDICTION_SCHEMA,
        }:
            raise ValueError(f"Unsupported prediction schema {schema!r}")
        written += _assignment_plots(source, schema, output, formats)
        written += _invisible_plots(source, schema, output, formats)
        written += _physics_plots(source, schema, output, formats)
        written += _met_plot(source, schema, output, formats)
    return written


def _summary_value(metrics: dict, key: str):
    if key == "assignment_accuracy":
        if metrics["assignment_formulation"] == "discriminative":
            return metrics["visible"]["complete"]["top1_efficiency"]
        return metrics["assignment"]["empirical_mode_accuracy"]
    if key == "energy_score":
        invisible = metrics.get("invisible")
        if invisible is None:
            return None
        if metrics["model_mode"] == "factorized":
            return invisible["truncated_assignment_weighted_mixture"][
                "energy_score_joint"
            ]
        return invisible["energy_score_joint"]
    if key == "sample_equivalent_nfe":
        return metrics.get("sample_equivalent_nfe_per_event")
    raise KeyError(key)


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
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for key, ylabel in (
        ("assignment_accuracy", "Assignment accuracy"),
        ("energy_score", "Joint invisible energy score"),
        ("sample_equivalent_nfe", "Sample-equivalent NFE / event"),
    ):
        values = [_summary_value(item, key) for item in metrics]
        available = [index for index, value in enumerate(values) if value is not None]
        if not available:
            continue
        fig, ax = plt.subplots()
        ax.bar(
            [labels[index] for index in available],
            [values[index] for index in available],
        )
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        written += _save(fig, output / f"comparison_{key}", formats)
    return written
