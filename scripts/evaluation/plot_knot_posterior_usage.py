#!/usr/bin/env python3
"""Evaluate point and population use of the frozen KNOT posterior samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chisquare, wasserstein_distance

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from joint_event_reconstruction.plotting_style import (
    COLORS,
    LABELS,
    controlled_ratio_ylim,
    decorate_axis,
    format_legend,
    plot_histogram_ratio,
    ratio_axes,
    save_figure,
    uncertainty_band,
)

EVALUATION = ROOT / "results/experiments/knot/matched-production-v1/evaluation-v1/full-10k"
DEFAULT_INPUT = EVALUATION / "J/posterior.h5"
DEFAULT_OUTPUT = EVALUATION / "posterior-usage-plotting-v1"
DRAW_SEED = 42011
PSEUDO_DATASETS = 128
MIN_RATIO_SUPPORT = 25.0
MIN_DIFFERENTIAL_SUPPORT = 50
BOOTSTRAP_REPLICATES = 200
NOMINAL_COVERAGE = (0.05, 0.10, 0.20, 0.32, 0.50, 0.68, 0.80, 0.90, 0.95, 0.99)

PARENT_OBSERVABLES: dict[str, dict[str, Any]] = {
    "mttbar": {
        "label": r"$m_{t\bar t}$ [GeV]",
        "unit": "GeV",
        "density_label": r"Normalised event density [GeV$^{-1}$]",
        "edges": np.linspace(0.0, 3500.0, 71),
        "migration_edges": np.asarray((0, 380, 440, 500, 650, 900, 3500), dtype=np.float64),
        "differential_edges": np.asarray((0, 380, 440, 500, 650, 900, 3500), dtype=np.float64),
        "positive_response": True,
    },
    "ptttbar": {
        "label": r"$p_T^{t\bar t}$ [GeV]",
        "unit": "GeV",
        "density_label": r"Normalised event density [GeV$^{-1}$]",
        "edges": np.linspace(0.0, 1500.0, 61),
        "differential_edges": np.asarray((0, 20, 50, 100, 200, 400, 1500), dtype=np.float64),
        "positive_response": True,
    },
    "yttbar": {
        "label": r"$y_{t\bar t}$",
        "unit": "",
        "density_label": "Normalised event density",
        "edges": np.linspace(-3.0, 3.0, 61),
        "differential_edges": np.asarray((-3, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 3), dtype=np.float64),
        "positive_response": False,
    },
    "top_pt": {
        "label": r"Top and antitop $p_T$ [GeV]",
        "unit": "GeV",
        "density_label": r"Normalised particle density [GeV$^{-1}$]",
        "edges": np.linspace(0.0, 1500.0, 61),
        "migration_edges": np.asarray((0, 60, 100, 150, 220, 320, 500, 1500), dtype=np.float64),
        "differential_edges": np.asarray((0, 60, 100, 150, 220, 320, 500, 1500), dtype=np.float64),
        "positive_response": True,
    },
}

NEUTRINO_OBSERVABLES: dict[str, dict[str, Any]] = {
    "neutrino_pt": {
        "label": r"Neutrino $p_T$ [GeV]",
        "unit": "GeV",
        "density_label": r"Normalised particle density [GeV$^{-1}$]",
        "edges": np.linspace(0.0, 800.0, 65),
    },
    "neutrino_pz": {
        "label": r"Neutrino $p_z$ [GeV]",
        "unit": "GeV",
        "density_label": r"Normalised particle density [GeV$^{-1}$]",
        "edges": np.linspace(-2000.0, 2000.0, 81),
    },
}

SPIN_OBSERVABLES: dict[str, dict[str, Any]] = {
    "cos_opening": {
        "label": r"Lepton opening-angle cosine",
        "unit": "",
        "density_label": "Normalised event density",
        "edges": np.linspace(-1.0, 1.0, 51),
    },
    "cos_theta_plus": {
        "label": r"Positive-lepton helicity cosine",
        "unit": "",
        "density_label": "Normalised event density",
        "edges": np.linspace(-1.0, 1.0, 51),
    },
    "cos_theta_minus": {
        "label": r"Negative-lepton helicity cosine",
        "unit": "",
        "density_label": "Normalised event density",
        "edges": np.linspace(-1.0, 1.0, 51),
    },
    "helicity_product": {
        "label": r"Helicity-cosine product",
        "unit": "",
        "density_label": "Normalised event density",
        "edges": np.linspace(-1.0, 1.0, 51),
    },
}


def _finite(values: np.ndarray, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains NaN or infinity")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _event_id_hash(event_ids: np.ndarray) -> str:
    values = np.asarray(event_ids)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("EVENT_ID must be a one-dimensional integer array")
    return hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()


def _draw_choice_hash(event_ids: np.ndarray, indices: np.ndarray) -> str:
    event_ids = np.asarray(event_ids, dtype="<i8")
    indices = np.asarray(indices, dtype="<i8")
    if event_ids.shape != indices.shape or event_ids.ndim != 1:
        raise ValueError("Event IDs and draw choices must be aligned one-dimensional arrays")
    order = np.argsort(event_ids, kind="stable")
    payload = event_ids[order].tobytes() + indices[order].tobytes()
    return hashlib.sha256(payload).hexdigest()


def event_keyed_uniforms(event_ids: np.ndarray, *, seed: int, stream: int = 0) -> np.ndarray:
    """PCFM SplitMix64 event-keyed uniforms; independent of row and batch order."""
    ids = np.asarray(event_ids)
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("Event-keyed draws require a one-dimensional integer EVENT_ID array")
    with np.errstate(over="ignore"):
        value = ids.astype(np.uint64, copy=True)
        value ^= np.uint64(seed)
        value ^= np.uint64(stream + 1) * np.uint64(0xD2B74407B1CE6E93)
        value += np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return ((value >> np.uint64(11)).astype(np.float64) + 0.5) / float(1 << 53)


def event_keyed_indices(
    event_ids: np.ndarray,
    sample_count: int,
    *,
    seed: int,
    stream: int = 0,
    repeats: int = 1,
) -> np.ndarray:
    if sample_count <= 0 or repeats <= 0:
        raise ValueError("Posterior sample and pseudo-dataset counts must be positive")
    uniforms = np.stack(
        [event_keyed_uniforms(event_ids, seed=seed, stream=stream + index) for index in range(repeats)],
        axis=0,
    )
    return np.minimum((uniforms * sample_count).astype(np.int64), sample_count - 1)


def _parent_observables(parents: np.ndarray, label: str) -> dict[str, np.ndarray]:
    parents = _finite(parents, f"{label} parent four-vectors")
    if parents.ndim not in {3, 4} or parents.shape[-2:] != (2, 4):
        raise ValueError(f"{label} parents must end in [top/antitop, E,px,py,pz], got {parents.shape}")
    total = parents.sum(axis=-2)
    mass2 = total[..., 0] ** 2 - np.square(total[..., 1:]).sum(axis=-1)
    if (mass2 < -1.0e-3).any() or (total[..., 0] <= np.abs(total[..., 3])).any():
        raise ValueError(f"{label} contains a spacelike or non-forward parent pair")
    rapidities = []
    for role, role_name in enumerate(("top", "antitop")):
        energy, pz = parents[..., role, 0], parents[..., role, 3]
        if (energy <= np.abs(pz)).any():
            raise ValueError(f"{label} {role_name} rapidity is undefined")
        rapidities.append(0.5 * np.log((energy + pz) / (energy - pz)))
    if (total[..., 0] <= np.abs(total[..., 3])).any():
        raise ValueError(f"{label} ttbar rapidity is undefined")
    return {
        "mttbar": np.sqrt(np.maximum(mass2, 0.0)),
        "top_pt": np.linalg.norm(parents[..., :, 1:3], axis=-1),
        "ptttbar": np.linalg.norm(total[..., 1:3], axis=-1),
        "yttbar": 0.5 * np.log((total[..., 0] + total[..., 3]) / (total[..., 0] - total[..., 3])),
    }


def _neutrino_observables(neutrinos: np.ndarray, label: str) -> dict[str, np.ndarray]:
    neutrinos = _finite(neutrinos, f"{label} neutrino four-vectors")
    if neutrinos.shape[-2:] != (2, 4):
        raise ValueError(f"{label} neutrinos must end in [top/antitop,E,px,py,pz], got {neutrinos.shape}")
    return {
        "neutrino_pt": np.linalg.norm(neutrinos[..., 1:3], axis=-1),
        "neutrino_pz": neutrinos[..., 3],
    }


def _load_posterior(
    path: Path, *, expected_model: str | None = None, include_spin: bool = False
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"KNOT posterior file is missing: {path}")
    with h5py.File(path, "r") as source:
        required = (
            "EVENT_ID",
            "TRUTH/parents",
            "parents",
            "TRUTH/neutrinos",
            "invisible",
            "parent_support",
            "invisible_support",
            "sample_index",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"{path}: missing required datasets {missing}")
        event_ids = source["EVENT_ID"][:]
        truth_parents = source["TRUTH/parents"][:]
        parent_samples = source["parents"][:]
        truth_neutrinos = source["TRUTH/neutrinos"][:]
        neutrino_samples = source["invisible"][:]
        parent_support = source["parent_support"][:].astype(bool)
        neutrino_support = source["invisible_support"][:].astype(bool)
        sample_index = source["sample_index"][:]
        attrs = {str(key): source.attrs[key] for key in source.attrs}
        spin_truth = {}
        spin_samples = {}
        if include_spin:
            for name in SPIN_OBSERVABLES:
                truth_key, samples_key = f"spin/truth/{name}", f"spin/paired/{name}"
                missing_spin = [key for key in (truth_key, samples_key) if key not in source]
                if missing_spin:
                    raise KeyError(f"{path}: missing required paired-spin datasets {missing_spin}")
                spin_truth[name] = source[truth_key][:]
                spin_samples[name] = source[samples_key][:]
    if event_ids.ndim != 1 or not np.issubdtype(event_ids.dtype, np.integer):
        raise ValueError(f"{path}: EVENT_ID must be a one-dimensional integer array")
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError(f"{path}: EVENT_ID contains duplicate events")
    n_events = len(event_ids)
    if truth_parents.shape != (n_events, 2, 4) or truth_neutrinos.shape != (n_events, 2, 4):
        raise ValueError(f"{path}: truth-role arrays do not match the {n_events} EVENT_ID rows")
    if parent_samples.ndim != 4 or parent_samples.shape[0] != n_events or parent_samples.shape[2:] != (2, 4):
        raise ValueError(f"{path}: parents must be [N_event,K,2,4], got {parent_samples.shape}")
    if neutrino_samples.shape != parent_samples.shape:
        raise ValueError(f"{path}: invisible shape {neutrino_samples.shape} does not match parents {parent_samples.shape}")
    sample_count = parent_samples.shape[1]
    if sample_index.shape != (n_events, sample_count):
        raise ValueError(f"{path}: sample_index does not align with [N_event,K]")
    if parent_support.shape != (n_events, sample_count, 2) or neutrino_support.shape != parent_support.shape:
        raise ValueError(f"{path}: posterior support arrays do not align with [N_event,K,role]")
    if not parent_support.all():
        bad = int(np.flatnonzero(~parent_support.all(axis=(1, 2)))[0])
        raise ValueError(f"{path}: missing parent support for EVENT_ID={int(event_ids[bad])}")
    if not neutrino_support.all():
        bad = int(np.flatnonzero(~neutrino_support.all(axis=(1, 2)))[0])
        raise ValueError(f"{path}: missing neutrino support for EVENT_ID={int(event_ids[bad])}")
    if expected_model is not None and str(attrs.get("factorisation", "")) != expected_model:
        raise ValueError(f"{path}: expected factorisation {expected_model}, found {attrs.get('factorisation')!r}")
    if include_spin:
        if "spin_definition" not in attrs:
            raise KeyError(f"{path}: missing spin_definition attribute")
        for name in SPIN_OBSERVABLES:
            if spin_truth[name].shape != (n_events,) or spin_samples[name].shape != (n_events, sample_count):
                raise ValueError(
                    f"{path}: spin/{name} must align as truth [N] and paired posterior [N,K]"
                )
            spin_truth[name] = _finite(spin_truth[name], f"{path} spin/truth/{name}")
            spin_samples[name] = _finite(spin_samples[name], f"{path} spin/paired/{name}")
    return {
        "path": path.resolve(),
        "event_ids": event_ids,
        "truth_parents": _finite(truth_parents, f"{path} TRUTH/parents"),
        "parent_samples": _finite(parent_samples, f"{path} parents"),
        "truth_neutrinos": _finite(truth_neutrinos, f"{path} TRUTH/neutrinos"),
        "neutrino_samples": _finite(neutrino_samples, f"{path} invisible"),
        "sample_index": sample_index,
        "attrs": attrs,
        "sample_count": sample_count,
        "spin_truth": spin_truth,
        "spin_samples": spin_samples,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_population(values: np.ndarray, truth: np.ndarray, *, representation: str) -> tuple[np.ndarray, np.ndarray]:
    """Flatten an observable while preserving one total weight per physics event."""
    values = _finite(values, f"{representation} population values")
    truth = _finite(truth, "population reference values")
    if representation == "posterior_distribution":
        if values.ndim != truth.ndim + 1 or values.shape[0] != truth.shape[0] or values.shape[2:] != truth.shape[1:]:
            raise ValueError(f"Posterior distribution {values.shape} is not [N,K]+{truth.shape[1:]} aligned to truth {truth.shape}")
        samples = values.shape[1]
        objects_per_event = int(np.prod(truth.shape[1:])) if truth.ndim > 1 else 1
        if samples <= 0 or objects_per_event <= 0:
            raise ValueError("Posterior distribution has empty event or particle support")
        weights = np.full(values.size, 1.0 / (samples * objects_per_event), dtype=np.float64)
        per_event = weights.reshape((len(truth), samples, objects_per_event)).sum(axis=(1, 2))
    else:
        if values.shape != truth.shape:
            raise ValueError(f"{representation} values {values.shape} do not match truth {truth.shape}")
        objects_per_event = int(np.prod(truth.shape[1:])) if truth.ndim > 1 else 1
        weights = np.full(values.size, 1.0 / objects_per_event, dtype=np.float64)
        per_event = weights.reshape((len(truth), objects_per_event)).sum(axis=1)
    if not np.allclose(per_event, 1.0, atol=1e-14, rtol=0.0):
        raise ValueError(f"{representation}: posterior weights do not sum to one per event")
    return values.reshape(-1), weights


def ordinary_event_residuals(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    representation: str,
    label: str,
) -> np.ndarray:
    """Return point residuals; a full posterior distribution is never a point estimator."""
    if representation == "posterior_distribution":
        raise ValueError(f"{label}: posterior_distribution is forbidden in ordinary event residuals")
    prediction = _finite(prediction, f"{label} point prediction")
    truth = _finite(truth, f"{label} truth")
    if prediction.ndim not in {1, 2} or prediction.shape != truth.shape:
        raise ValueError(f"{label}: point residual arrays must align one value per event/object, got {prediction.shape}/{truth.shape}")
    return prediction - truth


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = _finite(values, "weighted-quantile values").reshape(-1)
    weights = _finite(weights, "weighted-quantile weights").reshape(-1)
    if values.shape != weights.shape or (weights < 0.0).any() or not 0.0 <= quantile <= 1.0:
        raise ValueError("Weighted quantile values, weights, or requested quantile are invalid")
    order = np.argsort(values, kind="stable")
    x, w = values[order], weights[order]
    cumulative = np.cumsum(w)
    if not len(x) or cumulative[-1] <= 0.0:
        raise ValueError("Weighted quantile has no positive support")
    index = min(int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left")), len(x) - 1)
    return float(x[index])


def _weighted_ks_distance(a: np.ndarray, wa: np.ndarray, b: np.ndarray, wb: np.ndarray) -> float:
    a, wa = _finite(a, "KS reference values").reshape(-1), _finite(wa, "KS reference weights").reshape(-1)
    b, wb = _finite(b, "KS prediction values").reshape(-1), _finite(wb, "KS prediction weights").reshape(-1)
    if a.shape != wa.shape or b.shape != wb.shape or (wa < 0.0).any() or (wb < 0.0).any():
        raise ValueError("KS values and non-negative weights must align")
    if wa.sum() <= 0.0 or wb.sum() <= 0.0:
        raise ValueError("KS distributions must have positive total weight")
    points = np.unique(np.concatenate((a, b)))
    def cdf(x, w):
        order = np.argsort(x, kind="stable")
        sx, sw = x[order], w[order]
        cumulative = np.cumsum(sw) / sw.sum()
        indices = np.searchsorted(sx, points, side="right") - 1
        result = np.zeros(len(points), dtype=np.float64)
        selected = indices >= 0
        result[selected] = cumulative[indices[selected]]
        return result
    return float(np.max(np.abs(cdf(a, wa) - cdf(b, wb))))


def _crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """Mean empirical-posterior CRPS using the finite-sample energy-score identity."""
    samples = _finite(samples, "CRPS posterior samples")
    truth = _finite(truth, "CRPS truth")
    if samples.ndim != 2 or truth.shape != (samples.shape[0],) or samples.shape[1] == 0:
        raise ValueError(f"CRPS requires [event,K] samples and one truth value/event, got {samples.shape}/{truth.shape}")
    first = np.mean(np.abs(samples - truth[:, None]), axis=1)
    ordered = np.sort(samples, axis=1)
    k = samples.shape[1]
    coefficients = 2.0 * np.arange(k, dtype=np.float64) + 1.0 - k
    pair_term = (ordered @ coefficients) / float(k * k)
    scores = first - pair_term
    if (scores < -1e-10).any():
        raise ValueError("CRPS calculation became negative beyond floating-point tolerance")
    return float(np.mean(np.maximum(scores, 0.0)))


def _spectrum_metrics(truth: np.ndarray, prediction: np.ndarray, *, representation: str) -> dict[str, float]:
    truth_flat, truth_weights = _flatten_population(truth, truth, representation="posterior_mean")
    pred_flat, pred_weights = _flatten_population(prediction, truth, representation=representation)
    q = {name: _weighted_quantile(truth_flat, truth_weights, value) for name, value in (("q90", 0.9), ("q95", 0.95), ("q99", 0.99))}
    truth_iqr = _weighted_quantile(truth_flat, truth_weights, 0.75) - _weighted_quantile(truth_flat, truth_weights, 0.25)
    pred_iqr = _weighted_quantile(pred_flat, pred_weights, 0.75) - _weighted_quantile(pred_flat, pred_weights, 0.25)
    truth_width = _weighted_quantile(truth_flat, truth_weights, 0.84) - _weighted_quantile(truth_flat, truth_weights, 0.16)
    pred_width = _weighted_quantile(pred_flat, pred_weights, 0.84) - _weighted_quantile(pred_flat, pred_weights, 0.16)
    tail = {}
    for name, threshold in q.items():
        truth_fraction = float(np.average(truth_flat >= threshold, weights=truth_weights))
        pred_fraction = float(np.average(pred_flat >= threshold, weights=pred_weights))
        tail[f"{name}_truth_tail_fraction"] = truth_fraction
        tail[f"{name}_prediction_tail_fraction"] = pred_fraction
        tail[f"{name}_tail_fraction_ratio"] = pred_fraction / truth_fraction if truth_fraction > 0.0 else 0.0
    return {
        "wasserstein_1": float(wasserstein_distance(truth_flat, pred_flat, u_weights=truth_weights, v_weights=pred_weights)),
        "ks_distance": _weighted_ks_distance(truth_flat, truth_weights, pred_flat, pred_weights),
        "iqr_ratio": pred_iqr / truth_iqr if truth_iqr > 0.0 else 0.0,
        "central_68_width_ratio": pred_width / truth_width if truth_width > 0.0 else 0.0,
        "truth_mean": float(np.average(truth_flat, weights=truth_weights)),
        "prediction_mean": float(np.average(pred_flat, weights=pred_weights)),
        "truth_median": _weighted_quantile(truth_flat, truth_weights, 0.5),
        "prediction_median": _weighted_quantile(pred_flat, pred_weights, 0.5),
        **tail,
    }


def _histogram(values: np.ndarray, weights: np.ndarray, edges: np.ndarray, *, label: str) -> tuple[np.ndarray, np.ndarray]:
    values = _finite(values, f"{label} values").reshape(-1)
    weights = _finite(weights, f"{label} weights").reshape(-1)
    edges = _finite(edges, f"{label} edges").reshape(-1)
    if values.shape != weights.shape or len(edges) < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError(f"{label}: histogram values, weights, or edges are inconsistent")
    outside = (values < edges[0]) | (values > edges[-1])
    if outside.any():
        raise ValueError(f"{label}: {int(outside.sum())} values fall outside the physical plotting range [{edges[0]}, {edges[-1]}]")
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    expected = float(weights.sum())
    if not np.isclose(counts.sum(), expected, atol=1e-9 * max(1.0, expected), rtol=0.0):
        raise ValueError(f"{label}: histogram integral {counts.sum()} does not close to total weight {expected}")
    density = counts / expected / np.diff(edges)
    return counts.astype(np.float64), density.astype(np.float64)


def _representation_values(
    samples: np.ndarray, choices: np.ndarray
) -> dict[str, np.ndarray]:
    if samples.shape[0] != len(choices) or samples.shape[1] <= 0:
        raise ValueError("Posterior draws and event-keyed choices do not align")
    rows = np.arange(len(choices))
    if samples.ndim < 2:
        raise ValueError("Posterior observable must include an event and sample axis")
    selected = samples[rows, choices]
    return {
        "one_draw": _finite(selected, "one-draw observable"),
        "posterior_mean": _finite(np.mean(samples, axis=1, dtype=np.float64), "posterior-mean observable"),
        "posterior_distribution": _finite(samples, "full posterior observable"),
    }


def _residual_statistics(residual: np.ndarray) -> dict[str, float]:
    values = _finite(residual, "point-estimator residual").reshape(-1)
    return {
        "bias": float(np.mean(values)),
        "residual_sigma": float(np.std(values, ddof=0)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "mae": float(np.mean(np.abs(values))),
    }


def _plot_shape(
    output: Path,
    name: str,
    spec: dict[str, Any],
    truth: np.ndarray,
    values: dict[str, np.ndarray],
) -> tuple[list[Path], dict[str, Any], list[dict[str, Any]]]:
    edges = np.asarray(spec["edges"], dtype=np.float64)
    truth_flat, truth_weights = _flatten_population(truth, truth, representation="posterior_mean")
    truth_counts, truth_density = _histogram(truth_flat, truth_weights, edges, label=f"{name} Parton truth")
    fig, main, lower = ratio_axes()
    main.fill_between(
        edges,
        0.0,
        np.r_[truth_density, truth_density[-1]],
        step="post",
        color=COLORS["truth"],
        alpha=0.45,
        edgecolor="none",
        linewidth=0,
        label=LABELS["truth"],
        zorder=1,
    )
    ratio_values, ratio_colors = [], []
    output_rows = []
    support_metadata = {}
    for representation in ("one_draw", "posterior_mean", "posterior_distribution"):
        flat, weights = _flatten_population(values[representation], truth, representation=representation)
        counts, density = _histogram(flat, weights, edges, label=f"{name} {representation}")
        label = LABELS[{"one_draw": "draw", "posterior_mean": "mean", "posterior_distribution": "posterior"}[representation]]
        color = COLORS[{"one_draw": "draw", "posterior_mean": "mean", "posterior_distribution": "posterior"}[representation]]
        main.stairs(density, edges, color=color, linewidth=2.1, label=label, zorder=3)
        ratio, supported = plot_histogram_ratio(
            lower,
            counts,
            truth_counts,
            edges,
            method={"one_draw": "draw", "posterior_mean": "mean", "posterior_distribution": "posterior"}[representation],
            min_reference_events=MIN_RATIO_SUPPORT,
            label=label,
        )
        ratio_values.append(ratio)
        ratio_colors.append(color)
        support_metadata[representation] = {
            "supported_bins": int(supported.sum()),
            "total_bins": len(supported),
            "minimum_reference_weight": MIN_RATIO_SUPPORT,
        }
        for index, (low, high) in enumerate(pairwise(edges)):
            output_rows.append({
                "observable": name,
                "representation": representation,
                "bin_low": float(low),
                "bin_high": float(high),
                "weighted_count": float(counts[index]),
                "density": float(density[index]),
                "truth_weighted_count": float(truth_counts[index]),
                "ratio_supported": bool(supported[index]),
                "prediction_over_truth": float(ratio[index]) if supported[index] else "",
            })
    centers = 0.5 * (edges[:-1] + edges[1:])
    ratio_limit, clipped = controlled_ratio_ylim(lower, ratio_values, centers, colors=ratio_colors, center=1.0)
    main.set_ylabel(spec["density_label"])
    main.set_xlim(float(edges[0]), float(edges[-1]))
    lower.set_xlabel(spec["label"])
    decorate_axis(main, grid="y")
    decorate_axis(lower, grid="y")
    format_legend(main, ncol=2, loc="upper right", fontsize=10)
    files = save_figure(fig, output / f"{name}_shape")
    return files, {"ratio_y_limit": ratio_limit, "clipped_supported_bins": clipped, **support_metadata}, output_rows


def _plot_residual(
    output: Path,
    name: str,
    spec: dict[str, Any],
    truth: np.ndarray,
    values: dict[str, np.ndarray],
) -> tuple[list[Path], list[dict[str, Any]], list[dict[str, Any]]]:
    residuals: dict[str, np.ndarray] = {}
    rows = []
    for representation in ("one_draw", "posterior_mean"):
        residual = ordinary_event_residuals(values[representation], truth, representation=representation, label=name)
        residuals[representation] = residual
        rows.append({"observable": name, "estimator": representation, "objects": int(residual.size), **_residual_statistics(residual)})
    combined = np.abs(np.concatenate([residuals["one_draw"].reshape(-1), residuals["posterior_mean"].reshape(-1)]))
    extent = max(float(np.max(combined)), 1e-6)
    if spec["unit"] == "GeV":
        step = 25.0 if extent > 100.0 else 10.0 if extent > 30.0 else 2.0
    else:
        step = 0.1
    extent = max(step, step * math.ceil(extent / step))
    edges = np.linspace(-extent, extent, 81)
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    hist_rows = []
    objects_per_event = int(np.prod(truth.shape[1:])) if truth.ndim > 1 else 1
    for representation in ("one_draw", "posterior_mean"):
        residual = residuals[representation].reshape(-1)
        weights = np.full(len(residual), 1.0 / objects_per_event, dtype=np.float64)
        counts, density = _histogram(residual, weights, edges, label=f"{name} {representation} residual")
        key = "draw" if representation == "one_draw" else "mean"
        ax.stairs(density, edges, color=COLORS[key], linewidth=2.2, label=LABELS[key])
        for index, (low, high) in enumerate(pairwise(edges)):
            hist_rows.append({"observable": name, "estimator": representation, "bin_low": float(low), "bin_high": float(high), "weighted_count": float(counts[index]), "density": float(density[index])})
    ax.axvline(0.0, color="#30343B", linewidth=1.1)
    unit = f" [{spec['unit']}]" if spec["unit"] else ""
    ax.set_xlabel(f"Reconstructed - Parton truth{unit}")
    ax.set_ylabel("Normalised event density" if truth.ndim == 1 else "Normalised particle density")
    decorate_axis(ax, grid="y")
    format_legend(ax, ncol=2, loc="upper right", fontsize=10)
    compact = []
    for representation in ("one_draw", "posterior_mean"):
        stats = next(row for row in rows if row["estimator"] == representation)
        compact.append(f"{LABELS['draw' if representation == 'one_draw' else 'mean']}: bias {stats['bias']:.3g}, σ {stats['residual_sigma']:.3g}, RMSE {stats['rmse']:.3g}, MAE {stats['mae']:.3g}")
    ax.text(0.02, 0.97, "\n".join(compact), transform=ax.transAxes, va="top", ha="left", fontsize=8.6, color="#30343B")
    return save_figure(fig, output / f"{name}_residual"), rows, hist_rows


def _central_interval_bounds(sorted_draws: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray, float]:
    samples = sorted_draws.shape[1]
    low_index = int(np.floor((1.0 - level) * (samples + 1) / 2.0))
    high_index = int(np.floor((1.0 + level) * (samples + 1) / 2.0))
    low_index = int(np.clip(low_index, 0, samples - 1))
    high_index = int(np.clip(high_index, low_index, samples - 1))
    return sorted_draws[:, low_index], sorted_draws[:, high_index], float((high_index - low_index) / (samples + 1))


def _coverage_rows(samples: np.ndarray, truth: np.ndarray, *, observable: str) -> list[dict[str, Any]]:
    samples = _finite(samples, f"{observable} coverage samples")
    truth = _finite(truth, f"{observable} coverage truth")
    if samples.ndim == truth.ndim + 1:
        rows = int(np.prod(truth.shape))
        draws = np.moveaxis(samples, 1, -1).reshape((rows, samples.shape[1]))
        target = truth.reshape(-1)
    else:
        raise ValueError(f"{observable}: coverage inputs must align [N,K,...] with [N,...]")
    ordered = np.sort(draws, axis=1)
    result = []
    for nominal in NOMINAL_COVERAGE:
        low, high, finite_expectation = _central_interval_bounds(ordered, float(nominal))
        empirical = float(np.mean((target >= low) & (target <= high)))
        result.append({
            "diagnostic": "coverage",
            "observable": observable,
            "nominal_credibility": float(nominal),
            "empirical_coverage": empirical,
            "finite_draw_rank_expectation": finite_expectation,
            "objects": len(target),
            "posterior_samples_per_event": int(samples.shape[1]),
        })
    return result


def _rank_rows(samples: np.ndarray, truth: np.ndarray, event_ids: np.ndarray, *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = _finite(samples, "mttbar rank samples")
    truth = _finite(truth, "mttbar rank truth")
    if samples.ndim != 2 or truth.shape != (samples.shape[0],) or len(event_ids) != len(truth):
        raise ValueError("mttbar rank arrays must align as [N,K] and [N]")
    lower = np.sum(samples < truth[:, None], axis=1)
    ties = np.sum(samples == truth[:, None], axis=1)
    uniforms = event_keyed_uniforms(event_ids, seed=seed, stream=41)
    tie_offset = np.floor(uniforms * (ties + 1)).astype(np.int64)
    ranks = lower + tie_offset
    k = samples.shape[1]
    observed = np.bincount(ranks, minlength=k + 1)
    expected = np.full(k + 1, len(ranks) / (k + 1), dtype=np.float64)
    test = chisquare(observed, expected)
    rows = [{
        "diagnostic": "rank",
        "observable": "mttbar",
        "rank": int(rank),
        "events": int(observed[rank]),
        "fraction": float(observed[rank] / len(ranks)),
        "uniform_expected_fraction": float(1.0 / (k + 1)),
    } for rank in range(k + 1)]
    summary = {
        "events": len(ranks),
        "posterior_samples_per_event": int(k),
        "rank_bins": int(k + 1),
        "rank_definition": "number of posterior draws below truth; exact ties are broken uniformly over allowed discrete ranks using event-keyed SplitMix64",
        "chi_square": float(test.statistic),
        "degrees_of_freedom": int(k),
        "p_value": float(test.pvalue),
        "rank_fraction_rms_deviation_from_uniform": float(np.sqrt(np.mean(np.square(observed / len(ranks) - 1.0 / (k + 1))))),
        "rank_fraction_max_abs_deviation_from_uniform": float(np.max(np.abs(observed / len(ranks) - 1.0 / (k + 1)))),
    }
    return rows, summary


def _plot_coverage(
    output: Path,
    observable: str,
    label: str,
    rows: list[dict[str, Any]],
    *,
    filename: str | None = None,
) -> list[Path]:
    selected = [row for row in rows if row["diagnostic"] == "coverage" and row["observable"] == observable]
    if not selected:
        raise ValueError(f"No coverage points for {observable}")
    x = np.asarray([row["nominal_credibility"] for row in selected])
    y = np.asarray([row["empirical_coverage"] for row in selected])
    fig, ax = plt.subplots(figsize=(6.7, 6.0))
    ax.plot((0, 1), (0, 1), color="#737B84", linewidth=1.4, label="Ideal coverage")
    ax.plot(x, y, color=COLORS["posterior"], marker="o", linestyle="-", linewidth=2.0, label="KNOT J posterior")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Nominal credibility")
    ax.set_ylabel("Empirical coverage")
    decorate_axis(ax, grid="both")
    format_legend(ax, ncol=1, loc="upper left", fontsize=10)
    return save_figure(fig, output / f"{filename or observable}_coverage")


def _plot_rank(output: Path, rows: list[dict[str, Any]]) -> list[Path]:
    selected = [row for row in rows if row["diagnostic"] == "rank" and row["observable"] == "mttbar"]
    if not selected:
        raise ValueError("mttbar rank histogram rows are missing")
    ranks = np.asarray([row["rank"] for row in selected])
    fractions = np.asarray([row["fraction"] for row in selected])
    expected = 1.0 / len(selected)
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.bar(ranks, fractions, width=0.88, color=COLORS["posterior"], alpha=0.8, edgecolor="none", label="KNOT J posterior")
    ax.axhline(expected, color="#737B84", linewidth=1.4, label="Uniform rank")
    ax.set_xlabel("Number of posterior draws below truth")
    ax.set_ylabel("Event fraction")
    decorate_axis(ax, grid="y")
    format_legend(ax, ncol=1, loc="upper right", fontsize=10)
    return save_figure(fig, output / "mttbar_pit_rank")


def _plot_differential(
    output: Path,
    name: str,
    spec: dict[str, Any],
    truth: np.ndarray,
    values: dict[str, np.ndarray],
    *,
    seed: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    edges = np.asarray(spec["differential_edges"], dtype=np.float64)
    truth = _finite(truth, f"{name} differential truth")
    truth_flat = truth.reshape(-1)
    objects_per_event = int(np.prod(truth.shape[1:])) if truth.ndim > 1 else 1
    event_rows = np.repeat(np.arange(len(truth)), objects_per_event)
    if truth_flat.min() < edges[0] or truth_flat.max() > edges[-1]:
        raise ValueError(f"{name}: differential edges do not cover all truth values")
    fig, (bias_ax, response_ax) = plt.subplots(2, 1, figsize=(8.7, 7.2), sharex=True)
    table = []
    positive = bool(spec["positive_response"])
    used_centers: dict[str, list[float]] = {"one_draw": [], "posterior_mean": []}
    plot_values: dict[str, dict[str, list[float]]] = {
        rep: {"bias": [], "bias_low": [], "bias_high": [], "response": [], "response_low": [], "response_high": []}
        for rep in used_centers
    }
    rng = np.random.default_rng(seed)
    for bin_index, (low, high) in enumerate(pairwise(edges)):
        selected = (truth_flat >= low) & ((truth_flat < high) if bin_index < len(edges) - 2 else (truth_flat <= high))
        selected_events = np.unique(event_rows[selected])
        if len(selected_events) < MIN_DIFFERENTIAL_SUPPORT:
            continue
        local_rows = np.flatnonzero(selected)
        event_to_local = {int(event): index for index, event in enumerate(selected_events)}
        row_event_positions = np.asarray([event_to_local[int(event_rows[row])] for row in local_rows], dtype=np.int64)
        bootstrap_multiplicities = np.stack([
            np.bincount(
                rng.integers(0, len(selected_events), size=len(selected_events)),
                minlength=len(selected_events),
            )
            for _ in range(BOOTSTRAP_REPLICATES)
        ])
        for representation in ("one_draw", "posterior_mean"):
            prediction = ordinary_event_residuals(
                values[representation], truth, representation=representation, label=f"{name} differential"
            ).reshape(-1) + truth_flat
            residual = prediction[local_rows] - truth_flat[local_rows]
            if positive:
                response = prediction[local_rows] / truth_flat[local_rows]
                response_reference = 1.0
            else:
                response = prediction[local_rows]
                response_reference = float(np.mean(truth_flat[local_rows]))
            counts = bootstrap_multiplicities[:, row_event_positions]
            totals = counts.sum(axis=1)
            usable = totals > 0
            boot_bias = (counts[usable] @ residual) / totals[usable]
            boot_response = (counts[usable] @ response) / totals[usable]
            if len(boot_bias) < int(0.9 * BOOTSTRAP_REPLICATES):
                raise ValueError(f"{name}: too few nonempty event-bootstrap draws in differential bin {bin_index}")
            bias = float(np.mean(residual))
            response_value = float(np.mean(response))
            bias_low, bias_high = map(float, np.quantile(boot_bias, (0.16, 0.84)))
            response_low, response_high = map(float, np.quantile(boot_response, (0.16, 0.84)))
            center = float(np.mean(truth_flat[local_rows]))
            key = "draw" if representation == "one_draw" else "mean"
            used_centers[representation].append(center)
            plot_values[representation]["bias"].append(bias)
            plot_values[representation]["bias_low"].append(bias_low)
            plot_values[representation]["bias_high"].append(bias_high)
            plot_values[representation]["response"].append(response_value)
            plot_values[representation]["response_low"].append(response_low)
            plot_values[representation]["response_high"].append(response_high)
            table.append({
                "observable": name,
                "estimator": representation,
                "truth_bin": bin_index,
                "truth_bin_low": float(low),
                "truth_bin_high": float(high),
                "truth_mean": center,
                "events": len(selected_events),
                "objects": len(local_rows),
                "bias": bias,
                "bias_bootstrap_16": bias_low,
                "bias_bootstrap_84": bias_high,
                "response_definition": "mean(reco/truth) within bin" if positive else "mean(reco) versus mean(truth) within bin",
                "response": response_value,
                "response_reference": response_reference,
                "response_bootstrap_16": response_low,
                "response_bootstrap_84": response_high,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            })
    for representation in ("one_draw", "posterior_mean"):
        key = "draw" if representation == "one_draw" else "mean"
        centers = used_centers[representation]
        point = plot_values[representation]
        bias_ax.errorbar(
            centers,
            point["bias"],
            yerr=np.asarray([np.asarray(point["bias"]) - point["bias_low"], np.asarray(point["bias_high"]) - point["bias"]]),
            color=COLORS[key],
            marker="o",
            linestyle="-",
            capsize=2,
            label=LABELS[key],
        )
        if positive:
            response_ax.errorbar(
                centers,
                point["response"],
                yerr=np.asarray([np.asarray(point["response"]) - point["response_low"], np.asarray(point["response_high"]) - point["response"]]),
                color=COLORS[key],
                marker="o",
                linestyle="-",
                capsize=2,
                label=LABELS[key],
            )
        else:
            response_ax.errorbar(
                centers,
                point["response"],
                yerr=np.asarray([np.asarray(point["response"]) - point["response_low"], np.asarray(point["response_high"]) - point["response"]]),
                color=COLORS[key],
                marker="o",
                linestyle="-",
                capsize=2,
                label=LABELS[key],
            )
    bias_ax.axhline(0.0, color="#515963", linewidth=1.0)
    if positive:
        response_ax.axhline(1.0, color="#515963", linewidth=1.0)
        response_ax.set_ylabel(r"Mean response $\langle O_{reco}/O_{truth}\rangle$")
    else:
        low = float(edges[0]); high = float(edges[-1])
        response_ax.plot((low, high), (low, high), color="#515963", linewidth=1.0, label="Ideal response")
        response_ax.set_ylabel(r"Mean reconstructed $y_{t\bar t}$")
    unit = f" [{spec['unit']}]" if spec["unit"] else ""
    bias_ax.set_ylabel(f"Mean residual bias{unit}")
    response_ax.set_xlabel(f"Parton truth {spec['label']}")
    for axis in (bias_ax, response_ax):
        decorate_axis(axis, grid="both")
    format_legend(bias_ax, ncol=2, loc="best", fontsize=9.5)
    format_legend(response_ax, ncol=2, loc="best", fontsize=9.5)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.98, hspace=0.08)
    return save_figure(fig, output / f"{name}_differential_bias_response",), table


def _migration_values(truth: np.ndarray, reco: np.ndarray, *, representation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = _finite(truth, "migration truth")
    reco = _finite(reco, "migration reconstruction")
    n_events = len(truth)
    objects = int(np.prod(truth.shape[1:])) if truth.ndim > 1 else 1
    if representation == "posterior_distribution":
        if reco.ndim != truth.ndim + 1 or reco.shape[0] != n_events or reco.shape[2:] != truth.shape[1:]:
            raise ValueError("Posterior migration values must align as [N,K,...] with truth [N,...]")
        k = reco.shape[1]
        truth_values = np.broadcast_to(truth[:, None, ...], reco.shape).reshape(-1)
        reco_values = reco.reshape(-1)
        weights = np.full(reco_values.size, 1.0 / (k * objects), dtype=np.float64)
    else:
        if reco.shape != truth.shape:
            raise ValueError(f"{representation} migration values {reco.shape} do not align with truth {truth.shape}")
        truth_values, reco_values = truth.reshape(-1), reco.reshape(-1)
        weights = np.full(reco_values.size, 1.0 / objects, dtype=np.float64)
    event_weight = weights.reshape(n_events, -1).sum(axis=1)
    if not np.allclose(event_weight, 1.0, atol=1e-14, rtol=0.0):
        raise ValueError(f"{representation} migration weights do not close to one per event")
    return truth_values, reco_values, weights, event_weight


def _migration_matrix(truth: np.ndarray, reco: np.ndarray, edges: np.ndarray, *, representation: str, label: str) -> dict[str, Any]:
    truth_values, reco_values, weights, event_weight = _migration_values(truth, reco, representation=representation)
    edges = _finite(edges, f"{label} migration edges")
    outside = (truth_values < edges[0]) | (truth_values > edges[-1]) | (reco_values < edges[0]) | (reco_values > edges[-1])
    if outside.any():
        raise ValueError(f"{label}: {int(outside.sum())} truth/reconstruction values fall outside migration edges")
    raw, _, _ = np.histogram2d(truth_values, reco_values, bins=(edges, edges), weights=weights)
    total = float(weights.sum())
    if not np.isclose(raw.sum(), total, atol=1e-9 * max(1.0, total), rtol=0.0):
        raise ValueError(f"{label}: migration weight {raw.sum()} does not close to expected event weight {total}")
    # x is truth and y is reconstruction; columns therefore condition on truth bin.
    matrix = (raw.T / np.where(raw.sum(axis=1) > 0.0, raw.sum(axis=1), 1.0)[None, :]) * 100.0
    column_sums = matrix.sum(axis=0)
    truth_support = raw.sum(axis=1)
    active = truth_support > 0.0
    max_column_error = float(np.max(np.abs(column_sums[active] - 100.0))) if active.any() else float("inf")
    return {
        "raw": raw.T,
        "matrix_percent": matrix,
        "truth_support_weight": truth_support,
        "column_sums_percent": column_sums,
        "max_column_normalization_error": max_column_error,
        "max_event_weight_closure_error": float(np.max(np.abs(event_weight - 1.0))),
        "total_weight": total,
        "edges": edges,
    }


def _bin_labels(edges: np.ndarray) -> list[str]:
    labels = []
    for low, high in pairwise(edges):
        labels.append(f"{low:g}–{high:g}")
    return labels


def _plot_migration(output: Path, observable: str, representation: str, matrix: dict[str, Any]) -> list[Path]:
    edges = matrix["edges"]
    labels = _bin_labels(edges)
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    image = ax.imshow(matrix["matrix_percent"], origin="lower", cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel(f"Parton truth {observable}")
    ax.set_ylabel(f"Reconstructed {observable}")
    decorate_axis(ax, grid=None)
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Reconstruction probability within truth bin [%]")
    stem = f"{observable}_j_{representation}_migration"
    return save_figure(fig, output / stem)


def _plot_migration_grid(output: Path, observable: str, matrices: dict[str, dict[str, Any]]) -> list[Path]:
    edges = matrices["one_draw"]["edges"]
    labels = _bin_labels(edges)
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 6.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.875, bottom=0.30, top=0.88, wspace=0.12)
    image = None
    for ax, representation in zip(axes, ("one_draw", "posterior_mean", "posterior_distribution"), strict=True):
        image = ax.imshow(matrices[representation]["matrix_percent"], origin="lower", cmap="YlGnBu", vmin=0.0, vmax=100.0, aspect="auto")
        ax.set_title({"one_draw": "J one draw", "posterior_mean": "J posterior mean", "posterior_distribution": "J full posterior"}[representation])
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels)
        ax.tick_params(axis="both", labelsize=9)
        ax.set_xlabel(f"Parton truth {observable}")
        decorate_axis(ax, grid=None)
    axes[0].set_ylabel(f"Reconstructed {observable}")
    if image is not None:
        colorbar = fig.colorbar(image, cax=fig.add_axes((0.91, 0.30, 0.018, 0.56)))
        colorbar.set_label(r"$P(\mathrm{reco\ bin}\mid\mathrm{truth\ bin})$ [%]", fontsize=10)
        colorbar.ax.tick_params(labelsize=9)
    return save_figure(fig, output / f"{observable}_j_migration_comparison", tight_layout=False)


def _plot_one_draw_ensemble(
    output: Path,
    event_ids: np.ndarray,
    truth: np.ndarray,
    samples: np.ndarray,
    edges: np.ndarray,
    *,
    draw_seed: int,
    repeats: int,
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    if samples.ndim != 2 or truth.shape != (samples.shape[0],) or len(event_ids) != len(truth):
        raise ValueError("mttbar one-draw ensemble expects [N,K], [N], and aligned EVENT_ID")
    n_events, k = samples.shape
    seed_list = [draw_seed + index + 1 for index in range(repeats)]
    choices = np.stack([event_keyed_indices(event_ids, k, seed=seed, repeats=1)[0] for seed in seed_list])
    rows = np.arange(n_events)
    pseudo_values = samples[rows[None, :], choices]
    histograms = np.empty((repeats, len(edges) - 1), dtype=np.float64)
    for repeat, pseudo in enumerate(pseudo_values):
        counts, _ = _histogram(pseudo, np.ones(n_events), edges, label=f"mttbar pseudo-dataset {repeat}")
        histograms[repeat] = counts / n_events / np.diff(edges)
    posterior_values, posterior_weights = _flatten_population(samples, truth, representation="posterior_distribution")
    _, posterior_density = _histogram(posterior_values, posterior_weights, edges, label="mttbar full posterior ensemble")
    event_bin_probabilities = np.zeros((n_events, len(edges) - 1), dtype=np.float64)
    for sample_index in range(k):
        local_bins = np.searchsorted(edges, samples[:, sample_index], side="right") - 1
        local_bins[samples[:, sample_index] == edges[-1]] = len(edges) - 2
        np.add.at(event_bin_probabilities, (np.arange(n_events), local_bins), 1.0 / k)
    expected_density = event_bin_probabilities.mean(axis=0) / np.diff(edges)
    if not np.allclose(expected_density, posterior_density, atol=1e-13, rtol=0.0):
        raise ValueError("Event-normalised posterior spectrum differs from the per-event draw expectation")
    ensemble_mean = histograms.mean(axis=0)
    ensemble_se = np.sqrt(np.sum(event_bin_probabilities * (1.0 - event_bin_probabilities), axis=0)) / n_events / math.sqrt(repeats) / np.diff(edges)
    pulls = np.divide(ensemble_mean - posterior_density, ensemble_se, out=np.zeros_like(ensemble_mean), where=ensemble_se > 0.0)
    zero_variance_bad = (ensemble_se == 0.0) & (np.abs(ensemble_mean - posterior_density) > 1e-12)
    if zero_variance_bad.any() or (np.abs(pulls) > 5.0).any():
        raise ValueError(f"One-draw mttbar ensemble does not close to the posterior mean spectrum (max pull {np.max(np.abs(pulls)):.3g})")
    quantiles = np.quantile(histograms, (0.025, 0.16, 0.5, 0.84, 0.975), axis=0)
    table = []
    for index, (low, high) in enumerate(pairwise(edges)):
        table.append({
            "bin_low_GeV": float(low),
            "bin_high_GeV": float(high),
            "full_posterior_density": float(posterior_density[index]),
            "one_draw_mean_density": float(ensemble_mean[index]),
            "one_draw_ensemble_standard_error": float(ensemble_se[index]),
            "one_draw_q025_density": float(quantiles[0, index]),
            "one_draw_q16_density": float(quantiles[1, index]),
            "one_draw_median_density": float(quantiles[2, index]),
            "one_draw_q84_density": float(quantiles[3, index]),
            "one_draw_q975_density": float(quantiles[4, index]),
            "mean_minus_posterior_pull": float(pulls[index]),
        })
    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.stairs(posterior_density, edges, color=COLORS["posterior"], linewidth=2.4, label="J full posterior")
    uncertainty_band(ax, edges, quantiles[1], quantiles[3], color=COLORS["draw"], alpha=0.22)
    ax.plot(centers, quantiles[2], color=COLORS["draw"], linewidth=1.7, label="One-draw median")
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    ax.set_xlabel(r"$m_{t\bar t}$ [GeV]")
    ax.set_ylabel(r"Normalised event density [GeV$^{-1}$]")
    decorate_axis(ax, grid="y")
    format_legend(ax, ncol=2, loc="upper right", fontsize=10)
    files = save_figure(fig, output / "mttbar_one_draw_ensemble")
    summary = {
        "pseudo_datasets": int(repeats),
        "pseudo_dataset_seeds": seed_list,
        "posterior_samples_per_event": int(k),
        "events": int(n_events),
        "max_abs_mean_minus_posterior_pull": float(np.max(np.abs(pulls))),
        "max_abs_density_difference": float(np.max(np.abs(ensemble_mean - posterior_density))),
        "agreement_threshold_sigma": 5.0,
        "closure_passed": True,
        "interpretation": "The mean histogram over event-keyed one-draw pseudo-datasets agrees with the event-normalised full-posterior spectrum within finite pseudo-experiment fluctuations.",
    }
    return files, table, summary


def _plot_factorisation(
    output: Path,
    models: dict[str, dict[str, Any]],
    truth: np.ndarray,
    edges: np.ndarray,
    *,
    representation: str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    truth_values, truth_weights = _flatten_population(truth, truth, representation="posterior_mean")
    truth_counts, truth_density = _histogram(truth_values, truth_weights, edges, label="factorisation Parton truth")
    fig, main, lower = ratio_axes()
    main.fill_between(edges, 0.0, np.r_[truth_density, truth_density[-1]], step="post", color=COLORS["truth"], alpha=0.45, edgecolor="none", label=LABELS["truth"])
    ratios, colors, table = [], [], []
    for model in ("M", "C", "J"):
        values = models[model]["observables"]["mttbar"]
        if representation == "posterior_mean":
            values = np.mean(values, axis=1, dtype=np.float64)
        flat, weights = _flatten_population(values, truth, representation=representation)
        counts, density = _histogram(flat, weights, edges, label=f"{model} {representation}")
        main.stairs(density, edges, color=COLORS[model], linewidth=2.1, label=LABELS[model])
        ratio, _ = plot_histogram_ratio(lower, counts, truth_counts, edges, method=model, min_reference_events=MIN_RATIO_SUPPORT, label=LABELS[model])
        ratios.append(ratio); colors.append(COLORS[model])
        table.append({"observable": "mttbar", "representation": representation, "model": model, **_spectrum_metrics(truth, values, representation=representation)})
    centers = 0.5 * (edges[:-1] + edges[1:])
    controlled_ratio_ylim(lower, ratios, centers, colors=colors, center=1.0)
    main.set_ylabel(r"Normalised event density [GeV$^{-1}$]")
    main.set_xlim(float(edges[0]), float(edges[-1]))
    lower.set_xlabel(r"$m_{t\bar t}$ [GeV]")
    decorate_axis(main, grid="y"); decorate_axis(lower, grid="y")
    format_legend(main, ncol=2, loc="upper right", fontsize=10)
    files = save_figure(fig, output / f"mttbar_factorisation_{representation}")
    return files, table


def _plot_migration_matrix_csv(
    cell_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    observable: str,
    representation: str,
    matrix: dict[str, Any],
) -> None:
    edges = matrix["edges"]
    labels = _bin_labels(edges)
    values = matrix["matrix_percent"]
    raw = matrix["raw"]
    for reco_bin, reco_label in enumerate(labels):
        for truth_bin, truth_label in enumerate(labels):
            cell_rows.append({
                "observable": observable,
                "representation": representation,
                "truth_bin": truth_bin,
                "truth_label": truth_label,
                "truth_low": float(edges[truth_bin]),
                "truth_high": float(edges[truth_bin + 1]),
                "reco_bin": reco_bin,
                "reco_label": reco_label,
                "reco_low": float(edges[reco_bin]),
                "reco_high": float(edges[reco_bin + 1]),
                "weighted_migration_entries": float(raw[reco_bin, truth_bin]),
                "conditional_probability_percent": float(values[reco_bin, truth_bin]),
            })
    for index, (low, high) in enumerate(pairwise(edges)):
        summary_rows.append({
            "observable": observable,
            "representation": representation,
            "truth_bin": index,
            "truth_bin_low": float(low),
            "truth_bin_high": float(high),
            "truth_weight": float(matrix["truth_support_weight"][index]),
            "conditional_column_sum_percent": float(matrix["column_sums_percent"][index]),
            "max_column_normalization_error_percent": matrix["max_column_normalization_error"],
            "max_event_weight_closure_error": matrix["max_event_weight_closure_error"],
            "total_event_weight": matrix["total_weight"],
        })


def _plot_core_summary(path: Path, rows: list[dict[str, Any]], calibration: list[dict[str, Any]], k: int, n: int) -> None:
    lines = [
        "# KNOT posterior-usage plotting summary",
        "",
        f"The evaluation used {n:,} aligned events and K={k} stored posterior draws/event from the frozen KNOT-J posterior.",
        "One draw is selected once per event from EVENT_ID and seed 42011; the same sample index is used for all coupled J quantities. Posterior means are E[O(y)] computed after deriving each observable from each sample. Full-posterior population plots use weight 1/K per sample and ordinary residual functions reject this representation.",
        "",
        "## Point estimators",
        "",
        "| Observable | Estimator | Bias | Residual sigma | RMSE | MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['observable']} | {row['estimator']} | {row['bias']:.5g} | {row['residual_sigma']:.5g} | {row['rmse']:.5g} | {row['mae']:.5g} |")
    lines.extend(("", "## Calibration summary", "", "Coverage and rank tables are in `tables/calibration_metrics.csv`; figures are in `plots/calibration/` and `plots/spin/`."))
    for observable in (
        "mttbar",
        "top_pt",
        "neutrino_pz",
        "spin/cos_opening",
        "spin/cos_theta_plus",
        "spin/cos_theta_minus",
        "spin/helicity_product",
    ):
        selected = [row for row in calibration if row.get("diagnostic") == "coverage" and row["observable"] == observable and row["nominal_credibility"] in (0.68, 0.9)]
        if selected:
            shown = ", ".join(f"{row['nominal_credibility']:.0%} nominal → {row['empirical_coverage']:.3f} empirical" for row in selected)
            lines.append(f"- {observable}: {shown}.")
    lines.extend(("", "## Files", "", "Figures are saved as PNG and PDF. Numerical spectrum, residual, differential-response, migration, pseudo-dataset, and calibration results are in `tables/`. Full input hashes, draw choices, posterior weights, bin support, and closure checks are in `metadata/`."))
    path.write_text("\n".join(lines) + "\n")


def _score_sample_matrix(values: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = _finite(values, "proper-score posterior samples")
    truth = _finite(truth, "proper-score truth")
    if values.ndim != truth.ndim + 1 or values.shape[0] != truth.shape[0] or values.shape[2:] != truth.shape[1:]:
        raise ValueError(f"Proper-score arrays do not align: samples {values.shape}, truth {truth.shape}")
    matrix = np.moveaxis(values, 1, -1).reshape(-1, values.shape[1])
    return matrix, truth.reshape(-1)


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.posterior.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing KNOT posterior-usage output: {output}")
    if not 100 <= args.pseudo_datasets <= 200:
        raise ValueError("The one-draw closure requires between 100 and 200 pseudo-datasets")
    if args.draw_seed < 0:
        raise ValueError("Draw seed must be a non-negative integer")

    j = _load_posterior(input_path, expected_model="J", include_spin=True)
    if str(j["attrs"].get("sample_semantics", "")) != "coherent_event_hypothesis":
        raise ValueError(f"{input_path}: KNOT-J must store coherent event hypotheses to select one draw")
    if bool(j["attrs"].get("sealed_test_used", False)):
        raise ValueError(f"{input_path}: posterior declares sealed-test use; refusing this evaluation")
    event_ids = j["event_ids"]
    n_events, k = len(event_ids), int(j["sample_count"])
    if k <= 0:
        raise ValueError(f"{input_path}: posterior has no samples/event")
    model_root = input_path.parent.parent
    models = {"J": j}
    for model in ("M", "C"):
        path = model_root / model / "posterior.h5"
        models[model] = _load_posterior(path, expected_model=model)
        other = models[model]
        if not np.array_equal(other["event_ids"], event_ids):
            raise ValueError(f"{path}: EVENT_ID order differs from KNOT-J")
        if not np.array_equal(other["truth_parents"], j["truth_parents"]):
            raise ValueError(f"{path}: parent truth differs from KNOT-J on aligned EVENT_ID rows")
        if not np.array_equal(other["truth_neutrinos"], j["truth_neutrinos"]):
            raise ValueError(f"{path}: neutrino truth differs from KNOT-J on aligned EVENT_ID rows")
        if str(other["attrs"].get("spin_definition", "")) != str(j["attrs"].get("spin_definition", "")):
            raise ValueError(f"{path}: spin definition differs from KNOT-J")

    choices = event_keyed_indices(event_ids, k, seed=args.draw_seed, repeats=1)[0]
    replay = event_keyed_indices(event_ids, k, seed=args.draw_seed, repeats=1)[0]
    permuted_choices = event_keyed_indices(event_ids[::-1], k, seed=args.draw_seed, repeats=1)[0]
    draw_hash = _draw_choice_hash(event_ids, choices)
    if not np.array_equal(choices, replay):
        raise ValueError("Event-keyed one-draw choices did not reproduce exactly")
    if not np.array_equal(permuted_choices, choices[::-1]):
        raise ValueError("One-draw choices changed when EVENT_ID rows were permuted")
    if _draw_choice_hash(event_ids[::-1], permuted_choices) != draw_hash:
        raise ValueError("Draw-selection hash depends on input row order")

    truth_parent_obs = _parent_observables(j["truth_parents"], "KNOT-J truth")
    truth_neutrino_obs = _neutrino_observables(j["truth_neutrinos"], "KNOT-J truth")
    for model in models.values():
        model["observables"] = _parent_observables(model["parent_samples"], str(model["path"]))
    j_parent_values = {
        name: _representation_values(samples, choices)
        for name, samples in j["observables"].items()
    }
    j_neutrino_observables = _neutrino_observables(j["neutrino_samples"], "KNOT-J")
    j_neutrino_values = {
        name: _representation_values(samples, choices)
        for name, samples in j_neutrino_observables.items()
    }
    j_spin_values = {
        name: _representation_values(samples, choices)
        for name, samples in j["spin_samples"].items()
    }
    if j["sample_index"][np.arange(n_events), choices].shape != (n_events,):
        raise ValueError("One-draw sample indices did not yield exactly one hypothesis/event")

    for name, values in {**j_parent_values, **j_neutrino_values}.items():
        truth = truth_parent_obs[name] if name in truth_parent_obs else truth_neutrino_obs[name]
        if values["one_draw"].shape != truth.shape or values["posterior_mean"].shape != truth.shape:
            raise ValueError(f"{name}: draw and posterior mean must contain one value per event/object")
        try:
            ordinary_event_residuals(
                values["posterior_distribution"],
                truth,
                representation="posterior_distribution",
                label=name,
            )
        except ValueError:
            pass
        else:
            raise ValueError(f"{name}: full posterior unexpectedly passed ordinary residual validation")
    for name, values in j_spin_values.items():
        truth = j["spin_truth"][name]
        if values["one_draw"].shape != truth.shape or values["posterior_mean"].shape != truth.shape:
            raise ValueError(f"spin/{name}: draw and posterior mean must contain exactly one value per event")
        try:
            ordinary_event_residuals(
                values["posterior_distribution"],
                truth,
                representation="posterior_distribution",
                label=f"spin/{name}",
            )
        except ValueError:
            pass
        else:
            raise ValueError(f"spin/{name}: full posterior unexpectedly passed ordinary residual validation")

    output.mkdir(parents=True)
    dirs = {
        key: output / "plots" / key
        for key in ("core", "migration", "neutrino", "calibration", "factorisation", "spin")
    }
    for directory in (*dirs.values(), output / "tables", output / "metadata"):
        directory.mkdir(parents=True, exist_ok=True)

    input_manifest = {
        "primary_model": "J",
        "events": int(n_events),
        "posterior_samples_per_event": int(k),
        "event_id_sha256_ordered_little_endian_int64": _event_id_hash(event_ids),
        "truth_role_order": ["top", "antitop"],
        "neutrino_role_order": ["top.nu", "antitop.nu"],
        "four_vector_order": ["E", "px", "py", "pz"],
        "units": "GeV; rapidity dimensionless",
        "spin_truth_benchmark": "Hybrid closure: after-FSR truth parents with measured charge-ordered leptons; generator-level lepton truth is unavailable.",
        "spin_definition": json.loads(str(j["attrs"]["spin_definition"])),
        "models": {
            model: {
                "path": str(data["path"]),
                "size_bytes": data["path"].stat().st_size,
                "sha256": _sha256(data["path"]),
                "factorisation": str(data["attrs"].get("factorisation", "")),
                "sample_semantics": str(data["attrs"].get("sample_semantics", "")),
                "posterior_semantics": str(data["attrs"].get("posterior_semantics", "")),
                "posterior_samples_per_event": int(data["sample_count"]),
            }
            for model, data in models.items()
        },
    }
    _write_json(output / "metadata/input_manifest.json", input_manifest)
    _write_json(output / "metadata/draw_selection.json", {
        "draw_seed": int(args.draw_seed),
        "algorithm": "PCFM SplitMix64 event-keyed uniform, floor(u*K), version 1",
        "algorithm_version": "splitmix64_event_id_seed_stream_floor_v1",
        "key": "(fixed seed, EVENT_ID, stream)",
        "posterior_samples_per_event": int(k),
        "events": int(n_events),
        "draw_selection_sha256_order_independent_event_id_index_pairs": draw_hash,
        "event_id_sha256_ordered_little_endian_int64": _event_id_hash(event_ids),
        "row_order_invariance_validated": True,
        "same_choices_replay_validated": True,
        "coupled_hypothesis_policy": "The one selected sample index is shared across parent four-vectors, neutrino four-vectors, and all derived observables for each EVENT_ID.",
        "pseudo_dataset_count": int(args.pseudo_datasets),
        "pseudo_dataset_seeds": [int(args.draw_seed + index + 1) for index in range(args.pseudo_datasets)],
    })

    point_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    spectrum_hist_rows: list[dict[str, Any]] = []
    residual_hist_rows: list[dict[str, Any]] = []
    differential_rows: list[dict[str, Any]] = []
    migration_summary_rows: list[dict[str, Any]] = []
    migration_cell_rows: list[dict[str, Any]] = []
    plot_paths: list[str] = []
    ratio_support: dict[str, Any] = {}

    all_specs = {**PARENT_OBSERVABLES, **NEUTRINO_OBSERVABLES}
    all_truth = {**truth_parent_obs, **truth_neutrino_obs}
    all_values = {**j_parent_values, **j_neutrino_values}
    for name, spec in all_specs.items():
        truth = all_truth[name]
        values = all_values[name]
        for representation in ("one_draw", "posterior_mean"):
            residual = ordinary_event_residuals(values[representation], truth, representation=representation, label=name)
            point_rows.append({
                "observable": name,
                "estimator": representation,
                "events": int(n_events),
                "objects": int(residual.size),
                "unit": spec["unit"],
                **_residual_statistics(residual),
            })
        for representation in ("one_draw", "posterior_mean", "posterior_distribution"):
            metric = _spectrum_metrics(truth, values[representation], representation=representation)
            spectrum_rows.append({
                "family": "posterior_usage",
                "observable": name,
                "representation": representation,
                "events": int(n_events),
                "posterior_samples_per_event": int(k) if representation == "posterior_distribution" else "",
                "unit": spec["unit"],
                **metric,
            })
        sample_matrix, truth_flat = _score_sample_matrix(values["posterior_distribution"], truth)
        calibration_rows.append({
            "diagnostic": "proper_score",
            "observable": name,
            "representation": "posterior_distribution",
            "score": "CRPS",
            "score_value": _crps(sample_matrix, truth_flat),
            "objects": len(truth_flat),
            "posterior_samples_per_event": int(k),
        })
        calibration_rows.extend(_coverage_rows(values["posterior_distribution"], truth, observable=name))

        plot_group = "neutrino" if name in NEUTRINO_OBSERVABLES else "core"
        shape_files, support, shape_rows = _plot_shape(dirs[plot_group], name, spec, truth, values)
        plot_paths.extend(map(str, shape_files))
        ratio_support[name] = support
        spectrum_hist_rows.extend(shape_rows)
        residual_files, _, hist_rows = _plot_residual(dirs[plot_group], name, spec, truth, values)
        plot_paths.extend(map(str, residual_files))
        residual_hist_rows.extend(hist_rows)
        if name in PARENT_OBSERVABLES:
            differential_files, local_differential = _plot_differential(
                dirs["core"], name, spec, truth, values, seed=args.draw_seed + 80
            )
            plot_paths.extend(map(str, differential_files))
            differential_rows.extend(local_differential)

    for name, spec in SPIN_OBSERVABLES.items():
        truth = j["spin_truth"][name]
        values = j_spin_values[name]
        for representation in ("one_draw", "posterior_mean"):
            residual = ordinary_event_residuals(
                values[representation], truth, representation=representation, label=f"spin/{name}"
            )
            point_rows.append({
                "observable": f"spin/{name}",
                "estimator": representation,
                "events": int(n_events),
                "objects": int(residual.size),
                "unit": spec["unit"],
                **_residual_statistics(residual),
            })
        for representation in ("one_draw", "posterior_mean", "posterior_distribution"):
            spectrum_rows.append({
                "family": "posterior_usage",
                "observable": f"spin/{name}",
                "representation": representation,
                "events": int(n_events),
                "posterior_samples_per_event": int(k) if representation == "posterior_distribution" else "",
                "unit": spec["unit"],
                **_spectrum_metrics(truth, values[representation], representation=representation),
            })
        sample_matrix, truth_flat = _score_sample_matrix(values["posterior_distribution"], truth)
        spin_observable_name = f"spin/{name}"
        calibration_rows.append({
            "diagnostic": "proper_score",
            "observable": spin_observable_name,
            "representation": "posterior_distribution",
            "score": "CRPS",
            "score_value": _crps(sample_matrix, truth_flat),
            "objects": len(truth_flat),
            "posterior_samples_per_event": int(k),
        })
        calibration_rows.extend(
            _coverage_rows(values["posterior_distribution"], truth, observable=spin_observable_name)
        )
        shape_files, support, shape_rows = _plot_shape(
            dirs["spin"], f"spin_{name}", spec, truth, values
        )
        plot_paths.extend(map(str, shape_files))
        ratio_support[spin_observable_name] = support
        spectrum_hist_rows.extend(shape_rows)
        residual_files, _, hist_rows = _plot_residual(
            dirs["spin"], f"spin_{name}", spec, truth, values
        )
        plot_paths.extend(map(str, residual_files))
        residual_hist_rows.extend(hist_rows)
        coverage_files = _plot_coverage(
            dirs["spin"], spin_observable_name, spec["label"], calibration_rows, filename=name
        )
        plot_paths.extend(map(str, coverage_files))

    rank_rows, rank_summary = _rank_rows(
        j_parent_values["mttbar"]["posterior_distribution"],
        truth_parent_obs["mttbar"],
        event_ids,
        seed=args.draw_seed,
    )
    calibration_rows.extend(rank_rows)
    for observable, label in (
        ("mttbar", PARENT_OBSERVABLES["mttbar"]["label"]),
        ("top_pt", PARENT_OBSERVABLES["top_pt"]["label"]),
        ("neutrino_pz", NEUTRINO_OBSERVABLES["neutrino_pz"]["label"]),
    ):
        files = _plot_coverage(dirs["calibration"], observable, label, calibration_rows)
        plot_paths.extend(map(str, files))
    rank_files = _plot_rank(dirs["calibration"], calibration_rows)
    plot_paths.extend(map(str, rank_files))

    mtt_samples = j_parent_values["mttbar"]["posterior_distribution"]
    ensemble_files, ensemble_rows, ensemble_summary = _plot_one_draw_ensemble(
        dirs["core"], event_ids, truth_parent_obs["mttbar"], mtt_samples,
        PARENT_OBSERVABLES["mttbar"]["edges"], draw_seed=args.draw_seed,
        repeats=args.pseudo_datasets,
    )
    plot_paths.extend(map(str, ensemble_files))

    for observable in ("mttbar", "top_pt"):
        spec = PARENT_OBSERVABLES[observable]
        edges = np.asarray(spec["migration_edges"], dtype=np.float64)
        truth = truth_parent_obs[observable]
        matrices = {}
        truth_flat, truth_weights = _flatten_population(truth, truth, representation="posterior_mean")
        truth_support, _ = np.histogram(truth_flat, bins=edges, weights=truth_weights)
        if (truth_support < MIN_RATIO_SUPPORT).any():
            sparse = np.flatnonzero(truth_support < MIN_RATIO_SUPPORT).tolist()
            raise ValueError(f"{observable}: audited migration bins have fewer than {MIN_RATIO_SUPPORT} truth-equivalent events: {sparse}")
        for representation in ("one_draw", "posterior_mean", "posterior_distribution"):
            matrices[representation] = _migration_matrix(
                truth,
                j_parent_values[observable][representation],
                edges,
                representation=representation,
                label=f"{observable} J {representation}",
            )
            matrix = matrices[representation]
            if matrix["max_column_normalization_error"] > 1e-10:
                raise ValueError(f"{observable} {representation}: migration columns do not normalize to 100%")
            stem_rep = {"one_draw": "draw", "posterior_mean": "mean", "posterior_distribution": "posterior"}[representation]
            plot_paths.extend(map(str, _plot_migration(dirs["migration"], observable, stem_rep, matrix)))
            _plot_migration_matrix_csv(
                migration_cell_rows,
                migration_summary_rows,
                observable=observable,
                representation=representation,
                matrix=matrix,
            )
        plot_paths.extend(map(str, _plot_migration_grid(dirs["migration"], observable, matrices)))

    factorisation_rows = []
    for representation in ("posterior_distribution", "posterior_mean"):
        files, rows = _plot_factorisation(
            dirs["factorisation"],
            models,
            truth_parent_obs["mttbar"],
            PARENT_OBSERVABLES["mttbar"]["edges"],
            representation=representation,
        )
        plot_paths.extend(map(str, files))
        factorisation_rows.extend(rows)
    spectrum_rows.extend(factorisation_rows)

    _write_csv(output / "tables/point_metrics.csv", point_rows)
    _write_csv(output / "tables/spectrum_metrics.csv", spectrum_rows)
    _write_csv(output / "tables/calibration_metrics.csv", calibration_rows)
    _write_csv(output / "tables/migration_summary.csv", migration_summary_rows)
    _write_csv(output / "tables/migration_cells.csv", migration_cell_rows)
    _write_csv(output / "tables/spectrum_histograms.csv", spectrum_hist_rows)
    _write_csv(output / "tables/residual_histograms.csv", residual_hist_rows)
    _write_csv(output / "tables/differential_bias_response.csv", differential_rows)
    _write_csv(output / "tables/mttbar_one_draw_ensemble.csv", ensemble_rows)

    migration_validation = {
        observable: {
            representation: {
                "total_weight": matrix["total_weight"],
                "expected_event_weight": float(n_events),
                "maximum_event_weight_closure_error": matrix["max_event_weight_closure_error"],
                "maximum_truth-column_normalization_error_percent": matrix["max_column_normalization_error"],
                "truth_equivalent_support_per_bin": matrix["truth_support_weight"].tolist(),
            }
            for representation, matrix in representations.items()
        }
        for observable, representations in ((name, {rep: _migration_matrix(
            truth_parent_obs[name], j_parent_values[name][rep], PARENT_OBSERVABLES[name]["migration_edges"], representation=rep, label=f"{name} {rep} closure record"
        ) for rep in ("one_draw", "posterior_mean", "posterior_distribution")}) for name in ("mttbar", "top_pt"))
    }
    scalar_weight_error = abs(float(np.full(k, 1.0 / k, dtype=np.float64).sum()) - 1.0)
    particle_weight_error = abs(float(np.full(2 * k, 1.0 / (2 * k), dtype=np.float64).sum()) - 1.0)
    _write_json(output / "metadata/posterior_weight_validation.json", {
        "posterior_samples_per_event": int(k),
        "events": int(n_events),
        "posterior_weights": "w_ik = 1/K; particle observables divide each event's total weight equally over the explicit two-role particle axis",
        "scalar_event_total_weight_min": 1.0,
        "scalar_event_total_weight_max": 1.0,
        "maximum_scalar_event_weight_closure_error": scalar_weight_error,
        "particle_event_total_weight_min": 1.0,
        "particle_event_total_weight_max": 1.0,
        "maximum_particle_event_weight_closure_error": particle_weight_error,
        "draw_and_mean_exactly_one_value_per_event_or_particle": True,
        "full_posterior_rejected_by_ordinary_residual_helper": True,
        "mttbar_one_draw_ensemble_closure": ensemble_summary,
        "migration": migration_validation,
    })
    _write_json(output / "metadata/plotting_specification.json", {
        "primary_study": "KNOT posterior usage; KNOT-J",
        "posterior_samples_per_event": int(k),
        "events": int(n_events),
        "posterior_mean_definition": "For every nonlinear observable O, compute O for each stored posterior sample first and use E[O] = (1/K) sum_k O(y^(k)); do not compute O(E[y]).",
        "one_draw_definition": "Select one event-keyed sample index and reuse it for parents, neutrinos, spin-capable coupled quantities, and all derived observables.",
        "full_posterior_definition": "Use every stored sample with weight 1/K per event. When flattening a two-role particle axis, use 1/(2K) per role-sample so each physics event retains total weight one.",
        "ordinary_residual_policy": "Only one_draw and posterior_mean are accepted; posterior_distribution raises ValueError before any reduction or flattening.",
        "core_observables": {name: {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in spec.items() if key in {"label", "unit", "edges", "migration_edges", "differential_edges", "positive_response"}} for name, spec in PARENT_OBSERVABLES.items()},
        "neutrino_observables": {name: {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in spec.items() if key in {"label", "unit", "edges"}} for name, spec in NEUTRINO_OBSERVABLES.items()},
        "spin_observables": {name: {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in spec.items() if key in {"label", "unit", "edges"}} for name, spec in SPIN_OBSERVABLES.items()},
        "spin_truth_benchmark": "Hybrid closure using after-FSR truth parents and measured charge-ordered leptons; generator-level lepton truth is unavailable.",
        "spin_observable_definition": json.loads(str(j["attrs"]["spin_definition"])),
        "spin_posterior_mean_definition": "The paired spin observables are nonlinear quantities evaluated for each coherent stored posterior hypothesis. The posterior mean is E[O] over those K stored values.",
        "spin_excluded_control": {"delta_phi_ll": "Excluded from posterior-usage plots because it is the measured lab-frame lepton azimuth difference and is identical for every posterior draw; it remains a useful posterior-insensitive control in the existing spin analysis."},
        "ratio_support": ratio_support,
        "minimum_truth_support_for_ratio_and_migration_bins": MIN_RATIO_SUPPORT,
        "differential_minimum_event_support": MIN_DIFFERENTIAL_SUPPORT,
        "differential_event_cluster_bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "one_draw_pseudo_datasets": int(args.pseudo_datasets),
        "migration_convention": "Matrix x-axis is truth and y-axis is reconstruction; every column is normalized to 100% conditional on its truth bin.",
        "visual_style_source": "Port of the established PCFM plotting_style.py helper concepts; no runtime dependency on the PCFM checkout.",
        "plot_files": plot_paths,
    })
    _write_json(output / "metadata/calibration_rank_summary.json", rank_summary)
    _write_json(output / "metadata/one_draw_ensemble_closure.json", ensemble_summary)
    _plot_core_summary(output / "plotting_summary.md", point_rows, calibration_rows, k, n_events)

    return {
        "output": str(output),
        "events": int(n_events),
        "posterior_samples_per_event": int(k),
        "draw_selection_sha256": draw_hash,
        "plots": len(plot_paths),
        "mttbar_one_draw_ensemble_closure": ensemble_summary,
        "rank_summary": rank_summary,
        "migrations": migration_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draw-seed", type=int, default=DRAW_SEED)
    parser.add_argument("--pseudo-datasets", type=int, default=PSEUDO_DATASETS)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
