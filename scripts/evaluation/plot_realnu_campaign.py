#!/usr/bin/env python3
"""Make paired, truth-joined plots for the completed realnu production campaign."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "Q0": "No-query reconstruction baseline",
    "Q1": "Topology-query reconstruction",
    "PCFM_Q": "Direct parent-flow baseline",
    "J0": "Joint multitask parent model",
    "J_A": "Assignment-aware parent model",
    "J_AN": "Assignment + neutrino-aware parent model",
}
COLORS = {
    "Q0": "#2878B5",
    "Q1": "#C43C39",
    "PCFM_Q": "#65727E",
    "J0": "#E18A21",
    "J_A": "#27956D",
    "J_AN": "#8053A2",
}
STANDARD = ("Q0", "Q1", "J0", "J_A", "J_AN")
PARENT = ("PCFM_Q", "J0", "J_A", "J_AN")
PARENT_ROLES = ("top", "antitop")
NU_ROLES = ("top.nu", "antitop.nu")
COMPARISONS = (
    ("Q0", "Q1", "No-query → topology-query"),
    ("PCFM_Q", "J0", "Direct parent flow → joint multitask"),
    ("J0", "J_A", "Joint multitask → assignment-aware"),
    ("J_A", "J_AN", "Assignment-aware → assignment + neutrino-aware"),
)


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _attr_json(value, name: str):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    result = json.loads(str(value))
    if not isinstance(result, list):
        raise TypeError(f"{name} must be JSON list metadata")
    return result


def _join(ids: np.ndarray, source_ids: np.ndarray, source: np.ndarray, name: str):
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError(f"{name} contains duplicate EVENT_ID values")
    order = np.argsort(source_ids, kind="stable")
    pos = np.searchsorted(source_ids[order], ids)
    if np.any(pos >= len(order)) or not np.array_equal(source_ids[order[pos]], ids):
        raise ValueError(f"EVENT_ID join failed for {name}")
    return source[order[pos]]


def _p4_observables(p4: np.ndarray) -> dict[str, np.ndarray]:
    p4 = np.asarray(p4, dtype=np.float64)
    e, px, py, pz = np.moveaxis(p4, -1, 0)
    pt = np.hypot(px, py)
    m2 = e * e - px * px - py * py - pz * pz
    scale = np.maximum(e * e + px * px + py * py + pz * pz, 1.0)
    mass = np.where(m2 >= -1.0e-10 * scale, np.sqrt(np.maximum(m2, 0.0)), np.nan)
    eta = np.full_like(pt, np.nan)
    np.arcsinh(
        np.divide(pz, pt, out=np.zeros_like(pz), where=pt > 0), out=eta, where=pt > 0
    )
    y = np.full_like(e, np.nan)
    valid_y = (e + pz > 0) & (e - pz > 0)
    y[valid_y] = 0.5 * np.log((e[valid_y] + pz[valid_y]) / (e[valid_y] - pz[valid_y]))
    return {
        "E": e,
        "px": px,
        "py": py,
        "pz": pz,
        "pT": pt,
        "eta": eta,
        "y": y,
        "phi": np.arctan2(py, px),
        "mass": mass,
    }


def _nu_observables(p3: np.ndarray) -> dict[str, np.ndarray]:
    p3 = np.asarray(p3, dtype=np.float64)
    px, py, pz = np.moveaxis(p3, -1, 0)
    pt = np.hypot(px, py)
    eta = np.full_like(pt, np.nan)
    np.arcsinh(
        np.divide(pz, pt, out=np.zeros_like(pz), where=pt > 0), out=eta, where=pt > 0
    )
    return {
        "px": px,
        "py": py,
        "pz": pz,
        "pT": pt,
        "eta": eta,
        "phi": np.arctan2(py, px),
    }


def _wrap(delta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(delta), np.cos(delta))


def _mass(v: np.ndarray) -> np.ndarray:
    e, px, py, pz = np.moveaxis(v, -1, 0)
    m2 = e * e - px * px - py * py - pz * pz
    scale = np.maximum(e * e + px * px + py * py + pz * pz, 1.0)
    return np.where(m2 >= -1.0e-10 * scale, np.sqrt(np.maximum(m2, 0.0)), np.nan)


def _joint_medoid(
    samples: np.ndarray, *, scale: np.ndarray | None = None
) -> np.ndarray:
    """Select one whole event-level sample minimizing summed pairwise distance."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim < 3:
        raise ValueError("Joint medoid requires [events, samples, ...]")
    n, count = values.shape[:2]
    flat = values.reshape(n, count, -1)
    if scale is None:
        scale = np.nanstd(flat, axis=1)
    scale = np.asarray(scale, dtype=np.float64)
    if scale.ndim == 1:
        scale = np.broadcast_to(scale, (n, flat.shape[-1]))
    scale = np.where(np.isfinite(scale) & (scale > 1.0e-8), scale, 1.0)
    result = np.empty((n, *values.shape[2:]), dtype=values.dtype)
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        local = flat[start:stop] / scale[start:stop, None, :]
        delta = local[:, :, None, :] - local[:, None, :, :]
        distances = np.sqrt(np.square(delta).sum(axis=-1))
        medoid_index = distances.sum(axis=-1).argmin(axis=1)
        result[start:stop] = values[np.arange(start, stop), medoid_index]
    return result


def _energy_score(samples: np.ndarray, truth: np.ndarray, *, chunk: int = 128) -> float:
    """Multivariate energy score in the supplied feature units."""
    x = np.asarray(samples, dtype=np.float64)
    y = np.asarray(truth, dtype=np.float64)
    x = x.reshape(len(x), x.shape[1], -1)
    y = y.reshape(len(y), -1)
    total = 0.0
    count = 0
    for start in range(0, len(x), chunk):
        local = x[start : start + chunk]
        target = y[start : start + chunk]
        first = np.linalg.norm(local - target[:, None, :], axis=-1).mean(axis=1)
        pair = local[:, :, None, :] - local[:, None, :, :]
        second = 0.5 * np.linalg.norm(pair, axis=-1).mean(axis=(1, 2))
        score = first - second
        finite = np.isfinite(score)
        total += float(score[finite].sum())
        count += int(finite.sum())
    return total / count if count else float("nan")


def _scalar_energy_score(samples: np.ndarray, truth: np.ndarray) -> float:
    """Mean scalar energy score (CRPS) using the empirical posterior samples."""
    values = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    finite = np.isfinite(target) & np.isfinite(values).all(axis=1)
    values, target = values[finite], target[finite]
    if not len(values):
        return float("nan")
    count = values.shape[1]
    ordered = np.sort(values, axis=1)
    coefficients = 2 * np.arange(1, count + 1) - count - 1
    half_pair_distance = (ordered * coefficients[None, :]).sum(axis=1) / count**2
    crps = np.abs(values - target[:, None]).mean(axis=1) - half_pair_distance
    return float(crps.mean())


def _bootstrap_delta(
    a: np.ndarray, b: np.ndarray, *, seed: int, draws: int = 2000
) -> dict:
    """Paired bootstrap for metric (b - a), resampling shared events."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    difference = b[keep] - a[keep]
    if not len(difference):
        return {"events": 0, "mean_delta": None, "ci68": None, "ci95": None}
    rng = np.random.default_rng(seed)
    sample_size = len(difference)
    estimates = np.empty(draws, dtype=np.float64)
    batch_size = 32
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        indices = rng.integers(0, len(difference), size=(stop - start, sample_size))
        estimates[start:stop] = difference[indices].mean(axis=1)
    return {
        "events": len(difference),
        "mean_delta": float(difference.mean()),
        "ci68": np.quantile(estimates, [0.16, 0.84]).tolist(),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "bootstrap_draws": draws,
        "bootstrap_sample_size": sample_size,
    }


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "lines.linewidth": 1.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )


def _save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=170)
    plt.close(fig)


def _finite_common(truth: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    valid = np.isfinite(truth)
    for value in arrays.values():
        valid &= np.isfinite(value)
    return valid


def _common_edges(truth: np.ndarray, *, fixed: np.ndarray | None = None) -> np.ndarray:
    if fixed is not None:
        return np.asarray(fixed, dtype=np.float64)
    finite = np.asarray(truth)[np.isfinite(truth)]
    lo, hi = np.quantile(finite, [0.005, 0.995])
    if not hi > lo:
        lo, hi = lo - 1.0, hi + 1.0
    return np.linspace(lo, hi, 41)


def _bin_summary(
    truth: np.ndarray, pred: np.ndarray, edges: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers, bias, resolution, counts = [], [], [], []
    for low, high in itertools.pairwise(edges):
        selected = (
            np.isfinite(truth) & np.isfinite(pred) & (truth >= low) & (truth < high)
        )
        counts.append(int(selected.sum()))
        centers.append(0.5 * (low + high))
        if selected.any():
            residual = pred[selected] - truth[selected]
            bias.append(float(residual.mean()))
            resolution.append(float(residual.std()))
        else:
            bias.append(np.nan)
            resolution.append(np.nan)
    return (
        np.asarray(centers),
        np.asarray(bias),
        np.asarray(resolution),
        np.asarray(counts),
    )


def _plot_scalar_suite(
    output: Path,
    stem: str,
    title: str,
    unit: str,
    truth: np.ndarray,
    samples: dict[str, np.ndarray],
    *,
    perfect: np.ndarray | None = None,
    fixed_edges: np.ndarray | None = None,
    angular: bool = False,
) -> dict:
    """Write spectrum/residual/response/bias/resolution figures for a scalar."""
    sample_mean = {
        name: (
            np.arctan2(
                np.nanmean(np.sin(value), axis=1), np.nanmean(np.cos(value), axis=1)
            )
            if angular
            else np.nanmean(value, axis=1)
        )
        for name, value in samples.items()
    }
    common = np.isfinite(truth)
    for value in sample_mean.values():
        common &= np.isfinite(value)
    truth_c = truth[common]
    if not len(truth_c):
        return {"events": 0}
    sample_c = {name: value[common] for name, value in samples.items()}
    mean_c = {name: value[common] for name, value in sample_mean.items()}
    perfect_valid = common.copy()
    if perfect is not None:
        perfect_valid &= np.isfinite(perfect)
    edges = _common_edges(truth_c, fixed=fixed_edges)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.hist(
        truth_c,
        bins=edges,
        density=True,
        histtype="step",
        color="black",
        linewidth=2.1,
        label="Parton truth",
    )
    if perfect is not None:
        ax.hist(
            perfect[perfect_valid],
            bins=edges,
            density=True,
            histtype="step",
            color="#252525",
            linestyle="--",
            label="Perfect reco",
        )
    for mode, values in sample_c.items():
        ax.hist(
            values.reshape(-1),
            bins=edges,
            density=True,
            histtype="step",
            color=COLORS[mode],
            label=LABELS[mode],
        )
    ax.set(
        title=f"{title}: posterior predictive spectrum",
        xlabel=f"{title} [{unit}]" if unit else title,
        ylabel="Normalized density",
    )
    ax.legend(frameon=False, ncol=1)
    ax.text(
        0.02,
        0.97,
        f"Validation events: {len(truth_c):,}",
        transform=ax.transAxes,
        va="top",
    )
    _save(fig, output / f"{stem}_spectrum")

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    reference_density, _ = np.histogram(truth_c, bins=edges, density=True)
    centers_for_ratio = 0.5 * (edges[:-1] + edges[1:])
    for mode, values in sample_c.items():
        density, _ = np.histogram(values.reshape(-1), bins=edges, density=True)
        ratio = np.divide(
            density,
            reference_density,
            out=np.full_like(density, np.nan),
            where=reference_density > 0,
        )
        ax.step(
            centers_for_ratio,
            ratio,
            where="mid",
            color=COLORS[mode],
            label=LABELS[mode],
        )
    if perfect is not None:
        density, _ = np.histogram(perfect[perfect_valid], bins=edges, density=True)
        ratio = np.divide(
            density,
            reference_density,
            out=np.full_like(density, np.nan),
            where=reference_density > 0,
        )
        ax.step(
            centers_for_ratio,
            ratio,
            where="mid",
            color="#252525",
            linestyle="--",
            label="Perfect reco",
        )
    ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
    ax.set(
        xlabel=f"{title} [{unit}]" if unit else title,
        ylabel="Model / parton truth",
        title=f"{title}: spectrum ratio",
    )
    ax.legend(frameon=False, fontsize=7, ncol=2)
    _save(fig, output / f"{stem}_spectrum_ratio")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    residual_edges = np.linspace(-np.pi, np.pi, 51) if angular else None
    for mode, values in mean_c.items():
        residual = _wrap(values - truth_c) if angular else values - truth_c
        ax.hist(
            residual,
            bins=residual_edges if residual_edges is not None else 51,
            density=True,
            histtype="step",
            color=COLORS[mode],
            label=LABELS[mode],
        )
    if perfect is not None:
        residual = (
            _wrap(perfect[perfect_valid] - truth[perfect_valid])
            if angular
            else perfect[perfect_valid] - truth[perfect_valid]
        )
        ax.hist(
            residual,
            bins=residual_edges if residual_edges is not None else 51,
            density=True,
            histtype="step",
            color="#252525",
            linestyle="--",
            label="Perfect reco − parton",
        )
    ax.axvline(0.0, color="black", linewidth=1, linestyle=":")
    ax.set(
        title=f"{title}: residual of posterior observable mean",
        xlabel=f"Residual [{unit}]" if unit else "Residual",
        ylabel="Normalized density",
    )
    ax.legend(frameon=False)
    _save(fig, output / f"{stem}_residual")
    if perfect is not None:
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        for mode, values in sample_mean.items():
            valid = perfect_valid & np.isfinite(values)
            residual = (
                _wrap(values[valid] - perfect[valid])
                if angular
                else values[valid] - perfect[valid]
            )
            ax.hist(
                residual,
                bins=residual_edges if residual_edges is not None else 51,
                density=True,
                histtype="step",
                color=COLORS[mode],
                label=LABELS[mode],
            )
        ax.axvline(0.0, color="black", linewidth=1, linestyle=":")
        ax.set(
            title=f"{title}: model − perfect-reco residual",
            xlabel=f"Residual [{unit}]" if unit else "Residual",
            ylabel="Normalized density",
        )
        ax.legend(frameon=False, fontsize=7)
        _save(fig, output / f"{stem}_residual_vs_perfect")

    # Each panel is one model; use the same truth and prediction binning.
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.0), squeeze=False)
    for ax, (mode, values) in zip(axes.flat, mean_c.items(), strict=False):
        h = ax.hist2d(truth_c, values, bins=(edges, edges), cmap="Blues", cmin=1)
        ax.plot(
            [edges[0], edges[-1]],
            [edges[0], edges[-1]],
            color="#b22222",
            linestyle="--",
            linewidth=1,
        )
        ax.set(
            title=LABELS[mode],
            xlabel=f"Parton truth [{unit}]" if unit else "Parton truth",
            ylabel=f"Model reco [{unit}]" if unit else "Model reco",
        )
    for ax in axes.flat[len(mean_c) :]:
        ax.axis("off")
    fig.colorbar(h[3], ax=list(axes.flat), label="Events / bin")
    fig.suptitle(f"{title}: response of posterior observable mean")
    _save(fig, output / f"{stem}_response")

    fig, (ax_bias, ax_res) = plt.subplots(1, 2, figsize=(10.5, 4.5))
    metric_rows = {}
    for mode, values in mean_c.items():
        residual = _wrap(values - truth_c) if angular else values - truth_c
        bx, bias, resolution, counts = [], [], [], []
        for low, high in itertools.pairwise(edges):
            selected = (truth_c >= low) & (truth_c < high) & np.isfinite(residual)
            bx.append(0.5 * (low + high))
            counts.append(int(selected.sum()))
            bias.append(float(residual[selected].mean()) if selected.any() else np.nan)
            resolution.append(
                float(residual[selected].std()) if selected.any() else np.nan
            )
        bx, bias, resolution, counts = map(np.asarray, (bx, bias, resolution, counts))
        ax_bias.plot(
            bx, bias, marker="o", markersize=3, color=COLORS[mode], label=LABELS[mode]
        )
        ax_res.plot(
            bx,
            resolution,
            marker="o",
            markersize=3,
            color=COLORS[mode],
            label=LABELS[mode],
        )
        raw_res = residual
        metric_rows[mode] = {
            "events": len(truth_c),
            "bias": float(np.mean(raw_res)),
            "mae": float(np.mean(np.abs(raw_res))),
            "rmse": float(np.sqrt(np.mean(raw_res**2))),
            "resolution": float(np.std(raw_res)),
            "posterior_mean": float(np.mean(values)),
            "truth_mean": float(np.mean(truth_c)),
            "bin_centers": bx.tolist(),
            "bias_by_truth_bin": bias.tolist(),
            "resolution_by_truth_bin": resolution.tolist(),
            "events_by_truth_bin": counts.tolist(),
        }
        posterior = sample_c[mode]
        if angular:
            relative = _wrap(posterior - truth_c[:, None])
            low68, median, high68 = np.quantile(relative, [0.16, 0.5, 0.84], axis=1)
            low90, high90 = np.quantile(relative, [0.05, 0.95], axis=1)
            covered68 = (low68 <= 0.0) & (high68 >= 0.0)
            covered90 = (low90 <= 0.0) & (high90 >= 0.0)
            pit = np.mean(relative <= 0.0, axis=1)
            interval68 = high68 - low68
            interval90 = high90 - low90
            median_residual = median
        else:
            low68, median, high68 = np.quantile(posterior, [0.16, 0.5, 0.84], axis=1)
            low90, high90 = np.quantile(posterior, [0.05, 0.95], axis=1)
            covered68 = (truth_c >= low68) & (truth_c <= high68)
            covered90 = (truth_c >= low90) & (truth_c <= high90)
            pit = np.mean(posterior <= truth_c[:, None], axis=1)
            interval68 = high68 - low68
            interval90 = high90 - low90
            median_residual = median - truth_c
        metric_rows[mode].update(
            {
                "posterior_q16_mean": float(np.mean(low68)),
                "posterior_q50_bias": float(np.mean(median_residual)),
                "posterior_q50_mae": float(np.mean(np.abs(median_residual))),
                "posterior_q50_rmse": float(np.sqrt(np.mean(median_residual**2))),
                "posterior_q84_mean": float(np.mean(high68)),
                "central_68_coverage": float(np.mean(covered68)),
                "central_90_coverage": float(np.mean(covered90)),
                "mean_central_68_width": float(np.mean(interval68)),
                "mean_central_90_width": float(np.mean(interval90)),
                "pit_mean": float(np.mean(pit)),
                "posterior_width_actual_error_correlation": float(
                    np.corrcoef(interval68, np.abs(median_residual))[0, 1]
                )
                if np.std(interval68) and np.std(median_residual)
                else None,
            }
        )
    if perfect is not None:
        residual = (
            _wrap(perfect[perfect_valid] - truth[perfect_valid])
            if angular
            else perfect[perfect_valid] - truth[perfect_valid]
        )
        metric_rows["Perfect reco"] = {
            "events": int(perfect_valid.sum()),
            "bias": float(residual.mean()),
            "mae": float(np.abs(residual).mean()),
            "rmse": float(np.sqrt(np.square(residual).mean())),
            "resolution": float(residual.std()),
        }
        for mode, values in mean_c.items():
            valid = perfect_valid[common] & np.isfinite(values)
            model_residual = (
                _wrap(values[valid] - perfect[common][valid])
                if angular
                else values[valid] - perfect[common][valid]
            )
            metric_rows[mode].update(
                {
                    "model_minus_perfect_events": int(valid.sum()),
                    "model_minus_perfect_bias": float(model_residual.mean())
                    if valid.any()
                    else None,
                    "model_minus_perfect_mae": float(np.abs(model_residual).mean())
                    if valid.any()
                    else None,
                    "model_minus_perfect_rmse": float(
                        np.sqrt(np.square(model_residual).mean())
                    )
                    if valid.any()
                    else None,
                }
            )
    if all(value.shape[1] > 1 for value in sample_c.values()):
        posterior_dir = output.parent / "posterior"
        posterior_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        for mode, posterior in sample_c.items():
            if angular:
                residual_samples = _wrap(posterior - truth_c[:, None])
                q16, q50, q84 = np.quantile(residual_samples, [0.16, 0.5, 0.84], axis=1)
                width = q84 - q16
                error = np.abs(q50)
            else:
                q16, q50, q84 = np.quantile(posterior, [0.16, 0.5, 0.84], axis=1)
                width = q84 - q16
                error = np.abs(q50 - truth_c)
            ax.hist2d(width, error, bins=35, cmap="Blues", cmin=1, alpha=0.18)
            ax.scatter(
                width[:: max(1, len(width) // 3000)],
                error[:: max(1, len(error) // 3000)],
                s=3,
                alpha=0.2,
                color=COLORS[mode],
                label=LABELS[mode],
            )
        ax.set(
            xlabel="Central 68% posterior width",
            ylabel="Absolute posterior q50 error",
            title=f"{title}: posterior width vs actual error",
        )
        ax.legend(frameon=False, fontsize=7)
        _save(fig, posterior_dir / f"{stem}_width_vs_actual_error")
    ax_bias.axhline(0.0, color="black", linewidth=1, linestyle=":")
    ax_res.set_ylim(bottom=0.0)
    ax_bias.set(
        xlabel=f"Parton truth [{unit}]" if unit else "Parton truth",
        ylabel=f"Bias [{unit}]" if unit else "Bias",
        title="Bias vs truth",
    )
    ax_res.set(
        xlabel=f"Parton truth [{unit}]" if unit else "Parton truth",
        ylabel=f"Resolution [{unit}]" if unit else "Resolution",
        title="Resolution vs truth",
    )
    ax_bias.legend(frameon=False, fontsize=7)
    _save(fig, output / f"{stem}_bias_resolution")
    for component, values_by_mode, ylabel, title_suffix in (
        (
            "bias_vs_truth",
            "bias",
            f"Bias [{unit}]" if unit else "Bias",
            "Bias vs truth",
        ),
        (
            "resolution_vs_truth",
            "resolution",
            f"Resolution [{unit}]" if unit else "Resolution",
            "Resolution vs truth",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        for mode, record in metric_rows.items():
            if mode == "Perfect reco":
                continue
            ax.plot(
                record["bin_centers"],
                record[
                    "bias_by_truth_bin"
                    if values_by_mode == "bias"
                    else "resolution_by_truth_bin"
                ],
                marker="o",
                markersize=3,
                color=COLORS[mode],
                label=LABELS[mode],
            )
        if values_by_mode == "bias":
            ax.axhline(0.0, color="black", linewidth=1, linestyle=":")
        else:
            ax.set_ylim(bottom=0.0)
        ax.set(
            xlabel=f"{title} [{unit}]" if unit else title,
            ylabel=ylabel,
            title=f"{title}: {title_suffix}",
        )
        ax.legend(frameon=False, fontsize=7)
        _save(fig, output / f"{stem}_{component}")
    return {
        "events": len(truth_c),
        "unit": unit,
        "metrics": metric_rows,
        "truth_edges": edges.tolist(),
    }


def _load_truth_and_channels(manifest: dict, ids: np.ndarray) -> dict:
    truth_path = Path(manifest["inputs"]["truth_store"])
    measured_path = Path(manifest["inputs"]["measured_input"])
    with h5py.File(truth_path, "r") as truth:
        all_ids = np.asarray(truth["EVENT_ID"][:], dtype=np.int64)
        roles = _attr_json(truth.attrs["invisible_role_order"], "invisible_role_order")
        if roles != list(NU_ROLES):
            raise ValueError(
                f"Truth sidecar neutrino order is {roles}, expected {list(NU_ROLES)}"
            )
        neutrino_struct = _join(
            ids, all_ids, np.asarray(truth["LABELS/NEUTRINO"][:]), "neutrino truth"
        )
        neutrino = np.stack(
            [neutrino_struct[axis] for axis in ("px", "py", "pz")], axis=-1
        ).astype(np.float64)
        nu_support = _join(
            ids,
            all_ids,
            np.asarray(truth["SUPPORT/INVISIBLE_ROLE"][:], dtype=bool),
            "neutrino support",
        )
        parent_order = _attr_json(truth.attrs["partonic_roles"], "partonic_roles")
        if parent_order != list(PARENT_ROLES):
            raise ValueError(
                f"Partonic truth roles are {parent_order}, expected {list(PARENT_ROLES)}"
            )
        parent = np.stack(
            [
                _join(
                    ids,
                    all_ids,
                    np.asarray(truth[f"PARTONIC/{role}"][:], dtype=np.float64),
                    f"{role} truth",
                )
                for role in PARENT_ROLES
            ],
            axis=1,
        )
        parent_support = _join(
            ids,
            all_ids,
            np.asarray(truth["PARTONIC/SUPPORT"][:], dtype=bool),
            "parent support",
        )
        assignment_support = _join(
            ids,
            all_ids,
            np.asarray(
                truth["SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"][:], dtype=bool
            ),
            "assignment support",
        )
    with h5py.File(measured_path, "r") as measured:
        split = np.asarray(measured["SPLIT"][:], dtype=np.uint8)
        measured_ids = np.asarray(measured["EVENT_ID"][:], dtype=np.int64)
        validation_ids = measured_ids[split == 1]
        if len(validation_ids) != 100_342 or len(np.unique(validation_ids)) != len(
            validation_ids
        ):
            raise ValueError(
                "Measured input no longer has the frozen 100342-event validation split"
            )
        if np.any(split == 2):
            raise ValueError(
                "Reserved sealed-test rows are materialised in the measured input"
            )
        globals_val = np.asarray(measured["INPUTS/GLOBAL"][split == 1])
        jets_val = np.asarray(measured["INPUTS/JET"][split == 1])
    if set(map(int, ids)) != set(map(int, validation_ids)):
        raise ValueError(
            "Prediction EVENT_IDs do not exactly match the frozen validation membership"
        )
    globals_joined = _join(
        ids, validation_ids, globals_val, "measured validation globals"
    )
    jets_joined = _join(ids, validation_ids, jets_val, "measured validation jets")
    names = globals_joined.dtype.names or ()
    fields = {name.lower(): name for name in names}
    for required in ("njet", "nelectron", "nmuon", "met_met", "met_phi"):
        if required not in fields:
            raise KeyError(
                f"Measured global inputs lack required {required}; fields={names}"
            )
    raw_source_njet = np.asarray(globals_joined[fields["njet"]], dtype=np.float64)
    if not np.isfinite(raw_source_njet).all() or not np.allclose(
        raw_source_njet, np.rint(raw_source_njet), rtol=0.0, atol=1e-6
    ):
        raise ValueError("Measured source_njet contains non-finite or non-integer values")
    source_njet = np.rint(raw_source_njet).astype(np.int64)
    met_magnitude = np.asarray(globals_joined[fields["met_met"]], dtype=np.float64)
    met_phi = np.asarray(globals_joined[fields["met_phi"]], dtype=np.float64)
    if not np.isfinite(met_magnitude).all() or not np.isfinite(met_phi).all():
        raise ValueError("Measured MET magnitude/phi contains non-finite validation values")
    met = np.stack(
        [met_magnitude * np.cos(met_phi), met_magnitude * np.sin(met_phi)], axis=-1
    )
    model_visible_njet = (
        np.isfinite(jets_joined["pt"]) & (jets_joined["pt"] > 0)
    ).sum(axis=1).astype(np.int64)
    model_max_jets = int(jets_joined.shape[1])
    if model_max_jets != 8:
        raise ValueError(f"Frozen validation input must have eight jet slots, found {model_max_jets}")
    if not np.array_equal(model_visible_njet, np.minimum(source_njet, model_max_jets)):
        bad = np.flatnonzero(
            model_visible_njet != np.minimum(source_njet, model_max_jets)
        )[:10]
        raise ValueError(
            "Measured maxjets8 adapter mismatch: "
            f"events={int((model_visible_njet != np.minimum(source_njet, model_max_jets)).sum())}; "
            f"first_EVENT_IDs={ids[bad].tolist()}; "
            f"source_njet={source_njet[bad].tolist()}; "
            f"model_visible_njet={model_visible_njet[bad].tolist()}"
        )
    if int((source_njet > model_max_jets).sum()) != 164:
        raise ValueError(
            "Frozen maxjets8 truncation population changed: "
            f"expected=164, observed={int((source_njet > model_max_jets).sum())}"
        )
    ne = np.rint(globals_joined[fields["nelectron"]]).astype(np.int64)
    nm = np.rint(globals_joined[fields["nmuon"]]).astype(np.int64)
    channel = np.full(len(ids), "other", dtype=object)
    channel[(ne == 2) & (nm == 0)] = "ee"
    channel[(ne == 1) & (nm == 1)] = "eμ"
    channel[(ne == 0) & (nm == 2)] = "μμ"
    if assignment_support.shape != (len(ids),) or int(assignment_support.sum()) != 100_323:
        raise ValueError(
            "Frozen visible-assignment support changed: "
            f"shape={assignment_support.shape}; supported={int(assignment_support.sum())}"
        )
    if (
        nu_support.shape != (len(ids), 2)
        or not nu_support.all(axis=1).all()
        or parent_support.shape != (len(ids), 2)
        or not parent_support.all(axis=1).all()
    ):
        raise ValueError(
            "Frozen validation neutrino/parent truth support is not complete and role ordered"
        )
    return {
        "neutrino": neutrino,
        "neutrino_support": nu_support,
        "parents": parent,
        "parent_support": parent_support,
        "assignment_support": assignment_support,
        "source_njet": source_njet,
        "model_visible_njet": model_visible_njet,
        "model_max_jets": model_max_jets,
        "channel": channel,
        "met": met,
    }


def _load_standard(prediction_path: Path, expected_ids: np.ndarray) -> dict:
    with h5py.File(prediction_path, "r") as source:
        ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        role_order = _attr_json(source.attrs["invisible_roles"], "invisible_roles")
        if role_order != list(NU_ROLES):
            raise ValueError(f"{prediction_path}: wrong neutrino roles {role_order}")
        root_order = _attr_json(source.attrs["root_order"], "root_order")
        if len(root_order) != 2 or set(root_order) != set(PARENT_ROLES):
            raise ValueError(
                f"{prediction_path}: unsupported reconstruction roots {root_order}"
            )
        root_indices = [root_order.index(role) for role in PARENT_ROLES]
        roots = np.asarray(source["VISIBLE/root_candidate_indices"][:], dtype=np.int64)[
            :, :, root_indices
        ]
        samples = np.asarray(source["INVISIBLE/samples"][:, 0], dtype=np.float32)
        model_visible_njet = np.asarray(source["OBSERVED/njet"][:], dtype=np.int64)
        attr_mode = str(source.attrs.get("split", ""))
    if (
        attr_mode != "validation"
        or roots.shape[0] != len(ids)
        or samples.shape[0] != len(ids)
    ):
        raise ValueError(f"{prediction_path}: malformed non-validation prediction")
    return {
        "ids": ids,
        "roots": _join(expected_ids, ids, roots, f"{prediction_path} assignment"),
        "samples": _join(
            expected_ids, ids, samples, f"{prediction_path} neutrino posterior"
        ),
        "model_visible_njet": _join(
            expected_ids,
            ids,
            model_visible_njet,
            f"{prediction_path} model-visible njet",
        ),
        "posterior_samples": int(samples.shape[1]),
    }


def _load_parent(prediction_path: Path, expected_ids: np.ndarray) -> dict:
    with h5py.File(prediction_path, "r") as source:
        ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        roles = _attr_json(source.attrs["parent_roles"], "parent_roles")
        if roles != list(PARENT_ROLES):
            raise ValueError(f"{prediction_path}: wrong parent roles {roles}")
        samples = np.asarray(source["PARTONIC/parents"][:], dtype=np.float32)
        stored_pair = np.asarray(source["PARTONIC/ttbar"][:], dtype=np.float32)
        observables = {
            name: np.asarray(
                source[f"PARTONIC/observables/{name}"][:], dtype=np.float32
            )
            for name in (
                "cos_theta_star",
                "delta_phi_top_antitop",
                "delta_rapidity_top_antitop",
            )
        }
        conditioning = str(source.attrs.get("conditioning", ""))
        solver = str(source.attrs.get("solver", ""))
        steps = int(source.attrs.get("solver_steps", -1))
        nfe = int(source.attrs.get("nfe", -1))
        parent_samples = int(source.attrs.get("samples_per_event", samples.shape[1]))
        conditioning_contract = json.loads(
            str(source.attrs.get("conditioning_contract", "{}"))
        )
        conditioning_indices = {
            name: np.asarray(source[f"CONDITIONING/{name}"][:], dtype=np.int64)
            for name in (
                "assignment_hypothesis_index",
                "invisible_sample_index",
                "parent_sample_index",
            )
        }
        observables = {
            name: np.asarray(
                source[f"PARTONIC/observables/{name}"][:], dtype=np.float32
            )
            for name in (
                "cos_theta_star",
                "delta_phi_top_antitop",
                "delta_rapidity_top_antitop",
            )
        }
        if not np.allclose(stored_pair, samples.sum(axis=2), rtol=2e-6, atol=3e-4):
            raise ValueError(
                f"{prediction_path}: stored ttbar differs from top + antitop"
            )
        if conditioning == "P2":
            nu_roles = _attr_json(
                source.attrs["invisible_role_order"], "invisible_role_order"
            )
            if nu_roles != list(NU_ROLES):
                raise ValueError("J_AN P2 context is not [top.nu, antitop.nu]")
    return {
        "ids": ids,
        "samples": _join(
            expected_ids, ids, samples, f"{prediction_path} parent posterior"
        ),
        "observables": {
            key: _join(expected_ids, ids, value, f"{prediction_path} {key}")
            for key, value in observables.items()
        },
        "conditioning": conditioning,
        "conditioning_contract": conditioning_contract,
        "conditioning_indices": {
            key: _join(expected_ids, ids, value, f"{prediction_path} {key}")
            for key, value in conditioning_indices.items()
        },
        "solver": solver,
        "steps": steps,
        "nfe": nfe,
        "posterior_samples": parent_samples,
    }


def _load_candidate_vectors(
    manifest: dict, mode: str, expected_ids: np.ndarray
) -> dict:
    """Use the maintained structured collator for truth and assignment p4 vectors."""
    source_root = Path(manifest["source_root"])
    sys.path.insert(0, str(source_root / "src"))
    from joint_event_reconstruction.cli.run import build_data, load_visible_config
    from joint_event_reconstruction.configuration import load_experiment

    entry = manifest["modes"][mode]
    config = load_experiment(entry["config_path"])
    visible = load_visible_config(config)
    config.training.workers = 0
    data = build_data(config, visible, include_truth=True, device="cpu")
    data.setup("predict")
    if data.structured_collator.visible_store is None:
        raise RuntimeError(
            "Oracle assignment decomposition requires the validated visible truth sidecar"
        )
    node_order = list(data.structured_collator.spec.roots)
    if len(node_order) != 2 or set(node_order) != set(PARENT_ROLES):
        raise ValueError(f"Structured truth roots changed: {node_order}")
    ids_parts = []
    truth_index_parts = {role: [] for role in PARENT_ROLES}
    support_parts = {role: [] for role in PARENT_ROLES}
    candidate_parts = {role: [] for role in PARENT_ROLES}
    ownership_reference = {}
    for batch in data.val_dataloader():
        ids_parts.append(batch["event_index"].detach().cpu().numpy().astype(np.int64))
        vectors = batch["physics_vectors"].detach().cpu().numpy().astype(np.float64)
        for role in PARENT_ROLES:
            pool = batch["structured"]["pools"][role]
            ownership = pool["leaf_ownership"].detach().cpu().numpy().astype(np.float64)
            prior = ownership_reference.setdefault(role, ownership)
            if not np.array_equal(prior, ownership):
                raise ValueError(
                    f"{role} candidate ownership changed across validation batches"
                )
            candidate_parts[role].append(
                np.einsum("ct,btd->bcd", ownership, vectors, optimize=True)
            )
            truth_index_parts[role].append(
                pool["truth_index"].detach().cpu().numpy().astype(np.int64)
            )
            support_parts[role].append(
                pool["reconstructable"].detach().cpu().numpy().astype(bool)
            )
    del data, visible
    ids = np.concatenate(ids_parts)
    positions = np.argsort(ids, kind="stable")
    if not np.array_equal(ids[positions], np.sort(expected_ids)):
        raise ValueError(
            "Structured truth collator EVENT_IDs do not match validation membership"
        )
    lookup = {int(event_id): int(position) for position, event_id in enumerate(ids)}
    selected = np.asarray(
        [lookup[int(event_id)] for event_id in expected_ids], dtype=np.int64
    )
    result = {"candidate_vectors": {}, "truth_index": {}, "support": {}}
    for role in PARENT_ROLES:
        candidates = np.concatenate(candidate_parts[role], axis=0)[selected]
        result["candidate_vectors"][role] = candidates
        result["truth_index"][role] = np.concatenate(truth_index_parts[role])[selected]
        result["support"][role] = np.concatenate(support_parts[role])[selected]
    return result


def _gather_candidate(candidates: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.full((len(indices), candidates.shape[-1]), np.nan, dtype=np.float64)
    valid = (indices >= 0) & (indices < candidates.shape[1])
    rows = np.flatnonzero(valid)
    result[valid] = candidates[rows, indices[valid]]
    return result


def _assignment_products(
    output: Path, predictions: dict, truth: dict
) -> tuple[dict, dict]:
    assignment_dir = output / "assignment"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    correctness = {}
    truncation_rows = []
    support = truth["assignment_support"]
    for mode in STANDARD:
        roots = predictions[mode]["roots"][:, 0]
        target = predictions[mode]["candidate_truth_index"]
        role_correct = {}
        complete = support.copy()
        for index, role in enumerate(PARENT_ROLES):
            role_supported = (
                truth["assignment_support"]
                & predictions[mode]["candidate_support"][role]
                & (target[role] >= 0)
            )
            correct = (roots[:, index] == target[role]) & role_supported
            role_correct[role] = correct
            complete &= role_supported
            rows.append(
                {
                    "model": LABELS[mode],
                    "role": role,
                    "supported_events": int(role_supported.sum()),
                    "accuracy": float(correct.sum() / role_supported.sum())
                    if role_supported.any()
                    else None,
                    "null_fraction": float((roots[:, index][role_supported] < 0).mean())
                    if role_supported.any()
                    else None,
                }
            )
        event_correct = role_correct["top"] & role_correct["antitop"] & complete
        predictions[mode]["assignment_supported"] = complete
        predictions[mode]["assignment_complete_correct"] = event_correct
        correctness[mode] = event_correct
        for label, selected in (
            ("source_njet<=8", truth["source_njet"] <= 8),
            ("source_njet>8", truth["source_njet"] > 8),
        ):
            supported = complete & selected
            truncation_rows.append(
                {
                    "model": LABELS[mode],
                    "source_njet_group": label,
                    "selected_events": int(selected.sum()),
                    "supported_events": int(supported.sum()),
                    "support_fraction": float(supported.sum() / max(1, selected.sum())),
                    "complete_event_efficiency": float(event_correct[supported].mean())
                    if supported.any()
                    else None,
                }
            )

    source_njet = truth["source_njet"]
    model_visible_njet = truth["model_visible_njet"]
    if np.any(source_njet < 2) or np.any(model_visible_njet < 2):
        raise ValueError("The frozen dilepton validation contains fewer than two visible jets")
    source_njet_bins = np.arange(2, 10, dtype=np.int64)
    model_visible_bins = np.arange(2, 9, dtype=np.int64)
    assignment_rows = []
    transition_matrices = {}
    role_bin_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for role_index, role in enumerate(PARENT_ROLES):
        for mode in STANDARD:
            roots = predictions[mode]["roots"][:, 0, role_index]
            role_supported = (
                truth["assignment_support"]
                & predictions[mode]["candidate_support"][role]
                & (predictions[mode]["candidate_truth_index"][role] >= 0)
            )
            target_index = predictions[mode]["candidate_truth_index"][role]
            for definition, bin_values, bins in (
                (
                    "source_njet",
                    np.where(source_njet >= 9, 9, source_njet),
                    source_njet_bins,
                ),
                ("model_visible_njet", model_visible_njet, model_visible_bins),
            ):
                for n in bins:
                    selected = bin_values == n
                    supported = role_supported & selected
                    role_bin_rows.append(
                        {
                            "model": LABELS[mode],
                            "role": role,
                            "multiplicity_definition": definition,
                            "multiplicity_bin": ">=9"
                            if definition == "source_njet" and n == 9
                            else int(n),
                            "selected_events": int(selected.sum()),
                            "supported_events": int(supported.sum()),
                            "accuracy": float(
                                np.mean(roots[supported] == target_index[supported])
                            )
                            if supported.any()
                            else None,
                            "null_fraction": float((roots[supported] < 0).mean())
                            if supported.any()
                            else None,
                        }
                    )
            efficiency = []
            source_bins = np.where(source_njet >= 9, 9, source_njet)
            for n in source_njet_bins:
                selected = role_supported & (source_bins == n)
                efficiency.append(
                    float(
                        np.mean(
                            roots[selected]
                            == predictions[mode]["candidate_truth_index"][role][selected]
                        )
                    )
                    if selected.any()
                    else np.nan
                )
            axes[role_index].plot(
                source_njet_bins,
                efficiency,
                marker="o",
                markersize=3,
                color=COLORS[mode],
                label=LABELS[mode],
            )
        axes[role_index].set(
            xlabel="Source nJets (>=9 combined)",
            ylabel="Per-role assignment efficiency",
            title=role,
        )
        axes[role_index].set_xticks(source_njet_bins, [">=9" if n == 9 else str(n) for n in source_njet_bins])
        axes[role_index].set_ylim(0, 1.02)
        axes[role_index].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%")
        )
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Assignment efficiency by role and jet multiplicity")
    _save(fig, assignment_dir / "per_role_efficiency_vs_njet")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    by_source_njet = {}
    by_model_visible_njet = {}
    for mode in STANDARD:
        values, denominators = [], []
        for n in source_njet_bins:
            selected_bin = np.where(source_njet >= 9, 9, source_njet) == n
            supported = predictions[mode]["assignment_supported"] & selected_bin
            denominators.append(int(supported.sum()))
            values.append(float(correctness[mode][supported].mean()) if supported.any() else np.nan)
            assignment_rows.append(
                {
                    "model": LABELS[mode],
                    "multiplicity_definition": "source_njet",
                    "multiplicity_bin": ">=9" if n == 9 else int(n),
                    "selected_events": int(selected_bin.sum()),
                    "supported_events": int(supported.sum()),
                    "complete_event_efficiency": values[-1],
                }
            )
        by_source_njet[mode] = {
            (">=9" if int(n) == 9 else str(int(n))): {
                "selected_events": int((np.where(source_njet >= 9, 9, source_njet) == n).sum()),
                "supported_events": denominators[i],
                "efficiency": float(values[i]) if denominators[i] else None,
            }
            for i, n in enumerate(source_njet_bins)
        }
        ax.plot(
            source_njet_bins,
            values,
            marker="o",
            markersize=3,
            color=COLORS[mode],
            label=LABELS[mode],
        )
    ax.set(
        xlabel="Source nJets (>=9 combined)",
        ylabel="Complete-event assignment efficiency",
        title="Complete assignment efficiency by jet multiplicity",
    )
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    ax.legend(frameon=False, fontsize=7)
    ax.set_xticks(source_njet_bins, [">=9" if n == 9 else str(n) for n in source_njet_bins])
    ax.text(
        0.02,
        0.02,
        "Denominator: visible-assignment-supported events in each nJets bin",
        transform=ax.transAxes,
        fontsize=8,
    )
    _save(fig, assignment_dir / "complete_event_efficiency_vs_source_njet")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for mode in STANDARD:
        values, denominators = [], []
        for n in model_visible_bins:
            selected = predictions[mode]["assignment_supported"] & (model_visible_njet == n)
            denominators.append(int(selected.sum()))
            values.append(float(correctness[mode][selected].mean()) if selected.any() else np.nan)
            assignment_rows.append(
                {
                    "model": LABELS[mode],
                    "multiplicity_definition": "model_visible_njet",
                    "multiplicity_bin": int(n),
                    "selected_events": int((model_visible_njet == n).sum()),
                    "supported_events": int(selected.sum()),
                    "complete_event_efficiency": values[-1],
                }
            )
        by_model_visible_njet[mode] = {
            str(int(n)): {
                "selected_events": int((model_visible_njet == n).sum()),
                "supported_events": denominators[i],
                "efficiency": float(values[i]) if denominators[i] else None,
            }
            for i, n in enumerate(model_visible_bins)
        }
        ax.plot(
            model_visible_bins,
            values,
            marker="o",
            markersize=3,
            color=COLORS[mode],
            label=LABELS[mode],
        )
    ax.set(
        xlabel="Model-visible nJets",
        ylabel="Complete-event assignment efficiency",
        title="Complete assignment efficiency by model-visible jet multiplicity",
    )
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    ax.legend(frameon=False, fontsize=7)
    ax.text(
        0.02,
        0.02,
        "Denominator: visible-assignment-supported events in each model-visible nJets bin",
        transform=ax.transAxes,
        fontsize=8,
    )
    _save(fig, assignment_dir / "complete_event_efficiency_vs_model_visible_njet")

    for definition, bins, values in (
        ("source_njet", source_njet_bins, by_source_njet),
        ("model_visible_njet", model_visible_bins, by_model_visible_njet),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for mode in STANDARD:
            fractions = [
                values[mode][">=9" if definition == "source_njet" and n == 9 else str(int(n))][
                    "supported_events"
                ]
                / max(1, int((source_njet == n).sum()) if definition == "source_njet" and n < 9 else int((source_njet >= 9).sum()) if definition == "source_njet" else int((model_visible_njet == n).sum()))
                for n in bins
            ]
            ax.plot(bins, fractions, marker="o", color=COLORS[mode], label=LABELS[mode])
        ax.set(
            xlabel="Source nJets (>=9 combined)" if definition == "source_njet" else "Model-visible nJets",
            ylabel="Assignment-supported fraction",
            title=f"Assignment support by {definition.replace('_', ' ')}",
        )
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
        if definition == "source_njet":
            ax.set_xticks(bins, [">=9" if n == 9 else str(n) for n in bins])
        ax.legend(frameon=False, fontsize=7)
        _save(fig, assignment_dir / f"support_fraction_vs_{definition}")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    for ax, (first, second, title) in zip(
        axes, (COMPARISONS[0], COMPARISONS[2], COMPARISONS[3]), strict=True
    ):
        a = predictions[first]["assignment_complete_correct"]
        b = predictions[second]["assignment_complete_correct"]
        selected = (
            predictions[first]["assignment_supported"]
            & predictions[second]["assignment_supported"]
        )
        delta = (b[selected].mean() - a[selected].mean()) if selected.any() else np.nan
        matrix = np.array(
            [
                [np.sum(a[selected] & b[selected]), np.sum(a[selected] & ~b[selected])],
                [np.sum(~a[selected] & b[selected]), np.sum(~a[selected] & ~b[selected])],
            ],
            dtype=np.int64,
        )
        transition_matrices[f"{first}_to_{second}"] = {
            "denominator": int(selected.sum()),
            "row_first_model_correctness_column_second_model_correctness": matrix.tolist(),
            "percent_of_paired_supported_events": (
                (100.0 * matrix / selected.sum()).tolist() if selected.any() else None
            ),
        }
        ax.bar(
            ["First", "Second"],
            [a[selected].mean(), b[selected].mean()],
            color=[COLORS[first], COLORS[second]],
        )
        ax.set(
            title=f"{title}\npaired n={int(selected.sum()):,}; Δ={delta:+.3f}",
            ylabel="Complete-event efficiency",
            ylim=(0, 1.02),
        )
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
        ax.text(
            0.03,
            0.04,
            f"discordant: {int((a[selected] != b[selected]).sum()):,}",
            transform=ax.transAxes,
            fontsize=8,
        )
        # Keep the transition matrix in a separate inset-like PDF per comparison.
        fig_matrix, matrix_ax = plt.subplots(figsize=(5.2, 4.5))
        percent_matrix = 100.0 * matrix / selected.sum() if selected.any() else matrix.astype(float)
        mesh = matrix_ax.imshow(percent_matrix, vmin=0, vmax=100, cmap="Blues")
        for row_index in range(2):
            for column_index in range(2):
                matrix_ax.text(
                    column_index,
                    row_index,
                    f"{percent_matrix[row_index, column_index]:.1f}%\n(n={matrix[row_index, column_index]:,})",
                    ha="center",
                    va="center",
                    color="black",
                )
        matrix_ax.set_xticks([0, 1], ["correct", "incorrect"])
        matrix_ax.set_yticks([0, 1], ["correct", "incorrect"])
        matrix_ax.set(
            xlabel=LABELS[second],
            ylabel=LABELS[first],
            title=f"Paired transition; denominator={int(selected.sum()):,}",
        )
        fig_matrix.colorbar(mesh, ax=matrix_ax, label="Percent of common supported events")
        _save(fig_matrix, assignment_dir / f"transition_{first}_to_{second}")
    fig.tight_layout()
    _save(fig, assignment_dir / "paired_model_transitions")
    (assignment_dir / "paired_transition_matrices.json").write_text(
        json.dumps(transition_matrices, indent=2, allow_nan=False) + "\n"
    )

    summary = {
        "support_denominator": int(support.sum()),
        "complete_event": {
            mode: {
                "supported_events": int(
                    predictions[mode]["assignment_supported"].sum()
                ),
                "correct_events": int(correctness[mode].sum()),
                "efficiency": float(
                    correctness[mode].sum()
                    / max(1, predictions[mode]["assignment_supported"].sum())
                ),
                "by_source_njet": by_source_njet[mode],
                "by_model_visible_njet": by_model_visible_njet[mode],
            }
            for mode in STANDARD
        },
        "source_truncation": truncation_rows,
        "role_metrics": rows,
    }
    with (output / "tables/assignment.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "tables/assignment.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (output / "tables/assignment_efficiency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(assignment_rows[0]))
        writer.writeheader()
        writer.writerows(assignment_rows)
    with (output / "tables/assignment_truncation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(truncation_rows[0]))
        writer.writeheader()
        writer.writerows(truncation_rows)
    with (output / "tables/assignment_role_efficiency_by_multiplicity.csv").open(
        "w", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(role_bin_rows[0]))
        writer.writeheader()
        writer.writerows(role_bin_rows)
    return correctness, summary


def _neutrino_products(
    output: Path, predictions: dict, truth: dict, assignment_correct: dict
) -> tuple[dict, dict]:
    folder = output / "neutrinos"
    folder.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    point_estimators = {
        name: {} for name in ("single_draw", "cartesian_mean", "joint_medoid")
    }
    posterior_values: dict[str, dict[str, dict[str, np.ndarray]]] = {
        role: {} for role in NU_ROLES
    }
    for mode in STANDARD:
        samples = predictions[mode]["nu_samples"].astype(np.float64)
        if samples.ndim != 4 or samples.shape[2:] != (2, 3):
            raise ValueError(
                f"{mode} neutrino samples have unexpected shape {samples.shape}"
            )
        point_estimators["single_draw"][mode] = samples[:, 0]
        point_estimators["cartesian_mean"][mode] = samples.mean(axis=1)
        point_estimators["joint_medoid"][mode] = _joint_medoid(samples)
        for role_index, role in enumerate(NU_ROLES):
            role_samples = samples[:, :, role_index]
            role_truth = truth["neutrino"][:, role_index]
            sample_obs = _nu_observables(role_samples)
            truth_obs = _nu_observables(role_truth)
            dphi = _wrap(sample_obs["phi"] - truth_obs["phi"][:, None])
            d_eta = sample_obs["eta"] - truth_obs["eta"][:, None]
            delta_r = np.sqrt(d_eta**2 + dphi**2)
            role_samples_obs = dict(sample_obs)
            role_samples_obs["DeltaR"] = delta_r
            target_obs = dict(truth_obs)
            target_obs["DeltaR"] = np.zeros(len(role_truth))
            posterior_values[role][mode] = role_samples_obs
            for name, values in role_samples_obs.items():
                valid = (
                    truth["neutrino_support"][:, role_index]
                    & np.isfinite(target_obs[name])
                    & np.isfinite(values).all(axis=1)
                )
                if not valid.any():
                    continue
                truth_value = target_obs[name][valid]
                posterior = values[valid]
                if name == "phi":
                    residual_samples = _wrap(posterior - truth_value[:, None])
                    q05, q16, q50, q84, q95 = np.quantile(
                        residual_samples, [0.05, 0.16, 0.5, 0.84, 0.95], axis=1
                    )
                    median_residual = q50
                    width68 = q84 - q16
                    width90 = q95 - q05
                    coverage68 = np.mean((q16 <= 0) & (0 <= q84))
                    coverage90 = np.mean((q05 <= 0) & (0 <= q95))
                    pit_mean = float(np.mean(np.mean(residual_samples <= 0, axis=1)))
                else:
                    q05, q16, q50, q84, q95 = np.quantile(
                        posterior, [0.05, 0.16, 0.5, 0.84, 0.95], axis=1
                    )
                    median_residual = q50 - truth_value
                    width68 = q84 - q16
                    width90 = q95 - q05
                    coverage68 = np.mean((truth_value >= q16) & (truth_value <= q84))
                    coverage90 = np.mean((truth_value >= q05) & (truth_value <= q95))
                    pit_mean = float(
                        np.mean(np.mean(posterior <= truth_value[:, None], axis=1))
                    )
                metric_rows.append(
                    {
                        "model": LABELS[mode],
                        "role": role,
                        "observable": name,
                        "events": int(valid.sum()),
                        "posterior_q50_bias": float(np.mean(median_residual)),
                        "posterior_q50_mae": float(np.mean(np.abs(median_residual))),
                        "posterior_q50_rmse": float(
                            np.sqrt(np.mean(median_residual**2))
                        ),
                        "central_68_coverage": float(coverage68),
                        "central_90_coverage": float(coverage90),
                        "mean_central_68_width": float(np.nanmean(width68)),
                        "mean_central_90_width": float(np.nanmean(width90)),
                        "pit_mean": pit_mean,
                    }
                )
            # Event-level reducers use a complete correlated draw for joint medoid.
            reductions = {
                "single_draw": samples[:, 0, role_index],
                "cartesian_mean": samples[:, :, role_index].mean(axis=1),
                "joint_medoid": point_estimators["joint_medoid"][mode][:, role_index],
            }
            for reducer, points in reductions.items():
                point_obs = _nu_observables(points)
                for name, values in point_obs.items():
                    valid = (
                        truth["neutrino_support"][:, role_index]
                        & np.isfinite(values)
                        & np.isfinite(truth_obs[name])
                    )
                    if not valid.any():
                        continue
                    residual = (
                        _wrap(values[valid] - truth_obs[name][valid])
                        if name == "phi"
                        else values[valid] - truth_obs[name][valid]
                    )
                    metric_rows.append(
                        {
                            "model": LABELS[mode],
                            "role": role,
                            "observable": f"{name}:{reducer}",
                            "events": int(valid.sum()),
                            "bias": float(residual.mean()),
                            "mae": float(np.abs(residual).mean()),
                            "rmse": float(np.sqrt(np.square(residual).mean())),
                            "resolution": float(residual.std()),
                        }
                    )

    truth_mass = _mass(truth["parents"][:, 0] + truth["parents"][:, 1])
    region_masks = {
        "inclusive_validation": np.ones(len(truth_mass), dtype=bool),
        "threshold_mttbar_lt_400": (truth_mass >= 345) & (truth_mass < 400),
        "high_mass_mttbar_ge_700": truth_mass >= 700,
        "source_njet_le_8": truth["source_njet"] <= 8,
        "source_njet_gt_8": truth["source_njet"] > 8,
    }
    region_masks.update(
        {
            f"source_njet_{n}": truth["source_njet"] == n
            for n in range(2, 9)
        }
    )
    region_masks["source_njet_ge_9"] = truth["source_njet"] >= 9
    quality_rows = []
    quality_summary = {}
    for mode in STANDARD:
        quality_summary[mode] = {}
        samples = predictions[mode]["nu_samples"].astype(np.float64)
        for role_index, role in enumerate(NU_ROLES):
            quality_summary[mode][role] = {}
            for variable, posterior, target in (
                (
                    "pz",
                    samples[:, :, role_index, 2],
                    truth["neutrino"][:, role_index, 2],
                ),
                (
                    "pT",
                    np.hypot(
                        samples[:, :, role_index, 0],
                        samples[:, :, role_index, 1],
                    ),
                    np.hypot(
                        truth["neutrino"][:, role_index, 0],
                        truth["neutrino"][:, role_index, 1],
                    ),
                ),
            ):
                for region, region_mask in region_masks.items():
                    valid = (
                        region_mask
                        & truth["neutrino_support"][:, role_index]
                        & np.isfinite(target)
                        & np.isfinite(posterior).all(axis=1)
                    )
                    count = int(valid.sum())
                    record = {"events": count}
                    if count >= 100:
                        values = posterior[valid]
                        truth_values = target[valid]
                        q05, q16, q50, q84, q95 = np.quantile(
                            values, [0.05, 0.16, 0.50, 0.84, 0.95], axis=1
                        )
                        record.update(
                            {
                                "pit_mean": float(
                                    np.mean(
                                        np.mean(values <= truth_values[:, None], axis=1)
                                    )
                                ),
                                "central_68_coverage": float(
                                    np.mean((truth_values >= q16) & (truth_values <= q84))
                                ),
                                "central_90_coverage": float(
                                    np.mean((truth_values >= q05) & (truth_values <= q95))
                                ),
                                "mean_central_68_width": float(np.mean(q84 - q16)),
                                "mean_central_90_width": float(np.mean(q95 - q05)),
                                "scalar_energy_score": _scalar_energy_score(
                                    values, truth_values
                                ),
                            }
                        )
                    else:
                        record.update(
                            {
                                "pit_mean": None,
                                "central_68_coverage": None,
                                "central_90_coverage": None,
                                "mean_central_68_width": None,
                                "mean_central_90_width": None,
                                "scalar_energy_score": None,
                            }
                        )
                    quality_summary[mode][role].setdefault(region, {})[variable] = record
                    quality_rows.append(
                        {
                            "model": LABELS[mode],
                            "role": role,
                            "observable": variable,
                            "region": region,
                            **record,
                        }
                    )
    (output / "posterior/neutrino_quality_by_region.json").write_text(
        json.dumps(quality_summary, indent=2, allow_nan=False) + "\n"
    )
    with (output / "tables/neutrino_posterior_quality.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(quality_rows[0]))
        writer.writeheader()
        writer.writerows(quality_rows)

    # Posterior observable plots and cross-model point-estimator tables.
    for role_index, role in enumerate(NU_ROLES):
        truth_obs = _nu_observables(truth["neutrino"][:, role_index])
        for name in ("px", "py", "pz", "pT", "eta", "phi", "DeltaR"):
            samples_by_mode = {
                mode: posterior_values[role][mode][name] for mode in STANDARD
            }
            target_values = (
                np.zeros(len(truth["neutrino"]))
                if name == "DeltaR"
                else truth_obs[name]
            )
            _plot_scalar_suite(
                folder,
                f"{role.replace('.', '_')}_{name}",
                f"{role} {name}",
                "GeV"
                if name in {"px", "py", "pz", "pT"}
                else ("rad" if name == "phi" else ""),
                target_values,
                samples_by_mode,
                angular=name == "phi",
            )
        for mode in STANDARD:
            samples = predictions[mode]["nu_samples"].astype(np.float64)[
                :, :, role_index
            ]
            pt = np.hypot(samples[..., 0], samples[..., 1])
            target_pt = np.hypot(
                truth["neutrino"][:, role_index, 0], truth["neutrino"][:, role_index, 1]
            )
            pz = samples[..., 2]
            target_pz = truth["neutrino"][:, role_index, 2]
            for variable, samples_v, truth_v, label in (
                ("pT", pt, target_pt, "truth pT"),
                ("pz", pz, target_pz, "truth pz"),
            ):
                edges = np.quantile(
                    truth_v[np.isfinite(truth_v)], np.linspace(0, 1, 16)
                )
                centers = 0.5 * (edges[:-1] + edges[1:])
                q50 = np.quantile(samples_v, 0.5, axis=1)
                residual = q50 - truth_v
                bins = np.digitize(truth_v, edges[1:-1], right=False)
                bias = np.asarray(
                    [
                        np.mean(residual[bins == i]) if np.any(bins == i) else np.nan
                        for i in range(len(centers))
                    ]
                )
                resolution = np.asarray(
                    [
                        np.std(residual[bins == i]) if np.any(bins == i) else np.nan
                        for i in range(len(centers))
                    ]
                )
                fig, ax = plt.subplots(figsize=(6.8, 4.6))
                ax.plot(centers, bias, marker="o", color=COLORS[mode], label="Bias")
                ax.set(
                    xlabel=label,
                    ylabel=f"{variable} bias [GeV]",
                    title=f"{LABELS[mode]}: {role} bias vs {label}",
                )
                ax.axhline(0, color="black", linestyle=":")
                _save(
                    fig,
                    folder
                    / f"{mode}_{role.replace('.', '_')}_{variable}_bias_vs_truth",
                )
                fig, ax = plt.subplots(figsize=(6.8, 4.6))
                ax.plot(
                    centers,
                    resolution,
                    marker="o",
                    color=COLORS[mode],
                    label="Resolution",
                )
                ax.set(
                    xlabel=label,
                    ylabel=f"{variable} resolution [GeV]",
                    title=f"{LABELS[mode]}: {role} resolution vs {label}",
                )
                _save(
                    fig,
                    folder
                    / f"{mode}_{role.replace('.', '_')}_{variable}_resolution_vs_truth",
                )

        # Channel, both jet-count definitions, and assignment splits for pz.
        for split_name, split_values in (
            ("channel", truth["channel"]),
            ("source_njet", np.where(truth["source_njet"] >= 9, 9, truth["source_njet"])),
            ("model_visible_njet", truth["model_visible_njet"]),
            (
                "source_truncation",
                np.where(truth["source_njet"] > 8, "source_njet>8", "source_njet<=8"),
            ),
            ("assignment", None),
        ):
            fig, ax = plt.subplots(figsize=(7.0, 4.8))
            for mode in STANDARD:
                samples = predictions[mode]["nu_samples"].astype(np.float64)[
                    :, :, role_index, :
                ]
                q50 = np.quantile(samples[..., 2], 0.5, axis=1)
                residual = q50 - truth["neutrino"][:, role_index, 2]
                if split_name == "assignment":
                    local = assignment_correct[mode]
                    splits = (
                        ("correct", local),
                        (
                            "incorrect",
                            predictions[mode]["assignment_supported"] & ~local,
                        ),
                    )
                else:
                    unique = sorted(np.unique(split_values).tolist(), key=str)
                    splits = tuple(
                        (
                            ">=9" if split_name == "source_njet" and value == 9 else str(value),
                            split_values == value,
                        )
                        for value in unique
                    )
                for label, select in splits:
                    valid = (
                        select
                        & truth["neutrino_support"][:, role_index]
                        & np.isfinite(residual)
                    )
                    if valid.sum() < 100:
                        continue
                    linestyle = (
                        "-" if label in {"correct", "ee", "2", "3", "4"} else "--"
                    )
                    ax.hist(
                        residual[valid],
                        bins=60,
                        density=True,
                        histtype="step",
                        color=COLORS[mode],
                        linestyle=linestyle,
                        label=f"{LABELS[mode]}: {label}",
                    )
            ax.set(
                title=f"{role} pz residual by {split_name}",
                xlabel="Posterior q50 pz − truth [GeV]",
                ylabel="Normalized density",
            )
            ax.legend(frameon=False, fontsize=6, ncol=2)
            _save(fig, folder / f"{role.replace('.', '_')}_pz_residual_by_{split_name}")

    # Correlated-event posterior quality and nested-prefix stability.
    posterior_summary = {}
    for mode in STANDARD:
        samples = predictions[mode]["nu_samples"].astype(np.float64)
        target = truth["neutrino"]
        valid = truth["neutrino_support"].all(axis=1)
        full_medoid = _joint_medoid(samples[valid])
        full_pt_q50 = np.quantile(
            np.hypot(samples[valid, :, :, 0], samples[valid, :, :, 1]),
            0.5,
            axis=1,
        )
        full_pz_q50 = np.quantile(samples[valid, :, :, 2], 0.5, axis=1)
        posterior_summary[mode] = {
            "events": int(valid.sum()),
            "samples": int(samples.shape[1]),
            "joint_energy_score_momentum_GeV": _energy_score(
                samples[valid], target[valid]
            ),
            "observable_q50_prefix_stability": {},
            "sample_prefixes": {},
        }
        for count in (1, 4, 8, 16, 32, 64):
            if count > samples.shape[1]:
                continue
            prefix = samples[valid, :count]
            cart = prefix.mean(axis=1)
            medoid = _joint_medoid(prefix)
            pt_q50 = np.quantile(np.hypot(prefix[..., 0], prefix[..., 1]), 0.5, axis=1)
            pz_q50 = np.quantile(prefix[..., 2], 0.5, axis=1)
            posterior_summary[mode]["sample_prefixes"][str(count)] = {
                "cartesian_mean_rms_change_from_full": float(
                    np.sqrt(np.mean((cart - samples[valid].mean(axis=1)) ** 2))
                ),
                "joint_medoid_fraction_changed_from_full": float(
                    np.any(medoid != full_medoid, axis=(1, 2)).mean()
                )
                if count < samples.shape[1]
                else 0.0,
            }
            posterior_summary[mode]["observable_q50_prefix_stability"][str(count)] = {
                "top.nu_pT_rms_change_GeV": float(
                    np.sqrt(np.mean((pt_q50[:, 0] - full_pt_q50[:, 0]) ** 2))
                ),
                "antitop.nu_pT_rms_change_GeV": float(
                    np.sqrt(np.mean((pt_q50[:, 1] - full_pt_q50[:, 1]) ** 2))
                ),
                "top.nu_pz_rms_change_GeV": float(
                    np.sqrt(np.mean((pz_q50[:, 0] - full_pz_q50[:, 0]) ** 2))
                ),
                "antitop.nu_pz_rms_change_GeV": float(
                    np.sqrt(np.mean((pz_q50[:, 1] - full_pz_q50[:, 1]) ** 2))
                ),
            }
    (output / "posterior/neutrino_summary.json").write_text(
        json.dumps(posterior_summary, indent=2, allow_nan=False) + "\n"
    )
    if metric_rows:
        with (output / "tables/neutrino_metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(
                    dict.fromkeys(key for row in metric_rows for key in row)
                ),
            )
            writer.writeheader()
            writer.writerows(metric_rows)
    return point_estimators, posterior_summary


def _met_closure_products(output: Path, standard: dict, truth: dict) -> dict:
    folder = output / "neutrinos"
    folder.mkdir(parents=True, exist_ok=True)
    summary = {}
    rows = []
    for mode in STANDARD:
        samples = standard[mode]["nu_samples"].astype(np.float64)
        predicted_met = samples.sum(axis=2)[..., :2]
        residual = predicted_met - truth["met"][:, None, :]
        residual_magnitude = np.linalg.norm(residual, axis=-1)
        point = standard[mode].get("nu_joint_medoid")
        if point is None:
            point = _joint_medoid(samples)
        point_met = point.sum(axis=1)[:, :2]
        point_residual = point_met - truth["met"]
        summary[mode] = {
            "events": len(truth["met"]),
            "posterior_samples_per_event": int(samples.shape[1]),
            "delta_met_x": _residual_summary(residual[..., 0].reshape(-1)),
            "delta_met_y": _residual_summary(residual[..., 1].reshape(-1)),
            "delta_met_magnitude": {
                "samples": int(residual_magnitude.size),
                "median_gev": float(np.median(residual_magnitude)),
                "q16_gev": float(np.quantile(residual_magnitude, 0.16)),
                "q84_gev": float(np.quantile(residual_magnitude, 0.84)),
                "mean_gev": float(residual_magnitude.mean()),
            },
            "joint_medoid_point": {
                "delta_met_x": _residual_summary(point_residual[:, 0]),
                "delta_met_y": _residual_summary(point_residual[:, 1]),
                "delta_met_magnitude": {
                    "events": len(point_residual),
                    "median_gev": float(np.median(np.linalg.norm(point_residual, axis=-1))),
                    "mean_gev": float(np.linalg.norm(point_residual, axis=-1).mean()),
                },
            },
        }
        for axis, component in enumerate(("x", "y")):
            local = residual[..., axis].reshape(-1)
            local_summary = _residual_summary(local)
            rows.append(
                {
                    "model": LABELS[mode],
                    "quantity": f"delta_met_{component}",
                    **local_summary,
                }
            )
            fig, ax = plt.subplots(figsize=(6.8, 4.8))
            ax.hist(local, bins=80, histtype="step", color=COLORS[mode])
            ax.axvline(0, color="black", linestyle=":")
            ax.set(
                xlabel=f"Predicted neutrino sum − measured MET {component} [GeV]",
                ylabel="Posterior samples",
                title=f"{LABELS[mode]} MET closure in {component}",
            )
            _save(fig, folder / f"{mode}_met_residual_{component}")
        for axis, component in enumerate(("x", "y")):
            fig, ax = plt.subplots(figsize=(6.5, 5.4))
            ax.scatter(
                truth["met"][:, axis],
                point_met[:, axis],
                s=3,
                alpha=0.12,
                color=COLORS[mode],
                rasterized=True,
            )
            limits = [
                float(min(truth["met"][:, axis].min(), point_met[:, axis].min())),
                float(max(truth["met"][:, axis].max(), point_met[:, axis].max())),
            ]
            ax.plot(limits, limits, color="black", linestyle=":")
            ax.set(
                xlabel=f"Measured MET {component} [GeV]",
                ylabel=f"Joint-medoid neutrino sum {component} [GeV]",
                title=f"{LABELS[mode]}: MET {component} closure",
            )
            _save(fig, folder / f"{mode}_met_{component}_vs_predicted_neutrino_sum")
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.hist(residual_magnitude.reshape(-1), bins=80, histtype="step", color=COLORS[mode])
        ax.set(
            xlabel="|predicted neutrino sum − measured MET| [GeV]",
            ylabel="Posterior samples",
            title=f"{LABELS[mode]} MET residual magnitude",
        )
        _save(fig, folder / f"{mode}_met_residual_magnitude")
    (folder / "met_closure.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (output / "tables/met_closure.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summary


def _parent_products(
    output: Path,
    predictions: dict,
    truth: dict,
    perfect_parent: np.ndarray,
    assignment_correct: dict,
) -> tuple[dict, dict]:
    folder = output / "parents"
    tt_folder = output / "ttbar"
    threshold_folder = output / "threshold"
    spin_folder = output / "spin"
    for path in (
        folder,
        tt_folder,
        threshold_folder,
        spin_folder,
        output / "posterior",
    ):
        path.mkdir(parents=True, exist_ok=True)
    parent_samples = {
        mode: predictions[mode]["parent_samples"].astype(np.float64) for mode in PARENT
    }
    point_methods = {
        name: {} for name in ("single_draw", "cartesian_mean", "joint_medoid")
    }
    for mode, samples in parent_samples.items():
        point_methods["single_draw"][mode] = samples[:, 0]
        point_methods["cartesian_mean"][mode] = samples.mean(axis=1)
        point_methods["joint_medoid"][mode] = _joint_medoid(samples)

    truth_roles = [_p4_observables(truth["parents"][:, i]) for i in range(2)]
    parent_scalar_samples: dict[str, dict[str, np.ndarray]] = {}
    parent_metrics = []
    for role_index, role in enumerate(PARENT_ROLES):
        names = ("E", "px", "py", "pz", "pT", "eta", "y", "phi", "mass")
        truth_values = _p4_observables(truth["parents"][:, role_index])
        samples_by_observable = {}
        perfect_values = _p4_observables(perfect_parent[:, role_index])
        for name in names:
            samples_by_observable[name] = {}
            for mode, samples in parent_samples.items():
                samples_by_observable[name][mode] = _p4_observables(
                    samples[:, :, role_index]
                )[name]
            parent_scalar_samples[f"{role}/{name}"] = samples_by_observable[name]
            suite = _plot_scalar_suite(
                folder,
                f"{role}_{name}",
                f"Parton {role} {name}",
                "GeV"
                if name in {"E", "px", "py", "pz", "pT", "mass"}
                else ("rad" if name == "phi" else ""),
                truth_values[name],
                samples_by_observable[name],
                perfect=perfect_values[name],
                angular=name == "phi",
            )
            for mode in PARENT:
                metric = suite.get("metrics", {}).get(mode, {})
                parent_metrics.append(
                    {
                        "model": LABELS[mode],
                        "role": role,
                        "observable": name,
                        **{
                            key: value
                            for key, value in metric.items()
                            if not isinstance(value, list)
                        },
                    }
                )

    # Model-level posterior pair observables are derived from top + antitop for each sample.
    pair_truth = truth["parents"][:, 0] + truth["parents"][:, 1]
    pair_obs_truth = _p4_observables(pair_truth)
    truth_dphi = _wrap(truth_roles[0]["phi"] - truth_roles[1]["phi"])
    truth_dy = truth_roles[0]["y"] - truth_roles[1]["y"]
    truth_pair_obs = {
        "mttbar": _mass(pair_truth),
        "pT(ttbar)": pair_obs_truth["pT"],
        "y(ttbar)": pair_obs_truth["y"],
        "phi(ttbar)": pair_obs_truth["phi"],
        "E(ttbar)": pair_obs_truth["E"],
        "Delta phi(top, antitop)": truth_dphi,
        "Delta y(top, antitop)": truth_dy,
    }
    perfect_pair = perfect_parent[:, 0] + perfect_parent[:, 1]
    perfect_pair_obs0 = _p4_observables(perfect_pair)
    perfect_pair_obs = {
        "mttbar": _mass(perfect_pair),
        "pT(ttbar)": perfect_pair_obs0["pT"],
        "y(ttbar)": perfect_pair_obs0["y"],
        "phi(ttbar)": perfect_pair_obs0["phi"],
        "E(ttbar)": perfect_pair_obs0["E"],
        "Delta phi(top, antitop)": _wrap(
            _p4_observables(perfect_parent[:, 0])["phi"]
            - _p4_observables(perfect_parent[:, 1])["phi"]
        ),
        "Delta y(top, antitop)": _p4_observables(perfect_parent[:, 0])["y"]
        - _p4_observables(perfect_parent[:, 1])["y"],
    }
    pair_samples_observable: dict[str, dict[str, np.ndarray]] = {}
    pair_metrics = []
    observed_mass_max = max(
        float(np.nanmax(truth_pair_obs["mttbar"])),
        max(
            float(np.nanmax(_mass(samples[:, :, 0] + samples[:, :, 1])))
            for samples in parent_samples.values()
        ),
        float(np.nanmax(_mass(perfect_pair))),
    )
    open_edge = np.nextafter(max(1000.0, observed_mass_max), np.inf)
    mttbar_edges = np.asarray([345.0, 400.0, 500.0, 700.0, 1000.0, open_edge])
    for name, target in truth_pair_obs.items():
        samples_by_mode = {}
        for mode, samples in parent_samples.items():
            pair = samples[:, :, 0] + samples[:, :, 1]
            obs = _p4_observables(pair)
            if name == "mttbar":
                samples_by_mode[mode] = _mass(pair)
            elif name == "pT(ttbar)":
                samples_by_mode[mode] = obs["pT"]
            elif name == "y(ttbar)":
                samples_by_mode[mode] = obs["y"]
            elif name == "phi(ttbar)":
                samples_by_mode[mode] = obs["phi"]
            elif name == "E(ttbar)":
                samples_by_mode[mode] = obs["E"]
            elif name == "Delta phi(top, antitop)":
                samples_by_mode[mode] = _wrap(
                    _p4_observables(samples[:, :, 0])["phi"]
                    - _p4_observables(samples[:, :, 1])["phi"]
                )
            else:
                samples_by_mode[mode] = (
                    _p4_observables(samples[:, :, 0])["y"]
                    - _p4_observables(samples[:, :, 1])["y"]
                )
        pair_samples_observable[name] = samples_by_mode
        target_for_plot = target
        perfect_for_plot = perfect_pair_obs[name]
        fixed = mttbar_edges if name == "mttbar" else None
        suite = _plot_scalar_suite(
            tt_folder,
            name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", ""),
            name,
            "GeV"
            if name in {"mttbar", "pT(ttbar)", "E(ttbar)"}
            else ("rad" if name.startswith(("phi", "Delta phi")) else ""),
            target_for_plot,
            samples_by_mode,
            perfect=perfect_for_plot,
            fixed_edges=fixed,
            angular=name in {"phi(ttbar)", "Delta phi(top, antitop)"},
        )
        for mode in PARENT:
            metric = suite.get("metrics", {}).get(mode, {})
            pair_metrics.append(
                {
                    "model": LABELS[mode],
                    "observable": name,
                    **{
                        key: value
                        for key, value in metric.items()
                        if not isinstance(value, list)
                    },
                }
            )

    # Explicit threshold details and fine near-threshold view use common truth bins.
    truth_mass = truth_pair_obs["mttbar"]
    threshold_summary = {}
    for mode, samples in parent_samples.items():
        pair = samples[:, :, 0] + samples[:, :, 1]
        sample_mass = _mass(pair)
        medoid_mass = _mass(point_methods["joint_medoid"][mode][:, 0] + point_methods["joint_medoid"][mode][:, 1])
        point_error = medoid_mass - truth_mass
        bins = {}
        for lo, hi, label in (
            (345, 400, "threshold_345_400"),
            (400, 500, "400_500"),
            (500, 700, "500_700"),
            (700, 1000, "700_1000"),
            (1000, np.inf, "above_1000"),
        ):
            selected = (truth_mass >= lo) & (truth_mass < hi)
            residual = point_error[selected]
            lo68, hi68 = (
                np.quantile(sample_mass[selected], [0.16, 0.84], axis=1)
                if selected.any()
                else (np.array([]), np.array([]))
            )
            lo90, hi90 = (
                np.quantile(sample_mass[selected], [0.05, 0.95], axis=1)
                if selected.any()
                else (np.array([]), np.array([]))
            )
            bins[label] = {
                "events": int(selected.sum()),
                "bias_gev": float(residual.mean()) if selected.any() else None,
                "resolution_gev": float(residual.std()) if selected.any() else None,
                "mae_gev": float(np.abs(residual).mean()) if selected.any() else None,
                "rmse_gev": float(np.sqrt(np.mean(residual**2)))
                if selected.any()
                else None,
                "median_residual_gev": float(np.median(residual))
                if selected.any()
                else None,
                "relative_bias": float(np.mean(residual / truth_mass[selected]))
                if selected.any()
                else None,
                "central_68_coverage": float(
                    np.mean(
                        (truth_mass[selected] >= lo68) & (truth_mass[selected] <= hi68)
                    )
                )
                if selected.any()
                else None,
                "central_90_coverage": float(
                    np.mean(
                        (truth_mass[selected] >= lo90) & (truth_mass[selected] <= hi90)
                    )
                )
                if selected.any()
                else None,
                "mean_central_68_width_gev": float(np.mean(hi68 - lo68))
                if selected.any()
                else None,
                "mean_central_90_width_gev": float(np.mean(hi90 - lo90))
                if selected.any()
                else None,
            }
        threshold_summary[mode] = bins

    posterior_region_masks = {
        "inclusive_validation": np.ones(len(truth_mass), dtype=bool),
        "threshold_345_400": (truth_mass >= 345) & (truth_mass < 400),
        "high_mass_ge_700": truth_mass >= 700,
        "source_njet_le_8": truth["source_njet"] <= 8,
        "source_njet_gt_8": truth["source_njet"] > 8,
    }
    posterior_region_masks.update(
        {f"source_njet_{n}": truth["source_njet"] == n for n in range(2, 9)}
    )
    posterior_region_masks["source_njet_ge_9"] = truth["source_njet"] >= 9
    posterior_region_summary = {}
    posterior_quality_rows = []
    migration_summary = {}
    migration_labels = [
        "<345",
        "345–400",
        "400–500",
        "500–700",
        "700–1000",
        ">1000",
        "invalid",
    ]
    migration_truth_bin = np.digitize(truth_mass, [345, 400, 500, 700, 1000])
    for mode, samples in parent_samples.items():
        mass_samples = _mass(samples[:, :, 0] + samples[:, :, 1])
        posterior_region_summary[mode] = {}
        for region, region_mask in posterior_region_masks.items():
            valid = (
                region_mask
                & truth["parent_support"].all(axis=1)
                & np.isfinite(truth_mass)
                & np.isfinite(mass_samples).all(axis=1)
            )
            record = {"events": int(valid.sum())}
            if valid.sum() >= 100:
                local_samples = mass_samples[valid]
                target_values = truth_mass[valid]
                q05, q16, q50, q84, q95 = np.quantile(
                    local_samples, [0.05, 0.16, 0.50, 0.84, 0.95], axis=1
                )
                width68 = q84 - q16
                absolute_error = np.abs(q50 - target_values)
                correlation = (
                    float(np.corrcoef(width68, absolute_error)[0, 1])
                    if np.std(width68) > 0 and np.std(absolute_error) > 0
                    else None
                )
                record.update(
                    {
                        "pit_mean": float(
                            np.mean(
                                np.mean(local_samples <= target_values[:, None], axis=1)
                            )
                        ),
                        "central_68_coverage": float(
                            np.mean((target_values >= q16) & (target_values <= q84))
                        ),
                        "central_90_coverage": float(
                            np.mean((target_values >= q05) & (target_values <= q95))
                        ),
                        "mean_central_68_width_gev": float(width68.mean()),
                        "mean_central_90_width_gev": float((q95 - q05).mean()),
                        "scalar_energy_score_gev": _scalar_energy_score(
                            local_samples, target_values
                        ),
                        "width_vs_absolute_q50_error_correlation": correlation,
                    }
                )
            else:
                record.update(
                    {
                        "pit_mean": None,
                        "central_68_coverage": None,
                        "central_90_coverage": None,
                        "mean_central_68_width_gev": None,
                        "mean_central_90_width_gev": None,
                        "scalar_energy_score_gev": None,
                        "width_vs_absolute_q50_error_correlation": None,
                    }
                )
            posterior_region_summary[mode][region] = record
            posterior_quality_rows.append(
                {"model": LABELS[mode], "region": region, **record}
            )

        predicted_bin = np.digitize(mass_samples, [345, 400, 500, 700, 1000])
        predicted_bin[~np.isfinite(mass_samples)] = len(migration_labels) - 1
        matrix = np.zeros((len(migration_labels), len(migration_labels)), dtype=np.int64)
        for row_index in range(len(migration_labels)):
            selected = migration_truth_bin == row_index
            if selected.any():
                matrix[row_index] = np.bincount(
                    predicted_bin[selected].ravel(), minlength=len(migration_labels)
                )[: len(migration_labels)]
        row_sums = matrix.sum(axis=1, keepdims=True)
        fractions = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=np.float64),
            where=row_sums > 0,
        )
        migration_summary[mode] = {
            "truth_bin_labels": migration_labels,
            "predicted_bin_labels": migration_labels,
            "raw_posterior_sample_counts": matrix.tolist(),
            "row_normalized_probabilistic_migration": fractions.tolist(),
        }
        fig, ax = plt.subplots(figsize=(7.0, 5.6))
        mesh = ax.imshow(100 * fractions, vmin=0, vmax=100, cmap="Blues")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                ax.text(
                    column_index,
                    row_index,
                    f"{100 * fractions[row_index, column_index]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        ax.set_xticks(range(len(migration_labels)), migration_labels, rotation=30, ha="right")
        ax.set_yticks(range(len(migration_labels)), migration_labels)
        ax.set(
            xlabel="Posterior mttbar bin",
            ylabel="Parton-truth mttbar bin",
            title=f"{LABELS[mode]} posterior mttbar migration",
        )
        fig.colorbar(mesh, ax=ax, label="Posterior probability [%]")
        _save(fig, output / "posterior" / f"mttbar_migration_{mode}")
    (output / "posterior/parent_quality_by_region.json").write_text(
        json.dumps(posterior_region_summary, indent=2, allow_nan=False) + "\n"
    )
    (output / "posterior/mttbar_probabilistic_migration.json").write_text(
        json.dumps(migration_summary, indent=2, allow_nan=False) + "\n"
    )
    with (output / "tables/parent_posterior_quality.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(posterior_quality_rows[0]))
        writer.writeheader()
        writer.writerows(posterior_quality_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fine_edges = np.asarray([345, 360, 375, 390, 400, 415, 430, 450], dtype=float)
    for mode in parent_samples:
        medoid_mass = _mass(point_methods["joint_medoid"][mode][:, 0] + point_methods["joint_medoid"][mode][:, 1])
        centers, bias, resolution, _counts = _bin_summary(truth_mass, medoid_mass, fine_edges)
        axes[0].plot(centers, bias, marker="o", color=COLORS[mode], label=LABELS[mode])
        axes[1].plot(
            centers, resolution, marker="o", color=COLORS[mode], label=LABELS[mode]
        )
    axes[0].axhline(0, color="black", linestyle=":")
    for ax, ylabel, title in (
        (axes[0], "Bias [GeV]", "Near-threshold bias"),
        (axes[1], "Resolution [GeV]", "Near-threshold resolution"),
    ):
        ax.set(xlabel="Parton-truth mttbar [GeV]", ylabel=ylabel, title=title)
        ax.legend(frameon=False, fontsize=7)
    _save(fig, threshold_folder / "mttbar_first_55gev_above_threshold")
    coverage_labels = (
        "threshold_345_400",
        "400_500",
        "500_700",
        "700_1000",
        "above_1000",
    )
    coverage_ranges = ((345, 400), (400, 500), (500, 700), (700, 1000), (1000, np.inf))
    coverage_centers = np.asarray(
        [
            truth_mass[(truth_mass >= low) & (truth_mass < high)].mean()
            for low, high in coverage_ranges
        ]
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for mode in PARENT:
        for coverage, marker, linestyle in (
            ("central_68_coverage", "o", "-"),
            ("central_90_coverage", "s", "--"),
        ):
            values = [
                threshold_summary[mode][label].get(coverage, np.nan)
                for label in coverage_labels
            ]
            ax.plot(
                coverage_centers,
                values,
                marker=marker,
                linestyle=linestyle,
                color=COLORS[mode],
                label=f"{LABELS[mode]}: {coverage.split('_')[1]}%",
            )
    ax.axhline(0.68, color="#666666", linestyle=":", linewidth=1)
    ax.axhline(0.90, color="#aaaaaa", linestyle=":", linewidth=1)
    ax.set(
        xlabel="Parton-truth mttbar [GeV] (common bins)",
        ylabel="Empirical posterior coverage",
        title="mttbar posterior coverage vs truth",
    )
    ax.set_ylim(0.0, 1.02)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    _save(fig, threshold_folder / "mttbar_coverage_vs_truth")
    (output / "tables/threshold.json").write_text(
        json.dumps(threshold_summary, indent=2, allow_nan=False) + "\n"
    )

    # Official ordered top/antitop and diagnostic swap-aware four-vector comparison.
    ordering = {}
    for mode, samples in parent_samples.items():
        point = point_methods["joint_medoid"][mode]
        scale = np.nanstd(truth["parents"], axis=0).reshape(2, 4)
        scale = np.where(scale > 1e-8, scale, 1.0)
        ordered = np.sqrt(
            np.square((point - truth["parents"]) / scale[None, :, :]).sum(axis=(1, 2))
        )
        swapped = np.sqrt(
            np.square((point[:, ::-1] - truth["parents"]) / scale[None, :, :]).sum(
                axis=(1, 2)
            )
        )
        improves = swapped < ordered
        ordering[mode] = {
            "events": len(point),
            "swap_improves_fraction": float(improves.mean()),
            "ordered_mean_scaled_error": float(ordered.mean()),
            "best_swap_diagnostic_mean_scaled_error": float(
                np.minimum(ordered, swapped).mean()
            ),
            "ordered_error_by_assignment": {},
        }
        if mode in assignment_correct:
            good = assignment_correct[mode]
            support = predictions[mode]["assignment_supported"]
            for label, mask in (
                ("assignment_correct", good & support),
                ("assignment_incorrect", ~good & support),
            ):
                ordering[mode]["ordered_error_by_assignment"][label] = {
                    "events": int(mask.sum()),
                    "ordered_error": float(ordered[mask].mean())
                    if mask.any()
                    else None,
                    "swap_improves_fraction": float(improves[mask].mean())
                    if mask.any()
                    else None,
                }
    (output / "parents/top_antitop_ordering.json").write_text(
        json.dumps(ordering, indent=2, allow_nan=False) + "\n"
    )

    # Existing maintained angular observables only; no new c_hel/c_han convention is introduced.
    try:
        source_root = Path(os.environ.get("JER_REALNU_SOURCE_ROOT", ""))
        if not source_root.is_dir():
            raise RuntimeError("JER_REALNU_SOURCE_ROOT missing")
        sys.path.insert(0, str(source_root / "src"))
        import torch

        from joint_event_reconstruction.models.partonic import parent_pair_observables

        def angle_values(vectors: np.ndarray) -> dict[str, np.ndarray]:
            original_shape = vectors.shape[:-2]
            flat = vectors.reshape(-1, 2, 4)
            finite = np.isfinite(flat).all(axis=(1, 2))
            names = (
                "cos_theta_star",
                "delta_phi_top_antitop",
                "delta_rapidity_top_antitop",
            )
            result = {name: np.full(len(flat), np.nan) for name in names}
            if finite.any():
                calculated = parent_pair_observables(
                    torch.as_tensor(flat[finite], dtype=torch.float64)
                )
                for name in names:
                    result[name][finite] = calculated[name].detach().cpu().numpy()
            return {
                name: values.reshape(original_shape) for name, values in result.items()
            }

        truth_angles = angle_values(truth["parents"])
        perfect_angles = angle_values(perfect_parent)
        spin_observables = {
            "cos_theta_star": (
                truth_angles["cos_theta_star"],
                perfect_angles["cos_theta_star"],
                {
                    mode: np.asarray(
                        predictions[mode]["parent_observables"]["cos_theta_star"],
                        dtype=np.float64,
                    )
                    for mode in PARENT
                },
                "",
                False,
            ),
            "delta_phi_top_antitop": (
                truth_angles["delta_phi_top_antitop"],
                perfect_angles["delta_phi_top_antitop"],
                {
                    mode: np.asarray(
                        predictions[mode]["parent_observables"][
                            "delta_phi_top_antitop"
                        ],
                        dtype=np.float64,
                    )
                    for mode in PARENT
                },
                "rad",
                True,
            ),
            "delta_rapidity_top_antitop": (
                truth_angles["delta_rapidity_top_antitop"],
                perfect_angles["delta_rapidity_top_antitop"],
                {
                    mode: np.asarray(
                        predictions[mode]["parent_observables"][
                            "delta_rapidity_top_antitop"
                        ],
                        dtype=np.float64,
                    )
                    for mode in PARENT
                },
                "",
                False,
            ),
        }
        for name, (target, perfect, samples, unit, angular) in spin_observables.items():
            _plot_scalar_suite(
                spin_folder,
                name,
                name.replace("_", " "),
                unit,
                target,
                samples,
                perfect=perfect,
                angular=angular,
            )
        (spin_folder / "README.md").write_text(
            "Maintained parent-pair observables are plotted here: cos_theta_star, delta_phi_top_antitop, and delta_rapidity_top_antitop. "
            "The authoritative supervision sidecar contains LABELS/LEPTON with shape [events, 2] (matched-label codes) and SUPPORT/LEPTON_CHARGE_SIDE (charge-side labels), but it contains no charged-lepton four-vectors. "
            "The source HDF5 metadata has reconstructed INPUTS/ELECTRON and INPUTS/MUON vectors, LABELS/ELECTRON and LABELS/MUON, and TARGETS/NEUTRINO, TOP, and TRUTH; it has no truth charged-lepton four-vector target. "
            "Thus measured charge-labelled leptons are available but truth charged-lepton vectors are not. No c_hel or c_han convention is fabricated; these parton-truth spin observables remain unavailable.\n"
        )
    except (ImportError, KeyError, RuntimeError) as error:
        (spin_folder / "README.md").write_text(
            f"Maintained angular observable evaluation unavailable: {type(error).__name__}: {error}. "
            "The supervision sidecar stores charge-side label codes but no truth charged-lepton four-vectors; measured leptons cannot be labelled as parton-truth spin. c_hel and c_han were not rederived.\n"
        )

    # Posterior observable calibration, point estimators, sample-prefix stability.
    posterior_summary = {}
    point_rows = []
    for mode, samples in parent_samples.items():
        valid = truth["parent_support"].all(axis=1)
        posterior_summary[mode] = {
            "events": int(valid.sum()),
            "samples": int(samples.shape[1]),
            "joint_energy_score_parent_p4_GeV": _energy_score(
                samples[valid], truth["parents"][valid]
            ),
            "sample_prefixes": {},
        }
        full_medoid = _joint_medoid(samples[valid])
        full_mean = samples[valid].mean(axis=1)
        full_mass = _mass(samples[valid, :, 0] + samples[valid, :, 1])
        full_mttbar_q50 = np.quantile(full_mass, 0.5, axis=1)
        for count in (1, 4, 8, 16, 32, 64):
            if count > samples.shape[1]:
                continue
            prefix = samples[valid, :count]
            medoid = _joint_medoid(prefix)
            cart = prefix.mean(axis=1)
            prefix_mttbar_q50 = np.quantile(
                _mass(prefix[:, :, 0] + prefix[:, :, 1]), 0.5, axis=1
            )
            posterior_summary[mode]["sample_prefixes"][str(count)] = {
                "cartesian_mean_rms_change_from_full_GeV": float(
                    np.sqrt(np.mean((cart - full_mean) ** 2))
                ),
                "joint_medoid_fraction_changed_from_full": float(
                    np.any(medoid != full_medoid, axis=(1, 2)).mean()
                )
                if count < samples.shape[1]
                else 0.0,
                "mttbar_q50_rms_change_GeV": float(
                    np.sqrt(np.mean((prefix_mttbar_q50 - full_mttbar_q50) ** 2))
                ),
            }
        for reducer, point_map in point_methods.items():
            points = point_map[mode]
            point_obs = [_p4_observables(points[:, role]) for role in range(2)]
            pair = points[:, 0] + points[:, 1]
            pair_obs = _p4_observables(pair)
            for role_index, role in enumerate(PARENT_ROLES):
                target_obs = truth_roles[role_index]
                for name in ("pT", "y", "mass"):
                    valid_role = (
                        truth["parent_support"][:, role_index]
                        & np.isfinite(target_obs[name])
                        & np.isfinite(point_obs[role_index][name])
                    )
                    residual = (
                        point_obs[role_index][name][valid_role]
                        - target_obs[name][valid_role]
                    )
                    point_rows.append(
                        {
                            "model": LABELS[mode],
                            "reducer": reducer,
                            "role": role,
                            "observable": name,
                            "events": int(valid_role.sum()),
                            "bias": float(residual.mean()),
                            "mae": float(np.abs(residual).mean()),
                            "rmse": float(np.sqrt(np.square(residual).mean())),
                        }
                    )
            for name, values in (
                ("mttbar", _mass(pair)),
                ("pT(ttbar)", pair_obs["pT"]),
                ("y(ttbar)", pair_obs["y"]),
            ):
                residual = values - truth_pair_obs[name]
                point_rows.append(
                    {
                        "model": LABELS[mode],
                        "reducer": reducer,
                        "role": "ttbar",
                        "observable": name,
                        "events": int(np.isfinite(residual).sum()),
                        "bias": float(np.nanmean(residual)),
                        "mae": float(np.nanmean(np.abs(residual))),
                        "rmse": float(np.sqrt(np.nanmean(residual**2))),
                    }
                )
    (output / "posterior/parent_summary.json").write_text(
        json.dumps(posterior_summary, indent=2, allow_nan=False) + "\n"
    )
    with (output / "tables/parent_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                dict.fromkeys(key for row in parent_metrics for key in row)
            ),
        )
        writer.writeheader()
        writer.writerows(parent_metrics)
    with (output / "tables/ttbar_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(dict.fromkeys(key for row in pair_metrics for key in row)),
        )
        writer.writeheader()
        writer.writerows(pair_metrics)
    with (output / "tables/point_estimators.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(dict.fromkeys(key for row in point_rows for key in row)),
        )
        writer.writeheader()
        writer.writerows(point_rows)
    return point_methods, {
        "threshold": threshold_summary,
        "ordering": ordering,
        "posterior": posterior_summary,
        "posterior_region_quality": posterior_region_summary,
        "posterior_migration": migration_summary,
        "parent_metrics": parent_metrics,
        "ttbar_metrics": pair_metrics,
        "point_estimators": point_rows,
    }


def _oracle_products(
    output: Path, predictions: dict, truth: dict, candidates: dict
) -> dict:
    folder = output / "oracle_decomposition"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    decomposition = {}
    true_candidate = {}
    for role in PARENT_ROLES:
        true_candidate[role] = _gather_candidate(
            candidates["candidate_vectors"][role], candidates["truth_index"][role]
        )
    true_nu = truth["neutrino"]
    true_nu_e = np.linalg.norm(true_nu, axis=-1)
    truth_nu_p4 = np.concatenate([true_nu_e[..., None], true_nu], axis=-1)
    maintained_angles = None
    try:
        import torch

        source_root = Path(os.environ["JER_REALNU_SOURCE_ROOT"])
        sys.path.insert(0, str(source_root / "src"))
        from joint_event_reconstruction.models.partonic import parent_pair_observables

        maintained_angles = parent_pair_observables(
            torch.as_tensor(truth["parents"], dtype=torch.float64)
        )
        maintained_angles = {
            name: value.detach().cpu().numpy()
            for name, value in maintained_angles.items()
            if name
            in {
                "cos_theta_star",
                "delta_phi_top_antitop",
                "delta_rapidity_top_antitop",
            }
        }
    except (ImportError, KeyError):
        maintained_angles = None
    for mode in STANDARD:
        predicted_candidate = {}
        for index, role in enumerate(PARENT_ROLES):
            predicted_candidate[role] = _gather_candidate(
                candidates["candidate_vectors"][role],
                predictions[mode]["roots"][:, 0, index],
            )
        nu_samples = predictions[mode]["nu_samples"].astype(np.float64)
        nu_medoid = _joint_medoid(nu_samples)
        predicted_nu_p4 = np.concatenate(
            [np.linalg.norm(nu_medoid, axis=-1)[..., None], nu_medoid], axis=-1
        )
        variants = {
            "truth_assignment_truth_neutrinos_perfect_reco": np.stack(
                [
                    true_candidate[role] + truth_nu_p4[:, i]
                    for i, role in enumerate(PARENT_ROLES)
                ],
                axis=1,
            ),
            "predicted_assignment_truth_neutrinos": np.stack(
                [
                    predicted_candidate[role] + truth_nu_p4[:, i]
                    for i, role in enumerate(PARENT_ROLES)
                ],
                axis=1,
            ),
            "truth_assignment_predicted_neutrinos": np.stack(
                [
                    true_candidate[role] + predicted_nu_p4[:, i]
                    for i, role in enumerate(PARENT_ROLES)
                ],
                axis=1,
            ),
            "predicted_assignment_predicted_neutrinos_full_model": np.stack(
                [
                    predicted_candidate[role] + predicted_nu_p4[:, i]
                    for i, role in enumerate(PARENT_ROLES)
                ],
                axis=1,
            ),
        }
        decomposition[mode] = {}
        for label, parents in variants.items():
            pair = parents[:, 0] + parents[:, 1]
            mass = _mass(pair)
            truth_mass = _mass(truth["parents"][:, 0] + truth["parents"][:, 1])
            valid = np.isfinite(mass) & np.isfinite(truth_mass)
            residual = mass[valid] - truth_mass[valid]
            decomposition[mode][label] = {
                "events": int(valid.sum()),
                "mttbar_bias_gev": float(residual.mean()) if valid.any() else None,
                "mttbar_mae_gev": float(np.abs(residual).mean())
                if valid.any()
                else None,
                "mttbar_rmse_gev": float(np.sqrt(np.mean(residual**2)))
                if valid.any()
                else None,
            }
            observable_metrics = {}
            for role_index, role in enumerate(PARENT_ROLES):
                predicted_observables = _p4_observables(parents[:, role_index])
                target_observables = _p4_observables(truth["parents"][:, role_index])
                for observable in ("pT", "y", "mass"):
                    selected = (
                        truth["parent_support"][:, role_index]
                        & np.isfinite(predicted_observables[observable])
                        & np.isfinite(target_observables[observable])
                    )
                    local = (
                        predicted_observables[observable][selected]
                        - target_observables[observable][selected]
                    )
                    observable_metrics[f"{role}_{observable}"] = {
                        "events": int(selected.sum()),
                        "bias": float(local.mean()) if selected.any() else None,
                        "rmse": float(np.sqrt(np.square(local).mean()))
                        if selected.any()
                        else None,
                    }
            target_pair = truth["parents"][:, 0] + truth["parents"][:, 1]
            predicted_pair_observables = _p4_observables(pair)
            target_pair_observables = _p4_observables(target_pair)
            for observable, key in (
                ("mttbar", None),
                ("pT(ttbar)", "pT"),
                ("y(ttbar)", "y"),
            ):
                predicted_value = (
                    _mass(pair)
                    if observable == "mttbar"
                    else predicted_pair_observables[key]
                )
                target_value = (
                    _mass(target_pair)
                    if observable == "mttbar"
                    else target_pair_observables[key]
                )
                selected = np.isfinite(predicted_value) & np.isfinite(target_value)
                local = predicted_value[selected] - target_value[selected]
                observable_metrics[observable] = {
                    "events": int(selected.sum()),
                    "bias": float(local.mean()) if selected.any() else None,
                    "rmse": float(np.sqrt(np.square(local).mean()))
                    if selected.any()
                    else None,
                }
            if maintained_angles is not None:
                selected = np.isfinite(parents).all(axis=(1, 2))
                if selected.any():
                    import torch

                    angles = parent_pair_observables(
                        torch.as_tensor(parents[selected], dtype=torch.float64)
                    )
                    for observable in (
                        "cos_theta_star",
                        "delta_phi_top_antitop",
                        "delta_rapidity_top_antitop",
                    ):
                        predicted_value = angles[observable].detach().cpu().numpy()
                        target_value = maintained_angles[observable][selected]
                        local = (
                            _wrap(predicted_value - target_value)
                            if observable == "delta_phi_top_antitop"
                            else predicted_value - target_value
                        )
                        observable_metrics[observable] = {
                            "events": int(selected.sum()),
                            "bias": float(local.mean()),
                            "rmse": float(np.sqrt(np.square(local).mean())),
                        }
            decomposition[mode][label]["observables"] = observable_metrics
            rows.append(
                {
                    "model": LABELS[mode],
                    "reconstruction": label,
                    **{
                        key: value
                        for key, value in decomposition[mode][label].items()
                        if key != "observables"
                    },
                }
            )
        # Same event-level decomposition as spectra, with the shared parton-truth reference.
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        bins = np.linspace(345, 1500, 55)
        ax.hist(
            truth_mass,
            bins=bins,
            density=True,
            histtype="step",
            color="black",
            label="Parton truth",
        )
        for label, parents in variants.items():
            short = label.replace("_", " ")
            ax.hist(
                _mass(parents[:, 0] + parents[:, 1]),
                bins=bins,
                density=True,
                histtype="step",
                label=short,
            )
        ax.set(
            xlabel="mttbar [GeV]",
            ylabel="Normalized density",
            title=f"Oracle decomposition: {LABELS[mode]}",
        )
        ax.legend(frameon=False, fontsize=7)
        _save(fig, folder / f"{mode}_mttbar_decomposition")
    (folder / "oracle_metrics.json").write_text(
        json.dumps(decomposition, indent=2, allow_nan=False) + "\n"
    )
    with (output / "tables/oracle_decomposition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return decomposition


def _residual_summary(residual: np.ndarray) -> dict:
    residual = np.asarray(residual, dtype=np.float64)
    residual = residual[np.isfinite(residual)]
    if not len(residual):
        return {
            "events": 0,
            "bias": None,
            "mae": None,
            "rmse": None,
            "central_residual_width": None,
        }
    return {
        "events": len(residual),
        "bias": float(residual.mean()),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "central_residual_width": float(residual.std()),
    }


def _paired_metric_bootstrap(
    first_residual: np.ndarray,
    second_residual: np.ndarray,
    *,
    seed: int,
    draws: int = 2000,
) -> dict:
    """Paired bootstrap of second-minus-first residual metrics."""
    first = np.asarray(first_residual, dtype=np.float64)
    second = np.asarray(second_residual, dtype=np.float64)
    keep = np.isfinite(first) & np.isfinite(second)
    first, second = first[keep], second[keep]
    if not len(first):
        return {"events": 0, "metrics": {}, "bootstrap_draws": draws, "seed": seed}
    sample_size = len(first)
    observed = {
        "bias": second.mean() - first.mean(),
        "mae": np.abs(second).mean() - np.abs(first).mean(),
        "rmse": np.sqrt(np.square(second).mean())
        - np.sqrt(np.square(first).mean()),
        "central_residual_width": second.std() - first.std(),
    }
    estimates = {name: np.empty(draws, dtype=np.float64) for name in observed}
    rng = np.random.default_rng(seed)
    batch_size = 32
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        indices = rng.integers(0, len(first), size=(stop - start, sample_size))
        a, b = first[indices], second[indices]
        estimates["bias"][start:stop] = b.mean(axis=1) - a.mean(axis=1)
        estimates["mae"][start:stop] = np.abs(b).mean(axis=1) - np.abs(a).mean(axis=1)
        estimates["rmse"][start:stop] = np.sqrt(np.square(b).mean(axis=1)) - np.sqrt(
            np.square(a).mean(axis=1)
        )
        estimates["central_residual_width"][start:stop] = b.std(axis=1) - a.std(axis=1)
    return {
        "events": len(first),
        "bootstrap_draws": draws,
        "bootstrap_sample_size": sample_size,
        "seed": seed,
        "metrics": {
            name: {
                "delta_second_minus_first": float(observed[name]),
                "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
            }
            for name, values in estimates.items()
        },
    }


def _constituent_parent_vectors(
    mode: str, predictions: dict, candidates: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    roots = predictions[mode]["roots"][:, 0]
    neutrinos = predictions[mode].get("nu_joint_medoid")
    if neutrinos is None:
        neutrinos = _joint_medoid(predictions[mode]["nu_samples"].astype(np.float64))
    candidates_by_role = []
    root_valid = np.ones(len(roots), dtype=bool)
    for role_index, role in enumerate(PARENT_ROLES):
        indices = roots[:, role_index]
        root_valid &= (indices >= 0) & (
            indices < candidates["candidate_vectors"][role].shape[1]
        )
        candidates_by_role.append(
            _gather_candidate(candidates["candidate_vectors"][role], indices)
        )
    nu_energy = np.linalg.norm(neutrinos, axis=-1)
    neutrino_p4 = np.concatenate([nu_energy[..., None], neutrinos], axis=-1)
    constituent = np.stack(
        [candidates_by_role[index] + neutrino_p4[:, index] for index in range(2)],
        axis=1,
    )
    root_valid &= np.isfinite(constituent).all(axis=(1, 2))
    return constituent, neutrinos, root_valid


def _cross_head_products(
    output: Path, standard: dict, parent: dict, truth: dict, candidates: dict
) -> dict:
    folder = output / "cross_head"
    folder.mkdir(parents=True, exist_ok=True)
    summary = {}
    rows = []
    for mode in ("J0", "J_A", "J_AN"):
        constituent, neutrinos, roots_valid = _constituent_parent_vectors(
            mode, standard, candidates
        )
        direct = parent[mode].get("parent_joint_medoid")
        if direct is None:
            direct = _joint_medoid(parent[mode]["parent_samples"].astype(np.float64))
        valid = (
            roots_valid
            & truth["assignment_support"]
            & truth["neutrino_support"].all(axis=1)
            & truth["parent_support"].all(axis=1)
            & np.isfinite(direct).all(axis=(1, 2))
            & np.isfinite(truth["parents"]).all(axis=(1, 2))
        )
        truth_pair = truth["parents"][:, 0] + truth["parents"][:, 1]
        constituent_pair = constituent[:, 0] + constituent[:, 1]
        direct_pair = direct[:, 0] + direct[:, 1]
        truth_values = {
            "top_pT": _p4_observables(truth["parents"][:, 0])["pT"],
            "top_y": _p4_observables(truth["parents"][:, 0])["y"],
            "antitop_pT": _p4_observables(truth["parents"][:, 1])["pT"],
            "antitop_y": _p4_observables(truth["parents"][:, 1])["y"],
            "mttbar": _mass(truth_pair),
        }
        constituent_values = {
            "top_pT": _p4_observables(constituent[:, 0])["pT"],
            "top_y": _p4_observables(constituent[:, 0])["y"],
            "antitop_pT": _p4_observables(constituent[:, 1])["pT"],
            "antitop_y": _p4_observables(constituent[:, 1])["y"],
            "mttbar": _mass(constituent_pair),
        }
        direct_values = {
            "top_pT": _p4_observables(direct[:, 0])["pT"],
            "top_y": _p4_observables(direct[:, 0])["y"],
            "antitop_pT": _p4_observables(direct[:, 1])["pT"],
            "antitop_y": _p4_observables(direct[:, 1])["y"],
            "mttbar": _mass(direct_pair),
        }
        summary[mode] = {"events": 0, "observables": {}}
        for name, target_values_for_observable in truth_values.items():
            local = valid & np.isfinite(target_values_for_observable)
            local &= np.isfinite(constituent_values[name]) & np.isfinite(direct_values[name])
            target = target_values_for_observable[local]
            reco_constituent = constituent_values[name][local]
            reco_direct = direct_values[name][local]
            residual_constituent = reco_constituent - target
            residual_direct = reco_direct - target
            summary[mode]["events"] = int(local.sum())
            summary[mode]["observables"][name] = {
                "constituent_minus_truth": _residual_summary(residual_constituent),
                "parent_cfm_minus_truth": _residual_summary(residual_direct),
                "parent_cfm_minus_constituent": _residual_summary(
                    reco_direct - reco_constituent
                ),
                "parent_cfm_improves_fraction": float(
                    np.mean(np.abs(residual_direct) < np.abs(residual_constituent))
                )
                if local.any()
                else None,
                "constituent_improves_fraction": float(
                    np.mean(np.abs(residual_constituent) < np.abs(residual_direct))
                )
                if local.any()
                else None,
            }
            record = summary[mode]["observables"][name]
            rows.append(
                {
                    "model": LABELS[mode],
                    "observable": name,
                    "events": int(local.sum()),
                    "constituent_truth_bias": record["constituent_minus_truth"]["bias"],
                    "constituent_truth_rmse": record["constituent_minus_truth"]["rmse"],
                    "parent_truth_bias": record["parent_cfm_minus_truth"]["bias"],
                    "parent_truth_rmse": record["parent_cfm_minus_truth"]["rmse"],
                    "parent_minus_constituent_bias": record[
                        "parent_cfm_minus_constituent"
                    ]["bias"],
                    "parent_cfm_improves_fraction": record[
                        "parent_cfm_improves_fraction"
                    ],
                    "constituent_improves_fraction": record[
                        "constituent_improves_fraction"
                    ],
                }
            )
            fig, ax = plt.subplots(figsize=(6.4, 5.2))
            finite = np.isfinite(reco_constituent) & np.isfinite(reco_direct)
            lower = float(
                min(np.min(reco_constituent[finite]), np.min(reco_direct[finite]))
            )
            upper = float(
                max(np.max(reco_constituent[finite]), np.max(reco_direct[finite]))
            )
            ax.scatter(
                reco_constituent[finite],
                reco_direct[finite],
                s=3,
                alpha=0.12,
                color=COLORS[mode],
                rasterized=True,
            )
            ax.plot([lower, upper], [lower, upper], color="black", linestyle=":")
            ax.set(
                xlabel=f"Constituent reconstruction {name}",
                ylabel=f"Parent-CFM reconstruction {name}",
                title=f"{LABELS[mode]}: common {int(local.sum()):,} supported events",
            )
            _save(fig, folder / f"{mode}_{name}_parent_vs_constituent")

        if mode == "J_AN":
            target_nu = truth["neutrino"]
            predicted_nu = neutrinos
            target_obs = _nu_observables(target_nu)
            predicted_obs = _nu_observables(predicted_nu)
            dr = np.sqrt(
                (predicted_obs["eta"] - target_obs["eta"]) ** 2
                + _wrap(predicted_obs["phi"] - target_obs["phi"]) ** 2
            )
            max_dr = np.max(dr, axis=1)
            mass_truth = truth_values["mttbar"]
            mass_constituent = constituent_values["mttbar"]
            mass_direct = direct_values["mttbar"]
            local = valid & np.isfinite(max_dr) & np.isfinite(mass_truth)
            quartile_edges = np.unique(np.quantile(max_dr[local], [0, 0.25, 0.5, 0.75, 1]))
            correlation = []
            if len(quartile_edges) >= 2:
                bins = np.digitize(max_dr, quartile_edges[1:-1], right=True)
                for index in range(len(quartile_edges) - 1):
                    selected = local & (bins == index)
                    direct_better = np.abs(mass_direct - mass_truth) < np.abs(
                        mass_constituent - mass_truth
                    )
                    correlation.append(
                        {
                            "max_neutrino_deltaR_low": float(quartile_edges[index]),
                            "max_neutrino_deltaR_high": float(quartile_edges[index + 1]),
                            "events": int(selected.sum()),
                            "parent_cfm_mttbar_improves_fraction": float(
                                direct_better[selected].mean()
                            )
                            if selected.any()
                            else None,
                        }
                    )
            summary[mode]["parent_improvement_vs_neutrino_quality"] = correlation

    (folder / "cross_head_metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    if rows:
        with (output / "tables/cross_head.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def _posterior_physicality(output: Path, standard: dict, parent: dict) -> dict:
    result = {"filter_applied": False, "models": {}}
    for mode in STANDARD:
        samples = standard[mode]["nu_samples"].astype(np.float64)
        finite_components = np.isfinite(samples).all(axis=-1)
        p3_obs = _nu_observables(samples)
        finite_derived = np.ones(finite_components.shape, dtype=bool)
        for values in p3_obs.values():
            finite_derived &= np.isfinite(values)
        count = int(finite_components.size)
        result["models"][mode] = {
            "neutrino": {
                "raw_role_samples": count,
                "finite_components": int(finite_components.sum()),
                "finite_derived_pt_eta_phi": int(finite_derived.sum()),
                "finite_fraction": float(finite_components.mean()),
                "raw_samples_retained": True,
            }
        }
    for mode in PARENT:
        result["models"].setdefault(mode, {})
        samples = parent[mode]["parent_samples"].astype(np.float64)
        finite = np.isfinite(samples).all(axis=-1)
        energy = samples[..., 0]
        momentum2 = np.square(samples[..., 1:]).sum(axis=-1)
        mass2 = energy**2 - momentum2
        scale = np.maximum(energy**2 + momentum2, 1.0)
        positive_energy = energy > 0
        physical_mass2 = mass2 >= -1.0e-10 * scale
        physical = finite & positive_energy & physical_mass2
        roles = {}
        for role_index, role in enumerate(PARENT_ROLES):
            mass = _mass(samples[:, :, role_index])
            valid_mass = np.isfinite(mass)
            role_finite = finite[:, :, role_index]
            role_energy = positive_energy[:, :, role_index]
            role_mass2 = physical_mass2[:, :, role_index]
            roles[role] = {
                "raw_samples": int(role_finite.size),
                "finite_samples": int(role_finite.sum()),
                "nonpositive_energy_samples": int((role_finite & ~role_energy).sum()),
                "space_like_beyond_tolerance": int((role_finite & ~role_mass2).sum()),
                "physical_samples": int((role_finite & role_energy & role_mass2).sum()),
                "physical_fraction": float((role_finite & role_energy & role_mass2).mean()),
                "mass_gev_quantiles": {
                    f"q{quantile:g}": float(np.quantile(mass[valid_mass], quantile / 100))
                    for quantile in (1, 50, 99, 99.9)
                }
                if valid_mass.any()
                else None,
                "mass_over_500_gev_count": int((mass > 500).sum()),
                "mass_max_gev": float(np.nanmax(mass)) if valid_mass.any() else None,
            }
        pair = samples[:, :, 0] + samples[:, :, 1]
        pair_mass = _mass(pair)
        pair_finite = np.isfinite(pair).all(axis=-1) & np.isfinite(pair_mass)
        result["models"][mode]["parent"] = {
            "raw_event_samples": int(samples.shape[0] * samples.shape[1]),
            "finite_ordered_parent_samples": int(finite.all(axis=-1).sum()),
            "physical_ordered_parent_samples": int(physical.all(axis=-1).sum()),
            "physical_fraction": float(physical.all(axis=-1).mean()),
            "roles": roles,
            "mttbar_gev_quantiles": {
                f"q{quantile:g}": float(np.quantile(pair_mass[pair_finite], quantile / 100))
                for quantile in (1, 50, 99, 99.9)
            }
            if pair_finite.any()
            else None,
            "mttbar_over_3000_gev_count": int((pair_mass > 3000).sum()),
            "mttbar_max_gev": float(np.nanmax(pair_mass)) if pair_finite.any() else None,
            "accepted_fraction_if_physicality_filter_were_applied": float(
                physical.all(axis=-1).mean()
            ),
        }
    (output / "posterior/physicality.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def _population_reweighting(
    output: Path,
    standard: dict,
    parent: dict,
    truth: dict,
    assignment_correct: dict,
) -> dict:
    folder = output / "robustness"
    folder.mkdir(parents=True, exist_ok=True)
    truth_mass = _mass(truth["parents"][:, 0] + truth["parents"][:, 1])
    valid_mass = np.isfinite(truth_mass)
    region_masks = {
        "threshold_lt_400": valid_mass & (truth_mass < 400),
        "mid_400_700": valid_mass & (truth_mass >= 400) & (truth_mass < 700),
        "high_ge_700": valid_mass & (truth_mass >= 700),
    }
    counts = {name: int(mask.sum()) for name, mask in region_masks.items()}
    below_nominal_threshold = valid_mass & (truth_mass < 345)
    contract = _json(output / "evaluation_contract.json")
    expected_below = int(
        contract["spectrum_reweighting"]["expected_events_below_345_gev"]
    )
    if not valid_mass.all() or any(value == 0 for value in counts.values()):
        raise ValueError(
            f"Truth mttbar cannot define the frozen spectrum weights: counts={counts}, "
            f"below_345={int(below_nominal_threshold.sum())}, nonfinite={int((~valid_mass).sum())}"
        )
    if int(below_nominal_threshold.sum()) != expected_below:
        raise ValueError(
            "Truth mttbar below the nominal pair threshold differs from the frozen validation: "
            f"expected={expected_below}, observed={int(below_nominal_threshold.sum())}"
        )
    populations = {
        "nominal_validation": None,
        "approximately_equal_threshold_mid_high": (1 / 3, 1 / 3, 1 / 3),
        "threshold_enhanced_50_30_20": (0.50, 0.30, 0.20),
        "high_mass_enhanced_15_25_60": (0.15, 0.25, 0.60),
    }
    region_names = tuple(region_masks)
    weights_by_population = {}
    diagnostics = {}
    region_share = {
        name: counts[name] / len(truth_mass) for name in region_names
    }
    for population, target_shares in populations.items():
        if target_shares is None:
            weights = np.ones(len(truth_mass), dtype=np.float64)
        else:
            weights = np.zeros(len(truth_mass), dtype=np.float64)
            for name, share in zip(region_names, target_shares, strict=True):
                weights[region_masks[name]] = share / region_share[name]
            weights = np.minimum(weights, 5.0)
        weight_sum = float(weights.sum())
        weights_by_population[population] = weights
        diagnostics[population] = {
            "target_region_shares": (
                dict(zip(region_names, target_shares, strict=True))
                if target_shares is not None
                else {name: region_share[name] for name in region_names}
            ),
            "realized_region_shares": {
                name: float(weights[mask].sum() / weight_sum)
                for name, mask in region_masks.items()
            },
            "maximum_weight": float(weights.max()),
            "mean_weight": float(weights.mean()),
            "events": len(weights),
            "effective_sample_size": float(weight_sum**2 / np.square(weights).sum()),
            "effective_sample_size_fraction": float(
                (weight_sum**2 / np.square(weights).sum()) / len(weights)
            ),
            "capped_at_five": target_shares is not None,
        }

    metric_rows = []
    result = {
        "interpretation": "Population-reweighting sensitivity only; it is not out-of-distribution validation.",
        "truth_mttbar_region_counts": counts,
        "truth_mttbar_below_345_gev_events": int(below_nominal_threshold.sum()),
        "weight_cap": 5.0,
        "populations": diagnostics,
        "metrics": {},
    }
    for population, weights in weights_by_population.items():
        result["metrics"][population] = {"models": {}}
        for mode in STANDARD:
            item = {}
            supported = standard[mode]["assignment_supported"]
            local_weights = weights[supported]
            item["complete_assignment_efficiency"] = float(
                np.average(assignment_correct[mode][supported], weights=local_weights)
            )
            item["complete_assignment_supported_events"] = int(supported.sum())
            medoid = standard[mode].get("nu_joint_medoid")
            if medoid is None:
                medoid = _joint_medoid(
                    standard[mode]["nu_samples"].astype(np.float64)
                )
            for role_index, role in enumerate(NU_ROLES):
                support = truth["neutrino_support"][:, role_index]
                residual = medoid[:, role_index, 2] - truth["neutrino"][:, role_index, 2]
                selected = support & np.isfinite(residual)
                w = weights[selected]
                item[f"{role}_pz_bias_gev"] = float(np.average(residual[selected], weights=w))
                item[f"{role}_pz_mae_gev"] = float(
                    np.average(np.abs(residual[selected]), weights=w)
                )
            result["metrics"][population]["models"].setdefault(mode, {}).update(item)
            for name, value in item.items():
                metric_rows.append(
                    {
                        "population": population,
                        "model": LABELS[mode],
                        "metric": name,
                        "weighted_value": value,
                        "effective_sample_size": diagnostics[population][
                            "effective_sample_size"
                        ],
                    }
                )
        for mode in PARENT:
            medoid = parent[mode].get("parent_joint_medoid")
            if medoid is None:
                medoid = _joint_medoid(
                    parent[mode]["parent_samples"].astype(np.float64)
                )
            predicted_mass = _mass(medoid[:, 0] + medoid[:, 1])
            residual = predicted_mass - truth_mass
            selected = np.isfinite(residual)
            local_weights = weights[selected]
            item = {
                "mttbar_bias_gev": float(np.average(residual[selected], weights=local_weights)),
                "mttbar_mae_gev": float(
                    np.average(np.abs(residual[selected]), weights=local_weights)
                ),
                "mttbar_rmse_gev": float(
                    np.sqrt(np.average(np.square(residual[selected]), weights=local_weights))
                ),
            }
            result["metrics"][population]["models"].setdefault(mode, {}).update(item)
            for name, value in item.items():
                metric_rows.append(
                    {
                        "population": population,
                        "model": LABELS[mode],
                        "metric": name,
                        "weighted_value": value,
                        "effective_sample_size": diagnostics[population][
                            "effective_sample_size"
                        ],
                    }
                )
    (folder / "population_reweighting.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    with (output / "tables/population_reweighting.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    (folder / "README.md").write_text(
        "These deterministic truth-mttbar population weights test sensitivity to the validation spectrum. "
        f"The threshold region is below 400 GeV and includes {int(below_nominal_threshold.sum())} truth pairs below 345 GeV; "
        "the nominal 345–400 GeV response is reported separately. Mid is 400–700 GeV and high is >=700 GeV. Weights are capped at 5, "
        "and report realized region shares and effective sample size. This is not out-of-distribution validation. "
        "Conditional response versus truth mttbar remains in the parent and threshold products.\n"
    )
    return result


def _write_event_metrics(
    output: Path,
    ids: np.ndarray,
    truth: dict,
    standard: dict,
    parent: dict,
    assignment_correct: dict,
    candidates: dict,
) -> None:
    path = output / "event_metrics.h5"
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as target:
        target.attrs["schema"] = "realnu_event_metrics_v1"
        target.attrs["sealed_test_used"] = False
        target.attrs["neutrino_role_order"] = json.dumps(NU_ROLES)
        target.attrs["parent_role_order"] = json.dumps(PARENT_ROLES)
        target.attrs["source_njet_definition"] = (
            "round(INPUTS/GLOBAL.njet) joined by EVENT_ID"
        )
        target.attrs["model_visible_njet_definition"] = (
            "count finite INPUTS/JET.pt > 0 slots joined by EVENT_ID"
        )
        target.create_dataset("EVENT_ID", data=ids, compression="gzip", compression_opts=4)
        target.create_dataset("source_njet", data=truth["source_njet"])
        target.create_dataset("model_visible_njet", data=truth["model_visible_njet"])
        target.create_dataset(
            "channel", data=np.asarray(truth["channel"], dtype=object), dtype=string_type
        )
        truth_group = target.create_group("truth")
        truth_group.create_dataset("neutrinos", data=truth["neutrino"], compression="gzip")
        truth_group.create_dataset("parents", data=truth["parents"], compression="gzip")
        truth_group.create_dataset("met", data=truth["met"])
        truth_group.create_dataset(
            "mttbar", data=_mass(truth["parents"][:, 0] + truth["parents"][:, 1])
        )
        support_group = target.create_group("support")
        support_group.create_dataset("neutrino_roles", data=truth["neutrino_support"])
        support_group.create_dataset("parent_roles", data=truth["parent_support"])
        support_group.create_dataset("assignment_roles", data=truth["assignment_support"])
        support_group.create_dataset(
            "assignment_role_candidates",
            data=np.stack(
                [candidates["support"][role] for role in PARENT_ROLES], axis=1
            ),
        )
        models = target.create_group("models")
        for mode in STANDARD:
            group = models.create_group(mode)
            group.attrs["label"] = LABELS[mode]
            group.create_dataset(
                "assignment_supported", data=standard[mode]["assignment_supported"]
            )
            group.create_dataset("assignment_complete_correct", data=assignment_correct[mode])
            group.create_dataset(
                "assignment_root_indices", data=standard[mode]["roots"][:, 0]
            )
            assignment_role_supported = np.stack(
                [
                    truth["assignment_support"]
                    & candidates["support"][role]
                    & (candidates["truth_index"][role] >= 0)
                    for role in PARENT_ROLES
                ],
                axis=1,
            )
            assignment_role_correct = np.stack(
                [
                    (standard[mode]["roots"][:, 0, role_index]
                    == candidates["truth_index"][role])
                    & assignment_role_supported[:, role_index]
                    for role_index, role in enumerate(PARENT_ROLES)
                ],
                axis=1,
            )
            group.create_dataset(
                "assignment_truth_candidate_indices",
                data=np.stack(
                    [candidates["truth_index"][role] for role in PARENT_ROLES],
                    axis=1,
                ),
            )
            group.create_dataset(
                "assignment_role_supported", data=assignment_role_supported
            )
            group.create_dataset("assignment_role_correct", data=assignment_role_correct)
            nu_point = standard[mode].get("nu_joint_medoid")
            if nu_point is None:
                nu_point = _joint_medoid(
                    standard[mode]["nu_samples"].astype(np.float64)
                )
            group.create_dataset("neutrino_joint_medoid", data=nu_point, compression="gzip")
            group.create_dataset(
                "neutrino_residual_joint_medoid",
                data=nu_point - truth["neutrino"],
                compression="gzip",
            )
            neutrino_sum = nu_point.sum(axis=1)
            group.create_dataset("neutrino_momentum_sum", data=neutrino_sum)
            group.create_dataset(
                "met_residual_joint_medoid",
                data=neutrino_sum[:, :2] - truth["met"],
            )
            constituent, _, root_valid = _constituent_parent_vectors(
                mode, standard, candidates
            )
            group.create_dataset("constituent_parent", data=constituent, compression="gzip")
            group.create_dataset("constituent_parent_valid", data=root_valid)
            truth_candidate = []
            predicted_candidate = []
            for role_index, role in enumerate(PARENT_ROLES):
                truth_candidate.append(
                    _gather_candidate(
                        candidates["candidate_vectors"][role],
                        candidates["truth_index"][role],
                    )
                )
                predicted_candidate.append(
                    _gather_candidate(
                        candidates["candidate_vectors"][role],
                        standard[mode]["roots"][:, 0, role_index],
                    )
                )
            truth_nu = truth["neutrino"]
            truth_nu_p4 = np.concatenate(
                [np.linalg.norm(truth_nu, axis=-1)[..., None], truth_nu], axis=-1
            )
            predicted_nu_p4 = np.concatenate(
                [np.linalg.norm(nu_point, axis=-1)[..., None], nu_point], axis=-1
            )
            oracle_group = group.create_group("constituent_oracles")
            oracle_variants = {
                "perfect_reco": np.stack(
                    [truth_candidate[index] + truth_nu_p4[:, index] for index in range(2)],
                    axis=1,
                ),
                "predicted_assignment_truth_neutrinos": np.stack(
                    [predicted_candidate[index] + truth_nu_p4[:, index] for index in range(2)],
                    axis=1,
                ),
                "truth_assignment_predicted_neutrinos": np.stack(
                    [truth_candidate[index] + predicted_nu_p4[:, index] for index in range(2)],
                    axis=1,
                ),
                "predicted_assignment_predicted_neutrinos": constituent,
            }
            for name, vectors in oracle_variants.items():
                variant = oracle_group.create_group(name)
                variant.create_dataset("parents", data=vectors, compression="gzip")
                variant.create_dataset(
                    "parent_residual", data=vectors - truth["parents"], compression="gzip"
                )
                variant.create_dataset(
                    "mttbar",
                    data=_mass(vectors[:, 0] + vectors[:, 1]),
                )
            if mode in PARENT:
                parent_point = parent[mode].get("parent_joint_medoid")
                if parent_point is None:
                    parent_point = _joint_medoid(
                        parent[mode]["parent_samples"].astype(np.float64)
                    )
                group.create_dataset("parent_joint_medoid", data=parent_point, compression="gzip")
                group.create_dataset(
                    "parent_residual_joint_medoid",
                    data=parent_point - truth["parents"],
                    compression="gzip",
                )
                group.create_dataset(
                    "parent_mttbar_joint_medoid",
                    data=_mass(parent_point[:, 0] + parent_point[:, 1]),
                )
        group = models.create_group("PCFM_Q")
        group.attrs["label"] = LABELS["PCFM_Q"]
        parent_point = parent["PCFM_Q"].get("parent_joint_medoid")
        if parent_point is None:
            parent_point = _joint_medoid(
                parent["PCFM_Q"]["parent_samples"].astype(np.float64)
            )
        group.create_dataset("parent_joint_medoid", data=parent_point, compression="gzip")
        group.create_dataset(
            "parent_residual_joint_medoid",
            data=parent_point - truth["parents"],
            compression="gzip",
        )
        group.create_dataset(
            "parent_mttbar_joint_medoid",
            data=_mass(parent_point[:, 0] + parent_point[:, 1]),
        )


def _paired_table(
    output: Path, predictions: dict, truth: dict, assignment_correct: dict
) -> dict:
    rows = []
    results = {}
    for comparison_index, (first, second, label) in enumerate(COMPARISONS):
        item = {}
        if first in assignment_correct and second in assignment_correct:
            common = (
                predictions[first]["assignment_supported"]
                & predictions[second]["assignment_supported"]
            )
            bootstrap = _bootstrap_delta(
                assignment_correct[first][common].astype(float),
                assignment_correct[second][common].astype(float),
                seed=730241 + comparison_index,
                draws=2000,
            )
            first_correct = assignment_correct[first][common]
            second_correct = assignment_correct[second][common]
            transition = {
                "both_correct": int(np.sum(first_correct & second_correct)),
                "first_only_correct": int(np.sum(first_correct & ~second_correct)),
                "second_only_correct": int(np.sum(~first_correct & second_correct)),
                "both_incorrect": int(np.sum(~first_correct & ~second_correct)),
            }
            item["complete_assignment_efficiency"] = {
                **bootstrap,
                "common_supported_events": int(common.sum()),
                "first_efficiency": float(first_correct.mean()) if common.any() else None,
                "second_efficiency": float(second_correct.mean()) if common.any() else None,
                "paired_transition_counts": transition,
            }
        if first in STANDARD and second in STANDARD:
            first_point = predictions[first].get("nu_joint_medoid")
            if first_point is None:
                first_point = _joint_medoid(
                    predictions[first]["nu_samples"].astype(np.float64)
                )
            second_point = predictions[second].get("nu_joint_medoid")
            if second_point is None:
                second_point = _joint_medoid(
                    predictions[second]["nu_samples"].astype(np.float64)
                )
            for role_index, role in enumerate(NU_ROLES):
                target = truth["neutrino"][:, role_index]
                for name, first_values, second_values, target_value in (
                    (
                        "pz",
                        first_point[:, role_index, 2],
                        second_point[:, role_index, 2],
                        target[:, 2],
                    ),
                    (
                        "pT",
                        np.hypot(first_point[:, role_index, 0], first_point[:, role_index, 1]),
                        np.hypot(second_point[:, role_index, 0], second_point[:, role_index, 1]),
                        np.hypot(target[:, 0], target[:, 1]),
                    ),
                ):
                    support = truth["neutrino_support"][:, role_index]
                    first_residual = first_values[support] - target_value[support]
                    second_residual = second_values[support] - target_value[support]
                    item[f"{role}_{name}"] = _paired_metric_bootstrap(
                        first_residual,
                        second_residual,
                        seed=730241 + comparison_index * 31 + role_index * 7 + (name == "pT"),
                        draws=2000,
                    )
        if first in PARENT and second in PARENT:
            first_point = predictions[first].get("parent_joint_medoid")
            if first_point is None:
                first_point = _joint_medoid(
                    predictions[first]["parent_samples"].astype(np.float64)
                )
            second_point = predictions[second].get("parent_joint_medoid")
            if second_point is None:
                second_point = _joint_medoid(
                    predictions[second]["parent_samples"].astype(np.float64)
                )
            first_pair, second_pair = first_point[:, 0] + first_point[:, 1], second_point[:, 0] + second_point[:, 1]
            truth_pair = truth["parents"][:, 0] + truth["parents"][:, 1]
            first_values = {
                "mttbar": _mass(first_pair),
                "top_pT": _p4_observables(first_point[:, 0])["pT"],
                "top_y": _p4_observables(first_point[:, 0])["y"],
                "antitop_pT": _p4_observables(first_point[:, 1])["pT"],
                "antitop_y": _p4_observables(first_point[:, 1])["y"],
            }
            second_values = {
                "mttbar": _mass(second_pair),
                "top_pT": _p4_observables(second_point[:, 0])["pT"],
                "top_y": _p4_observables(second_point[:, 0])["y"],
                "antitop_pT": _p4_observables(second_point[:, 1])["pT"],
                "antitop_y": _p4_observables(second_point[:, 1])["y"],
            }
            target_values = {
                "mttbar": _mass(truth_pair),
                "top_pT": _p4_observables(truth["parents"][:, 0])["pT"],
                "top_y": _p4_observables(truth["parents"][:, 0])["y"],
                "antitop_pT": _p4_observables(truth["parents"][:, 1])["pT"],
                "antitop_y": _p4_observables(truth["parents"][:, 1])["y"],
            }
            for metric_index, name in enumerate(first_values):
                support = np.isfinite(target_values[name])
                first_residual = first_values[name][support] - target_values[name][support]
                second_residual = second_values[name][support] - target_values[name][support]
                item[name] = _paired_metric_bootstrap(
                    first_residual,
                    second_residual,
                    seed=730241 + comparison_index * 31 + metric_index,
                    draws=2000,
                )
        results[f"{first}_vs_{second}"] = item
        for metric_name, record in item.items():
            rows.append({"comparison": label, "metric": metric_name, **record})
    with (output / "tables/paired_bootstrap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row))
        )
        writer.writeheader()
        writer.writerows(rows)
    (output / "tables/paired_bootstrap.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    return results


def _choose_final_architecture(parent_summary: dict) -> dict:
    candidates = ("PCFM_Q", "J0", "J_A", "J_AN")
    point_rows = parent_summary["point_estimators"]
    by_key = {}
    for row in point_rows:
        if row.get("reducer") == "joint_medoid":
            by_key[(row["model"], row["role"], row["observable"])] = row.get("rmse")
    metrics = (
        ("inclusive_mttbar_rmse_gev", "ttbar", "mttbar"),
        ("top_pT_rmse_gev", "top", "pT"),
        ("top_rapidity_rmse", "top", "y"),
        ("antitop_pT_rmse_gev", "antitop", "pT"),
        ("antitop_rapidity_rmse", "antitop", "y"),
    )
    values = {}
    for mode in candidates:
        row = {}
        for metric_name, role, observable in metrics:
            row[metric_name] = by_key.get((LABELS[mode], role, observable))
        row["threshold_mttbar_rmse_gev"] = parent_summary["threshold"][mode][
            "threshold_345_400"
        ]["rmse_gev"]
        if any(value is None or not np.isfinite(value) for value in row.values()):
            raise RuntimeError(
                f"Cannot freeze the final architecture: {mode} lacks finite medoid metrics {row}"
            )
        values[mode] = row
    frontier = []
    for mode in candidates:
        candidate_values = np.asarray(list(values[mode].values()), dtype=np.float64)
        dominated = False
        for other in candidates:
            if other == mode:
                continue
            other_values = np.asarray(list(values[other].values()), dtype=np.float64)
            if np.all(other_values <= candidate_values) and np.any(
                other_values < candidate_values
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(mode)
    selected = frontier[0]
    return {
        "chosen_mode": selected,
        "chosen_architecture": LABELS[selected],
        "selection_metrics": list(values[selected]),
        "metrics_by_mode": values,
        "pareto_frontier": frontier,
        "selection_rule": (
            "Remove Pareto-dominated candidates across inclusive mttbar RMSE, 345–400 GeV mttbar RMSE, "
            "ordered top/antitop pT RMSE, and ordered top/antitop rapidity RMSE. "
            "From the non-dominated set select the least-conditioned architecture in order PCFM_Q, J0, J_A, J_AN."
        ),
        "aggregate_score_used": False,
    }


def _write_final_reports(
    output: Path,
    report: dict,
    assignment_summary: dict,
    neutrino_summary: dict,
    parent_summary: dict,
    cross_head: dict,
    oracle: dict,
    paired: dict,
    reweighting: dict,
    physicality: dict,
) -> dict:
    final_dir = output / "report"
    final_dir.mkdir(parents=True, exist_ok=True)
    selection = _choose_final_architecture(parent_summary)
    decision = {
        "status": "frozen_on_validation_only",
        "training_source_sha": report["training_source_sha"],
        "evaluation_recovery_sha": report.get("evaluation_recovery_sha"),
        "chosen_architecture": selection["chosen_architecture"],
        "chosen_mode": selection["chosen_mode"],
        "architecture_selection": selection,
        "nominal_coherent_event_estimator": "joint posterior medoid; select one complete correlated top/antitop sample",
        "posterior_sample_count": 64,
        "flow_solver_and_nfe": report["runtime"],
        "physicality_treatment": {
            "posterior_samples_filtered": False,
            "raw_physicality_audit": physicality,
        },
        "truth_and_reference_definitions": report["references"],
        "mttbar_binning_gev": {
            "primary": [345, 400, 500, 700, 1000, "open-ended"],
            "fine_threshold": [345, 360, 375, 390, 400, 415, 430, 450],
        },
        "primary_metrics": [
            "complete-event assignment efficiency",
            "neutrino pT and pz residuals",
            "ordered top/antitop pT and rapidity RMSE",
            "mttbar RMSE",
            "345–400 GeV mttbar RMSE",
            "posterior PIT, 68% and 90% coverage, interval width, and energy score",
        ],
        "classical_baseline_status": (output / "classical_baseline_status.md").read_text(),
        "spin_observable_status": (output / "spin/README.md").read_text(),
        "validation_only_decisions": [
            "Use the frozen 100342-event validation EVENT_ID set and declared support masks.",
            "Physics-facing multiplicity is source_njet; model_visible_njet is retained as a separate maxjets8 diagnostic.",
            "Reuse all five original standard prediction files; reuse the passing PCFM_Q parent validation.",
            "Derive ttbar only as the four-vector sum of ordered top and antitop.",
            "Use truth assignment plus truth neutrinos for perfect reconstruction and keep assignment/neutrino oracles separate.",
            "Retain every raw posterior sample; no physicality filter is applied.",
            "Use truth-mttbar population reweighting as a sensitivity check only, not as OOD validation.",
            "No c_hel/c_han convention is added because truth charged-lepton four-vectors are unavailable.",
            "No classical dilepton baseline is integrated because no validated local implementation passed the reuse audit.",
            "No training, checkpoint, split, scaler, cache, architecture, or sealed-test input was changed or used.",
        ],
        "posterior_summary": {
            "neutrino": neutrino_summary,
            "parent": parent_summary["posterior"],
            "parent_region_quality": parent_summary["posterior_region_quality"],
            "parent_mttbar_migration": parent_summary["posterior_migration"],
        },
        "cross_head": cross_head,
        "oracle_decomposition": oracle,
        "paired_comparisons": paired,
        "population_reweighting": reweighting,
        "sealed_test_used": False,
    }
    (final_dir / "decision_freeze.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n"
    )
    (final_dir / "DECISION_FREEZE.md").write_text(
        "# Validation decision freeze\n\n"
        f"- Chosen architecture: **{selection['chosen_architecture']}** (`{selection['chosen_mode']}`).\n"
        f"- Pareto frontier: {', '.join(selection['pareto_frontier'])}.\n"
        "- Nominal event estimator: joint posterior medoid, selecting one correlated whole-event top/antitop sample.\n"
        "- Posterior count: 64 nested deterministic samples per event.\n"
        "- Physicality: raw samples retained and audited; no sample filter applied.\n"
        "- Sealed test used: false.\n\n"
        f"Selection rule: {selection['selection_rule']}\n\n"
        "The full machine-readable choices, NFE, references, binning, comparisons, and validation-only decisions are in `decision_freeze.json`.\n"
    )

    q0q1 = paired.get("Q0_vs_Q1", {})
    q0q1_assign = q0q1.get("complete_assignment_efficiency", {})
    pair_labels = (
        ("PCFM_Q_vs_J0", "Direct parent flow → joint multitask"),
        ("J0_vs_J_A", "Joint multitask → assignment-aware"),
        ("J_A_vs_J_AN", "Assignment-aware → assignment + neutrino-aware"),
    )
    lines = [
        "# Final real-neutrino validation report",
        "",
        "**Evaluation:** R2 validation only; sealed test rows read: 0.",
        f"**Validation events:** {report['validation_events']:,}.",
        f"**Training source:** `{report['training_source_sha']}`.",
        f"**Selected final architecture:** {selection['chosen_architecture']} (`{selection['chosen_mode']}`).",
        "",
        "## A. Do topology queries help?",
        "",
        "The table reports paired-event complete-assignment performance and the joint-medoid neutrino response. Assignment differences use the common support mask; neutrino quantities use the full role support.",
        "",
        "| Model | Assignment support | Complete-event efficiency | top.nu pz RMSE [GeV] | antitop.nu pz RMSE [GeV] |",
        "|---|---:|---:|---:|---:|",
    ]
    neutrino_rows = []
    with (output / "tables/neutrino_metrics.csv").open(newline="") as stream:
        neutrino_rows = list(csv.DictReader(stream))
    pz_metrics = {}
    for row in neutrino_rows:
        if row.get("observable") == "pz:joint_medoid":
            pz_metrics[(row["model"], row["role"])] = row.get("rmse")
    for mode in STANDARD:
        assigned = assignment_summary["complete_event"][mode]
        top_rmse = pz_metrics.get((LABELS[mode], "top.nu"), "")
        anti_rmse = pz_metrics.get((LABELS[mode], "antitop.nu"), "")
        lines.append(
            f"| {LABELS[mode]} | {assigned['supported_events']:,} | {assigned['efficiency']:.4f} | {float(top_rmse):.3f} | {float(anti_rmse):.3f} |"
        )
    lines.extend(
        [
            "",
            f"Q0 → Q1 paired assignment delta: {q0q1_assign.get('mean_delta', float('nan')):+.4f}, 95% bootstrap CI {q0q1_assign.get('ci95')}; common supported events {q0q1_assign.get('common_supported_events', 0):,}.",
            "The same comparison includes paired pT and pz bias, MAE, RMSE, and residual-width intervals in `tables/paired_bootstrap.json`.",
            "Predicted-neutrino-system x/y closure against measured MET and the residual magnitude are in `neutrinos/met_closure.json` and `tables/met_closure.csv`; no closure constraint is imposed.",
            "",
            "## B–D. Does parent conditioning improve direct parent inference?",
            "",
            "| Comparison | mttbar RMSE delta [GeV] | 345–400 GeV RMSE delta [GeV] |",
            "|---|---:|---:|",
        ]
    )
    for key, label in pair_labels:
        record = paired.get(key, {}).get("mttbar", {})
        delta = record.get("metrics", {}).get("rmse", {}).get("delta_second_minus_first")
        ci = record.get("metrics", {}).get("rmse", {}).get("ci95")
        first_mode, second_mode = key.split("_vs_")
        first_threshold = parent_summary["threshold"][first_mode]["threshold_345_400"]["rmse_gev"]
        second_threshold = parent_summary["threshold"][second_mode]["threshold_345_400"]["rmse_gev"]
        threshold_delta = second_threshold - first_threshold
        lines.append(
            f"| {label} | {delta:+.3f} (95% CI {ci}) | {threshold_delta:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Negative RMSE deltas favour the second model. The architecture choice uses a Pareto frontier across inclusive and threshold mttbar plus ordered top/antitop pT and rapidity; it does not combine observables into an arbitrary score.",
            "",
            "## E. Does the parent CFM add value beyond constituent reconstruction?",
            "",
            "For J0, J_A, and J_AN, the parent head is compared with the sum of the predicted visible-root four-vector and the joint-medoid neutrino four-vector on the same supported events.",
            "",
            "| Model | Observable | Events | Parent-CFM RMSE | Constituent RMSE | Parent-CFM improves |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for mode in ("J0", "J_A", "J_AN"):
        for name, record in cross_head[mode]["observables"].items():
            lines.append(
                f"| {LABELS[mode]} | {name} | {record['parent_cfm_minus_truth']['events']:,} | {record['parent_cfm_minus_truth']['rmse']:.3f} | {record['constituent_minus_truth']['rmse']:.3f} | {record['parent_cfm_improves_fraction']:.3f} |"
            )
    lines.extend(
        [
            "",
            "The J_AN parent-improvement fraction versus max(top/antitop neutrino ΔR) is in `cross_head/cross_head_metrics.json`.",
            "",
            "## F. Where does the remaining error originate?",
            "",
            "Perfect reconstruction uses truth assignment and truth neutrinos. Assignment-only, neutrino-only, and full constituent diagnostics are reported separately from direct parent reconstruction.",
            "",
            "| Model | Reconstruction variant | mttbar bias [GeV] | MAE [GeV] | RMSE [GeV] |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for mode in STANDARD:
        for variant, record in oracle[mode].items():
            lines.append(
                f"| {LABELS[mode]} | {variant.replace('_', ' ')} | {record['mttbar_bias_gev']:.3f} | {record['mttbar_mae_gev']:.3f} | {record['mttbar_rmse_gev']:.3f} |"
            )
    lines.extend(
        [
            "",
            "The decomposition keeps model-minus-perfect-reco, perfect-reco-minus-parton, and model-minus-parton residual products distinct in the event-level file and observable tables.",
            "",
            "## G. How does performance behave at mttbar threshold?",
            "",
            "| Model | Truth bin | Events | Bias [GeV] | Resolution [GeV] | MAE [GeV] | RMSE [GeV] |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in PARENT:
        for bin_name, record in parent_summary["threshold"][mode].items():
            if record["events"] == 0:
                continue
            lines.append(
                f"| {LABELS[mode]} | {bin_name.replace('_', ' ')} | {record['events']:,} | {record['bias_gev']:.3f} | {record['resolution_gev']:.3f} | {record['mae_gev']:.3f} | {record['rmse_gev']:.3f} |"
            )
    lines.extend(
        [
            "",
            "The local classical-baseline search result is below; no unvalidated approximation was added.",
            "",
            (output / "classical_baseline_status.md").read_text().strip(),
            "",
            "## H. How calibrated is the posterior?",
            "",
            "Point-estimator accuracy and posterior calibration are separate. PIT, 68%/90% coverage, interval widths, scalar energy scores, width-versus-error checks, and probabilistic mttbar migration are reported for inclusive, threshold, high-mass, and source-njet regions where support permits.",
            "",
            "Neutrino calibration: `tables/neutrino_posterior_quality.csv`; parent/mttbar calibration: `tables/parent_posterior_quality.csv`; physicality: `posterior/physicality.json`.",
            "",
            "## I. Is there evidence of mass-spectrum regression or central-prior pull?",
            "",
            "Conditional bias and resolution versus truth mttbar are in `threshold/` and `ttbar/`. The reweighting exercise is a population-sensitivity diagnostic only and does not establish OOD robustness.",
            "",
            "| Population | Effective sample size | Maximum event weight |",
            "|---|---:|---:|",
        ]
    )
    for name, record in reweighting["populations"].items():
        lines.append(
            f"| {name.replace('_', ' ')} | {record['effective_sample_size']:.1f} / {record['events']:,} | {record['maximum_weight']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Spin and angular observables",
            "",
            (output / "spin/README.md").read_text().strip(),
            "",
            "## Frozen decision",
            "",
            f"The validation-only selection is **{selection['chosen_architecture']}**. The Pareto frontier is `{selection['pareto_frontier']}`; the least-conditioned non-dominated candidate was selected. The machine-readable decision freeze is `report/decision_freeze.json`.",
            "",
            "No sealed-test rows were read.",
            "",
        ]
    )
    (final_dir / "FINAL_REALNU_VALIDATION.md").write_text("\n".join(lines))
    compute_rows = []
    for mode, model_label in LABELS.items():
        item = report["runtime"].get(mode, {})
        compute_rows.append(
            {
                "model": model_label,
                "conditioning": item.get("conditioning", "not applicable"),
                "solver": item.get("solver", "not applicable"),
                "steps": item.get("steps", "not applicable"),
                "nfe": item.get("nfe", "not applicable"),
                "posterior_samples": item.get("posterior_samples", "not applicable"),
            }
        )
    with (output / "tables/compute.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compute_rows[0]))
        writer.writeheader()
        writer.writerows(compute_rows)
    (output / "tables/compute.json").write_text(
        json.dumps(compute_rows, indent=2, allow_nan=False) + "\n"
    )
    return decision


def run(
    manifest_path: Path,
    output: Path,
    posterior_manifest_path: Path | None = None,
) -> dict:
    _style()
    manifest = _json(manifest_path)
    if (
        manifest.get("test_rows_read") != 0
        or manifest.get("inputs", {}).get("reserved_test_rows_materialised") != 0
    ):
        raise RuntimeError("Campaign manifest does not prove sealed-test exclusion")
    if manifest.get("evaluation_recovery"):
        contract_path = output / "evaluation_contract.json"
        contract = _json(contract_path)
        if (
            contract.get("evaluation_recovery_sha")
            != manifest["evaluation_recovery"].get("source_sha")
            or contract.get("training_source_sha") != manifest["source_commit"]
            or contract.get("reserved_test_used") is not False
        ):
            raise RuntimeError(
                "Frozen evaluation contract does not match the R2 campaign manifest"
            )
    output.mkdir(parents=True, exist_ok=True)
    allowed_existing = {
        "manifests",
        "evaluation_contract.json",
        "EVALUATION_CONTRACT.md",
        "classical_baseline_status.md",
        "core",
        "posterior64",
    }
    unexpected = [path.name for path in output.iterdir() if path.name not in allowed_existing]
    if unexpected:
        raise FileExistsError(
            f"Refusing to overwrite existing evaluation products under {output}: {unexpected}"
        )
    source_root = Path(manifest["source_root"]).resolve()
    project_root = source_root.parents[2]
    os.environ["JER_REALNU_SOURCE_ROOT"] = str(source_root)
    os.environ["JER_REALNU_PROJECT_ROOT"] = str(project_root)
    os.environ["JER_PROJECT_ROOT"] = str(project_root)
    os.environ["JER_SOURCE_ROOT"] = str(source_root)
    os.environ["JER_SOURCE_SHA"] = str(manifest["source_commit"])
    os.environ["JER_SOURCE_BRANCH"] = str(manifest["source_branch"])
    os.environ["PYTHONPATH"] = (
        str(source_root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    posterior_manifest = None
    if posterior_manifest_path is not None:
        posterior_manifest = _json(posterior_manifest_path.expanduser().resolve())
        if (
            posterior_manifest.get("status") != "passed"
            or posterior_manifest.get("source_commit") != manifest["source_commit"]
            or int(posterior_manifest.get("sample_count", 0)) < 64
        ):
            raise RuntimeError(
                "The 64-sample posterior extension is incomplete or from another source"
            )
    for folder in (
        "manifests",
        "event_metrics",
        "assignment",
        "neutrinos",
        "parents",
        "cross_head",
        "ttbar",
        "threshold",
        "ordering",
        "spin",
        "posterior",
        "posterior_convergence",
        "oracle_decomposition",
        "robustness",
        "classical",
        "tables",
        "report",
    ):
        (output / folder).mkdir(exist_ok=True)

    standard_h5: dict[str, Path] = {}
    parent_h5: dict[str, Path] = {}
    validation_records = {}
    for mode, entry in manifest["modes"].items():
        path = Path(entry["validation_output"]) / "validation.json"
        record = _json(path)
        if (
            record.get("status") != "passed"
            or record.get("reserved_test_rows_read") != 0
        ):
            raise RuntimeError(
                f"{mode} validation is not passed or does not exclude sealed test"
            )
        validation_records[mode] = record
        standard = record.get("standard_validation")
        parent = record.get("parent_validation")
        if posterior_manifest is not None:
            products = posterior_manifest.get("products", {}).get(mode, {})
            if standard is not None:
                standard = products.get("standard")
            if parent is not None:
                parent = products.get("parent")
            if (record.get("standard_validation") is not None and not standard) or (
                record.get("parent_validation") is not None and not parent
            ):
                raise RuntimeError(
                    f"Posterior extension is missing {mode} validation products"
                )
        if standard:
            standard_h5[mode] = Path(standard["prediction"])
        if parent:
            parent_h5[mode] = Path(parent["prediction"])
    if set(standard_h5) != set(STANDARD) or set(parent_h5) != set(PARENT):
        raise RuntimeError(
            "All five standard and four parent validation products are required"
        )

    with h5py.File(standard_h5["Q0"], "r") as source:
        ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
    if len(ids) != 100_342 or len(np.unique(ids)) != len(ids):
        raise ValueError(
            "Q0 standard validation prediction has wrong EVENT_ID population"
        )
    truth = _load_truth_and_channels(manifest, ids)
    truth["ids"] = ids
    standard = {mode: _load_standard(standard_h5[mode], ids) for mode in STANDARD}
    parent = {mode: _load_parent(parent_h5[mode], ids) for mode in PARENT}
    for mode in STANDARD:
        if not np.array_equal(
            standard[mode]["model_visible_njet"], truth["model_visible_njet"]
        ):
            raise ValueError(
                f"{mode} prediction njet differs from aligned model-visible jet count"
            )
        standard[mode]["nu_samples"] = standard[mode].pop("samples")
    for mode in PARENT:
        parent[mode]["parent_samples"] = parent[mode].pop("samples")
        parent[mode]["parent_observables"] = parent[mode].pop("observables")
        if mode == "J_AN" and (
            parent[mode]["conditioning"] != "P2"
            or parent[mode]["solver"] != "heun"
            or parent[mode]["steps"] != 32
        ):
            raise ValueError(
                "J_AN parent prediction did not retain the live P2 conditioning and 32-step Heun parent flow"
            )
        if mode == "J_AN" and parent[mode]["nfe"] != 64:
            raise ValueError(
                "J_AN parent prediction does not have the expected 64-NFE Heun sampling path"
            )
        if mode == "J_AN":
            contract = parent[mode]["conditioning_contract"]
            indices = parent[mode]["conditioning_indices"]
            samples_per_event = parent[mode]["parent_samples"].shape[1]
            if (
                contract.get("invisible_context")
                != "predicted_CFM_sample_top_antitop_role_order_one_sample_per_event"
                or not np.all(indices["assignment_hypothesis_index"] == -1)
                or not np.all(indices["invisible_sample_index"] == 0)
                or not np.array_equal(
                    indices["parent_sample_index"],
                    np.broadcast_to(
                        np.arange(samples_per_event, dtype=np.int64),
                        indices["parent_sample_index"].shape,
                    ),
                )
            ):
                raise ValueError(
                    "J_AN parent prediction lost the one-sample P2 neutrino conditioning contract"
                )
    if posterior_manifest is not None:
        for mode in STANDARD:
            if standard[mode]["posterior_samples"] < 64:
                raise RuntimeError(
                    f"{mode} standard posterior did not store at least 64 samples"
                )
        for mode in PARENT:
            if parent[mode]["posterior_samples"] < 64:
                raise RuntimeError(
                    f"{mode} parent posterior did not store at least 64 samples"
                )
    candidate = _load_candidate_vectors(manifest, "Q0", ids)
    for mode in STANDARD:
        standard[mode]["candidate_truth_index"] = candidate["truth_index"]
        standard[mode]["candidate_support"] = candidate["support"]

    assignment_correct, assignment_summary = _assignment_products(
        output, standard, truth
    )
    # Candidate vectors are the same for every shared topology/configuration.
    perfect_roots = {
        role: _gather_candidate(
            candidate["candidate_vectors"][role], candidate["truth_index"][role]
        )
        for role in PARENT_ROLES
    }
    truth_nu_e = np.linalg.norm(truth["neutrino"], axis=-1)
    truth_nu_p4 = np.concatenate([truth_nu_e[..., None], truth["neutrino"]], axis=-1)
    perfect_parent = np.stack(
        [
            perfect_roots[role] + truth_nu_p4[:, index]
            for index, role in enumerate(PARENT_ROLES)
        ],
        axis=1,
    )
    neutrino_points, neutrino_summary = _neutrino_products(
        output, standard, truth, assignment_correct
    )
    for mode in STANDARD:
        standard[mode]["nu_joint_medoid"] = neutrino_points["joint_medoid"][mode]
    parent_points, parent_summary = _parent_products(
        output, parent, truth, perfect_parent, assignment_correct
    )
    for mode in PARENT:
        parent[mode]["parent_joint_medoid"] = parent_points["joint_medoid"][mode]
    met_closure = _met_closure_products(output, standard, truth)
    physicality = _posterior_physicality(output, standard, parent)
    cross_head = _cross_head_products(output, standard, parent, truth, candidate)
    oracle = _oracle_products(output, standard, truth, candidate)
    paired = _paired_table(output, {**standard, **parent}, truth, assignment_correct)
    reweighting = _population_reweighting(
        output, standard, parent, truth, assignment_correct
    )
    _write_event_metrics(
        output,
        ids,
        truth,
        standard,
        parent,
        assignment_correct,
        candidate,
    )
    (output / "posterior_convergence/convergence.json").write_text(
        json.dumps(
            {
                "nested_prefixes": [1, 4, 8, 16, 32, 64],
                "neutrino": neutrino_summary,
                "parent": parent_summary["posterior"],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    (output / "classical/status.md").write_text(
        (output / "classical_baseline_status.md").read_text()
    )
    (output / "event_metrics/README.md").write_text(
        "EVENT_ID-keyed truth, support, point-estimator, residual, assignment, and constituent-oracle arrays are in `../event_metrics.h5`.\n"
    )
    (output / "ordering/top_antitop_ordering.json").write_text(
        (output / "parents/top_antitop_ordering.json").read_text()
    )
    report = {
        "status": "passed",
        "campaign": manifest["campaign"],
        "training_source_sha": manifest["source_commit"],
        "evaluation_recovery_sha": manifest.get("evaluation_recovery", {}).get("source_sha"),
        "posterior_extension_manifest": (
            str(posterior_manifest_path.resolve())
            if posterior_manifest_path is not None
            else None
        ),
        "evaluation_source": str(Path(__file__).resolve()),
        "validation_events": len(ids),
        "sealed_test_used": False,
        "neutrino_role_order": list(NU_ROLES),
        "parent_role_order": list(PARENT_ROLES),
        "references": {
            "parton_truth": "authoritative after-FSR top/antitop sidecar",
            "perfect_reco": "truth-assigned measured visible constituents + truth neutrinos",
            "model_reco": "truth-free production prediction",
        },
        "assignment": assignment_summary,
        "neutrino_posterior": neutrino_summary,
        "parent_and_ttbar": parent_summary,
        "posterior_physicality": physicality,
        "met_closure": met_closure,
        "cross_head": cross_head,
        "oracle_decomposition": oracle,
        "paired_comparisons": paired,
        "population_reweighting": reweighting,
        "production_validation_records": {
            mode: {
                "validation_path": str(
                    Path(manifest["modes"][mode]["validation_output"])
                    / "validation.json"
                ),
                "status": validation_records[mode]["status"],
            }
            for mode in manifest["modes"]
        },
        "runtime": {
            mode: {
                "posterior_samples": parent[mode]["posterior_samples"],
                "conditioning": parent[mode]["conditioning"],
                "solver": parent[mode]["solver"],
                "steps": parent[mode]["steps"],
                "nfe": parent[mode]["nfe"],
            }
            for mode in PARENT
        },
    }
    (output / "evaluation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    rows = [
        "# Real-neutrino and parent-CFM validation",
        "",
        f"**Status:** {report['status']}",
        f"**Training source:** `{manifest['source_commit']}`",
        f"**Validation events:** {len(ids):,}",
        "**Sealed test used:** false",
        "",
        "## Model names",
        "",
    ]
    rows.extend(f"- {LABELS[mode]}" for mode in manifest["modes"])
    rows.extend(
        [
            "",
            "## Main comparisons",
            "",
            "| Question | Comparison | Validation evidence |",
            "|---|---|---|",
        ]
    )
    rows.extend(
        [
            "| Do topology queries help? | No-query → topology-query | Assignment efficiency vs nJets; role-separated neutrino residuals and posterior calibration |",
            "| Does multitask learning help parent inference? | Direct parent flow → joint multitask | Top/antitop kinematics; mttbar and ttbar; threshold response |",
            "| Does predicted assignment context help? | Joint multitask → assignment-aware | Paired parent and system-level validation |",
            "| Does predicted neutrino context help beyond assignment? | Assignment-aware → assignment + neutrino-aware | Paired parent and system-level validation |",
            "",
            "## Headline assignment efficiencies",
            "",
            "| Model | Supported events | Complete-event efficiency |",
            "|---|---:|---:|",
        ]
    )
    for mode in STANDARD:
        item = assignment_summary["complete_event"][mode]
        rows.append(
            f"| {LABELS[mode]} | {item['supported_events']:,} | {item['efficiency']:.4f} |"
        )
    rows.extend(
        [
            "",
            "## Threshold mttbar",
            "",
            "| Model | 345–400 GeV events | Bias [GeV] | Resolution [GeV] | RMSE [GeV] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for mode in PARENT:
        item = parent_summary["threshold"][mode]["threshold_345_400"]
        rows.append(
            f"| {LABELS[mode]} | {item['events']:,} | {item['bias_gev'] if item['bias_gev'] is not None else float('nan'):.3f} | {item['resolution_gev'] if item['resolution_gev'] is not None else float('nan'):.3f} | {item['rmse_gev'] if item['rmse_gev'] is not None else float('nan'):.3f} |"
        )
    rows.extend(
        [
            "",
            "## Physics and posterior products",
            "",
            "The evaluation tree contains assignment, neutrino, parent, ttbar, threshold, spin/angular, posterior, oracle-decomposition, paired-bootstrap and machine-readable table products. Top and antitop are always kept in charge-labelled order; ttbar is derived as their four-vector sum.",
            "The posterior predictor preserves whole-event sample association for the joint medoid. Observable quantiles are calculated after evaluating each posterior sample, so q50(mttbar) is not confused with mttbar of component-wise median vectors.",
            (output / "spin/README.md").read_text().strip(),
            "",
            "## Output folders",
            "",
        ]
    )
    rows.extend(
        f"- `{folder}/`"
        for folder in (
            "assignment",
            "neutrinos",
            "parents",
            "ttbar",
            "threshold",
            "spin",
            "posterior",
            "oracle_decomposition",
            "robustness",
            "tables",
        )
    )
    rows.extend([""])
    (output / "REPORT.md").write_text("\n".join(rows))
    if posterior_manifest is not None:
        decision = _write_final_reports(
            output,
            report,
            assignment_summary,
            neutrino_summary,
            parent_summary,
            cross_head,
            oracle,
            paired,
            reweighting,
            physicality,
        )
        report["decision_freeze"] = decision
        (output / "evaluation.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--posterior-manifest", type=Path)
    args = parser.parse_args()
    result = run(
        args.manifest.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.posterior_manifest,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "events": result["validation_events"],
                "training_source_sha": result["training_source_sha"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
