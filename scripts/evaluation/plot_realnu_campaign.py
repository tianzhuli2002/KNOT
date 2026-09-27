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


def _bootstrap_delta(
    a: np.ndarray, b: np.ndarray, *, seed: int, draws: int = 500
) -> dict:
    """Paired bootstrap for metric (b - a), resampling shared events."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    difference = b[keep] - a[keep]
    if not len(difference):
        return {"events": 0, "mean_delta": None, "ci68": None, "ci95": None}
    rng = np.random.default_rng(seed)
    sample_size = min(10_000, len(difference))
    estimates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        sample = rng.choice(difference, size=sample_size, replace=True)
        estimates[index] = sample.mean()
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
        globals_all = np.asarray(measured["INPUTS/GLOBAL"][:])
        globals_val = globals_all[split == 1]
    if set(map(int, ids)) != set(map(int, validation_ids)):
        raise ValueError(
            "Prediction EVENT_IDs do not exactly match the frozen validation membership"
        )
    globals_joined = _join(
        ids, validation_ids, globals_val, "measured validation globals"
    )
    names = globals_joined.dtype.names or ()
    fields = {name.lower(): name for name in names}
    for required in ("njet", "nelectron", "nmuon"):
        if required not in fields:
            raise KeyError(
                f"Measured global inputs lack required {required}; fields={names}"
            )
    njet = np.rint(globals_joined[fields["njet"]]).astype(np.int64)
    ne = np.rint(globals_joined[fields["nelectron"]]).astype(np.int64)
    nm = np.rint(globals_joined[fields["nmuon"]]).astype(np.int64)
    channel = np.full(len(ids), "other", dtype=object)
    channel[(ne == 2) & (nm == 0)] = "ee"
    channel[(ne == 1) & (nm == 1)] = "eμ"
    channel[(ne == 0) & (nm == 2)] = "μμ"
    return {
        "neutrino": neutrino,
        "neutrino_support": nu_support,
        "parents": parent,
        "parent_support": parent_support,
        "assignment_support": assignment_support,
        "njet": njet,
        "channel": channel,
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
        njet = np.asarray(source["OBSERVED/njet"][:], dtype=np.int64)
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
        "njet": _join(expected_ids, ids, njet, f"{prediction_path} njet"),
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
    support = truth["assignment_support"].copy()
    for mode in STANDARD:
        roots = predictions[mode]["roots"][:, 0]
        target = predictions["candidate_truth_index"]
        role_correct = {}
        complete = support.copy()
        for index, role in enumerate(PARENT_ROLES):
            role_supported = predictions["candidate_support"][role] & (
                target[role] >= 0
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

    njet = truth["njet"]
    njet_bins = np.unique(njet)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for role_index, role in enumerate(PARENT_ROLES):
        for mode in STANDARD:
            roots = predictions[mode]["roots"][:, 0, role_index]
            role_supported = predictions[mode]["candidate_support"][role] & (
                predictions[mode]["candidate_truth_index"][role] >= 0
            )
            efficiency = [
                (
                    (roots == predictions[mode]["candidate_truth_index"][role])
                    & role_supported
                    & (njet == n)
                ).sum()
                / max(1, (role_supported & (njet == n)).sum())
                for n in njet_bins
            ]
            axes[role_index].plot(
                njet_bins,
                efficiency,
                marker="o",
                markersize=3,
                color=COLORS[mode],
                label=LABELS[mode],
            )
        axes[role_index].set(
            xlabel="Source nJets", ylabel="Per-role assignment efficiency", title=role
        )
        axes[role_index].set_ylim(0, 1.02)
        axes[role_index].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%")
        )
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("Assignment efficiency by role and jet multiplicity")
    _save(fig, assignment_dir / "per_role_efficiency_vs_njet")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    by_njet = {}
    for mode in STANDARD:
        values, denominators = [], []
        for n in njet_bins:
            selected = predictions[mode]["assignment_supported"] & (njet == n)
            denominators.append(int(selected.sum()))
            values.append(
                float(correctness[mode][selected].mean()) if selected.any() else np.nan
            )
        by_njet[mode] = {
            str(int(n)): {"supported_events": denominators[i], "efficiency": values[i]}
            for i, n in enumerate(njet_bins)
        }
        ax.plot(
            njet_bins,
            values,
            marker="o",
            markersize=3,
            color=COLORS[mode],
            label=LABELS[mode],
        )
    ax.set(
        xlabel="Source nJets",
        ylabel="Complete-event assignment efficiency",
        title="Complete assignment efficiency by jet multiplicity",
    )
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    ax.legend(frameon=False, fontsize=7)
    ax.text(
        0.02,
        0.02,
        "Denominator: visible-assignment-supported events in each nJets bin",
        transform=ax.transAxes,
        fontsize=8,
    )
    _save(fig, assignment_dir / "complete_event_efficiency_vs_njet")

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
        (
            np.stack([a[selected], b[selected]], axis=1).sum(axis=0)
            if selected.any()
            else np.zeros(2)
        )
        delta = (b[selected].mean() - a[selected].mean()) if selected.any() else np.nan
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
    fig.tight_layout()
    _save(fig, assignment_dir / "paired_model_transitions")

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
                "by_njet": by_njet[mode],
            }
            for mode in STANDARD
        },
        "role_metrics": rows,
    }
    with (output / "tables/assignment.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "tables/assignment.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
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
    truth["neutrino_support"].all(axis=1)

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
                q16, q50, q84 = np.quantile(posterior, [0.16, 0.5, 0.84], axis=1)
                if name == "phi":
                    median_residual = _wrap(q50 - truth_value)
                    width = _wrap(q84 - q16)
                    _wrap(posterior - truth_value[:, None])
                    coverage68 = np.mean(
                        np.abs(_wrap(truth_value - q50)) <= np.abs(width) / 2
                    )
                else:
                    median_residual = q50 - truth_value
                    width = q84 - q16
                    posterior - truth_value[:, None]
                    coverage68 = np.mean((truth_value >= q16) & (truth_value <= q84))
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
                        "mean_central_68_width": float(np.nanmean(width)),
                        "pit_mean": float(
                            np.mean(np.mean(posterior <= truth_value[:, None], axis=1))
                        ),
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

        # Channel, jet count, and assignment-correct/incorrect splits for pz.
        for split_name, split_values in (
            ("channel", truth["channel"]),
            ("njet", truth["njet"]),
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
                        (str(value), split_values == value) for value in unique
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
    mttbar_edges = np.asarray([345.0, 400.0, 500.0, 700.0, 1000.0, 3000.0])
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
        posterior_mean = sample_mass.mean(axis=1)
        point_error = posterior_mean - truth_mass
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
                "rmse_gev": float(np.sqrt(np.mean(residual**2)))
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fine_edges = np.asarray([345, 355, 365, 375, 385, 400], dtype=float)
    for mode, samples in parent_samples.items():
        mass_samples = _mass(samples[:, :, 0] + samples[:, :, 1])
        mean = mass_samples.mean(axis=1)
        centers, bias, resolution, _counts = _bin_summary(truth_mass, mean, fine_edges)
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
            "The repository has no maintained c_hel or c_han implementation with the requested charge-labelled-lepton convention; those observables are explicitly unavailable and were not rederived.\n"
        )
    except (ImportError, KeyError, RuntimeError) as error:
        (spin_folder / "README.md").write_text(
            f"Maintained angular observable evaluation unavailable: {type(error).__name__}: {error}\n"
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
        "parent_metrics": parent_metrics,
        "ttbar_metrics": pair_metrics,
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


def _paired_table(
    output: Path, predictions: dict, truth: dict, assignment_correct: dict
) -> dict:
    rows = []
    results = {}
    for first, second, label in COMPARISONS:
        item = {}
        if first in assignment_correct and second in assignment_correct:
            common = (
                predictions[first]["assignment_supported"]
                & predictions[second]["assignment_supported"]
            )
            item["complete_assignment_efficiency"] = _bootstrap_delta(
                assignment_correct[first][common].astype(float),
                assignment_correct[second][common].astype(float),
                seed=1102,
            )
        if first in STANDARD and second in STANDARD:
            for role_index, role in enumerate(NU_ROLES):
                target = truth["neutrino"][:, role_index]
                for name, comp in (("pz", 2), ("pT", -1)):

                    def metric(
                        mode,
                        role_index=role_index,
                        comp=comp,
                        target=target,
                    ):
                        sample = predictions[mode]["nu_samples"][
                            :, :, role_index
                        ].astype(np.float64)
                        values = (
                            sample[..., comp]
                            if comp >= 0
                            else np.hypot(sample[..., 0], sample[..., 1])
                        )
                        q50 = np.quantile(values, 0.5, axis=1)
                        truth_value = (
                            target[:, comp]
                            if comp >= 0
                            else np.hypot(target[:, 0], target[:, 1])
                        )
                        return np.abs(q50 - truth_value)

                    item[f"{role}_{name}_mae"] = _bootstrap_delta(
                        metric(first), metric(second), seed=1200 + role_index * 7 + comp
                    )
        if first in PARENT and second in PARENT:
            for name in ("mttbar", "pT(ttbar)", "y(ttbar)"):

                def metric(mode, name=name):
                    samples = predictions[mode]["parent_samples"].astype(np.float64)
                    pair = samples[:, :, 0] + samples[:, :, 1]
                    obs = (
                        _mass(pair)
                        if name == "mttbar"
                        else _p4_observables(pair)["pT" if name == "pT(ttbar)" else "y"]
                    )
                    target = (
                        _mass(truth["parents"][:, 0] + truth["parents"][:, 1])
                        if name == "mttbar"
                        else _p4_observables(
                            truth["parents"][:, 0] + truth["parents"][:, 1]
                        )["pT" if name == "pT(ttbar)" else "y"]
                    )
                    return np.abs(np.quantile(obs, 0.5, axis=1) - target)

                item[f"{name}_mae"] = _bootstrap_delta(
                    metric(first), metric(second), seed=1300 + len(name)
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
    output.mkdir(parents=True, exist_ok=False)
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
    ):
        (output / folder).mkdir()

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
        if not np.array_equal(standard[mode]["njet"], truth["njet"]):
            raise ValueError(
                f"{mode} observed nJets differs from measured validation input"
            )
        standard[mode]["nu_samples"] = standard[mode].pop("samples")
    for mode in PARENT:
        parent[mode]["parent_samples"] = parent[mode].pop("samples")
        parent[mode]["parent_observables"] = parent[mode].pop("observables")
        if mode == "J_AN" and (
            parent[mode]["conditioning"] != "P2"
            or parent[mode]["solver"] != "heun"
            or parent[mode]["steps"] != 8
        ):
            raise ValueError(
                "J_AN production validation did not retain the live P2, 8-step Heun contract"
            )
        if mode == "J_AN" and parent[mode]["nfe"] != 16:
            raise ValueError(
                "J_AN parent validation does not have the expected 16 NFE Heun sampling path"
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
    _neutrino_points, neutrino_summary = _neutrino_products(
        output, standard, truth, assignment_correct
    )
    _parent_points, parent_summary = _parent_products(
        output, parent, truth, perfect_parent, assignment_correct
    )
    oracle = _oracle_products(output, standard, truth, candidate)
    paired = _paired_table(output, {**standard, **parent}, truth, assignment_correct)

    # Store compact machine-readable event-level quantities to support paired rereads.
    paired_path = output / "tables/event_level_metrics.npz"
    event_products = {
        "EVENT_ID": ids,
        "njet": truth["njet"],
        "channel": truth["channel"].astype("U8"),
        "visible_assignment_support": truth["assignment_support"],
    }
    for mode in STANDARD:
        event_products[f"{mode}_assignment_correct"] = assignment_correct[mode]
        event_products[f"{mode}_assignment_supported"] = standard[mode][
            "assignment_supported"
        ]
    for mode in STANDARD:
        for role_index, role in enumerate(NU_ROLES):
            samples = standard[mode]["nu_samples"].astype(np.float64)[:, :, role_index]
            event_products[f"{mode}_{role.replace('.', '_')}_pz_q50"] = np.quantile(
                samples[..., 2], 0.5, axis=1
            )
            event_products[f"{mode}_{role.replace('.', '_')}_pt_q50"] = np.quantile(
                np.hypot(samples[..., 0], samples[..., 1]), 0.5, axis=1
            )
    for mode in PARENT:
        samples = parent[mode]["parent_samples"].astype(np.float64)
        pair = samples[:, :, 0] + samples[:, :, 1]
        event_products[f"{mode}_mttbar_q50"] = np.quantile(_mass(pair), 0.5, axis=1)
    np.savez_compressed(paired_path, **event_products)

    # Runtime provenance, physics conventions, data counts and supported outputs.
    (output / "robustness/README.md").write_text(
        "Nominal validation is a frozen 100342-event sample and no sealed test data were read. "
        "The report retains EVENT_ID-keyed predictions and event-level metrics so a future alternative-generator validation can reuse the trained checkpoints. "
        "A reweighted validation can be added by supplying weights keyed to these EVENT_ID values and recomputing weighted observables; no truth-spectrum correction was applied here.\n"
    )
    report = {
        "status": "passed",
        "campaign": manifest["campaign"],
        "training_source_sha": manifest["source_commit"],
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
        "oracle_decomposition": oracle,
        "paired_comparisons": paired,
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
