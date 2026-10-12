"""Small PCFM-derived visual style layer for KNOT evaluation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "truth": "#C9CFD6",
    "draw": "#D99645",
    "mean": "#287C75",
    "posterior": "#9A78B8",
    "M": "#D96C75",
    "C": "#55A69B",
    "J": "#246B73",
}

LABELS = {
    "truth": "Parton truth",
    "draw": "J one draw",
    "mean": "J posterior mean",
    "posterior": "J full posterior",
    "M": "KNOT M posterior",
    "C": "KNOT C posterior",
    "J": "KNOT J posterior",
}

RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "mathtext.fontset": "dejavusans",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "axes.labelcolor": "#30343B",
    "text.color": "#30343B",
    "xtick.color": "#30343B",
    "ytick.color": "#30343B",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#555B63",
    "axes.grid": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
    "legend.fontsize": 11,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 250,
}


def apply_style() -> None:
    plt.rcParams.update(RC_PARAMS)


def decorate_axis(ax, *, grid: str | None = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.8)
    ax.set_axisbelow(True)
    if grid in {"x", "y", "both"}:
        ax.grid(axis=grid, color="#D8DCE1", linewidth=0.8, alpha=0.5)


def ratio_axes(
    *, figsize: tuple[float, float] = (9.4, 7.2), ylabel: str = "Prediction / truth"
):
    fig, (main, lower) = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": (7, 3)},
    )
    fig.subplots_adjust(hspace=0.04)
    decorate_axis(main)
    decorate_axis(lower)
    lower.axhline(1.0, color="#30343B", linewidth=1.1)
    lower.set_ylabel(ylabel)
    return fig, main, lower


def plot_truth_density(ax, values: np.ndarray, bins: np.ndarray, *, weights=None, label="Parton truth"):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return ax.hist(
        values,
        bins=bins,
        weights=None if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1),
        density=True,
        histtype="stepfilled",
        color=COLORS["truth"],
        edgecolor="none",
        linewidth=0,
        alpha=0.42,
        label=label,
        zorder=1,
    )


def plot_histogram_ratio(
    ax,
    numerator_counts: np.ndarray,
    reference_counts: np.ndarray,
    bins: np.ndarray,
    *,
    method: str,
    min_reference_events: float = 25.0,
    label: str | None = None,
):
    numerator = np.asarray(numerator_counts, dtype=np.float64)
    reference = np.asarray(reference_counts, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    if numerator.shape != reference.shape or numerator.shape != (len(bins) - 1,):
        raise ValueError("Ratio histogram counts must match the supplied edges")
    if np.any(numerator < 0.0) or np.any(reference < 0.0):
        raise ValueError("Ratio histogram counts must be non-negative")
    n_total, r_total = float(numerator.sum()), float(reference.sum())
    if n_total <= 0.0 or r_total <= 0.0:
        raise ValueError("Cannot form a ratio from an empty distribution")
    widths = np.diff(bins)
    n_density = numerator / n_total / widths
    r_density = reference / r_total / widths
    supported = (reference >= min_reference_events) & (r_density > 0.0)
    ratio = np.divide(n_density, r_density, out=np.full_like(n_density, np.nan), where=supported)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(
        centers,
        ratio,
        color=COLORS[method],
        linewidth=1.9,
        label=label or LABELS[method],
    )
    return ratio, supported


def controlled_ratio_ylim(
    ax,
    ratios: list[np.ndarray],
    x_values: np.ndarray,
    *,
    colors: list[str] | None = None,
    center: float = 1.0,
    robust_quantile: float = 0.95,
) -> tuple[float, int]:
    finite = [np.asarray(r, dtype=np.float64)[np.isfinite(r)] for r in ratios]
    finite = [r for r in finite if r.size]
    if not finite:
        raise ValueError("Cannot set ratio limits without supported finite bins")
    if not 0.0 < robust_quantile < 1.0:
        raise ValueError("Ratio limit quantile must be between zero and one")
    limit = max(0.05, float(np.quantile(np.concatenate([np.abs(r - center) for r in finite]), robust_quantile)) * 1.15)
    limit = float(np.ceil(limit * 20.0) / 20.0)
    x_values = np.asarray(x_values, dtype=np.float64)
    if colors is not None and len(colors) != len(ratios):
        raise ValueError("Ratio colors must match the number of curves")
    lower_limit = max(0.0, center - limit) if center > 0.0 else center - limit
    clipped = 0
    for index, values in enumerate(ratios):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != x_values.shape:
            raise ValueError("Ratio points and bin centers do not align")
        high = np.isfinite(values) & (values > center + limit)
        low = np.isfinite(values) & (values < lower_limit)
        color = colors[index] if colors is not None else "#747A82"
        clipped += int(high.sum() + low.sum())
        if high.any():
            ax.scatter(x_values[high], np.full(high.sum(), center + limit), marker="^", s=24, color=color, clip_on=False, zorder=6)
        if low.any():
            ax.scatter(x_values[low], np.full(low.sum(), center - limit), marker="v", s=24, color=color, clip_on=False, zorder=6)
    ax.set_ylim(lower_limit, center + limit)
    return limit, clipped


def uncertainty_band(ax, edges: np.ndarray, low: np.ndarray, high: np.ndarray, *, color: str, alpha: float = 0.2):
    edges = np.asarray(edges, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    if low.shape != (len(edges) - 1,) or high.shape != low.shape:
        raise ValueError("Uncertainty band values must match histogram bins")
    return ax.fill_between(
        edges,
        np.r_[low, low[-1]],
        np.r_[high, high[-1]],
        step="post",
        color=color,
        alpha=alpha,
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )


def format_legend(ax, *, ncol: int = 2, loc: str = "best", fontsize: float = 10.5):
    return ax.legend(
        frameon=False,
        ncol=ncol,
        loc=loc,
        fontsize=fontsize,
        handlelength=2.2,
        columnspacing=0.9,
        labelspacing=0.35,
    )


def save_figure(
    fig,
    path: str | Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
    tight_layout: bool = True,
) -> list[Path]:
    base = Path(path)
    base.parent.mkdir(parents=True, exist_ok=True)
    if tight_layout:
        fig.tight_layout()
    outputs = []
    for extension in formats:
        extension = extension.lower().lstrip(".")
        if extension not in {"png", "pdf"}:
            raise ValueError(f"Unsupported figure format {extension!r}")
        target = base.with_suffix(f".{extension}")
        fig.savefig(target, bbox_inches="tight", dpi=250 if extension == "png" else None)
        outputs.append(target)
    plt.close(fig)
    return outputs


apply_style()
