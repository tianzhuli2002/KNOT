"""Shared plotting style for transformer reconstruction and representation studies."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, PercentFormatter

RANDOM_COLOUR = "#B8B8B8"
TRUTH_COLOUR = "#202020"
CORRECT_COLOUR = "#2A9D8F"
INCORRECT_COLOUR = "#C44E52"
BACKGROUND_COLOUR = "#7A7A7A"
SIGNAL_COLOUR = "#3B82F6"
FM_COLOUR = "#2A9D8F"
NONFM_COLOUR = "#E07A5F"

SIGNAL_SAMPLE_COLOURS = ("#C44E52", "#D95F02", "#E07A5F", "#D4A017")
BACKGROUND_SAMPLE_COLOURS = ("#3B82F6", "#228B60", "#7C3AED", "#2A9D8F", "#6B7280", "#8B5A2B")

COMPONENT_LABELS = {
    "blep": r"$b_{\mathrm{lep}}$",
    "bhad": r"$b_{\mathrm{had}}$",
    "Whad": r"$W_{\mathrm{had}}$",
    "tlep_b": r"$b_{\mathrm{lep}}+\ell$",
    "tlep": r"$t_{\mathrm{lep}}$",
    "thad": r"$t_{\mathrm{had}}$",
    "H": r"$H\to b\bar{b}$",
    "complete ttbar1L topology": r"Complete $t\bar{t}$",
    "complete ttH topology": r"Complete $t\bar{t}H$",
}
COMPONENT_COLOURS = {
    "blep": "#2563EB",
    "bhad": "#7C3AED",
    "Whad": "#228B60",
    "tlep_b": "#2563EB",
    "tlep": "#2A9D8F",
    "thad": "#D95F02",
    "H": "#D4A017",
    "complete ttbar1L topology": "#202020",
    "complete ttH topology": "#202020",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7.4, 5.5),
            "figure.dpi": 130,
            "savefig.dpi": 250,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 11.5,
            "axes.labelsize": 12.5,
            "axes.titlesize": 13.5,
            "axes.linewidth": 1.05,
            "axes.grid": False,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "xtick.minor.size": 2.8,
            "ytick.minor.size": 2.8,
            "legend.frameon": False,
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def decorate_axis(ax, *, title: str | None = None, minor_ticks: bool = True) -> None:
    ax.tick_params(which="both", direction="in", top=True, right=True)
    if minor_ticks:
        if ax.get_xscale() == "linear":
            ax.xaxis.set_minor_locator(AutoMinorLocator())
        if ax.get_yscale() == "linear":
            ax.yaxis.set_minor_locator(AutoMinorLocator())
    if title:
        ax.set_title(title, pad=10)


def efficiency_axis(ax, *, percent: bool = True) -> None:
    decorate_axis(ax)
    ax.set_ylim(0.0, 1.05)
    if percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)


def save_figure(fig, output_path: str | Path, formats: Iterable[str] = ("pdf", "png")) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    for fmt in formats:
        fmt = str(fmt).lower().lstrip(".")
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = 250
        fig.savefig(stem.with_suffix(f".{fmt}"), **kwargs)
    plt.close(fig)


configure_matplotlib()
