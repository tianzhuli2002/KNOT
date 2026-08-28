#!/usr/bin/env python3
"""Plot compact cross-model summaries from run_study.py summary.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGSIZE = (8.0, 5.2)


def _label(frame: pd.DataFrame) -> pd.Series:
    return frame["variant_label"].astype(str) + ": " + frame["prediction_label"].astype(str)


def _aggregate(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    selected = frame.loc[frame[metric].notna()].copy()
    if selected.empty:
        return selected
    selected["model_label"] = _label(selected)
    return (
        selected.groupby(
            ["topology", "variant", "prediction", "model_label"], as_index=False
        )[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def _finish(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


def _save(fig, base: Path, formats: tuple[str, ...]) -> list[Path]:
    fig.tight_layout()
    written = []
    for extension in formats:
        path = base.with_suffix(f".{extension}")
        fig.savefig(path, bbox_inches="tight", dpi=220 if extension == "png" else None)
        written.append(path)
    plt.close(fig)
    return written


def plot_summary(
    summary: Path,
    output: Path,
    *,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[Path]:
    frame = pd.read_csv(summary)
    output.mkdir(parents=True, exist_ok=True)
    written = []

    common = _aggregate(frame, "common_non_null_assignment_accuracy")
    for topology, local in common.groupby("topology"):
        local = local.sort_values("mean")
        fig, ax = plt.subplots(figsize=FIGSIZE)
        y = np.arange(len(local))
        errors = local["std"].fillna(0.0).to_numpy()
        ax.errorbar(
            local["mean"],
            y,
            xerr=errors,
            marker="o",
            linestyle="none",
            capsize=3,
        )
        ax.set_yticks(y, local["model_label"])
        ax.set_xlim(0.0, 1.02)
        ax.set_xlabel("Assignment accuracy on common truth-non-NULL subset")
        ax.set_title(str(topology))
        _finish(ax)
        written += _save(fig, output / f"{topology}_common_assignment", formats)

    energy = frame.loc[
        frame["energy_score"].notna()
        & frame["sample_equivalent_nfe_per_event"].notna()
    ].copy()
    for topology, local in energy.groupby("topology"):
        local["model_label"] = _label(local)
        fig, ax = plt.subplots(figsize=FIGSIZE)
        for label, values in local.groupby("model_label"):
            ax.scatter(
                values["sample_equivalent_nfe_per_event"],
                values["energy_score"],
                label=label,
            )
        ax.set_xlabel("Sample-equivalent NFE / event")
        ax.set_ylabel("Joint invisible energy score")
        ax.set_title(str(topology))
        ax.legend(frameon=False, fontsize=8)
        _finish(ax)
        written += _save(fig, output / f"{topology}_energy_vs_nfe", formats)

    compatibility = frame.loc[frame["compatibility_combined_top1"].notna()].copy()
    for topology, local in compatibility.groupby("topology"):
        local["model_label"] = _label(local)
        local = local.sort_values("model_label")
        x = np.arange(len(local))
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(
            x,
            local["compatibility_assignment_only_top1"],
            marker="o",
            label="assignment only",
        )
        ax.plot(
            x,
            local["compatibility_combined_top1"],
            marker="o",
            label="compatibility reranked",
        )
        ax.set_xticks(x, local["model_label"], rotation=35, ha="right")
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Top-1 complete-assignment efficiency")
        ax.set_title(str(topology))
        ax.legend(frameon=False)
        _finish(ax)
        written += _save(fig, output / f"{topology}_compatibility_reranking", formats)

    if not written:
        raise RuntimeError(f"No plottable study summary metrics found in {summary}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf")
    )
    args = parser.parse_args()
    written = plot_summary(
        args.summary.expanduser().resolve(),
        args.output.expanduser().resolve(),
        formats=tuple(args.formats),
    )
    print(f"Wrote {len(written)} study summary figures to {args.output}")


if __name__ == "__main__":
    main()
