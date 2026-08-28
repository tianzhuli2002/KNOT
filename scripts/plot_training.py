#!/usr/bin/env python3
"""Plot maintained Lightning loss curves from a TensorBoard run directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


MAINTAINED_LOSSES = (
    "loss",
    "reconstruction_loss",
    "invisible_loss",
    "compatibility_loss",
)


def _series(accumulator: EventAccumulator, tag: str):
    if tag not in accumulator.Tags().get("scalars", []):
        return []
    return [(event.step, event.value) for event in accumulator.Scalars(tag)]


def _plot_metric(accumulator: EventAccumulator, metric: str, output: Path) -> Path | None:
    available = {
        split: _series(accumulator, f"{split}_{metric}")
        for split in ("train", "val", "test")
    }
    available = {split: values for split, values in available.items() if values}
    if not available:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    for split, values in available.items():
        x, y = zip(*values)
        ax.plot(x, y, label=split)
    ax.set_xlabel("logged step")
    ax.set_ylabel(metric.replace("_", " "))
    if len(available) > 1:
        ax.legend()
    fig.tight_layout()
    target = output / f"{metric}.pdf"
    fig.savefig(target, bbox_inches="tight")
    fig.savefig(target.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    accumulator = EventAccumulator(str(args.log_dir.expanduser().resolve()))
    accumulator.Reload()
    written = [
        path
        for metric in MAINTAINED_LOSSES
        if (path := _plot_metric(accumulator, metric, output)) is not None
    ]
    if not written:
        tags = accumulator.Tags().get("scalars", [])
        raise RuntimeError(
            "No maintained JER loss tags were found in the TensorBoard run; "
            f"available scalar tags={tags}"
        )
    print(f"Wrote {len(written)} training diagnostic figure sets to {output}")


if __name__ == "__main__":
    main()
