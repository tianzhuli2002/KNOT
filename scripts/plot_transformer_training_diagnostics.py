#!/usr/bin/env python
"""Compact learning curves from a Lightning TensorBoard run directory."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.m0.evaluation.plot_style import decorate_axis


def _series(accumulator, tag):
    return [(event.step, event.value) for event in accumulator.Scalars(tag)] if tag in accumulator.Tags().get("scalars", []) else []


def _plot(series, path, ylabel):
    if not series:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, values in series.items():
        if values:
            x, y = zip(*values); ax.plot(x, y, label=label)
    ax.set_xlabel("epoch / validation step"); ax.set_ylabel(ylabel)
    if len(ax.lines) > 1:
        ax.legend()
    decorate_axis(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Transformer training diagnostics from a TensorBoard run.",
        epilog="Example: python scripts/plot_transformer_training_diagnostics.py --log-dir training_logs/Transformer --output-dir diagnostics",
    )
    parser.add_argument("--log-dir", required=True, help="Lightning/TensorBoard run directory.")
    parser.add_argument("--output-dir", required=True, help="Directory for diagnostic PDF and PNG plots.")
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    accumulator = EventAccumulator(args.log_dir); accumulator.Reload()
    tags = accumulator.Tags().get("scalars", [])
    _plot({tag: _series(accumulator, tag) for tag in tags if tag in {"train_loss", "val_loss", "train_reconstruction_loss", "val_reconstruction_loss", "train_classification_loss", "val_classification_loss"}}, output / "reco_learning_loss.pdf", "loss")
    _plot({tag: _series(accumulator, tag) for tag in tags if tag.startswith("val_reco_") and tag.endswith("_top1_accuracy")}, output / "reco_learning_top1.pdf", "top-1 efficiency")
    _plot({"val_auc": _series(accumulator, "val_auc")}, output / "reco_learning_auc.pdf", "validation AUC")


if __name__ == "__main__":
    main()
