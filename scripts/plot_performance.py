#!/usr/bin/env python3
"""Plot maintained factorized, DFM and joint-flow prediction products."""

from __future__ import annotations

import argparse
from pathlib import Path

from joint_event_reconstruction.plotting import plot_comparison, plot_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        nargs="+",
        required=True,
        help="One or more maintained prediction HDF5 files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--formats", nargs="+", default=("png", "pdf"), choices=("png", "pdf")
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Optional labels for multi-file comparison plots.",
    )
    args = parser.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    formats = tuple(args.formats)
    if args.labels is not None and len(args.labels) != len(args.predictions):
        parser.error("--labels must contain one label per prediction file")

    for index, prediction in enumerate(args.predictions):
        label = (
            args.labels[index]
            if args.labels is not None
            else Path(prediction).stem
        )
        destination = output / label
        written = plot_prediction(prediction, destination, formats=formats)
        print(f"{prediction}: wrote {len(written)} plot/metric products to {destination}")

    if len(args.predictions) > 1:
        comparison = plot_comparison(
            args.predictions,
            output / "comparison",
            labels=args.labels,
            formats=formats,
        )
        print(f"comparison: wrote {len(comparison)} products")


if __name__ == "__main__":
    main()
