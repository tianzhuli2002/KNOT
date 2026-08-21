#!/usr/bin/env python3
"""Plot Transformer classification and reconstruction performance from a prediction manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.m0.evaluation.performance import run_performance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing Transformer prediction manifest using source-event jet multiplicity.",
        epilog="Example: python scripts/plot_performance.py /path/to/prediction_manifest.json",
    )
    parser.add_argument("prediction_manifest", type=Path, help="Prediction manifest produced by the predict operation.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing incomplete or incompatible performance directory.")
    parser.add_argument("--diagnostics", action="store_true", help="Write selected candidate and input-multiplicity diagnostics.")
    parser.add_argument("--no-kinematics", action="store_true", help="Skip configured reconstructed-kinematics products.")
    args = parser.parse_args()
    run_performance(
        None,
        None,
        {
            "performance": {
                "prediction_manifest": str(args.prediction_manifest),
                "output_dir": None,
                "diagnostics": bool(args.diagnostics),
                "kinematics": not args.no_kinematics,
                "overwrite": bool(args.overwrite),
            }
        },
    )


if __name__ == "__main__":
    main()
