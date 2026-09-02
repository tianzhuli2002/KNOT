#!/usr/bin/env python3
"""Apply a fixed APS calibrator to a ranked assignment prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py

from joint_event_reconstruction.calibration.assignment_sets import analyze_aps
from joint_event_reconstruction.prediction_schema import validate_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("calibrator", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    validate_prediction(args.prediction)
    calibrator = json.loads(args.calibrator.read_text())
    with h5py.File(args.prediction, "r") as source:
        entropy = None
        nodes = source["VISIBLE/nodes"]
        if len(nodes):
            import numpy as np

            entropy = np.stack([nodes[name]["entropy"][:] for name in nodes], axis=1).mean(axis=1)
        result = analyze_aps(
            source["VISIBLE/constrained_probabilities"][:],
            source["TRUTH/topk_assignment"][:],
            calibrator,
            njet=source["OBSERVED/njet"][:],
            entropy=entropy,
        )
        result["prediction_file"] = str(args.prediction.resolve())
        result["calibrator_file"] = str(args.calibrator.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
