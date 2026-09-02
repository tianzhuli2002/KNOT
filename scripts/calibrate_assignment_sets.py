#!/usr/bin/env python3
"""Fit a validation-only finite-sample APS calibrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py

from joint_event_reconstruction.calibration.assignment_sets import calibrate_aps
from joint_event_reconstruction.prediction_schema import validate_prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    validate_prediction(args.prediction)
    with h5py.File(args.prediction, "r") as source:
        if str(source.attrs.get("split")) not in {"validation", "val"}:
            raise ValueError("APS calibration requires a validation prediction")
        probabilities = source["VISIBLE/constrained_probabilities"][:]
        truth = source["TRUTH/topk_assignment"][:]
        event_ids = source["EVENT_ID"][:]
        result = calibrate_aps(probabilities, truth)
        result.update(
            {
                "source_prediction": str(args.prediction.resolve()),
                "source_prediction_identity": str(source.attrs.get("prediction_identity", "unknown")),
                "checkpoint": str(source.attrs["checkpoint"]),
                "split": str(source.attrs["split"]),
                "event_id_fingerprint": hashlib.sha256(event_ids.tobytes()).hexdigest(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
