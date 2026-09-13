#!/usr/bin/env python3
"""Run validation-only assignment-set diagnostics on Study-12 predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.calibration.assignment_sets import analyze_aps, calibrate_aps
from joint_event_reconstruction.prediction_schema import validate_prediction


def _read(path: Path) -> tuple[dict, dict]:
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        if str(source.attrs.get("split")) not in {"validation", "val"}:
            raise ValueError(f"Analysis 14 requires validation predictions: {path}")
        probabilities = np.asarray(source["VISIBLE/constrained_probabilities"][:])
        truth = np.asarray(source["TRUTH/topk_assignment"][:])
        event_id = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        nodes = source["VISIBLE/nodes"]
        entropy = np.stack([nodes[name]["entropy"][:] for name in nodes], axis=1).mean(axis=1)
        calibrator = calibrate_aps(probabilities, truth)
        analysis = analyze_aps(
            probabilities,
            truth,
            calibrator,
            njet=np.asarray(source["OBSERVED/njet"][:]),
            entropy=entropy,
        )
        analysis.update({
            "source_prediction": str(path.resolve()),
            "event_id_fingerprint": hashlib.sha256(event_id.tobytes()).hexdigest(),
            "calibrator": calibrator,
            "selection_biased_diagnostic": True,
        })
        return calibrator, analysis


def _parse(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        label, path_text = value.split("=", 1)
        if label in result:
            raise ValueError(f"Duplicate Analysis-14 prediction label: {label}")
        result[label] = Path(path_text).expanduser().resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predictions = _parse(args.prediction)
    if not predictions:
        raise ValueError("Analysis 14 requires at least one prediction")
    reports = {}
    for label, path in predictions.items():
        _, reports[label] = _read(path)
    result = {
        "schema": "study14_assignment_set_uncertainty_v1",
        "selection_split": "validation",
        "scientific_winner_selected": False,
        "selection_use": "human review only; this diagnostic never selects a visible encoder",
        "calibration_note": (
            "APS calibration and reporting use the maintained validation-only machinery. "
            "The final reconstruction system must reserve the sealed test split."
        ),
        "predictions": reports,
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis14_assignment_sets.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    lines = [
        "# Analysis 14 — assignment-set uncertainty",
        "",
        "Validation-only APS diagnostic for human review. It never selects a visible encoder.",
        "",
        "| prediction | events | outside top-K | 68% mean set size | 90% mean set size | 95% mean set size |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, report in reports.items():
        targets = report["targets"]
        lines.append(
            f"| {label} | {report['events']} | {report['truth_outside_candidate_support_fraction']:.5f} | "
            f"{targets['0.68']['mean_set_size']:.3f} | {targets['0.90']['mean_set_size']:.3f} | "
            f"{targets['0.95']['mean_set_size']:.3f} |"
        )
    (output / "analysis14_assignment_sets.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
