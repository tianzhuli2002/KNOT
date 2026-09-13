#!/usr/bin/env python3
"""Measure paired assignment transitions after posterior-informed correction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.prediction_schema import validate_prediction


def _read(path: Path) -> dict[str, np.ndarray]:
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        event_id = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        correct = np.asarray(source["TRUTH/topk_assignment"][:, 0], dtype=bool)
        nodes = source["VISIBLE/nodes"]
        names = sorted(nodes.keys())
        entropy = np.stack([nodes[name]["entropy"][:] for name in names], axis=1).mean(axis=1)
        margin = np.stack([nodes[name]["margin"][:] for name in names], axis=1).mean(axis=1)
        result = {
            "event_id": event_id,
            "correct": correct,
            "entropy": np.asarray(entropy, dtype=np.float64),
            "margin": np.asarray(margin, dtype=np.float64),
            "njet": np.asarray(source["OBSERVED/njet"][:], dtype=np.int64),
        }
        if "INVISIBLE/samples" in source:
            samples = np.asarray(source["INVISIBLE/samples"][:, 0], dtype=np.float64)
            truth = np.asarray(source["TRUTH/NEUTRINO"][:], dtype=np.float64)
            posterior_mean = samples.mean(axis=1)
            result["posterior_width"] = np.asarray(
                source["INVISIBLE/std"][:, 0], dtype=np.float64
            ).mean(axis=(1, 2))
            result["neutrino_error"] = np.sqrt(
                ((posterior_mean - truth) ** 2).mean(axis=(1, 2))
            )
            if "OBSERVED/MET" in source:
                observed_met = np.asarray(source["OBSERVED/MET"][:, 1:3], dtype=np.float64)
                result["met_residual"] = np.linalg.norm(
                    posterior_mean[:, :, :2].sum(axis=1) - observed_met, axis=1
                )
        return result


def _quantile_groups(values: np.ndarray, prefix: str) -> list[tuple[str, np.ndarray]]:
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    return [
        (
            f"{prefix}_q{index + 1}[{edges[index]:.6g},{edges[index + 1]:.6g}]",
            (values >= edges[index])
            & (values <= edges[index + 1] if index == 3 else values < edges[index + 1]),
        )
        for index in range(4)
    ]


def _paired(before: np.ndarray, after: np.ndarray) -> dict:
    return {
        "events": int(len(before)),
        "wrong_to_correct": int((~before & after).sum()),
        "correct_to_wrong": int((before & ~after).sum()),
        "stable_correct": int((before & after).sum()),
        "stable_wrong": int((~before & ~after).sum()),
        "wrong_to_correct_fraction": float((~before & after).mean()),
        "correct_to_wrong_fraction": float((before & ~after).mean()),
        "accuracy_before": float(before.mean()),
        "accuracy_after": float(after.mean()),
        "paired_delta": float(after.astype(np.int8).mean() - before.astype(np.int8).mean()),
    }


def analyse(base_path: Path, feedback_path: Path) -> dict:
    base = _read(base_path)
    feedback = _read(feedback_path)
    if not np.array_equal(base["event_id"], feedback["event_id"]):
        raise ValueError("Feedback transition predictions must share ordered EVENT_IDs")
    before = base["correct"]
    after = feedback["correct"]
    transition = {
        "wrong_to_correct": ~before & after,
        "correct_to_wrong": before & ~after,
        "stable_correct": before & after,
        "stable_wrong": ~before & ~after,
    }
    stratification = {}
    for name, values in (
        ("posterior_width", feedback.get("posterior_width")),
        ("neutrino_error", feedback.get("neutrino_error")),
        ("met_residual", feedback.get("met_residual")),
        ("assignment_entropy", base["entropy"]),
        ("assignment_top1_top2_margin", base["margin"]),
    ):
        if values is None:
            continue
        stratification[name] = {}
        for label, mask in _quantile_groups(np.asarray(values, dtype=np.float64), name):
            stratification[name][label] = {
                "events": int(mask.sum()),
                **{
                    transition_name: int((transition_mask & mask).sum())
                    for transition_name, transition_mask in transition.items()
                },
                "transition_gain": int(
                    (transition["wrong_to_correct"] & mask).sum()
                    - (transition["correct_to_wrong"] & mask).sum()
                ),
            }
    stratification["njet"] = {
        str(int(value)): {
            "events": int((base["njet"] == value).sum()),
            **{
                transition_name: int((transition_mask & (base["njet"] == value)).sum())
                for transition_name, transition_mask in transition.items()
            },
        }
        for value in np.unique(base["njet"])
    }
    return {
        "schema": "study04_feedback_transition_diagnostics_v1",
        "selection_split": "validation",
        "scientific_winner_selected": False,
        "base_prediction": str(base_path.resolve()),
        "feedback_prediction": str(feedback_path.resolve()),
        "events": int(len(before)),
        "transitions": _paired(before, after),
        "stratified_transitions": stratification,
        "interpretation": (
            "Posterior feedback is judged from paired EVENT_ID transitions. The four "
            "transition classes are retained explicitly; no recurrence winner is selected."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--feedback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.base.expanduser().resolve(), args.feedback.expanduser().resolve())
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "feedback_transitions.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")
    transitions = result["transitions"]
    (output / "feedback_transitions.md").write_text(
        "\n".join(
            [
                "# Study 04 posterior-feedback transitions",
                "",
                "Validation-only paired EVENT_ID diagnostics; no scientific winner is selected.",
                "",
                f"- wrong → correct: {transitions['wrong_to_correct']} ({transitions['wrong_to_correct_fraction']:.5f})",
                f"- correct → wrong: {transitions['correct_to_wrong']} ({transitions['correct_to_wrong_fraction']:.5f})",
                f"- stable correct: {transitions['stable_correct']}",
                f"- stable wrong: {transitions['stable_wrong']}",
                f"- paired accuracy delta: {transitions['paired_delta']:+.5f}",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
