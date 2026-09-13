#!/usr/bin/env python3
"""Stratify invisible-posterior performance by assignment quality."""

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


def _groups(values: np.ndarray, prefix: str) -> list[tuple[str, np.ndarray]]:
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    result = []
    for index in range(4):
        low, high = edges[index], edges[index + 1]
        mask = (values >= low) & (values <= high if index == 3 else values < high)
        result.append((f"{prefix}_q{index + 1}[{low:.6g},{high:.6g}]", mask))
    return result


def _summary(mask: np.ndarray, correct: np.ndarray, entropy: np.ndarray, margin: np.ndarray,
             njet: np.ndarray, truth_rank: np.ndarray, invisible: dict | None) -> dict:
    count = int(mask.sum())
    result = {
        "events": count,
        "assignment_top1_accuracy": float(correct[mask].mean()) if count else None,
        "assignment_entropy_mean": float(entropy[mask].mean()) if count else None,
        "assignment_top1_top2_margin_mean": float(margin[mask].mean()) if count else None,
        "njet_mean": float(njet[mask].mean()) if count else None,
        "truth_rank_mean": (
            float(truth_rank[mask].mean()) if count and np.any(truth_rank[mask] > 0) else None
        ),
    }
    if invisible is not None:
        for name, values in invisible.items():
            selected = values[mask]
            result[f"{name}_mean"] = float(selected.mean()) if len(selected) else None
    return result


def _read(path: Path) -> dict:
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        if str(source.attrs.get("split")) not in {"validation", "val"}:
            raise ValueError(f"Assignment-conditioning diagnostics require validation: {path}")
        event_id = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        topk = np.asarray(source["TRUTH/topk_assignment"][:], dtype=bool)
        nodes = source["VISIBLE/nodes"]
        names = sorted(nodes.keys())
        entropy = np.stack([nodes[name]["entropy"][:] for name in names], axis=1).mean(axis=1)
        margin = np.stack([nodes[name]["margin"][:] for name in names], axis=1).mean(axis=1)
        njet = np.asarray(source["OBSERVED/njet"][:], dtype=np.int64)
        correct = topk[:, 0]
        truth_rank = np.where(topk.any(axis=1), np.argmax(topk, axis=1) + 1, -1)
        invisible = None
        if "INVISIBLE/samples" in source:
            all_samples = np.asarray(source["INVISIBLE/samples"][:], dtype=np.float64)
            samples = all_samples[:, 0]
            truth = np.asarray(source["TRUTH/NEUTRINO"][:], dtype=np.float64)
            mean = samples.mean(axis=1)
            invisible = {
                "posterior_width": np.asarray(source["INVISIBLE/std"][:, 0], dtype=np.float64).mean(axis=(1, 2)),
                "neutrino_rmse": np.sqrt(((mean - truth) ** 2).mean(axis=(1, 2))),
            }
            if topk.shape[1] > 1 and samples.shape[1] > 1:
                present = topk.any(axis=1)
                branch = np.argmax(topk, axis=1)
                rows = np.flatnonzero(present)
                branch_mean = np.full_like(truth, np.nan)
                branch_mean[rows] = all_samples[rows, branch[rows]].mean(axis=1)
                branch_error = np.sqrt(((branch_mean - truth) ** 2).mean(axis=(1, 2)))
                invisible["truth_branch_rmse_when_in_topk"] = branch_error
        return {
            "event_id": event_id,
            "correct": correct,
            "topk": topk,
            "truth_rank": truth_rank,
            "entropy": np.asarray(entropy, dtype=np.float64),
            "margin": np.asarray(margin, dtype=np.float64),
            "njet": njet,
            "invisible": invisible,
            "path": str(path.resolve()),
        }


def analyse(values: list[str]) -> dict:
    parsed = {}
    for item in values:
        label, path = item.split("=", 1)
        if label in parsed:
            raise ValueError(f"Duplicate conditioning prediction label: {label}")
        parsed[label] = _read(Path(path).expanduser().resolve())
    if not parsed:
        raise ValueError("At least one assignment-conditioning prediction is required")
    reference = next(iter(parsed.values()))
    reports = {}
    for label, value in parsed.items():
        if not np.array_equal(reference["event_id"], value["event_id"]):
            raise ValueError(f"Assignment-conditioning predictions do not share ordered EVENT_IDs: {label}")
        groups = [
            ("all", np.ones(len(value["correct"]), dtype=bool)),
            ("assignment_top1_correct", value["correct"]),
            ("assignment_top1_wrong", ~value["correct"]),
            ("truth_in_topk", value["topk"].any(axis=1)),
            ("truth_not_in_topk", ~value["topk"].any(axis=1)),
            ("truth_rank_1", value["truth_rank"] == 1),
            ("truth_rank_2", value["truth_rank"] == 2),
            ("truth_rank_3_plus", value["truth_rank"] >= 3),
        ]
        groups.extend(_groups(value["entropy"], "assignment_entropy"))
        groups.extend(_groups(value["margin"], "assignment_margin"))
        groups.extend((f"njet_{int(n)}", value["njet"] == n) for n in np.unique(value["njet"]))
        reports[label] = {
            "prediction": value["path"],
            "events": int(len(value["event_id"])),
            "top1_accuracy": float(value["correct"].mean()),
            "topk_truth_coverage": value["topk"].mean(axis=0).tolist(),
            "truth_in_topk_fraction": float(value["topk"].any(axis=1).mean()),
            "truth_rank_counts": {
                str(rank): int((value["truth_rank"] == rank).sum())
                for rank in range(1, value["topk"].shape[1] + 1)
            },
            "strata": {
                name: _summary(mask, value["correct"], value["entropy"], value["margin"], value["njet"], value["truth_rank"], value["invisible"])
                for name, mask in groups
            },
        }
    return {
        "schema": "study03_assignment_conditioning_diagnostics_v1",
        "selection_split": "validation",
        "scientific_winner_selected": False,
        "interpretation": (
            "Compare conditioning when the selected assignment is correct, and quantify "
            "posterior damage after a wrong committed assignment. Hypothesis variants "
            "separate truth-in-top-K from truth-outside-top-K events."
        ),
        "predictions": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.prediction)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "assignment_conditioning.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n")
    lines = [
        "# Study 03 assignment → neutrino diagnostics",
        "",
        "Validation-only, event-paired diagnostics. No scientific winner is selected.",
        "",
        "| prediction | events | assignment top-1 | truth in top-K | correct-assignment posterior RMSE | wrong-assignment posterior RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, report in result["predictions"].items():
        strata = report["strata"]
        correct = strata["assignment_top1_correct"].get("neutrino_rmse_mean")
        wrong = strata["assignment_top1_wrong"].get("neutrino_rmse_mean")
        lines.append(
            f"| {label} | {report['events']} | {report['top1_accuracy']:.5f} | "
            f"{report['truth_in_topk_fraction']:.5f} | "
            f"{correct if correct is not None else float('nan'):.5g} | "
            f"{wrong if wrong is not None else float('nan'):.5g} |"
        )
    (output / "assignment_conditioning.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
