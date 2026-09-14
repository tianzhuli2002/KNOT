#!/usr/bin/env python3
"""Write the compact human review for the authorised Study-05 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mode(values: np.ndarray) -> np.ndarray:
    modes = np.empty(len(values), dtype=values.dtype)
    for event, samples in enumerate(values):
        unique, counts = np.unique(samples, return_counts=True)
        modes[event] = unique[np.argmax(counts)]
    return modes


def _correct(path: Path, formulation: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as source:
        event_id = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        if formulation == "discriminative":
            correct = np.asarray(source["TRUTH/topk_assignment"][:, 0], dtype=bool)
        else:
            states = np.asarray(source["HYPOTHESES/assignment_state"][:])
            truth = np.asarray(source["TRUTH/assignment_state"][:])
            correct = _mode(states) == truth
    return event_id, correct


def _paired_bootstrap(delta: np.ndarray, *, seed: int = 505, samples: int = 1000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = delta[generator.integers(0, len(delta), len(delta))].mean()
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def review(root: Path) -> dict:
    root = root.expanduser().resolve()
    rows = []
    paired = []
    source_files = {}
    for seed in (42, 43, 44):
        base = root / "ttbar2L"
        m1 = base / "m1" / f"seed_{seed}" / "predictions/top1.h5"
        dfm16 = base / "dfm" / f"seed_{seed}" / "predictions/s64_t16.h5"
        dfm8 = base / "dfm" / f"seed_{seed}" / "predictions/s64_t8.h5"
        for label, path in (("m1", m1), ("dfm_s64_t16", dfm16), ("dfm_s64_t8", dfm8)):
            if not path.is_file():
                raise FileNotFoundError(f"Study-05 review input is missing: {path}")
            source_files[label + f"_seed_{seed}"] = {
                "path": str(path),
                "sha256": _sha256(path),
            }

        m1_ids, m1_correct = _correct(m1, "discriminative")
        dfm16_ids, dfm16_correct = _correct(dfm16, "flow_matching")
        dfm8_ids, dfm8_correct = _correct(dfm8, "flow_matching")
        if not (np.array_equal(m1_ids, dfm16_ids) and np.array_equal(m1_ids, dfm8_ids)):
            raise ValueError(f"Study-05 prediction EVENT_IDs are not paired for seed {seed}")
        delta = m1_correct.astype(np.int8) - dfm16_correct.astype(np.int8)
        paired.append(
            {
                "seed": seed,
                "m1_minus_dfm_s64_t16_mean": float(delta.mean()),
                "m1_correct_events": int(m1_correct.sum()),
                "dfm_s64_t16_correct_events": int(dfm16_correct.sum()),
                "paired_bootstrap_95": list(_paired_bootstrap(delta)),
                "events": int(len(delta)),
            }
        )
        rows.extend(
            [
                {"model": "M1", "seed": seed, "accuracy": float(m1_correct.mean())},
                {"model": "DFM S=64,T=16", "seed": seed, "accuracy": float(dfm16_correct.mean())},
                {"model": "DFM S=64,T=8", "seed": seed, "accuracy": float(dfm8_correct.mean())},
            ]
        )

    aggregate = {}
    for model in ("M1", "DFM S=64,T=16", "DFM S=64,T=8"):
        values = np.asarray([row["accuracy"] for row in rows if row["model"] == model])
        aggregate[model] = {
            "mean": float(values.mean()),
            "seed_std": float(values.std(ddof=1)),
            "seeds": [42, 43, 44],
        }
    result = {
        "schema": "study05_assignment_formulation_review_v1",
        "selection_split": "validation",
        "selected_decision": "assignment_formulation=discriminative",
        "aggregate": aggregate,
        "per_seed": rows,
        "paired_m1_minus_dfm_s64_t16": paired,
        "inference_cost": {
            "M1": {
                "description": "one discriminative assignment forward",
                "network_calls_per_event": 1,
            },
            "DFM S=64,T=16": {
                "description": "64 posterior assignment samples x 16 flow steps",
                "network_calls_per_event": 1024,
                "relative_to_m1": 1024,
            },
            "DFM S=64,T=8": {
                "description": "64 posterior assignment samples x 8 flow steps",
                "network_calls_per_event": 512,
                "relative_to_m1": 512,
            },
        },
        "source_prediction_files": source_files,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/studies/discrete_assignment"),
        help="Study-05 result directory.",
    )
    args = parser.parse_args()
    result = review(args.root)
    output = args.root.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "assignment_formulation_review.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Study 05 assignment-formulation review",
        "",
        "Selection split: validation. The authorised gate is `assignment_formulation=discriminative`.",
        "",
        "| model | mean accuracy | seed std | inference calls/event |",
        "|---|---:|---:|---:|",
    ]
    for model, summary in result["aggregate"].items():
        cost = result["inference_cost"][model]["network_calls_per_event"]
        lines.append(f"| {model} | {summary['mean']:.6f} | {summary['seed_std']:.6f} | {cost} |")
    lines.extend(["", "## Paired M1 minus strongest DFM (S=64,T=16)", ""])
    for row in result["paired_m1_minus_dfm_s64_t16"]:
        low, high = row["paired_bootstrap_95"]
        lines.append(
            f"- seed {row['seed']}: {row['m1_minus_dfm_s64_t16_mean']:.6f} "
            f"(bootstrap 95% interval {low:.6f} to {high:.6f})"
        )
    (output / "assignment_formulation_review.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
