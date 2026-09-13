#!/usr/bin/env python3
"""Summarise paired M1 data scaling and the limited M3 diagnostic."""

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


def _read(path: Path) -> tuple[np.ndarray, np.ndarray]:
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        if str(source.attrs.get("split")) not in {"validation", "val"}:
            raise ValueError(f"Data-scaling analysis requires validation predictions: {path}")
        return (
            np.asarray(source["EVENT_ID"][:], dtype=np.int64),
            np.asarray(source["TRUTH/topk_assignment"][:, 0], dtype=bool),
        )


def _parse(items: list[str]) -> dict[int, dict[int, Path]]:
    result: dict[int, dict[int, Path]] = {}
    for item in items:
        label, path_text = item.split("=", 1)
        try:
            kind, count_text, seed_text = label.split(":")
            count = int(count_text)
            seed = int(seed_text)
        except ValueError as error:
            raise ValueError("Prediction labels must be M1:n_events:seed or M3:n_events:seed") from error
        if kind not in {"M1", "M3"} or count <= 0 or seed < 0:
            raise ValueError(f"Invalid data-scaling prediction label: {label}")
        key = 1 if kind == "M1" else 3
        encoded = (count << 16) + seed
        if encoded in result.setdefault(key, {}):
            raise ValueError(f"Duplicate data-scaling prediction: {label}")
        result[key][encoded] = Path(path_text).expanduser().resolve()
    return result


def analyse(m1_items: list[str], m3_items: list[str]) -> dict:
    parsed_m1 = _parse(m1_items).get(1, {})
    parsed_m3 = _parse(m3_items).get(3, {})
    if not parsed_m1:
        raise ValueError("At least one M1 scaling prediction is required")
    per_seed = []
    rows = []
    for encoded, path in sorted(parsed_m1.items()):
        count, seed = divmod(encoded, 1 << 16)
        event_id, correct = _read(path)
        row = {
            "model": "M1",
            "train_events": count,
            "seed": seed,
            "events": int(len(event_id)),
            "accuracy": float(correct.mean()),
            "prediction": str(path),
        }
        rows.append(row)
    for encoded, path in sorted(parsed_m3.items()):
        count, seed = divmod(encoded, 1 << 16)
        event_id, correct = _read(path)
        m1_path = parsed_m1.get(encoded)
        if m1_path is None:
            raise ValueError(f"M3 diagnostic {count}:{seed} has no paired M1 prediction")
        m1_event_id, m1_correct = _read(m1_path)
        if not np.array_equal(event_id, m1_event_id):
            raise ValueError(f"M1/M3 predictions do not share ordered EVENT_IDs at {count}:{seed}")
        rows.append({
            "model": "M3_diagnostic",
            "train_events": count,
            "seed": seed,
            "events": int(len(event_id)),
            "accuracy": float(correct.mean()),
            "prediction": str(path),
            "m1_accuracy": float(m1_correct.mean()),
            "m1_minus_m3": float(m1_correct.mean() - correct.mean()),
        })
    aggregate = []
    for model in ("M1", "M3_diagnostic"):
        for count in sorted({row["train_events"] for row in rows if row["model"] == model}):
            values = [row["accuracy"] for row in rows if row["model"] == model and row["train_events"] == count]
            aggregate.append({
                "model": model,
                "train_events": count,
                "seeds": len(values),
                "accuracy_mean": float(np.mean(values)),
                "accuracy_seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            })
    paired = [row for row in rows if row["model"] == "M3_diagnostic"]
    return {
        "schema": "study11_data_scaling_diagnostic_v1",
        "selection_split": "validation",
        "scientific_winner_selected": False,
        "primary_model": "M1 one-pass selected visible system",
        "secondary_model": "M3 recurrent + soft-feedback diagnostic only",
        "per_seed": rows,
        "aggregate": aggregate,
        "paired_m1_minus_m3": [
            {
                "train_events": row["train_events"],
                "seed": row["seed"],
                "delta": row["m1_minus_m3"],
            }
            for row in paired
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", action="append", default=[], metavar="M1:n_events:seed=PATH")
    parser.add_argument("--m3", action="append", default=[], metavar="M3:n_events:seed=PATH")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyse(args.m1, args.m3)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_scaling_recurrence.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Study 11 data scaling",
        "",
        "M1 is the primary scaling curve. M3 is a limited labelled diagnostic and does not reopen Study 01.",
        "",
        "| model | training events | seed count | mean accuracy | seed std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in result["aggregate"]:
        lines.append(f"| {row['model']} | {row['train_events']} | {row['seeds']} | {row['accuracy_mean']:.5f} | {row['accuracy_seed_std']:.5f} |")
    (output / "data_scaling_recurrence.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
