#!/usr/bin/env python3
"""Measure event-level error complementarity for the Study-12 encoders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np


EXPECTED_VARIANTS = (
    "transformer_baseline",
    "fourvector_transformer",
    "relational_transformer",
    "lgatr",
    "lgatr_slim",
)


def _prediction(path: Path) -> dict[str, np.ndarray | list]:
    if not path.is_file():
        raise FileNotFoundError(f"Study-12 prediction does not exist: {path}")
    with h5py.File(path, "r") as source:
        required = {"EVENT_ID", "OBSERVED/njet", "TRUTH/topk_assignment", "VISIBLE/nodes"}
        missing = sorted(name for name in required if name not in source)
        if missing:
            raise KeyError(f"Study-12 prediction {path} is missing {missing}")
        event_id = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        topk = np.asarray(source["TRUTH/topk_assignment"][:], dtype=bool)
        if topk.ndim != 2 or not topk.shape[1]:
            raise ValueError(f"Study-12 truth top-K assignment has invalid shape: {path}")
        nodes = source["VISIBLE/nodes"]
        node_names = sorted(nodes.keys())
        if not node_names:
            raise ValueError(f"Study-12 prediction has no visible assignment nodes: {path}")
        entropy = np.stack(
            [np.asarray(nodes[name]["entropy"][:], dtype=np.float64) for name in node_names],
            axis=1,
        ).mean(axis=1)
        margin = np.stack(
            [np.asarray(nodes[name]["margin"][:], dtype=np.float64) for name in node_names],
            axis=1,
        ).mean(axis=1)
        njet = np.asarray(source["OBSERVED/njet"][:], dtype=np.int64)
        if len(event_id) == 0 or len(np.unique(event_id)) != len(event_id):
            raise ValueError(f"Study-12 prediction EVENT_ID must be non-empty and unique: {path}")
        arrays = (event_id, topk, entropy, margin, njet)
        if any(len(value) != len(event_id) for value in arrays[1:]):
            raise ValueError(f"Study-12 prediction fields have mismatched lengths: {path}")
        return {
            "event_id": event_id,
            "correct": topk[:, 0],
            "topk_coverage": topk.mean(axis=0),
            "entropy": entropy,
            "margin": margin,
            "njet": njet,
            "path": str(path.resolve()),
        }


def _same_events(reference: dict, candidate: dict, label: str) -> None:
    if not np.array_equal(reference["event_id"], candidate["event_id"]):
        raise ValueError(
            f"Study-12 complementarity requires identical ordered EVENT_IDs; mismatch in {label}"
        )


def _bootstrap_interval(values: np.ndarray, seed: int, replicates: int) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, replicates, 128):
        count = min(128, replicates - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.extend(values[indices].mean(axis=1).tolist())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _quantile_groups(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    finite = np.isfinite(values)
    if not finite.all():
        raise ValueError("Complementarity stratification variable contains non-finite values")
    edges = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    groups = []
    labels = []
    for index in range(4):
        lower, upper = edges[index], edges[index + 1]
        if index == 3:
            mask = (values >= lower) & (values <= upper)
        else:
            mask = (values >= lower) & (values < upper)
        groups.append(mask)
        labels.append(f"q{index + 1}[{lower:.6g},{upper:.6g}]")
    return np.asarray(groups), labels


def _stratification(
    disagreement: np.ndarray,
    a_correct: np.ndarray,
    b_correct: np.ndarray,
    njet: np.ndarray,
    entropy: np.ndarray,
    margin: np.ndarray,
) -> dict:
    def metrics(groups: np.ndarray, labels: list[str]) -> dict:
        result = {}
        for label, mask in zip(labels, groups, strict=True):
            count = int(mask.sum())
            disagree = disagreement & mask
            result[label] = {
                "events": count,
                "disagreement_fraction": float(disagree.mean()) if count else float("nan"),
                "a_accuracy_on_disagreement": (
                    float(a_correct[disagree].mean()) if disagree.any() else float("nan")
                ),
                "b_accuracy_on_disagreement": (
                    float(b_correct[disagree].mean()) if disagree.any() else float("nan")
                ),
                "both_correct_fraction": (
                    float((a_correct & b_correct & mask).sum() / count)
                    if count else float("nan")
                ),
                "both_wrong_fraction": (
                    float((~a_correct & ~b_correct & mask).sum() / count)
                    if count else float("nan")
                ),
            }
        return result

    njet_values = np.unique(njet)
    njet_groups = np.asarray([njet == value for value in njet_values])
    entropy_groups, entropy_labels = _quantile_groups(entropy)
    margin_groups, margin_labels = _quantile_groups(margin)
    return {
        "njet": metrics(njet_groups, [str(int(value)) for value in njet_values]),
        "assignment_entropy_quartile": metrics(entropy_groups, entropy_labels),
        "top1_top2_margin_quartile": metrics(margin_groups, margin_labels),
    }


def pair_record(a_name: str, a: dict, b_name: str, b: dict, replicates: int) -> dict:
    correct_a = np.asarray(a["correct"], dtype=bool)
    correct_b = np.asarray(b["correct"], dtype=bool)
    both_correct = correct_a & correct_b
    a_only = correct_a & ~correct_b
    b_only = ~correct_a & correct_b
    both_wrong = ~correct_a & ~correct_b
    disagreement = a_only | b_only
    delta = correct_a.astype(np.float64) - correct_b.astype(np.float64)
    seed_text = f"{a_name}|{b_name}".encode()
    bootstrap_seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:8], "little")
    n_events = len(correct_a)
    return {
        "a": a_name,
        "b": b_name,
        "events": n_events,
        "both_correct": int(both_correct.sum()),
        "a_correct_b_wrong": int(a_only.sum()),
        "a_wrong_b_correct": int(b_only.sum()),
        "both_wrong": int(both_wrong.sum()),
        "both_correct_fraction": float(both_correct.mean()),
        "a_correct_b_wrong_fraction": float(a_only.mean()),
        "a_wrong_b_correct_fraction": float(b_only.mean()),
        "both_wrong_fraction": float(both_wrong.mean()),
        "oracle_union_accuracy": float((correct_a | correct_b).mean()),
        "disagreement_fraction": float(disagreement.mean()),
        "a_accuracy_on_disagreement": float(correct_a[disagreement].mean()) if disagreement.any() else float("nan"),
        "b_accuracy_on_disagreement": float(correct_b[disagreement].mean()) if disagreement.any() else float("nan"),
        "paired_accuracy_delta_a_minus_b": float(delta.mean()),
        "paired_bootstrap_interval_a_minus_b": _bootstrap_interval(
            delta, bootstrap_seed, replicates
        ),
        "stratified": _stratification(
            disagreement,
            correct_a,
            correct_b,
            np.asarray(a["njet"]),
            0.5 * (np.asarray(a["entropy"]) + np.asarray(b["entropy"])),
            0.5 * (np.asarray(a["margin"]) + np.asarray(b["margin"])),
        ),
    }


def _matrix(records: list[dict], names: tuple[str, ...], field: str) -> np.ndarray:
    matrix = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    for index, name in enumerate(names):
        same = [record for record in records if record["a"] == name and record["b"] == name]
        if same:
            matrix[index, index] = float(same[0][field])
    for record in records:
        a, b = names.index(record["a"]), names.index(record["b"])
        matrix[a, b] = matrix[b, a] = float(record[field])
    return matrix


def _write_matrix(path: Path, matrix: np.ndarray, names: tuple[str, ...], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_title(title)
    for row in range(len(names)):
        for column in range(len(names)):
            if np.isfinite(matrix[row, column]):
                axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color="white")
    figure.colorbar(image, ax=axis, label="fraction / accuracy")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _parse_prediction(value: str) -> tuple[str, str, Path]:
    try:
        key, path_text = value.split("=", 1)
        seed, variant = key.split("/", 1)
    except ValueError as exc:
        raise ValueError(
            "--prediction must be SEED/VARIANT=PATH, e.g. seed_42/lgatr=predictions.h5"
        ) from exc
    if not seed.startswith("seed_") or not seed[5:].isdigit():
        raise ValueError(f"Prediction seed must be seed_N, found {seed!r}")
    if variant not in EXPECTED_VARIANTS:
        raise ValueError(f"Unknown Study-12 variant {variant!r}")
    return seed, variant, Path(path_text).expanduser().resolve()


def analyse(values: list[str], output: Path, replicates: int = 1000) -> dict:
    parsed = [_parse_prediction(value) for value in values]
    grouped: dict[str, dict[str, dict]] = {}
    for seed, variant, path in parsed:
        if variant in grouped.setdefault(seed, {}):
            raise ValueError(f"Duplicate Study-12 prediction for {seed}/{variant}")
        grouped[seed][variant] = _prediction(path)
    if not grouped:
        raise ValueError("At least one Study-12 prediction is required")

    per_seed = {}
    pair_records = []
    for seed in sorted(grouped):
        missing = sorted(set(EXPECTED_VARIANTS) - set(grouped[seed]))
        if missing:
            raise ValueError(f"Study-12 seed {seed} is missing variants {missing}")
        reference = grouped[seed][EXPECTED_VARIANTS[0]]
        for variant in EXPECTED_VARIANTS[1:]:
            _same_events(reference, grouped[seed][variant], f"{seed}/{variant}")
        records = []
        for a_name, b_name in combinations(EXPECTED_VARIANTS, 2):
            records.append(
                pair_record(a_name, grouped[seed][a_name], b_name, grouped[seed][b_name], replicates)
            )
        per_seed[seed] = {
            "events": int(len(reference["event_id"])),
            "variants": {
                name: {
                    "path": grouped[seed][name]["path"],
                    "accuracy": float(np.asarray(grouped[seed][name]["correct"]).mean()),
                    "topk_coverage": np.asarray(grouped[seed][name]["topk_coverage"]).tolist(),
                    "mean_entropy": float(np.asarray(grouped[seed][name]["entropy"]).mean()),
                    "mean_margin": float(np.asarray(grouped[seed][name]["margin"]).mean()),
                }
                for name in EXPECTED_VARIANTS
            },
            "pairs": records,
        }
        pair_records.extend({"seed": seed, **record} for record in records)

    aggregate = []
    for a_name, b_name in combinations(EXPECTED_VARIANTS, 2):
        records = [
            record for record in pair_records if record["a"] == a_name and record["b"] == b_name
        ]
        aggregate_record = {"a": a_name, "b": b_name, "seeds": len(records)}
        numeric_fields = (
            "both_correct_fraction",
            "a_correct_b_wrong_fraction",
            "a_wrong_b_correct_fraction",
            "both_wrong_fraction",
            "oracle_union_accuracy",
            "disagreement_fraction",
            "a_accuracy_on_disagreement",
            "b_accuracy_on_disagreement",
            "paired_accuracy_delta_a_minus_b",
        )
        for field in numeric_fields:
            values_for_field = np.asarray([record[field] for record in records], dtype=np.float64)
            finite = values_for_field[np.isfinite(values_for_field)]
            aggregate_record[f"{field}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
            aggregate_record[f"{field}_seed_std"] = (
                float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            )
        deltas = np.concatenate(
            [
                np.asarray(
                    grouped[record["seed"]][a_name]["correct"], dtype=np.float64
                )
                - np.asarray(grouped[record["seed"]][b_name]["correct"], dtype=np.float64)
                for record in records
            ]
        )
        aggregate_record["paired_bootstrap_interval_a_minus_b"] = _bootstrap_interval(
            deltas, 99173 + len(aggregate), replicates
        )
        aggregate.append(aggregate_record)

    output.mkdir(parents=True, exist_ok=True)
    per_seed_dir = output / "per_seed"
    per_seed_dir.mkdir(exist_ok=True)
    for seed, payload in per_seed.items():
        (per_seed_dir / f"{seed}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"
        )
        lines = [f"# Study-12 complementarity: {seed}", "", "No scientific winner is selected by this diagnostic.", ""]
        for record in payload["pairs"]:
            lines.append(
                f"- `{record['a']} vs {record['b']}`: oracle union "
                f"{record['oracle_union_accuracy']:.5f}, disagreement "
                f"{record['disagreement_fraction']:.5f}, paired delta "
                f"{record['paired_accuracy_delta_a_minus_b']:+.5f}."
            )
        (per_seed_dir / f"{seed}.md").write_text("\n".join(lines) + "\n")

    payload = {
        "schema": "study12_pairwise_complementarity_v1",
        "selection_split": "validation",
        "scientific_winner_selected": False,
        "variants": list(EXPECTED_VARIANTS),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }
    (output / "pairwise_complementarity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    with (output / "pairwise_complementarity.csv").open("w", newline="") as handle:
        fieldnames = ["scope", "seed", "a", "b", "metric", "value"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for seed, seed_payload in per_seed.items():
            for record in seed_payload["pairs"]:
                for metric in (
                    "both_correct_fraction",
                    "a_correct_b_wrong_fraction",
                    "a_wrong_b_correct_fraction",
                    "both_wrong_fraction",
                    "oracle_union_accuracy",
                    "disagreement_fraction",
                    "a_accuracy_on_disagreement",
                    "b_accuracy_on_disagreement",
                    "paired_accuracy_delta_a_minus_b",
                ):
                    writer.writerow({"scope": "seed", "seed": seed, "a": record["a"], "b": record["b"], "metric": metric, "value": record[metric]})
        for record in aggregate:
            for metric in (
                "both_correct_fraction_mean",
                "a_correct_b_wrong_fraction_mean",
                "a_wrong_b_correct_fraction_mean",
                "both_wrong_fraction_mean",
                "oracle_union_accuracy_mean",
                "disagreement_fraction_mean",
                "a_accuracy_on_disagreement_mean",
                "b_accuracy_on_disagreement_mean",
                "paired_accuracy_delta_a_minus_b_mean",
            ):
                writer.writerow({"scope": "aggregate", "seed": "", "a": record["a"], "b": record["b"], "metric": metric, "value": record[metric]})

    seed_records = []
    for seed_payload in per_seed.values():
        seed_records.extend(seed_payload["pairs"])
    aggregate_for_matrix = [
        {
            "a": record["a"],
            "b": record["b"],
            "oracle_union_accuracy": float(
                np.mean([item["oracle_union_accuracy"] for item in seed_records if item["a"] == record["a"] and item["b"] == record["b"]])
            ),
            "disagreement_fraction": float(
                np.mean([item["disagreement_fraction"] for item in seed_records if item["a"] == record["a"] and item["b"] == record["b"]])
            ),
        }
        for record in aggregate
    ]
    diagonal = []
    for name in EXPECTED_VARIANTS:
        accuracy = float(np.mean([seed_payload["variants"][name]["accuracy"] for seed_payload in per_seed.values()]))
        diagonal.append({"a": name, "b": name, "oracle_union_accuracy": accuracy, "disagreement_fraction": 0.0})
    matrix_records = aggregate_for_matrix + diagonal
    _write_matrix(output / "oracle_union_matrix.png", _matrix(matrix_records, EXPECTED_VARIANTS, "oracle_union_accuracy"), EXPECTED_VARIANTS, "Study 12 oracle union accuracy")
    _write_matrix(output / "oracle_union_matrix.pdf", _matrix(matrix_records, EXPECTED_VARIANTS, "oracle_union_accuracy"), EXPECTED_VARIANTS, "Study 12 oracle union accuracy")
    _write_matrix(output / "disagreement_matrix.png", _matrix(matrix_records, EXPECTED_VARIANTS, "disagreement_fraction"), EXPECTED_VARIANTS, "Study 12 disagreement fraction")
    _write_matrix(output / "disagreement_matrix.pdf", _matrix(matrix_records, EXPECTED_VARIANTS, "disagreement_fraction"), EXPECTED_VARIANTS, "Study 12 disagreement fraction")

    lines = [
        "# Study 12 human-review package",
        "",
        "This package reports event-level error complementarity on paired validation EVENT_IDs. It does not select a scientific winner.",
        "",
        "| pair | oracle union | disagreement | paired delta (A-B) | bootstrap interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for record in aggregate:
        interval = record["paired_bootstrap_interval_a_minus_b"]
        lines.append(
            f"| {record['a']} vs {record['b']} | {record['oracle_union_accuracy_mean']:.5f} | "
            f"{record['disagreement_fraction_mean']:.5f} | "
            f"{record['paired_accuracy_delta_a_minus_b_mean']:+.5f} | "
            f"[{interval[0]:+.5f}, {interval[1]:+.5f}] |"
        )
    (output / "study12_human_review.md").write_text("\n".join(lines) + "\n")
    (output / "study12_human_review.json").write_text(
        json.dumps(
            {"scientific_winner_selected": False, "aggregate": aggregate},
            indent=2,
            sort_keys=True,
            allow_nan=True,
        )
        + "\n"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    result = analyse(args.prediction, args.output, args.bootstrap_replicates)
    print(json.dumps({"output": str(args.output.resolve()), "seeds": sorted(result["per_seed"])}, indent=2))


if __name__ == "__main__":
    main()
