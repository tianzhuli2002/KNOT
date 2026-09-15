#!/usr/bin/env python3
"""Summarise validation top-K headroom without regenerating a posterior.

The canonical factorised prediction files currently store the selected visible
hypothesis only (``K=1``).  This script reports that fact explicitly instead of
silently treating a top-1 file as a top-K oracle.  Files produced by the
optional ``HYPOTHESES`` extension can additionally provide future offline
truth-neutrino scores, but those scores are never inferred from ordinary
posterior samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REQUESTED_TOP_K = (1, 2, 3, 5, 10)


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _bin_summary(
    truth_topk: np.ndarray,
    labels: np.ndarray,
    available_k: int,
) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for label in sorted(set(labels.tolist())):
        selected = labels == label
        entry: dict[str, float | int | None] = {"events": int(selected.sum())}
        for requested in REQUESTED_TOP_K:
            if requested > available_k:
                entry[f"top_{requested}"] = None
            else:
                entry[f"top_{requested}"] = float(
                    truth_topk[selected, :requested].any(axis=1).mean()
                )
        result[str(label)] = entry
    return result


def _quantile_labels(values: np.ndarray, prefix: str) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=object)
    edges = np.unique(np.quantile(values, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))
    if len(edges) < 2:
        return np.full(values.shape, f"{prefix}:constant", dtype=object)
    labels = np.searchsorted(edges[1:-1], values, side="right")
    return np.asarray(
        [f"{prefix}:q{int(index) + 1}" for index in labels], dtype=object
    )


def _read_one(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Coupling analysis prediction does not exist: {path}")
    with h5py.File(path, "r") as source:
        split = _text(source.attrs.get("split", ""))
        if split != "validation":
            raise ValueError(
                f"Coupling analysis only accepts validation predictions: {path} has split={split!r}"
            )
        event_id = source["EVENT_ID"][:]
        scores = source["VISIBLE/assignment_log_scores"][:]
        truth_topk = source["TRUTH/topk_assignment"][:]
        if scores.ndim != 2 or truth_topk.shape != scores.shape:
            raise ValueError(f"Invalid visible/top-K shapes in {path}")
        count, available_k = scores.shape
        if count == 0 or len(np.unique(event_id)) != count:
            raise ValueError(f"Prediction has empty or duplicate EVENT_ID values: {path}")
        njet = source["OBSERVED/njet"][:]
        if njet.shape != (count,):
            raise ValueError(f"Invalid Njet shape in {path}")
        entropy_values = []
        margin_values = []
        for node in source["VISIBLE/nodes"]:
            entropy_values.append(source[f"VISIBLE/nodes/{node}/entropy"][:])
            margin_values.append(source[f"VISIBLE/nodes/{node}/margin"][:])
        if not entropy_values or not margin_values:
            raise ValueError(f"Visible assignment diagnostics are missing in {path}")
        entropy = np.stack(entropy_values, axis=1).mean(axis=1)
        margin = np.stack(margin_values, axis=1).mean(axis=1)
        if not np.isfinite(entropy).all() or not np.isfinite(margin).all():
            raise ValueError(f"Visible assignment diagnostics are non-finite in {path}")

        truth_topk = truth_topk.astype(bool, copy=False)
        coverage = {}
        for requested in REQUESTED_TOP_K:
            coverage[str(requested)] = (
                None
                if requested > available_k
                else float(truth_topk[:, :requested].any(axis=1).mean())
            )
        njet_labels = np.asarray(
            [
                "njet:0-3" if value <= 3 else
                "njet:4-5" if value <= 5 else
                "njet:6-7" if value <= 7 else
                "njet:8+"
                for value in njet
            ],
            dtype=object,
        )
        hypotheses = source.get("HYPOTHESES")
        truth_neutrino = None
        if hypotheses is not None and "truth_neutrino_analytic_score" in hypotheses:
            values = hypotheses["truth_neutrino_analytic_score"][:]
            if values.shape != (count, available_k):
                raise ValueError(
                    f"Invalid truth-neutrino score shape in {path}: {values.shape}"
                )
            truth_neutrino = values

        record = {
            "prediction": str(path.resolve()),
            "events": int(count),
            "split": split,
            "topology_signature": _text(source.attrs.get("topology_signature", "")),
            "physics_signature": _text(source.attrs.get("physics_signature", "none")),
            "available_top_k": int(available_k),
            "coverage": coverage,
            "by_njet": _bin_summary(truth_topk, njet_labels, available_k),
            "by_entropy": _bin_summary(
                truth_topk, _quantile_labels(entropy, "entropy"), available_k
            ),
            "by_margin": _bin_summary(
                truth_topk, _quantile_labels(margin, "margin"), available_k
            ),
            "baseline_top1_accuracy": float(truth_topk[:, 0].mean()),
            "truth_neutrino_reranking": {
                "status": "available" if truth_neutrino is not None else "unavailable",
                "reason": (
                    None
                    if truth_neutrino is not None
                    else "Prediction contains no explicit truth-neutrino hypothesis scores"
                ),
            },
            "transition_analysis": {
                "status": "unavailable",
                "reason": "A truth-neutrino reranked top-K score is required",
            },
            "physics_feature_ablation": {
                "status": "unavailable",
                "reason": "A stored per-hypothesis physics feature artifact is required",
            },
        }
        if truth_neutrino is not None:
            # This branch is intentionally explicit and only consumes a dataset
            # named for the truth-neutrino oracle; ordinary posterior scores are
            # not relabelled as an oracle.
            reranked = np.argmax(truth_neutrino, axis=1)
            baseline_correct = truth_topk[:, 0]
            reranked_correct = truth_topk[np.arange(count), reranked]
            record["truth_neutrino_reranking"] = {
                "status": "available",
                "accuracy": float(reranked_correct.mean()),
                "delta_vs_visible": float(
                    reranked_correct.mean() - baseline_correct.mean()
                ),
            }
            record["transition_analysis"] = {
                "status": "available",
                "baseline_correct_to_remains_correct": int(
                    (baseline_correct & reranked_correct).sum()
                ),
                "baseline_correct_to_broken": int(
                    (baseline_correct & ~reranked_correct).sum()
                ),
                "baseline_wrong_to_rescued": int(
                    (~baseline_correct & reranked_correct).sum()
                ),
                "baseline_wrong_to_remains_wrong": int(
                    (~baseline_correct & ~reranked_correct).sum()
                ),
            }
    return record


def build_review(predictions: list[Path], output: Path) -> tuple[Path, Path]:
    records = [_read_one(path.expanduser().resolve()) for path in predictions]
    if not records:
        raise ValueError("At least one prediction is required")
    output.parent.mkdir(parents=True, exist_ok=True)
    review = {
        "study": "compatibility_reranking",
        "topology": "ttbar2L",
        "selection_split": "validation",
        "test_split_used": False,
        "requested_top_k": list(REQUESTED_TOP_K),
        "records": records,
        "decision": {
            "status": (
                "oracle_available"
                if any(
                    item["truth_neutrino_reranking"]["status"] == "available"
                    for item in records
                )
                else "insufficient_top_k_oracle_artifact"
            ),
            "interpretation": (
                "Top-1 is the visible baseline. K>1 coverage and truth-neutrino reranking "
                "require an explicitly stored top-K hypothesis artifact; no headroom is "
                "claimed when the supplied prediction has K=1."
            ),
        },
    }
    json_path = output.with_suffix(".json")
    md_path = output.with_suffix(".md")
    json_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Invisible-to-assignment coupling oracle",
        "",
        "Selection split: `validation`. Test split used: `False`.",
        "",
        "| prediction | events | available K | K=1 | K=2 | K=3 | K=5 | K=10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        values = record["coverage"]
        rendered = [
            "—" if values[str(k)] is None else f"{values[str(k)]:.6f}"
            for k in REQUESTED_TOP_K
        ]
        lines.append(
            f"| {Path(record['prediction']).name} | {record['events']} | "
            f"{record['available_top_k']} | " + " | ".join(rendered) + " |"
        )
    lines += [
        "",
        "The supplied canonical factorised predictions contain only K=1. The K>1 "
        "coverage, truth-neutrino reranking, transitions, and feature ablations are "
        "therefore reported as unavailable rather than inferred from a repeated top-1 state.",
        "",
        "Decision: `insufficient_top_k_oracle_artifact`; this is not evidence that "
        "coupling has zero scientific headroom.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction", type=Path, action="append", required=True,
        help="Validation prediction HDF5; repeat for per-seed records.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/studies/compatibility_reranking/coupling_oracle"),
    )
    args = parser.parse_args()
    for path in build_review(args.prediction, args.output.expanduser().resolve()):
        print(path)


if __name__ == "__main__":
    main()
