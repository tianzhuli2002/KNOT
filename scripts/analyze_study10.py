#!/usr/bin/env python3
"""Aggregate the controlled Study-10 hypothesis-evaluation comparison.

The script consumes validation prediction products only.  The optional
``HYPOTHESES`` group is the deployable/offline physics product; when it is
absent the script reports the missing comparison explicitly rather than
repeating a top-1 hypothesis and calling it a top-K result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.prediction_schema import validate_prediction


ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
REQUESTED_TOP_K = (1, 2, 3, 5, 10)


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _transitions(baseline: np.ndarray, selected: np.ndarray) -> dict[str, int]:
    return {
        "baseline_correct_to_remains_correct": int((baseline & selected).sum()),
        "baseline_correct_to_broken": int((baseline & ~selected).sum()),
        "baseline_wrong_to_rescued": int((~baseline & selected).sum()),
        "baseline_wrong_to_remains_wrong": int((~baseline & ~selected).sum()),
    }


def _labels(values: np.ndarray, prefix: str) -> np.ndarray:
    edges = np.unique(np.quantile(values, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))
    if len(edges) < 2:
        return np.full(values.shape, f"{prefix}:constant", dtype=object)
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return np.asarray([f"{prefix}:q{int(value) + 1}" for value in bins], dtype=object)


def _stratified(
    labels: np.ndarray,
    baseline: np.ndarray,
    ceiling: np.ndarray,
    selected: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        result[str(label)] = {
            "events": int(mask.sum()),
            "visible_accuracy": float(baseline[mask].mean()),
            "top_k_ceiling": float(ceiling[mask].mean()),
            "selected_accuracy": float(selected[mask].mean()),
            "delta_vs_visible": float(selected[mask].mean() - baseline[mask].mean()),
        }
    return result


def _ranking(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, score, -np.inf)
    return np.argsort(-masked, axis=1, kind="stable")


def _record(path: Path) -> dict:
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        split = _text(source.attrs.get("split", ""))
        if split != "validation":
            raise ValueError(f"Study 10 accepts validation predictions only: {path}")
        event_id = source["EVENT_ID"][:]
        scores = source["VISIBLE/assignment_log_scores"][:].astype(np.float64)
        truth = source["TRUTH/topk_assignment"][:].astype(bool, copy=False)
        valid = source["VISIBLE/valid_hypotheses"][:].astype(bool, copy=False)
        njet = source["OBSERVED/njet"][:]
        entropy = np.stack(
            [source[f"VISIBLE/nodes/{node}/entropy"][:] for node in source["VISIBLE/nodes"]],
            axis=1,
        ).mean(axis=1)
        margin = np.stack(
            [source[f"VISIBLE/nodes/{node}/margin"][:] for node in source["VISIBLE/nodes"]],
            axis=1,
        ).mean(axis=1)
        if split != "validation" or len(event_id) == 0 or len(np.unique(event_id)) != len(event_id):
            raise ValueError(f"Invalid Study-10 event identity in {path}")
        if scores.shape != truth.shape or valid.shape != truth.shape:
            raise ValueError(f"Inconsistent Study-10 hypothesis shapes in {path}")
        hypotheses = source.get("HYPOTHESES")
        has_hypotheses = hypotheses is not None
        branch_count = (
            int(source["INVISIBLE"].attrs.get("unique_assignment_conditioning_branches", -1))
            if "INVISIBLE" in source else None
        )
        has_matched_control = (
            hypotheses is not None and "matched_visible_only_log_score" in hypotheses
        )
        scorer_type = (
            _text(hypotheses.attrs.get("scorer_type", ""))
            if hypotheses is not None else ""
        )
        analytic = None
        learned = None
        truth_oracle = None
        if hypotheses is not None:
            if "PHYSICS/analytic_score" in hypotheses:
                analytic = hypotheses["PHYSICS/analytic_score"][:].astype(np.float64)
            if "final_log_score" in hypotheses and scorer_type.startswith("learned"):
                learned = hypotheses["final_log_score"][:].astype(np.float64)
            if "truth_neutrino_analytic_score" in hypotheses:
                truth_oracle = hypotheses["truth_neutrino_analytic_score"][:].astype(np.float64)
            for name, values in (("analytic", analytic), ("learned", learned), ("truth_oracle", truth_oracle)):
                if values is not None and values.shape != scores.shape:
                    raise ValueError(f"Study-10 {name} score shape disagrees with visible hypotheses")

    baseline_ranking = _ranking(scores, valid)
    baseline_correct = truth[np.arange(len(truth)), baseline_ranking[:, 0]]
    ceiling = truth[np.arange(len(truth)), :].any(axis=1)
    njet_labels = np.asarray(
        [
            "njet:0-3" if value <= 3 else
            "njet:4-5" if value <= 5 else
            "njet:6-7" if value <= 7 else "njet:8+"
            for value in njet
        ],
        dtype=object,
    )
    strata = {
        "njet": njet_labels,
        "entropy": _labels(entropy, "entropy"),
        "margin": _labels(margin, "margin"),
    }
    result: dict[str, object] = {
        "prediction": str(path.resolve()),
        "events": int(len(event_id)),
        "selection_split": "validation",
        "test_split_used": False,
        "available_top_k": int(truth.shape[1]),
        "posterior_branches": branch_count,
        "top_k_coverage": {
            str(k): float(truth[:, :k].any(axis=1).mean()) if k <= truth.shape[1] else None
            for k in REQUESTED_TOP_K
        },
        "visible_baseline": {
            "accuracy": float(baseline_correct.mean()),
            "transitions": _transitions(baseline_correct, baseline_correct),
        },
        "physics_feature_ablation": {
            "status": "stored_in_hypothesis_artifact" if has_hypotheses else "unavailable",
            "note": "Feature ablations must be evaluated from the stored reduced feature artifact; no posterior is regenerated.",
        },
        "matched_visible_only_control": {
            "status": "stored_in_hypothesis_artifact" if has_matched_control else "unavailable",
        },
        "analytic_physics": {"status": "available" if analytic is not None else "unavailable"},
        "learned_physics": {"status": "available" if learned is not None else "unavailable"},
        "truth_neutrino_oracle": {"status": "available" if truth_oracle is not None else "unavailable"},
    }

    if analytic is not None:
        scan = {}
        for alpha in ALPHAS:
            ranking = _ranking(scores + alpha * analytic, valid)
            selected = truth[np.arange(len(truth)), ranking[:, 0]]
            scan[str(alpha)] = {
                "accuracy": float(selected.mean()),
                "delta_vs_visible": float(selected.mean() - baseline_correct.mean()),
                "transitions": _transitions(baseline_correct, selected),
                "by": {
                    name: _stratified(labels, baseline_correct, ceiling, selected)
                    for name, labels in strata.items()
                },
            }
        result["analytic_physics"] = {
            "status": "available",
            "alpha_scan": scan,
            "score_is_normalised_likelihood": False,
        }
    if learned is not None:
        ranking = _ranking(learned, valid)
        selected = truth[np.arange(len(truth)), ranking[:, 0]]
        result["learned_physics"] = {
            "status": "available",
            "accuracy": float(selected.mean()),
            "delta_vs_visible": float(selected.mean() - baseline_correct.mean()),
            "transitions": _transitions(baseline_correct, selected),
            "by": {
                name: _stratified(labels, baseline_correct, ceiling, selected)
                for name, labels in strata.items()
            },
        }
    if truth_oracle is not None:
        ranking = _ranking(truth_oracle, valid)
        selected = truth[np.arange(len(truth)), ranking[:, 0]]
        result["truth_neutrino_oracle"] = {
            "status": "available",
            "accuracy": float(selected.mean()),
            "delta_vs_visible": float(selected.mean() - baseline_correct.mean()),
            "transitions": _transitions(baseline_correct, selected),
        }
    return result


def build_review(predictions: list[Path], output: Path) -> tuple[Path, Path]:
    if not predictions:
        raise ValueError("Study 10 requires at least one prediction")
    records = [_record(path.expanduser().resolve()) for path in predictions]
    review = {
        "study": "compatibility_reranking",
        "topology": "ttbar2L",
        "selection_split": "validation",
        "test_split_used": False,
        "records": records,
        "controlled_comparisons": [
            "visible baseline",
            "matched visible-only control",
            "analytic physics compatibility",
            "learned physics correction",
            "truth-neutrino oracle",
            "top-K ceiling",
        ],
        "decision": "review_required",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    md_path = output.with_suffix(".md")
    json_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Study 10 — invisible physics and assignment",
        "",
        "Selection split: `validation`. Test split used: `False`.",
        "",
        "The truth-neutrino row is an oracle headroom diagnostic, not a deployable model.",
        "The analytic score is an unnormalised compatibility/evidence-like score.",
        "",
        "| prediction | events | visible | top-K ceiling | analytic/learned status |",
        "|---|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| {Path(record['prediction']).name} | {record['events']} | "
            f"{record['visible_baseline']['accuracy']:.6f} | "
            f"{record['top_k_coverage'].get(str(record['available_top_k'])):.6f} | "
            f"{record['analytic_physics']['status']} / {record['learned_physics']['status']} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in build_review(args.prediction, args.output.expanduser().resolve()):
        print(path)


if __name__ == "__main__":
    main()
