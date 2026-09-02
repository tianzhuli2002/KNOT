#!/usr/bin/env python3
"""Analyse event-by-event assignment changes across recurrent inference depth."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from joint_event_reconstruction.prediction_schema import PREDICTION_SCHEMA
from joint_event_reconstruction.study_metrics import assert_same_events


def _read(path: Path) -> dict:
    with h5py.File(path, "r") as source:
        if str(source.attrs["schema_version"]) != PREDICTION_SCHEMA:
            raise ValueError("Recurrent-depth analysis requires discriminative predictions")
        truth_topk = source["TRUTH/topk_assignment"][:].astype(bool)
        probabilities = source["VISIBLE/constrained_probabilities"][:]
        nodes = source["VISIBLE/nodes"]
        root_order = json.loads(str(source.attrs["root_order"]))
        entropy = np.stack(
            [nodes[name]["entropy"][:] for name in root_order], axis=1
        )
        margin = np.stack([nodes[name]["margin"][:] for name in root_order], axis=1)
        reconstructable = np.stack(
            [nodes[name]["reconstructable"][:] for name in root_order], axis=1
        ).all(axis=1)
        per_role = {}
        for name in root_order:
            node = nodes[name]
            role_support = node["reconstructable"][:].astype(bool)
            role_correct = node["selected_candidate"][:] == node["truth_candidate"][:]
            supported = int(role_support.sum())
            per_role[name] = {
                "reconstructable_events": supported,
                "accuracy_on_reconstructable": (
                    float(role_correct[role_support].mean())
                    if supported
                    else float("nan")
                ),
                "mean_truth_probability_on_reconstructable": (
                    float(node["truth_probability"][:][role_support].mean())
                    if supported
                    else float("nan")
                ),
                "mean_entropy": float(node["entropy"][:].mean()),
                "mean_margin": float(node["margin"][:].mean()),
            }
        complete_margin = (
            probabilities[:, 0] - probabilities[:, 1]
            if probabilities.shape[1] >= 2
            else np.full(len(probabilities), np.nan)
        )
        return {
            "events": source["EVENT_ID"][:],
            "correct": truth_topk[:, 0],
            "topk_truth": truth_topk.any(axis=1),
            "truth_probability": (probabilities * truth_topk).sum(axis=1),
            "retained_mass": probabilities.sum(axis=1),
            "complete_margin": complete_margin,
            "mean_node_entropy": entropy.mean(axis=1),
            "mean_node_margin": margin.mean(axis=1),
            "per_role": per_role,
            "njet": source["OBSERVED/njet"][:],
            "reconstructable": reconstructable,
            "support": str(source.attrs["assignment_support"]),
        }


def _se(value: float, count: int) -> float:
    return float(np.sqrt(value * (1.0 - value) / count)) if count else float("nan")


def _paired(before: np.ndarray, after: np.ndarray, seed: int = 12345) -> dict:
    before = np.asarray(before, dtype=bool)
    after = np.asarray(after, dtype=bool)
    delta = after.astype(np.int8) - before.astype(np.int8)
    values, counts = np.unique(delta, return_counts=True)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(delta), counts / counts.sum(), size=5000)
    bootstrap = (draws * values[None, :]).sum(axis=1) / len(delta)
    return {
        "events": int(len(delta)),
        "before_accuracy": float(before.mean()),
        "after_accuracy": float(after.mean()),
        "delta_accuracy": float(delta.mean()),
        "paired_bootstrap_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wrong_to_correct": int((~before & after).sum()),
        "correct_to_wrong": int((before & ~after).sum()),
        "stable_correct": int((before & after).sum()),
        "stable_wrong": int((~before & ~after).sum()),
        "net_transition_gain": int((~before & after).sum() - (before & ~after).sum()),
    }


def _quantile_groups(values: np.ndarray) -> list[tuple[str, np.ndarray]]:
    edges = np.quantile(values, (0.0, 0.25, 0.5, 0.75, 1.0))
    groups = []
    for index in range(4):
        lower, upper = edges[index], edges[index + 1]
        mask = (values >= lower) & (
            values <= upper if index == 3 else values < upper
        )
        groups.append((f"q{index + 1}", mask))
    return groups


def _stratified(
    initial: dict,
    final: dict,
) -> dict:
    first = initial["correct"]
    last = final["correct"]

    def summarize(groups):
        return {
            label: _paired(first[mask], last[mask])
            for label, mask in groups
            if mask.any()
        }

    order = np.argsort(initial["mean_node_entropy"], kind="stable")
    entropy_rank = np.empty(len(order), dtype=np.float64)
    entropy_rank[order] = np.arange(len(order)) / max(len(order) - 1, 1)
    order = np.argsort(-initial["mean_node_margin"], kind="stable")
    inverse_margin_rank = np.empty(len(order), dtype=np.float64)
    inverse_margin_rank[order] = np.arange(len(order)) / max(len(order) - 1, 1)
    ambiguity = 0.5 * (entropy_rank + inverse_margin_rank)

    njet_groups = [
        (str(int(value)), initial["njet"] == value)
        for value in np.unique(initial["njet"])
    ]
    return {
        "njet": summarize(njet_groups),
        "initial_node_entropy_quartile": summarize(
            _quantile_groups(initial["mean_node_entropy"])
        ),
        "initial_node_margin_quartile": summarize(
            _quantile_groups(initial["mean_node_margin"])
        ),
        "initial_combined_ambiguity_quartile": summarize(
            _quantile_groups(ambiguity)
        ),
        "reconstructability": summarize(
            [
                ("reconstructable", initial["reconstructable"]),
                ("not_reconstructable", ~initial["reconstructable"]),
            ]
        ),
        "btag_ambiguity": "not stored in maintained prediction schema; intentionally deferred",
    }


def analyse(paths: list[Path], depths: list[int]) -> dict:
    if len(paths) != len(depths) or len(paths) < 2:
        raise ValueError("Provide at least two predictions with one depth per file")
    order = np.argsort(depths)
    depths = [depths[index] for index in order]
    paths = [paths[index] for index in order]
    if len(set(depths)) != len(depths) or any(depth <= 0 for depth in depths):
        raise ValueError("Recurrent depths must be positive and unique")
    assert_same_events(paths)
    values = [_read(path) for path in paths]
    if len({item["support"] for item in values}) != 1:
        raise ValueError("Recurrent depth scan changed assignment support")

    count = len(values[0]["correct"])
    depth_metrics = []
    for depth, path, item in zip(depths, paths, values):
        accuracy = float(item["correct"].mean())
        depth_metrics.append(
            {
                "depth": depth,
                "prediction": str(path),
                "events": count,
                "accuracy": accuracy,
                "standard_error": _se(accuracy, count),
                "topk_truth_coverage": float(item["topk_truth"].mean()),
                "mean_truth_probability": float(item["truth_probability"].mean()),
                "mean_retained_assignment_mass": float(item["retained_mass"].mean()),
                "mean_complete_top1_top2_margin": float(item["complete_margin"].mean()),
                "mean_node_entropy": float(item["mean_node_entropy"].mean()),
                "mean_node_margin": float(item["mean_node_margin"].mean()),
                "per_role": item["per_role"],
            }
        )

    transitions = []
    for previous_depth, next_depth, previous, current in zip(
        depths[:-1], depths[1:], values[:-1], values[1:]
    ):
        before = previous["correct"]
        after = current["correct"]
        paired = _paired(before, after, seed=12345 + next_depth)
        transitions.append(
            {
                "from_depth": previous_depth,
                "to_depth": next_depth,
                **paired,
                "wrong_to_correct_fraction": float((~before & after).mean()),
                "correct_to_wrong_fraction": float((before & ~after).mean()),
            }
        )

    correctness = np.stack([item["correct"] for item in values], axis=1)
    ever_correct = correctness.any(axis=1)
    first_correct_index = np.argmax(correctness, axis=1)
    first_correct_depth = np.where(
        ever_correct,
        np.asarray(depths)[first_correct_index],
        -1,
    )
    final_correct = correctness[:, -1]
    overthought = ever_correct & ~final_correct
    recovered = ~correctness[:, 0] & final_correct
    return {
        "events": count,
        "assignment_support": values[0]["support"],
        "depths": depth_metrics,
        "transitions": transitions,
        "first_to_final_paired": _paired(
            correctness[:, 0], correctness[:, -1], seed=12445
        ),
        "stratified_first_to_final": _stratified(values[0], values[-1]),
        "event_level": {
            "ever_correct_fraction": float(ever_correct.mean()),
            "final_correct_fraction": float(final_correct.mean()),
            "wrong_at_first_but_correct_final_fraction": float(recovered.mean()),
            "correct_at_some_depth_but_wrong_final_fraction": float(overthought.mean()),
            "first_correct_depth_counts": {
                str(depth): int((first_correct_depth == depth).sum()) for depth in depths
            },
            "never_correct": int((~ever_correct).sum()),
        },
    }


def _plots(result: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    depths = np.asarray([item["depth"] for item in result["depths"]])
    accuracy = np.asarray([item["accuracy"] for item in result["depths"]])
    error = np.asarray([item["standard_error"] for item in result["depths"]])

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.errorbar(depths, accuracy, yerr=error, marker="o", capsize=3)
    ax.set_xlabel("Recurrent inference depth R")
    ax.set_ylabel("Top-1 complete-assignment efficiency")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(depths)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "accuracy_vs_depth.pdf", bbox_inches="tight")
    fig.savefig(output / "accuracy_vs_depth.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    transitions = result["transitions"]
    x = np.arange(len(transitions))
    labels = [f"{item['from_depth']}→{item['to_depth']}" for item in transitions]
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.plot(
        x,
        [item["wrong_to_correct_fraction"] for item in transitions],
        marker="o",
        label="wrong → correct",
    )
    ax.plot(
        x,
        [item["correct_to_wrong_fraction"] for item in transitions],
        marker="o",
        label="correct → wrong",
    )
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Depth transition")
    ax.set_ylabel("Fraction of all events")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "transitions_vs_depth.pdf", bbox_inches="tight")
    fig.savefig(output / "transitions_vs_depth.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--depths", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [path.expanduser().resolve() for path in args.predictions]
    result = analyse(paths, args.depths)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "recurrent_depth.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    with (output / "recurrent_depth.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result["depths"][0].keys())
        writer.writeheader()
        writer.writerows(result["depths"])
    _plots(result, output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
