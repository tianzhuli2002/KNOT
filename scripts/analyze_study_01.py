#!/usr/bin/env python3
"""Study 01 paired assignment and recurrent-depth analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.analyze_recurrent_depth import _paired, _plots, _read, analyse
from joint_event_reconstruction.study_metrics import assert_same_events


COMPARISONS = {
    "one_pass": ("m1", "default", "one-pass structured assignment"),
    "recurrent_r4": ("m2", "r4", "recurrent assignment refinement, R=4"),
    "recurrent_soft_r4": (
        "m3",
        "r4",
        "recurrent assignment + soft feedback, R=4",
    ),
    "parameter_control_m2": (
        "m2_parameter_control",
        "default",
        "parameter-matched non-recurrent control for recurrence",
    ),
    "compute_control_m2": (
        "m2_compute_control",
        "default",
        "training-compute-matched non-recurrent control for recurrence",
    ),
    "parameter_control_m3": (
        "m3_parameter_control",
        "default",
        "parameter-matched non-recurrent control for soft feedback",
    ),
    "compute_control_m3": (
        "m3_compute_control",
        "default",
        "training-compute-matched non-recurrent control for soft feedback",
    ),
}


def _prediction(root: Path, topology: str, variant: str, seed: int, name: str) -> Path:
    return (
        root
        / "visible_recurrence"
        / topology
        / variant
        / f"seed_{seed}"
        / "predictions"
        / f"{name}.h5"
    )


def analyze(root: Path, topology: str, seed: int) -> dict:
    recurrent = {}
    for variant in ("m2", "m3"):
        paths = [
            _prediction(root, topology, variant, seed, f"r{depth}")
            for depth in (1, 2, 3, 4, 8)
        ]
        recurrent[variant] = analyse(paths, [1, 2, 3, 4, 8])

    paths = {
        name: _prediction(root, topology, variant, seed, prediction)
        for name, (variant, prediction, _) in COMPARISONS.items()
    }
    assert_same_events(list(paths.values()))
    values = {name: _read(path) for name, path in paths.items()}
    if len({value["support"] for value in values.values()}) != 1:
        raise ValueError("Study 01 controls changed assignment support")

    baseline = values["one_pass"]["correct"]
    paired = {
        name: {
            "label": COMPARISONS[name][2],
            "prediction": str(paths[name]),
            **_paired(baseline, value["correct"], seed=12345 + index),
        }
        for index, (name, value) in enumerate(values.items())
        if name != "one_pass"
    }
    control_pairs = {
        "recurrent_r4_vs_m2_parameter_control": (
            "parameter_control_m2",
            "recurrent_r4",
            "R=4 recurrence vs M2 parameter-matched non-recurrent control",
        ),
        "recurrent_r4_vs_m2_training_compute_control": (
            "compute_control_m2",
            "recurrent_r4",
            "R=4 recurrence vs M2 training-compute-matched non-recurrent control",
        ),
        "recurrent_soft_r4_vs_m3_parameter_control": (
            "parameter_control_m3",
            "recurrent_soft_r4",
            "R=4 recurrence + soft feedback vs M3 parameter-matched non-recurrent control",
        ),
        "recurrent_soft_r4_vs_m3_training_compute_control": (
            "compute_control_m3",
            "recurrent_soft_r4",
            "R=4 recurrence + soft feedback vs M3 training-compute-matched non-recurrent control",
        ),
    }
    paired_controls = {}
    for index, (name, (control, target, label)) in enumerate(control_pairs.items()):
        paired_controls[name] = {
            "label": label,
            "control": COMPARISONS[control][2],
            "target": COMPARISONS[target][2],
            **_paired(
                values[control]["correct"],
                values[target]["correct"],
                seed=14000 + index,
            ),
        }
    result = {
        "study": "visible_recurrence",
        "topology": topology,
        "model_seed": seed,
        "selection_split": "validation",
        "assignment_support": values["one_pass"]["support"],
        "identical_ordered_event_ids": True,
        "recurrent_depth": recurrent,
        "paired_vs_one_pass": paired,
        "paired_vs_controls": paired_controls,
    }

    destination = root / "visible_recurrence" / topology / "analysis" / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "study01.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    for variant, depth_result in recurrent.items():
        _plots(depth_result, destination / variant)

    names = list(values)
    accuracy = np.asarray([values[name]["correct"].mean() for name in names])
    errors = np.sqrt(accuracy * (1.0 - accuracy) / len(baseline))
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    ax.errorbar(np.arange(len(names)), accuracy, yerr=errors, fmt="o", capsize=3)
    ax.set_xticks(
        np.arange(len(names)),
        [COMPARISONS[name][2] for name in names],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Complete-assignment accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(destination / "controls.png", dpi=220, bbox_inches="tight")
    fig.savefig(destination / "controls.pdf", bbox_inches="tight")
    plt.close(fig)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("results/studies"))
    parser.add_argument("--topology", default="ttbar2L")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = analyze(args.output_root.expanduser().resolve(), args.topology, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
