#!/usr/bin/env python3
"""Find non-recurrent visible controls matched to M2/M3 parameter counts.

Parameter matching is a design calculation, not a physics result. This tool keeps the
semantic candidate head fixed and varies only the non-recurrent visible Transformer depth
and feed-forward width. Runtime matching should subsequently use benchmark_synthetic.py
on the shortlisted candidates.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from joint_event_reconstruction.configuration import validate_experiment
from joint_event_reconstruction.training.joint import JointEventModule


def _assignment_only(base, *, target: str):
    config = copy.deepcopy(base)
    config.model.mode = "factorized"
    config.model.assignment.formulation = "discriminative"
    config.model.invisible.enabled = False
    config.model.coupling.invisible_to_assignment.enabled = False
    config.model.compatibility.enabled = False
    config.training.loss_weights.assignment = 1.0
    config.training.loss_weights.invisible = 0.0
    config.training.loss_weights.compatibility = 0.0
    config.training.freeze.visible = False
    config.training.freeze.assignment = False
    config.training.freeze.invisible = False
    config.training.freeze.compatibility = False
    config.model.assignment.reasoning.enabled = target in {"m2", "m3"}
    config.model.assignment.feedback.enabled = target == "m3"
    validate_experiment(config)
    return config


def _parameters(config, visible) -> tuple[int, int]:
    module = JointEventModule(
        config,
        visible,
        invisible_scaler=None,
        synthetic_data=True,
    )
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return total, trainable


def match(
    experiment: Path,
    *,
    target: str,
    max_layers: int,
    feedforward_min: int,
    feedforward_max: int,
    feedforward_step: int,
    top: int,
) -> dict:
    if target not in {"m2", "m3"}:
        raise ValueError("target must be m2 or m3")
    if max_layers <= 0 or feedforward_min <= 0 or feedforward_max < feedforward_min:
        raise ValueError("Invalid control search bounds")
    if feedforward_step <= 0 or top <= 0:
        raise ValueError("feedforward_step and top must be positive")

    base = OmegaConf.load(experiment)
    visible = OmegaConf.load(base.visible_config)
    target_config = _assignment_only(base, target=target)
    target_total, target_trainable = _parameters(target_config, visible)

    controls = []
    for layers in range(1, max_layers + 1):
        for feedforward in range(
            feedforward_min, feedforward_max + 1, feedforward_step
        ):
            control = _assignment_only(base, target="m1")
            control.model.visible.layers = layers
            control.model.visible.d_feedforward = feedforward
            validate_experiment(control)
            total, trainable = _parameters(control, visible)
            controls.append(
                {
                    "visible_layers": layers,
                    "visible_d_feedforward": feedforward,
                    "parameters": total,
                    "trainable_parameters": trainable,
                    "absolute_parameter_difference": abs(total - target_total),
                    "relative_parameter_difference": (total - target_total) / target_total,
                    "overrides": [
                        "model.assignment.reasoning.enabled=false",
                        "model.assignment.feedback.enabled=false",
                        f"model.visible.layers={layers}",
                        f"model.visible.d_feedforward={feedforward}",
                    ],
                }
            )
    controls.sort(
        key=lambda item: (
            item["absolute_parameter_difference"],
            item["visible_layers"],
            item["visible_d_feedforward"],
        )
    )
    return {
        "purpose": "RECURRENCE_CONTROL_DESIGN_NOT_PHYSICS",
        "experiment": str(experiment.resolve()),
        "target": target,
        "target_parameters": target_total,
        "target_trainable_parameters": target_trainable,
        "target_overrides": [
            "model.assignment.reasoning.enabled=true",
            f"model.assignment.feedback.enabled={str(target == 'm3').lower()}",
        ],
        "closest_nonrecurrent_controls": controls[:top],
        "next_step": (
            "Benchmark the closest few controls and the recurrent target with identical "
            "synthetic batch/inference settings; choose a separate compute-matched control "
            "from measured runtime rather than parameter count."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/ttbar1L.yaml"),
    )
    parser.add_argument("--target", choices=("m2", "m3"), default="m3")
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--feedforward-min", type=int, default=64)
    parser.add_argument("--feedforward-max", type=int, default=1536)
    parser.add_argument("--feedforward-step", type=int, default=32)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = match(
        args.experiment.expanduser().resolve(),
        target=args.target,
        max_layers=args.max_layers,
        feedforward_min=args.feedforward_min,
        feedforward_max=args.feedforward_max,
        feedforward_step=args.feedforward_step,
        top=args.top,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
