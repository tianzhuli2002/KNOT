#!/usr/bin/env python3
"""Generate parameter- and compute-matched non-recurrent controls for M2/M3."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from joint_event_reconstruction.configuration import load_experiment, validate_experiment
from joint_event_reconstruction.training.joint import JointEventModule
from scripts.smoke_synthetic import build_synthetic_batch


def _assignment_only(base, target: str):
    config = copy.deepcopy(base)
    config.model.mode = "factorized"
    config.model.assignment.formulation = "discriminative"
    config.model.invisible.enabled = False
    config.model.coupling.invisible_to_assignment.enabled = False
    config.model.compatibility.enabled = False
    config.training.trainable = ["visible", "assignment"]
    config.training.loss_weights.assignment = 1.0
    config.training.loss_weights.invisible = 0.0
    config.training.loss_weights.compatibility = 0.0
    config.model.assignment.reasoning.enabled = target in {"m2", "m3"}
    config.model.assignment.feedback.enabled = target == "m3"
    validate_experiment(config)
    return config


def _parameters(config, visible) -> tuple[int, int]:
    module = JointEventModule(config, visible, invisible_scaler=None, synthetic_data=True)
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return total, trainable


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _runtime(
    config,
    visible,
    batch,
    *,
    device: str,
    recurrent_steps: int | None,
    warmup: int,
    iterations: int,
) -> dict:
    module = JointEventModule(
        config, visible, invisible_scaler=None, synthetic_data=True
    ).to(device)

    def training_step() -> None:
        module.zero_grad(set_to_none=True)
        loss = module(batch, recurrent_steps=recurrent_steps)["loss"]
        loss.backward()

    module.train()
    for _ in range(warmup):
        training_step()
    _sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iterations):
        training_step()
    _sync(device)
    train_seconds = (time.perf_counter() - start) / iterations
    train_memory = (
        torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    )

    module.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            module(batch, recurrent_steps=recurrent_steps)
        _sync(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iterations):
            module(batch, recurrent_steps=recurrent_steps)
        _sync(device)
        inference_seconds = (time.perf_counter() - start) / iterations
        inference_memory = (
            torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
        )

    del module
    if device == "cuda":
        torch.cuda.empty_cache()
    return {
        "training_seconds_per_step": train_seconds,
        "inference_seconds_per_call": inference_seconds,
        "training_peak_memory_mib": train_memory,
        "inference_peak_memory_mib": inference_memory,
    }


def _control_patch(layers: int, feedforward: int) -> dict:
    return {
        "model": {
            "encoder": {"layers": layers, "d_feedforward": feedforward},
            "assignment": {"method": "discriminative"},
        },
        "training": {"trainable": ["visible", "assignment"]},
    }


def _target_patch(target: str, steps: int) -> dict:
    recurrence = {"train_steps": [1, 2, 3, steps], "eval_steps": steps}
    if target == "m3":
        recurrence["feedback"] = "soft"
    return {
        "model": {
            "assignment": {
                "method": "discriminative",
                "recurrence": recurrence,
            }
        },
        "training": {"trainable": ["visible", "assignment"]},
    }


def match(
    experiment: Path,
    *,
    target: str,
    recurrent_steps: int,
    max_layers: int,
    feedforward_min: int,
    feedforward_max: int,
    feedforward_step: int,
    top: int,
    benchmark_candidates: int,
    device: str,
    events: int,
    warmup: int,
    iterations: int,
) -> dict:
    if target not in {"m2", "m3"}:
        raise ValueError("target must be m2 or m3")
    if recurrent_steps <= 0 or max_layers <= 0:
        raise ValueError("recurrent_steps and max_layers must be positive")
    if feedforward_min <= 0 or feedforward_max < feedforward_min:
        raise ValueError("Invalid feed-forward search bounds")
    if feedforward_step <= 0 or top <= 0 or benchmark_candidates <= 0:
        raise ValueError("feedforward_step, top and benchmark_candidates must be positive")
    if events < 2 or warmup < 0 or iterations <= 0:
        raise ValueError("events>=2, warmup>=0 and iterations>0 are required")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA recurrence-control benchmark requested but CUDA is unavailable")

    base = load_experiment(experiment)
    visible, batch, _ = build_synthetic_batch(base, events=events, device=device)
    target_config = _assignment_only(base, target)
    target_config.model.assignment.reasoning.train_steps = [1, 2, 3, recurrent_steps]
    target_config.model.assignment.reasoning.eval_steps = recurrent_steps
    validate_experiment(target_config)
    target_total, target_trainable = _parameters(target_config, visible)

    controls = []
    for layers in range(1, max_layers + 1):
        for feedforward in range(
            feedforward_min, feedforward_max + 1, feedforward_step
        ):
            control = _assignment_only(base, "m1")
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
                    "patch": _control_patch(layers, feedforward),
                }
            )
    controls.sort(
        key=lambda item: (
            item["absolute_parameter_difference"],
            item["visible_layers"],
            item["visible_d_feedforward"],
        )
    )

    target_runtime = _runtime(
        target_config,
        visible,
        batch,
        device=device,
        recurrent_steps=recurrent_steps,
        warmup=warmup,
        iterations=iterations,
    )
    benchmarked = []
    for candidate in controls[:benchmark_candidates]:
        control = _assignment_only(base, "m1")
        control.model.visible.layers = candidate["visible_layers"]
        control.model.visible.d_feedforward = candidate["visible_d_feedforward"]
        runtime = _runtime(
            control,
            visible,
            batch,
            device=device,
            recurrent_steps=None,
            warmup=warmup,
            iterations=iterations,
        )
        train_ratio = (
            runtime["training_seconds_per_step"]
            / target_runtime["training_seconds_per_step"]
        )
        inference_ratio = (
            runtime["inference_seconds_per_call"]
            / target_runtime["inference_seconds_per_call"]
        )
        benchmarked.append(
            {
                **candidate,
                "runtime": runtime,
                "training_time_ratio": train_ratio,
                "inference_time_ratio": inference_ratio,
                "compute_distance": abs(math.log(train_ratio)) + abs(math.log(inference_ratio)),
            }
        )
    benchmarked.sort(key=lambda item: item["compute_distance"])

    return {
        "purpose": "RECURRENCE_CONTROL_DESIGN_NOT_PHYSICS",
        "experiment": str(experiment.resolve()),
        "target": target,
        "target_recurrent_steps": recurrent_steps,
        "target_parameters": target_total,
        "target_trainable_parameters": target_trainable,
        "target_patch": _target_patch(target, recurrent_steps),
        "target_runtime": target_runtime,
        "benchmark": {
            "device": device,
            "events": events,
            "warmup": warmup,
            "iterations": iterations,
            "compute_distance": "|log(train_time_ratio)| + |log(inference_time_ratio)|",
        },
        "parameter_matched_control": controls[0],
        "compute_matched_control": benchmarked[0],
        "closest_parameter_controls": controls[:top],
        "benchmarked_controls": benchmarked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("configs/experiments/ttbar2L.yaml"),
    )
    parser.add_argument("--target", choices=("m2", "m3"), default="m3")
    parser.add_argument("--recurrent-steps", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=12)
    parser.add_argument("--feedforward-min", type=int, default=64)
    parser.add_argument("--feedforward-max", type=int, default=1536)
    parser.add_argument("--feedforward-step", type=int, default=32)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--benchmark-candidates", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--events", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = match(
        args.experiment.expanduser().resolve(),
        target=args.target,
        recurrent_steps=args.recurrent_steps,
        max_layers=args.max_layers,
        feedforward_min=args.feedforward_min,
        feedforward_max=args.feedforward_max,
        feedforward_step=args.feedforward_step,
        top=args.top,
        benchmark_candidates=args.benchmark_candidates,
        device=args.device,
        events=args.events,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
