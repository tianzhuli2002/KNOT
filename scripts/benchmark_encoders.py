#!/usr/bin/env python3
"""Engineering-only comparison of the maintained visible encoder backbones."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from joint_event_reconstruction.configuration import (
    load_experiment,
    validate_experiment,
)
from joint_event_reconstruction.training.joint import JointEventModule
from scripts.smoke_synthetic import build_synthetic_batch


ENCODERS = ("transformer", "physics_transformer", "lgatr", "lgatr_slim")


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _configure(experiment: Path, encoder: str):
    if encoder not in ENCODERS:
        raise ValueError(f"Unknown encoder {encoder!r}; choose from {ENCODERS}")
    config = load_experiment(experiment, [f"model.encoder.type={encoder}"])
    config = copy.deepcopy(config)
    # This is a visible-backbone profile; downstream assignment capacity is
    # intentionally kept identical for every row.
    if encoder in {"transformer", "physics_transformer"}:
        config.model.visible.layers = 1
    else:
        config.model.visible.blocks = 1
    config.model.assignment.formulation = "discriminative"
    config.model.assignment.feedback.enabled = False
    config.model.invisible.enabled = False
    config.model.coupling.invisible_to_assignment.enabled = False
    config.model.compatibility.enabled = False
    config.training.trainable = ["visible", "assignment"]
    config.training.loss_weights.assignment = 1.0
    config.training.loss_weights.invisible = 0.0
    config.training.loss_weights.compatibility = 0.0
    config.training.runtime.compile = False
    config.training.runtime.activation_checkpointing = False
    validate_experiment(config)
    return config


def _timed(function, *, device: str) -> float:
    _sync(device)
    start = time.perf_counter()
    function()
    _sync(device)
    return time.perf_counter() - start


def benchmark(
    experiment: Path,
    *,
    encoders: tuple[str, ...],
    device: str,
    events: int,
    warmup: int,
    steps: int,
    compile_model: bool,
) -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    base = load_experiment(experiment)
    visible_config, batch, _ = build_synthetic_batch(base, events=events, device=device)
    result = {
        "purpose": "ENGINEERING_PROFILE_ONLY_NOT_SCIENTIFIC_COMPARISON",
        "experiment": str(experiment.resolve()),
        "device": device,
        "precision": "32-true",
        "compile": bool(compile_model),
        "warmup_steps": warmup,
        "measured_steps": steps,
        "encoders": {},
    }
    for encoder in encoders:
        config = _configure(experiment, encoder)
        module = JointEventModule(config, visible_config, synthetic_data=True).to(device)
        model = module
        if compile_model:
            model = torch.compile(module)
            # Compilation is deliberately reported separately from steady-state
            # timings; the first call below pays the trace cost.
        compile_seconds = None
        if compile_model:
            compile_seconds = _timed(lambda: model(batch), device=device)
        optimiser = torch.optim.AdamW(
            [parameter for parameter in module.parameters() if parameter.requires_grad],
            lr=1.0e-4,
            fused=False,
        )

        def forward():
            model(batch)

        def train_step():
            optimiser.zero_grad(set_to_none=True)
            model(batch)["loss"].backward()
            optimiser.step()

        for _ in range(warmup):
            train_step()
        _sync(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        forward_seconds = sum(
            _timed(forward, device=device) for _ in range(steps)
        ) / steps
        train_seconds = sum(
            _timed(train_step, device=device) for _ in range(steps)
        ) / steps
        result["encoders"][encoder] = {
            "encoder_parameters": sum(parameter.numel() for parameter in module.event_model.visible.parameters()),
            "total_model_parameters": sum(parameter.numel() for parameter in module.parameters()),
            "forward_ms": 1000.0 * forward_seconds,
            "training_step_ms": 1000.0 * train_seconds,
            "compile_startup_ms": None if compile_seconds is None else 1000.0 * compile_seconds,
            "cuda_peak_memory_mib": (
                torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path("configs/experiments/ttbar2L.yaml"))
    parser.add_argument("--encoders", default=",".join(ENCODERS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--events", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.events <= 0 or args.warmup < 0 or args.steps <= 0:
        raise ValueError("events and steps must be positive; warmup must be non-negative")
    encoders = tuple(value.strip() for value in args.encoders.split(",") if value.strip())
    result = benchmark(
        args.experiment.expanduser().resolve(),
        encoders=encoders,
        device=args.device,
        events=args.events,
        warmup=args.warmup,
        steps=args.steps,
        compile_model=args.compile_model,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered)


if __name__ == "__main__":
    main()
