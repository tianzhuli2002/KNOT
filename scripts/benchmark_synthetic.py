#!/usr/bin/env python3
"""File-free throughput/memory benchmark for maintained reconstruction modes.

The benchmark measures model execution on one fixed synthetic in-memory batch. It excludes
HDF5 conversion/loading and is a software profile, not a physics-performance result.
"""

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

from joint_event_reconstruction.configuration import load_experiment, validate_experiment
from joint_event_reconstruction.topology.assignments import AssignmentSpace
from joint_event_reconstruction.training.joint import JointEventModule
from scripts.smoke_synthetic import build_synthetic_batch


MODES = ("visible", "cfm", "dfm", "joint")


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _configure(base, mode: str):
    if mode not in MODES:
        raise ValueError(f"Unknown benchmark mode {mode!r}")
    config = copy.deepcopy(base)
    assignment = config.model.assignment
    config.model.mode = "factorized"
    assignment.reasoning.enabled = False
    assignment.formulation = "discriminative"
    assignment.feedback.enabled = False
    config.model.invisible.enabled = False
    config.model.coupling.invisible_to_assignment.enabled = False
    config.model.compatibility.enabled = False

    if mode == "visible":
        config.training.trainable = ["visible", "assignment"]
        config.training.loss_weights.assignment = 1.0
        config.training.loss_weights.invisible = 0.0
    elif mode == "cfm":
        config.model.invisible.enabled = True
        config.model.invisible.formulation = "flow_matching"
        config.model.invisible.conditioning.assignment = "none"
        config.model.invisible.conditioning.allow_truth_oracle = False
        config.training.trainable = ["visible", "assignment", "invisible"]
        config.training.loss_weights.assignment = 1.0
        config.training.loss_weights.invisible = 1.0
    elif mode == "dfm":
        assignment.formulation = "flow_matching"
        config.training.trainable = ["visible", "assignment"]
        config.training.loss_weights.assignment = 1.0
        config.training.loss_weights.invisible = 0.0
    else:
        config.model.mode = "joint_flow"
        assignment.formulation = "flow_matching"
        config.model.invisible.enabled = True
        config.model.invisible.formulation = "flow_matching"
        config.model.invisible.conditioning.assignment = "none"
        config.model.invisible.conditioning.allow_truth_oracle = False
        config.training.trainable = ["visible", "assignment", "invisible"]
        config.training.loss_weights.assignment = 1.0
        config.training.loss_weights.invisible = 1.0
    config.training.loss_weights.compatibility = 0.0
    validate_experiment(config)
    return config


def _optimizer(module: JointEventModule):
    configured = module.configure_optimizers()
    return configured["optimizer"] if isinstance(configured, dict) else configured


def _train_benchmark(module, batch, *, device, warmup, iterations):
    module.train()
    optimizer = _optimizer(module)

    def one_step():
        optimizer.zero_grad(set_to_none=True)
        loss = module(batch)["loss"]
        loss.backward()
        optimizer.step()

    for _ in range(warmup):
        one_step()
    _sync(device)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(iterations):
        one_step()
    _sync(device)
    elapsed = time.perf_counter() - start
    events = iterations * len(batch["tokens"])
    return {
        "iterations": iterations,
        "batch_size": len(batch["tokens"]),
        "seconds": elapsed,
        "seconds_per_step": elapsed / iterations,
        "events_per_second": events / elapsed,
        "cuda_peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
        "optimizer": type(optimizer).__name__,
        "fused_adamw": bool(isinstance(optimizer, torch.optim.AdamW) and optimizer.defaults.get("fused", False)),
    }


def _inference_call(mode, module, batch, *, posterior_samples, flow_steps, joint_sample_chunk_size):
    visible = module.event_model.encode_visible(batch)
    if mode == "visible":
        return module(batch), 1
    if mode == "cfm":
        output = module(batch)
        generated, nfe = module.event_model.invisible.sample(
            output["visible"], output["invisible_parent_context"],
            samples=posterior_samples, budget=flow_steps,
        )
        return generated, nfe * posterior_samples
    if mode == "dfm":
        sampled = module.event_model.assignment_flow.sample_many(
            visible, batch["structured"], module.event_model.reconstruction,
            samples=posterior_samples, steps=flow_steps,
        )
        return sampled["sampled_assignment_state"], flow_steps * posterior_samples
    sampled = module.event_model.joint_flow.sample(
        visible, batch["structured"], module.event_model.reconstruction,
        module.event_model.invisible, samples=posterior_samples, steps=flow_steps,
        sample_chunk_size=joint_sample_chunk_size,
    )
    return sampled["invisible"], flow_steps * posterior_samples


def _inference_benchmark(
    mode, module, batch, *, device, warmup, iterations,
    posterior_samples, flow_steps, joint_sample_chunk_size,
):
    module.eval()
    equivalent_nfe = None
    with torch.inference_mode():
        for _ in range(warmup):
            _, equivalent_nfe = _inference_call(
                mode, module, batch, posterior_samples=posterior_samples,
                flow_steps=flow_steps, joint_sample_chunk_size=joint_sample_chunk_size,
            )
        _sync(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iterations):
            _, equivalent_nfe = _inference_call(
                mode, module, batch, posterior_samples=posterior_samples,
                flow_steps=flow_steps, joint_sample_chunk_size=joint_sample_chunk_size,
            )
        _sync(device)
        elapsed = time.perf_counter() - start
    events = iterations * len(batch["tokens"])
    return {
        "iterations": iterations,
        "batch_size": len(batch["tokens"]),
        "posterior_samples": posterior_samples if mode != "visible" else 0,
        "flow_steps": flow_steps if mode in {"cfm", "dfm", "joint"} else 0,
        "sample_equivalent_network_evaluations_per_event": int(equivalent_nfe),
        "joint_sample_chunk_size": joint_sample_chunk_size if mode == "joint" else None,
        "seconds": elapsed,
        "seconds_per_call": elapsed / iterations,
        "events_per_second": events / elapsed,
        "cuda_peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
    }


def benchmark(
    experiment: Path,
    *,
    modes: tuple[str, ...],
    device: str,
    events: int,
    warmup: int,
    iterations: int,
    posterior_samples: int,
    flow_steps: int,
    joint_sample_chunk_size: int,
) -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    torch.set_float32_matmul_precision("high")
    base = load_experiment(experiment)
    visible, batch, scaler = build_synthetic_batch(base, events=events, device=device)
    space = AssignmentSpace.from_structured(batch["structured"])
    results = {
        "purpose": "SOFTWARE_PROFILE_ONLY_NOT_PHYSICS",
        "experiment": str(experiment.resolve()),
        "device": device,
        "events": events,
        "assignment_state_count": space.state_count,
        "invisible_role_count": len(scaler.role_names),
        "modes": {},
    }

    for mode in modes:
        config = _configure(base, mode)
        module = JointEventModule(
            config,
            visible,
            invisible_scaler=scaler if bool(config.model.invisible.enabled) else None,
            synthetic_data=True,
        ).to(device)
        results["modes"][mode] = {
            "total_parameters": sum(parameter.numel() for parameter in module.parameters()),
            "trainable_parameters": sum(
                parameter.numel() for parameter in module.parameters() if parameter.requires_grad
            ),
            "training": _train_benchmark(
                module, batch, device=device, warmup=warmup, iterations=iterations
            ),
            "inference": _inference_benchmark(
                mode, module, batch, device=device, warmup=warmup, iterations=iterations,
                posterior_samples=posterior_samples, flow_steps=flow_steps,
                joint_sample_chunk_size=joint_sample_chunk_size,
            ),
        }
        del module
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=Path("configs/experiments/ttbar2L.yaml"))
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--events", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--posterior-samples", type=int, default=8)
    parser.add_argument("--flow-steps", type=int, default=8)
    parser.add_argument("--joint-sample-chunk-size", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("events", "warmup", "iterations", "posterior_samples", "flow_steps", "joint_sample_chunk_size"):
        value = getattr(args, name)
        if name == "warmup":
            if value < 0:
                raise ValueError("--warmup must be non-negative")
        elif value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    results = benchmark(
        args.experiment, modes=tuple(args.modes), device=args.device, events=args.events,
        warmup=args.warmup, iterations=args.iterations,
        posterior_samples=args.posterior_samples, flow_steps=args.flow_steps,
        joint_sample_chunk_size=args.joint_sample_chunk_size,
    )
    payload = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
