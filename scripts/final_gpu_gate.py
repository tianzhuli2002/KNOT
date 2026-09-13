#!/usr/bin/env python3
"""Run the consolidated final JER engineering gate on one CUDA worker.

This is an engineering gate, not a scientific comparison.  It deliberately
uses one deterministic real-cache batch for all visible backbones and keeps
the reconstruction checks small enough for a short Slurm allocation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from joint_event_reconstruction.cli.run import build_data
from joint_event_reconstruction.configuration import (
    load_experiment,
    load_visible_config,
    validate_experiment,
)
from joint_event_reconstruction.evaluation import evaluate_predictions
from joint_event_reconstruction.prediction import predict_hdf5
from joint_event_reconstruction.prediction_schema import validate_prediction
from joint_event_reconstruction.studies import load_study
from joint_event_reconstruction.training.joint import JointEventModule
from scripts.run_study import _public_run_config
from scripts.smoke_flows import configure as configure_flow


VISIBLE_VARIANTS = (
    "transformer_baseline",
    "fourvector_transformer",
    "relational_transformer",
    "lgatr",
    "lgatr_slim",
)
CONTROL_VARIANTS = (
    "m2",
    "m2_compute_control",
    "m3",
    "m3_compute_control",
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _move(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    return value


def _subset(value, size: int, batch_size: int):
    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == batch_size:
            return value[:size]
        return value
    if isinstance(value, dict):
        return {key: _subset(item, size, batch_size) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_subset(item, size, batch_size) for item in value)
    if isinstance(value, list):
        return [_subset(item, size, batch_size) for item in value]
    return value


def _finite_grads(module: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(gradients) and all(torch.isfinite(value).all().item() for value in gradients)


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _train_step(module, batch, optimiser, *, recurrent_steps: int | None = None) -> float:
    optimiser.zero_grad(set_to_none=True)
    output = module(batch, recurrent_steps=recurrent_steps)
    loss = output["loss"]
    if not torch.isfinite(loss).item():
        raise RuntimeError("non-finite loss in GPU gate")
    loss.backward()
    if not _finite_grads(module):
        raise RuntimeError("missing or non-finite gradients in GPU gate")
    optimiser.step()
    return float(loss.detach().cpu())


def _timed_steps(
    module,
    batch,
    *,
    warmup: int,
    measured: int,
    device: torch.device,
    recurrent_steps: int | None = None,
):
    optimiser = torch.optim.AdamW(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=1.0e-4,
        fused=False,
    )
    module.train()
    for _ in range(warmup):
        _train_step(module, batch, optimiser, recurrent_steps=recurrent_steps)
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    values = []
    losses = []
    for _ in range(measured):
        _sync(device)
        start = time.perf_counter()
        losses.append(
            _train_step(module, batch, optimiser, recurrent_steps=recurrent_steps)
        )
        _sync(device)
        values.append(time.perf_counter() - start)
    return {
        "mean_training_step_ms": 1000.0 * sum(values) / len(values),
        "step_times_ms": [1000.0 * value for value in values],
        "losses": losses,
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None
        ),
    }


def _config_limits(config, batch_size: int):
    config.data.max_events = {"train": batch_size, "val": batch_size, "test": 2}
    config.training.batch_size = batch_size
    config.training.workers = 0
    config.training.precision = "32-true"
    config.training.runtime.compile = False
    config.training.runtime.activation_checkpointing = False
    return config


def _build_real_data(config, batch_size: int):
    config = _config_limits(config, batch_size)
    visible = load_visible_config(config)
    for split in ("train", "val", "test"):
        visible.dataset[f"{split}_batch_size"] = batch_size if split != "test" else 2
        visible.dataset[f"{split}_num_workers"] = 0
        visible.dataset[f"{split}_persistent_workers"] = False
    data = build_data(config, visible)
    data.setup("fit")
    return config, visible, data


def _flow_config(base, mode: str, *, coordinate: str | None = None):
    config = configure_flow(base, mode)
    if coordinate is not None:
        config.model.invisible.coordinates.type = coordinate
        config.model.invisible.coordinates.basis_convention = (
            "helmert_orthonormal_v1"
            if coordinate in {"met_residual", "met_exact"}
            else None
        )
    validate_experiment(config)
    return config


def _cache_metadata(config) -> dict:
    cache = Path(config.data.tensor_cache).expanduser().resolve()
    manifest = cache.with_suffix(".json")
    if not cache.is_file() or not manifest.is_file():
        raise RuntimeError(f"GPU gate requires a prepared cache and manifest: {cache}")
    metadata = json.loads(manifest.read_text())
    return {
        "path": str(cache),
        "size_bytes": cache.stat().st_size,
        "manifest": str(manifest),
        "cache_version": metadata.get("cache_version"),
        "event_count": metadata.get("event_count"),
        "physics_vector_representation": metadata.get("physics_vector_representation"),
        "state": "HIT",
    }


def _check_git(expected_sha: str) -> str:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if actual != expected_sha:
        raise RuntimeError(f"GPU gate SHA mismatch: expected {expected_sha}, observed {actual}")
    if status:
        raise RuntimeError(f"GPU gate requires a clean worktree; observed {status!r}")
    return actual


def _visible_configs(experiment: Path, output: Path):
    study = load_study(ROOT / "configs/studies/12_encoder_backbone.yaml")
    result = {}
    for name in VISIBLE_VARIANTS:
        _, _, resolved = _public_run_config(
            study, "ttbar2L", study.variant_map[name], 42, output, ()
        )
        _config_limits(resolved, 128)
        result[name] = resolved
    return result


def _control_configs(output: Path):
    study = load_study(ROOT / "configs/studies/01_visible_recurrence.yaml")
    result = {}
    for name in CONTROL_VARIANTS:
        _, _, resolved = _public_run_config(
            study, "ttbar2L", study.variant_map[name], 42, output, ()
        )
        _config_limits(resolved, 128)
        result[name] = resolved
    return result


def _record_visible(configs, visible, batch, device):
    result = {}
    for name in VISIBLE_VARIANTS:
        module = JointEventModule(configs[name], visible, synthetic_data=False).to(device)
        result[name] = {
            "parameters": _parameter_count(module),
            "precision": "32-true",
            **_timed_steps(module, batch, warmup=1, measured=1, device=device),
        }
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def _record_controls(configs, visible, batch, device):
    result = {}
    for name in CONTROL_VARIANTS:
        module = JointEventModule(configs[name], visible, synthetic_data=False).to(device)
        result[name] = {
            "parameters": _parameter_count(module),
            "precision": "32-true",
            "recurrent_steps": 4 if name in {"m2", "m3"} else None,
            **_timed_steps(
                module,
                batch,
                warmup=2,
                measured=5,
                device=device,
                recurrent_steps=4 if name in {"m2", "m3"} else None,
            ),
        }
        del module
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["ratios"] = {
        "m2_compute_control_over_m2": (
            result["m2_compute_control"]["mean_training_step_ms"]
            / result["m2"]["mean_training_step_ms"]
        ),
        "m3_compute_control_over_m3": (
            result["m3_compute_control"]["mean_training_step_ms"]
            / result["m3"]["mean_training_step_ms"]
        ),
    }
    for name, ratio in result["ratios"].items():
        if not 0.95 <= ratio <= 1.05:
            raise RuntimeError(
                f"{name}={ratio:.4f} is outside the required 0.95-1.05 GPU control range"
            )
    result["ratio_tolerance"] = {"lower": 0.95, "upper": 1.05, "status": "PASS"}
    return result


def _run_reconstruction_paths(base, data_cart, data_met, visible_cart, visible_met, batch_cart, batch_met, device):
    configurations = {
        "cfm_cartesian": (_flow_config(base, "cfm_no_assignment"), data_cart, visible_cart, batch_cart),
        "cfm_met_residual": (_flow_config(base, "cfm_no_assignment", coordinate="met_residual"), data_met, visible_met, batch_met),
        "cfm_conditional_ot": (_flow_config(base, "cfm_no_assignment"), data_cart, visible_cart, batch_cart),
        "jointflow_met_residual": (_flow_config(base, "joint_flow", coordinate="met_residual"), data_met, visible_met, batch_met),
        "cfm_truth_oracle": (_flow_config(base, "cfm_no_assignment"), data_cart, visible_cart, batch_cart),
    }
    configurations["cfm_conditional_ot"][0].model.invisible.flow_matching.coupling = {
        "method": "conditional_ot", "condition_weight": 1.0
    }
    configurations["cfm_truth_oracle"][0].model.invisible.conditioning.assignment = "truth"
    configurations["cfm_truth_oracle"][0].model.invisible.conditioning.allow_truth_oracle = True
    for config, _, _, _ in configurations.values():
        validate_experiment(config)

    result = {}
    modules = {}
    for name, (config, data, visible, batch) in configurations.items():
        small_batch = _subset(batch, min(8, len(batch["tokens"])), len(batch["tokens"]))
        scaler = data.invisible_scaler
        module = JointEventModule(
            config, visible, invisible_scaler=scaler, synthetic_data=False
        ).to(device)
        result[name] = {
            "parameters": _parameter_count(module),
            "precision": "32-true",
            **_timed_steps(module, small_batch, warmup=0, measured=1, device=device),
        }
        modules[name] = (module, config, data, visible)
    return result, modules


def _checkpoint_prediction(modules, output: Path, device: torch.device):
    module, config, data, visible = modules["cfm_met_residual"]
    checkpoint_path = output / "checkpoint-roundtrip.ckpt"
    checkpoint = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "invisible_scaler": module.scaler_metadata,
    }
    torch.save(checkpoint, checkpoint_path)
    fresh = JointEventModule(
        config,
        visible,
        invisible_scaler=data.invisible_scaler,
        synthetic_data=False,
    ).to(device)
    fresh.validate_checkpoint_metadata(checkpoint)
    fresh.load_state_dict(checkpoint["state_dict"], strict=True)
    prediction_path = output / "prediction.h5"
    predict_hdf5(
        fresh,
        data,
        data.invisible_scaler,
        prediction_path,
        split="test",
        top_k=2,
        posterior_samples=2,
        recurrent_steps=None,
        device=str(device),
        prediction_seed=12345,
        checkpoint_path=checkpoint_path,
        compatibility_weight=1.0,
    )
    validate_prediction(prediction_path)
    evaluation = evaluate_predictions(prediction_path)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_reload": "PASS",
        "prediction": str(prediction_path),
        "prediction_schema": "PASS",
        "evaluation": "PASS",
        "evaluation_events": evaluation["events"],
        "prediction_architecture_signature": evaluation["architecture_signature"],
    }


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Final GPU gate requires CUDA")
    device = torch.device("cuda")
    actual_sha = _check_git(args.expected_sha)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = load_experiment(args.experiment.expanduser().resolve())
    cache = _cache_metadata(base)

    base_flow_cart = _flow_config(base, "cfm_no_assignment")
    base_flow_met = _flow_config(base, "joint_flow", coordinate="met_residual")
    _, visible_cart, data_cart = _build_real_data(base_flow_cart, args.batch_size)
    _, visible_met, data_met = _build_real_data(base_flow_met, args.batch_size)
    batch_cart_cpu = next(iter(data_cart.val_dataloader()))
    batch_met_cpu = next(iter(data_met.val_dataloader()))
    ids_cart = batch_cart_cpu["event_index"].clone()
    ids_met = batch_met_cpu["event_index"].clone()
    if not torch.equal(ids_cart, ids_met):
        raise RuntimeError("Cartesian and MET-residual gate data selected different EVENT_IDs")
    batch_cart = _move(batch_cart_cpu, device)
    batch_met = _move(batch_met_cpu, device)

    visible_configs = _visible_configs(args.experiment, output)
    visible_result = _record_visible(visible_configs, visible_met, batch_met, device)
    controls = _control_configs(output)
    control_result = _record_controls(controls, visible_met, batch_met, device)
    reconstruction, modules = _run_reconstruction_paths(
        base, data_cart, data_met, visible_cart, visible_met, batch_cart, batch_met, device
    )
    checkpoint = _checkpoint_prediction(modules, output, device)
    summary = {
        "status": "PASS",
        "purpose": "ENGINEERING_GATE_ONLY_NOT_SCIENTIFIC_COMPARISON",
        "git_sha": actual_sha,
        "gpu": torch.cuda.get_device_name(device),
        "precision": "32-true",
        "cache": cache,
        "batch_size": args.batch_size,
        "selected_event_ids": [int(value) for value in ids_cart.tolist()],
        "visible_encoders": visible_result,
        "compute_controls": control_result,
        "reconstruction_paths": reconstruction,
        "checkpoint_prediction": checkpoint,
        "pass_markers": {
            "cache_hit": True,
            "five_visible_encoders": True,
            "compute_controls": True,
            "representative_reconstruction": True,
            "checkpoint_reload": True,
            "prediction_schema_evaluation": True,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=ROOT / "configs/experiments/ttbar2L.yaml")
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    try:
        run(args)
    except Exception as exc:
        output = args.output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "FAIL",
            "expected_sha": args.expected_sha,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
        (output / "summary.json").write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
