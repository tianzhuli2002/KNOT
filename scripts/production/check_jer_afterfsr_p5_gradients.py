#!/usr/bin/env python3
"""Check one configured P5 training batch for finite loss and gradients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.fabric.utilities.apply_func import move_data_to_device

from joint_event_reconstruction.cli.run import build_data, load_trained
from joint_event_reconstruction.configuration import (
    load_experiment,
    load_visible_config,
)


def check(config_path: Path, checkpoint_path: Path, overrides=()):
    torch.set_float32_matmul_precision("high")
    config = load_experiment(config_path, overrides)
    if str(config.training.precision) != "16-mixed":
        raise ValueError(f"P5 smoke expected 16-mixed precision, found {config.training.precision}")
    L.seed_everything(int(config.experiment.seed), workers=True)
    visible = load_visible_config(config)
    data = build_data(config, visible)
    data.setup("fit")
    module = load_trained(config, visible, data, str(checkpoint_path))
    device = torch.device("cuda")
    module.to(device)
    module.train()
    batch = move_data_to_device(next(iter(data.train_dataloader())), device)

    seen_gradients = set()
    nonfinite_gradients = []
    hooks = []

    def inspect_gradient(name):
        def hook(gradient):
            values = gradient.coalesce().values() if gradient.is_sparse else gradient
            if not torch.isfinite(values).all():
                nonfinite_gradients.append(name)
            seen_gradients.add(name)
            return gradient

        return hook

    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            hooks.append(parameter.register_hook(inspect_gradient(name)))

    optimizer_config = module.configure_optimizers()
    optimizer = (
        optimizer_config["optimizer"]
        if isinstance(optimizer_config, dict)
        else optimizer_config
    )
    optimizer.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = module(batch)
        loss = output["loss"]
    if loss.ndim != 0 or not torch.isfinite(loss).all():
        raise FloatingPointError(f"P5 smoke training loss is non-finite or non-scalar: {loss}")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("P5 smoke batch produced no trainable gradients")
    if nonfinite_gradients:
        raise FloatingPointError(f"P5 smoke produced non-finite gradients: {nonfinite_gradients[:10]}")
    for gradient in gradients:
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        if not torch.isfinite(values).all():
            raise FloatingPointError("P5 smoke produced a non-finite gradient after AMP unscale")
    if not seen_gradients:
        raise RuntimeError("Gradient hooks did not observe any trainable gradient")

    for hook in hooks:
        hook.remove()
    return {
        "status": "finite_gradient_smoke_passed",
        "loss": float(loss.detach().cpu()),
        "finite_gradient_tensors": len(gradients),
        "trainable_parameters_with_gradient": len(seen_gradients),
        "trainable_parameter_count": sum(parameter.requires_grad for parameter in module.parameters()),
        "training_batch_events": len(batch["tokens"]),
        "precision": str(config.training.precision),
        "checkpoint": str(checkpoint_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The P5 gradient smoke requires the allocated CUDA device")
    print(json.dumps(check(args.config, args.checkpoint, args.overrides), indent=2), flush=True)


if __name__ == "__main__":
    main()
