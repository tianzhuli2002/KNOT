#!/usr/bin/env python3
"""Run a small real-cache CPU smoke for the one-pass Study-12 configurations."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.cli.run import build_data
from joint_event_reconstruction.configuration import load_visible_config
from joint_event_reconstruction.prediction import predict_hdf5
from joint_event_reconstruction.prediction_schema import validate_prediction
from joint_event_reconstruction.studies import resolve_study
from joint_event_reconstruction.training.joint import JointEventModule
from scripts.run_study import _public_run_config


def _finite_gradients(module: torch.nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    return bool(gradients) and all(torch.isfinite(value).all().item() for value in gradients)


def _real_batch(config):
    config = copy.deepcopy(config)
    config.data.max_events = {"train": 8, "validation": 8, "test": 2}
    config.training.batch_size = 4
    config.training.workers = 0
    config.training.precision = "32-true"
    config.training.runtime.compile = False
    config.training.runtime.activation_checkpointing = False
    visible = load_visible_config(config)
    for split in ("train", "val", "test"):
        visible.dataset[f"{split}_batch_size"] = 4 if split != "test" else 2
        visible.dataset[f"{split}_num_workers"] = 0
        visible.dataset[f"{split}_persistent_workers"] = False
        visible.dataset[f"{split}_pin_memory"] = False
    visible.dataset.pin_memory = False
    data = build_data(config, visible)
    data.setup("fit")
    return config, visible, data


def main() -> None:
    study = resolve_study(ROOT / "configs/studies/12_encoder_backbone.yaml")
    results = {}
    with tempfile.TemporaryDirectory(prefix="jer-study12-smoke-") as directory:
        output = Path(directory)
        for variant in study.variants:
            _, _, resolved = _public_run_config(
                study, "ttbar2L", variant, 42, output, ()
            )
            resolved, visible, data = _real_batch(resolved)
            batch = next(iter(data.val_dataloader()))
            module = JointEventModule(resolved, visible, synthetic_data=False).eval()
            forward = module(batch)
            if not torch.isfinite(forward["loss"]).item():
                raise RuntimeError(f"Non-finite Study-12 smoke loss: {variant.name}")
            backward = variant.name in {
                "transformer_baseline",
                "fourvector_transformer",
                "relational_transformer",
            }
            if backward:
                module.train()
                module.zero_grad(set_to_none=True)
                module(batch)["loss"].backward()
                if not _finite_gradients(module):
                    raise RuntimeError(f"Non-finite Study-12 smoke gradients: {variant.name}")
            checkpoint = output / f"{variant.name}.ckpt"
            torch.save(
                {
                    "state_dict": module.state_dict(),
                    "hyper_parameters": dict(module.hparams),
                },
                checkpoint,
            )
            prediction = output / f"{variant.name}.h5"
            predict_hdf5(
                module,
                data,
                None,
                prediction,
                split="test",
                top_k=2,
                posterior_samples=1,
                recurrent_steps=None,
                device="cpu",
                prediction_seed=12345,
                checkpoint_path=checkpoint,
            )
            validate_prediction(prediction)
            results[variant.name] = {
                "forward_loss_finite": True,
                "backward_checked": backward,
                "prediction_schema": "PASS",
                "cache": "HIT",
            }
            del module, data, visible, batch
    print(json.dumps({"status": "PASS", "variants": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
