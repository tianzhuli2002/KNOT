#!/usr/bin/env python3
"""Validate shared-encoder gradient policy and optional-head targets on real data."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.training.joint import JointEventModule


def _clone(config):
    return OmegaConf.create(OmegaConf.to_container(config, resolve=False))


def _norm(module) -> float:
    values = [parameter.grad.detach().square().sum() for parameter in module.parameters() if parameter.grad is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def _finite_gradients(module) -> bool:
    return all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def _make_module(config):
    visible = load_visible_config(config)
    data = build_data(config, visible)
    data.setup("fit")
    module = JointEventModule(
        config,
        visible,
        invisible_scaler=data.invisible_scaler,
        point_scaler=data.point_scaler,
    )
    batch = next(iter(data.train_dataloader()))
    return module, data, batch


def _policy_report(base, policy: str) -> dict:
    config = _clone(base)
    config.training.shared_encoder_gradients = policy
    module, _, batch = _make_module(config)
    module.train()
    report = {}

    module.zero_grad(set_to_none=True)
    output = module(batch)
    output["assignment_loss"].backward()
    report["assignment_loss"] = {
        "visible": _norm(module.event_model.visible),
        "assignment": _norm(module.event_model.reconstruction),
        "invisible": _norm(module.event_model.invisible),
        "finite": _finite_gradients(module),
    }

    module.zero_grad(set_to_none=True)
    output = module(batch)
    output["invisible_loss"].backward()
    report["cfm_before_update"] = {
        "visible": _norm(module.event_model.visible),
        "assignment": _norm(module.event_model.reconstruction),
        "invisible": _norm(module.event_model.invisible),
        "finite": _finite_gradients(module),
    }

    optimizer = torch.optim.Adam((parameter for parameter in module.parameters() if parameter.requires_grad), lr=1.0e-4)
    module.zero_grad(set_to_none=True)
    module(batch)["loss"].backward()
    optimizer.step()
    module.zero_grad(set_to_none=True)
    output = module(batch)
    output["invisible_loss"].backward()
    report["cfm_after_combined_update"] = {
        "visible": _norm(module.event_model.visible),
        "assignment": _norm(module.event_model.reconstruction),
        "invisible": _norm(module.event_model.invisible),
        "finite": _finite_gradients(module),
    }
    report["policy"] = policy
    return report


def _optional_report(base, kind: str, parent_target: Path) -> dict:
    config = _clone(base)
    config.training.trainable = list(config.training.trainable) + [kind]
    config.training.loss_weights[kind] = 1.0
    if kind == "point":
        config.model.heads.point_invisible.enabled = True
    else:
        roles = ["top_e", "top_mu"]
        config.model.heads.parent_four_vector.enabled = True
        config.model.heads.parent_four_vector.roles = roles
        config.model.heads.parent_four_vector.target_path = str(parent_target)
        config.data.parent.path = str(parent_target)
        config.data.parent.roles = roles
    module, _, batch = _make_module(config)
    module.train()
    module.zero_grad(set_to_none=True)
    output = module(batch)
    loss_key = "point_loss" if kind == "point" else "parent_loss"
    output[loss_key].backward()
    head = module.event_model.point_head if kind == "point" else module.event_model.parent_head
    return {
        "loss": float(output[loss_key].detach()),
        "output_shape": list(output["point_momenta_scaled" if kind == "point" else "parent_four_vectors"].shape),
        "visible_gradient": _norm(module.event_model.visible),
        "head_gradient": _norm(head),
        "finite": _finite_gradients(module),
        "parent_target_semantics": (
            "hybrid_visible_reconstructed_plus_truth_invisible_v1" if kind == "parent" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20260923)
    base = load_experiment(args.config.expanduser().resolve())
    # The diagnostic encoder keeps this gate fast; production configs remain L-GATr.
    base.model.visible.type = "transformer"
    base.model.visible.layers = 2
    base.model.visible.d_feedforward = 384
    report = {
        "data": {"config": str(args.config.resolve()), "real_data": True, "batch_size": int(base.training.batch_size)},
        "gradient_policy": {policy: _policy_report(base, policy) for policy in ("both", "assignment_only")},
        "optional_heads": {
            "point": _optional_report(base, "point", args.parent_target.resolve()),
            "parent": _optional_report(base, "parent", args.parent_target.resolve()),
        },
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
