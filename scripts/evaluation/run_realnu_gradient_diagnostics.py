#!/usr/bin/env python3
"""Measure bounded assignment/neutrino/parent gradient interaction on real train events."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset


PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
SOURCE_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = PROJECT_ROOT / (
    "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/"
    "realnu_top_antitop_20260927_release2"
)
OUTPUT_PATH = PRODUCTION_ROOT / "evaluation_v3_diagnostics/query_diagnostics/gradient_interaction.json"
MODES = ("J0", "J_A", "J_AN")
EVENT_COUNT = 32


def _gradient_vector(gradients, parameters) -> torch.Tensor:
    flattened = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        if gradient is None:
            flattened.append(torch.zeros(parameter.numel(), dtype=torch.float64))
        else:
            flattened.append(gradient.detach().reshape(-1).to(device="cpu", dtype=torch.float64))
    return torch.cat(flattened) if flattened else torch.empty(0, dtype=torch.float64)


def _norm_cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = float(left.norm() * right.norm())
    if denominator == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _mode(mode: str) -> dict:
    os.environ["JER_PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["JER_REALNU_PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["JER_REALNU_SOURCE_ROOT"] = str(SOURCE_ROOT)
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    sys.path.insert(0, str(SOURCE_ROOT / "scripts/production"))
    from joint_event_reconstruction.cli.run import build_data, load_trained, load_visible_config
    import run_realnu_production as runner

    config, _, _ = runner.load_and_validate_config(
        mode, runner.MODE_CONFIGS[mode], PRODUCTION_ROOT / mode
    )
    visible_config = load_visible_config(config)
    data = build_data(config, visible_config, include_truth=True, device="cpu")
    data.setup("fit")
    if data.train_dataset is None or data.structured_collator.partonic_store is None:
        raise RuntimeError(f"{mode}: gradient diagnostic lacks frozen training truth support")
    if len(data.train_dataset) < EVENT_COUNT:
        raise ValueError(f"{mode}: frozen training split is smaller than requested fixture")
    data.train_dataset = Subset(data.train_dataset, range(EVENT_COUNT))
    batch = next(iter(data.train_dataloader()))
    ids = batch["event_index"].cpu().numpy().astype("<i8", copy=False)
    module = load_trained(
        config,
        visible_config,
        data,
        str(Path(json.loads((PRODUCTION_ROOT / mode / "production_training.json").read_text())["checkpoint"]).resolve()),
    ).cpu()
    module.train()
    torch.manual_seed(56001)
    output = module(batch, compute_objectives=True)
    losses = {
        "assignment": output.get("assignment_loss"),
        "neutrino_cfm": output.get("invisible_loss"),
        "parent_cfm": output.get("partonic_cfm_loss"),
    }
    if any(value is None or not torch.isfinite(value) for value in losses.values()):
        raise RuntimeError(f"{mode}: not all three task losses were finite and present")

    module_parameters = dict(module.named_parameters())
    groups = {
        "visible_encoder": [
            (name, parameter) for name, parameter in module_parameters.items()
            if name.startswith("event_model.visible.") and parameter.requires_grad
        ],
        "query_block": [
            (name, parameter) for name, parameter in module_parameters.items()
            if name.startswith("event_model.topology_queries.") and parameter.requires_grad
        ],
    }
    result = {
        "event_count": int(len(ids)),
        "training_EVENT_ID_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "sealed_test_used": False,
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "groups": {},
    }
    for group_name, named_parameters in groups.items():
        names, parameters = zip(*named_parameters, strict=True)
        gradients_by_loss = {}
        loss_records = {}
        for loss_name, loss in losses.items():
            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=True,
                allow_unused=True,
            )
            vector = _gradient_vector(gradients, parameters)
            gradients_by_loss[loss_name] = vector
            loss_records[loss_name] = {
                "gradient_l2_norm": float(vector.norm()),
                "nonzero_parameter_tensors": int(sum(g is not None and bool(torch.count_nonzero(g)) for g in gradients)),
                "parameter_tensor_count": len(parameters),
            }
        cosine = {}
        loss_names = list(gradients_by_loss)
        for left_index, left_name in enumerate(loss_names):
            for right_name in loss_names[left_index + 1:]:
                cosine[f"{left_name}__{right_name}"] = _norm_cosine(
                    gradients_by_loss[left_name], gradients_by_loss[right_name]
                )
        result["groups"][group_name] = {
            "parameters": len(parameters),
            "parameter_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
            "losses": loss_records,
            "pairwise_cosine": cosine,
        }
    record = json.loads((PRODUCTION_ROOT / mode / "production_training.json").read_text())
    result["checkpoint"] = record["checkpoint"]
    result["checkpoint_sha256"] = record["checkpoint_sha256"]
    return result


def run(output_path: Path = OUTPUT_PATH) -> dict:
    torch.set_num_threads(4)
    records = {mode: _mode(mode) for mode in MODES}
    event_hashes = {value["training_EVENT_ID_sha256"] for value in records.values()}
    if len(event_hashes) != 1:
        raise ValueError("Representative train EVENT_IDs differ across J0/J_A/J_AN")
    result = {
        "status": "passed",
        "split": "frozen train split",
        "events_per_mode": EVENT_COUNT,
        "training_EVENT_ID_sha256": next(iter(event_hashes)),
        "parent_roles": ["top", "antitop"],
        "neutrino_roles": ["top.nu", "antitop.nu"],
        "sealed_test_used": False,
        "method": "autograd.grad on separate finite assignment, invisible CFM, and parent CFM objectives from one fixed representative real batch; flattened gradients only over shared visible encoder and query block",
        "models": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    print(json.dumps(run(args.output.expanduser().resolve()), indent=2))


if __name__ == "__main__":
    main()
