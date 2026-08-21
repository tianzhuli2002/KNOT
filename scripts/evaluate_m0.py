#!/usr/bin/env python3
"""Evaluate a frozen M0 checkpoint on a bounded real-data test sample."""

from __future__ import annotations

import argparse
import json

import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.m0.data.module import TransformerDataModule
from joint_event_reconstruction.m0.model import TransformerModel
from joint_event_reconstruction.m0.prediction import _decode_batch_topologies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-events", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.max_events <= 0 or args.batch_size <= 0:
        parser.error("--max-events and --batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["hyper_parameters"])
    config.pop("optimise", None)
    config.dataset.cache.enabled = False
    for split in ("train", "val", "test"):
        config.dataset[f"{split}_batch_size"] = args.batch_size
        config.dataset[f"{split}_num_workers"] = 0
    config.dataset.pin_memory = args.device == "cuda"
    config.dataset.persistent_workers = False

    data = TransformerDataModule(config)
    data.setup("test")
    model = TransformerModel(config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()

    totals = {"events": 0, "decoded_valid": 0, "full_event_correct": 0, "full_reconstructable": 0}
    role_correct: dict[str, float] = {}
    role_active: dict[str, float] = {}
    loss_sum = 0.0
    for batch in data.test_dataloader():
        remaining = args.max_events - totals["events"]
        if remaining <= 0:
            break
        if len(batch["labels"]) > remaining:
            def trim(value):
                if torch.is_tensor(value) and value.shape and value.shape[0] == len(batch["labels"]):
                    return value[:remaining]
                if isinstance(value, dict):
                    return {key: trim(item) for key, item in value.items()}
                return value
            batch = trim(batch)
        batch = model.transfer_batch_to_device(batch, device, 0)
        with torch.inference_mode():
            output = model(
                batch["tokens"], batch["type_ids"], batch["padding_mask"],
                labels=batch["labels"], reco=batch["reco"],
            )
        count = len(batch["labels"])
        totals["events"] += count
        loss_sum += float(output["loss"]) * count
        decoded = _decode_batch_topologies(output, batch, str(config.general.topology))
        totals["decoded_valid"] += sum(bool(item["reco_valid"]) for item in decoded)
        totals["full_event_correct"] += sum(bool(item["complete_topology_truth_correct"]) for item in decoded)
        required_roles = {"Whad", "thad", "tlep"}
        if str(config.general.topology) == "ttH":
            required_roles.add("H")
        reconstructable = torch.ones(count, dtype=torch.bool, device=device)
        for pool in batch["reco"]["pools"].values():
            truth = pool["truth_class"]
            for class_index, role in enumerate(pool["class_names"][1:], start=1):
                if role in required_roles:
                    reconstructable &= (truth == class_index).any(dim=1)
        totals["full_reconstructable"] += int(reconstructable.sum())
        for metrics in output["reconstruction_metrics"].values():
            for key, value in metrics.items():
                if not key.endswith("_top1_accuracy"):
                    continue
                role = key.removesuffix("_top1_accuracy")
                active = float(metrics[f"{role}_class_count"])
                role_correct[role] = role_correct.get(role, 0.0) + float(value) * active
                role_active[role] = role_active.get(role, 0.0) + active

    result = {
        **totals,
        "mean_loss": loss_sum / totals["events"],
        "decode_valid_fraction": totals["decoded_valid"] / totals["events"],
        "full_event_efficiency": totals["full_event_correct"] / totals["events"],
        "full_event_efficiency_reconstructable": (
            totals["full_event_correct"] / totals["full_reconstructable"]
            if totals["full_reconstructable"] else None
        ),
        "role_local_top1": {
            role: role_correct[role] / role_active[role] for role in sorted(role_active) if role_active[role]
        },
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "topology": str(config.general.topology),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
