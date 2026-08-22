#!/usr/bin/env python3
"""Bounded real-data metrics for a trained M1 structured checkpoint."""

from __future__ import annotations

import argparse
import json
import time

from omegaconf import OmegaConf
import torch

from joint_event_reconstruction.data.structured import StructuredDataModule
from joint_event_reconstruction.topology.candidates import CandidateRecord
from joint_event_reconstruction.topology.decoder import ScoredCandidate, decode_topk
from joint_event_reconstruction.training.structured import StructuredEventModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-events", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    experiment = OmegaConf.load(args.experiment)
    visible = OmegaConf.load(experiment.visible_config)
    for split in ("train", "val", "test"):
        visible.dataset[f"{split}_batch_size"] = args.batch_size
        visible.dataset[f"{split}_num_workers"] = 0
    visible.dataset.persistent_workers = False
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    data = StructuredDataModule(visible, experiment.topology_config, tensor_cache=experiment.data.tensor_cache)
    data.setup("test")
    module = StructuredEventModule(experiment, visible)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    module.validate_checkpoint_metadata(checkpoint)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device(args.device)
    module.to(device).eval()
    counts = {
        node: {"reconstructable": 0, "correct": 0, "truth_probability": 0.0, "entropy": 0.0}
        for node in module.spec.dag_order
    }
    events = full_reconstructable = full_top1 = full_top3 = 0
    started = time.perf_counter()
    for batch in data.test_dataloader():
        if events >= args.max_events:
            break
        remaining = args.max_events - events
        if len(batch["tokens"]) > remaining:
            batch_size = len(batch["tokens"])
            def trim(value):
                if torch.is_tensor(value) and value.shape and value.shape[0] == batch_size:
                    return value[:remaining]
                if isinstance(value, dict):
                    return {key: trim(item) for key, item in value.items()}
                return value
            batch = trim(batch)
        batch = module.transfer_batch_to_device(batch, device, 0)
        with torch.inference_mode():
            output = module(batch)
        size = len(batch["tokens"])
        root_active = torch.ones(size, dtype=torch.bool, device=device)
        for node in module.spec.dag_order:
            pool = batch["structured"]["pools"][node]
            probability = output["assignment_probabilities"][node]
            prediction = probability.argmax(dim=1) - 1
            active = pool["reconstructable"]
            truth_column = (pool["truth_index"] + 1).clamp_min(0)
            truth_probability = probability.gather(1, truth_column.unsqueeze(1)).squeeze(1)
            entropy = -(probability * probability.clamp_min(1e-12).log()).sum(dim=1)
            counts[node]["reconstructable"] += int(active.sum())
            counts[node]["correct"] += int(((prediction == pool["truth_index"]) & active).sum())
            counts[node]["truth_probability"] += float(truth_probability[active].sum())
            counts[node]["entropy"] += float(entropy[active].sum())
            if node in module.spec.roots:
                root_active &= active
        full_reconstructable += int(root_active.sum())

        for event in range(size):
            if not root_active[event]:
                continue
            scored = {}
            for root in module.spec.roots:
                pool = batch["structured"]["pools"][root]
                values = []
                for index in pool["candidate_mask"][event].nonzero(as_tuple=False).flatten().tolist():
                    owned = pool["leaf_ownership"][index].nonzero(as_tuple=False).flatten().tolist()
                    record = CandidateRecord(
                        root, index, (), (), frozenset(("TOKEN", int(token)) for token in owned),
                        bool(index == int(pool["truth_index"][event])),
                    )
                    probability_value = output["assignment_probabilities"][root][event, index + 1]
                    values.append(ScoredCandidate(record, float(probability_value)))
                scored[root] = tuple(values)
            assignments = decode_topk(module.spec, scored, k=3)
            full_top1 += int(bool(assignments and assignments[0].truth))
            full_top3 += int(any(item.truth for item in assignments))
        events += size

    elapsed = time.perf_counter() - started
    result = {
        "events": events,
        "full_reconstructable": full_reconstructable,
        "full_top1": full_top1 / full_reconstructable if full_reconstructable else None,
        "full_top3": full_top3 / full_reconstructable if full_reconstructable else None,
        "events_per_second": events / elapsed,
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        "nodes": {
            node: {
                "reconstructable": value["reconstructable"],
                "top1": value["correct"] / value["reconstructable"] if value["reconstructable"] else None,
                "mean_truth_probability": value["truth_probability"] / value["reconstructable"] if value["reconstructable"] else None,
                "mean_entropy": value["entropy"] / value["reconstructable"] if value["reconstructable"] else None,
            } for node, value in counts.items()
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
