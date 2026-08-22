#!/usr/bin/env python3
"""Real-data forward/backward and constrained-decode smoke for M1."""

from __future__ import annotations

import argparse
import json

from omegaconf import OmegaConf
import torch

from joint_event_reconstruction.data.structured import StructuredDataModule
from joint_event_reconstruction.topology.candidates import CandidateRecord
from joint_event_reconstruction.topology.decoder import ScoredCandidate, decode_topk
from joint_event_reconstruction.training.structured import StructuredEventModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--events", type=int, default=32)
    args = parser.parse_args()
    experiment = OmegaConf.load(args.experiment)
    visible = OmegaConf.load(experiment.visible_config)
    visible.dataset.cache.enabled = False
    for split in ("train", "val", "test"):
        visible.dataset[f"{split}_batch_size"] = args.events
        visible.dataset[f"{split}_num_workers"] = 0
    visible.dataset.pin_memory = False
    visible.dataset.persistent_workers = False
    data = StructuredDataModule(visible, experiment.topology_config)
    data.setup("validate")
    batch = next(iter(data.val_dataloader()))
    module = StructuredEventModule(experiment, visible)
    module.eval()
    output = module(batch)
    if not torch.isfinite(output["loss"]):
        raise RuntimeError("M1 smoke produced non-finite loss")

    pair_differences = {}
    generator = torch.Generator().manual_seed(123)
    for name, encoder in module.event_model.reconstruction.pair_encoders.items():
        members = torch.randn((3, 5, 2, int(experiment.model.d_model)), generator=generator)
        with torch.inference_mode():
            difference = (encoder(members) - encoder(members.flip(2))).abs().max()
        pair_differences[name] = float(difference)
        if difference != 0:
            raise RuntimeError(f"Exchange symmetry is not exact for {name}: max_abs={float(difference)}")

    output["loss"].backward()
    finite_gradients = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )
    if not finite_gradients:
        raise RuntimeError("M1 smoke produced a non-finite gradient")

    local = {}
    for node in module.spec.dag_order:
        pool = batch["structured"]["pools"][node]
        prediction = output["assignment_probabilities"][node].argmax(dim=1) - 1
        active = pool["reconstructable"]
        local[node] = {
            "reconstructable": int(active.sum()),
            "truth_top1": int(((prediction == pool["truth_index"]) & active).sum()),
            "null_correct": int(((prediction == -1) & ~active).sum()),
        }

    decoded = 0
    decoded_truth = 0
    for event in range(len(batch["tokens"])):
        scored = {}
        for root in module.spec.roots:
            pool = batch["structured"]["pools"][root]
            values = []
            for index in pool["candidate_mask"][event].nonzero(as_tuple=False).flatten().tolist():
                owned_tokens = pool["leaf_ownership"][index].nonzero(as_tuple=False).flatten().tolist()
                ownership = frozenset(("TOKEN", int(token)) for token in owned_tokens)
                candidate = CandidateRecord(root, index, (), (), ownership,
                                            bool(index == int(pool["truth_index"][event])))
                probability = output["assignment_probabilities"][root][event, index + 1].detach()
                values.append(ScoredCandidate(candidate, float(probability)))
            scored[root] = tuple(values)
        assignments = decode_topk(module.spec, scored, k=1)
        if assignments:
            decoded += 1
            decoded_truth += int(assignments[0].truth)

    summary = {
        "events": len(batch["tokens"]),
        "loss": float(output["loss"].detach()),
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
        "pair_symmetry_max_abs": pair_differences,
        "finite_gradients": finite_gradients,
        "local": local,
        "decoded": decoded,
        "decoded_truth": decoded_truth,
        "candidate_shapes": {
            node: list(pool["candidate_mask"].shape)
            for node, pool in batch["structured"]["pools"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
