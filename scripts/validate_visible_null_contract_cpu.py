"""Exercise structured NULL-target semantics on one bounded CPU event."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.training.joint import JointEventModule


def run(experiment: str | Path, output: str | Path) -> dict:
    config = load_experiment(
        experiment,
        [
            "data.events.train=12",
            "data.events.validation=8",
            "training.loss_weights.invisible=0",
            "training.loss_weights.partonic_cfm=0",
        ],
    )
    visible = load_visible_config(config)
    data = build_data(config, visible)
    data.setup("fit")
    item = data.train_dataset[0]
    event_id = int(item["event_index"])

    store = data.structured_collator.visible_store
    if store is None:
        raise RuntimeError("NULL-target fixture requires the real visible-truth sidecar store")
    position = int(np.searchsorted(store.event_ids, event_id))
    if position >= len(store.event_ids) or int(store.event_ids[position]) != event_id:
        raise KeyError(f"Visible truth store does not contain EVENT_ID={event_id}")

    jet_labels = store.labels["JET"][position]
    jet_labels[np.isfinite(jet_labels)] = 0.0
    store.support[position] = False
    batch = data.structured_collator([item])
    for role in ("top", "antitop"):
        pool = batch["structured"]["pools"][role]
        if bool(pool["reconstructable"][0]) or int(pool["truth_index"][0]) != -1:
            raise AssertionError(f"Synthetic unsupported event did not produce NULL target for {role}")

    module = JointEventModule(
        config,
        visible,
        invisible_scaler=data.invisible_scaler,
        point_scaler=data.point_scaler,
        partonic_scaler=data.partonic_scaler,
        parent_cfm_scaler=data.partonic_scaler,
        synthetic_data=bool(
            data.structured_collator.invisible_store is not None
            and data.structured_collator.invisible_store.synthetic
        ),
    )
    module.train()
    result = module(batch, compute_objectives=True)
    loss = result["assignment_loss"]
    if not torch.isfinite(loss):
        raise AssertionError("NULL-target assignment loss is non-finite")
    losses = result["node_losses"]
    logits = result["assignment_logits"]
    details = {}
    for role in ("top", "antitop"):
        expected = torch.nn.functional.cross_entropy(
            logits[role], torch.zeros(1, dtype=torch.long)
        )
        if not torch.allclose(losses[role], expected, rtol=0.0, atol=1.0e-7):
            raise AssertionError(f"NULL target class was not retained for {role}")
        details[role] = {
            "reconstructable": bool(batch["structured"]["pools"][role]["reconstructable"][0]),
            "truth_index": int(batch["structured"]["pools"][role]["truth_index"][0]),
            "null_target_class": 0,
            "candidate_logit_shape": list(logits[role].shape),
            "loss": float(losses[role].detach()),
        }
    loss.backward()
    query_gradient_norm = sum(
        float(parameter.grad.detach().norm())
        for parameter in module.event_model.topology_queries.parameters()
        if parameter.grad is not None
    )
    if not np.isfinite(query_gradient_norm) or query_gradient_norm <= 0.0:
        raise AssertionError("NULL assignment loss did not reach shared topology queries")

    result = {
        "status": "passed",
        "event_id": event_id,
        "synthetic_change": "all finite jet-role labels changed to background in memory; sidecar file untouched",
        "roots": details,
        "assignment_loss": float(loss.detach()),
        "query_gradient_norm": query_gradient_norm,
        "null_loss_retained": True,
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
