"""CPU release checks for the topology-query and explicit partonic head path.

This is intentionally a bounded contract test, not a performance benchmark.
It uses the versioned synthetic fixture and records numerical gradient norms,
physical round-trip errors, topology-sized tensor shapes, and the two shared
encoder controls used by the development study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.cli.run import build_data
from joint_event_reconstruction.configuration import load_visible_config
from joint_event_reconstruction.data.partonic import PartonicScaler
from joint_event_reconstruction.models.partonic import (
    PartonicFourVectorHead,
    partonic_features,
)
from joint_event_reconstruction.models.topology_queries import TopologyQueryBlock
from joint_event_reconstruction.public_config import load_public_experiment
from joint_event_reconstruction.topology.compiler import load_topology
from joint_event_reconstruction.training.joint import JointEventModule


ROOT = Path(__file__).resolve().parents[1]


def _norm(module, prefix: str) -> float:
    values = [
        parameter.grad.detach().float().norm() ** 2
        for name, parameter in module.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def _parameter_counts(module) -> dict[str, int]:
    prefixes = {
        "visible": "event_model.visible.",
        "queries": "event_model.topology_queries.",
        "assignment": "event_model.reconstruction.",
        "invisible": "event_model.invisible.",
        "partonic": "event_model.partonic_head.",
    }
    return {
        name: sum(
            parameter.numel()
            for key, parameter in module.named_parameters()
            if key.startswith(prefix) and parameter.requires_grad
        )
        for name, prefix in prefixes.items()
    }


def _batch_and_module(config):
    visible = load_visible_config(config)
    visible.dataset.pin_memory = False
    visible.dataset.persistent_workers = False
    data = build_data(config, visible, device="cpu")
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    module = JointEventModule(
        config,
        visible,
        invisible_scaler=data.invisible_scaler,
        point_scaler=data.point_scaler,
        partonic_scaler=data.partonic_scaler,
        synthetic_data=True,
    )
    return data, batch, module


def _loss_gradients(module, batch, output, keys):
    result = {}
    for key in keys:
        module.zero_grad(set_to_none=True)
        output[key].backward(retain_graph=True)
        result[key] = {
            "visible": _norm(module, "event_model.visible."),
            "queries": _norm(module, "event_model.topology_queries."),
            "assignment": _norm(module, "event_model.reconstruction."),
            "invisible": _norm(module, "event_model.invisible."),
            "partonic": _norm(module, "event_model.partonic_head."),
        }
    module.zero_grad(set_to_none=True)
    return result


def _topology_shape_checks() -> dict:
    torch.manual_seed(20260923)
    result = {}
    for count in (1, 2, 3, 5):
        roles = [
            {
                "name": f"role_{index}",
                "node": None,
                "kind": "global" if index == 0 else "role",
                "related": [],
            }
            for index in range(count)
        ]
        policy_results = {}
        for policy in ("full", "topology", "topology_plus_global"):
            block = TopologyQueryBlock(
                8,
                2,
                query_roles=roles,
                interaction=policy,
                layers=1,
                relation_edges=tuple((index, 0) for index in range(1, count)),
            )
            output = block(
                torch.randn(2, 8),
                torch.randn(2, 6, 8),
                torch.zeros(2, 6, dtype=torch.bool),
            )
            policy_results[policy] = {
                "states": list(output["states"].shape),
                "summary": list(output["summary"].shape),
                "context": list(output["context"].shape),
                "roles": list(output["roles"]),
            }
        result[str(count)] = policy_results
    # A role-conditioned head must follow declared role order rather than an
    # assumed electron/muon or parent count.
    head = PartonicFourVectorHead(8, ("role_2", "role_0"))
    # Replace only the decoder in this shape/ordering contract with an
    # identity projection so the selected query slots are directly visible.
    head.output = torch.nn.Linear(8, 4, bias=False)
    with torch.no_grad():
        head.output.weight.zero_()
        head.output.weight[:, :4] = torch.eye(4)
    query_states = torch.zeros(2, 3, 8)
    query_states[:, 0, 0] = 10.0
    query_states[:, 2, 0] = 30.0
    values = head(query_states, ("role_0", "role_1", "role_2"))
    if not torch.equal(values[:, 0, 0], torch.full((2,), 30.0)) or not torch.equal(
        values[:, 1, 0], torch.full((2,), 10.0)
    ):
        raise AssertionError("Partonic head did not preserve declared role ordering")
    result["partonic_head_role_order"] = {
        "shape": list(values.shape),
        "roles": list(head.roles),
    }
    result["available_topology_invisible_role_counts"] = {
        name: len(
            load_topology(ROOT / "configs" / "topology" / filename).invisible_roles
        )
        for name, filename in {
            "one_invisible": "ttbar1L.yaml",
            "two_invisible": "ttbar2L.yaml",
        }.items()
    }
    result["no_invisible_fixture"] = "not_available_in_maintained_topology_fixtures"
    return result


def _variant(base_path: Path, overrides: list[str]):
    return load_public_experiment(base_path, overrides=overrides)


def _run_full_checks(base_path: Path, overrides: list[str] | None = None) -> dict:
    common = [
        "data.events.train=4",
        "data.events.validation=2",
        "data.events.test=2",
        "training.batch_size=2",
    ]
    config = _variant(base_path, common + (overrides or []))
    data, batch, module = _batch_and_module(config)
    module.train()
    output = module(batch)
    losses = ["assignment_loss", "invisible_loss", "partonic_loss", "loss"]
    gradients = _loss_gradients(module, batch, output, losses)

    # The selected CFM network uses zero-initialised output weights.  Its
    # first loss evaluation consequently has no context derivative, but after
    # one legitimate CFM update the context path must be non-zero.
    optimiser = torch.optim.AdamW(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=1.0e-3,
    )
    module.zero_grad(set_to_none=True)
    output["invisible_loss"].backward()
    optimiser.step()
    module.zero_grad(set_to_none=True)
    after_update = module(batch)
    after_update["invisible_loss"].backward()
    gradients["invisible_loss_after_one_update"] = {
        "visible": _norm(module, "event_model.visible."),
        "queries": _norm(module, "event_model.topology_queries."),
        "assignment": _norm(module, "event_model.reconstruction."),
        "invisible": _norm(module, "event_model.invisible."),
        "partonic": _norm(module, "event_model.partonic_head."),
    }

    truth = batch["partonic_truth"]
    roundtrip = data.partonic_scaler.inverse(data.partonic_scaler.transform(truth))
    invisible_state = data.invisible_coordinates.encode(
        batch["invisible_truth"], batch["met_xy"]
    )
    invisible_roundtrip = data.invisible_coordinates.decode_physical(
        invisible_state, batch["met_xy"]
    )["momenta"]
    prediction_without_targets = dict(batch)
    for key in (
        "partonic_truth",
        "partonic_support",
        "partonic_truth_scaled",
    ):
        prediction_without_targets.pop(key, None)
    module.eval()
    with torch.no_grad():
        with_truth = module(batch)["partonic_four_vectors"]
        without_truth = module(prediction_without_targets)["partonic_four_vectors"]

    # Continued-control policy: assignment reaches the shared encoder/query
    # path, while CFM and partonic losses use detached shared states.
    control_config = _variant(
        base_path,
        common
        + [
            "training.shared_encoder_gradients=assignment_only",
            "training.partonic_shared_encoder_gradients=assignment_only",
        ],
    )
    _, control_batch, control_module = _batch_and_module(control_config)
    control_module.train()
    control_output = control_module(control_batch)
    control_gradients = _loss_gradients(
        control_module,
        control_batch,
        control_output,
        ["assignment_loss", "invisible_loss", "partonic_loss", "loss"],
    )

    # Factorised freezing: only the CFM branch remains trainable in this
    # contract control; it must not expose visible/query/assignment gradients.
    factorized_config = _variant(
        base_path,
        common
        + [
            "training.trainable=[invisible]",
            "training.shared_encoder_gradients=assignment_only",
            "training.partonic_shared_encoder_gradients=assignment_only",
        ],
    )
    _, factorized_batch, factorized_module = _batch_and_module(factorized_config)
    factorized_module.train()
    factorized_output = factorized_module(factorized_batch)
    factorized_gradients = _loss_gradients(
        factorized_module,
        factorized_batch,
        factorized_output,
        ["loss"],
    )

    return {
        "config": str(base_path.resolve()),
        "batch_shapes": {
            "tokens": list(batch["tokens"].shape),
            "invisible_truth": list(batch["invisible_truth"].shape),
            "partonic_truth": list(batch["partonic_truth"].shape),
            "partonic_support": list(batch["partonic_support"].shape),
        },
        "partonic_scaler_representation": data.partonic_scaler.representation,
        "partonic_roundtrip_max_abs": float((roundtrip - truth).abs().max()),
        "invisible_roundtrip_max_abs": float(
            (invisible_roundtrip - batch["invisible_truth"]).abs().max()
        ),
        "truth_independent_partonic_prediction_max_abs": float(
            (with_truth - without_truth).abs().max()
        ),
        "parameter_counts": _parameter_counts(module),
        "optimizer_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
        "gradients_joint": gradients,
        "gradients_continued_control": control_gradients,
        "gradients_factorized_freeze": factorized_gradients,
        "topology_shapes": _topology_shape_checks(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    result = _run_full_checks(
        Path(args.experiment).expanduser().resolve(), args.overrides
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
