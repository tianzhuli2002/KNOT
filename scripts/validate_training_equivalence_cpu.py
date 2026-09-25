"""Check cached batching and objective fast-path equivalence on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.training.joint import JointEventModule


def _same(name, left, right, *, atol: float = 1.0e-6, rtol: float = 1.0e-6) -> float:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.dtype == torch.bool or not left.is_floating_point():
            if not torch.equal(left, right):
                raise AssertionError(f"{name} differs")
            return 0.0
        if not torch.allclose(left, right, atol=atol, rtol=rtol, equal_nan=True):
            difference = float((left - right).detach().abs().nan_to_num().max())
            raise AssertionError(f"{name} differs; max_abs={difference}")
        return float((left - right).detach().abs().nan_to_num().max())
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{name} keys differ")
        return max(
            (_same(f"{name}.{key}", left[key], right[key], atol=atol, rtol=rtol) for key in left),
            default=0.0,
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f"{name} lengths differ")
        return max(
            (_same(f"{name}[{index}]", a, b, atol=atol, rtol=rtol) for index, (a, b) in enumerate(zip(left, right, strict=True))),
            default=0.0,
        )
    if left != right:
        raise AssertionError(f"{name} differs: {left!r} != {right!r}")
    return 0.0


def _reference_truth(collator, batch) -> dict[str, torch.Tensor]:
    source_labels, _ = collator.visible_store.lookup(batch["event_index"])
    raw = torch.full(batch["type_ids"].shape, float("nan"), dtype=torch.float32)
    for source_name, values in source_labels.items():
        positions = (collator._type_layout == collator.source_type_ids[source_name]).nonzero().flatten()
        raw[:, positions] = values
    truth_by_node = {}
    result = {}
    for node_name in collator.spec.dag_order:
        template = collator._templates[node_name]
        ownership = template["leaf_ownership"]
        valid = ~(batch["padding_mask"].unsqueeze(1) & ownership.unsqueeze(0)).any(dim=2)
        truth = torch.ones((len(raw), len(ownership)), dtype=torch.bool)
        for role in collator.spec.nodes[node_name].leaf_roles:
            if role.invisible:
                continue
            indices = template["leaf_indices"][role.key]
            values = raw[:, indices.reshape(-1)].reshape(len(raw), len(indices), role.multiplicity)
            expected = template["truth_values"][role.key].to(values.dtype)
            truth &= (values.sort(dim=2).values == expected).all(dim=2)
        for role in collator.spec.nodes[node_name].composite_roles:
            truth &= truth_by_node[role.child][:, template["child_indices"][role.key]]
        truth &= valid
        truth_by_node[node_name] = truth
        result[node_name] = truth
    return result


def run(experiment: str | Path, cache_path: str | Path, output: str | Path) -> dict:
    torch.set_num_threads(1)
    config = load_experiment(
        experiment,
        [
            f"data.cache={Path(cache_path).expanduser().resolve()}",
            "training.loss_weights.partonic_cfm=0",
        ],
    )
    visible = load_visible_config(config)
    data = build_data(config, visible, device="cpu")
    data.setup("fit")
    dataset = data.train_dataset
    positions = [0, 1]
    individual = [dataset[index] for index in positions]
    batched = dataset.__getitems__(positions)
    single_batch = data.structured_collator(individual)
    batch_batch = data.structured_collator(batched)

    measured_error = _same("batched measured input", single_batch, batch_batch)
    reference_truth = _reference_truth(data.structured_collator, single_batch)
    truth_errors = {}
    for name, expected in reference_truth.items():
        pool = single_batch["structured"]["pools"][name]
        candidate_count = expected.sum(dim=1)
        reconstructable = candidate_count == 1
        index = expected.to(torch.long).argmax(dim=1)
        index = torch.where(reconstructable, index, torch.full_like(index, -1))
        _same(f"{name} sparse candidate validity", pool["candidate_mask"], ~(single_batch["padding_mask"].unsqueeze(1) & pool["leaf_ownership"].unsqueeze(0)).any(dim=2))
        _same(f"{name} truth candidate count", candidate_count, expected.sum(dim=1))
        _same(f"{name} truth index", pool["truth_index"], index)
        _same(f"{name} reconstructability", pool["reconstructable"], reconstructable)
        truth_errors[name] = {
            "candidate_count": [int(value) for value in candidate_count.tolist()],
            "reconstructable": [bool(value) for value in reconstructable.tolist()],
        }

    def make_module():
        return JointEventModule(
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

    initial_state = make_module().state_dict()
    rich = make_module().eval()
    fast = make_module().eval()
    rich.load_state_dict(initial_state)
    fast.load_state_dict(initial_state)
    torch.manual_seed(7052026)
    rich_output = rich(single_batch, compute_objectives=False)
    torch.manual_seed(7052026)
    fast_output = fast(batch_batch, compute_objectives=True)
    output_names = ("assignment_logits", "assignment_probabilities", "candidate_embeddings")
    output_errors = {
        name: _same(name, rich_output[name], fast_output[name]) for name in output_names
    }
    loss_names = ("assignment_loss", "invisible_loss", "loss")
    loss_errors = {
        name: _same(name, rich_output[name], fast_output[name]) for name in loss_names
    }
    rich_output["loss"].backward()
    fast_output["loss"].backward()
    gradient_errors = {}
    gradient_norms = {"visible": 0.0, "assignment": 0.0, "invisible": 0.0, "queries": 0.0}
    prefixes = {
        "visible": "event_model.visible.",
        "assignment": "event_model.reconstruction.",
        "invisible": "event_model.invisible.",
        "queries": "event_model.topology_queries.",
    }
    rich_parameters = dict(rich.named_parameters())
    fast_parameters = dict(fast.named_parameters())
    if rich_parameters.keys() != fast_parameters.keys():
        raise AssertionError("Rich and objective paths have different model parameter names")
    for name, rich_parameter in rich_parameters.items():
        fast_parameter = fast_parameters[name]
        _same(f"gradient {name}", rich_parameter.grad, fast_parameter.grad, atol=2.0e-5, rtol=2.0e-5)
        if rich_parameter.grad is not None:
            for component, prefix in prefixes.items():
                if name.startswith(prefix):
                    gradient_norms[component] += float(rich_parameter.grad.detach().norm())
                    break
    if min(gradient_norms.values()) <= 0.0:
        raise AssertionError(f"Expected gradients in all active components: {gradient_norms}")

    rich_optimizer = torch.optim.AdamW(rich.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    fast_optimizer = torch.optim.AdamW(fast.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
    rich_optimizer.step()
    fast_optimizer.step()
    optimizer_error = max(
        _same(f"optimizer parameter {name}", rich_parameters[name].detach(), fast_parameters[name].detach(), atol=2.0e-6, rtol=2.0e-6)
        for name in rich_parameters
    )

    result = {
        "status": "passed",
        "device": "cpu",
        "event_ids": [int(value) for value in single_batch["event_index"].tolist()],
        "batched_retrieval_max_abs": measured_error,
        "candidate_truth": truth_errors,
        "forward_output_max_abs": output_errors,
        "loss_max_abs": loss_errors,
        "gradient_max_abs": "all named gradients matched at atol=2e-5, rtol=2e-5",
        "gradient_norms": gradient_norms,
        "optimizer_update_max_abs": optimizer_error,
        "comparison": "rich prediction-capable forward vs objective-aware training forward, identical weights/seeds",
    }
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.experiment, args.cache, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
