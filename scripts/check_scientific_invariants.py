#!/usr/bin/env python3
"""Focused contracts that protect study identity and event selection."""

from __future__ import annotations

import json
import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.cli.run import _load_initial_components
from joint_event_reconstruction.configuration import load_experiment, load_visible_config
from joint_event_reconstruction.data.subsets import deterministic_subset_positions
from joint_event_reconstruction.training.joint import (
    JointEventModule,
    active_architecture,
)


def _assert_close(first, second, name: str, tolerance: float = 2.0e-5) -> None:
    difference = (first - second).abs().max().item()
    if difference > tolerance:
        raise AssertionError(f"{name} changed under a symmetry transformation: {difference}")


def _architecture(overrides: list[str] | None = None) -> str:
    config = load_experiment("configs/experiments/ttbar2L.yaml", overrides or [])
    return json.dumps(active_architecture(config.model), sort_keys=True)


def check_architecture_signature_contract() -> None:
    reference = _architecture()
    inference_only = (
        "prediction.seed=7",
        "prediction.top_k=5",
        "prediction.samples=64",
        "prediction.steps=16",
        "prediction.solver=euler",
        "prediction.sample_chunk_size=4",
        "prediction.nfe=33",
        "prediction.compatibility_weight=2.0",
    )
    for override in inference_only:
        if _architecture([override]) != reference:
            raise AssertionError(f"Inference-only override changed architecture: {override}")

    architecture_changes = (
        "model.encoder.layers=5",
        "model.assignment.recurrence.train_steps=[1,2]",
        "model.assignment.recurrence.feedback=soft",
        "model.invisible.method=cfm",
    )
    for override in architecture_changes:
        if _architecture([override]) == reference:
            raise AssertionError(f"Training/architecture override was omitted: {override}")

    c2_base = [
        "model.assignment.recurrence.train_steps=[1,2]",
        "model.assignment.recurrence.eval_steps=2",
        "model.invisible.method=cfm",
        "model.c2={}",
    ]
    c2_reference = _architecture(c2_base)
    if _architecture(
        [*c2_base, "model.c2.training_sampler.steps=8"]
    ) != c2_reference:
        raise AssertionError("Equivalent C2 sampler override changed architecture")
    if _architecture(
        [*c2_base, "model.c2.training_sampler.steps=9"]
    ) == c2_reference:
        raise AssertionError("C2 training sampler change was omitted from identity")
    if _architecture(
        [
            *c2_base,
            "prediction.steps=17",
            "prediction.solver=euler",
        ]
    ) != c2_reference:
        raise AssertionError("Prediction sampler changed C2 architecture identity")


def _component_signatures(overrides: list[str] | None = None) -> dict[str, str]:
    config = load_experiment("configs/experiments/ttbar2L.yaml", overrides or [])
    from joint_event_reconstruction.training.joint import architecture_component_signatures

    return architecture_component_signatures(config.model)


def check_partial_checkpoint_contract() -> None:
    import torch

    base = load_experiment("configs/experiments/ttbar2L.yaml")
    parent = JointEventModule(base, load_visible_config(base), synthetic_data=True)
    with tempfile.TemporaryDirectory(prefix="jer-partial-checkpoint-") as directory:
        checkpoint_path = f"{directory}/parent.ckpt"
        torch.save(
            {
                "state_dict": parent.state_dict(),
                "hyper_parameters": dict(parent.hparams),
                "invisible_scaler": parent.scaler_metadata,
            },
            checkpoint_path,
        )

        target_config = load_experiment(
            "configs/experiments/ttbar2L.yaml", ["model.invisible.method=cfm"]
        )
        target = JointEventModule(
            target_config, load_visible_config(target_config), synthetic_data=True
        )
        unselected = {
            key: value.detach().clone()
            for key, value in target.state_dict().items()
            if key.startswith("event_model.invisible.")
        }
        _load_initial_components(target, checkpoint_path, ("visible", "assignment"))
        parent_state = parent.state_dict()
        target_state = target.state_dict()
        for key, value in parent_state.items():
            if key.startswith(("event_model.visible.", "event_model.reconstruction.")):
                if not torch.equal(target_state[key], value):
                    raise AssertionError(f"Selected checkpoint value was not loaded: {key}")
        for key, value in unselected.items():
            if not torch.equal(target_state[key], value):
                raise AssertionError(f"Unselected checkpoint component changed: {key}")

        changed_config = load_experiment(
            "configs/experiments/ttbar2L.yaml", ["model.encoder.dropout=0.2"]
        )
        changed = JointEventModule(
            changed_config, load_visible_config(changed_config), synthetic_data=True
        )
        try:
            _load_initial_components(changed, checkpoint_path, ("visible",))
        except ValueError as error:
            if "component signatures" not in str(error):
                raise AssertionError(f"Unexpected partial-load error: {error}") from error
        else:
            raise AssertionError("Changed selected component was accepted")


def check_component_signature_contract() -> None:
    reference = _component_signatures()
    inference_only = (
        "prediction.seed=7",
        "prediction.top_k=5",
        "prediction.samples=64",
        "prediction.steps=16",
        "prediction.solver=euler",
        "prediction.sample_chunk_size=4",
        "prediction.nfe=33",
        "prediction.compatibility_weight=2.0",
    )
    for override in inference_only:
        if _component_signatures([override]) != reference:
            raise AssertionError(f"Inference-only override changed component identity: {override}")

    if _component_signatures(["model.encoder.dropout=0.2"])["visible"] == reference["visible"]:
        raise AssertionError("Visible architecture change was omitted from component identity")
    if _component_signatures(["model.assignment.recurrence.train_steps=[1,2]"])["assignment"] == reference["assignment"]:
        raise AssertionError("Assignment architecture change was omitted from component identity")

    c2_base = [
        "model.assignment.recurrence.train_steps=[1,2]",
        "model.assignment.recurrence.eval_steps=2",
        "model.invisible.method=cfm",
        "model.c2={}",
    ]
    c2_reference = _component_signatures(c2_base)
    if _component_signatures([*c2_base, "model.c2.training_sampler.steps=9"])["c2"] == c2_reference["c2"]:
        raise AssertionError("C2 training sampler change was omitted from component identity")


def check_nested_subsets() -> None:
    event_ids = np.arange(10_000, dtype=np.uint64) * 17 + 3
    previous: set[int] = set()
    for count in (100, 250, 500, 1000):
        positions = deterministic_subset_positions(event_ids, count, seed=42)
        repeated = deterministic_subset_positions(event_ids, count, seed=42)
        np.testing.assert_array_equal(positions, repeated)
        selected = set(event_ids[positions].tolist())
        if not previous < selected:
            raise AssertionError(f"Stable subsets are not strictly nested at {count}")
        previous = selected


def check_loader_split_seed_contract() -> None:
    base = load_experiment("configs/experiments/ttbar2L.yaml")
    changed_model_seed = load_experiment(
        "configs/experiments/ttbar2L.yaml", ["experiment.seed=123"]
    )
    for experiment in (base, changed_model_seed):
        visible = load_visible_config(experiment)
        if int(visible.general.seed) != int(experiment.data.subset_seed):
            raise AssertionError(
                "Visible-loader split seed must mirror data.subset_seed, not "
                "experiment.seed"
            )
    if int(base.experiment.seed) == int(changed_model_seed.experiment.seed):
        raise AssertionError("Loader seed contract test did not change model seed")


def check_truth_oracle_unreconstructable_contract() -> None:
    """Truth-oracle conditioning must mask, rather than index, missing parents."""
    import torch

    from scripts.smoke_synthetic import build_synthetic_batch

    config = load_experiment(
        "configs/experiments/ttbar2L.yaml",
        [
            "model.invisible.method=cfm",
            "model.invisible.conditioning=truth_oracle",
            "training.trainable=[visible,assignment,invisible]",
            "training.loss_weights.compatibility=0.0",
        ],
    )
    visible, batch, scaler = build_synthetic_batch(config, events=2, device="cpu")
    node = next(iter(batch["structured"]["pools"]))
    pool = batch["structured"]["pools"][node]
    pool["truth_index"][0] = -1
    pool["reconstructable"][0] = False
    module = JointEventModule(
        config, visible, invisible_scaler=scaler, synthetic_data=True
    )
    output = module(batch)
    if not torch.isfinite(output["loss"]):
        raise AssertionError("Truth-oracle masking produced a non-finite loss")
    if bool(output["invisible_training_mask"][0]):
        raise AssertionError("Unreconstructable truth-oracle event was not masked")

    from joint_event_reconstruction.physics import compose_assignment_state
    from joint_event_reconstruction.prediction import _compose_physics, _truth_node_indices

    valid_batch = copy.deepcopy(batch)
    valid_batch["structured"]["pools"][node]["truth_index"][0] = 0
    valid_batch["structured"]["pools"][node]["reconstructable"][0] = True
    module = JointEventModule(
        config, visible, invisible_scaler=scaler, synthetic_data=True
    )
    samples = torch.zeros(
        (2, 1, 2, len(module.spec.invisible_roles), 3), dtype=torch.float32
    )
    composites, _, _ = _compose_physics(
        module,
        valid_batch,
        [(), ()],
        samples,
        truth_assignment=True,
    )
    truth_indices = _truth_node_indices(module.spec, valid_batch, 1)
    oracle_invisible = {
        role.key: samples[1, 0, 0, role_index]
        for role_index, role in enumerate(module.spec.invisible_roles)
    }
    expected = compose_assignment_state(
        module.spec,
        valid_batch["structured"]["pools"],
        valid_batch["physics_vectors"][1],
        truth_indices,
        oracle_invisible,
    )
    for name in module.spec.dag_order:
        expected_values = torch.stack([expected[name], expected[name]])
        if not torch.allclose(
            torch.from_numpy(composites[name][1, 0]),
            expected_values,
            equal_nan=True,
        ):
            raise AssertionError(
                "Truth-oracle composite output did not use the truth assignment"
            )

    alternative, _, _ = _compose_physics(
        module,
        valid_batch,
        [[object()], [object()]],
        samples,
        truth_assignment=True,
    )
    for name in module.spec.dag_order:
        if not np.allclose(composites[name], alternative[name], equal_nan=True):
            raise AssertionError(
                "Truth-oracle composite depends on model-predicted assignment"
            )

    changed_samples = samples.clone()
    changed_samples[1, 0, 0, 0, 0] = 1.0
    changed, _, _ = _compose_physics(
        module,
        valid_batch,
        [(), ()],
        changed_samples,
        truth_assignment=True,
    )
    if all(
        np.allclose(composites[name][1, 0, 0], changed[name][1, 0, 0])
        for name in module.spec.dag_order
    ):
        raise AssertionError(
            "Truth-oracle composite is insensitive to its generated invisible sample"
        )

    unsupported, _, _ = _compose_physics(
        module,
        batch,
        [(), ()],
        samples,
        truth_assignment=True,
    )
    for name in module.spec.dag_order:
        if not np.isnan(unsupported[name][0]).all():
            raise AssertionError(
                "Unsupported truth-oracle composite must be explicitly all-NaN"
            )


def _permuted_same_type_batch(batch: dict, positions) -> dict:
    """Permute same-type token positions and all static candidate indices with them."""
    import torch

    result = copy.deepcopy(batch)
    token_count = batch["tokens"].shape[1]
    permutation = torch.arange(token_count)
    positions = torch.as_tensor(positions, dtype=torch.long)
    permutation[positions] = positions.flip(0)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(token_count)
    for name in ("tokens", "type_ids", "padding_mask", "physics_vectors"):
        if name in result:
            result[name] = result[name][:, permutation]
    for pool in result["structured"]["pools"].values():
        pool["leaf_ownership"] = pool["leaf_ownership"][:, permutation]
        pool["leaf_indices"] = {
            role: torch.where(
                indices >= 0, inverse[indices.clamp_min(0)], indices
            )
            for role, indices in pool["leaf_indices"].items()
        }
    return result


def check_exchange_and_permutation_contract() -> None:
    import torch

    from scripts.smoke_synthetic import build_synthetic_batch
    from joint_event_reconstruction.models.reconstruction.structured import (
        SymmetricPairEncoder,
        SymmetricSetEncoder,
    )

    torch.manual_seed(701)
    pair = SymmetricPairEncoder(8, 0.0).eval()
    pair_members = torch.randn(2, 3, 2, 8)
    _assert_close(pair(pair_members), pair(pair_members.flip(2)), "pair exchange symmetry", 0.0)

    set_encoder = SymmetricSetEncoder(8, 0.0).eval()
    set_members = torch.randn(2, 3, 4, 8)
    _assert_close(
        set_encoder(set_members),
        set_encoder(set_members[:, :, (2, 0, 3, 1)]),
        "set exchange symmetry",
        2.0e-5,
    )

    base = load_experiment("configs/experiments/ttbar2L.yaml")
    visible, batch, scaler = build_synthetic_batch(base, events=2, device="cpu")
    model = JointEventModule(
        base, visible, invisible_scaler=scaler, synthetic_data=True
    ).eval()
    permuted = _permuted_same_type_batch(batch, (0, 1, 2, 3, 4, 5, 6, 7))
    with torch.inference_mode():
        original = model(batch)
        transformed = model(permuted)
    _assert_close(
        original["visible"]["pooled"],
        transformed["visible"]["pooled"],
        "visible same-type permutation equivariance",
    )
    for node in model.spec.dag_order:
        _assert_close(
            original["assignment_logits"][node],
            transformed["assignment_logits"][node],
            f"{node} assignment permutation equivariance",
        )


def _check_one_freeze_case(
    config,
    visible,
    batch,
    scaler,
    expected_prefixes: tuple[str, ...],
    name: str,
) -> None:
    import torch

    module = JointEventModule(
        config, visible, invisible_scaler=scaler, synthetic_data=True
    )
    frozen = {
        key: value.detach().clone()
        for key, value in module.named_parameters()
        if not value.requires_grad
    }
    output = module(batch)
    if not torch.isfinite(output["loss"]):
        raise AssertionError(f"{name} produced a non-finite loss")
    module.zero_grad(set_to_none=True)
    output["loss"].backward()
    intended = [
        parameter
        for key, parameter in module.named_parameters()
        if parameter.requires_grad
        and key.startswith(expected_prefixes)
        and parameter.grad is not None
    ]
    if not intended or not all(torch.isfinite(parameter.grad).all() for parameter in intended):
        raise AssertionError(f"{name} did not produce finite intended gradients")
    optimiser = torch.optim.AdamW(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=1.0e-4,
        fused=False,
    )
    optimiser.step()
    for key, before in frozen.items():
        if not torch.equal(module.state_dict()[key], before):
            raise AssertionError(f"{name} changed frozen parameter {key}")


def check_freeze_contract() -> None:
    from scripts.smoke_all import configure as configure_factorized
    from scripts.smoke_flows import configure as configure_flow
    from scripts.smoke_synthetic import build_synthetic_batch

    base = load_experiment("configs/experiments/ttbar2L.yaml")
    visible, batch, scaler = build_synthetic_batch(base, events=2, device="cpu")

    assignment = copy.deepcopy(base)
    assignment.training.trainable = ["visible", "assignment"]
    assignment.training.loss_weights.invisible = 0.0
    assignment.training.loss_weights.compatibility = 0.0
    _check_one_freeze_case(
        assignment,
        visible,
        batch,
        scaler,
        ("event_model.visible.", "event_model.reconstruction."),
        "visible+assignment",
    )

    invisible = configure_flow(base, "cfm_no_assignment")
    invisible.training.trainable = ["invisible"]
    invisible.training.loss_weights.assignment = 0.0
    invisible.training.loss_weights.invisible = 1.0
    _check_one_freeze_case(
        invisible,
        visible,
        batch,
        scaler,
        ("event_model.invisible.",),
        "invisible-only",
    )

    c2 = configure_factorized(base, "edm_c2_detached")
    c2.training.trainable = ["assignment", "c2"]
    c2.training.loss_weights.assignment = 1.0
    c2.training.loss_weights.invisible = 0.0
    c2.training.loss_weights.compatibility = 0.0
    _check_one_freeze_case(
        c2,
        visible,
        batch,
        scaler,
        (
            "event_model.reconstruction.",
            "event_model.reasoner.",
            "event_model.feedback.",
            "event_model.posterior_feedback.",
        ),
        "assignment+c2",
    )

    compatibility = configure_factorized(base, "joint_compatibility")
    compatibility.training.trainable = ["compatibility"]
    compatibility.training.loss_weights.assignment = 0.0
    compatibility.training.loss_weights.invisible = 0.0
    compatibility.training.loss_weights.compatibility = 1.0
    _check_one_freeze_case(
        compatibility,
        visible,
        batch,
        scaler,
        ("event_model.compatibility.",),
        "compatibility-only",
    )

    joint = configure_flow(base, "joint_flow")
    joint.training.trainable = ["assignment", "invisible"]
    joint.training.loss_weights.assignment = 1.0
    joint.training.loss_weights.invisible = 1.0
    _check_one_freeze_case(
        joint,
        visible,
        batch,
        scaler,
        ("event_model.joint_flow.", "event_model.invisible."),
        "joint assignment+invisible",
    )


def main() -> None:
    check_architecture_signature_contract()
    check_component_signature_contract()
    check_partial_checkpoint_contract()
    check_nested_subsets()
    check_loader_split_seed_contract()
    check_truth_oracle_unreconstructable_contract()
    check_exchange_and_permutation_contract()
    check_freeze_contract()
    print(
        "PASS: architecture, partial-checkpoint, split-seed, nested-subset, "
        "exchange, permutation and freeze contracts"
    )


if __name__ == "__main__":
    main()
