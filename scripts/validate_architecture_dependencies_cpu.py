"""Validate shared-query, objective-dependency and support contracts on CPU."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.invisible import InvisibleScaler, InvisibleTruthStore
from joint_event_reconstruction.models.topology_queries import TopologyQueryBlock
from joint_event_reconstruction.training.joint import JointEventModule


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = ROOT / "configs/experiments"
OUTPUT_DEFAULT = (
    ROOT
    / "results/experiments/partonic_ttbar2L_2M_v1/architecture_dependencies_cpu.json"
)


def _norm(module, prefix: str) -> float:
    values = [
        parameter.grad.detach().float().norm().square()
        for name, parameter in module.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    ]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def _module(config, visible, data) -> JointEventModule:
    store = data.structured_collator.invisible_store
    synthetic = bool(store.synthetic) if store is not None else bool(
        data.structured_collator.partonic_store is not None
        and data.structured_collator.partonic_store.synthetic
    )
    return JointEventModule(
        config,
        visible,
        invisible_scaler=data.invisible_scaler,
        point_scaler=data.point_scaler,
        partonic_scaler=data.partonic_scaler,
        parent_cfm_scaler=data.partonic_scaler,
        synthetic_data=synthetic,
    )


def _load(config_name: str, overrides: list[str] | None = None):
    config = load_experiment(EXPERIMENT_ROOT / config_name, overrides or [])
    visible = load_visible_config(config)
    data = build_data(config, visible, device="cpu")
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    return config, visible, data, batch, _module(config, visible, data)


def _mask_and_padding_checks() -> dict:
    roles = (
        {"name": "top", "node": "top", "kind": "parent", "related": ["ttbar"]},
        {"name": "antitop", "node": "antitop", "kind": "parent", "related": ["ttbar"]},
        {"name": "ttbar", "node": None, "kind": "global", "related": ["top", "antitop"]},
    )
    relation_edges = ((0, 2), (1, 2))
    expected_topology = torch.tensor(
        [[False, True, False], [True, False, False], [False, False, False]],
        dtype=torch.bool,
    )
    masks = {}
    blocks = {}
    for policy in ("full", "topology", "topology_plus_global"):
        block = TopologyQueryBlock(
            8,
            2,
            query_roles=roles,
            interaction=policy,
            layers=1,
            relation_edges=relation_edges,
            required_node_names=("top", "antitop"),
            invisible_role_names=("top.nu", "antitop.nu"),
            invisible_role_nodes=("top", "antitop"),
            topology_identity="charge_side_fixture_v1",
        ).eval()
        blocks[policy] = block
        observed = block.query_attention_mask.cpu()
        expected = (
            torch.zeros_like(observed)
            if policy == "full"
            else expected_topology
        )
        if not torch.equal(observed, expected):
            raise AssertionError(
                f"{policy} attention mask differs from compiled relationships: {observed}"
            )
        masks[policy] = observed.tolist()

    block = blocks["topology_plus_global"]
    torch.manual_seed(82017)
    pooled = torch.randn(2, 8)
    tokens = torch.randn(2, 5, 8)
    padding = torch.tensor(
        [[False, False, True, True, True], [False, False, False, True, True]]
    )
    changed = tokens.clone()
    changed[padding] = torch.randn_like(changed[padding]) * 1000.0
    with torch.no_grad():
        first = block(pooled, tokens, padding)
        second = block(pooled, changed, padding)
    if not torch.allclose(first.query_states, second.query_states, atol=1e-6, rtol=1e-6):
        raise AssertionError("Padded token values changed the topology-query states")
    if first.node_query_indices != (("top", 0), ("antitop", 1)):
        raise AssertionError(f"Compiled node-query mapping changed: {first.node_query_indices}")
    if first.invisible_role_names != ("top.nu", "antitop.nu"):
        raise AssertionError(f"Invisible role order changed: {first.invisible_role_names}")
    if first.parent_role_names != ("top", "antitop"):
        raise AssertionError(f"Parent role order changed: {first.parent_role_names}")
    if first.query_states.shape != (2, 3, 8) or first.invisible_contexts.shape != (2, 2, 8):
        raise AssertionError("Topology-query state contract returned unexpected dimensions")
    return {
        "status": "passed",
        "attention_masks_blocked_true": masks,
        "padding_invariance_max_abs": float(
            (first.query_states - second.query_states).abs().max()
        ),
        "node_query_indices": list(first.node_query_indices),
        "invisible_role_names": list(first.invisible_role_names),
        "parent_role_names": list(first.parent_role_names),
        "query_shape": list(first.query_states.shape),
        "invisible_context_shape": list(first.invisible_contexts.shape),
    }


def _query_mode_comparison() -> dict:
    q1_config, q1_visible, data, batch, q1_module = _load(
        "partonic_ttbar2L_2M_q1_cpu.public.yaml"
    )
    q0_config = load_experiment(
        EXPERIMENT_ROOT / "partonic_ttbar2L_2M_q0_cpu.public.yaml"
    )
    q0_visible = load_visible_config(q0_config)
    for key in ("input", "inference_input", "visible_truth", "invisible_truth"):
        if getattr(q0_config.data, key, None) != getattr(q1_config.data, key, None):
            raise AssertionError(f"Q0/Q1 data contract differs for data.{key}")
    if (
        q0_config.input_definition != q1_config.input_definition
        or q0_config.data.truth_bindings != q1_config.data.truth_bindings
        or q0_config.data.max_events != q1_config.data.max_events
        or q0_config.data.subset_seed != q1_config.data.subset_seed
        or q0_config.experiment.seed != q1_config.experiment.seed
    ):
        raise AssertionError("Q0/Q1 input ordering, truth labels, split subset, or seed differs")
    if q0_config.topology_config != q1_config.topology_config:
        raise AssertionError("Q0/Q1 compiled topology configuration differs")
    for path in (
        "model.latent",
        "model.visible",
        "model.assignment",
        "model.mode",
        "model.invisible",
        "training.loss_weights",
        "training.fused_adamw",
        "training.learning_rate",
        "training.weight_decay",
        "training.batch_size",
        "training.epochs",
        "training.precision",
    ):
        q0_value = q0_config
        q1_value = q1_config
        for field in path.split("."):
            q0_value = q0_value[field] if isinstance(q0_value, dict) else getattr(q0_value, field)
            q1_value = q1_value[field] if isinstance(q1_value, dict) else getattr(q1_value, field)
        if q0_value != q1_value:
            raise AssertionError(f"Q0/Q1 controlled comparison differs for {path}")

    q0_module = _module(q0_config, q0_visible, data)
    if q0_module.event_model.topology_queries is not None:
        raise AssertionError("Q0 unexpectedly constructed topology queries")
    if q1_module.event_model.topology_queries is None:
        raise AssertionError("Q1 did not construct the shared topology-query block")
    q0_module.train()
    q1_module.train()
    q0_output = q0_module(batch, compute_objectives=True)
    q1_output = q1_module(batch, compute_objectives=True)
    for name, output in (("Q0", q0_output), ("Q1", q1_output)):
        for loss_name in ("assignment_loss", "invisible_loss", "loss"):
            value = output[loss_name]
            if not torch.isfinite(value):
                raise AssertionError(f"{name} produced non-finite {loss_name}")
    if "query_states" in q0_output["visible"] or "query_states" not in q1_output["visible"]:
        raise AssertionError("No-query and query-enabled paths did not remain distinct")
    if "invisible_role_context" in q0_output["visible"] or "invisible_role_context" not in q1_output["visible"]:
        raise AssertionError("No-query and query-enabled neutrino conditioning paths were conflated")
    q1_role_context = q1_output["visible"]["invisible_role_context"]
    if torch.allclose(q1_role_context[:, 0], q1_role_context[:, 1]):
        raise AssertionError("Query-enabled neutrino roles collapsed to one identical context")
    q1_module.zero_grad(set_to_none=True)
    q1_output["assignment_loss"].backward(retain_graph=True)
    query_gradient = _norm(q1_module, "event_model.topology_queries.")
    if query_gradient <= 0.0:
        raise AssertionError("Q1 assignment loss did not reach the shared query parameters")
    q1_module.zero_grad(set_to_none=True)
    q1_output["invisible_loss"].backward()
    zero_initialization_gradient = _norm(
        q1_module, "event_model.topology_queries.invisible_projection."
    )
    cfm_optimizer = torch.optim.AdamW(q1_module.parameters(), lr=0.001)
    cfm_optimizer.step()
    q1_module.zero_grad(set_to_none=True)
    q1_after_update = q1_module(batch, compute_objectives=True)
    q1_after_update["invisible_loss"].backward()
    invisible_query_gradient = _norm(
        q1_module, "event_model.topology_queries.invisible_projection."
    )
    if invisible_query_gradient <= 0.0:
        raise AssertionError("Q1 neutrino CFM loss did not reach role-specific query projections")
    return {
        "status": "passed",
        "event_ids": [int(value) for value in batch["event_index"].tolist()],
        "Q0_query_module": "absent",
        "Q1_query_module": "present",
        "Q1_role_context_max_abs_separation": float(
            (q1_role_context[:, 0] - q1_role_context[:, 1]).detach().abs().max()
        ),
        "Q0_assignment_loss": float(q0_output["assignment_loss"].detach()),
        "Q1_assignment_loss": float(q1_output["assignment_loss"].detach()),
        "Q1_assignment_query_gradient_norm": query_gradient,
        "Q1_initial_zero_output_role_projection_gradient_norm": zero_initialization_gradient,
        "Q1_neutrino_role_projection_gradient_norm": invisible_query_gradient,
        "input_contract": "identical measured events, topology, truth sidecars, and input features",
    }


def _truth_store_partial_support_check() -> dict:
    role_names = ("top.nu", "antitop.nu")
    event_ids = np.asarray([15, 16, 17], dtype=np.int64)
    dtype = np.dtype([(name, "<f4") for name in ("px", "py", "pz")])
    records = np.zeros((3, 2), dtype=dtype)
    records["px"] = [[1.0, 5.0], [3.0, np.nan], [np.nan, 9.0]]
    records["py"] = [[2.0, 6.0], [4.0, np.nan], [np.nan, 10.0]]
    records["pz"] = [[3.0, 7.0], [5.0, np.nan], [np.nan, 11.0]]
    support = np.asarray([[True, True], [True, False], [False, True]])
    with tempfile.TemporaryDirectory(prefix="jer-invisible-support-") as temporary:
        path = Path(temporary) / "truth.h5"
        with h5py.File(path, "w") as destination:
            destination.create_dataset("EVENT_ID", data=event_ids)
            destination.create_dataset("LABELS/NEUTRINO", data=records)
            support_dataset = destination.create_dataset(
                "SUPPORT/INVISIBLE_ROLE", data=support
            )
            support_dataset.attrs["role_order"] = json.dumps(role_names)
            destination.attrs["invisible_roles"] = json.dumps(role_names)
            destination.attrs["synthetic"] = False
        store = InvisibleTruthStore(path, role_names, allow_synthetic=False)
        values, observed_support = store.lookup_with_support(event_ids)
        if not torch.equal(observed_support, torch.from_numpy(support)):
            raise AssertionError("InvisibleTruthStore changed the explicit role support mask")
        if not torch.isfinite(values).all() or torch.count_nonzero(values[1, 1]) or torch.count_nonzero(values[2, 0]):
            raise AssertionError("Unsupported neutrino placeholders were not safely zeroed")
        try:
            store.lookup(event_ids)
        except ValueError:
            pass
        else:
            raise AssertionError("Unmasked lookup accepted partially unsupported neutrino truth")
        scaler = InvisibleScaler.fit(values, role_names, support_mask=observed_support)
        if scaler.mean[0] != (2.0, 3.0, 4.0) or scaler.mean[1] != (7.0, 8.0, 9.0):
            raise AssertionError(f"Invisible scaler used unsupported rows: {scaler.mean}")
    return {
        "status": "passed",
        "event_ids": event_ids.tolist(),
        "role_support_counts": support.sum(axis=0).astype(int).tolist(),
        "train_only_supported_role_means": [list(row) for row in scaler.mean],
        "unsupported_nan_targets_masked": True,
    }


def _collated_partial_support_check() -> dict:
    config_name = "partonic_ttbar2L_2M_q1_cpu.public.yaml"
    config, visible, data, _, _ = _load(config_name)
    source_store = data.structured_collator.invisible_store
    if source_store is None:
        raise AssertionError("Q1 support fixture has no invisible truth store")
    item_count = min(4, len(data.train_dataset))
    reference_items = [data.train_dataset[index] for index in range(item_count)]
    event_ids = np.asarray(
        [int(item["event_index"]) for item in reference_items], dtype=np.int64
    )
    if len(event_ids) != 4:
        raise AssertionError("Collator support fixture requires four development train events")
    support = np.ones((len(source_store.event_ids), 2), dtype=np.bool_)
    for event_id in event_ids[1::2]:
        row = int(np.searchsorted(source_store.event_ids, event_id))
        if row >= len(source_store.event_ids) or source_store.event_ids[row] != event_id:
            raise AssertionError(f"Fixture source store is missing event {event_id}")
        support[row] = False

    with tempfile.TemporaryDirectory(prefix="jer-invisible-collator-") as temporary:
        path = Path(temporary) / "partial_support.h5"
        record_dtype = np.dtype([(name, "<f4") for name in ("px", "py", "pz")])
        records = np.empty((len(source_store.event_ids), 2), dtype=record_dtype)
        for component_index, component_name in enumerate(("px", "py", "pz")):
            records[component_name] = source_store.values[..., component_index]
        with h5py.File(path, "w") as destination:
            destination.create_dataset("EVENT_ID", data=source_store.event_ids)
            destination.create_dataset("LABELS/NEUTRINO", data=records)
            support_dataset = destination.create_dataset(
                "SUPPORT/INVISIBLE_ROLE", data=support
            )
            support_dataset.attrs["role_order"] = json.dumps(source_store.role_names)
            destination.attrs["invisible_roles"] = json.dumps(source_store.role_names)
            destination.attrs["synthetic"] = True
            destination.attrs["units"] = "GeV"

        fixture_config = load_experiment(
            EXPERIMENT_ROOT / config_name,
            [f"data.invisible_truth={path}"],
        )
        fixture_visible = load_visible_config(fixture_config)
        fixture_data = build_data(fixture_config, fixture_visible, device="cpu")
        fixture_data.setup("fit")
        batch_items = [fixture_data.train_dataset[index] for index in range(item_count)]
        batch = fixture_data.structured_collator(batch_items)
        rows = np.searchsorted(source_store.event_ids, batch["event_index"].numpy())
        expected_support = torch.from_numpy(support[rows])
        if not torch.equal(batch["invisible_support"], expected_support):
            raise AssertionError("StructuredCollator changed or misaligned role support")
        if batch["invisible_all_supported"]:
            raise AssertionError("StructuredCollator marked a partially supported store complete")
        joint_fraction = float(expected_support.all(dim=1).float().mean())
        if joint_fraction != 0.5:
            raise AssertionError(f"Fixture should contain half supported events, found {joint_fraction}")
        output = _module(fixture_config, fixture_visible, fixture_data)(
            batch, compute_objectives=True
        )
        if not torch.isfinite(output["invisible_loss"]):
            raise AssertionError("Partially supported collated neutrino batch has non-finite loss")
        observed_fraction = float(output["invisible_physical_support_fraction"])
        if observed_fraction != joint_fraction:
            raise AssertionError("Model neutrino support reduction disagrees with collated mask")
    return {
        "status": "passed",
        "event_ids": event_ids.tolist(),
        "role_support_shape": list(batch["invisible_support"].shape),
        "joint_support_fraction": joint_fraction,
        "collated_placeholder_finite": bool(torch.isfinite(batch["invisible_truth_scaled"]).all()),
        "finite_masked_cfm_loss": float(output["invisible_loss"].detach()),
    }


def _instrument(module, counts: dict[str, int]) -> None:
    module.event_model.visible.register_forward_hook(
        lambda *_: counts.__setitem__("lgatr", counts["lgatr"] + 1)
    )
    if module.event_model.topology_queries is not None:
        module.event_model.topology_queries.register_forward_hook(
            lambda *_: counts.__setitem__("queries", counts["queries"] + 1)
        )
    module.event_model.reconstruction.register_forward_hook(
        lambda *_: counts.__setitem__("assignment", counts["assignment"] + 1)
    )
    if module.event_model.invisible is not None:
        invisible = module.event_model.invisible
        original_loss = invisible.loss
        original_sample = invisible.sample

        def count_loss(*args, _original=original_loss, **kwargs):
            counts["invisible_loss"] += 1
            return _original(*args, **kwargs)

        def count_sample(*args, _original=original_sample, **kwargs):
            counts["invisible_sample"] += 1
            return _original(*args, **kwargs)

        invisible.loss = count_loss
        invisible.sample = count_sample
    if module.event_model.parent_cfm is not None:
        parent_cfm = module.event_model.parent_cfm
        original_loss = parent_cfm.loss

        def count_parent_loss(*args, _original=original_loss, **kwargs):
            counts["parent_cfm_loss"] += 1
            return _original(*args, **kwargs)

        parent_cfm.loss = count_parent_loss


def _dependency_and_gradient_checks() -> dict:
    cases = {}
    gradients = {}
    specifications = (
        (
            "A1",
            "partonic_ttbar2L_2M_q1_cpu.public.yaml",
            ["training.loss_weights.invisible=0"],
        ),
        (
            "N1",
            "partonic_ttbar2L_2M_q1_cpu.public.yaml",
            ["training.loss_weights.assignment=0"],
        ),
        ("P0", "partonic_ttbar2L_2M_p0_cpu.public.yaml", []),
        (
            "P1",
            "partonic_ttbar2L_2M_p1_cpu.public.yaml",
            ["training.loss_weights.assignment=0"],
        ),
        ("J0", "partonic_ttbar2L_2M_j0_synthetic_cpu.public.yaml", []),
        ("J0_soft", "partonic_ttbar2L_2M_j0_soft_synthetic_cpu.public.yaml", []),
        ("P2", "partonic_ttbar2L_2M_p2_synthetic_cpu.public.yaml", []),
    )
    p0_data = None
    p0_batch = None
    p0_config = None
    p0_visible = None
    for name, filename, overrides in specifications:
        config, visible, data, batch, module = _load(filename, overrides)
        if name == "P0":
            p0_data, p0_batch, p0_config, p0_visible = data, batch, config, visible
        module.train()
        counts = {
            "lgatr": 0,
            "queries": 0,
            "assignment": 0,
            "invisible_loss": 0,
            "invisible_sample": 0,
            "parent_cfm_loss": 0,
        }
        _instrument(module, counts)
        output = module(batch, compute_objectives=True)
        for key in ("loss", "partonic_cfm_loss"):
            if key in output and not torch.isfinite(output[key]):
                raise AssertionError(f"{name} produced non-finite {key}")
        if counts["lgatr"] != 1 or counts["queries"] != 1:
            raise AssertionError(f"{name} did not encode/query exactly once: {counts}")
        expected_parent_calls = 0 if name in {"A1", "N1"} else 1
        if counts["parent_cfm_loss"] != expected_parent_calls:
            raise AssertionError(
                f"{name} executed an unexpected number of parent CFM calls: {counts}"
            )
        if name == "A1" and (
            counts["assignment"] != 1
            or counts["invisible_loss"]
            or counts["parent_cfm_loss"]
        ):
            raise AssertionError(f"Assignment-only training executed inactive objectives: {counts}")
        if name == "N1" and (
            counts["assignment"]
            or counts["invisible_loss"] != 1
            or counts["parent_cfm_loss"]
        ):
            raise AssertionError(f"Neutrino-only training executed inactive objectives: {counts}")
        if name == "P0" and (counts["assignment"] or counts["invisible_sample"]):
            raise AssertionError(f"P0 unnecessarily executed upstream prediction: {counts}")
        if name == "P1" and (counts["assignment"] != 1 or counts["invisible_sample"]):
            raise AssertionError(f"P1 must score assignment without neutrino sampling: {counts}")
        if name == "J0" and (counts["assignment"] != 1 or counts["invisible_loss"] != 1 or counts["invisible_sample"]):
            raise AssertionError(f"Uncoupled J0 executed the wrong dependency path: {counts}")
        if name == "J0_soft" and (counts["assignment"] != 1 or counts["invisible_loss"] != 1 or counts["invisible_sample"]):
            raise AssertionError(f"Soft J0 did not reuse soft assignments without sampling: {counts}")
        if name == "P2" and (counts["assignment"] != 1 or counts["invisible_sample"] != 1):
            raise AssertionError(f"P2 did not execute predicted assignment and one neutrino sample: {counts}")
        cases[name] = {"counts": counts, "status": "passed"}

        if name in {"P0", "P1"}:
            module.zero_grad(set_to_none=True)
            output["partonic_cfm_loss"].backward()
            query_norm = _norm(module, "event_model.topology_queries.")
            parent_norm = _norm(module, "event_model.parent_cfm.")
            assignment_norm = _norm(module, "event_model.reconstruction.")
            if query_norm <= 0.0 or parent_norm <= 0.0:
                raise AssertionError(f"{name} parent CFM missed query/model gradients")
            if name == "P0" and assignment_norm != 0.0:
                raise AssertionError("P0 parent loss unexpectedly reached assignment scoring")
            if name == "P1" and assignment_norm <= 0.0:
                raise AssertionError("P1 parent loss did not reach its predicted assignment context")
            gradients[name] = {
                "parent_query_gradient_norm": query_norm,
                "parent_cfm_gradient_norm": parent_norm,
                "assignment_gradient_norm": assignment_norm,
            }

    if p0_data is None or p0_batch is None or p0_config is None or p0_visible is None:
        raise RuntimeError("P0 gradient fixture was not created")
    detached_config = load_experiment(
        EXPERIMENT_ROOT / "partonic_ttbar2L_2M_p0_cpu.public.yaml",
        ["training.parent_cfm_shared_encoder_gradients=assignment_only"],
    )
    detached = _module(detached_config, p0_visible, p0_data)
    detached.train()
    detached_output = detached(p0_batch, compute_objectives=True)
    detached.zero_grad(set_to_none=True)
    detached_output["partonic_cfm_loss"].backward()
    detached_query_norm = _norm(detached, "event_model.topology_queries.")
    detached_parent_norm = _norm(detached, "event_model.parent_cfm.")
    if detached_query_norm != 0.0 or detached_parent_norm <= 0.0:
        raise AssertionError(
            "Configured parent shared-encoder detachment did not isolate CFM gradients"
        )
    gradients["P0_detached"] = {
        "parent_query_gradient_norm": detached_query_norm,
        "parent_cfm_gradient_norm": detached_parent_norm,
        "policy": "parent_cfm_shared_encoder_gradients=assignment_only",
    }
    return {"cases": cases, "parent_cfm_gradients": gradients}


def _unsupported_neutrino_loss_check() -> dict:
    _, _, _, batch, module = _load("partonic_ttbar2L_2M_j0_soft_synthetic_cpu.public.yaml")
    if module.event_model.invisible is None:
        raise AssertionError("Synthetic joint fixture did not construct the neutrino CFM")
    if getattr(module.event_model.invisible.formulation.coupling, "method", None) != "independent":
        raise AssertionError("Unsupported-row invariant check requires independent CFM source coupling")
    count = len(batch["tokens"])
    if count < 2:
        raise AssertionError("Unsupported-target fixture needs at least two events")
    support = torch.ones((count, 2), dtype=torch.bool)
    support[count // 2 :] = False

    def prepared(seed: int) -> dict:
        value = copy.deepcopy(batch)
        value["invisible_support"] = support.clone()
        value["invisible_all_supported"] = False
        torch.manual_seed(seed)
        value["invisible_truth_scaled"][count // 2 :] = torch.randn_like(
            value["invisible_truth_scaled"][count // 2 :]
        )
        return value

    module.train()
    first_batch = prepared(31002)
    second_batch = prepared(31003)
    torch.manual_seed(31001)
    first = module(first_batch, compute_objectives=True)["invisible_loss"]
    torch.manual_seed(31001)
    second = module(second_batch, compute_objectives=True)["invisible_loss"]
    if not torch.isfinite(first) or not torch.allclose(first, second, atol=0.0, rtol=0.0):
        raise AssertionError("Unsupported neutrino placeholders changed the supported-event CFM loss")

    unsupported = copy.deepcopy(batch)
    unsupported["invisible_support"] = torch.zeros((count, 2), dtype=torch.bool)
    unsupported["invisible_all_supported"] = False
    torch.manual_seed(31004)
    unsupported["invisible_truth_scaled"] = torch.randn_like(
        unsupported["invisible_truth_scaled"]
    )
    module.zero_grad(set_to_none=True)
    zero_loss = module(unsupported, compute_objectives=True)["invisible_loss"]
    if not zero_loss.requires_grad or not torch.isfinite(zero_loss) or float(zero_loss.detach()) != 0.0:
        raise AssertionError("All-unsupported neutrino batch was not a finite gradient-safe zero")
    zero_loss.backward()
    gradients = [parameter.grad for parameter in module.event_model.invisible.parameters()]
    if not any(gradient is not None for gradient in gradients):
        raise AssertionError("All-unsupported neutrino loss lost its gradient graph")
    if not all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("All-unsupported neutrino loss produced non-finite gradients")
    return {
        "status": "passed",
        "partial_support_rows": int(support.all(dim=1).sum()),
        "unsupported_placeholder_invariant": True,
        "all_unsupported_loss": float(zero_loss.detach()),
        "all_unsupported_has_grad_fn": bool(zero_loss.requires_grad),
        "device": "cpu",
    }


def run() -> dict:
    torch.set_num_threads(1)
    torch.manual_seed(82017)
    return {
        "device": "cpu",
        "mask_padding_and_role_contract": _mask_and_padding_checks(),
        "no_query_vs_query": _query_mode_comparison(),
        "objective_dependencies_and_parent_gradients": _dependency_and_gradient_checks(),
        "invisible_truth_store_support": _truth_store_partial_support_check(),
        "structured_collator_partial_support": _collated_partial_support_check(),
        "unsupported_neutrino_targets": _unsupported_neutrino_loss_check(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    result = run()
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
