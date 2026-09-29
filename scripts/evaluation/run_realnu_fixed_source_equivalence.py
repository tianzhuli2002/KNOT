#!/usr/bin/env python3
"""Compare accepted and diagnostics implementations on explicit validation fixtures."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
TRAINING_SOURCE = PROJECT_ROOT / "runtime/source/cb2b60445499f78421503d4fccc4a43cae1444fd"
PARENT_ROLES = ("top", "antitop")
NEUTRINO_ROLES = ("top.nu", "antitop.nu")
MODES = ("PCFM_Q", "Q1")
EVENT_COUNT = 64
ATOL = 1.0e-4
RTOL = 5.0e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value) -> dict | list:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _write_fixture(
    path: Path,
    *,
    mode: str,
    ids: np.ndarray,
    batch: dict,
    visible: dict,
    parent_source: np.ndarray,
    parent_time: np.ndarray,
    neutrino_sample_source: np.ndarray | None,
    neutrino_loss_source: np.ndarray | None,
    neutrino_loss_time: np.ndarray | None,
    truth: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as target:
        target.attrs["mode"] = mode
        target.attrs["split"] = "validation"
        target.attrs["event_count"] = EVENT_COUNT
        target.attrs["parent_roles"] = json.dumps(PARENT_ROLES)
        target.attrs["neutrino_roles"] = json.dumps(NEUTRINO_ROLES)
        target.attrs["truth_free_conditioning"] = True
        target.attrs["sealed_test_used"] = False
        target.create_dataset("EVENT_ID", data=ids)
        measured = target.create_group("MEASURED")
        for name in ("physics_vectors", "tokens", "type_ids", "padding_mask", "met_xy"):
            measured.create_dataset(name, data=batch[name].detach().cpu().numpy())
        conditioning = target.create_group("CONDITIONING")
        for name in (
            "pooled", "encoder_pooled", "query_states", "query_context",
            "parent_role_context", "invisible_role_context",
        ):
            value = visible.get(name)
            if value is not None and hasattr(value, "detach"):
                conditioning.create_dataset(name, data=value.detach().cpu().numpy())
        if "parent_role_names" in visible:
            conditioning.attrs["parent_role_names"] = json.dumps(visible["parent_role_names"])
        if "invisible_role_names" in visible:
            conditioning.attrs["invisible_role_names"] = json.dumps(visible["invisible_role_names"])
        fixed = target.create_group("FIXED_SOURCE")
        fixed.create_dataset("parent_x0", data=parent_source)
        fixed.create_dataset("parent_t", data=parent_time)
        if neutrino_sample_source is not None:
            fixed.create_dataset("neutrino_sample_x0", data=neutrino_sample_source)
            fixed.create_dataset("neutrino_loss_x0", data=neutrino_loss_source)
            fixed.create_dataset("neutrino_loss_t", data=neutrino_loss_time)
        truth_group = target.create_group("TRUTH_SIDE_CAR")
        for key, values in truth.items():
            parent, name = key.split("/", 1)
            truth_group.require_group(parent).create_dataset(name, data=values)
        if "structured" in batch:
            masks = target.create_group("MEASURED/STRUCTURED_MASKS")
            for node, values in batch["structured"].get("pools", {}).items():
                node_group = masks.require_group(node)
                for key, value in values.items():
                    if hasattr(value, "dtype") and value.dtype == __import__("torch").bool:
                        node_group.create_dataset(key, data=value.detach().cpu().numpy())


def _read_tree(group, prefix: str = "") -> dict[str, np.ndarray]:
    result = {}
    for name, value in group.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Group):
            result.update(_read_tree(value, key))
        else:
            result[key] = np.asarray(value[:])
    return result


def _load_fixture(path: Path, mode: str) -> dict:
    with h5py.File(path, "r") as source:
        if source.attrs.get("mode") != mode or source.attrs.get("split") != "validation":
            raise ValueError(f"Fixed fixture has the wrong mode/split: {path}")
        if not bool(source.attrs.get("truth_free_conditioning", False)):
            raise ValueError(f"Fixture does not separate truth from conditioning: {path}")
        if bool(source.attrs.get("sealed_test_used", True)):
            raise ValueError(f"Fixture violates sealed_test_used=false: {path}")
        if _json(source.attrs["parent_roles"]) != list(PARENT_ROLES):
            raise ValueError("Fixture parent roles are not [top, antitop]")
        if _json(source.attrs["neutrino_roles"]) != list(NEUTRINO_ROLES):
            raise ValueError("Fixture neutrino roles are not [top.nu, antitop.nu]")
        result = {
            "ids": np.asarray(source["EVENT_ID"][:], dtype=np.int64),
            "measured": {name: np.asarray(source[f"MEASURED/{name}"][:]) for name in source["MEASURED"] if name != "STRUCTURED_MASKS"},
            "parent_source": np.asarray(source["FIXED_SOURCE/parent_x0"][:]),
            "parent_time": np.asarray(source["FIXED_SOURCE/parent_t"][:]),
            "truth": _read_tree(source["TRUTH_SIDE_CAR"]),
        }
        fixed = source["FIXED_SOURCE"]
        result["neutrino_sample_source"] = (
            np.asarray(fixed["neutrino_sample_x0"][:])
            if "neutrino_sample_x0" in fixed else None
        )
        result["neutrino_loss_source"] = (
            np.asarray(fixed["neutrino_loss_x0"][:])
            if "neutrino_loss_x0" in fixed else None
        )
        result["neutrino_loss_time"] = (
            np.asarray(fixed["neutrino_loss_t"][:])
            if "neutrino_loss_t" in fixed else None
        )
        return result


def _truth_join(event_table: Path, ids: np.ndarray, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    with h5py.File(event_table, "r") as source:
        if str(source.attrs.get("sealed_test_used", "")).lower() != "false":
            raise ValueError("Accepted R2 event table does not establish sealed_test_used=false")
        source_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        order = np.argsort(source_ids, kind="stable")
        positions = np.searchsorted(source_ids[order], ids)
        if np.any(positions >= len(order)) or not np.array_equal(source_ids[order[positions]], ids):
            raise ValueError("Fixed validation EVENT_ID join failed")
        result = {}
        for key in keys:
            result[key] = np.asarray(source[key][order[positions]])
        return result


def _record_arrays(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _worker(args) -> None:
    import inspect
    import torch
    from torch.utils.data import Subset

    from joint_event_reconstruction.cli.run import build_data, load_trained, load_visible_config
    from joint_event_reconstruction.flows.continuous import linear_path
    from joint_event_reconstruction.flows.solvers import integrate
    from joint_event_reconstruction.models.partonic import (
        four_vectors_from_partonic_features,
        partonic_features,
        parent_pair_observables,
    )

    source_root = Path(args.source_root).resolve()
    project_root = Path(args.project_root).resolve()
    production_root = Path(args.production_root).resolve()
    event_table = production_root / "evaluation_r2/core/event_metrics.h5"
    fixture_path = Path(args.fixture).resolve()
    output_path = Path(args.output).resolve()
    sys.path.insert(0, str(source_root / "scripts/production"))
    import run_realnu_production as runner

    torch.set_num_threads(4)
    config, _, _ = runner.load_and_validate_config(
        args.mode,
        runner.MODE_CONFIGS[args.mode],
        production_root / args.mode,
    )
    visible_config = load_visible_config(config)
    visible_config.general.mode = "predict"
    visible_config.dataset.val_batch_size = EVENT_COUNT
    visible_config.dataset.val_num_workers = 2
    visible_config.dataset.pin_memory = False
    visible_config.dataset.persistent_workers = True
    visible_config.dataset.prefetch_factor = 2
    data = build_data(config, visible_config, include_truth=False, device="cpu")
    data.setup("predict")

    if fixture_path.exists():
        fixture = _load_fixture(fixture_path, args.mode)
        expected_ids = fixture["ids"]
    else:
        with h5py.File(event_table, "r") as source:
            if source.attrs.get("schema") != "realnu_event_metrics_v1":
                raise ValueError("Unexpected R2 event table schema")
            if str(source.attrs.get("sealed_test_used", "")).lower() != "false":
                raise ValueError("R2 event table violates sealed_test_used=false")
            if _json(source.attrs["parent_role_order"]) != list(PARENT_ROLES):
                raise ValueError("R2 parent role order is invalid")
            if _json(source.attrs["neutrino_role_order"]) != list(NEUTRINO_ROLES):
                raise ValueError("R2 neutrino role order is invalid")
            expected_ids = np.asarray(source["EVENT_ID"][:EVENT_COUNT], dtype=np.int64)
        fixture = None

    if len(expected_ids) != EVENT_COUNT or len(np.unique(expected_ids)) != EVENT_COUNT:
        raise ValueError("Fixed-source fixture must contain 64 unique validation EVENT_IDs")
    data.val_dataset = Subset(data.val_dataset, range(EVENT_COUNT))
    batch = next(iter(data.val_dataloader()))
    ids = batch["event_index"].detach().cpu().numpy().astype(np.int64, copy=False)
    if not np.array_equal(ids, expected_ids):
        raise ValueError("Measured validation order differs from fixed fixture EVENT_ID order")
    if fixture is not None:
        for name, expected in fixture["measured"].items():
            observed = batch[name].detach().cpu().numpy()
            if not np.array_equal(observed, expected):
                raise ValueError(f"Measured fixture tensor changed: {name}")

    record = json.loads((production_root / args.mode / "production_training.json").read_text())
    checkpoint = Path(record["checkpoint"]).resolve()
    module = load_trained(config, visible_config, data, str(checkpoint)).cpu().eval()
    model = module.event_model
    with torch.no_grad():
        visible = model.encode_visible(batch)

    truth_keys = ("truth/parents", "support/parent_roles")
    if args.mode == "Q1":
        truth_keys += ("truth/neutrinos", "support/neutrino_roles")
    truth = _truth_join(event_table, ids, truth_keys)
    parent_source = (
        fixture["parent_source"] if fixture is not None
        else torch.randn((EVENT_COUNT, 1, 8), generator=torch.Generator().manual_seed(27182)).numpy()
    )
    parent_time = (
        fixture["parent_time"] if fixture is not None
        else torch.linspace(0.025, 0.975, EVENT_COUNT, dtype=torch.float32).numpy()
    )
    neutrino_sample_source = None
    neutrino_loss_source = None
    neutrino_loss_time = None
    if args.mode == "Q1":
        invisible = model.invisible
        target_momenta = torch.as_tensor(truth["truth/neutrinos"], dtype=torch.float32)
        target_scaled = invisible.encode_target(
            target_momenta, batch["met_xy"], data.invisible_scaler
        )
        if fixture is None:
            sample_generator = torch.Generator().manual_seed(314159)
            neutrino_sample_source = torch.randn(
                (EVENT_COUNT, 1, *target_scaled.shape[1:]),
                generator=sample_generator,
                dtype=target_scaled.dtype,
            ).numpy()
            torch.manual_seed(161803)
            neutrino_loss_source = torch.randn(
                target_scaled.shape, dtype=target_scaled.dtype
            ).numpy()
            neutrino_loss_time = torch.rand(
                EVENT_COUNT, dtype=target_scaled.dtype
            ).numpy()
        else:
            neutrino_sample_source = fixture["neutrino_sample_source"]
            neutrino_loss_source = fixture["neutrino_loss_source"]
            neutrino_loss_time = fixture["neutrino_loss_time"]

    if fixture is None:
        _write_fixture(
            fixture_path,
            mode=args.mode,
            ids=ids,
            batch=batch,
            visible=visible,
            parent_source=parent_source,
            parent_time=parent_time,
            neutrino_sample_source=neutrino_sample_source,
            neutrino_loss_source=neutrino_loss_source,
            neutrino_loss_time=neutrino_loss_time,
            truth=truth,
        )

    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "mode": args.mode,
        "source_root": str(source_root),
        "source_commit": subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "event_count": EVENT_COUNT,
        "event_id_order": ids.tolist(),
        "parent_roles": list(PARENT_ROLES),
        "neutrino_roles": list(NEUTRINO_ROLES) if args.mode == "Q1" else [],
        "truth_free_conditioning": True,
        "sealed_test_used": False,
        "device": "cpu",
    }
    for name in ("query_states", "query_context", "parent_role_context", "invisible_role_context"):
        value = visible.get(name)
        if torch.is_tensor(value):
            arrays[f"visible_{name}"] = value.detach().cpu().numpy()

    if args.mode == "PCFM_Q":
        parent_cfm = model.parent_cfm
        query_states = visible.get("parent_role_context")
        if query_states is None or tuple(visible.get("parent_role_names", ())) != PARENT_ROLES:
            raise ValueError("PCFM_Q fixed fixture lacks ordered parent query states")
        measured = torch.cat((visible["pooled"].unsqueeze(1), visible["tokens"]), dim=1)
        measured_mask = torch.cat((
            torch.zeros((EVENT_COUNT, 1), dtype=torch.bool), visible["padding_mask"]
        ), dim=1)
        role_embedding = parent_cfm.parent_role_embedding.unsqueeze(0)
        parent_context = query_states + role_embedding + parent_cfm.condition_kind_embedding[0]
        truth_parent = torch.as_tensor(truth["truth/parents"], dtype=torch.float32)
        truth_support = torch.as_tensor(truth["support/parent_roles"], dtype=torch.bool).all(dim=1)
        target = partonic_features(truth_parent, validate_physics=False)
        target_scaled = ((target - parent_cfm.target_mean) / parent_cfm.target_std).reshape(EVENT_COUNT, 8)
        x0 = torch.as_tensor(parent_source[:, 0], dtype=torch.float32)
        time = torch.as_tensor(parent_time, dtype=torch.float32)
        state, target_velocity = linear_path(x0, target_scaled, time)
        prediction = parent_cfm.flow.velocity(
            state, time, measured, measured_mask, parent_context
        )
        active = parent_cfm.flow.active_mask.to(dtype=target_scaled.dtype)
        per_event = ((prediction - target_velocity).square() * active).reshape(EVENT_COUNT, -1).sum(dim=1) / active.sum()
        loss = (per_event * truth_support.to(per_event.dtype)).sum() / truth_support.sum().clamp_min(1)
        module.zero_grad(set_to_none=True)
        loss.backward()
        arrays.update({
            "parent_target_encoded": target.detach().numpy(),
            "parent_target_scaled": target_scaled.detach().numpy(),
            "parent_x0": x0.numpy(),
            "parent_t": time.numpy(),
            "parent_xt": state.detach().numpy(),
            "parent_ut": target_velocity.detach().numpy(),
            "parent_velocity": prediction.detach().numpy(),
            "parent_per_event_loss": per_event.detach().numpy(),
            "parent_support": truth_support.numpy(),
        })
        for name, parameter in (
            ("parent_input_projection_gradient", parent_cfm.flow.denoiser.input_projection.weight),
            ("parent_role_embedding_gradient", parent_cfm.parent_role_embedding),
        ):
            if parameter.grad is None:
                raise RuntimeError(f"Missing parent-CFM gradient for {name}")
            arrays[name] = parameter.grad.detach().cpu().numpy()
        with torch.no_grad():
            sample_kwargs = {
                "samples": 1, "steps": 8, "solver": "heun",
            }
            if "source_state" in inspect.signature(parent_cfm.sample).parameters:
                sample_kwargs["source_state"] = torch.as_tensor(parent_source, dtype=torch.float32)
            else:
                sample_kwargs["generator"] = torch.Generator().manual_seed(27182)
            sampled = parent_cfm.sample(measured, measured_mask, query_states, **sample_kwargs)
        arrays["parent_sample"] = sampled["parents"].detach().cpu().numpy()
        arrays["parent_pair"] = sampled["pair"]["ttbar"].detach().cpu().numpy()
        metadata["parent_solver"] = "heun"
        metadata["parent_steps"] = 8
        metadata["parent_nfe"] = int(sampled["nfe"])
    else:
        assignment = model.reconstruction(
            visible["tokens"], visible["pooled"], batch["structured"],
            node_context=visible.get("node_context"), calculate_loss=False,
        )
        for group_name in ("assignment_logits", "assignment_probabilities"):
            values = assignment[group_name]
            for role, tensor in values.items():
                arrays[f"{group_name}_{role}"] = tensor.detach().cpu().numpy()
        role_parent = model.invisible.role_parent_context(
            assignment,
            batch["structured"],
            visible["pooled"],
            assignment.get("node_context", visible.get("node_context")),
            mode=model.invisible.assignment_mode,
            query_role_context=visible.get("invisible_role_context"),
            query_role_names=tuple(visible.get("invisible_role_names", ())),
        )
        parent_context = model.invisible.coordinates.parent_context(role_parent, visible["pooled"])
        target_momenta = torch.as_tensor(truth["truth/neutrinos"], dtype=torch.float32)
        target_scaled = model.invisible.encode_target(target_momenta, batch["met_xy"], data.invisible_scaler)
        support = torch.as_tensor(truth["support/neutrino_roles"], dtype=torch.bool).all(dim=1)
        sample_source = torch.as_tensor(
            neutrino_sample_source, dtype=target_scaled.dtype
        )
        active = model.invisible.formulation.active_mask.to(dtype=target_scaled.dtype)
        torch.manual_seed(161803)
        generated_raw_source = torch.randn(target_scaled.shape, dtype=target_scaled.dtype)
        generated_time = torch.rand(EVENT_COUNT, dtype=target_scaled.dtype)
        torch.manual_seed(161803)
        loss_kwargs = {}
        if "source_state" in inspect.signature(model.invisible.loss).parameters:
            loss_kwargs = {
                "source_state": torch.as_tensor(
                    neutrino_loss_source, dtype=target_scaled.dtype
                ),
                "flow_time": torch.as_tensor(
                    neutrino_loss_time, dtype=target_scaled.dtype
                ),
            }
        invisible_loss = model.invisible.loss(
            target_scaled,
            visible,
            parent_context,
            role_context=role_parent,
            event_mask=support,
            **loss_kwargs,
        )
        arrays.update({
            "neutrino_target_scaled": target_scaled.detach().numpy(),
            "neutrino_x0": (
                loss_kwargs["source_state"] * active
                if loss_kwargs else generated_raw_source * active
            ).detach().numpy(),
            "neutrino_t": (
                loss_kwargs["flow_time"]
                if loss_kwargs else invisible_loss["time"]
            ).detach().numpy(),
            "neutrino_xt": invisible_loss["state"].detach().numpy(),
            "neutrino_ut": invisible_loss["target_velocity"].detach().numpy(),
            "neutrino_velocity": invisible_loss["prediction"].detach().numpy(),
            "neutrino_per_event_loss": invisible_loss["per_event_loss"].detach().numpy(),
            "neutrino_support": support.detach().numpy(),
        })
        module.zero_grad(set_to_none=True)
        invisible_loss["loss"].backward()
        selected_gradients = 0
        for name, parameter in model.invisible.formulation.named_parameters():
            if parameter.grad is not None and parameter.grad.numel():
                arrays[f"neutrino_gradient_{selected_gradients}"] = parameter.grad.detach().cpu().numpy()
                metadata[f"neutrino_gradient_{selected_gradients}_parameter"] = name
                selected_gradients += 1
                if selected_gradients == 2:
                    break
        with torch.no_grad():
            sample_kwargs = {
                "samples": 1,
                "budget": min(8, int(model.invisible.inference_budget)),
                "sampler": str(model.invisible.sampler),
                "generator": torch.Generator().manual_seed(314159),
            }
            if "source_state" in inspect.signature(model.invisible.sample).parameters:
                sample_kwargs["source_state"] = sample_source
            generated, nfe = model.invisible.sample(visible, parent_context, **sample_kwargs)
            physical = model.invisible.decode_state(generated, batch["met_xy"], data.invisible_scaler)
        arrays["assignment_candidate_masks"] = np.concatenate([
            np.asarray(value.detach().cpu().numpy()).reshape(EVENT_COUNT, -1)
            for node in batch["structured"]["pools"].values()
            for key, value in node.items()
            if torch.is_tensor(value) and value.dtype == torch.bool
        ], axis=1) if any(
            torch.is_tensor(value) and value.dtype == torch.bool
            for node in batch["structured"]["pools"].values() for value in node.values()
        ) else np.empty((EVENT_COUNT, 0), dtype=bool)
        arrays["neutrino_sample_scaled"] = generated.detach().cpu().numpy()
        arrays["neutrino_sample_physical"] = physical.detach().cpu().numpy()
        metadata["neutrino_solver"] = str(model.invisible.sampler)
        metadata["neutrino_steps"] = int(sample_kwargs["budget"])
        metadata["neutrino_nfe"] = int(nfe)
    _record_arrays(output_path, arrays, metadata)


def _compare(reference: Path, observed: Path) -> dict:
    result = {"reference": str(reference), "observed": str(observed), "arrays": {}, "status": "passed"}
    with np.load(reference, allow_pickle=False) as ref, np.load(observed, allow_pickle=False) as new:
        if set(ref.files) != set(new.files):
            raise ValueError(f"Output keys differ: reference-only={sorted(set(ref.files)-set(new.files))}, new-only={sorted(set(new.files)-set(ref.files))}")
        for key in ref.files:
            left = np.asarray(ref[key])
            right = np.asarray(new[key])
            if left.shape != right.shape:
                raise ValueError(f"{key}: shape differs: {left.shape} vs {right.shape}")
            finite_left, finite_right = np.isfinite(left), np.isfinite(right)
            if not np.array_equal(finite_left, finite_right):
                raise ValueError(f"{key}: finite masks differ")
            difference = left[finite_left].astype(np.float64) - right[finite_right].astype(np.float64)
            max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
            rms = float(np.sqrt(np.mean(difference * difference))) if difference.size else 0.0
            relative = float(np.linalg.norm(difference) / max(np.linalg.norm(left[finite_left].astype(np.float64)), 1e-15)) if difference.size else 0.0
            passed = bool(np.allclose(left, right, atol=ATOL, rtol=RTOL, equal_nan=True))
            result["arrays"][key] = {"max_abs": max_abs, "rms": rms, "relative_l2": relative, "passed": passed}
            if not passed:
                result["status"] = "failed"
    return result


def run(args) -> dict:
    production_root = args.production_root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    source_roots = {"reference": args.reference_source.expanduser().resolve(), "diagnostics": args.diagnostics_source.expanduser().resolve()}
    output_root = args.output_root.expanduser().resolve()
    expected_output = production_root / "evaluation_v3_diagnostics"
    if not output_root.is_relative_to(expected_output):
        raise ValueError(f"Equivalence outputs must stay under {expected_output}")
    if output_root == production_root / "evaluation_r2" or production_root / "evaluation_r2" in output_root.parents:
        raise ValueError("Equivalence output may not be written under evaluation_r2")
    fixture_root = output_root / "fixtures"
    result_root = output_root / "implementations"
    fixture_root.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)
    records = []
    modes = (args.mode,) if args.mode else MODES
    for mode in modes:
        fixture = fixture_root / f"{mode}_validation64.h5"
        for version, source_root in source_roots.items():
            if version == "reference" and fixture.exists():
                fixture.unlink()
            output = result_root / f"{mode}_{version}.npz"
            env = os.environ.copy()
            source_path = str(source_root)
            env.update({
                "JER_REALNU_SOURCE_ROOT": source_path,
                "JER_REALNU_PROJECT_ROOT": str(project_root),
                "JER_SOURCE_ROOT": source_path,
                "JER_SOURCE_SHA": subprocess.check_output(["git", "-C", source_path, "rev-parse", "HEAD"], text=True).strip(),
                "PYTHONPATH": os.pathsep.join((str(source_root / "src"), str(source_root / "scripts/production"), env.get("PYTHONPATH", ""))),
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            })
            command = [
                sys.executable, str(Path(__file__).resolve()), "--worker",
                "--mode", mode, "--source-root", source_path,
                "--project-root", str(project_root), "--production-root", str(production_root),
                "--fixture", str(fixture), "--output", str(output),
            ]
            subprocess.run(command, env=env, check=True)
        comparison = _compare(result_root / f"{mode}_reference.npz", result_root / f"{mode}_diagnostics.npz")
        comparison["mode"] = mode
        comparison["events"] = EVENT_COUNT
        records.append(comparison)
    completed_modes = {record["mode"] for record in records}
    for mode in MODES:
        if mode in completed_modes:
            continue
        reference = result_root / f"{mode}_reference.npz"
        observed = result_root / f"{mode}_diagnostics.npz"
        if reference.is_file() and observed.is_file():
            comparison = _compare(reference, observed)
            comparison["mode"] = mode
            comparison["events"] = EVENT_COUNT
            records.append(comparison)
    records.sort(key=lambda record: MODES.index(record["mode"]))
    summary = {
        "status": "passed" if all(item["status"] == "passed" for item in records) else "failed",
        "gate": "REALNU_FIXED_SOURCE_IMPLEMENTATION_EQUIVALENCE",
        "event_count_per_mode": EVENT_COUNT,
        "parent_roles": list(PARENT_ROLES),
        "neutrino_roles": list(NEUTRINO_ROLES),
        "truth_free_conditioning": True,
        "sealed_test_used": False,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "modes": records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / "equivalence_summary.json"
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=MODES, help="Run one mode instead of both equivalence gates")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--reference-source", type=Path, default=TRAINING_SOURCE)
    parser.add_argument("--diagnostics-source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v3_diagnostics/audit/deterministic_equivalence"),
    )
    args = parser.parse_args()
    if args.worker:
        _worker(args)
        return
    summary = run(args)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
