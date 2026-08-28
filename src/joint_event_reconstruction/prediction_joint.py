"""Prediction writer for sampled joint assignment-invisible Flow Matching hypotheses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from joint_event_reconstruction.topology.assignments import AssignmentSpace


JOINT_PREDICTION_SCHEMA = "joint_event_reconstruction_joint_flow_predictions_v1"


def _create(group: h5py.Group, name: str, tail: tuple[int, ...], dtype) -> h5py.Dataset:
    return group.create_dataset(name, (0,) + tail, maxshape=(None,) + tail, dtype=dtype, chunks=True)


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    old = len(dataset)
    dataset.resize(old + len(values), axis=0)
    dataset[old:] = values


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def predict_joint_flow_hdf5(
    module,
    data,
    scaler,
    output_path: str | Path,
    *,
    split: str,
    posterior_samples: int,
    steps: int,
    device: str,
    checkpoint_path: str = "",
) -> None:
    if posterior_samples <= 0 or steps <= 0:
        raise ValueError("Joint posterior samples and flow steps must be positive")
    if module.event_model.model_mode != "joint_flow":
        raise ValueError("Joint-flow prediction requires a joint_flow checkpoint")
    if scaler is None:
        raise ValueError("Joint-flow prediction requires the trained invisible scaler")

    loader = {
        "train": data.train_dataloader,
        "validation": data.val_dataloader,
        "test": data.test_dataloader,
    }[split]()
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    module = module.to(device).eval()
    datasets = None
    with h5py.File(temporary, "w") as destination, torch.inference_mode():
        source_path = Path(data.h5_path).expanduser().resolve()
        source_stat = source_path.stat()
        destination.attrs.update({
            "schema_version": JOINT_PREDICTION_SCHEMA,
            "model_mode": "joint_flow",
            "topology_signature": module.spec.signature,
            "architecture_signature": module.hparams.architecture_signature,
            "checkpoint": str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else "untrained",
            "split": split,
            "posterior_samples": posterior_samples,
            "flow_steps": steps,
            "actual_nfe": steps,
            "recurrent_state": bool(module.event_model.joint_flow.recurrent_state),
            "invisible_roles": json.dumps([role.key for role in module.spec.invisible_roles]),
            "scaler": json.dumps(scaler.as_dict()),
            "assignment_probabilities_are_exact_likelihoods": False,
            "source_data": str(source_path),
            "source_data_size_bytes": source_stat.st_size,
            "source_data_mtime_ns": source_stat.st_mtime_ns,
        })

        for batch in loader:
            batch = _move(batch, device)
            visible = module.event_model.visible(
                batch["tokens"], batch["type_ids"], batch["padding_mask"]
            )
            generated = module.event_model.joint_flow.sample(
                visible,
                batch["structured"],
                samples=posterior_samples,
                steps=steps,
            )
            space = AssignmentSpace.from_structured(batch["structured"])
            states = generated["assignment_state"]
            flat = states.reshape(-1)
            root_indices = space.root_candidate_indices[flat].reshape(
                len(states), posterior_samples, len(space.roots)
            )
            node_indices = space.node_candidate_indices[flat].reshape(
                len(states), posterior_samples, len(space.nodes)
            )
            invisible = scaler.inverse(generated["invisible"])
            truth_hit = states == space.truth_state.unsqueeze(1)

            jet_type = data.schema.source_type_ids.get("JET")
            if jet_type is None:
                raise RuntimeError("Prediction schema has no JET source for Njet diagnostics")
            njet = (
                (batch["type_ids"] == jet_type) & ~batch["padding_mask"]
            ).sum(dim=1)

            if datasets is None:
                hypotheses = destination.create_group("HYPOTHESES")
                truth = destination.create_group("TRUTH")
                observed = destination.create_group("OBSERVED")
                datasets = {
                    "event": _create(destination, "EVENT_ID", (), "<i8"),
                    "state": _create(hypotheses, "assignment_state", (posterior_samples,), "<i4"),
                    "roots": _create(hypotheses, "root_candidate_indices", (posterior_samples, len(space.roots)), "<i4"),
                    "nodes": _create(hypotheses, "node_candidate_indices", (posterior_samples, len(space.nodes)), "<i4"),
                    "invisible": _create(hypotheses, "invisible", (posterior_samples, len(module.spec.invisible_roles), 3), "<f4"),
                    "truth_state": _create(truth, "assignment_state", (), "<i4"),
                    "truth_hit": _create(truth, "hypothesis_assignment", (posterior_samples,), "?"),
                    "nu_truth": _create(truth, "NEUTRINO", (len(module.spec.invisible_roles), 3), "<f4"),
                    "njet": _create(observed, "njet", (), "<i2"),
                }
                destination.attrs["root_order"] = json.dumps(list(space.roots))
                destination.attrs["node_order"] = json.dumps(list(space.nodes))
                destination.attrs["assignment_state_count"] = space.state_count

            _append(datasets["event"], batch["event_index"].cpu().numpy().astype(np.int64))
            _append(datasets["state"], states.cpu().numpy().astype(np.int32))
            _append(datasets["roots"], root_indices.cpu().numpy().astype(np.int32))
            _append(datasets["nodes"], node_indices.cpu().numpy().astype(np.int32))
            _append(datasets["invisible"], invisible.cpu().numpy().astype(np.float32))
            _append(datasets["truth_state"], space.truth_state.cpu().numpy().astype(np.int32))
            _append(datasets["truth_hit"], truth_hit.cpu().numpy())
            _append(datasets["nu_truth"], batch["invisible_truth"].cpu().numpy().astype(np.float32))
            _append(datasets["njet"], njet.cpu().numpy().astype(np.int16))

    validate_joint_prediction(temporary)
    os.replace(temporary, target)


def validate_joint_prediction(path: str | Path) -> None:
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema_version") != JOINT_PREDICTION_SCHEMA:
            raise ValueError(f"Invalid joint-flow prediction schema in {path}")
        count = len(source["EVENT_ID"])
        if count == 0 or len(np.unique(source["EVENT_ID"][:])) != count:
            raise ValueError(f"Prediction output {path} is empty or has duplicate EVENT_ID")
        for name in (
            "HYPOTHESES/assignment_state",
            "HYPOTHESES/root_candidate_indices",
            "HYPOTHESES/node_candidate_indices",
            "HYPOTHESES/invisible",
            "TRUTH/assignment_state",
            "TRUTH/NEUTRINO",
        ):
            if len(source[name]) != count:
                raise ValueError(f"Prediction output has inconsistent {name} length")
        if not np.isfinite(source["HYPOTHESES/invisible"][:]).all():
            raise ValueError("Joint-flow invisible hypotheses contain non-finite values")
