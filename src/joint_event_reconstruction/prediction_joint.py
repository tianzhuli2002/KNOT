"""Prediction writer for sampled joint assignment-invisible Flow Matching hypotheses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from joint_event_reconstruction.assignment_semantics import (
    NULL_INCLUDING_SUPPORT,
    hdf5_support_attributes,
)
from joint_event_reconstruction.physics import compose_assignment_state
from joint_event_reconstruction.prediction_schema import (
    JOINT_PREDICTION_SCHEMA,
    validate_joint_prediction,
)
from joint_event_reconstruction.topology.assignments import AssignmentSpace


def _create(group: h5py.Group, name: str, tail: tuple[int, ...], dtype) -> h5py.Dataset:
    return group.create_dataset(
        name, (0,) + tail, maxshape=(None,) + tail, dtype=dtype, chunks=True
    )


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


def _physics_hypotheses(
    module,
    batch,
    space,
    node_indices,
    invisible,
    physical_invisible=None,
    truth_physical=None,
):
    if "physics_vectors" not in batch:
        return None, None, None

    values = {node: [] for node in module.spec.dag_order}
    for event in range(len(node_indices)):
        event_values = {node: [] for node in module.spec.dag_order}
        for sample in range(node_indices.shape[1]):
            invisible_map = {
                role.key: invisible[event, sample, role_index]
                for role_index, role in enumerate(module.spec.invisible_roles)
            }
            physical_map = None
            if physical_invisible is not None:
                physical_map = {
                    role.key: physical_invisible[event, sample, role_index]
                    for role_index, role in enumerate(module.spec.invisible_roles)
                }
            composed = compose_assignment_state(
                module.spec,
                batch["structured"]["pools"],
                batch["physics_vectors"][event],
                node_indices[event, sample],
                invisible_map,
                invisible_four_vectors=physical_map,
            )
            for node in module.spec.dag_order:
                event_values[node].append(composed[node])
        for node in module.spec.dag_order:
            values[node].append(torch.stack(event_values[node]))
    composites = {
        node: torch.stack(event_values).cpu().numpy().astype(np.float32)
        for node, event_values in values.items()
    }

    truth_supported = torch.stack(
        [
            batch["structured"]["pools"][role.node]["truth_index"] >= 0
            for role in module.spec.invisible_roles
        ],
        dim=1,
    ).all(dim=1)
    truth_indices = space.node_candidate_indices[space.truth_state]
    truth_values = {node: [] for node in module.spec.dag_order}
    for event in range(len(truth_indices)):
        if not bool(truth_supported[event]):
            for node in module.spec.dag_order:
                truth_values[node].append(
                    torch.full(
                        (4,),
                        float("nan"),
                        dtype=invisible.dtype,
                        device=invisible.device,
                    )
                )
            continue
        invisible_map = {
            role.key: batch["invisible_truth"][event, role_index]
            for role_index, role in enumerate(module.spec.invisible_roles)
        }
        truth_map = None
        if truth_physical is not None:
            truth_map = {
                role.key: truth_physical[event, role_index]
                for role_index, role in enumerate(module.spec.invisible_roles)
            }
        composed = compose_assignment_state(
            module.spec,
            batch["structured"]["pools"],
            batch["physics_vectors"][event],
            truth_indices[event],
            invisible_map,
            invisible_four_vectors=truth_map,
        )
        for node in module.spec.dag_order:
            truth_values[node].append(composed[node])
    truth_composites = {
        node: torch.stack(event_values).cpu().numpy().astype(np.float32)
        for node, event_values in truth_values.items()
    }

    observed_met = None
    met_type = module.hparams.source_type_ids.get("MET")
    if met_type is not None:
        met_mask = batch["type_ids"] == int(met_type)
        if not (met_mask.sum(dim=1) == 1).all():
            raise RuntimeError(
                "Joint prediction physics output requires exactly one MET token"
            )
        observed_met = (
            batch["physics_vectors"][met_mask]
            .reshape(len(batch["tokens"]), 4)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return composites, truth_composites, observed_met


def _met_exact_truth_projection(module, batch):
    """Build the explicit MET-exact projected target for provenance/evaluation."""
    coordinates = module.event_model.invisible.coordinates
    if coordinates.coordinate_type != "met_exact":
        return None
    truth = batch["invisible_truth"]
    state = coordinates.encode(truth, batch["met_xy"])
    projected = coordinates.decode(state, batch["met_xy"])
    diagnostics = coordinates.projection_diagnostics(truth, batch["met_xy"])
    if diagnostics is None:
        raise RuntimeError("MET-exact coordinates did not provide projection diagnostics")
    return {
        "projected": projected,
        "px_shift": diagnostics["truth_projected_px_shift"],
        "py_shift": diagnostics["truth_projected_py_shift"],
        "per_role_magnitude": diagnostics["per_role_transverse_projection_magnitude"],
    }


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
    prediction_seed: int,
    checkpoint_path: str = "",
    sample_chunk_size: int = 1,
) -> None:
    if posterior_samples <= 0 or steps <= 0 or sample_chunk_size <= 0:
        raise ValueError(
            "Joint posterior samples, flow steps and sample_chunk_size must be positive"
        )
    if module.event_model.model_mode != "joint_flow":
        raise ValueError("Joint-flow prediction requires a joint_flow checkpoint")
    if scaler is None:
        raise ValueError("Joint-flow prediction requires the trained invisible scaler")

    sample_chunk_size = min(sample_chunk_size, posterior_samples)
    chunk_count = (posterior_samples + sample_chunk_size - 1) // sample_chunk_size
    loader = {
        "train": data.train_dataloader,
        "validation": data.val_dataloader,
        "test": data.test_dataloader,
    }[split]()
    iterator = iter(loader)
    sampling_generator = torch.Generator(device=device).manual_seed(int(prediction_seed))
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
        destination.attrs.update(
            {
                "schema_version": JOINT_PREDICTION_SCHEMA,
                "model_mode": "joint_flow",
                "assignment_formulation": "flow_matching",
                **hdf5_support_attributes(NULL_INCLUDING_SUPPORT),
                "invisible_formulation": "flow_matching",
                "topology_signature": module.spec.signature,
                "architecture_signature_version": int(
                    module.hparams.architecture_signature_version
                ),
                "architecture_signature": module.hparams.architecture_signature,
                "checkpoint": (
                    str(Path(checkpoint_path).expanduser().resolve())
                    if checkpoint_path
                    else "untrained"
                ),
                "split": split,
                "prediction_seed": int(prediction_seed),
                "posterior_samples": posterior_samples,
                "flow_steps": steps,
                "actual_nfe": steps,
                "nfe_per_sample": steps,
                "sample_chunk_size": sample_chunk_size,
                "posterior_chunk_count": chunk_count,
                "batched_network_evaluations_per_event": steps * chunk_count,
                "sample_equivalent_nfe_per_event": steps * posterior_samples,
                "recurrent_state": bool(module.event_model.joint_flow.recurrent_state),
                "invisible_roles": json.dumps(
                    [role.key for role in module.spec.invisible_roles]
                ),
                "invisible_coordinates": module.hparams.invisible_coordinates,
                "invisible_coordinate_metadata": json.dumps(
                    module.hparams.invisible_coordinate_metadata
                ),
                "invisible_mass_shell": json.dumps(
                    getattr(module.hparams, "invisible_mass_shell", None)
                ),
                "invisible_generator_backbone": json.dumps(
                    getattr(module.hparams, "invisible_generator_backbone", None)
                ),
                "flow_coupling": json.dumps(module.hparams.flow_coupling),
                "physics_vector_representation": "E_px_py_pz_GeV_v1",
                "candidate_representation": module.hparams.candidate_representation,
                "assignment_state_representation": module.hparams.assignment_state_representation,
                "visible_encoder": module.hparams.visible_encoder,
                "visible_encoder_config": json.dumps(
                    module.hparams.visible_encoder_config
                ),
                "invisible_role_count": int(module.hparams.invisible_role_count),
                "scaler": json.dumps(scaler.as_dict()),
                "assignment_probabilities_are_exact_likelihoods": False,
                "source_data": str(source_path),
                "source_data_size_bytes": source_stat.st_size,
                "source_data_mtime_ns": source_stat.st_mtime_ns,
            }
        )

        for batch in iterator:
            batch = _move(batch, device)
            visible = module.event_model.encode_visible(batch)
            generated = module.event_model.joint_flow.sample(
                visible,
                batch["structured"],
                module.event_model.reconstruction,
                module.event_model.invisible,
                samples=posterior_samples,
                steps=steps,
                sample_chunk_size=sample_chunk_size,
                generator=sampling_generator,
                met_xy=batch["met_xy"],
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
            truth_root_indices = space.root_candidate_indices[space.truth_state]
            invisible = module.event_model.invisible.decode_state(
                generated["invisible"], batch["met_xy"], scaler
            )
            physical_invisible = module.event_model.invisible.decode_physical_state(
                generated["invisible"], batch["met_xy"], scaler
            )["four_vector"]
            truth_state = module.event_model.invisible.coordinates.encode(
                batch["invisible_truth"], batch["met_xy"]
            )
            truth_physical = module.event_model.invisible.coordinates.decode_physical(
                truth_state, batch["met_xy"]
            )["four_vector"]
            projection = _met_exact_truth_projection(module, batch)
            truth_hit = states == space.truth_state.unsqueeze(1)
            composites, truth_composites, observed_met = _physics_hypotheses(
                module,
                batch,
                space,
                node_indices,
                invisible,
                physical_invisible,
                truth_physical,
            )

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
                    "state": _create(
                        hypotheses, "assignment_state", (posterior_samples,), "<i4"
                    ),
                    "roots": _create(
                        hypotheses,
                        "root_candidate_indices",
                        (posterior_samples, len(space.roots)),
                        "<i4",
                    ),
                    "nodes": _create(
                        hypotheses,
                        "node_candidate_indices",
                        (posterior_samples, len(space.nodes)),
                        "<i4",
                    ),
                    "invisible": _create(
                        hypotheses,
                        "invisible",
                        (posterior_samples, len(module.spec.invisible_roles), 3),
                        "<f4",
                    ),
                    "invisible_four_vectors": _create(
                        hypotheses,
                        "invisible_four_vectors",
                        (posterior_samples, len(module.spec.invisible_roles), 4),
                        "<f4",
                    ),
                    "truth_state": _create(truth, "assignment_state", (), "<i4"),
                    "truth_roots": _create(
                        truth,
                        "root_candidate_indices",
                        (len(space.roots),),
                        "<i4",
                    ),
                    "truth_hit": _create(
                        truth, "hypothesis_assignment", (posterior_samples,), "?"
                    ),
                    "nu_truth": _create(
                        truth,
                        "NEUTRINO",
                        (len(module.spec.invisible_roles), 3),
                        "<f4",
                    ),
                    "truth_four_vector": _create(
                        truth,
                        "INVISIBLE_FOUR_VECTOR",
                        (len(module.spec.invisible_roles), 4),
                        "<f4",
                    ),
                    "njet": _create(observed, "njet", (), "<i2"),
                }
                if projection is not None:
                    projection_group = truth.create_group("MET_EXACT_PROJECTION")
                    projection_group.attrs["definition"] = (
                        "orthogonal equal transverse shift to measured MET"
                    )
                    datasets["projected_truth"] = _create(
                        projection_group,
                        "NEUTRINO_PROJECTED",
                        tuple(projection["projected"].shape[1:]),
                        "<f4",
                    )
                    datasets["projection_px_shift"] = _create(
                        projection_group, "px_shift", (), "<f4"
                    )
                    datasets["projection_py_shift"] = _create(
                        projection_group, "py_shift", (), "<f4"
                    )
                    datasets["projection_per_role_magnitude"] = _create(
                        projection_group,
                        "per_role_transverse_magnitude",
                        (projection["per_role_magnitude"].shape[1],),
                        "<f4",
                    )
                if composites is not None:
                    composite_group = destination.create_group("COMPOSITES")
                    truth_composite_group = truth.create_group("COMPOSITES")
                    for node in module.spec.dag_order:
                        datasets[f"composite/{node}"] = _create(
                            composite_group, node, (posterior_samples, 4), "<f4"
                        )
                        datasets[f"truth_composite/{node}"] = _create(
                            truth_composite_group, node, (4,), "<f4"
                        )
                if observed_met is not None:
                    datasets["observed_met"] = _create(observed, "MET", (4,), "<f4")
                destination.attrs["root_order"] = json.dumps(list(space.roots))
                destination.attrs["node_order"] = json.dumps(list(space.nodes))
                destination.attrs["assignment_state_count"] = space.state_count

            _append(
                datasets["event"],
                batch["event_index"].cpu().numpy().astype(np.int64),
            )
            _append(datasets["state"], states.cpu().numpy().astype(np.int32))
            _append(datasets["roots"], root_indices.cpu().numpy().astype(np.int32))
            _append(datasets["nodes"], node_indices.cpu().numpy().astype(np.int32))
            _append(
                datasets["invisible"], invisible.cpu().numpy().astype(np.float32)
            )
            _append(
                datasets["invisible_four_vectors"],
                physical_invisible.cpu().numpy().astype(np.float32),
            )
            _append(
                datasets["truth_state"],
                space.truth_state.cpu().numpy().astype(np.int32),
            )
            _append(
                datasets["truth_roots"],
                truth_root_indices.cpu().numpy().astype(np.int32),
            )
            _append(datasets["truth_hit"], truth_hit.cpu().numpy())
            _append(
                datasets["nu_truth"],
                batch["invisible_truth"].cpu().numpy().astype(np.float32),
            )
            _append(
                datasets["truth_four_vector"],
                truth_physical.cpu().numpy().astype(np.float32),
            )
            if projection is not None:
                _append(
                    datasets["projected_truth"],
                    projection["projected"].cpu().numpy().astype(np.float32),
                )
                _append(
                    datasets["projection_px_shift"],
                    projection["px_shift"].cpu().numpy().astype(np.float32),
                )
                _append(
                    datasets["projection_py_shift"],
                    projection["py_shift"].cpu().numpy().astype(np.float32),
                )
                _append(
                    datasets["projection_per_role_magnitude"],
                    projection["per_role_magnitude"]
                    .cpu()
                    .numpy()
                    .astype(np.float32),
                )
            _append(datasets["njet"], njet.cpu().numpy().astype(np.int16))
            if composites is not None:
                for node in module.spec.dag_order:
                    _append(datasets[f"composite/{node}"], composites[node])
                    _append(
                        datasets[f"truth_composite/{node}"], truth_composites[node]
                    )
            if observed_met is not None:
                _append(datasets["observed_met"], observed_met)

    validate_joint_prediction(temporary)
    os.replace(temporary, target)
