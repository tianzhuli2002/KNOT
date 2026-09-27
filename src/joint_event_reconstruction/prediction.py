"""Validated HDF5 prediction for discriminative factorized reconstruction."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch

from joint_event_reconstruction.assignment_semantics import (
    NON_NULL_SUPPORT,
    hdf5_support_attributes,
)
from joint_event_reconstruction.physics import compose_assignment_state
from joint_event_reconstruction.prediction_reductions import (
    POINT_ESTIMATORS,
    physical_four_vectors_from_momenta,
    reduce_joint_posterior,
)
from joint_event_reconstruction.prediction_schema import (
    PREDICTION_SCHEMA,
    validate_prediction,
)
from joint_event_reconstruction.topology.candidates import CandidateRecord
from joint_event_reconstruction.topology.decoder import ScoredCandidate, decode_complete


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    old = len(dataset)
    dataset.resize(old + len(values), axis=0)
    dataset[old:] = values


def create_prediction_dataset(
    group: h5py.Group, name: str, tail: tuple[int, ...], dtype
) -> h5py.Dataset:
    # ``Tensor.shape`` is a torch.Size, whose ``__radd__`` preserves the
    # torch.Size type.  Passing ``(None,) + torch.Size(...)`` to h5py then
    # makes h5py construct a torch.Size containing the extendible ``None``
    # dimension, which fails before the first prediction is written.  HDF5
    # dataset shapes are ordinary Python tuples; materialise that contract
    # before adding the extendible leading dimension.
    try:
        tail = tuple(int(dimension) for dimension in tail)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Prediction dataset {name!r} has a non-integer trailing shape: {tail!r}"
        ) from error
    return group.create_dataset(
        name,
        (0,) + tail,
        maxshape=(None,) + tail,
        dtype=dtype,
        chunks=True,
    )


def decode_batch(spec, batch: dict, output: dict, k: int):
    """Decode exact top-k complete non-NULL assignment hypotheses."""
    decoded = []
    for event in range(len(batch["tokens"])):
        scored = {}
        for root in spec.roots:
            pool = batch["structured"]["pools"][root]
            values = []
            valid = (
                pool["candidate_mask"][event]
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            for index in valid:
                ownership = (
                    pool["leaf_ownership"][index]
                    .nonzero(as_tuple=False)
                    .flatten()
                    .tolist()
                )
                record = CandidateRecord(
                    root,
                    index,
                    (),
                    (),
                    frozenset(("TOKEN", int(token)) for token in ownership),
                    bool(index == int(pool["truth_index"][event])),
                )
                probability = float(
                    output["assignment_probabilities"][root][event, index + 1]
                    .detach()
                    .cpu()
                )
                values.append(ScoredCandidate(record, probability))
            scored[root] = tuple(values)
        result = decode_complete(spec, scored, k=k)
        if not result.assignments:
            raise RuntimeError(
                f"EVENT_ID={int(batch['event_index'][event])} has no complete "
                "non-null assignment hypothesis"
            )
        if len(result.assignments) < k:
            result = replace(
                result,
                assignments=result.assignments
                + (result.assignments[-1],) * (k - len(result.assignments)),
            )
        decoded.append(result)
    return decoded


def node_indices_for_hypothesis(spec, batch: dict, event: int, hypothesis) -> torch.Tensor:
    selected = dict(hypothesis.roots)
    for node in reversed(spec.dag_order):
        if node not in selected or selected[node] < 0:
            continue
        for role in spec.nodes[node].composite_roles:
            child = int(
                batch["structured"]["pools"][node]["child_indices"][role.key][
                    selected[node]
                ]
            )
            previous = selected.setdefault(role.child, child)
            if previous != child:
                raise RuntimeError(
                    f"Decoded hypothesis has inconsistent child {role.child}"
                )
    return torch.tensor(
        [selected.get(node, -1) for node in spec.dag_order],
        dtype=torch.long,
        device=batch["tokens"].device,
    )


def truth_node_indices(spec, batch: dict, event: int) -> torch.Tensor:
    return torch.stack(
        [
            batch["structured"]["pools"][node]["truth_index"][event]
            for node in spec.dag_order
        ]
    ).to(torch.long)


def compose_physics(
    module,
    batch: dict,
    assignments,
    samples: torch.Tensor,
    *,
    physical_samples: torch.Tensor | None = None,
    truth_physical: torch.Tensor | None = None,
    truth_assignment: bool = False,
):
    if "physics_vectors" not in batch:
        return None, None, None
    batch_size, _, posterior_samples = samples.shape[:3]
    composite_values = {node: [] for node in module.spec.dag_order}
    truth_values = {node: [] for node in module.spec.dag_order}

    for event in range(batch_size):
        event_values = {node: [] for node in module.spec.dag_order}
        truth_supported = all(
            int(batch["structured"]["pools"][role.node]["truth_index"][event]) >= 0
            for role in module.spec.invisible_roles
        )
        if truth_assignment and not truth_supported:
            missing = torch.full(
                (samples.shape[1], posterior_samples, 4),
                float("nan"),
                dtype=samples.dtype,
                device=samples.device,
            )
            for node in module.spec.dag_order:
                composite_values[node].append(missing.clone())
            for node in module.spec.dag_order:
                # A truth-oracle composite is undefined when any conditioned
                # parent is absent from the retained visible input.  Never
                # pass the -1 sentinel to the composition primitive: that
                # would accidentally select the last candidate.
                truth_values[node].append(
                    torch.full(
                        (4,),
                        float("nan"),
                        dtype=samples.dtype,
                        device=samples.device,
                    )
                )
            continue
        hypotheses = range(samples.shape[1]) if truth_assignment else assignments[event]
        for hypothesis_index, hypothesis in enumerate(hypotheses):
            indices = (
                truth_node_indices(module.spec, batch, event)
                if truth_assignment
                else node_indices_for_hypothesis(module.spec, batch, event, hypothesis)
            )
            hypothesis_values = {node: [] for node in module.spec.dag_order}
            for sample_index in range(posterior_samples):
                invisible = {
                    role.key: samples[
                        event, hypothesis_index, sample_index, role_index
                    ]
                    for role_index, role in enumerate(module.spec.invisible_roles)
                }
                physical = None
                if physical_samples is not None:
                    physical = {
                        role.key: physical_samples[
                            event, hypothesis_index, sample_index, role_index
                        ]
                        for role_index, role in enumerate(module.spec.invisible_roles)
                    }
                composed = compose_assignment_state(
                    module.spec,
                    batch["structured"]["pools"],
                    batch["physics_vectors"][event],
                    indices,
                    invisible,
                    invisible_four_vectors=physical,
                )
                for node in module.spec.dag_order:
                    hypothesis_values[node].append(composed[node])
            for node in module.spec.dag_order:
                event_values[node].append(torch.stack(hypothesis_values[node]))
        for node in module.spec.dag_order:
            composite_values[node].append(torch.stack(event_values[node]))

        if truth_supported:
            truth_indices = truth_node_indices(module.spec, batch, event)
            truth_invisible = {
                role.key: batch["invisible_truth"][event, role_index]
                for role_index, role in enumerate(module.spec.invisible_roles)
            }
            truth_direct = None
            if truth_physical is not None:
                truth_direct = {
                    role.key: truth_physical[event, role_index]
                    for role_index, role in enumerate(module.spec.invisible_roles)
                }
            truth_composed = compose_assignment_state(
                module.spec,
                batch["structured"]["pools"],
                batch["physics_vectors"][event],
                truth_indices,
                truth_invisible,
                invisible_four_vectors=truth_direct,
            )
            for node in module.spec.dag_order:
                truth_values[node].append(truth_composed[node])
        else:
            # Normal predicted-assignment products retain the event, but a
            # physically undefined truth composite must remain explicitly
            # undefined for downstream support-aware evaluation.
            for node in module.spec.dag_order:
                truth_values[node].append(
                    torch.full(
                        (4,),
                        float("nan"),
                        dtype=samples.dtype,
                        device=samples.device,
                    )
                )

    composites = {
        node: torch.stack(values).cpu().numpy().astype(np.float32)
        for node, values in composite_values.items()
    }
    truth_composites = {
        node: torch.stack(values).cpu().numpy().astype(np.float32)
        for node, values in truth_values.items()
    }

    observed_met = None
    met_type = module.hparams.source_type_ids.get("MET")
    if met_type is not None:
        met_mask = batch["type_ids"] == int(met_type)
        if not (met_mask.sum(dim=1) == 1).all():
            raise RuntimeError("Prediction physics output requires exactly one MET token")
        observed_met = (
            batch["physics_vectors"][met_mask]
            .reshape(batch_size, 4)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    return composites, truth_composites, observed_met


def _generate_invisible(
    module,
    batch: dict,
    output: dict,
    decoded,
    *,
    top_k: int,
    posterior_samples: int,
    sampling_generator: torch.Generator,
):
    generator = module.event_model.invisible
    if generator is None:
        return None, 0, 0
    mode = generator.assignment_mode
    if mode == "hypothesis":
        parent = generator.assignment_parent_context(
            output,
            batch["structured"],
            output["visible"]["pooled"],
            [item.assignments for item in decoded],
            output.get("node_context"),
        )
        scaled, nfe = generator.sample_hypotheses(
            output["visible"], parent, samples=posterior_samples,
            generator=sampling_generator,
        )
        return scaled, nfe, top_k

    parent = generator.parent_context(
        output,
        batch["structured"],
        output["visible"]["pooled"],
        output.get("node_context"),
        mode=mode,
    )
    generated, nfe = generator.sample(
        output["visible"], parent, samples=posterior_samples,
        generator=sampling_generator,
    )
    # Common storage is [B,K,M,Rnu,3]. Non-hypothesis modes have one unique
    # posterior branch; K contains repeated views and is not counted as new samples.
    scaled = generated.unsqueeze(1).expand(-1, top_k, -1, -1, -1)
    return scaled, nfe, 1


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


def predict_hdf5(
    module,
    data,
    scaler,
    output_path: str | Path,
    *,
    split: str,
    top_k: int = 1,
    posterior_samples: int = 1,
    recurrent_steps: int | None = None,
    device: str = "cpu",
    prediction_seed: int,
    checkpoint_path: str = "",
    compatibility_weight: float = 1.0,
    point_estimator: str = "none",
) -> None:
    if split not in {"train", "validation", "test"}:
        raise ValueError(
            f"Prediction split must be one of train, validation, test; got {split!r}"
        )
    if top_k <= 0 or posterior_samples <= 0:
        raise ValueError("Prediction K and M must be positive")
    if point_estimator not in {"none", *POINT_ESTIMATORS, "direct_point_head"}:
        raise ValueError(f"Unknown point estimator {point_estimator!r}")
    if point_estimator == "direct_point_head" and module.event_model.point_head is None:
        raise ValueError("direct_point_head inference requires an enabled point head")
    loader = {
        "train": data.train_dataloader,
        "validation": data.val_dataloader,
        "test": data.test_dataloader,
    }[split]()
    iterator = iter(loader)
    sampling_generator = torch.Generator(device=device).manual_seed(int(prediction_seed))
    visible_truth_available = bool(data.structured_collator.require_truth)
    invisible_truth_available = data.structured_collator.invisible_store is not None
    truth_composites_available = (
        visible_truth_available and invisible_truth_available
    )
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
        invisible_truth_contract = module.hparams.get("invisible_truth_contract")
        invisible_provenance = {}
        if isinstance(invisible_truth_contract, dict):
            role_order = list(invisible_truth_contract.get("invisible_role_order", []))
            configured_roles = [role.key for role in module.spec.invisible_roles]
            if role_order != configured_roles:
                raise ValueError(
                    "Checkpoint invisible-role provenance differs from prediction model: "
                    f"checkpoint={role_order}, model={configured_roles}"
                )
            invisible_provenance = {
                "invisible_role_convention": invisible_truth_contract[
                    "invisible_role_convention"
                ],
                "invisible_role_order": json.dumps(role_order),
                "neutrino_truth_definition": invisible_truth_contract["truth_definition"],
                "invisible_units": invisible_truth_contract["units"],
                "source_file_sha256": invisible_truth_contract["source_file_sha256"],
                "source_manifest_sha256": invisible_truth_contract[
                    "source_manifest_sha256"
                ],
                "split_membership_sha256": invisible_truth_contract[
                    "split_membership_sha256"
                ],
            }
        assignment_mode = (
            module.event_model.invisible.assignment_mode
            if module.event_model.invisible is not None
            else "disabled"
        )
        stored_posterior_samples = (
            1 if point_estimator == "direct_point_head" else posterior_samples
        )
        destination.attrs.update(
            {
                "schema_version": PREDICTION_SCHEMA,
                "model_mode": "factorized",
                "assignment_formulation": "discriminative",
                **hdf5_support_attributes(NON_NULL_SUPPORT),
                "topology_signature": module.spec.signature,
                "physics_signature": module.spec.physics_signature or "none",
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
                "visible_truth_available": visible_truth_available,
                "invisible_truth_available": invisible_truth_available,
                "truth_composites_available": truth_composites_available,
                "prediction_seed": int(prediction_seed),
                "recurrent_steps": -1 if recurrent_steps is None else recurrent_steps,
                "invisible_formulation": module.hparams.invisible_formulation or "disabled",
                "invisible_assignment_conditioning": assignment_mode,
                "invisible_hypotheses_conditioned_on_complete_assignment": (
                    assignment_mode == "hypothesis"
                ),
                "K": top_k,
                "M": stored_posterior_samples,
                "point_estimator": str(point_estimator),
                "invisible_roles": json.dumps(
                    [role.key for role in module.spec.invisible_roles]
                ),
                "invisible_coordinates": (
                    module.hparams.invisible_coordinates or "disabled"
                ),
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
                "assignment_state_representation": (
                    module.hparams.assignment_state_representation or "disabled"
                ),
                "visible_encoder": module.hparams.visible_encoder,
                "visible_encoder_config": json.dumps(
                    module.hparams.visible_encoder_config
                ),
                "invisible_role_count": int(module.hparams.invisible_role_count),
                **invisible_provenance,
                "topology_query_roles": json.dumps(
                    list(getattr(module.event_model, "query_names", ()))
                ),
                "partonic_roles": json.dumps(
                    list(getattr(module.event_model, "partonic_roles", ()))
                ),
                "scaler": (
                    json.dumps(scaler.as_dict()) if scaler is not None else "null"
                ),
                "compatibility_is_normalised_likelihood": False,
                "source_data": str(source_path),
                "source_data_size_bytes": source_stat.st_size,
                "source_data_mtime_ns": source_stat.st_mtime_ns,
            }
        )

        for batch in iterator:
            batch = to_device(batch, device)
            model_batch = batch
            if point_estimator == "direct_point_head":
                # Direct-head latency must not include an unnecessary CFM loss
                # evaluation.  Truth remains in ``batch`` for evaluation.
                model_batch = dict(batch)
                model_batch.pop("invisible_truth_scaled", None)
                model_batch.pop("invisible_truth_point_scaled", None)
            output = module(model_batch, recurrent_steps=recurrent_steps)
            decoded = decode_batch(module.spec, batch, output, top_k)
            assignments = [item.assignments for item in decoded]
            roots = module.spec.roots
            root_indices = np.asarray(
                [
                    [
                        [dict(hypothesis.roots)[root] for root in roots]
                        for hypothesis in event
                    ]
                    for event in assignments
                ],
                dtype=np.int32,
            )
            log_scores = np.asarray(
                [[hyp.log_score for hyp in event] for event in assignments],
                dtype=np.float32,
            )
            valid_hypotheses = np.asarray(
                [np.arange(top_k) < item.valid_assignment_count for item in decoded],
                dtype=np.bool_,
            )
            log_scores = np.where(valid_hypotheses, log_scores, -1.0e30).astype(
                np.float32
            )
            logz = np.asarray(
                [item.log_normalisation for item in decoded], dtype=np.float32
            )
            constrained = np.where(
                valid_hypotheses, np.exp(log_scores - logz[:, None]), 0.0
            ).astype(np.float32)
            truth = np.asarray(
                [[hyp.truth for hyp in event] for event in assignments],
                dtype=np.bool_,
            )
            truth &= valid_hypotheses

            jet_type = data.schema.source_type_ids.get("JET")
            if jet_type is None:
                raise RuntimeError("Prediction schema has no JET source for Njet diagnostics")
            njet = (
                (batch["type_ids"] == jet_type) & ~batch["padding_mask"]
            ).sum(dim=1).cpu().numpy().astype(np.int16)

            samples = mean = std = None
            physical_samples = physical_mean = physical_std = truth_physical = None
            point_momenta = point_four_vectors = None
            parent_four_vectors = output.get("parent_four_vectors")
            partonic_four_vectors = output.get("partonic_four_vectors")
            projection = None
            composites = truth_composites = observed_met = None
            nfe = 0
            unique_branches = 0
            compatibility = final_score = final_rank = None
            should_generate_cfm = (
                module.event_model.invisible is not None
                and point_estimator != "direct_point_head"
            )
            if should_generate_cfm:
                scaled, nfe, unique_branches = _generate_invisible(
                    module,
                    batch,
                    output,
                    decoded,
                    top_k=top_k,
                    posterior_samples=posterior_samples,
                    sampling_generator=sampling_generator,
                )
                samples_tensor = module.event_model.invisible.decode_state(
                    scaled, batch["met_xy"], scaler
                )
                physical_samples = module.event_model.invisible.decode_physical_state(
                    scaled, batch["met_xy"], scaler
                )["four_vector"]
                truth_available = "invisible_truth" in batch
                truth_physical = None
                if truth_available:
                    truth_state = module.event_model.invisible.coordinates.encode(
                        batch["invisible_truth"], batch["met_xy"]
                    )
                    truth_physical = module.event_model.invisible.coordinates.decode_physical(
                        truth_state, batch["met_xy"]
                    )["four_vector"]
                samples = samples_tensor.cpu().numpy().astype(np.float32)
                mean = samples.mean(axis=2)
                std = samples.std(axis=2)
                mean_tensor = reduce_joint_posterior(
                    samples_tensor, "cfm_cartesian_mean"
                )
                physical_mean = physical_four_vectors_from_momenta(
                    module.event_model.invisible.coordinates, mean_tensor
                ).cpu().numpy().astype(np.float32)
                # A one-draw posterior is a supported point-estimator mode;
                # its empirical spread is defined as zero rather than the
                # undefined unbiased estimator with one observation.
                physical_std = physical_samples.std(
                    dim=2, unbiased=False
                ).cpu().numpy().astype(np.float32)
                if point_estimator != "none":
                    medoid_scale = None
                    if point_estimator == "cfm_joint_medoid":
                        point_scaler = module.event_model.point_scaler
                        if point_scaler is not None:
                            medoid_scale = samples_tensor.new_tensor(point_scaler.std)
                        elif scaler is not None:
                            medoid_scale = samples_tensor.new_tensor(
                                scaler.std[: len(module.spec.invisible_roles)]
                            )
                    point_momenta = reduce_joint_posterior(
                        samples_tensor, point_estimator, scale=medoid_scale
                    )
                    point_four_vectors = physical_four_vectors_from_momenta(
                        module.event_model.invisible.coordinates, point_momenta
                    )
                projection = (
                    _met_exact_truth_projection(module, batch)
                    if truth_available
                    else None
                )
                composites, truth_composites, observed_met = compose_physics(
                    module,
                    batch,
                    assignments,
                    samples_tensor,
                    physical_samples=physical_samples,
                    truth_physical=truth_physical,
                    truth_assignment=assignment_mode == "truth",
                )

                if module.event_model.compatibility is not None:
                    assignment_summary = []
                    for event, hypotheses in enumerate(assignments):
                        event_summary = []
                        for hypothesis in hypotheses:
                            selected = dict(hypothesis.roots)
                            event_summary.append(
                                torch.stack(
                                    [
                                        output["candidate_embeddings"][root][
                                            event, selected[root]
                                        ]
                                        for root in roots
                                    ]
                                ).mean(dim=0)
                            )
                        assignment_summary.append(torch.stack(event_summary))
                    assignment_summary = torch.stack(assignment_summary)
                    event_context = output["visible"]["pooled"][:, None, None].expand(
                        -1, top_k, posterior_samples, -1
                    )
                    assignment_context = assignment_summary[:, :, None].expand(
                        -1, -1, posterior_samples, -1
                    )
                    compatibility_samples = module.event_model.compatibility(
                        event_context, assignment_context, scaled
                    )
                    compatibility = (
                        compatibility_samples.mean(dim=2)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    final_score = log_scores + compatibility_weight * compatibility
                    final_score = np.where(
                        valid_hypotheses, final_score, -1.0e30
                    ).astype(np.float32)
                    final_rank = np.argsort(
                        -final_score, axis=1, kind="stable"
                    ).astype(np.int32)
            elif point_estimator == "direct_point_head":
                point_scaler = module.event_model.point_scaler
                if point_scaler is None or "point_momenta_scaled" not in output:
                    raise RuntimeError("Direct point-head prediction has no scaler/output")
                direct = point_scaler.inverse(output["point_momenta_scaled"])
                point_momenta = direct.unsqueeze(1).expand(-1, top_k, -1, -1)
                point_four_vectors = physical_four_vectors_from_momenta(
                    module.event_model.invisible.coordinates, point_momenta
                )
                truth_available = "invisible_truth" in batch
                truth_physical = None
                if truth_available:
                    truth_state = module.event_model.invisible.coordinates.encode(
                        batch["invisible_truth"], batch["met_xy"]
                    )
                    truth_physical = module.event_model.invisible.coordinates.decode_physical(
                        truth_state, batch["met_xy"]
                    )["four_vector"]
                direct_samples = point_momenta.unsqueeze(2)
                composites, truth_composites, observed_met = compose_physics(
                    module,
                    batch,
                    assignments,
                    direct_samples,
                    physical_samples=point_four_vectors.unsqueeze(2),
                    truth_physical=truth_physical,
                    truth_assignment=assignment_mode == "truth",
                )

            if datasets is None:
                visible_group = destination.create_group("VISIBLE")
                invisible_group = destination.create_group("INVISIBLE")
                joint_group = destination.create_group("JOINT")
                truth_group = destination.create_group("TRUTH")
                truth_group.attrs["visible_available"] = bool(
                    destination.attrs["visible_truth_available"]
                )
                truth_group.attrs["invisible_available"] = bool(
                    destination.attrs["invisible_truth_available"]
                )
                observed_group = destination.create_group("OBSERVED")
                datasets = {
                    "event": create_prediction_dataset(destination, "EVENT_ID", (), "<i8"),
                    "roots": create_prediction_dataset(
                        visible_group,
                        "root_candidate_indices",
                        root_indices.shape[1:],
                        "<i4",
                    ),
                    "log_scores": create_prediction_dataset(
                        visible_group, "assignment_log_scores", (top_k,), "<f4"
                    ),
                    "logz": create_prediction_dataset(visible_group, "exact_logZ", (), "<f4"),
                    "probabilities": create_prediction_dataset(
                        visible_group, "constrained_probabilities", (top_k,), "<f4"
                    ),
                    "valid_hypotheses": create_prediction_dataset(
                        visible_group, "valid_hypotheses", (top_k,), "?"
                    ),
                    "truth_assignment": create_prediction_dataset(
                        truth_group, "topk_assignment", (top_k,), "?"
                    ),
                    "njet": create_prediction_dataset(observed_group, "njet", (), "<i2"),
                }
                destination.attrs["root_order"] = json.dumps(list(roots))
                for node in module.spec.dag_order:
                    node_group = visible_group.create_group(f"nodes/{node}")
                    datasets[f"{node}/selected"] = create_prediction_dataset(
                        node_group, "selected_candidate", (), "<i4"
                    )
                    datasets[f"{node}/truth"] = create_prediction_dataset(
                        node_group, "truth_candidate", (), "<i4"
                    )
                    datasets[f"{node}/reconstructable"] = create_prediction_dataset(
                        node_group, "reconstructable", (), "?"
                    )
                    datasets[f"{node}/truth_probability"] = create_prediction_dataset(
                        node_group, "truth_probability", (), "<f4"
                    )
                    datasets[f"{node}/entropy"] = create_prediction_dataset(
                        node_group, "entropy", (), "<f4"
                    )
                    datasets[f"{node}/margin"] = create_prediction_dataset(
                        node_group, "margin", (), "<f4"
                    )
                if samples is not None:
                    datasets["samples"] = create_prediction_dataset(
                        invisible_group, "samples", samples.shape[1:], "<f4"
                    )
                    datasets["mean"] = create_prediction_dataset(
                        invisible_group, "mean", mean.shape[1:], "<f4"
                    )
                    datasets["std"] = create_prediction_dataset(
                        invisible_group, "std", std.shape[1:], "<f4"
                    )
                    datasets["four_vectors"] = create_prediction_dataset(
                        invisible_group,
                        "four_vectors",
                        physical_samples.shape[1:],
                        "<f4",
                    )
                    datasets["four_vector_mean"] = create_prediction_dataset(
                        invisible_group,
                        "four_vector_mean",
                        physical_mean.shape[1:],
                        "<f4",
                    )
                    datasets["four_vector_std"] = create_prediction_dataset(
                        invisible_group,
                        "four_vector_std",
                        physical_std.shape[1:],
                        "<f4",
                    )
                    if "invisible_truth" in batch:
                        datasets["nu_truth"] = create_prediction_dataset(
                            truth_group,
                            "NEUTRINO",
                            tuple(batch["invisible_truth"].shape[1:]),
                            "<f4",
                        )
                    if truth_physical is not None:
                        datasets["truth_four_vector"] = create_prediction_dataset(
                            truth_group,
                            "INVISIBLE_FOUR_VECTOR",
                            tuple(truth_physical.shape[1:]),
                            "<f4",
                        )
                    if projection is not None:
                        projection_group = truth_group.create_group(
                            "MET_EXACT_PROJECTION"
                        )
                        projection_group.attrs["definition"] = (
                            "orthogonal equal transverse shift to measured MET"
                        )
                        datasets["projected_truth"] = create_prediction_dataset(
                            projection_group,
                            "NEUTRINO_PROJECTED",
                            tuple(projection["projected"].shape[1:]),
                            "<f4",
                        )
                        datasets["projection_px_shift"] = create_prediction_dataset(
                            projection_group, "px_shift", (), "<f4"
                        )
                        datasets["projection_py_shift"] = create_prediction_dataset(
                            projection_group, "py_shift", (), "<f4"
                        )
                        datasets["projection_per_role_magnitude"] = create_prediction_dataset(
                            projection_group,
                            "per_role_transverse_magnitude",
                            (projection["per_role_magnitude"].shape[1],),
                            "<f4",
                        )
                    invisible_group.attrs["actual_nfe"] = nfe
                    invisible_group.attrs[
                        "unique_assignment_conditioning_branches"
                    ] = unique_branches
                    invisible_group.attrs[
                        "sample_equivalent_nfe_per_event"
                    ] = nfe * unique_branches * posterior_samples
                    if composites is not None:
                        composite_group = destination.create_group("COMPOSITES")
                        truth_composite_group = (
                            truth_group.create_group("COMPOSITES")
                            if bool(destination.attrs["truth_composites_available"])
                            else None
                        )
                        for node, values in composites.items():
                            datasets[f"composite/{node}"] = create_prediction_dataset(
                                composite_group, node, values.shape[1:], "<f4"
                            )
                            if truth_composite_group is not None:
                                datasets[f"truth_composite/{node}"] = create_prediction_dataset(
                                    truth_composite_group, node, (4,), "<f4"
                                )
                    if observed_met is not None:
                        datasets["observed_met"] = create_prediction_dataset(
                            observed_group, "MET", (4,), "<f4"
                        )
                if samples is None:
                    if composites is not None:
                        composite_group = destination.create_group("COMPOSITES")
                        truth_composite_group = (
                            truth_group.create_group("COMPOSITES")
                            if bool(destination.attrs["truth_composites_available"])
                            else None
                        )
                        for node, values in composites.items():
                            datasets[f"composite/{node}"] = create_prediction_dataset(
                                composite_group, node, values.shape[1:], "<f4"
                            )
                            if truth_composite_group is not None:
                                datasets[f"truth_composite/{node}"] = create_prediction_dataset(
                                    truth_composite_group, node, (4,), "<f4"
                                )
                    if observed_met is not None:
                        datasets["observed_met"] = create_prediction_dataset(
                            observed_group, "MET", (4,), "<f4"
                        )
                if point_momenta is not None:
                    datasets["point_momenta"] = create_prediction_dataset(
                        invisible_group, "point_momenta", point_momenta.shape[1:], "<f4"
                    )
                    datasets["point_four_vectors"] = create_prediction_dataset(
                        invisible_group, "point_four_vectors", point_four_vectors.shape[1:], "<f4"
                    )
                    invisible_group.attrs["point_estimator"] = str(point_estimator)
                    invisible_group.attrs["point_output_rule"] = str(point_estimator)
                    if samples is None and "invisible_truth" in batch:
                        datasets["nu_truth"] = create_prediction_dataset(
                            truth_group, "NEUTRINO", tuple(batch["invisible_truth"].shape[1:]), "<f4"
                        )
                    if samples is None and truth_physical is not None:
                        datasets["truth_four_vector"] = create_prediction_dataset(
                            truth_group, "INVISIBLE_FOUR_VECTOR", tuple(truth_physical.shape[1:]), "<f4"
                        )
                if parent_four_vectors is not None:
                    parent_group = destination.create_group("PARENTS")
                    datasets["parent_four_vectors"] = create_prediction_dataset(
                        parent_group, "direct_four_vectors", parent_four_vectors.shape[1:], "<f4"
                    )
                    parent_group.attrs["roles"] = json.dumps(list(module.event_model.parent_roles))
                    parent_group.attrs["output_rule"] = "direct_parent_head"
                if partonic_four_vectors is not None:
                    partonic_group = destination.create_group("PARTONIC")
                    datasets["partonic_four_vectors"] = create_prediction_dataset(
                        partonic_group,
                        "direct_four_vectors",
                        partonic_four_vectors.shape[1:],
                        "<f4",
                    )
                    partonic_group.attrs["roles"] = json.dumps(
                        list(module.event_model.partonic_roles)
                    )
                    partonic_group.attrs["output_rule"] = "direct_partonic_query_head"
                    if "partonic_truth" in batch:
                        datasets["partonic_truth"] = create_prediction_dataset(
                            truth_group,
                            "PARTONIC_FOUR_VECTORS",
                            tuple(batch["partonic_truth"].shape[1:]),
                            "<f4",
                        )
                        datasets["partonic_support"] = create_prediction_dataset(
                            truth_group,
                            "PARTONIC_SUPPORT",
                            tuple(batch["partonic_support"].shape[1:]),
                            "?",
                        )
                if compatibility is not None:
                    datasets["compatibility"] = create_prediction_dataset(
                        joint_group, "compatibility_logit", (top_k,), "<f4"
                    )
                    datasets["final_score"] = create_prediction_dataset(
                        joint_group, "final_score", (top_k,), "<f4"
                    )
                    datasets["final_rank"] = create_prediction_dataset(
                        joint_group, "rank_indices", (top_k,), "<i4"
                    )
                    joint_group.attrs["compatibility_weight"] = compatibility_weight

            _append(
                datasets["event"],
                batch["event_index"].cpu().numpy().astype(np.int64),
            )
            _append(datasets["roots"], root_indices)
            _append(datasets["log_scores"], log_scores)
            _append(datasets["logz"], logz)
            _append(datasets["probabilities"], constrained)
            _append(datasets["valid_hypotheses"], valid_hypotheses)
            _append(datasets["truth_assignment"], truth)
            _append(datasets["njet"], njet)
            for node in module.spec.dag_order:
                probabilities = output["assignment_probabilities"][node]
                selected = probabilities.argmax(dim=1) - 1
                local_truth = batch["structured"]["pools"][node]["truth_index"]
                safe = local_truth.clamp_min(0)
                rows = torch.arange(len(probabilities), device=probabilities.device)
                truth_probability = probabilities[rows, safe + 1] * (local_truth >= 0)
                entropy = -(
                    probabilities * probabilities.clamp_min(1e-12).log()
                ).sum(dim=1)
                top2 = probabilities.topk(
                    min(2, probabilities.shape[1]), dim=1
                ).values
                margin = (
                    top2[:, 0] - top2[:, 1]
                    if top2.shape[1] == 2
                    else top2[:, 0]
                )
                _append(
                    datasets[f"{node}/selected"],
                    selected.cpu().numpy().astype(np.int32),
                )
                _append(
                    datasets[f"{node}/truth"],
                    local_truth.cpu().numpy().astype(np.int32),
                )
                _append(
                    datasets[f"{node}/reconstructable"],
                    batch["structured"]["pools"][node]["reconstructable"]
                    .cpu()
                    .numpy(),
                )
                _append(
                    datasets[f"{node}/truth_probability"],
                    truth_probability.cpu().numpy().astype(np.float32),
                )
                _append(
                    datasets[f"{node}/entropy"],
                    entropy.cpu().numpy().astype(np.float32),
                )
                _append(
                    datasets[f"{node}/margin"],
                    margin.cpu().numpy().astype(np.float32),
                )
            if samples is not None:
                _append(datasets["samples"], samples)
                _append(datasets["mean"], mean)
                _append(datasets["std"], std)
                _append(
                    datasets["four_vectors"],
                    physical_samples.cpu().numpy().astype(np.float32),
                )
                _append(datasets["four_vector_mean"], physical_mean)
                _append(datasets["four_vector_std"], physical_std)
                if "nu_truth" in datasets:
                    _append(
                        datasets["nu_truth"],
                        batch["invisible_truth"].cpu().numpy().astype(np.float32),
                    )
                if "truth_four_vector" in datasets:
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
                if composites is not None:
                    for node, values in composites.items():
                        _append(datasets[f"composite/{node}"], values)
                        if f"truth_composite/{node}" in datasets:
                            _append(
                                datasets[f"truth_composite/{node}"], truth_composites[node]
                            )
                if observed_met is not None:
                    _append(datasets["observed_met"], observed_met)
            if point_momenta is not None:
                if samples is None:
                    if composites is not None:
                        for node, values in composites.items():
                            _append(datasets[f"composite/{node}"], values)
                            if f"truth_composite/{node}" in datasets:
                                _append(
                                    datasets[f"truth_composite/{node}"],
                                    truth_composites[node],
                                )
                    if observed_met is not None:
                        _append(datasets["observed_met"], observed_met)
                _append(
                    datasets["point_momenta"],
                    point_momenta.cpu().numpy().astype(np.float32),
                )
                _append(
                    datasets["point_four_vectors"],
                    point_four_vectors.cpu().numpy().astype(np.float32),
                )
                if samples is None and "nu_truth" in datasets:
                    _append(
                        datasets["nu_truth"],
                        batch["invisible_truth"].cpu().numpy().astype(np.float32),
                    )
                if samples is None and "truth_four_vector" in datasets:
                    _append(
                        datasets["truth_four_vector"],
                        truth_physical.cpu().numpy().astype(np.float32),
                    )
            if parent_four_vectors is not None:
                _append(
                    datasets["parent_four_vectors"],
                    parent_four_vectors.cpu().numpy().astype(np.float32),
                )
            if partonic_four_vectors is not None:
                _append(
                    datasets["partonic_four_vectors"],
                    partonic_four_vectors.cpu().numpy().astype(np.float32),
                )
                if "partonic_truth" in batch:
                    _append(
                        datasets["partonic_truth"],
                        batch["partonic_truth"].cpu().numpy().astype(np.float32),
                    )
                    _append(
                        datasets["partonic_support"],
                        batch["partonic_support"].cpu().numpy(),
                    )
            if compatibility is not None:
                _append(datasets["compatibility"], compatibility)
                _append(datasets["final_score"], final_score)
                _append(datasets["final_rank"], final_rank)

    validate_prediction(temporary)
    os.replace(temporary, target)


def to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


# Private aliases remain for older internal diagnostics; maintained new code
# should import the public names above.
_create = create_prediction_dataset
_truth_node_indices = truth_node_indices
_compose_physics = compose_physics
_node_indices = node_indices_for_hypothesis
_to_device = to_device


__all__ = [
    "compose_physics",
    "create_prediction_dataset",
    "decode_batch",
    "node_indices_for_hypothesis",
    "predict_hdf5",
    "to_device",
    "truth_node_indices",
]
