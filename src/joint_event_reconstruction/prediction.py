"""Validated, bounded HDF5 prediction for complete joint event hypotheses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from joint_event_reconstruction.topology.candidates import CandidateRecord
from joint_event_reconstruction.topology.decoder import ScoredCandidate, decode_complete
from joint_event_reconstruction.physics import compose_four_vectors


PREDICTION_SCHEMA = "joint_event_reconstruction_predictions_v1"


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    old = len(dataset)
    dataset.resize(old + len(values), axis=0)
    dataset[old:] = values


def _create(group: h5py.Group, name: str, tail: tuple[int, ...], dtype) -> h5py.Dataset:
    return group.create_dataset(name, (0,) + tail, maxshape=(None,) + tail, dtype=dtype, chunks=True)


def decode_batch(spec, batch: dict, output: dict, k: int):
    decoded = []
    for event in range(len(batch["tokens"])):
        scored = {}
        for root in spec.roots:
            pool = batch["structured"]["pools"][root]
            values = []
            valid = pool["candidate_mask"][event].nonzero(as_tuple=False).flatten().tolist()
            for index in valid:
                ownership = pool["leaf_ownership"][index].nonzero(as_tuple=False).flatten().tolist()
                record = CandidateRecord(
                    root, index, (), (), frozenset(("TOKEN", int(token)) for token in ownership),
                    bool(index == int(pool["truth_index"][event])),
                )
                probability = float(output["assignment_probabilities"][root][event, index + 1].detach().cpu())
                values.append(ScoredCandidate(record, probability))
            scored[root] = tuple(values)
        result = decode_complete(spec, scored, k=k)
        if len(result.assignments) < k:
            raise RuntimeError(f"EVENT_ID={int(batch['event_index'][event])} has only {len(result.assignments)} complete hypotheses")
        decoded.append(result)
    return decoded


def predict_hdf5(
    module, data, scaler, output_path: str | Path, *, split: str = "test",
    top_k: int = 1, posterior_samples: int = 1, recurrent_steps: int | None = None,
    device: str = "cpu", checkpoint_path: str = "", compatibility_lambda: float = 1.0,
) -> None:
    if top_k <= 0 or posterior_samples <= 0:
        raise ValueError("Prediction K and M must be positive")
    loader = {"train": data.train_dataloader, "validation": data.val_dataloader, "test": data.test_dataloader}[split]()
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
            "schema_version": PREDICTION_SCHEMA, "topology_signature": module.spec.signature,
            "architecture_signature": module.hparams.architecture_signature,
            "checkpoint": str(Path(checkpoint_path).expanduser().resolve()) if checkpoint_path else "untrained",
            "split": split, "recurrent_steps": -1 if recurrent_steps is None else recurrent_steps,
            "invisible_formulation": module.hparams.invisible_formulation or "disabled",
            "K": top_k, "M": posterior_samples,
            "invisible_roles": json.dumps([role.key for role in module.spec.invisible_roles]),
            "scaler": json.dumps(scaler.as_dict()) if scaler is not None else "null",
            "compatibility_is_normalised_likelihood": False,
            "source_data": str(source_path), "source_data_size_bytes": source_stat.st_size,
            "source_data_mtime_ns": source_stat.st_mtime_ns,
        })
        for batch in loader:
            batch = _to_device(batch, device)
            output = module(batch, recurrent_steps=recurrent_steps)
            decoded = decode_batch(module.spec, batch, output, top_k)
            assignments = [item.assignments for item in decoded]
            roots = module.spec.roots
            root_indices = np.asarray([
                [[dict(hypothesis.roots)[root] for root in roots] for hypothesis in event]
                for event in assignments
            ], dtype=np.int32)
            log_scores = np.asarray([[hyp.log_score for hyp in event] for event in assignments], dtype=np.float32)
            logz = np.asarray([item.log_normalisation for item in decoded], dtype=np.float32)
            constrained = np.exp(log_scores - logz[:, None]).astype(np.float32)
            truth = np.asarray([[hyp.truth for hyp in event] for event in assignments], dtype=np.bool_)

            samples = mean = std = None
            composites = None
            truth_composites = None
            observed_met = None
            nfe = 0
            compatibility = final_score = final_rank = None
            if module.event_model.invisible is not None:
                parent = module.event_model.invisible.assignment_parent_context(
                    output, batch["structured"], output["visible"]["pooled"], assignments,
                    output.get("node_context"),
                )
                scaled, nfe = module.event_model.invisible.sample_hypotheses(
                    output["visible"], parent, samples=posterior_samples,
                )
                samples = scaler.inverse(scaled).cpu().numpy().astype(np.float32)
                mean = samples.mean(axis=2)
                std = samples.std(axis=2)
                if "physics_vectors" in batch:
                    composite_values = {node: [] for node in module.spec.dag_order}
                    for event, hypotheses in enumerate(assignments):
                        event_values = {node: [] for node in module.spec.dag_order}
                        for hypothesis_index, hypothesis in enumerate(hypotheses):
                            selected = dict(hypothesis.roots)
                            for node in reversed(module.spec.dag_order):
                                if node not in selected:
                                    continue
                                for role in module.spec.nodes[node].composite_roles:
                                    selected[role.child] = int(batch["structured"]["pools"][node]["child_indices"][role.key][selected[node]])
                            visible_leaves = {}
                            for node, candidate_index in selected.items():
                                for role in module.spec.nodes[node].leaf_roles:
                                    if role.invisible:
                                        continue
                                    indices = batch["structured"]["pools"][node]["leaf_indices"][role.key][candidate_index]
                                    visible_leaves[role.key] = batch["physics_vectors"][event, indices]
                            hypothesis_values = {node: [] for node in module.spec.dag_order}
                            for sample_index in range(posterior_samples):
                                invisible = {
                                    role.key: torch.as_tensor(samples[event, hypothesis_index, sample_index, role_index], device=batch["tokens"].device)
                                    for role_index, role in enumerate(module.spec.invisible_roles)
                                }
                                composed = compose_four_vectors(module.spec, visible_leaves, invisible)
                                for node in module.spec.dag_order:
                                    hypothesis_values[node].append(composed[node])
                            for node in module.spec.dag_order:
                                event_values[node].append(torch.stack(hypothesis_values[node]))
                        for node in module.spec.dag_order:
                            composite_values[node].append(torch.stack(event_values[node]))
                    composites = {
                        node: torch.stack(values).cpu().numpy().astype(np.float32)
                        for node, values in composite_values.items()
                    }
                    truth_values = {node: [] for node in module.spec.dag_order}
                    for event in range(len(batch["tokens"])):
                        if not all(bool(batch["structured"]["pools"][node]["reconstructable"][event]) for node in module.spec.dag_order):
                            for node in module.spec.dag_order:
                                truth_values[node].append(torch.full((4,), float("nan"), device=batch["tokens"].device))
                            continue
                        visible_leaves = {}
                        for node in module.spec.dag_order:
                            selected_truth = int(batch["structured"]["pools"][node]["truth_index"][event])
                            for role in module.spec.nodes[node].leaf_roles:
                                if not role.invisible:
                                    indices = batch["structured"]["pools"][node]["leaf_indices"][role.key][selected_truth]
                                    visible_leaves[role.key] = batch["physics_vectors"][event, indices]
                        invisible_truth = {
                            role.key: batch["invisible_truth"][event, role_index]
                            for role_index, role in enumerate(module.spec.invisible_roles)
                        }
                        composed = compose_four_vectors(module.spec, visible_leaves, invisible_truth)
                        for node in module.spec.dag_order:
                            truth_values[node].append(composed[node])
                    truth_composites = {
                        node: torch.stack(values).cpu().numpy().astype(np.float32)
                        for node, values in truth_values.items()
                    }
                    met_type = data.schema.source_type_ids.get("MET")
                    if met_type is not None:
                        met_mask = batch["type_ids"] == met_type
                        if not (met_mask.sum(dim=1) == 1).all():
                            raise RuntimeError("Prediction physics output requires exactly one MET token")
                        observed_met = batch["physics_vectors"][met_mask].reshape(len(batch["tokens"]), 4).cpu().numpy().astype(np.float32)
                if module.event_model.compatibility is not None:
                    assignment_summary = []
                    for event, hypotheses in enumerate(assignments):
                        event_summary = []
                        for hypothesis in hypotheses:
                            selected = dict(hypothesis.roots)
                            event_summary.append(torch.stack([
                                output["candidate_embeddings"][root][event, selected[root]] for root in roots
                            ]).mean(dim=0))
                        assignment_summary.append(torch.stack(event_summary))
                    assignment_summary = torch.stack(assignment_summary)
                    event_context = output["visible"]["pooled"][:, None, None].expand(-1, top_k, posterior_samples, -1)
                    assignment_context = assignment_summary[:, :, None].expand(-1, -1, posterior_samples, -1)
                    compatibility_samples = module.event_model.compatibility(
                        event_context, assignment_context, scaled,
                    )
                    compatibility = compatibility_samples.mean(dim=2).cpu().numpy().astype(np.float32)
                    final_score = log_scores + compatibility_lambda * compatibility
                    final_rank = np.argsort(-final_score, axis=1, kind="stable").astype(np.int32)

            if datasets is None:
                visible_group = destination.create_group("VISIBLE")
                invisible_group = destination.create_group("INVISIBLE")
                joint_group = destination.create_group("JOINT")
                truth_group = destination.create_group("TRUTH")
                observed_group = destination.create_group("OBSERVED")
                datasets = {
                    "event": _create(destination, "EVENT_ID", (), "<i8"),
                    "roots": _create(visible_group, "root_candidate_indices", root_indices.shape[1:], "<i4"),
                    "log_scores": _create(visible_group, "assignment_log_scores", (top_k,), "<f4"),
                    "logz": _create(visible_group, "exact_logZ", (), "<f4"),
                    "probabilities": _create(visible_group, "constrained_probabilities", (top_k,), "<f4"),
                    "truth_assignment": _create(truth_group, "topk_assignment", (top_k,), "?"),
                }
                destination.attrs["root_order"] = json.dumps(list(roots))
                for node in module.spec.dag_order:
                    node_group = visible_group.create_group(f"nodes/{node}")
                    datasets[f"{node}/selected"] = _create(node_group, "selected_candidate", (), "<i4")
                    datasets[f"{node}/truth"] = _create(node_group, "truth_candidate", (), "<i4")
                    datasets[f"{node}/reconstructable"] = _create(node_group, "reconstructable", (), "?")
                    datasets[f"{node}/truth_probability"] = _create(node_group, "truth_probability", (), "<f4")
                    datasets[f"{node}/entropy"] = _create(node_group, "entropy", (), "<f4")
                    datasets[f"{node}/margin"] = _create(node_group, "margin", (), "<f4")
                if samples is not None:
                    datasets["samples"] = _create(invisible_group, "samples", samples.shape[1:], "<f4")
                    datasets["mean"] = _create(invisible_group, "mean", mean.shape[1:], "<f4")
                    datasets["std"] = _create(invisible_group, "std", std.shape[1:], "<f4")
                    datasets["nu_truth"] = _create(truth_group, "NEUTRINO", tuple(batch["invisible_truth"].shape[1:]), "<f4")
                    invisible_group.attrs["actual_nfe"] = nfe
                    if composites is not None:
                        composite_group = destination.create_group("COMPOSITES")
                        truth_composite_group = truth_group.create_group("COMPOSITES")
                        for node, values in composites.items():
                            datasets[f"composite/{node}"] = _create(composite_group, node, values.shape[1:], "<f4")
                            datasets[f"truth_composite/{node}"] = _create(truth_composite_group, node, (4,), "<f4")
                    if observed_met is not None:
                        datasets["observed_met"] = _create(observed_group, "MET", (4,), "<f4")
                if compatibility is not None:
                    datasets["compatibility"] = _create(joint_group, "compatibility_logit", (top_k,), "<f4")
                    datasets["final_score"] = _create(joint_group, "final_score", (top_k,), "<f4")
                    datasets["final_rank"] = _create(joint_group, "rank_indices", (top_k,), "<i4")
                    joint_group.attrs["compatibility_lambda"] = compatibility_lambda
            _append(datasets["event"], batch["event_index"].cpu().numpy().astype(np.int64))
            _append(datasets["roots"], root_indices); _append(datasets["log_scores"], log_scores)
            _append(datasets["logz"], logz); _append(datasets["probabilities"], constrained)
            _append(datasets["truth_assignment"], truth)
            for node in module.spec.dag_order:
                probabilities = output["assignment_probabilities"][node]
                selected = probabilities.argmax(dim=1) - 1
                local_truth = batch["structured"]["pools"][node]["truth_index"]
                safe = local_truth.clamp_min(0)
                rows = torch.arange(len(probabilities), device=probabilities.device)
                truth_probability = probabilities[rows, safe + 1] * (local_truth >= 0)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
                top2 = probabilities.topk(min(2, probabilities.shape[1]), dim=1).values
                margin = top2[:, 0] - top2[:, 1] if top2.shape[1] == 2 else top2[:, 0]
                _append(datasets[f"{node}/selected"], selected.cpu().numpy().astype(np.int32))
                _append(datasets[f"{node}/truth"], local_truth.cpu().numpy().astype(np.int32))
                _append(datasets[f"{node}/reconstructable"], batch["structured"]["pools"][node]["reconstructable"].cpu().numpy())
                _append(datasets[f"{node}/truth_probability"], truth_probability.cpu().numpy().astype(np.float32))
                _append(datasets[f"{node}/entropy"], entropy.cpu().numpy().astype(np.float32))
                _append(datasets[f"{node}/margin"], margin.cpu().numpy().astype(np.float32))
            if samples is not None:
                _append(datasets["samples"], samples); _append(datasets["mean"], mean); _append(datasets["std"], std)
                _append(datasets["nu_truth"], batch["invisible_truth"].cpu().numpy().astype(np.float32))
                if composites is not None:
                    for node, values in composites.items():
                        _append(datasets[f"composite/{node}"], values)
                        _append(datasets[f"truth_composite/{node}"], truth_composites[node])
                if observed_met is not None:
                    _append(datasets["observed_met"], observed_met)
            if compatibility is not None:
                _append(datasets["compatibility"], compatibility); _append(datasets["final_score"], final_score)
                _append(datasets["final_rank"], final_rank)
    validate_prediction(temporary)
    os.replace(temporary, target)


def validate_prediction(path: str | Path) -> None:
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema_version") != PREDICTION_SCHEMA:
            raise ValueError(f"Invalid prediction schema in {path}")
        count = len(source["EVENT_ID"])
        if count == 0 or len(np.unique(source["EVENT_ID"][:])) != count:
            raise ValueError(f"Prediction output {path} is empty or has duplicate EVENT_ID")
        for name in ("VISIBLE/assignment_log_scores", "VISIBLE/exact_logZ", "VISIBLE/constrained_probabilities"):
            if len(source[name]) != count or not np.isfinite(source[name][:]).all():
                raise ValueError(f"Prediction output has invalid {name}")


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value
