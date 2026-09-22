#!/usr/bin/env python3
"""Produce and analyse the controlled Study-10 scorer comparison.

Each model seed is predicted once at Kmax=10, Mmax=64.  The compact feature
artifact then supplies all scorer, ablation and nested-prefix evaluations; no
scorer epoch or K/M point calls the flow again.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.cli.run import build_data, load_trained
from joint_event_reconstruction.configuration import load_experiment, load_visible_config
from joint_event_reconstruction.prediction import predict_hdf5
from joint_event_reconstruction.prediction_schema import validate_prediction
from joint_event_reconstruction.runtime_paths import runtime_paths
from joint_event_reconstruction.study10 import (
    FEATURE_ABLATIONS,
    K_PREFIXES,
    M_PREFIXES,
    coupling_efficiency,
    nested_prefix_reduction,
    rank_correctness,
    select_alpha,
    top_k_coverage,
    train_scorers,
    transition_counts,
    visible_only_scorer_input,
)
from joint_event_reconstruction.topology import (
    PhysicsFeatureEngine,
    PhysicsFeatureScaler,
    PosteriorReducer,
    TopologyKinematics,
    load_topology,
)
from joint_event_reconstruction.topology.truth import compile_truth
from joint_event_reconstruction.topology.physics import LearnedHypothesisScorer


REQUESTED_K = K_PREFIXES
REQUESTED_M = M_PREFIXES
_FEATURE_NAMES = {
    "W_mass_only": lambda name: name.startswith("W_") and name.endswith("_mass"),
    "top_masses_only": lambda name: name in {"top_e_mass", "top_mu_mass"},
    "top_mass_balance": lambda name: name == "top_mass_balance",
    "MET_residual_only": lambda name: name.startswith("met_closure."),
    "W_plus_top": lambda name: (
        (name.startswith("W_") and name.endswith("_mass"))
        or name in {"top_e_mass", "top_mu_mass"}
    ),
    "all": lambda name: True,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_training_sha(checkpoint: Path) -> str:
    identity_path = checkpoint.parent.parent / "artifact_identity.json"
    if not identity_path.is_file():
        raise FileNotFoundError(
            f"Study 10 checkpoint has no training identity: {identity_path}"
        )
    payload = json.loads(identity_path.read_text())
    identity = payload.get("identity")
    training_sha = identity.get("git_commit") if isinstance(identity, dict) else None
    if not isinstance(training_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", training_sha):
        raise ValueError(f"Study 10 checkpoint identity has no training SHA: {identity_path}")
    return training_sha


def _source_sha() -> str:
    value = runtime_paths().source_sha
    if re.fullmatch(r"[0-9a-f]{40}", value):
        return value
    raise ValueError(f"Invalid Study-10 source SHA: {value}")


def _prediction_seed(path: Path) -> int:
    with h5py.File(path, "r") as source:
        return int(source.attrs.get("study_seed", source.attrs.get("prediction_seed", -1)))


@dataclass(frozen=True)
class PredictionRecord:
    path: Path
    event_id: np.ndarray
    states: np.ndarray
    visible_log_score: np.ndarray
    visible_probability: np.ndarray
    truth: np.ndarray
    valid: np.ndarray
    njet: np.ndarray
    entropy: np.ndarray
    margin: np.ndarray
    posterior: np.ndarray
    split: str
    branch_count: int


def _read_prediction(path: Path) -> PredictionRecord:
    path = path.expanduser().resolve()
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        split = source.attrs.get("split", "")
        split = split.decode() if isinstance(split, bytes) else str(split)
        if split not in {"train", "validation"}:
            raise ValueError(f"Study 10 producer accepts train/validation predictions only: {path}")
        event_id = source["EVENT_ID"][:]
        states = source["VISIBLE/root_candidate_indices"][:]
        visible_log_score = source["VISIBLE/assignment_log_scores"][:].astype(np.float32, copy=False)
        visible_probability = source["VISIBLE/constrained_probabilities"][:].astype(np.float32, copy=False)
        truth = source["TRUTH/topk_assignment"][:].astype(bool, copy=False)
        valid = source["VISIBLE/valid_hypotheses"][:].astype(bool, copy=False)
        posterior = source["INVISIBLE/four_vectors"][:].astype(np.float32, copy=False)
        entropy = np.stack(
            [source[f"VISIBLE/nodes/{node}/entropy"][:] for node in source["VISIBLE/nodes"]], axis=1
        ).mean(axis=1).astype(np.float32)
        margin = np.stack(
            [source[f"VISIBLE/nodes/{node}/margin"][:] for node in source["VISIBLE/nodes"]], axis=1
        ).mean(axis=1).astype(np.float32)
        njet = source["OBSERVED/njet"][:].astype(np.int16, copy=False)
        branch_count = int(source["INVISIBLE"].attrs.get("unique_assignment_conditioning_branches", -1))
    if states.ndim != 3 or states.shape[:2] != visible_log_score.shape:
        raise ValueError(f"Study 10 prediction has invalid assignment state shape: {path}")
    if visible_probability.shape != visible_log_score.shape or truth.shape != visible_log_score.shape:
        raise ValueError(f"Study 10 prediction score/truth shape mismatch: {path}")
    if posterior.ndim != 5 or posterior.shape[:2] != visible_log_score.shape:
        raise ValueError(f"Study 10 prediction has invalid posterior shape: {path}")
    if posterior.shape[1] < 10 or posterior.shape[2] < 64:
        raise ValueError(
            f"Study 10 requires one prediction at Kmax=10,Mmax=64: {path}; "
            f"found K={posterior.shape[1]}, M={posterior.shape[2]}"
        )
    if branch_count != 1:
        raise ValueError(f"Study 10 requires one unconditioned posterior branch: {path}")
    return PredictionRecord(
        path, event_id, states, visible_log_score, visible_probability, truth, valid,
        njet, entropy, margin, posterior, split, branch_count,
    )


def _collect_split(data, split: str, expected_event_id: np.ndarray) -> dict:
    loader = {"train": data.train_dataloader, "validation": data.val_dataloader}[split]()
    batches = list(loader)
    if not batches:
        raise ValueError(f"Study 10 {split} loader produced no events")

    event_id = torch.cat(
        [batch["event_index"] for batch in batches]
    ).cpu().numpy()
    expected_event_id = np.asarray(expected_event_id)

    if event_id.ndim != 1 or expected_event_id.ndim != 1:
        raise ValueError(f"Study 10 {split} EVENT_ID arrays must be one-dimensional")
    if event_id.dtype.kind not in "iu" or expected_event_id.dtype.kind not in "iu":
        raise ValueError(f"Study 10 {split} EVENT_ID arrays must contain integers")
    if len(np.unique(event_id)) != len(event_id):
        raise ValueError(f"Study 10 {split} data loader contains duplicate EVENT_IDs")
    if len(np.unique(expected_event_id)) != len(expected_event_id):
        raise ValueError(f"Study 10 {split} predictions contain duplicate EVENT_IDs")

    if (
        len(event_id) != len(expected_event_id)
        or not np.array_equal(np.sort(event_id), np.sort(expected_event_id))
    ):
        missing = np.setdiff1d(expected_event_id, event_id)
        extra = np.setdiff1d(event_id, expected_event_id)
        raise ValueError(
            f"Study 10 {split} prediction/loader EVENT_ID populations differ: "
            f"prediction_count={len(expected_event_id)}, loader_count={len(event_id)}, "
            f"missing_from_loader={missing[:10].tolist()}, "
            f"extra_in_loader={extra[:10].tolist()}"
        )

    # Prediction files define the target event order. Each loader row must
    # carry all its physics information and pool masks into that same order.
    positions = {int(identifier): i for i, identifier in enumerate(event_id)}
    indices = np.asarray(
        [positions[int(identifier)] for identifier in expected_event_id],
        dtype=np.int64,
    )
    permutation = torch.from_numpy(indices)

    aligned_event_id = event_id[indices]
    if not np.array_equal(aligned_event_id, expected_event_id):
        raise ValueError(f"Study 10 {split} EVENT_ID alignment failed")

    def aligned(name: str) -> torch.Tensor:
        values = torch.cat([batch[name] for batch in batches], dim=0)
        return values.index_select(0, permutation)

    spec = data.structured_collator.spec
    pools = {}
    for node in spec.dag_order:
        first = batches[0]["structured"]["pools"][node]
        candidate_mask = torch.cat(
            [
                batch["structured"]["pools"][node]["candidate_mask"]
                for batch in batches
            ],
            dim=0,
        )
        pools[node] = {
            "candidate_mask": candidate_mask.index_select(0, permutation),
            "leaf_indices": first["leaf_indices"],
            "child_indices": first["child_indices"],
        }

    return {
        "event_id": aligned_event_id,
        "physics_vectors": aligned("physics_vectors"),
        "met_xy": aligned("met_xy"),
        "invisible_truth": aligned("invisible_truth"),
        "pools": pools,
    }


def _fit_scaler(data, spec, kinematics, features, source_identity: str) -> PhysicsFeatureScaler:
    residuals = []
    names = None
    for batch in data.train_dataloader():
        truth_indices = torch.stack(
            [batch["structured"]["pools"][node]["truth_index"] for node in spec.dag_order], dim=-1
        ).unsqueeze(1)
        valid = truth_indices.ge(0).all(dim=-1)
        objects = kinematics.compose(
            batch["structured"]["pools"], batch["physics_vectors"], truth_indices,
            invisible_momenta=batch["invisible_truth"].unsqueeze(1),
        )
        result = features.evaluate(objects, observed_met=batch["met_xy"])
        names = result.feature_names if names is None else names
        if result.feature_names != names:
            raise ValueError("Study 10 physics feature order changed across training batches")
        values = result.tensor[:, 0, 0]
        selected = valid[:, 0] & torch.isfinite(values).all(dim=-1)
        if selected.any():
            residuals.append(values[selected].cpu())
    if not residuals:
        raise RuntimeError("Study 10 could not fit a training-derived physics scaler")
    return PhysicsFeatureScaler.fit(
        torch.cat(residuals), names, topology_signature=spec.signature,
        physics_signature=features.spec.signature, source_data_identity=source_identity,
    )


def _feature_masks(names: tuple[str, ...]) -> list[torch.Tensor]:
    masks = []
    for label in FEATURE_ABLATIONS:
        selected = [_FEATURE_NAMES[label](name) for name in names]
        if not any(selected):
            raise ValueError(f"Study 10 feature ablation {label} selected no features")
        masks.append(torch.tensor(selected, dtype=torch.bool))
    return masks


def _write_feature_artifact(
    path: Path,
    prediction: PredictionRecord,
    data_split: dict,
    scaler: PhysicsFeatureScaler,
    spec,
    kinematics: TopologyKinematics,
    features: PhysicsFeatureEngine,
    chunk_size: int,
    source_sha: str,
    checkpoint: Path,
    study_manifest_sha256: str,
    gate_digest: str,
    data_subset_identity: dict,
) -> None:
    reducer = PosteriorReducer()
    means, stds, rmss, evidences, oracle_scores, ablations, nested = [], [], [], [], [], [], []
    total = len(prediction.event_id)
    masks: list[torch.Tensor] | None = None
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        pools = {
            node: {
                "candidate_mask": values["candidate_mask"][start:stop],
                "leaf_indices": values["leaf_indices"],
                "child_indices": values["child_indices"],
            }
            for node, values in data_split["pools"].items()
        }
        states = torch.from_numpy(prediction.states[start:stop].astype(np.int64))
        posterior = torch.from_numpy(prediction.posterior[start:stop])
        valid = torch.from_numpy(prediction.valid[start:stop])
        objects = kinematics.compose(
            pools, data_split["physics_vectors"][start:stop], states,
            invisible_four_vectors=posterior,
        )
        result = features.evaluate(objects, observed_met=data_split["met_xy"][start:stop])
        scaled = scaler.transform(
            result.tensor, feature_names=result.feature_names,
            topology_signature=spec.signature, physics_signature=result.physics_signature,
        )
        valid_samples = valid.unsqueeze(-1).expand(-1, -1, scaled.shape[2])
        reduction = reducer.reduce(scaled, valid_samples=valid_samples, valid_hypotheses=valid)
        means.append(reduction.mean.detach().cpu().numpy())
        stds.append(reduction.std.detach().cpu().numpy())
        rmss.append(reduction.rms.detach().cpu().numpy())
        evidences.append(reduction.analytic_log_evidence.detach().cpu().numpy())
        prefix = nested_prefix_reduction(scaled, valid)
        nested.append(prefix.detach().cpu().numpy())
        if masks is None:
            masks = _feature_masks(result.feature_names)
        ablation_scores = []
        for mask in masks:
            masked = scaled[..., mask]
            ablation = reducer.reduce(masked, valid_samples=valid_samples, valid_hypotheses=valid)
            ablation_scores.append(ablation.analytic_log_evidence.detach().cpu().numpy())
        ablations.append(np.stack(ablation_scores, axis=-1))

        truth_objects = kinematics.compose(
            pools, data_split["physics_vectors"][start:stop], states,
            invisible_momenta=data_split["invisible_truth"][start:stop].unsqueeze(1).unsqueeze(1),
        )
        truth_result = features.evaluate(
            truth_objects, observed_met=data_split["met_xy"][start:stop]
        )
        truth_scaled = scaler.transform(
            truth_result.tensor, feature_names=truth_result.feature_names,
            topology_signature=spec.signature, physics_signature=truth_result.physics_signature,
        )
        truth_reduction = reducer.reduce(
            truth_scaled,
            valid_samples=valid.unsqueeze(-1),
            valid_hypotheses=valid,
        )
        oracle_scores.append(truth_reduction.analytic_log_evidence.detach().cpu().numpy())

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with h5py.File(temporary, "w") as destination:
        destination.attrs.update({
            "schema": "jer_study10_features_v1",
            "split": prediction.split,
            "source_sha": source_sha,
            "study_manifest_sha256": study_manifest_sha256,
            "resolved_gate_digest": gate_digest,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "checkpoint_training_sha": _checkpoint_training_sha(checkpoint),
            "data_subset_identity": json.dumps(data_subset_identity, sort_keys=True),
            "prediction_settings": json.dumps(
                {
                    "top_k": 10,
                    "posterior_samples": 64,
                    "flow_steps": 4,
                    "solver": "heun",
                    "posterior_branches": prediction.branch_count,
                },
                sort_keys=True,
            ),
            "prediction": str(prediction.path),
            "prediction_sha256": _sha256(prediction.path),
            "topology_signature": spec.signature,
            "physics_signature": spec.physics_signature or "none",
            "physics_scaler_sha256": scaler.identity_sha256,
            "physics_scaler_fit_split": "train",
            "feature_names": json.dumps(list(scaler.feature_names)),
            "Kmax": 10,
            "Mmax": 64,
            "posterior_branches": prediction.branch_count,
            "test_split_used": False,
        })
        destination.create_dataset("EVENT_ID", data=prediction.event_id)
        destination.create_dataset("visible_log_score", data=prediction.visible_log_score, compression="gzip")
        destination.create_dataset("visible_probability", data=prediction.visible_probability, compression="gzip")
        destination.create_dataset("truth", data=prediction.truth, compression="gzip")
        destination.create_dataset("valid_hypotheses", data=prediction.valid, compression="gzip")
        destination.create_dataset("njet", data=prediction.njet)
        destination.create_dataset("assignment_entropy", data=prediction.entropy)
        destination.create_dataset("top1_top2_margin", data=prediction.margin)
        physics = destination.create_group("PHYSICS")
        physics.create_dataset("mean", data=np.concatenate(means), compression="gzip")
        physics.create_dataset("std", data=np.concatenate(stds), compression="gzip")
        physics.create_dataset("rms", data=np.concatenate(rmss), compression="gzip")
        physics.create_dataset("analytic_score", data=np.concatenate(evidences), compression="gzip")
        physics.create_dataset("truth_neutrino_score", data=np.concatenate(oracle_scores), compression="gzip")
        physics.create_dataset("ablation_scores", data=np.concatenate(ablations), compression="gzip")
        physics.create_dataset("nested_analytic_scores", data=np.concatenate(nested, axis=2), compression="gzip")
        physics.attrs["ablation_names"] = json.dumps(list(FEATURE_ABLATIONS))
        physics.attrs["K_prefixes"] = json.dumps(list(REQUESTED_K))
        physics.attrs["M_prefixes"] = json.dumps(list(REQUESTED_M))
    temporary.replace(path)


def _attribute_text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _validate_feature_tensors(features: dict, path: Path) -> None:
    event_id = features["event_id"]
    truth = features["truth"]
    valid = features["valid"]
    if event_id.ndim != 1 or truth.ndim != 2 or valid.shape != truth.shape:
        raise ValueError(f"Study-10 feature tensors have invalid event/mask shape: {path}")
    if truth.dtype is not torch.bool or valid.dtype is not torch.bool:
        raise ValueError(f"Study-10 feature tensors have non-boolean masks: {path}")
    batch, candidates = truth.shape
    if batch <= 0 or candidates != 10:
        raise ValueError(f"Study-10 feature tensors require [B,10] masks: {path}")
    if len(torch.unique(event_id)) != batch:
        raise ValueError(f"Study-10 feature tensors contain duplicate EVENT_IDs: {path}")
    if (truth & ~valid).any():
        raise ValueError(f"Study-10 feature tensors mark truth on invalid candidates: {path}")
    if truth.sum(dim=1).gt(1).any():
        raise ValueError(f"Study-10 feature tensors contain multiple truth candidates: {path}")
    if (~valid).all(dim=1).any():
        raise ValueError(f"Study-10 feature tensors contain an event with no valid candidate: {path}")

    for name in ("visible_log_score", "visible_probability", "analytic", "oracle"):
        value = features[name]
        if value.shape != (batch, candidates):
            raise ValueError(f"Study-10 feature tensor has invalid shape for {name}: {path}")
        if not torch.isfinite(value[valid]).all():
            raise FloatingPointError(
                f"Study-10 feature tensor {name} is non-finite for a valid candidate: {path}"
            )
    widths = set()
    for name in ("physics_mean", "physics_std", "physics_rms"):
        value = features[name]
        if value.ndim != 3 or value.shape[:2] != (batch, candidates):
            raise ValueError(f"Study-10 physics feature tensor has invalid shape for {name}: {path}")
        widths.add(value.shape[-1])
        if not torch.isfinite(value[valid]).all():
            raise FloatingPointError(
                f"Study-10 physics feature tensor {name} is non-finite for a valid candidate: {path}"
            )
    if len(widths) != 1:
        raise ValueError(f"Study-10 reduced physics feature widths disagree: {path}")
    for name in ("entropy", "margin"):
        value = features[name]
        if value.shape not in {(batch,), (batch, candidates)}:
            raise ValueError(f"Study-10 feature tensor has invalid shape for {name}: {path}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Study-10 feature tensor {name} is non-finite: {path}")
    ablations = features["ablations"]
    if ablations.ndim != 3 or ablations.shape[:2] != (batch, candidates):
        raise ValueError(f"Study-10 ablation feature tensor has an invalid shape: {path}")
    if not torch.isfinite(ablations[valid]).all():
        raise FloatingPointError(f"Study-10 ablation scores are non-finite for valid candidates: {path}")
    nested = features["nested"]
    if nested.shape != (len(REQUESTED_K), len(REQUESTED_M), batch, candidates):
        raise ValueError(f"Study-10 nested K/M tensor has an invalid shape: {path}")
    for k_index, k in enumerate(REQUESTED_K):
        active = valid[:, :k]
        for m_index in range(len(REQUESTED_M)):
            active_values = nested[k_index, m_index, :, :k][active]
            if not torch.isfinite(active_values).all():
                raise FloatingPointError(f"Study-10 nested K/M scores are non-finite: {path}")


def _load_features(path: Path) -> dict[str, torch.Tensor | list[str] | str]:
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema", "") != "jer_study10_features_v1":
            raise ValueError(f"Unsupported Study-10 feature artifact: {path}")
        if source.attrs.get("test_split_used", True):
            raise ValueError(f"Study-10 feature artifact claims test data: {path}")
        required_provenance = (
            "source_sha",
            "study_manifest_sha256",
            "resolved_gate_digest",
            "checkpoint_sha256",
            "checkpoint_training_sha",
            "data_subset_identity",
            "prediction_settings",
            "physics_scaler_fit_split",
        )
        missing_provenance = [
            name for name in required_provenance
            if not str(source.attrs.get(name, "")).strip()
        ]
        if missing_provenance:
            raise ValueError(
                f"Study-10 feature artifact is missing provenance {missing_provenance}: {path}"
            )
        feature_names = json.loads(source.attrs["feature_names"])
        features = {
            "event_id": torch.from_numpy(source["EVENT_ID"][:]),
            "visible_log_score": torch.from_numpy(source["visible_log_score"][:]),
            "visible_probability": torch.from_numpy(source["visible_probability"][:]),
            "truth": torch.from_numpy(source["truth"][:]).bool(),
            "valid": torch.from_numpy(source["valid_hypotheses"][:]).bool(),
            "entropy": torch.from_numpy(source["assignment_entropy"][:]),
            "margin": torch.from_numpy(source["top1_top2_margin"][:]),
            "physics_mean": torch.from_numpy(source["PHYSICS/mean"][:]),
            "physics_std": torch.from_numpy(source["PHYSICS/std"][:]),
            "physics_rms": torch.from_numpy(source["PHYSICS/rms"][:]),
            "analytic": torch.from_numpy(source["PHYSICS/analytic_score"][:]),
            "oracle": torch.from_numpy(source["PHYSICS/truth_neutrino_score"][:]),
            "ablations": torch.from_numpy(source["PHYSICS/ablation_scores"][:]),
            "nested": torch.from_numpy(source["PHYSICS/nested_analytic_scores"][:]),
            "feature_names": feature_names,
            "source_sha": _attribute_text(source.attrs["source_sha"]),
            "feature_producer_source_sha": _attribute_text(source.attrs["source_sha"]),
            "study_manifest_sha256": _attribute_text(source.attrs["study_manifest_sha256"]),
            "resolved_gate_digest": _attribute_text(source.attrs["resolved_gate_digest"]),
            "checkpoint_sha256": _attribute_text(source.attrs["checkpoint_sha256"]),
            "checkpoint_training_sha": _attribute_text(source.attrs["checkpoint_training_sha"]),
            "data_subset_identity": _attribute_text(source.attrs["data_subset_identity"]),
            "physics_scaler_sha256": _attribute_text(source.attrs["physics_scaler_sha256"]),
            "physics_scaler_fit_split": _attribute_text(source.attrs["physics_scaler_fit_split"]),
        }
    _validate_feature_tensors(features, path)
    return features


def _atomic_json(path: Path, payload: object) -> None:
    """Publish one analysis record only after it has been written completely."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _feature_artifact_contract(
    path: Path,
    *,
    expected_split: str,
    expected_source_sha: str | None = None,
    expected_feature_producer_sha: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_gate_digest: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict:
    """Validate the cheap, provenance-bearing part of one feature artifact.

    This deliberately opens and inspects the HDF5 file rather than treating a
    non-empty pathname as a completed computation.  The full tensor checks are
    still performed by ``_load_features`` before scorer training.
    """
    required = {
        "EVENT_ID": ("EVENT_ID",),
        "visible_log_score": ("visible_log_score",),
        "visible_probability": ("visible_probability",),
        "truth": ("truth",),
        "valid_hypotheses": ("valid_hypotheses",),
        "physics_mean": ("PHYSICS/mean",),
        "physics_std": ("PHYSICS/std",),
        "physics_rms": ("PHYSICS/rms",),
        "analytic_score": ("PHYSICS/analytic_score",),
        "truth_neutrino_score": ("PHYSICS/truth_neutrino_score",),
        "ablation_scores": ("PHYSICS/ablation_scores",),
        "nested_analytic_scores": ("PHYSICS/nested_analytic_scores",),
    }
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Study-10 feature artifact is absent or empty: {path}")
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema") != "jer_study10_features_v1":
            raise ValueError(f"Unsupported Study-10 feature artifact: {path}")
        split = str(source.attrs.get("split", ""))
        if split != expected_split:
            raise ValueError(f"Study-10 feature split mismatch for {path}: {split!r}")
        if bool(source.attrs.get("test_split_used", True)):
            raise ValueError(f"Study-10 feature artifact claims test data: {path}")
        for key, candidates in required.items():
            if not any(candidate in source for candidate in candidates):
                raise ValueError(f"Study-10 feature artifact is missing {key}: {path}")
        event_id = source["EVENT_ID"][:]
        if event_id.ndim != 1 or len(np.unique(event_id)) != len(event_id):
            raise ValueError(f"Study-10 feature artifact has invalid EVENT_IDs: {path}")
        count = len(event_id)
        truth = source["truth"][:].astype(bool, copy=False)
        valid = source["valid_hypotheses"][:].astype(bool, copy=False)
        if truth.shape != valid.shape or truth.shape[0] != count or truth.shape[1] != 10:
            raise ValueError(f"Study-10 feature artifact has invalid truth/valid shape: {path}")
        if np.any(truth & ~valid):
            raise ValueError(f"Study-10 feature artifact marks truth on invalid candidates: {path}")
        if np.any(valid.sum(axis=1) == 0):
            raise ValueError(f"Study-10 feature artifact has an event with no valid candidate: {path}")
        if np.any(truth.sum(axis=1) > 1):
            raise ValueError(f"Study-10 feature artifact has multiple truth candidates: {path}")
        for name in (
            "visible_log_score", "visible_probability", "truth",
            "valid_hypotheses", "PHYSICS/mean", "PHYSICS/std", "PHYSICS/rms",
            "PHYSICS/analytic_score", "PHYSICS/truth_neutrino_score",
            "PHYSICS/ablation_scores",
        ):
            if source[name].shape[0] != count:
                raise ValueError(f"Study-10 feature event dimension mismatch in {path}: {name}")
        nested = source["PHYSICS/nested_analytic_scores"]
        if nested.shape != (len(REQUESTED_K), len(REQUESTED_M), count, 10):
            raise ValueError(f"Study-10 feature nested K/M shape is invalid: {path}")
        attrs = {
            "source_sha": _attribute_text(source.attrs.get("source_sha", "")),
            "study_manifest_sha256": _attribute_text(source.attrs.get("study_manifest_sha256", "")),
            "resolved_gate_digest": _attribute_text(source.attrs.get("resolved_gate_digest", "")),
            "checkpoint_sha256": _attribute_text(source.attrs.get("checkpoint_sha256", "")),
            "checkpoint_training_sha": _attribute_text(source.attrs.get("checkpoint_training_sha", "")),
            "checkpoint": _attribute_text(source.attrs.get("checkpoint", "")),
            "physics_scaler_sha256": _attribute_text(source.attrs.get("physics_scaler_sha256", "")),
            "physics_scaler_fit_split": _attribute_text(source.attrs.get("physics_scaler_fit_split", "")),
            "prediction_sha256": _attribute_text(source.attrs.get("prediction_sha256", "")),
        }
    expected = {
        "source_sha": expected_feature_producer_sha or expected_source_sha,
        "study_manifest_sha256": expected_manifest_sha256,
        "resolved_gate_digest": expected_gate_digest,
        "checkpoint_sha256": expected_checkpoint_sha256,
    }
    for name, value in expected.items():
        if value is not None and attrs[name] != str(value):
            raise ValueError(
                f"Study-10 feature provenance mismatch for {path}: "
                f"{name}={attrs[name]!r}, expected {value!r}"
            )
    if not all(
        attrs[name]
        for name in (
            "source_sha",
            "study_manifest_sha256",
            "resolved_gate_digest",
            "checkpoint_sha256",
            "checkpoint_training_sha",
            "checkpoint",
            "physics_scaler_sha256",
            "prediction_sha256",
        )
    ):
        raise ValueError(f"Study-10 feature artifact has incomplete provenance: {path}")
    if not re.fullmatch(r"[0-9a-f]{40}", attrs["source_sha"]):
        raise ValueError(f"Study-10 feature artifact has an invalid producer source SHA: {path}")
    if attrs["physics_scaler_fit_split"] != "train":
        raise ValueError(f"Study-10 physics scaler was not fit on train: {path}")
    supported = (truth & valid).any(axis=1)
    return {
        "event_ids": event_id,
        "events_total": count,
        "events_supported": int(supported.sum()),
        "events_unsupported": int((~supported).sum()),
        "feature_producer_source_sha": attrs["source_sha"],
        **attrs,
    }


def _seed_record_contract(
    path: Path,
    *,
    expected_seed: int,
    expected_source_sha: str,
    expected_manifest_sha256: str,
    expected_gate_digest: str,
    expected_feature_producer_sha: str | None = None,
    expected_checkpoint: Path | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict:
    """Validate one resumable seed result and all of its retained products."""
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"Study-10 seed record is absent or empty: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Study-10 seed record is not valid JSON: {path}: {error}") from error
    if not isinstance(payload, dict) or int(payload.get("seed", -1)) != int(expected_seed):
        raise ValueError(f"Study-10 seed record has the wrong seed: {path}")
    for name, expected in (
        ("source_sha", expected_source_sha),
        ("study_manifest_sha256", expected_manifest_sha256),
        ("resolved_gate_digest", expected_gate_digest),
    ):
        if payload.get(name) != expected:
            raise ValueError(
                f"Study-10 seed record provenance mismatch: {path}: "
                f"{name}={payload.get(name)!r}, expected {expected!r}"
            )
    feature_producer_sha = payload.get("feature_producer_source_sha")
    if not isinstance(feature_producer_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", feature_producer_sha):
        raise ValueError(f"Study-10 seed record has no feature-producer source SHA: {path}")
    if expected_feature_producer_sha is not None and feature_producer_sha != expected_feature_producer_sha:
        raise ValueError(
            f"Study-10 seed record feature-producer mismatch: {path}: "
            f"feature_producer_source_sha={feature_producer_sha!r}, "
            f"expected {expected_feature_producer_sha!r}"
        )
    if expected_checkpoint is not None and Path(str(payload.get("checkpoint", ""))).resolve() != expected_checkpoint.resolve():
        raise ValueError(f"Study-10 seed record checkpoint path mismatch: {path}")
    if expected_checkpoint_sha256 is not None and payload.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError(f"Study-10 seed record checkpoint checksum mismatch: {path}")
    checkpoint_training_sha = payload.get("checkpoint_training_sha")
    if not isinstance(checkpoint_training_sha, str) or not re.fullmatch(
        r"[0-9a-f]{40}", checkpoint_training_sha
    ):
        raise ValueError(f"Study-10 seed record has no checkpoint training SHA: {path}")
    if expected_checkpoint is not None and checkpoint_training_sha != _checkpoint_training_sha(expected_checkpoint):
        raise ValueError(f"Study-10 seed record checkpoint training identity mismatch: {path}")
    try:
        scorer_training_epochs = int(payload["scorer_training_epochs"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Study-10 seed record has no scorer-training epoch count: {path}") from error
    if scorer_training_epochs <= 0:
        raise ValueError(f"Study-10 seed record has an invalid scorer-training epoch count: {path}")
    comparisons = payload.get("comparisons")
    required_comparisons = {
        "visible_baseline", "matched_visible_only_control", "analytic_physics",
        "learned_physics", "truth_neutrino_oracle", "top_k_ceiling",
        "coupling_efficiency", "feature_ablations", "nested_K_M_convergence",
    }
    if not isinstance(comparisons, dict) or not required_comparisons <= set(comparisons):
        raise ValueError(f"Study-10 seed record is missing scientific controls: {path}")
    checkpoint_sha256 = str(payload.get("checkpoint_sha256", ""))
    train_path = Path(str(payload.get("train_feature_artifact", "")))
    validation_path = Path(str(payload.get("validation_feature_artifact", "")))
    train = _feature_artifact_contract(
        train_path, expected_split="train", expected_source_sha=expected_source_sha,
        expected_feature_producer_sha=feature_producer_sha,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_gate_digest=expected_gate_digest,
        expected_checkpoint_sha256=checkpoint_sha256,
    )
    validation = _feature_artifact_contract(
        validation_path, expected_split="validation", expected_source_sha=expected_source_sha,
        expected_feature_producer_sha=feature_producer_sha,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_gate_digest=expected_gate_digest,
        expected_checkpoint_sha256=checkpoint_sha256,
    )
    for artifact in (train, validation):
        if artifact["checkpoint_training_sha"] != checkpoint_training_sha:
            raise ValueError(f"Study-10 feature artifact checkpoint training identity mismatch: {path}")
        if Path(artifact["checkpoint"]).resolve() != Path(str(payload.get("checkpoint", ""))).resolve():
            raise ValueError(f"Study-10 feature artifact checkpoint path mismatch: {path}")
    if set(train["event_ids"].tolist()) & set(validation["event_ids"].tolist()):
        raise ValueError(f"Study-10 seed record train/validation populations overlap: {path}")
    population_fields = {
        "training_events_total": train["events_total"],
        "training_events_supported": train["events_supported"],
        "training_events_unsupported": train["events_unsupported"],
        "validation_events_total": validation["events_total"],
        "validation_events_supported": validation["events_supported"],
        "validation_events_unsupported": validation["events_unsupported"],
    }
    for name, expected in population_fields.items():
        if int(payload.get(name, -1)) != int(expected):
            raise ValueError(
                f"Study-10 seed record population mismatch: {path}: "
                f"{name}={payload.get(name)!r}, expected {expected!r}"
            )
    scorer_artifacts = payload.get("scorer_artifacts")
    if not isinstance(scorer_artifacts, dict) or not {"physics", "visible_only"} <= set(scorer_artifacts):
        raise ValueError(f"Study-10 seed record has no scorer identities: {path}")
    for name in ("physics", "visible_only"):
        item = scorer_artifacts[name]
        scorer_path = Path(str(item.get("path", "")))
        if not scorer_path.is_file() or scorer_path.stat().st_size <= 0:
            raise ValueError(f"Study-10 scorer artifact is missing: {scorer_path}")
        if _sha256(scorer_path) != item.get("sha256"):
            raise ValueError(f"Study-10 scorer artifact checksum mismatch: {scorer_path}")
        state = torch.load(scorer_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Study-10 scorer artifact is not a state dict: {scorer_path}")
    return payload


def _input_tensors(features: dict) -> tuple[torch.Tensor, torch.Tensor]:
    analytic = features["analytic"].float()
    valid = features["valid"].bool()
    candidate_mask = valid.unsqueeze(-1)

    def sanitise(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.ndim == 1:
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Study-10 {name} is non-finite")
            return value
        if value.ndim == 2:
            mask = valid
        elif value.ndim == 3 and value.shape[:2] == valid.shape:
            mask = candidate_mask
        else:
            raise ValueError(f"Study-10 scorer input has an invalid shape for {name}")
        if mask.shape != value.shape:
            mask = mask.expand_as(value)
        if not torch.isfinite(value[mask]).all():
            raise FloatingPointError(
                f"Study-10 {name} is non-finite for a valid hypothesis"
            )
        return torch.where(torch.isfinite(value), value, torch.zeros_like(value))

    analytic_for_scorer = sanitise(analytic, "analytic score")
    visible_log_score = sanitise(features["visible_log_score"].float(), "visible log score")
    visible_probability = sanitise(features["visible_probability"].float(), "visible probability")
    physics_mean = sanitise(features["physics_mean"].float(), "physics mean")
    physics_std = sanitise(features["physics_std"].float(), "physics std")
    physics_rms = sanitise(features["physics_rms"].float(), "physics rms")
    entropy = sanitise(features["entropy"].float(), "assignment entropy")
    margin = sanitise(features["margin"].float(), "top-1/top-2 margin")
    reduction = torch.cat(
        (physics_mean, physics_std, physics_rms, analytic_for_scorer.unsqueeze(-1)),
        dim=-1,
    )
    physics = torch.cat(
        (reduction, visible_log_score.unsqueeze(-1), visible_probability.unsqueeze(-1)),
        dim=-1,
    ).float()
    visible = visible_only_scorer_input(
        visible_log_score, visible_probability, entropy, margin,
        target_features=physics.shape[-1],
    )
    return physics, visible


def _transitions_for(score: torch.Tensor, features: dict) -> dict:
    baseline = rank_correctness(features["visible_log_score"], features["truth"], features["valid"])
    selected = rank_correctness(score, features["truth"], features["valid"])
    supported = (features["truth"] & features["valid"]).any(dim=1)
    supported_accuracy = (
        float(selected[supported].float().mean()) if supported.any() else None
    )
    return {
        "accuracy": float(selected.float().mean()),
        "supported_only_accuracy": supported_accuracy,
        "delta_vs_visible": float(selected.float().mean() - baseline.float().mean()),
        "transitions": transition_counts(baseline, selected),
    }


def _population_counts(features: dict) -> dict[str, int | float]:
    truth = features["truth"]
    valid = features["valid"]
    supported = (truth & valid).any(dim=1)
    total = int(truth.shape[0])
    supported_count = int(supported.sum())
    unsupported_count = total - supported_count
    return {
        "events_total": total,
        "events_supported": supported_count,
        "events_unsupported": unsupported_count,
        "supported_fraction": supported_count / total if total else 0.0,
    }


def _population_truth_diagnostics(
    train: dict,
    validation: dict,
    experiment: Path,
) -> dict:
    """Classify unsupported top-K events using the canonical truth contract.

    This reads only the raw labels/object masks for the already selected
    train/validation EVENT_IDs.  It does not rebuild assignment pools or run a
    model.  ``compile_truth`` is the same topology truth utility used by the
    structured data boundary, so the diagnostic distinguishes configured-input
    unavailability from a correct assignment ranked below stored K=10.
    """
    config = load_experiment(experiment, ["training.workers=0"])
    spec = load_topology(
        config.topology_config,
        OmegaConf.to_container(config.data.truth_bindings, resolve=True),
    )
    sources = tuple(sorted({role.source for role in spec.visible_leaf_roles}))
    if not sources:
        raise ValueError("Study-10 population diagnostic found no visible truth sources")

    identities = [json.loads(features["data_subset_identity"]) for features in (train, validation)]
    source_paths = {
        str(identity.get("data") or identity.get("source", {}).get("path", ""))
        for identity in identities
    }
    if len(source_paths) != 1 or not next(iter(source_paths)):
        raise ValueError("Study-10 feature artifacts do not share one canonical data source")
    source_path = Path(next(iter(source_paths))).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Study-10 population diagnostic source is missing: {source_path}")

    selected_widths = {
        source: int(config.input_definition.collections[source].count)
        for source in sources
    }
    with h5py.File(source_path, "r") as source_file:
        full_widths = {
            source: int(source_file[f"LABELS/{source}"].shape[1])
            for source in sources
        }
    roots = tuple(spec.roots)

    def classify(features: dict) -> dict:
        event_ids = features["event_id"].cpu().numpy()
        supported = ((features["truth"] & features["valid"]).any(dim=1)).cpu().numpy()
        with h5py.File(source_path, "r") as source_file:
            raw_event_ids = source_file["EVENT_ID"][:]
            positions = np.searchsorted(raw_event_ids, event_ids)
            if np.any(positions >= len(raw_event_ids)) or not np.array_equal(
                raw_event_ids[positions], event_ids
            ):
                raise ValueError(
                    f"Study-10 population diagnostic EVENT_IDs are absent from {source_path}"
                )
            order = np.argsort(positions)
            sorted_positions = positions[order]
            labels = {}
            masks = {}
            for source_name in sources:
                label_values = np.asarray(source_file[f"LABELS/{source_name}"][sorted_positions])
                input_values = source_file[f"INPUTS/{source_name}"][sorted_positions]
                labels[source_name] = torch.from_numpy(label_values)
                masks[source_name] = torch.from_numpy(
                    np.isfinite(input_values["pt"]) & (input_values["pt"] > 0.0)
                )

        inverse = np.argsort(order)
        labels = {name: values[inverse] for name, values in labels.items()}
        masks = {name: values[inverse] for name, values in masks.items()}
        selected_truth = compile_truth(
            spec, labels, masks, selected_widths=selected_widths
        )
        full_truth = compile_truth(spec, labels, masks, selected_widths=full_widths)
        selected_complete = torch.stack(
            [selected_truth.nodes[root].visible_reconstructable for root in roots], dim=1
        ).all(dim=1).numpy()
        full_complete = torch.stack(
            [full_truth.nodes[root].visible_reconstructable for root in roots], dim=1
        ).all(dim=1).numpy()
        if np.any(supported & ~selected_complete):
            raise ValueError(
                "Study-10 stored truth support disagrees with canonical selected-input truth"
            )
        unsupported = ~supported
        truncation = unsupported & selected_complete
        outside_width = unsupported & ~selected_complete & full_complete
        unavailable = unsupported & ~full_complete
        return {
            "events_total": int(len(event_ids)),
            "events_supported_top_k": int(supported.sum()),
            "events_unsupported_top_k": int(unsupported.sum()),
            "complete_candidate_supported": int(selected_complete.sum()),
            "complete_candidate_unsupported": int((~selected_complete).sum()),
            "stored_top_k_truncation": int(truncation.sum()),
            "truth_outside_configured_input_width": int(outside_width.sum()),
            "truth_unavailable_or_undefined": int(unavailable.sum()),
            "support_vs_complete_candidate_mismatch": int(
                np.count_nonzero(supported != selected_complete)
            ),
        }

    return {
        "status": "classified_from_canonical_truth",
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "candidate_definition": {
            "stored_K": 10,
            "configured_input_widths": selected_widths,
            "full_source_widths": full_widths,
            "truth_matching": "topology.truth.compile_truth",
            "object_validity": "finite positive pt",
        },
        "train": classify(train),
        "validation": classify(validation),
    }


def _generate_prediction(experiment: Path, checkpoint: Path, output: Path, split: str, device: str) -> Path:
    if output.is_file():
        record = _read_prediction(output)
        if record.split == split:
            return output
    overrides = [
        "inference.assignment.top_k=10",
        "inference.invisible.samples_per_assignment=64",
        "inference.invisible.flow.steps=4",
        "inference.invisible.flow.solver=heun",
        "training.workers=0",
    ]
    config = load_experiment(experiment, overrides)
    visible = load_visible_config(config)
    data = build_data(config, visible)
    if device == "cpu":
        data.config.pin_memory = False
    data.setup("predict")
    module = load_trained(config, visible, data, str(checkpoint)).to(device).eval()
    predict_hdf5(
        module, data, data.invisible_scaler, output, split=split, top_k=10,
        posterior_samples=64, device=device, prediction_seed=int(config.inference.seed),
        checkpoint_path=str(checkpoint),
    )
    return output


def _build_one_seed(
    validation_prediction: Path,
    training_prediction: Path | None,
    experiment: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    chunk_size: int,
    seed: int,
    study_manifest_sha256: str,
    gate_digest: str,
) -> tuple[Path, Path]:
    validation = _read_prediction(validation_prediction)
    if validation.posterior.shape[1:] != (10, 64, validation.posterior.shape[3], 4):
        validation_prediction = _generate_prediction(
            experiment, checkpoint, output.with_name(f"seed_{seed}_validation_prediction.h5"), "validation", device
        )
        validation = _read_prediction(validation_prediction)
    if training_prediction is None:
        training_prediction = output.with_name(f"seed_{seed}_training_prediction.h5")
    training_prediction = _generate_prediction(
        experiment, checkpoint, training_prediction, "train", device
    )
    training = _read_prediction(training_prediction)
    paths = runtime_paths()
    config = load_experiment(experiment, ["training.workers=0"])
    visible = load_visible_config(config)
    data = build_data(config, visible)
    if device == "cpu":
        data.config.pin_memory = False
    data.setup("predict")
    spec = load_topology(
        config.topology_config, OmegaConf.to_container(config.data.truth_bindings, resolve=True)
    )
    if spec.physics is None or tuple(spec.roots) != tuple(spec.dag_order):
        raise ValueError("Study 10 producer currently requires root-only ttbar2L assignment nodes")
    kinematics = TopologyKinematics(spec)
    physics = PhysicsFeatureEngine(spec.physics)
    source_identity = json.dumps(
        {"data": str(data.h5_path), "splits": data.split_metadata}, sort_keys=True
    )
    scaler = _fit_scaler(data, spec, kinematics, physics, source_identity)
    train_data = _collect_split(data, "train", training.event_id)
    validation_data = _collect_split(data, "validation", validation.event_id)
    train_path = output.parent / f"seed_{seed}_train_features.h5"
    validation_path = output.parent / f"seed_{seed}_validation_features.h5"
    _write_feature_artifact(
        train_path, training, train_data, scaler, spec, kinematics, physics, chunk_size,
        _source_sha(), checkpoint, study_manifest_sha256, gate_digest, data.split_metadata,
    )
    _write_feature_artifact(
        validation_path, validation, validation_data, scaler, spec, kinematics, physics, chunk_size,
        _source_sha(), checkpoint, study_manifest_sha256, gate_digest, data.split_metadata,
    )
    return train_path, validation_path


def _analyse_seed(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    epochs: int,
    seed: int,
    *,
    experiment: Path | None = None,
    population_diagnostics: dict | None = None,
) -> dict:
    train = _load_features(train_path)
    validation = _load_features(validation_path)
    if population_diagnostics is None and experiment is not None:
        population_diagnostics = _population_truth_diagnostics(train, validation, experiment)
    train_population = _population_counts(train)
    validation_population = _population_counts(validation)
    train_support = (train["truth"] & train["valid"]).any(dim=1)
    if not train_support.any():
        raise ValueError("Study-10 analysis has zero truth-supported training events")
    train_physics, train_visible = _input_tensors(train)
    validation_physics, validation_visible = _input_tensors(validation)
    scorers = train_scorers(
        train["visible_log_score"].float(), train["visible_probability"].float(),
        train_physics, train_visible, train["truth"], train["valid"], epochs=epochs,
    )
    scorer_dir = output_dir / "scorers"
    scorer_dir.mkdir(parents=True, exist_ok=True)
    physics_path = scorer_dir / f"seed_{seed}_physics.pt"
    visible_path = scorer_dir / f"seed_{seed}_visible_only.pt"
    for model, path in (
        (scorers.physics, physics_path),
        (scorers.visible_only, visible_path),
    ):
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        torch.save(model.state_dict(), temporary)
        temporary.replace(path)
    reloaded_physics = LearnedHypothesisScorer(
        scorers.physics_input_features, hidden=scorers.hidden_features
    )
    reloaded_visible = LearnedHypothesisScorer(
        scorers.physics_input_features, hidden=scorers.hidden_features
    )
    reloaded_physics.load_state_dict(torch.load(physics_path, map_location="cpu", weights_only=True))
    reloaded_visible.load_state_dict(torch.load(visible_path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        if not torch.allclose(
            scorers.physics(validation_physics), reloaded_physics(validation_physics),
            atol=1.0e-6, rtol=1.0e-6,
        ) or not torch.allclose(
            scorers.visible_only(validation_visible), reloaded_visible(validation_visible),
            atol=1.0e-6, rtol=1.0e-6,
        ):
            raise RuntimeError("Study-10 learned scorer reload is not deterministic")
    alpha, alpha_scan = select_alpha(
        train["visible_log_score"][train_support], train["analytic"][train_support],
        train["truth"][train_support], train["valid"][train_support],
        (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
    )
    with torch.no_grad():
        baseline = validation["visible_log_score"]
        alpha_zero_score = baseline
        if not torch.isfinite(alpha_zero_score).all():
            raise FloatingPointError("Study-10 alpha=0 control contains non-finite scores")
        physics_score = baseline + scorers.physics(validation_physics)
        visible_only_score = baseline + scorers.visible_only(validation_visible)
        analytic_score = (
            baseline
            if alpha == 0.0
            else baseline + alpha * validation["analytic"]
        )
        oracle_score = validation["oracle"]
    records = {
        "visible_baseline": _transitions_for(baseline, validation),
        "matched_visible_only_control": _transitions_for(visible_only_score, validation),
        "analytic_physics": {**_transitions_for(analytic_score, validation), "alpha_selected_on_train": alpha},
        "learned_physics": _transitions_for(physics_score, validation),
        "truth_neutrino_oracle": _transitions_for(oracle_score, validation),
        "top_k_ceiling": {
            str(k): float(top_k_coverage(validation["truth"], validation["valid"], k).float().mean())
            for k in REQUESTED_K
        },
    }
    records["coupling_efficiency"] = coupling_efficiency(
        records["visible_baseline"]["accuracy"],
        records["learned_physics"]["accuracy"],
        records["truth_neutrino_oracle"]["accuracy"],
    )
    ablation_results = {}
    for index, name in enumerate(FEATURE_ABLATIONS):
        ablation_alpha, scan = select_alpha(
            train["visible_log_score"][train_support], train["ablations"][train_support][..., index],
            train["truth"][train_support], train["valid"][train_support],
            (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
        )
        score = validation["visible_log_score"] + ablation_alpha * validation["ablations"][..., index]
        ablation_results[name] = {
            **_transitions_for(score, validation),
            "alpha_selected_on_train": ablation_alpha,
            "train_alpha_scan": scan,
        }
    records["feature_ablations"] = ablation_results
    convergence = {}
    nested = validation["nested"]
    for k_index, k in enumerate(REQUESTED_K):
        for m_index, m in enumerate(REQUESTED_M):
            score = validation["visible_log_score"] + alpha * nested[k_index, m_index]
            score = score[:, :k]
            truth = validation["truth"][:, :k]
            valid = validation["valid"][:, :k]
            convergence[f"K{k}_M{m}"] = {
                "assignment_accuracy": float(rank_correctness(score, truth, valid).float().mean()),
                "top_k_coverage": float(top_k_coverage(truth, valid, k).float().mean()),
            }
    records["nested_K_M_convergence"] = convergence
    return {
        "seed": int(seed),
        "scorer_training_epochs": int(epochs),
        "train_feature_artifact": str(train_path),
        "validation_feature_artifact": str(validation_path),
        "physics_input_features": scorers.physics_input_features,
        "hidden_features": scorers.hidden_features,
        "alpha_scan": alpha_scan,
        "scorer_artifacts": {
            "physics": {"path": str(physics_path), "sha256": _sha256(physics_path)},
            "visible_only": {"path": str(visible_path), "sha256": _sha256(visible_path)},
        },
        "comparisons": records,
        "feature_producer_source_sha": validation["feature_producer_source_sha"],
        "training_events_total": train_population["events_total"],
        "training_events_supported": train_population["events_supported"],
        "training_events_unsupported": train_population["events_unsupported"],
        "training_supported_fraction": train_population["supported_fraction"],
        "validation_events_total": validation_population["events_total"],
        "validation_events_supported": validation_population["events_supported"],
        "validation_events_unsupported": validation_population["events_unsupported"],
        "validation_supported_fraction": validation_population["supported_fraction"],
        "population_diagnostics": population_diagnostics or {
            "status": "stored_top_k_only",
            "description": "No valid correct candidate among the stored top ten.",
        },
    }


def _seed_record_path(output: Path, seed: int) -> Path:
    return output.parent / "study10_seed_records" / f"seed_{seed}.json"


def _run_one_seed(
    *,
    prediction: Path,
    training_prediction: Path | None,
    experiment: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    chunk_size: int,
    epochs: int,
    seed: int,
    study_manifest_sha256: str,
    gate_digest: str,
    feature_producer_source_sha: str,
    seed_record: Path,
) -> dict:
    """Run or reuse one seed without publishing the three-seed review."""
    expected_source = _source_sha()
    try:
        return _seed_record_contract(
            seed_record,
            expected_seed=seed,
            expected_source_sha=expected_source,
            expected_manifest_sha256=study_manifest_sha256,
            expected_gate_digest=gate_digest,
            expected_feature_producer_sha=feature_producer_source_sha,
            expected_checkpoint=checkpoint,
            expected_checkpoint_sha256=_sha256(checkpoint),
        )
    except (OSError, ValueError, KeyError, RuntimeError, EOFError):
        pass

    feature_dir = output.parent / "feature_artifacts"
    train_path = feature_dir / f"seed_{seed}_train_features.h5"
    validation_path = feature_dir / f"seed_{seed}_validation_features.h5"
    record_feature_producer = feature_producer_source_sha
    try:
        _feature_artifact_contract(
            train_path,
            expected_split="train",
            expected_source_sha=expected_source,
            expected_feature_producer_sha=feature_producer_source_sha,
            expected_manifest_sha256=study_manifest_sha256,
            expected_gate_digest=gate_digest,
            expected_checkpoint_sha256=_sha256(checkpoint),
        )
        _feature_artifact_contract(
            validation_path,
            expected_split="validation",
            expected_source_sha=expected_source,
            expected_feature_producer_sha=feature_producer_source_sha,
            expected_manifest_sha256=study_manifest_sha256,
            expected_gate_digest=gate_digest,
            expected_checkpoint_sha256=_sha256(checkpoint),
        )
    except (OSError, ValueError, KeyError, RuntimeError):
        placeholder = feature_dir / f"seed_{seed}_placeholder.h5"
        train_path, validation_path = _build_one_seed(
            prediction,
            training_prediction,
            experiment,
            checkpoint,
            placeholder,
            device,
            chunk_size,
            seed,
            study_manifest_sha256,
            gate_digest,
        )
        record_feature_producer = expected_source
    analysis = _analyse_seed(
        train_path,
        validation_path,
        output.parent,
        epochs=epochs,
        seed=seed,
        experiment=experiment,
    )
    if analysis["feature_producer_source_sha"] != record_feature_producer:
        raise ValueError(
            "Study-10 feature producer changed between contract validation and analysis: "
            f"{analysis['feature_producer_source_sha']} != {record_feature_producer}"
        )
    analysis.update({
        "seed": int(seed),
        "source_sha": expected_source,
        "analysis_source_sha": expected_source,
        "feature_producer_source_sha": record_feature_producer,
        "study_manifest_sha256": study_manifest_sha256,
        "resolved_gate_digest": gate_digest,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_training_sha": _checkpoint_training_sha(checkpoint),
        "seed_record_path": str(seed_record),
    })
    _atomic_json(seed_record, analysis)
    return _seed_record_contract(
        seed_record,
        expected_seed=seed,
        expected_source_sha=expected_source,
        expected_manifest_sha256=study_manifest_sha256,
        expected_gate_digest=gate_digest,
        expected_feature_producer_sha=record_feature_producer,
        expected_checkpoint=checkpoint,
        expected_checkpoint_sha256=_sha256(checkpoint),
    )


def _write_review(output: Path, records: list[dict], study_manifest_sha256: str, gate_digest: str) -> None:
    if {int(record["seed"]) for record in records} != {42, 43, 44}:
        raise ValueError("Study-10 finalisation requires exactly seed records 42, 43 and 44")
    source_values = {record.get("source_sha") for record in records}
    if len(source_values) != 1:
        raise ValueError("Study-10 seed records were produced by different source revisions")
    analysis_values = {record.get("analysis_source_sha", record.get("source_sha")) for record in records}
    if len(analysis_values) != 1 or analysis_values != source_values:
        raise ValueError("Study-10 seed records have inconsistent analysis-source provenance")
    producer_values = {record.get("feature_producer_source_sha") for record in records}
    if len(producer_values) != 1 or None in producer_values:
        raise ValueError("Study-10 seed records have inconsistent feature-producer provenance")
    review = {
        "study": "compatibility_reranking",
        "selection_split": "validation",
        "training_split_for_scorers": True,
        "test_split_used": False,
        "study_manifest_sha256": study_manifest_sha256,
        "resolved_gate_digest": gate_digest,
        "source_sha": records[0]["source_sha"],
        "analysis_source_sha": records[0].get("analysis_source_sha", records[0]["source_sha"]),
        "feature_producer_source_sha": records[0]["feature_producer_source_sha"],
        "Kmax": 10,
        "Mmax": 64,
        "K_prefixes": list(REQUESTED_K),
        "M_prefixes": list(REQUESTED_M),
        "comparisons": [
            "visible baseline",
            "capacity-matched visible-only learned control",
            "analytic physics compatibility",
            "learned physics correction",
            "truth-neutrino oracle",
            "top-K ceiling",
        ],
        "records": sorted(records, key=lambda record: int(record["seed"])),
        "population": {
            "training": [
                {
                    "seed": record["seed"],
                    "total": record["training_events_total"],
                    "supported": record["training_events_supported"],
                    "unsupported": record["training_events_unsupported"],
                }
                for record in sorted(records, key=lambda item: int(item["seed"]))
            ],
            "validation": [
                {
                    "seed": record["seed"],
                    "total": record["validation_events_total"],
                    "supported": record["validation_events_supported"],
                    "unsupported": record["validation_events_unsupported"],
                }
                for record in sorted(records, key=lambda item: int(item["seed"]))
            ],
        },
        "decision": "review_required",
        "finalisation": {
            "seed_record_paths": [
                str(record.get("seed_record_path", _seed_record_path(output, int(record["seed"]))))
                for record in sorted(records, key=lambda item: int(item["seed"]))
            ],
            "expensive_analysis_reused": True,
            "test_split_used": False,
        },
    }
    _atomic_json(output.with_suffix(".json"), review)
    lines = [
        "# Study 10 — controlled invisible physics/assignment coupling",
        "",
        "Scorers are fitted on the training split and evaluated once on validation.",
        "The truth-neutrino row is a non-deployable oracle. Test data are sealed.",
        "",
        "| seed | train supported | train unsupported | validation supported | validation unsupported | visible | visible-only control | analytic | learned | truth-ν oracle | Rcoupling |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in review["records"]:
        comparisons = record["comparisons"]
        coupling = comparisons["coupling_efficiency"].get("Rcoupling")
        lines.append(
            f"| {record['seed']} | {record['training_events_supported']}/{record['training_events_total']} | "
            f"{record['training_events_unsupported']} | "
            f"{record['validation_events_supported']}/{record['validation_events_total']} | "
            f"{record['validation_events_unsupported']} | "
            f"{comparisons['visible_baseline']['accuracy']:.6f} | "
            f"{comparisons['matched_visible_only_control']['accuracy']:.6f} | "
            f"{comparisons['analytic_physics']['accuracy']:.6f} | "
            f"{comparisons['learned_physics']['accuracy']:.6f} | "
            f"{comparisons['truth_neutrino_oracle']['accuracy']:.6f} | "
            f"{coupling if coupling is not None else '—'} |"
        )
    markdown = output.with_suffix(".md")
    temporary = markdown.with_name(f".{markdown.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(markdown)


def _finalize_records(output: Path, seed_records: list[Path], study_manifest_sha256: str, gate_digest: str) -> None:
    expected_source = _source_sha()
    if not seed_records:
        seed_records = [_seed_record_path(output, seed) for seed in (42, 43, 44)]
    records = [
        _seed_record_contract(
            path,
            expected_seed=seed,
            expected_source_sha=expected_source,
            expected_manifest_sha256=study_manifest_sha256,
            expected_gate_digest=gate_digest,
        )
        for seed, path in zip((42, 43, 44), seed_records, strict=True)
    ]
    _write_review(output, records, study_manifest_sha256, gate_digest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", default=[])
    parser.add_argument("--training-prediction", action="append", default=[])
    parser.add_argument("--experiment", action="append", default=[])
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--study-manifest-sha256", required=True)
    parser.add_argument("--gate-digest", required=True)
    parser.add_argument(
        "--feature-producer-source-sha",
        help="Source SHA that produced reusable feature artifacts; defaults to this analysis source.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seed-record", action="append", type=Path, default=[])
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if args.finalize:
        if args.prediction or args.training_prediction or args.experiment or args.checkpoint or args.seed is not None:
            raise ValueError("--finalize cannot be combined with prediction or seed execution arguments")
        _finalize_records(
            output,
            [path.expanduser().resolve() for path in args.seed_record],
            args.study_manifest_sha256,
            args.gate_digest,
        )
        print(output.with_suffix(".json"))
        print(output.with_suffix(".md"))
        return
    if not args.prediction or not args.experiment or not args.checkpoint:
        raise ValueError("Seed analysis requires --prediction, --experiment and --checkpoint")
    if len(args.experiment) != len(args.prediction) or len(args.checkpoint) != len(args.prediction):
        raise ValueError("Each Study-10 validation prediction needs one experiment and checkpoint")
    if args.training_prediction and len(args.training_prediction) != len(args.prediction):
        raise ValueError("--training-prediction must be supplied once per validation prediction")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Study 10 feature production requested CUDA but no GPU is available")
    if args.seed is not None and len(args.prediction) != 1:
        raise ValueError("A single prediction is required for --seed execution")
    if args.seed_record and len(args.seed_record) != (1 if args.seed is not None else len(args.prediction)):
        raise ValueError("--seed-record count does not match the selected seed analyses")
    results = []
    feature_producer_source_sha = args.feature_producer_source_sha or _source_sha()
    if not re.fullmatch(r"[0-9a-f]{40}", feature_producer_source_sha):
        raise ValueError(
            f"Invalid Study-10 feature producer source SHA: {feature_producer_source_sha}"
        )
    for index, validation in enumerate(args.prediction):
        train_prediction = args.training_prediction[index] if args.training_prediction else None
        seed = args.seed if args.seed is not None else index + 42
        record_path = (
            args.seed_record[index].expanduser().resolve()
            if args.seed_record
            else _seed_record_path(output, seed)
        )
        results.append(_run_one_seed(
            prediction=Path(validation),
            training_prediction=None if train_prediction is None else Path(train_prediction),
            experiment=Path(args.experiment[index]),
            checkpoint=Path(args.checkpoint[index]),
            output=output,
            device=args.device,
            chunk_size=args.chunk_size,
            epochs=args.epochs,
            seed=seed,
            study_manifest_sha256=args.study_manifest_sha256,
            gate_digest=args.gate_digest,
            feature_producer_source_sha=feature_producer_source_sha,
            seed_record=record_path,
        ))
    if args.seed is None:
        _write_review(output, results, args.study_manifest_sha256, args.gate_digest)
    else:
        print(record_path)


if __name__ == "__main__":
    main()
