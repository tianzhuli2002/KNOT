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
    with h5py.File(path, "w") as destination:
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
        feature_names = json.loads(source["PHYSICS"].attrs["feature_names"] if "feature_names" in source["PHYSICS"].attrs else source.attrs["feature_names"])
        return {
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
            "source_sha": str(source.attrs["source_sha"]),
            "physics_scaler_sha256": str(source.attrs["physics_scaler_sha256"]),
            "physics_scaler_fit_split": str(source.attrs["physics_scaler_fit_split"]),
        }


def _input_tensors(features: dict) -> tuple[torch.Tensor, torch.Tensor]:
    analytic = features["analytic"].float()
    valid = features["valid"]
    if not torch.isfinite(analytic[valid]).all():
        raise FloatingPointError("Study-10 analytic score is non-finite for a valid hypothesis")
    # Invalid hypotheses are masked before ranking/loss evaluation. Their
    # stored -inf sentinel must nevertheless be finite while it passes through
    # the learned scorer, otherwise ordinary linear layers create NaN gradients.
    analytic_for_scorer = torch.where(
        torch.isfinite(analytic), analytic, torch.zeros_like(analytic)
    )
    reduction = torch.cat(
        (features["physics_mean"], features["physics_std"], features["physics_rms"], analytic_for_scorer.unsqueeze(-1)),
        dim=-1,
    )
    physics = torch.cat(
        (reduction, features["visible_log_score"].unsqueeze(-1), features["visible_probability"].unsqueeze(-1)),
        dim=-1,
    ).float()
    visible = visible_only_scorer_input(
        features["visible_log_score"].float(), features["visible_probability"].float(),
        features["entropy"].float(), features["margin"].float(), target_features=physics.shape[-1],
    )
    return physics, visible


def _transitions_for(score: torch.Tensor, features: dict) -> dict:
    baseline = rank_correctness(features["visible_log_score"], features["truth"], features["valid"])
    selected = rank_correctness(score, features["truth"], features["valid"])
    return {
        "accuracy": float(selected.float().mean()),
        "delta_vs_visible": float(selected.float().mean() - baseline.float().mean()),
        "transitions": transition_counts(baseline, selected),
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


def _analyse_seed(train_path: Path, validation_path: Path, output_dir: Path, epochs: int, seed: int) -> dict:
    train = _load_features(train_path)
    validation = _load_features(validation_path)
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
    torch.save(scorers.physics.state_dict(), physics_path)
    torch.save(scorers.visible_only.state_dict(), visible_path)
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
        train["visible_log_score"], train["analytic"], train["truth"], train["valid"],
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
            train["visible_log_score"], train["ablations"][..., index], train["truth"], train["valid"],
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
        "train_feature_artifact": str(train_path),
        "validation_feature_artifact": str(validation_path),
        "physics_input_features": scorers.physics_input_features,
        "hidden_features": scorers.hidden_features,
        "alpha_scan": alpha_scan,
        "comparisons": records,
        "source_sha": validation["source_sha"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True)
    parser.add_argument("--training-prediction", action="append", default=[])
    parser.add_argument("--experiment", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--study-manifest-sha256", required=True)
    parser.add_argument("--gate-digest", required=True)
    args = parser.parse_args()
    if len(args.experiment) != len(args.prediction) or len(args.checkpoint) != len(args.prediction):
        raise ValueError("Each Study-10 validation prediction needs one experiment and checkpoint")
    if args.training_prediction and len(args.training_prediction) != len(args.prediction):
        raise ValueError("--training-prediction must be supplied once per validation prediction")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Study 10 feature production requested CUDA but no GPU is available")
    output = args.output.expanduser().resolve()
    feature_dir = output.parent / "feature_artifacts"
    results = []
    for index, validation in enumerate(args.prediction):
        train_prediction = args.training_prediction[index] if args.training_prediction else None
        train_path, validation_path = _build_one_seed(
            Path(validation),
            None if train_prediction is None else Path(train_prediction),
            Path(args.experiment[index]), Path(args.checkpoint[index]),
            feature_dir / f"seed_{index + 42}_placeholder.h5", args.device, args.chunk_size, index + 42,
            args.study_manifest_sha256,
            args.gate_digest,
        )
        results.append(_analyse_seed(train_path, validation_path, output.parent, args.epochs, index + 42))
    review = {
        "study": "compatibility_reranking",
        "selection_split": "validation",
        "training_split_for_scorers": True,
        "test_split_used": False,
        "study_manifest_sha256": args.study_manifest_sha256,
        "resolved_gate_digest": args.gate_digest,
        "source_sha": _source_sha(),
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
        "records": results,
        "decision": "review_required",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Study 10 — controlled invisible physics/assignment coupling",
        "",
        "Scorers are fitted on the training split and evaluated once on validation.",
        "The truth-neutrino row is a non-deployable oracle. Test data are sealed.",
        "",
        "| seed | visible | visible-only control | analytic | learned | truth-ν oracle | Rcoupling |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in results:
        comparisons = record["comparisons"]
        lines.append(
            f"| {record['seed']} | {comparisons['visible_baseline']['accuracy']:.6f} | "
            f"{comparisons['matched_visible_only_control']['accuracy']:.6f} | "
            f"{comparisons['analytic_physics']['accuracy']:.6f} | "
            f"{comparisons['learned_physics']['accuracy']:.6f} | "
            f"{comparisons['truth_neutrino_oracle']['accuracy']:.6f} | "
            f"{comparisons['coupling_efficiency']['Rcoupling'] if comparisons['coupling_efficiency']['Rcoupling'] is not None else '—'} |"
        )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(output.with_suffix(".json"))
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
