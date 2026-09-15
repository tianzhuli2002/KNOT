#!/usr/bin/env python3
"""Run the validation-only top-K and truth-neutrino coupling oracle.

The input prediction files are visible-assignment products.  No invisible model
is rerun: the true neutrino four-vectors are used only as an oracle for the
physics reranking diagnostic.  Training residual scales are fitted from the
training split with the correct assignment and truth invisible state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.cli.run import build_data
from joint_event_reconstruction.configuration import load_experiment, load_visible_config
from joint_event_reconstruction.topology import (
    PhysicsFeatureEngine,
    PhysicsFeatureScaler,
    PosteriorReducer,
    TopologyKinematics,
    load_topology,
)


REQUESTED_TOP_K = (1, 2, 3, 5, 10)
ABLATIONS = {
    "W_masses": lambda name: name in {"W_e_mass", "W_mu_mass"},
    "top_masses": lambda name: name in {"top_e_mass", "top_mu_mass"},
    "top_mass_balance": lambda name: name == "top_mass_balance",
    "MET_closure": lambda name: name.startswith("met_closure."),
    "W_plus_top": lambda name: name in {
        "W_e_mass", "W_mu_mass", "top_e_mass", "top_mu_mass"
    },
    "all": lambda name: True,
}


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    )
    return result.stdout.strip()


def _configure_cpu_data(config):
    visible = load_visible_config(config)
    for name in ("train_num_workers", "val_num_workers", "test_num_workers"):
        setattr(visible, name, 0)
    visible.pin_memory = False
    visible.persistent_workers = False
    config.model.invisible.enabled = True
    return visible


def _static_pools(batches: list[dict], spec) -> dict:
    result = {}
    for node_name in spec.dag_order:
        first = batches[0]["structured"]["pools"][node_name]
        result[node_name] = {
            "candidate_mask": torch.cat(
                [batch["structured"]["pools"][node_name]["candidate_mask"]
                 for batch in batches], dim=0
            ),
            "leaf_indices": first["leaf_indices"],
            "child_indices": first["child_indices"],
        }
    return result


def _collect_validation(data, spec) -> dict:
    batches = list(data.val_dataloader())
    if not batches:
        raise ValueError("Validation loader produced no events")
    return {
        "event_id": torch.cat([batch["event_index"] for batch in batches]).numpy(),
        "physics_vectors": torch.cat(
            [batch["physics_vectors"] for batch in batches], dim=0
        ),
        "met_xy": torch.cat([batch["met_xy"] for batch in batches], dim=0),
        "invisible_truth": torch.cat(
            [batch["invisible_truth"] for batch in batches], dim=0
        ),
        "pools": _static_pools(batches, spec),
    }


def _truth_node_indices(batch: dict, spec) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.stack(
        [batch["structured"]["pools"][name]["truth_index"] for name in spec.dag_order],
        dim=-1,
    ).unsqueeze(1)
    valid = indices.ge(0).all(dim=-1)
    return indices, valid


def _fit_scaler(data, spec, kinematics, features, source_identity: str) -> PhysicsFeatureScaler:
    residuals = []
    feature_names = None
    for batch in data.train_dataloader():
        indices, valid = _truth_node_indices(batch, spec)
        objects = kinematics.compose(
            batch["structured"]["pools"],
            batch["physics_vectors"],
            indices,
            invisible_momenta=batch["invisible_truth"].unsqueeze(1),
        )
        result = features.evaluate(objects, observed_met=batch["met_xy"])
        if feature_names is None:
            feature_names = result.feature_names
        elif result.feature_names != feature_names:
            raise ValueError("Physics feature order changed between training batches")
        values = result.tensor[:, 0, 0]
        selected = valid[:, 0] & torch.isfinite(values).all(dim=-1)
        if selected.any():
            residuals.append(values[selected].cpu())
    if not residuals:
        raise RuntimeError("No reconstructable train truth rows were available for scaling")
    return PhysicsFeatureScaler.fit(
        torch.cat(residuals, dim=0),
        feature_names,
        topology_signature=spec.signature,
        physics_signature=features.spec.signature,
        source_data_identity=source_identity,
        split="train",
    )


def _read_prediction(path: Path, spec, expected_event_ids: np.ndarray) -> dict:
    with h5py.File(path, "r") as source:
        split = source.attrs.get("split", "")
        if isinstance(split, bytes):
            split = split.decode()
        if split != "validation":
            raise ValueError(f"Oracle requires validation predictions: {path} has split={split!r}")
        event_id = source["EVENT_ID"][:]
        if not np.array_equal(event_id, expected_event_ids):
            raise ValueError(f"Prediction EVENT_ID order does not match the validation loader: {path}")
        raw_root_order = source.attrs["root_order"]
        if isinstance(raw_root_order, bytes):
            raw_root_order = raw_root_order.decode()
        root_order = json.loads(str(raw_root_order))
        root_indices = source["VISIBLE/root_candidate_indices"][:]
        count, top_k, root_count = root_indices.shape
        if tuple(root_order) != tuple(spec.roots) or root_count != len(spec.roots):
            raise ValueError(
                f"Prediction root order is incompatible with topology: {path}; "
                f"prediction={root_order}, topology={spec.roots}"
            )
        if tuple(spec.dag_order) != tuple(spec.roots):
            raise ValueError("The oracle currently requires all assignment nodes to be roots")
        truth = source["TRUTH/topk_assignment"][:].astype(bool, copy=False)
        valid = source["VISIBLE/valid_hypotheses"][:].astype(bool, copy=False)
        scores = source["VISIBLE/assignment_log_scores"][:].astype(np.float32, copy=False)
        if truth.shape != (count, top_k) or valid.shape != truth.shape or scores.shape != truth.shape:
            raise ValueError(f"Prediction has inconsistent top-K shapes: {path}")
        if not valid.any(axis=1).all():
            raise ValueError(f"Prediction contains an event without a valid hypothesis: {path}")
        entropies = []
        margins = []
        for node in spec.roots:
            entropies.append(source[f"VISIBLE/nodes/{node}/entropy"][:])
            margins.append(source[f"VISIBLE/nodes/{node}/margin"][:])
        return {
            "path": str(path.resolve()),
            "event_id": event_id,
            "root_indices": root_indices,
            "truth": truth,
            "valid": valid,
            "scores": scores,
            "probability": source["VISIBLE/constrained_probabilities"][:].astype(np.float32, copy=False),
            "njet": source["OBSERVED/njet"][:].astype(np.int16, copy=False),
            "entropy": np.stack(entropies, axis=1).mean(axis=1),
            "margin": np.stack(margins, axis=1).mean(axis=1),
        }


def _labels(values: np.ndarray, prefix: str) -> np.ndarray:
    edges = np.unique(np.quantile(values, [0.0, 1 / 3, 2 / 3, 1.0]))
    if len(edges) < 2:
        return np.full(values.shape, f"{prefix}:constant", dtype=object)
    bins = np.searchsorted(edges[1:-1], values, side="right")
    return np.asarray([f"{prefix}:q{int(value) + 1}" for value in bins], dtype=object)


def _stratified(
    labels: np.ndarray,
    baseline: np.ndarray,
    ceiling: np.ndarray,
    correct: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    result = {}
    for label in sorted(set(labels.tolist())):
        selected = labels == label
        count = int(selected.sum())
        result[str(label)] = {
            "events": count,
            "visible_accuracy": float(baseline[selected].mean()),
            "top_k_ceiling": float(ceiling[selected].mean()),
            "reranked_accuracy": float(correct[selected].mean()),
            "delta_vs_visible": float(correct[selected].mean() - baseline[selected].mean()),
        }
    return result


def _transition_metrics(baseline: np.ndarray, correct: np.ndarray) -> dict[str, int | float]:
    rescued = (~baseline) & correct
    damaged = baseline & (~correct)
    return {
        "baseline_correct_to_remains_correct": int((baseline & correct).sum()),
        "baseline_correct_to_broken": int(damaged.sum()),
        "baseline_wrong_to_rescued": int(rescued.sum()),
        "baseline_wrong_to_remains_wrong": int((~baseline & ~correct).sum()),
        "rescue_fraction_of_baseline_wrong": (
            float(rescued.sum() / (~baseline).sum()) if (~baseline).any() else 0.0
        ),
        "damage_fraction_of_baseline_correct": (
            float(damaged.sum() / baseline.sum()) if baseline.any() else 0.0
        ),
    }


def _write_artifact(
    path: Path,
    prediction: dict,
    feature_names: tuple[str, ...],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_rms: np.ndarray,
    truth_features: np.ndarray,
    valid_hypotheses: np.ndarray,
    scaler: PhysicsFeatureScaler,
    spec,
    source_sha: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as destination:
        destination.attrs.update({
            "artifact_type": "validation_hypothesis_coupling",
            "selection_split": "validation",
            "test_split_used": False,
            "source_sha": source_sha,
            "topology_signature": spec.signature,
            "physics_signature": spec.physics_signature or "none",
            "physics_feature_space": "train_scaled_residual",
            "physics_scaler_sha256": scaler.identity_sha256,
            "prediction_identity": prediction["path"],
            "normalised_likelihood": False,
        })
        destination.create_dataset("EVENT_ID", data=prediction["event_id"])
        destination.create_dataset("state", data=prediction["root_indices"], compression="gzip")
        destination.create_dataset("visible_log_score", data=prediction["scores"], compression="gzip")
        destination.create_dataset("visible_probability", data=prediction["probability"], compression="gzip")
        destination.create_dataset("truth", data=prediction["truth"], compression="gzip")
        destination.create_dataset("valid_hypotheses", data=valid_hypotheses, compression="gzip")
        destination.create_dataset("njet", data=prediction["njet"])
        destination.create_dataset("assignment_entropy", data=prediction["entropy"].astype(np.float32))
        destination.create_dataset("top1_top2_margin", data=prediction["margin"].astype(np.float32))
        physics = destination.create_group("PHYSICS")
        physics.attrs["feature_names"] = json.dumps(list(feature_names))
        physics.create_dataset("feature_mean", data=feature_mean, compression="gzip")
        physics.create_dataset("feature_std", data=feature_std, compression="gzip")
        physics.create_dataset("feature_rms", data=feature_rms, compression="gzip")
        physics.create_dataset("truth_neutrino_features", data=truth_features, compression="gzip")


def _run_one(
    prediction_path: Path,
    validation: dict,
    scaler: PhysicsFeatureScaler,
    spec,
    kinematics: TopologyKinematics,
    features: PhysicsFeatureEngine,
    *,
    output: Path,
    artifact_dir: Path,
    alpha: float,
    chunk_size: int,
    source_sha: str,
) -> dict:
    prediction = _read_prediction(prediction_path, spec, validation["event_id"])
    count = len(prediction["event_id"])
    if count != len(validation["event_id"]):
        raise ValueError(f"Prediction and validation data lengths disagree: {prediction_path}")
    reducer = PosteriorReducer()
    methods = list(ABLATIONS)
    correct = {method: [] for method in methods}
    feature_mean = []
    feature_std = []
    feature_rms = []
    truth_features = []
    valid_hypotheses = []

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        pools = {
            node: {
                "candidate_mask": values["candidate_mask"][start:stop],
                "leaf_indices": values["leaf_indices"],
                "child_indices": values["child_indices"],
            }
            for node, values in validation["pools"].items()
        }
        node_indices = torch.from_numpy(prediction["root_indices"][start:stop].astype(np.int64))
        valid = torch.from_numpy(prediction["valid"][start:stop])
        objects = kinematics.compose(
            pools,
            validation["physics_vectors"][start:stop],
            node_indices,
            invisible_momenta=validation["invisible_truth"][start:stop].unsqueeze(1),
        )
        result = features.evaluate(
            objects, observed_met=validation["met_xy"][start:stop]
        )
        scaled = scaler.transform(
            result.tensor,
            feature_names=result.feature_names,
            topology_signature=spec.signature,
            physics_signature=result.physics_signature,
        )
        stored_features = torch.nan_to_num(
            scaled[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0
        ).numpy()
        feature_mean.append(stored_features)
        feature_std.append(np.zeros_like(stored_features))
        feature_rms.append(np.abs(stored_features))
        truth_features.append(stored_features)
        valid_hypotheses.append(prediction["valid"][start:stop])
        visible = torch.from_numpy(prediction["scores"][start:stop])
        for method in methods:
            selected = [index for index, name in enumerate(result.feature_names) if ABLATIONS[method](name)]
            if not selected:
                raise ValueError(f"Physics ablation {method} selected no features")
            reduction = reducer.reduce(
                scaled[..., selected],
                valid_samples=valid.unsqueeze(-1),
                valid_hypotheses=valid,
            )
            final = (visible + alpha * reduction.analytic_log_evidence).masked_fill(~valid, -torch.inf)
            ranking = final.argmax(dim=1).numpy()
            truth = prediction["truth"][start:stop]
            correct[method].append(truth[np.arange(stop - start), ranking])

    truth = prediction["truth"]
    baseline = truth[:, 0]
    ceiling = truth.any(axis=1)
    summary_methods = {}
    njet_labels = np.asarray([
        "njet:0-3" if value <= 3 else "njet:4-5" if value <= 5 else
        "njet:6-7" if value <= 7 else "njet:8+"
        for value in prediction["njet"]
    ], dtype=object)
    strata = {
        "njet": njet_labels,
        "entropy": _labels(prediction["entropy"], "entropy"),
        "margin": _labels(prediction["margin"], "margin"),
    }
    for method in methods:
        method_correct = np.concatenate(correct[method]).astype(bool, copy=False)
        summary_methods[method] = {
            "accuracy": float(method_correct.mean()),
            "delta_vs_visible": float(method_correct.mean() - baseline.mean()),
            "transitions": _transition_metrics(baseline, method_correct),
            "by": {
                name: _stratified(labels, baseline, ceiling, method_correct)
                for name, labels in strata.items()
            },
        }

    mean = np.concatenate(feature_mean, axis=0).astype(np.float32, copy=False)
    std = np.concatenate(feature_std, axis=0).astype(np.float32, copy=False)
    rms = np.concatenate(feature_rms, axis=0).astype(np.float32, copy=False)
    truth_feature_array = np.concatenate(truth_features, axis=0).astype(np.float32, copy=False)
    valid_hypothesis_array = np.concatenate(valid_hypotheses, axis=0).astype(bool, copy=False)
    seed_name = prediction_path.parent.parent.name
    artifact = artifact_dir / f"{seed_name}.h5"
    _write_artifact(
        artifact, prediction, scaler.feature_names, mean, std, rms, truth_feature_array,
        valid_hypothesis_array, scaler, spec, source_sha,
    )
    record = {
        "prediction": prediction["path"],
        "events": count,
        "available_top_k": int(prediction["truth"].shape[1]),
        "coverage": {
            str(k): (
                float(truth[:, :k].any(axis=1).mean())
                if k <= truth.shape[1] else None
            )
            for k in REQUESTED_TOP_K
        },
        "visible_baseline_accuracy": float(baseline.mean()),
        "top_k_ceiling_accuracy": float(ceiling.mean()),
        "methods": summary_methods,
        "offline_artifact": str(artifact),
        "scaler": scaler.as_dict(),
        "physics_feature_names": list(scaler.feature_names),
    }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, action="append", required=True)
    parser.add_argument("--experiment", type=Path, default=Path("configs/experiments/ttbar2L.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/studies/compatibility_reranking/coupling_oracle_real.json"))
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    config = load_experiment(str(args.experiment))
    visible = _configure_cpu_data(config)
    data = build_data(config, visible)
    data.setup("predict")
    spec = load_topology(
        config.topology_config,
        OmegaConf.to_container(config.data.truth_bindings, resolve=True),
    )
    if spec.physics is None:
        raise ValueError("The selected topology has no physics compatibility block")
    kinematics = TopologyKinematics(spec)
    features = PhysicsFeatureEngine(spec.physics)
    scaler = _fit_scaler(
        data, spec, kinematics, features, str(Path(data.h5_path).resolve())
    )
    validation = _collect_validation(data, spec)
    source_sha = _git_sha()
    output = args.output.expanduser().resolve()
    artifact_dir = (
        args.artifact_dir.expanduser().resolve()
        if args.artifact_dir is not None
        else output.parent / "hypothesis_artifacts"
    )
    records = [
        _run_one(
            path.expanduser().resolve(), validation, scaler, spec, kinematics, features,
            output=output, artifact_dir=artifact_dir, alpha=args.alpha,
            chunk_size=args.chunk_size, source_sha=source_sha,
        )
        for path in args.prediction
    ]
    review = {
        "study": "compatibility_reranking",
        "topology": "ttbar2L",
        "selection_split": "validation",
        "test_split_used": False,
        "source_sha": source_sha,
        "alpha": args.alpha,
        "requested_top_k": list(REQUESTED_TOP_K),
        "records": records,
        "decision": {
            "status": "oracle_available",
            "interpretation": (
                "Truth-neutrino reranking is an oracle, not a deployable model. "
                "K=10 is unavailable because the supplied locked assignment artifact is top-5."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    markdown = output.with_suffix(".md")
    lines = [
        "# Truth-neutrino coupling oracle",
        "",
        "Selection split: `validation`. Test split used: `False`.",
        "",
        "| seed artifact | K=1 | K=2 | K=3 | K=5 | K=10 | visible | top-K ceiling | all physics |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        values = record["coverage"]
        method = record["methods"]["all"]["accuracy"]
        lines.append(
            f"| {Path(record['offline_artifact']).name} | "
            + " | ".join("—" if values[str(k)] is None else f"{values[str(k)]:.6f}" for k in REQUESTED_TOP_K)
            + f" | {record['visible_baseline_accuracy']:.6f} | {record['top_k_ceiling_accuracy']:.6f} | {method:.6f} |"
        )
    lines += [
        "",
        "The physics score is an unnormalised compatibility/evidence-like score, not a normalised likelihood.",
        "W/top/MET ablations, transition counts, and ambiguity-stratified results are in the JSON artifact.",
        "K=10 is unavailable because the maintained L-GATr assignment products contain K=5 hypotheses.",
    ]
    markdown.write_text("\n".join(lines) + "\n")
    print(output)
    print(markdown)


if __name__ == "__main__":
    main()
