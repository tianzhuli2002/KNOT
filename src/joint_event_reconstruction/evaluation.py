"""Compact maintained metrics for visible and sample-based invisible predictions."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.prediction import validate_prediction


def _energy_score(samples: np.ndarray, truth: np.ndarray) -> float:
    samples = samples.reshape(len(samples), samples.shape[1], -1)
    truth = truth.reshape(len(truth), -1)
    data_term = np.linalg.norm(samples - truth[:, None], axis=-1).mean(axis=1)
    pair = samples[:, :, None] - samples[:, None, :]
    sample_term = np.linalg.norm(pair, axis=-1).mean(axis=(1, 2))
    return float(np.mean(data_term - 0.5 * sample_term))


def _kinematics(vectors: np.ndarray) -> dict[str, np.ndarray]:
    momentum = vectors[..., 1:]
    pt = np.linalg.norm(momentum[..., :2], axis=-1)
    mass = np.sqrt(
        np.maximum(vectors[..., 0] ** 2 - np.sum(momentum ** 2, axis=-1), 0)
    )
    eta = np.arcsinh(
        np.divide(momentum[..., 2], np.maximum(pt, 1.0e-12))
    )
    phi = np.arctan2(momentum[..., 1], momentum[..., 0])
    return {"mass": mass, "pt": pt, "eta": eta, "phi": phi}


def _wrapped_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(first - second), np.cos(first - second))


def bootstrap_mean_interval(
    values: np.ndarray, *, replicas: int = 1000, seed: int = 42
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Bootstrap input must be a non-empty event-level vector")
    rng = np.random.default_rng(seed)
    unique = np.unique(values)
    if np.isin(unique, (0.0, 1.0)).all():
        means = rng.binomial(len(values), values.mean(), size=replicas) / len(values)
    else:
        means = np.asarray([
            values[rng.integers(0, len(values), size=len(values))].mean()
            for _ in range(replicas)
        ])
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def evaluate_predictions(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        version = source.attrs.get("architecture_signature_version")
        result = {
            "prediction_file": str(path),
            "events": len(source["EVENT_ID"]),
            "topology_signature": str(source.attrs["topology_signature"]),
            "architecture_signature_version": (
                None if version is None else int(version)
            ),
            "architecture_signature": str(source.attrs["architecture_signature"]),
            "model_mode": "factorized",
            "assignment_formulation": "discriminative",
            "invisible_formulation": str(
                source.attrs.get("invisible_formulation", "disabled")
            ),
            "top_k_assignments": int(source.attrs["K"]),
            "posterior_samples": int(source.attrs["M"]),
            "recurrent_steps": int(source.attrs["recurrent_steps"]),
            "visible": {},
        }
        if "INVISIBLE" in source and "actual_nfe" in source["INVISIBLE"].attrs:
            result["nfe_per_invisible_sample"] = int(
                source["INVISIBLE"].attrs["actual_nfe"]
            )
            result["network_calls_per_event"] = (
                int(source["INVISIBLE"].attrs["actual_nfe"])
                * int(source.attrs["K"])
                * int(source.attrs["M"])
            )

        nodes = source["VISIBLE/nodes"]
        for name in nodes:
            node = nodes[name]
            reconstructable = node["reconstructable"][:].astype(bool)
            selected = node["selected_candidate"][:]
            truth = node["truth_candidate"][:]
            correct = selected == truth
            result["visible"][name] = {
                "events": int(len(correct)),
                "reconstructable": int(reconstructable.sum()),
                "raw_efficiency": float(correct.mean()),
                "conditional_efficiency": (
                    float(correct[reconstructable].mean())
                    if reconstructable.any() else None
                ),
                "mean_truth_probability": (
                    float(node["truth_probability"][:][reconstructable].mean())
                    if reconstructable.any() else None
                ),
                "mean_entropy": float(node["entropy"][:].mean()),
                "mean_margin": float(node["margin"][:].mean()),
                "null_rate": float((selected < 0).mean()),
                "conditional_efficiency_bootstrap_95": (
                    list(bootstrap_mean_interval(correct[reconstructable]))
                    if reconstructable.any() else None
                ),
            }
        truth_topk = source["TRUTH/topk_assignment"][:]
        result["visible"]["complete"] = {
            "top1_efficiency": float(truth_topk[:, 0].mean()),
            "topk_coverage": float(truth_topk.any(axis=1).mean()),
            "truth_rank_mean_reciprocal": float(np.mean([
                1.0 / (np.flatnonzero(row)[0] + 1) if row.any() else 0.0
                for row in truth_topk
            ])),
            "mean_exact_logZ": float(source["VISIBLE/exact_logZ"][:].mean()),
            "probabilities_are_exact_constrained": True,
        }
        njet = source["OBSERVED/njet"][:]
        result["visible"]["complete"]["top1_vs_njet"] = {
            str(int(value)): {
                "events": int((njet == value).sum()),
                "efficiency": float(truth_topk[njet == value, 0].mean()),
            }
            for value in np.unique(njet)
        }
        if "INVISIBLE/samples" in source:
            samples = source["INVISIBLE/samples"][:, 0]
            truth = source["TRUTH/NEUTRINO"][:]
            residual = samples - truth[:, None]
            mean = samples.mean(axis=1)
            roles = json.loads(source.attrs["invisible_roles"])
            invisible = {"energy_score_joint": _energy_score(samples, truth), "roles": {}}
            levels = (0.50, 0.68, 0.90, 0.95)
            for role_index, role_name in enumerate(roles):
                role_result = {}
                for component, component_index in zip(("px", "py", "pz"), range(3)):
                    values = residual[:, :, role_index, component_index]
                    mean_residual = (
                        mean[:, role_index, component_index]
                        - truth[:, role_index, component_index]
                    )
                    component_result = {
                        "bias": float(mean_residual.mean()),
                        "rmse": float(np.sqrt(np.mean(mean_residual ** 2))),
                        "median_residual": float(np.median(mean_residual)),
                        "central_68_width": float(
                            np.quantile(mean_residual, 0.84)
                            - np.quantile(mean_residual, 0.16)
                        ),
                    }
                    for level in levels:
                        low = np.quantile(
                            samples[:, :, role_index, component_index],
                            (1 - level) / 2,
                            axis=1,
                        )
                        high = np.quantile(
                            samples[:, :, role_index, component_index],
                            1 - (1 - level) / 2,
                            axis=1,
                        )
                        target = truth[:, role_index, component_index]
                        component_result[f"coverage_{int(level * 100)}"] = float(
                            ((target >= low) & (target <= high)).mean()
                        )
                    role_result[component] = component_result
                role_samples = samples[:, :, role_index]
                role_truth = truth[:, role_index]
                role_result["energy_score"] = _energy_score(role_samples, role_truth)
                invisible["roles"][role_name] = role_result
            result["invisible"] = invisible
        if "COMPOSITES" in source and "TRUTH/COMPOSITES" in source:
            physics = {}
            for node in source["COMPOSITES"]:
                predicted = source[f"COMPOSITES/{node}"][:, 0].mean(axis=1)
                truth_vector = source[f"TRUTH/COMPOSITES/{node}"][:]
                finite = np.isfinite(truth_vector).all(axis=1)
                predicted_kinematics = _kinematics(predicted[finite])
                truth_kinematics = _kinematics(truth_vector[finite])
                node_result = {"events": int(finite.sum())}
                for observable in ("mass", "pt", "eta", "phi"):
                    residual = (
                        _wrapped_delta(
                            predicted_kinematics[observable],
                            truth_kinematics[observable],
                        )
                        if observable == "phi"
                        else predicted_kinematics[observable]
                        - truth_kinematics[observable]
                    )
                    node_result[observable] = {
                        "bias": float(residual.mean()),
                        "rmse": float(np.sqrt(np.mean(residual ** 2))),
                        "central_68_width": float(
                            np.quantile(residual, 0.84)
                            - np.quantile(residual, 0.16)
                        ),
                    }
                physics[node] = node_result
            if "OBSERVED/MET" in source and "INVISIBLE/samples" in source:
                invisible_samples = source["INVISIBLE/samples"][:, 0]
                predicted_px = invisible_samples[..., 0].sum(axis=2).mean(axis=1)
                predicted_py = invisible_samples[..., 1].sum(axis=2).mean(axis=1)
                met = source["OBSERVED/MET"][:]
                dx = predicted_px - met[:, 1]
                dy = predicted_py - met[:, 2]
                physics["met_closure"] = {
                    "px_bias": float(dx.mean()),
                    "py_bias": float(dy.mean()),
                    "transverse_rmse": float(np.sqrt(np.mean(dx ** 2 + dy ** 2))),
                }
            result["physics"] = physics
        if "JOINT/final_score" in source:
            ranking = source["JOINT/rank_indices"][:]
            reranked_truth = np.take_along_axis(truth_topk, ranking, axis=1)
            result["compatibility"] = {
                "assignment_only_top1": float(truth_topk[:, 0].mean()),
                "combined_top1": float(reranked_truth[:, 0].mean()),
                "mean_logit": float(source["JOINT/compatibility_logit"][:].mean()),
            }
        return result
