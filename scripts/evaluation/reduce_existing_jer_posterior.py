#!/usr/bin/env python3
"""Evaluate truth-independent point reductions on an existing JER posterior."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np


def _p4(momentum: np.ndarray) -> np.ndarray:
    energy = np.sqrt(np.square(momentum).sum(axis=-1))
    return np.concatenate((energy[..., None], momentum), axis=-1)


def _kinematics(momentum: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pt = np.hypot(momentum[..., 0], momentum[..., 1])
    phi = np.arctan2(momentum[..., 1], momentum[..., 0])
    eta = np.arcsinh(np.divide(momentum[..., 2], np.maximum(pt, 1.0e-8)))
    return pt, eta, phi


def _summary(prediction: np.ndarray, truth: np.ndarray) -> dict:
    residual = prediction - truth
    p_pt, p_eta, p_phi = _kinematics(prediction)
    t_pt, t_eta, t_phi = _kinematics(truth)
    dphi = np.arctan2(np.sin(p_phi - t_phi), np.cos(p_phi - t_phi))
    delta_r = np.hypot(p_eta - t_eta, dphi)
    values = {
        "px": residual[..., 0], "py": residual[..., 1], "pz": residual[..., 2],
        "pt": p_pt - t_pt, "eta": p_eta - t_eta, "phi": dphi, "delta_r": delta_r,
        "momentum_magnitude": np.linalg.norm(prediction, axis=-1) - np.linalg.norm(truth, axis=-1),
    }
    result = {}
    for name, value in values.items():
        flat = np.asarray(value, dtype=np.float64).reshape(-1)
        result[name] = {
            "bias": float(np.mean(flat)),
            "mae": float(np.mean(np.abs(flat))),
            "rmse": float(np.sqrt(np.mean(np.square(flat)))),
            "quantiles_abs": {
                str(q): float(np.quantile(np.abs(flat), q)) for q in (0.5, 0.9, 0.95, 0.99)
            },
        }
    return result


def _parent_summary(prediction: np.ndarray, truth: np.ndarray) -> dict:
    valid = np.isfinite(prediction).all(axis=-1) & np.isfinite(truth).all(axis=-1)
    residual = prediction - truth
    mass2_pred = np.square(prediction[..., 0]) - np.square(prediction[..., 1:]).sum(axis=-1)
    mass2_truth = np.square(truth[..., 0]) - np.square(truth[..., 1:]).sum(axis=-1)
    pred_pt = np.hypot(prediction[..., 1], prediction[..., 2])
    truth_pt = np.hypot(truth[..., 1], truth[..., 2])
    pred_rapidity = np.arctanh(np.divide(prediction[..., 3], np.maximum(prediction[..., 0], 1.0e-8)))
    truth_rapidity = np.arctanh(np.divide(truth[..., 3], np.maximum(truth[..., 0], 1.0e-8)))
    values = {
        "E": residual[..., 0], "px": residual[..., 1], "py": residual[..., 2], "pz": residual[..., 3],
        "mass": np.sqrt(np.maximum(mass2_pred, 0.0)) - np.sqrt(np.maximum(mass2_truth, 0.0)),
        "pt": pred_pt - truth_pt, "rapidity": pred_rapidity - truth_rapidity,
    }
    result = {}
    for name, value in values.items():
        flat = np.asarray(value, dtype=np.float64)[valid].reshape(-1)
        result[name] = {
            "bias": None if len(flat) == 0 else float(np.mean(flat)),
            "mae": None if len(flat) == 0 else float(np.mean(np.abs(flat))),
            "rmse": None if len(flat) == 0 else float(np.sqrt(np.mean(np.square(flat)))),
        }
    result["valid_events"] = int(valid.sum())
    result["invalid_events"] = int((~valid).sum())
    result["predicted_mass2_negative_fraction"] = float(np.mean(mass2_pred[valid] < 0.0)) if valid.any() else None
    result["truth_mass2_negative_fraction"] = float(np.mean(mass2_truth[valid] < 0.0)) if valid.any() else None
    return result


def _medoid(samples: np.ndarray, scale: np.ndarray, chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Return one complete sampled configuration and its sample index."""
    n_events, n_samples, roles, dimensions = samples.shape
    scores = np.empty((n_events, n_samples), dtype=np.float32)
    normalised = samples / scale[None, None, :, :]
    for start in range(0, n_events, chunk):
        stop = min(n_events, start + chunk)
        difference = normalised[start:stop, :, None, :, :] - normalised[start:stop, None, :, :, :]
        distances = np.sqrt(np.square(difference).sum(axis=(-1, -2)))
        scores[start:stop] = distances.mean(axis=-1)
    index = scores.argmin(axis=-1)
    selected = samples[np.arange(n_events), index]
    return selected, index


def evaluate(prediction_path: Path, output_path: Path, *, nfe_per_draw: int, wall_time_seconds: float | None) -> dict:
    started = time.perf_counter()
    with h5py.File(prediction_path, "r") as source:
        samples = np.asarray(source["INVISIBLE/samples"][:, 0], dtype=np.float32)
        truth = np.asarray(source["TRUTH/NEUTRINO"][:], dtype=np.float32)
        event_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        scaler = json.loads(str(source.attrs["scaler"]))
        scale = np.asarray(scaler["std"], dtype=np.float32)
        parent_samples = {
            role: np.asarray(source[f"COMPOSITES/{role}"][:, 0], dtype=np.float32)
            for role in ("top_e", "top_mu")
        }
        parent_truth = {
            role: np.asarray(source[f"TRUTH/COMPOSITES/{role}"][:], dtype=np.float32)
            for role in ("top_e", "top_mu")
        }
        attrs = {key: (value.item() if hasattr(value, "item") else value) for key, value in source.attrs.items()}

    if samples.ndim != 4 or samples.shape[2:] != truth.shape[1:]:
        raise ValueError(f"Posterior/truth shapes are incompatible: samples={samples.shape}, truth={truth.shape}")
    results = {}
    for count in (1, 16, min(32, samples.shape[1])):
        nested = samples[:, :count]
        medoid, medoid_index = _medoid(nested, scale)
        mean = nested.mean(axis=1)
        one = nested[:, 0]
        strategies = {"cfm_single_draw": one, "cfm_cartesian_mean": mean, "cfm_joint_medoid": medoid}
        results[str(count)] = {}
        for name, momentum in strategies.items():
            record = {
                "per_role": {
                    role: _summary(momentum[:, role], truth[:, role])
                    for role in range(truth.shape[1])
                },
                "posterior_sample_count": int(count),
                "solver_network_evaluations_per_draw": int(nfe_per_draw),
                "sample_equivalent_network_evaluations": int(count * nfe_per_draw),
            }
            if name == "cfm_joint_medoid":
                record["selected_sample_index_histogram"] = {
                    str(index): int(np.sum(medoid_index == index)) for index in range(count)
                }
            parent = {}
            for role_index, role in enumerate(parent_samples):
                sampled_parent = parent_samples[role][:, :count]
                visible = sampled_parent - _p4(nested[:, :, role_index])
                if name == "cfm_single_draw":
                    point_parent = sampled_parent[:, 0]
                elif name == "cfm_cartesian_mean":
                    point_parent = visible.mean(axis=1) + _p4(mean[:, role_index])
                else:
                    point_parent = sampled_parent[np.arange(len(sampled_parent)), medoid_index]
                parent[role] = _parent_summary(point_parent, parent_truth[role])
            record["parent_from_same_point_solution"] = parent
            results[str(count)][name] = record

    report = {
        "source_prediction": str(prediction_path.resolve()),
        "events": int(len(event_ids)),
        "event_id_sha256": __import__("hashlib").sha256(np.ascontiguousarray(event_ids).tobytes()).hexdigest(),
        "source_metadata": attrs,
        "strategies": results,
        "posterior_diagnostics_reference": str(
            prediction_path.parent.parent / "evaluation" / f"{prediction_path.stem}.json"
        ),
        "compute": {
            "nfe_per_draw": int(nfe_per_draw),
            "recorded_slurm_wall_time_seconds": wall_time_seconds,
            "reduction_cpu_wall_time_seconds": float(time.perf_counter() - started),
            "gpu_memory_bytes": None,
            "note": "The full posterior calibration/energy-score result remains in the existing Study-08 evaluation JSON; reductions are point-estimator diagnostics.",
        },
        "validation_population": "historical Study-08 validation; its event IDs are not the new matched manifest",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nfe-per-draw", type=int, required=True)
    parser.add_argument("--wall-time-seconds", type=float)
    args = parser.parse_args()
    report = evaluate(args.prediction.expanduser().resolve(), args.output.expanduser().resolve(), nfe_per_draw=args.nfe_per_draw, wall_time_seconds=args.wall_time_seconds)
    print(json.dumps({"output": str(args.output.resolve()), "events": report["events"], "strategies": list(report["strategies"].keys())}, indent=2))


if __name__ == "__main__":
    main()
