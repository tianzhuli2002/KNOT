#!/usr/bin/env python3
"""Evaluate nested posterior prefixes without regenerating stochastic samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.evaluation import (
    _weighted_crps_rows,
    _weighted_quantile_rows,
)
from joint_event_reconstruction.prediction_schema import (
    validate_joint_prediction,
    validate_prediction,
)


K_VALUES = (1, 2, 3, 5, 10)
M_VALUES = (1, 4, 8, 16, 32, 64)
S_VALUES = (1, 4, 8, 16, 32, 64)


def _require_validation_split(source: h5py.File, path: Path) -> None:
    if source.attrs.get("split") != "validation":
        raise ValueError(
            "Posterior convergence is a validation-selection diagnostic; "
            f"expected split='validation' in {path}, found {source.attrs.get('split')!r}"
        )


class _Mean:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.total2 = 0.0

    def add(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        self.count += len(values)
        self.total += float(values.sum())
        self.total2 += float(np.square(values).sum())

    def result(self) -> dict:
        if not self.count:
            raise ValueError("Cannot summarize an empty posterior scan")
        mean = self.total / self.count
        variance = max(self.total2 / self.count - mean * mean, 0.0)
        error = np.sqrt(variance / self.count)
        return {
            "mean": mean,
            "standard_error": float(error),
            "normal_95": [float(mean - 1.96 * error), float(mean + 1.96 * error)],
            "events": self.count,
        }


def _posterior_scores(
    samples: np.ndarray, truth: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (
        samples.ndim != 4
        or truth.ndim != 3
        or samples.shape[0] != truth.shape[0]
        or samples.shape[2:] != truth.shape[1:]
        or truth.shape[2] != 3
        or not np.isfinite(samples).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("Posterior scan arrays must be finite [N,S,Rnu,3] and [N,Rnu,3]")
    mean = np.sum(samples * weights[:, :, None, None], axis=1)
    squared = np.mean(np.square(mean - truth), axis=(1, 2))
    flat = samples.reshape(len(samples), samples.shape[1], -1).astype(np.float64)
    target = truth.reshape(len(truth), -1).astype(np.float64)
    energy = np.empty(len(samples), dtype=np.float64)
    for event in range(len(samples)):
        distances = np.linalg.norm(flat[event] - target[event], axis=1)
        pair = np.linalg.norm(
            flat[event, :, None] - flat[event, None, :], axis=2
        )
        local_weights = weights[event]
        energy[event] = np.sum(local_weights * distances) - 0.5 * np.sum(
            local_weights[:, None] * local_weights[None, :] * pair
        )
    components = []
    coverage = []
    for role in range(truth.shape[1]):
        for component in range(3):
            values = samples[:, :, role, component]
            target_component = truth[:, role, component]
            components.append(
                _weighted_crps_rows(
                    values,
                    target_component,
                    weights,
                )
            )
            low = _weighted_quantile_rows(values, weights, 0.16)
            high = _weighted_quantile_rows(values, weights, 0.84)
            coverage.append(
                (target_component >= low) & (target_component <= high)
            )
    return (
        squared,
        np.mean(np.stack(components, axis=1), axis=1),
        energy,
        np.mean(np.stack(coverage, axis=1), axis=1),
    )


def _finish(mse: _Mean, crps: _Mean, energy: _Mean, coverage: _Mean) -> dict:
    mse_result = mse.result()
    return {
        "posterior_mean_component_rmse": float(np.sqrt(mse_result["mean"])),
        "posterior_mean_component_mse": mse_result,
        "mean_component_crps": crps.result(),
        "energy_score": energy.result(),
        "central_68_coverage": coverage.result(),
    }


def _factorized(path: Path, chunk_size: int) -> dict:
    validate_prediction(path)
    with h5py.File(path, "r") as source:
        _require_validation_split(source, path)
        shape = source["INVISIBLE/samples"].shape
        _, k_max, m_max, _, _ = shape
        if int(source.attrs["K"]) != k_max or int(source.attrs["M"]) != m_max:
            raise ValueError(
                "Factorized posterior K/M attributes do not match stored sample shape"
            )
        conditioning = str(
            source.attrs.get("invisible_assignment_conditioning", "")
        )
        if conditioning != "hypothesis":
            raise ValueError(
                "Factorized posterior convergence requires hypothesis-conditioned "
                f"samples, found invisible_assignment_conditioning={conditioning!r}"
            )
        conditioning_branches = int(
            source["INVISIBLE"].attrs.get(
                "unique_assignment_conditioning_branches", -1
            )
        )
        if conditioning_branches != k_max:
            raise ValueError(
                "Factorized posterior conditioning-branch metadata does not match "
                f"stored K={k_max}: {conditioning_branches}"
            )
        if k_max < max(K_VALUES) or m_max < max(M_VALUES):
            raise ValueError(f"Factorized posterior {shape} is smaller than K=10, M=64")
        scans = {
            **{
                f"K={k},M={m_max}": (_Mean(), _Mean(), _Mean(), _Mean())
                for k in K_VALUES
            },
            **{
                f"K={k_max},M={m}": (_Mean(), _Mean(), _Mean(), _Mean())
                for m in M_VALUES
            },
        }
        assignment = {
            str(k): {"truth": 0, "events": 0, "retained_mass": _Mean()}
            for k in K_VALUES
        }
        for start in range(0, shape[0], chunk_size):
            stop = min(start + chunk_size, shape[0])
            samples = source["INVISIBLE/samples"][start:stop]
            truth = source["TRUTH/NEUTRINO"][start:stop]
            probabilities = source["VISIBLE/constrained_probabilities"][start:stop]
            log_scores = source["VISIBLE/assignment_log_scores"][start:stop]
            truth_topk = source["TRUTH/topk_assignment"][start:stop]
            if (
                not np.isfinite(probabilities).all()
                or (probabilities < 0).any()
                or not truth_topk.dtype == np.bool_
            ):
                raise ValueError("Factorized posterior assignment inputs are invalid")
            if (np.diff(log_scores, axis=1) > 1.0e-6).any():
                raise ValueError(
                    "Factorized posterior hypotheses are not stored in descending score order"
                )
            if not np.isfinite(truth).all():
                raise ValueError("Factorized posterior truth contains non-finite values")
            total_probability = probabilities.sum(axis=1)
            if (total_probability <= 0).any() or not np.isfinite(total_probability).all():
                raise ValueError("Factorized assignment probabilities must have positive sums")
            for k in K_VALUES:
                retained = probabilities[:, :k].sum(axis=1)
                if (retained <= 0).any() or not np.isfinite(retained).all():
                    raise ValueError(f"No positive retained assignment mass at K={k}")
                assignment[str(k)]["truth"] += int(truth_topk[:, :k].any(axis=1).sum())
                assignment[str(k)]["events"] += len(truth)
                assignment[str(k)]["retained_mass"].add(retained)
                # The maximum (K, M) point is also present in the M scan.
                # Evaluate it once so its uncertainty has the correct event count.
                if k == k_max:
                    continue
                branch = probabilities[:, :k] / retained[:, None]
                weights = np.repeat(branch[:, :, None] / m_max, m_max, axis=2)
                flat = samples[:, :k].reshape(len(truth), k * m_max, *truth.shape[1:])
                mse, crps, energy, coverage = _posterior_scores(
                    flat, truth, weights.reshape(len(truth), k * m_max)
                )
                scans[f"K={k},M={m_max}"][0].add(mse)
                scans[f"K={k},M={m_max}"][1].add(crps)
                scans[f"K={k},M={m_max}"][2].add(energy)
                scans[f"K={k},M={m_max}"][3].add(coverage)
            branch = probabilities / probabilities.sum(axis=1, keepdims=True)
            for m in M_VALUES:
                weights = np.repeat(branch[:, :, None] / m, m, axis=2)
                flat = samples[:, :, :m].reshape(
                    len(truth), k_max * m, *truth.shape[1:]
                )
                mse, crps, energy, coverage = _posterior_scores(
                    flat, truth, weights.reshape(len(truth), k_max * m)
                )
                scans[f"K={k_max},M={m}"][0].add(mse)
                scans[f"K={k_max},M={m}"][1].add(crps)
                scans[f"K={k_max},M={m}"][2].add(energy)
                scans[f"K={k_max},M={m}"][3].add(coverage)
    return {
        "prediction": str(path),
        "generated_once": {"Kmax": k_max, "Mmax": m_max},
        "common_random_numbers": True,
        "scan": {name: _finish(*values) for name, values in scans.items()},
        "assignment": {
            k: {
                "topk_truth_coverage": value["truth"] / value["events"],
                "retained_assignment_mass": value["retained_mass"].result(),
            }
            for k, value in assignment.items()
        },
    }


def _joint(path: Path, chunk_size: int) -> dict:
    validate_joint_prediction(path)
    with h5py.File(path, "r") as source:
        _require_validation_split(source, path)
        shape = source["HYPOTHESES/invisible"].shape
        _, s_max, _, _ = shape
        if int(source.attrs["posterior_samples"]) != s_max:
            raise ValueError(
                "Joint posterior samples attribute does not match stored sample shape"
            )
        if s_max < max(S_VALUES):
            raise ValueError(f"Joint posterior {shape} is smaller than S=64")
        scans = {str(s): (_Mean(), _Mean(), _Mean(), _Mean()) for s in S_VALUES}
        assignment = {str(s): {"correct": 0, "events": 0} for s in S_VALUES}
        for start in range(0, shape[0], chunk_size):
            stop = min(start + chunk_size, shape[0])
            samples = source["HYPOTHESES/invisible"][start:stop]
            states = source["HYPOTHESES/assignment_state"][start:stop]
            truth_state = source["TRUTH/assignment_state"][start:stop]
            truth = source["TRUTH/NEUTRINO"][start:stop]
            if not np.isfinite(truth).all():
                raise ValueError("Joint posterior truth contains non-finite values")
            for s in S_VALUES:
                weights = np.full((len(truth), s), 1.0 / s)
                mse, crps, energy, coverage = _posterior_scores(
                    samples[:, :s], truth, weights
                )
                scans[str(s)][0].add(mse)
                scans[str(s)][1].add(crps)
                scans[str(s)][2].add(energy)
                scans[str(s)][3].add(coverage)
                mode = np.asarray(
                    [
                        np.unique(row[:s], return_counts=True)[0][
                            np.argmax(np.unique(row[:s], return_counts=True)[1])
                        ]
                        for row in states
                    ]
                )
                assignment[str(s)]["correct"] += int((mode == truth_state).sum())
                assignment[str(s)]["events"] += len(truth)
    return {
        "prediction": str(path),
        "generated_once": {"Smax": s_max},
        "common_random_numbers": True,
        "scan": {name: _finish(*values) for name, values in scans.items()},
        "assignment_empirical_mode_accuracy": {
            s: value["correct"] / value["events"] for s, value in assignment.items()
        },
    }


def _parse_scan_key(value: str) -> dict[str, int]:
    return {
        name: int(number)
        for name, number in (item.split("=", 1) for item in value.split(","))
    }


def _save_scan_plot(
    values: list[int],
    result: list[dict],
    *,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = np.asarray([item["mean"] for item in result], dtype=np.float64)
    errors = np.asarray([item["standard_error"] for item in result], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.errorbar(values, means, yerr=errors, marker="o", capsize=3)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plots(result: dict, directory: Path) -> list[str]:
    """Write small quality-versus-posterior-size diagnostics."""
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    factorized_scan = result["factorized"]["scan"]
    k_entries = sorted(
        (
            _parse_scan_key(key)["K"],
            value,
        )
        for key, value in factorized_scan.items()
        if key.startswith("K=") and _parse_scan_key(key)["M"] == max(M_VALUES)
    )
    m_entries = sorted(
        (
            _parse_scan_key(key)["M"],
            value,
        )
        for key, value in factorized_scan.items()
        if key.startswith("K=") and _parse_scan_key(key)["K"] == max(K_VALUES)
    )
    for label, entries, xlabel in (
        ("K", k_entries, "Retained assignment hypotheses K"),
        ("M", m_entries, "Invisible samples per assignment M"),
    ):
        axis_values = [value for value, _ in entries]
        for metric, y_label in (
            ("energy_score", "Energy score"),
            ("mean_component_crps", "Mean component CRPS"),
        ):
            path = directory / f"factorized_{metric}_vs_{label}.png"
            _save_scan_plot(
                axis_values,
                [item[metric] for _, item in entries],
                xlabel=xlabel,
                ylabel=y_label,
                path=path,
            )
            written.append(str(path))

    joint_scan = result["joint"]["scan"]
    s_entries = sorted((int(key), value) for key, value in joint_scan.items())
    for metric, y_label in (
        ("energy_score", "Energy score"),
        ("mean_component_crps", "Mean component CRPS"),
    ):
        path = directory / f"joint_{metric}_vs_S.png"
        _save_scan_plot(
            [value for value, _ in s_entries],
            [item[metric] for _, item in s_entries],
            xlabel="Joint posterior samples S",
            ylabel=y_label,
            path=path,
        )
        written.append(str(path))
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorized", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    factorized = args.factorized.expanduser().resolve()
    joint = args.joint.expanduser().resolve()
    with h5py.File(factorized, "r") as first, h5py.File(joint, "r") as second:
        _require_validation_split(first, factorized)
        _require_validation_split(second, joint)
        if not np.array_equal(first["EVENT_ID"][:], second["EVENT_ID"][:]):
            raise ValueError("Posterior convergence requires identical ordered EVENT_ID")
    result = {
        "selection_split": "validation",
        "factorized": _factorized(factorized, args.chunk_size),
        "joint": _joint(joint, args.chunk_size),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result["plots"] = _plots(result, output.with_name(f"{output.stem}_plots"))
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
