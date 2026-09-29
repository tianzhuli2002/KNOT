#!/usr/bin/env python3
"""Build paired validation point estimates and posterior summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
PRODUCTION_ROOT = PROJECT_ROOT / (
    "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/"
    "realnu_top_antitop_20260927_release2"
)
EVENT_TABLE = PRODUCTION_ROOT / "evaluation_r2/core/event_metrics.h5"
ANGLE_OBSERVABLES = {
    "top_phi",
    "antitop_phi",
    "phittbar",
    "delta_phi_top_antitop",
}
OBSERVABLES = (
    "top_mass",
    "antitop_mass",
    "top_pt",
    "antitop_pt",
    "top_rapidity",
    "antitop_rapidity",
    "top_phi",
    "antitop_phi",
    "mttbar",
    "ptttbar",
    "yttbar",
    "phittbar",
    "delta_phi_top_antitop",
    "delta_rapidity_top_antitop",
    "cos_theta_star",
)
ESTIMATORS = ("single_draw", "observable_mean", "observable_q50", "joint_medoid")


def _sha256_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.remainder(angle + np.pi, 2.0 * np.pi) - np.pi


def _observables(parents: np.ndarray) -> dict[str, np.ndarray]:
    """Derive event observables from (...,2,4) future-timelike parents."""
    parents = np.asarray(parents, dtype=np.float64)
    if parents.ndim < 3 or parents.shape[-2:] != (2, 4):
        raise ValueError(f"Expected parent vectors ending in [2,4], found {parents.shape}")
    if not np.isfinite(parents).all():
        raise ValueError("Parent posterior contains non-finite four-vectors")
    top, antitop = parents[..., 0, :], parents[..., 1, :]
    pair = top + antitop

    def mass(vector):
        mass2 = vector[..., 0].square() if hasattr(vector[..., 0], "square") else vector[..., 0] ** 2
        mass2 = mass2 - np.sum(vector[..., 1:] ** 2, axis=-1)
        if np.any(mass2 < -1.0e-3):
            raise ValueError("Parent posterior contains a spacelike four-vector")
        return np.sqrt(np.maximum(mass2, 0.0))

    def rapidity(vector):
        plus, minus = vector[..., 0] + vector[..., 3], vector[..., 0] - vector[..., 3]
        if np.any(plus <= 0.0) or np.any(minus <= 0.0):
            raise ValueError("Parent posterior contains an invalid rapidity argument")
        return 0.5 * (np.log(plus) - np.log(minus))

    top_phi = np.arctan2(top[..., 2], top[..., 1])
    antitop_phi = np.arctan2(antitop[..., 2], antitop[..., 1])
    top_y, antitop_y = rapidity(top), rapidity(antitop)
    pair_energy, pair_momentum = pair[..., 0], pair[..., 1:]
    beta = pair_momentum / pair_energy[..., None]
    beta2 = np.sum(beta * beta, axis=-1)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta2, 1.0e-12))
    beta_dot_top = np.sum(beta * top[..., 1:], axis=-1)
    coefficient = (
        (gamma - 1.0) * beta_dot_top / np.maximum(beta2, 1.0e-12)
        - gamma * top[..., 0]
    )
    top_rest = top[..., 1:] + coefficient[..., None] * beta
    axis_norm = np.linalg.norm(pair_momentum, axis=-1)
    axis = np.zeros_like(pair_momentum)
    axis[..., 2] = 1.0
    nonzero = axis_norm > 1.0e-8
    axis[nonzero] = pair_momentum[nonzero] / axis_norm[nonzero, None]
    cos_theta = np.sum(top_rest * axis, axis=-1) / np.maximum(
        np.linalg.norm(top_rest, axis=-1), 1.0e-12
    )
    return {
        "top_mass": mass(top),
        "antitop_mass": mass(antitop),
        "top_pt": np.linalg.norm(top[..., 1:3], axis=-1),
        "antitop_pt": np.linalg.norm(antitop[..., 1:3], axis=-1),
        "top_rapidity": top_y,
        "antitop_rapidity": antitop_y,
        "top_phi": top_phi,
        "antitop_phi": antitop_phi,
        "mttbar": mass(pair),
        "ptttbar": np.linalg.norm(pair[..., 1:3], axis=-1),
        "yttbar": rapidity(pair),
        "phittbar": np.arctan2(pair[..., 2], pair[..., 1]),
        "delta_phi_top_antitop": _wrap(top_phi - antitop_phi),
        "delta_rapidity_top_antitop": top_y - antitop_y,
        "cos_theta_star": np.clip(cos_theta, -1.0, 1.0),
    }


def _circular_mean(values: np.ndarray, axis: int = 1) -> np.ndarray:
    return np.arctan2(np.mean(np.sin(values), axis=axis), np.mean(np.cos(values), axis=axis))


def _circular_median(values: np.ndarray) -> np.ndarray:
    center = _circular_mean(values)
    offsets = _wrap(values - center[:, None])
    return _wrap(center + np.median(offsets, axis=1))


def _joint_medoid(states: np.ndarray, *, chunk_size: int = 512) -> np.ndarray:
    """Choose the posterior sample minimizing summed Euclidean distance in scaled 8D."""
    states = np.asarray(states, dtype=np.float32)
    if states.ndim != 3 or states.shape[-1] != 8:
        raise ValueError(f"Joint parent states must be [N,M,8], found {states.shape}")
    result = np.empty(len(states), dtype=np.int64)
    for start in range(0, len(states), chunk_size):
        stop = min(start + chunk_size, len(states))
        values = states[start:stop]
        distances = np.linalg.norm(
            values[:, :, None, :] - values[:, None, :, :], axis=-1
        )
        result[start:stop] = distances.sum(axis=-1).argmin(axis=-1)
    return result


def _truth_join(table_path: Path, event_ids: np.ndarray) -> dict[str, np.ndarray]:
    with h5py.File(table_path, "r") as table:
        if table.attrs.get("schema") != "realnu_event_metrics_v1":
            raise ValueError("Accepted event table has an unexpected schema")
        if bool(table.attrs.get("sealed_test_used", True)):
            raise ValueError("Accepted event table does not establish sealed_test_used=false")
        table_ids = np.asarray(table["EVENT_ID"][:], dtype=np.int64)
        if len(np.unique(table_ids)) != len(table_ids):
            raise ValueError("Accepted event table contains duplicate EVENT_IDs")
        order = np.argsort(table_ids, kind="stable")
        positions = np.searchsorted(table_ids[order], event_ids)
        if np.any(positions >= len(order)) or not np.array_equal(table_ids[order[positions]], event_ids):
            raise ValueError("Prediction EVENT_IDs do not exactly join to validation truth")
        row = order[positions]
        support = np.asarray(table["support/parent_roles"][:][row], dtype=bool)
        if support.shape != (len(event_ids), 2) or not support.all():
            missing = int((~support.all(axis=1)).sum())
            raise ValueError(
                f"Validation truth has {missing} unsupported parent events in the requested sample"
            )
        result = {
            "parents": np.asarray(table["truth/parents"][:][row], dtype=np.float64),
            "mttbar": np.asarray(table["truth/mttbar"][:][row], dtype=np.float64),
            "channel": np.asarray(table["channel"][:][row]),
            "source_njet": np.asarray(table["source_njet"][:][row], dtype=np.int64),
            "model_visible_njet": np.asarray(table["model_visible_njet"][:][row], dtype=np.int64),
            "support_parent_roles": support,
        }
    return result


def _region_masks(truth: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    count = len(truth["parents"])
    masks = {"inclusive": np.ones(count, dtype=bool)}
    mass = np.asarray(truth["mttbar"], dtype=np.float64)
    if mass.shape != (count,) or not np.isfinite(mass).all():
        raise ValueError("Truth mttbar must be finite and aligned with the validation events")
    mass_regions = {
        "truth_mttbar/<345": mass < 345.0,
        "truth_mttbar/345-400": (mass >= 345.0) & (mass < 400.0),
        "truth_mttbar/400-500": (mass >= 400.0) & (mass < 500.0),
        "truth_mttbar/500-700": (mass >= 500.0) & (mass < 700.0),
        "truth_mttbar/700-1000": (mass >= 700.0) & (mass <= 1000.0),
        "truth_mttbar/>1000": mass > 1000.0,
    }
    if not np.all(np.sum(np.stack(tuple(mass_regions.values())), axis=0) == 1):
        raise RuntimeError("Truth-mttbar regions do not partition the validation events")
    masks.update(mass_regions)
    channel = np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value)
         for value in truth["channel"]]
    )
    for value in np.unique(channel):
        masks[f"channel/{value}"] = channel == value
    for name in ("model_visible_njet", "source_njet"):
        values = truth[name]
        for value in np.unique(values):
            masks[f"{name}/{int(value)}"] = values == value
    return masks


def _metric_record(
    prediction: np.ndarray,
    truth: np.ndarray,
    sample_values: np.ndarray,
    posterior_mean: np.ndarray,
    mask: np.ndarray,
    *,
    observable: str,
) -> dict[str, float | int | None]:
    if mask.dtype != bool or mask.shape != truth.shape:
        raise ValueError("Validation region mask does not align with observable rows")
    if not mask.any():
        empty: dict[str, float | int | None] = {
            "events": 0,
            "bias": None,
            "std": None,
            "sigma68": None,
            "mae": None,
            "rmse": None,
            "median_absolute_error": None,
            "posterior_width_68_median": None,
            "posterior_width_68_mean": None,
            "posterior_width_90_median": None,
            "posterior_width_90_mean": None,
            "truth_coverage_68": None,
            "truth_coverage_90": None,
        }
        if observable in {
            "top_mass", "antitop_mass", "top_pt", "antitop_pt", "mttbar", "ptttbar"
        }:
            empty.update(
                {
                    "fractional_bias": None,
                    "fractional_sigma68": None,
                    "fractional_rmse": None,
                }
            )
        return empty
    prediction = np.asarray(prediction, dtype=np.float64)[mask]
    truth = np.asarray(truth, dtype=np.float64)[mask]
    samples = np.asarray(sample_values, dtype=np.float64)[mask]
    center = np.asarray(posterior_mean, dtype=np.float64)[mask]
    residual = _wrap(prediction - truth) if observable in ANGLE_OBSERVABLES else prediction - truth
    bias = float(np.mean(residual))
    absolute = np.abs(residual)
    residual_q16, residual_q84 = np.quantile(residual, [0.16, 0.84])
    sigma68 = float((residual_q84 - residual_q16) / 2.0)
    fractional_observables = {
        "top_mass", "antitop_mass", "top_pt", "antitop_pt", "mttbar", "ptttbar"
    }
    result: dict[str, float | int | None] = {
        "events": int(mask.sum()),
        "bias": bias,
        "std": float(np.std(residual)),
        "sigma68": sigma68,
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "median_absolute_error": float(np.median(absolute)),
    }
    if observable in fractional_observables:
        truth_scale = float(np.mean(truth))
        if not np.isfinite(truth_scale) or truth_scale <= 0.0:
            raise ValueError(
                f"Cannot form fractional {observable} metrics with mean truth {truth_scale}"
            )
        result.update(
            {
                "fractional_bias": bias / truth_scale,
                "fractional_sigma68": sigma68 / truth_scale,
                "fractional_rmse": float(np.sqrt(np.mean(residual * residual))) / truth_scale,
            }
        )

    if observable not in ANGLE_OBSERVABLES:
        q16, q84 = np.quantile(samples, [0.16, 0.84], axis=1)
        q05, q95 = np.quantile(samples, [0.05, 0.95], axis=1)
        covered68 = (truth >= q16) & (truth <= q84)
        covered90 = (truth >= q05) & (truth <= q95)
        posterior_width68 = q84 - q16
        posterior_width90 = q95 - q05
    else:
        offsets = _wrap(samples - center[:, None])
        q16, q84 = np.quantile(offsets, [0.16, 0.84], axis=1)
        q05, q95 = np.quantile(offsets, [0.05, 0.95], axis=1)
        truth_offset = _wrap(truth - center)
        covered68 = (truth_offset >= q16) & (truth_offset <= q84)
        covered90 = (truth_offset >= q05) & (truth_offset <= q95)
        posterior_width68 = q84 - q16
        posterior_width90 = q95 - q05
    result.update(
        {
            "posterior_width_68_median": float(np.median(posterior_width68)),
            "posterior_width_68_mean": float(np.mean(posterior_width68)),
            "posterior_width_90_median": float(np.median(posterior_width90)),
            "posterior_width_90_mean": float(np.mean(posterior_width90)),
            "truth_coverage_68": float(np.mean(covered68)),
            "truth_coverage_90": float(np.mean(covered90)),
        }
    )
    return result


def reduce(
    predictions_path: Path,
    *,
    case: str,
    samples: int,
    event_table: Path,
    summary_path: Path,
    arrays_path: Path,
    metrics_csv_path: Path | None = None,
) -> dict:
    predictions_path = predictions_path.expanduser().resolve()
    event_table = event_table.expanduser().resolve()
    summary_path = summary_path.expanduser().resolve()
    arrays_path = arrays_path.expanduser().resolve()
    with h5py.File(predictions_path, "r") as source:
        if source.attrs.get("prediction_schema") != "realnu_parent_solver_scan_v1":
            raise ValueError(f"Unsupported parent scan schema: {predictions_path}")
        if not bool(source.attrs.get("truth_free_input", False)):
            raise ValueError("Parent predictions must record truth_free_input=true")
        if bool(source.attrs.get("sealed_test_used", True)):
            raise ValueError("Parent predictions must record sealed_test_used=false")
        if "TRUTH" in source or "TRUTH_SIDE_CAR" in source:
            raise ValueError("Truth-bearing parent predictions are not accepted")
        if case not in source["CASES"]:
            raise KeyError(f"Parent scan has no requested solver case {case!r}")
        event_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError("Parent prediction file contains duplicate EVENT_IDs")
        group = source[f"CASES/{case}"]
        parent_all = np.asarray(group["parents"][:], dtype=np.float32)
        scaled_all = np.asarray(group["scaled_states"][:], dtype=np.float32)
        nfe = int(group.attrs["nfe_per_sample"])
        solver_steps = int(group.attrs["steps"])
        solver = str(group.attrs["solver"])
        velocity_backbone = str(source.attrs["velocity_backbone"])
        conditioning = str(source.attrs["conditioning"])
        source_bank_sha = str(source.attrs["source_bank_sha256"])
        checkpoint_sha = str(source.attrs["checkpoint_sha256"])
    if samples <= 0 or samples > parent_all.shape[1]:
        raise ValueError(f"samples must be in [1,{parent_all.shape[1]}]")
    parents = parent_all[:, :samples]
    scaled = scaled_all[:, :samples]
    if parents.shape != (len(event_ids), samples, 2, 4) or scaled.shape != (len(event_ids), samples, 8):
        raise ValueError("Parent prediction and scaled-state arrays do not align")
    truth = _truth_join(event_table, event_ids)
    truth_observables = {
        name: values[:, 0]
        for name, values in _observables(truth["parents"][:, None]).items()
    }
    if not np.allclose(
        truth_observables["mttbar"], truth["mttbar"], rtol=5.0e-6, atol=1.0e-3
    ):
        maximum = float(np.max(np.abs(truth_observables["mttbar"] - truth["mttbar"])))
        raise ValueError(
            "Truth mttbar does not close against the exact EVENT_ID-joined parents; "
            f"maximum difference={maximum} GeV"
        )
    sample_observables = _observables(parents)
    mean_estimator = {}
    median_estimator = {}
    for name, values in sample_observables.items():
        if name in ANGLE_OBSERVABLES:
            mean_estimator[name] = _circular_mean(values)
            median_estimator[name] = _circular_median(values)
        else:
            mean_estimator[name] = values.mean(axis=1)
            median_estimator[name] = np.median(values, axis=1)
    medoid_index = _joint_medoid(scaled)
    event_index = np.arange(len(event_ids))
    medoid_parents = parents[event_index, medoid_index]
    medoid_observables = _observables(medoid_parents)
    # This is the maintained production parent-pair energy score, in physical p4 units.
    from plot_realnu_campaign import _energy_score as joint_parent_energy_score

    energy_score = joint_parent_energy_score(parents, truth["parents"])
    point_estimates = {
        "single_draw": {name: values[:, 0] for name, values in sample_observables.items()},
        "observable_mean": mean_estimator,
        "observable_q50": median_estimator,
        "joint_medoid": medoid_observables,
    }
    if set(point_estimates) != set(ESTIMATORS):
        raise RuntimeError("Parent point-estimator set changed unexpectedly")

    arrays = {
        "EVENT_ID": event_ids,
        "channel": np.asarray(
            [
                (value.decode("utf-8") if isinstance(value, bytes) else str(value)).encode("utf-8")
                for value in truth["channel"]
            ], dtype="S8"
        ),
        "source_njet": truth["source_njet"],
        "model_visible_njet": truth["model_visible_njet"],
        "support_parent_roles": truth["support_parent_roles"],
        "joint_medoid_sample_index": medoid_index,
        "parent_joint_medoid": medoid_parents.astype(np.float32),
    }
    for name in OBSERVABLES:
        arrays[f"truth__{name}"] = truth_observables[name].astype(np.float32)
        arrays[f"posterior_q16__{name}"] = np.quantile(
            sample_observables[name], 0.16, axis=1
        ).astype(np.float32) if name not in ANGLE_OBSERVABLES else np.quantile(
            _wrap(sample_observables[name] - mean_estimator[name][:, None]), 0.16, axis=1
        ).astype(np.float32) + mean_estimator[name].astype(np.float32)
        arrays[f"posterior_q84__{name}"] = np.quantile(
            sample_observables[name], 0.84, axis=1
        ).astype(np.float32) if name not in ANGLE_OBSERVABLES else np.quantile(
            _wrap(sample_observables[name] - mean_estimator[name][:, None]), 0.84, axis=1
        ).astype(np.float32) + mean_estimator[name].astype(np.float32)
        if name in ANGLE_OBSERVABLES:
            arrays[f"posterior_q16__{name}"] = _wrap(
                arrays[f"posterior_q16__{name}"]
            ).astype(np.float32)
            arrays[f"posterior_q84__{name}"] = _wrap(
                arrays[f"posterior_q84__{name}"]
            ).astype(np.float32)
        for estimator, values in point_estimates.items():
            arrays[f"pred__{estimator}__{name}"] = np.asarray(
                values[name], dtype=np.float32
            )

    regions = _region_masks(truth)
    report = {}
    for estimator, predictions in point_estimates.items():
        report[estimator] = {}
        for name in OBSERVABLES:
            report[estimator][name] = {
                region: _metric_record(
                    predictions[name],
                    truth_observables[name],
                    sample_observables[name],
                    mean_estimator[name],
                    mask,
                    observable=name,
                )
                for region, mask in regions.items()
            }
    metadata = {
        "status": "passed",
        "schema": "realnu_parent_posterior_reduction_v1",
        "predictions": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "event_table": str(event_table),
        "case": case,
        "solver": solver,
        "steps": solver_steps,
        "nfe_per_sample": nfe,
        "samples": samples,
        "velocity_backbone": velocity_backbone,
        "conditioning": conditioning,
        "checkpoint_sha256": checkpoint_sha,
        "source_bank_sha256": source_bank_sha,
        "events": int(len(event_ids)),
        "event_id_sha256": _sha256_ids(event_ids),
        "parent_role_order": ["top", "antitop"],
        "truth_free_conditioning": True,
        "sealed_test_used": False,
        "estimators": {
            "single_draw": "sample index 0 from the fixed common source bank",
            "observable_mean": "posterior mean per observable; circular mean for angles",
            "observable_q50": "posterior median per observable; circular median for angles",
            "joint_medoid": "sample minimizing summed Euclidean distance in train-scaled joint 8D parent state",
        },
        "metric_definitions": {
            "bias": "mean wrapped residual for angular observables, otherwise mean residual",
            "std": "population standard deviation of event residuals",
            "sigma68": "half the central 16-84 percent width of the event residual distribution",
            "fractional_metrics": "bias, sigma68, or RMSE divided by mean truth in the region for positive dimensional observables",
            "truth_coverage_68": "fraction of exact truth values inside each event's central posterior 16-84 percent interval",
            "truth_coverage_90": "fraction of exact truth values inside each event's central posterior 5-95 percent interval",
            "joint_energy_score_parent_p4_gev": "maintained realnu parent-pair multivariate energy score in the physical eight-component (top, antitop) four-vector space; lower is better",
            "truth_mttbar_regions": {
                "<345": "mttbar < 345 GeV",
                "345-400": "345 <= mttbar < 400 GeV",
                "400-500": "400 <= mttbar < 500 GeV",
                "500-700": "500 <= mttbar < 700 GeV",
                "700-1000": "700 <= mttbar <= 1000 GeV",
                ">1000": "mttbar > 1000 GeV",
            },
        },
        "angular_observables": sorted(ANGLE_OBSERVABLES),
        "region_counts": {name: int(mask.sum()) for name, mask in regions.items()},
        "metrics": report,
        "joint_energy_score_parent_p4_gev": energy_score,
        "arrays": str(arrays_path),
    }
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_csv_path = (
        metrics_csv_path.expanduser().resolve()
        if metrics_csv_path is not None
        else summary_path.with_name(f"{summary_path.stem}_metrics.csv")
    )
    metric_fields = (
        "bias", "std", "sigma68", "mae", "rmse", "median_absolute_error",
        "fractional_bias", "fractional_sigma68", "fractional_rmse",
        "posterior_width_68_median", "posterior_width_68_mean",
        "posterior_width_90_median", "posterior_width_90_mean",
        "truth_coverage_68", "truth_coverage_90",
    )
    csv_fields = (
        "velocity_backbone", "conditioning", "case", "solver", "steps",
        "nfe_per_sample", "samples", "events", "estimator", "observable",
        "region", *metric_fields,
    )
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for estimator, observables in report.items():
            for observable, region_records in observables.items():
                for region, values in region_records.items():
                    writer.writerow({
                        "velocity_backbone": velocity_backbone,
                        "conditioning": conditioning,
                        "case": case,
                        "solver": solver,
                        "steps": solver_steps,
                        "nfe_per_sample": nfe,
                        "samples": samples,
                        "events": values["events"],
                        "estimator": estimator,
                        "observable": observable,
                        "region": region,
                        **{field: values.get(field) for field in metric_fields},
                    })
    metadata["metrics_csv"] = str(metrics_csv_path)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{arrays_path.name}.", suffix=".tmp", dir=arrays_path.parent
    )
    os.close(fd)
    temporary_arrays = Path(temporary_name)
    try:
        with temporary_arrays.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary_arrays, arrays_path)
    finally:
        temporary_arrays.unlink(missing_ok=True)
    summary_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--case", default="heun_steps_32")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--event-table", type=Path, default=EVENT_TABLE)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--metrics-csv", type=Path)
    args = parser.parse_args()
    result = reduce(
        args.predictions,
        case=args.case,
        samples=args.samples,
        event_table=args.event_table,
        summary_path=args.summary,
        arrays_path=args.arrays,
        metrics_csv_path=args.metrics_csv,
    )
    print(json.dumps({
        "status": result["status"],
        "summary": str(args.summary.expanduser().resolve()),
        "arrays": str(args.arrays.expanduser().resolve()),
        "metrics_csv": result["metrics_csv"],
        "events": result["events"],
        "samples": result["samples"],
        "event_id_sha256": result["event_id_sha256"],
        "source_bank_sha256": result["source_bank_sha256"],
        "joint_energy_score_parent_p4_gev": result["joint_energy_score_parent_p4_gev"],
        "sealed_test_used": result["sealed_test_used"],
    }, indent=2))


if __name__ == "__main__":
    main()
