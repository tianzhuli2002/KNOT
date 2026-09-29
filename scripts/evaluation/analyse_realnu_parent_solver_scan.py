#!/usr/bin/env python3
"""Reduce the paired PCFM_Q scan into solver and posterior-budget evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from reduce_realnu_parent_posterior import (
    ANGLE_OBSERVABLES,
    ESTIMATORS,
    OBSERVABLES,
    _circular_mean,
    _circular_median,
    _joint_medoid,
    _metric_record,
    _observables,
    _region_masks,
    _truth_join,
    _wrap,
)

PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
PRODUCTION_ROOT = PROJECT_ROOT / (
    "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/"
    "realnu_top_antitop_20260927_release2"
)
EVENT_TABLE = PRODUCTION_ROOT / "evaluation_r2/core/event_metrics.h5"
SAMPLE_COUNTS = (1, 2, 4, 8, 16, 32)
EQUAL_NFE = ((32, 1), (16, 2), (8, 4), (4, 8))
KEY_OBSERVABLES = (
    "mttbar", "top_pt", "antitop_pt", "top_rapidity", "antitop_rapidity",
    "top_phi", "antitop_phi", "ptttbar", "yttbar",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _estimates(sample_observables: dict[str, np.ndarray], states: np.ndarray) -> dict:
    means, medians = {}, {}
    for name, values in sample_observables.items():
        if name in ANGLE_OBSERVABLES:
            means[name] = _circular_mean(values)
            medians[name] = _circular_median(values)
        else:
            means[name] = values.mean(axis=1)
            medians[name] = np.median(values, axis=1)
    medoid_index = _joint_medoid(states)
    event_index = np.arange(len(states))
    medoid_parents = sample_observables["_parents"][event_index, medoid_index]
    medoid = _observables(medoid_parents)
    return {
        "single_draw": {name: values[:, 0] for name, values in sample_observables.items() if name != "_parents"},
        "observable_mean": means,
        "observable_q50": medians,
        "joint_medoid": medoid,
    }


def _metric_rows(
    estimates: dict[str, dict[str, np.ndarray]],
    sample_observables: dict[str, np.ndarray],
    truth_observables: dict[str, np.ndarray],
    regions: dict[str, np.ndarray],
) -> dict:
    results = {}
    for estimator in ESTIMATORS:
        results[estimator] = {}
        for observable in OBSERVABLES:
            selected_regions = (
                regions
                if observable == "mttbar"
                else {"inclusive": regions["inclusive"]}
            )
            results[estimator][observable] = {
                region: _metric_record(
                    estimates[estimator][observable],
                    truth_observables[observable],
                    sample_observables[observable],
                    estimates["observable_mean"][observable],
                    mask,
                    observable=observable,
                )
                for region, mask in selected_regions.items()
            }
    return results


def _truth_observables(truth_parents: np.ndarray) -> dict[str, np.ndarray]:
    values = _observables(truth_parents[:, None])
    return {name: array[:, 0] for name, array in values.items()}


def analyse(predictions: Path, output_dir: Path, *, event_table: Path = EVENT_TABLE) -> dict:
    predictions = predictions.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    event_table = event_table.expanduser().resolve()
    if not predictions.is_file():
        raise FileNotFoundError(f"PCFM_Q scan output is missing: {predictions}")
    if not event_table.is_file():
        raise FileNotFoundError(f"Accepted validation event table is missing: {event_table}")

    with h5py.File(predictions, "r") as source:
        if source.attrs.get("prediction_schema") != "realnu_parent_solver_scan_v1":
            raise ValueError("Unsupported parent solver-scan prediction schema")
        if not bool(source.attrs.get("truth_free_input", False)):
            raise ValueError("Parent solver scan must record truth_free_input=true")
        if bool(source.attrs.get("sealed_test_used", True)):
            raise ValueError("Parent solver scan must record sealed_test_used=false")
        if "TRUTH" in source or "TRUTH_SIDE_CAR" in source:
            raise ValueError("Solver scan output must not contain truth arrays")
        event_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        source_bank_sha = str(source.attrs["source_bank_sha256"])
        checkpoint_sha = str(source.attrs["checkpoint_sha256"])
        solver = str(source.attrs["solver"])
        backbone = str(source.attrs["velocity_backbone"])
        conditioning = str(source.attrs["conditioning"])
        cases = {}
        for name in source["CASES"]:
            group = source[f"CASES/{name}"]
            parents = np.asarray(group["parents"][:], dtype=np.float32)
            states = np.asarray(group["scaled_states"][:], dtype=np.float32)
            case_solver = str(group.attrs["solver"])
            case_steps = int(group.attrs["steps"])
            actual_nfe = int(group.attrs["nfe_per_sample"])
            expected_nfe = 2 * case_steps if case_solver == "heun" else case_steps
            if case_solver != solver or actual_nfe != expected_nfe:
                raise ValueError(
                    f"Solver/NFE metadata mismatch in {name}: solver={case_solver}, "
                    f"steps={case_steps}, recorded NFE={actual_nfe}, expected={expected_nfe}"
                )
            cases[name] = {
                "parents": parents,
                "states": states,
                "steps": case_steps,
                "nfe_per_sample": actual_nfe,
                "elapsed_seconds": (
                    float(group.attrs["elapsed_seconds"])
                    if "elapsed_seconds" in group.attrs else None
                ),
            }
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("PCFM_Q scan contains duplicate EVENT_IDs")
    truth = _truth_join(event_table, event_ids)
    truth_observables = _truth_observables(truth["parents"])
    if not np.allclose(
        truth_observables["mttbar"], truth["mttbar"], rtol=5.0e-6, atol=1.0e-3
    ):
        raise ValueError("Truth mttbar closure failed after exact EVENT_ID joining")
    regions = _region_masks(truth)

    result_cases = {}
    for case_name, case in cases.items():
        parents = case["parents"]
        states = case["states"]
        if parents.ndim != 4 or parents.shape[0] != len(event_ids) or parents.shape[-2:] != (2, 4):
            raise ValueError(f"Malformed parent predictions in {case_name}: {parents.shape}")
        if states.shape != (len(event_ids), parents.shape[1], 8):
            raise ValueError(f"Malformed scaled states in {case_name}: {states.shape}")
        if not np.isfinite(parents).all() or not np.isfinite(states).all():
            raise ValueError(f"Non-finite parent predictions in {case_name}")
        result_cases[case_name] = {
            "steps": case["steps"],
            "nfe_per_sample": case["nfe_per_sample"],
            "elapsed_seconds_at_max_samples": case["elapsed_seconds"],
            "sample_counts": {},
        }
        max_samples = min(parents.shape[1], max(SAMPLE_COUNTS))
        for sample_count in SAMPLE_COUNTS:
            if sample_count > max_samples:
                continue
            sample_parents = parents[:, :sample_count]
            sample_states = states[:, :sample_count]
            raw_observables = _observables(sample_parents)
            raw_observables["_parents"] = sample_parents
            estimates = _estimates(raw_observables, sample_states)
            sample_observables = {
                name: values
                for name, values in raw_observables.items()
                if name != "_parents"
            }
            result_cases[case_name]["sample_counts"][str(sample_count)] = {
                "samples": sample_count,
                "nfe_per_sample": case["nfe_per_sample"],
                "aggregate_nfe_per_event": case["nfe_per_sample"] * sample_count,
                "metrics": _metric_rows(
                    estimates, sample_observables, truth_observables, regions
                ),
            }

    reference_name = "heun_steps_32"
    solver_convergence = {}
    if reference_name in cases:
        ref_case = cases[reference_name]
        ref_parent = ref_case["parents"]
        ref_states = ref_case["states"]
        for sample_count in (1, 8, 32):
            if sample_count > ref_parent.shape[1]:
                continue
            ref_obs = _observables(ref_parent[:, :sample_count])
            ref_obs["_parents"] = ref_parent[:, :sample_count]
            ref_est = _estimates(ref_obs, ref_states[:, :sample_count])
            solver_convergence[str(sample_count)] = {}
            for case_name, case in cases.items():
                if case["steps"] == 32:
                    continue
                obs = _observables(case["parents"][:, :sample_count])
                obs["_parents"] = case["parents"][:, :sample_count]
                estimates = _estimates(obs, case["states"][:, :sample_count])
                solver_convergence[str(sample_count)][case_name] = {}
                for estimator in ("single_draw", "observable_mean", "observable_q50", "joint_medoid"):
                    observable_deltas = {}
                    for observable in ("mttbar", "top_phi", "antitop_phi"):
                        delta = (
                            estimates[estimator][observable]
                            - ref_est[estimator][observable]
                        )
                        reconstruction = (
                            ref_est[estimator][observable]
                            - truth_observables[observable]
                        )
                        if observable in ANGLE_OBSERVABLES:
                            delta = _wrap(delta)
                            reconstruction = _wrap(reconstruction)
                        delta_rmse = float(np.sqrt(np.mean(delta * delta)))
                        reconstruction_rmse = float(
                            np.sqrt(np.mean(reconstruction * reconstruction))
                        )
                        observable_deltas[observable] = {
                            "events": int(len(delta)),
                            "delta_bias": float(np.mean(delta)),
                            "delta_mae": float(np.mean(np.abs(delta))),
                            "delta_rmse": delta_rmse,
                            "delta_q95_abs": float(np.quantile(np.abs(delta), 0.95)),
                            "reference_reconstruction_rmse": reconstruction_rmse,
                            "solver_delta_over_reconstruction_rmse": delta_rmse
                            / max(reconstruction_rmse, 1.0e-12),
                            "unit": "GeV" if observable == "mttbar" else "rad",
                        }
                    solver_convergence[str(sample_count)][case_name][estimator] = {
                        "observables": observable_deltas,
                    }

    equal_nfe = []
    for step_count, sample_count in EQUAL_NFE:
        case_name = f"heun_steps_{step_count}"
        if case_name in result_cases and str(sample_count) in result_cases[case_name]["sample_counts"]:
            equal_nfe.append(
                {
                    "solver": "heun",
                    "steps": step_count,
                    "samples": sample_count,
                    "nfe_per_sample": 2 * step_count,
                    "aggregate_nfe_per_event": 2 * step_count * sample_count,
                    "metrics": result_cases[case_name]["sample_counts"][str(sample_count)]["metrics"],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "status": "completed",
        "schema": "realnu_parent_solver_scan_analysis_v1",
        "predictions": str(predictions),
        "predictions_sha256": _sha256(predictions),
        "event_table": str(event_table),
        "events": int(len(event_ids)),
        "event_id_sha256": hashlib.sha256(np.asarray(event_ids, dtype="<i8").tobytes()).hexdigest(),
        "source_bank_sha256": source_bank_sha,
        "checkpoint_sha256": checkpoint_sha,
        "velocity_backbone": backbone,
        "conditioning": conditioning,
        "solver": solver,
        "max_samples": int(max(case["parents"].shape[1] for case in cases.values())),
        "sealed_test_used": False,
        "truth_free_input": True,
        "region_counts": {name: int(mask.sum()) for name, mask in regions.items()},
        "metric_scope": {
            "all_regions": "inclusive plus the six required truth-mttbar ranges",
            "other_observables": "inclusive only",
            "sigma68": "half central 16-84 percent event-residual width",
            "compute_cost": "NFE per sample and aggregate NFE per event are exact; per-case wall time is included only when measured in the scan file",
        },
        "equal_nfe_cases": equal_nfe,
        "solver_convergence_vs_heun32": solver_convergence,
        "cases": result_cases,
    }
    summary_path = output_dir / "PCFM_Q_solver_scan_analysis.json"
    summary_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    rows = []
    for case_name, case in result_cases.items():
        for sample_count, item in case["sample_counts"].items():
            for estimator in ESTIMATORS:
                for observable in KEY_OBSERVABLES:
                    inclusive = item["metrics"][estimator][observable]["inclusive"]
                    rows.append(
                        {
                            "case": case_name,
                            "steps": case["steps"],
                            "samples": int(sample_count),
                            "estimator": estimator,
                            "observable": observable,
                            "nfe_per_sample": case["nfe_per_sample"],
                            "aggregate_nfe_per_event": case["nfe_per_sample"] * int(sample_count),
                            "events": inclusive["events"],
                            "bias": inclusive["bias"],
                            "std": inclusive["std"],
                            "sigma68": inclusive["sigma68"],
                            "mae": inclusive["mae"],
                            "rmse": inclusive["rmse"],
                            "elapsed_seconds_at_max_samples": case["elapsed_seconds_at_max_samples"],
                        }
                    )
    csv_path = output_dir / "PCFM_Q_solver_scan_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "completed", "summary": str(summary_path), "metrics_csv": str(csv_path), "events": len(event_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-table", type=Path, default=EVENT_TABLE)
    args = parser.parse_args()
    print(json.dumps(analyse(args.predictions, args.output_dir, event_table=args.event_table), indent=2))


if __name__ == "__main__":
    main()
