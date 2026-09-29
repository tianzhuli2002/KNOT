#!/usr/bin/env python3
"""Compare two source-paired validation posterior reductions with event bootstrap."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from reduce_realnu_parent_posterior import (
    ANGLE_OBSERVABLES,
    ESTIMATORS,
    OBSERVABLES,
    _wrap,
)

BOOTSTRAP_OBSERVABLES = ("mttbar", "top_phi", "antitop_phi")
BOOTSTRAP_PLAN = {
    "mttbar": {
        "inclusive": ("mae", "rmse"),
        "truth_mttbar/345-400": ("bias", "rmse"),
        "truth_mttbar/>1000": ("rmse",),
    },
    "top_phi": {"inclusive": ("mae", "rmse")},
    "antitop_phi": {"inclusive": ("mae", "rmse")},
}


def _load(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    metadata = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if metadata.get("schema") != "realnu_parent_posterior_reduction_v1":
        raise ValueError(f"Unsupported parent-reduction summary: {path}")
    if metadata.get("sealed_test_used") is not False:
        raise ValueError(f"Parent-reduction summary does not establish sealed_test_used=false: {path}")
    arrays_path = Path(metadata["arrays"]).expanduser().resolve()
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    return metadata, arrays


def _regions(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ids = arrays["EVENT_ID"]
    masks = {"inclusive": np.ones(len(ids), dtype=bool)}
    mass = np.asarray(arrays["truth__mttbar"], dtype=np.float64)
    mass_regions = {
        "truth_mttbar/<345": mass < 345.0,
        "truth_mttbar/345-400": (mass >= 345.0) & (mass < 400.0),
        "truth_mttbar/400-500": (mass >= 400.0) & (mass < 500.0),
        "truth_mttbar/500-700": (mass >= 500.0) & (mass < 700.0),
        "truth_mttbar/700-1000": (mass >= 700.0) & (mass <= 1000.0),
        "truth_mttbar/>1000": mass > 1000.0,
    }
    if mass.shape != (len(ids),) or not np.isfinite(mass).all():
        raise ValueError("Truth mttbar must be finite and aligned with paired EVENT_IDs")
    if not np.all(np.sum(np.stack(tuple(mass_regions.values())), axis=0) == 1):
        raise ValueError("Truth-mttbar bootstrap regions do not partition validation events")
    masks.update(mass_regions)
    channels = np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value)
         for value in arrays["channel"]]
    )
    for value in np.unique(channels):
        masks[f"channel/{value}"] = channels == value
    for name in ("model_visible_njet", "source_njet"):
        for value in np.unique(arrays[name]):
            masks[f"{name}/{int(value)}"] = arrays[name] == value
    return masks


def _metric_value(residual: np.ndarray, metric: str) -> float:
    if metric == "bias":
        return float(np.mean(residual))
    if metric == "mae":
        return float(np.mean(np.abs(residual)))
    if metric == "rmse":
        return float(np.sqrt(np.mean(residual * residual)))
    raise ValueError(f"Unsupported paired bootstrap metric {metric!r}")


def _paired_bootstrap_delta(
    candidate_residual: np.ndarray,
    reference_residual: np.ndarray,
    *,
    metric: str,
    replicates: int,
    rng: np.random.Generator,
    chunk_replicates: int = 8,
) -> dict[str, float | int]:
    if candidate_residual.shape != reference_residual.shape or candidate_residual.ndim != 1:
        raise ValueError("Paired residual vectors must be one-dimensional and aligned")
    count = len(candidate_residual)
    if count == 0:
        raise ValueError("Cannot bootstrap an empty validation region")
    point = _metric_value(candidate_residual, metric) - _metric_value(
        reference_residual, metric
    )
    bootstrap = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, chunk_replicates):
        stop = min(start + chunk_replicates, replicates)
        indices = rng.integers(0, count, size=(stop - start, count), dtype=np.int32)
        candidate_chunk = candidate_residual[indices]
        reference_chunk = reference_residual[indices]
        if metric == "bias":
            candidate_values = candidate_chunk.mean(axis=1)
            reference_values = reference_chunk.mean(axis=1)
        elif metric == "mae":
            candidate_values = np.abs(candidate_chunk).mean(axis=1)
            reference_values = np.abs(reference_chunk).mean(axis=1)
        elif metric == "rmse":
            candidate_values = np.sqrt(np.mean(candidate_chunk * candidate_chunk, axis=1))
            reference_values = np.sqrt(np.mean(reference_chunk * reference_chunk, axis=1))
        else:
            raise ValueError(f"Unsupported paired bootstrap metric {metric!r}")
        bootstrap[start:stop] = candidate_values - reference_values
    return {
        "events": count,
        "candidate_minus_reference": point,
        "metric": metric,
        "bootstrap_95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_95_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_probability_candidate_lower": float(np.mean(bootstrap < 0.0)),
        "bootstrap_replicates": int(replicates),
    }


def compare(
    reference_summary: Path,
    candidate_summary: Path,
    *,
    output: Path,
    csv_output: Path | None = None,
    replicates: int = 500,
    seed: int = 42011,
    bootstrap_regions: bool = True,
    reference_estimator: str | None = None,
    candidate_estimator: str | None = None,
) -> dict:
    if replicates < 50:
        raise ValueError("Require at least 50 paired bootstrap replicates")
    reference_meta, reference = _load(reference_summary)
    candidate_meta, candidate = _load(candidate_summary)
    for name in ("source_bank_sha256", "case", "solver", "steps", "samples"):
        if reference_meta.get(name) != candidate_meta.get(name):
            raise ValueError(
                f"Paired parent comparison requires matching {name}: "
                f"reference={reference_meta.get(name)!r}, candidate={candidate_meta.get(name)!r}"
            )
    reference_ids = np.asarray(reference["EVENT_ID"], dtype=np.int64)
    candidate_ids = np.asarray(candidate["EVENT_ID"], dtype=np.int64)
    if len(np.unique(reference_ids)) != len(reference_ids) or len(np.unique(candidate_ids)) != len(candidate_ids):
        raise ValueError("Paired parent summaries contain duplicate EVENT_IDs")
    candidate_order_by_id = {int(event_id): index for index, event_id in enumerate(candidate_ids)}
    if set(map(int, reference_ids)) != set(candidate_order_by_id):
        raise ValueError("Paired parent summaries must cover the exact same validation EVENT_ID set")
    reorder = np.asarray([candidate_order_by_id[int(event_id)] for event_id in reference_ids])
    for name in ("channel", "source_njet", "model_visible_njet", "support_parent_roles"):
        if not np.array_equal(reference[name], candidate[name][reorder]):
            raise ValueError(f"Validation region/truth support differs after EVENT_ID join: {name}")
    for name in OBSERVABLES:
        if not np.allclose(
            reference[f"truth__{name}"], candidate[f"truth__{name}"][reorder],
            rtol=1.0e-6, atol=1.0e-5, equal_nan=True,
        ):
            raise ValueError(f"Truth observable differs after exact EVENT_ID join: {name}")

    if (reference_estimator is None) != (candidate_estimator is None):
        raise ValueError("Set both estimator names or neither")
    if reference_estimator is not None:
        if reference_estimator not in ESTIMATORS or candidate_estimator not in ESTIMATORS:
            raise ValueError(f"Estimator names must be selected from {ESTIMATORS}")
        estimator_pairs = [(reference_estimator, candidate_estimator)]
    else:
        estimator_pairs = [(estimator, estimator) for estimator in ESTIMATORS]

    regions = _regions(reference)
    active_plan = BOOTSTRAP_PLAN
    if not bootstrap_regions:
        active_plan = {
            observable: {"inclusive": region_map["inclusive"]}
            for observable, region_map in BOOTSTRAP_PLAN.items()
            if "inclusive" in region_map
        }
    rng = np.random.default_rng(seed)
    comparisons = {}
    for reference_name, candidate_name in estimator_pairs:
        pair_name = (
            candidate_name
            if candidate_name == reference_name
            else f"{candidate_name}_minus_{reference_name}"
        )
        comparisons[pair_name] = {}
        for observable in BOOTSTRAP_OBSERVABLES:
            truth = reference[f"truth__{observable}"].astype(np.float64)
            reference_value = reference[
                f"pred__{reference_name}__{observable}"
            ].astype(np.float64)
            candidate_value = candidate[
                f"pred__{candidate_name}__{observable}"
            ][reorder].astype(np.float64)
            if observable in ANGLE_OBSERVABLES:
                reference_residual = _wrap(reference_value - truth)
                candidate_residual = _wrap(candidate_value - truth)
            else:
                reference_residual = reference_value - truth
                candidate_residual = candidate_value - truth
            selected_regions = {
                name: regions[name]
                for name in active_plan[observable]
            }
            comparisons[pair_name][observable] = {}
            for region, mask in selected_regions.items():
                if not mask.any():
                    comparisons[pair_name][observable][region] = {
                        metric: {"events": 0, "status": "empty region"}
                        for metric in active_plan[observable][region]
                    }
                    continue
                comparisons[pair_name][observable][region] = {
                    metric: _paired_bootstrap_delta(
                        candidate_residual[mask],
                        reference_residual[mask],
                        metric=metric,
                        replicates=replicates,
                        rng=rng,
                    )
                    for metric in active_plan[observable][region]
                }

    result = {
        "status": "passed",
        "schema": "realnu_parent_paired_comparison_v1",
        "reference": {
            "summary": str(reference_summary.expanduser().resolve()),
            "model": reference_meta["velocity_backbone"],
            "conditioning": reference_meta["conditioning"],
            "checkpoint_sha256": reference_meta["checkpoint_sha256"],
        },
        "candidate": {
            "summary": str(candidate_summary.expanduser().resolve()),
            "model": candidate_meta["velocity_backbone"],
            "conditioning": candidate_meta["conditioning"],
            "checkpoint_sha256": candidate_meta["checkpoint_sha256"],
        },
        "estimators": [
            {"reference": reference_name, "candidate": candidate_name}
            for reference_name, candidate_name in estimator_pairs
        ],
        "case": reference_meta["case"],
        "solver": reference_meta["solver"],
        "steps": reference_meta["steps"],
        "nfe_per_sample": reference_meta["nfe_per_sample"],
        "samples": reference_meta["samples"],
        "events": int(len(reference_ids)),
        "event_id_sha256": reference_meta["event_id_sha256"],
        "source_bank_sha256": reference_meta["source_bank_sha256"],
        "bootstrap": {
            "metrics_by_observable_and_region": {
                observable: {
                    region: list(metrics) for region, metrics in regions.items()
                }
                for observable, regions in active_plan.items()
            },
            "resampling_unit": "validation EVENT_ID, paired across models",
            "replicates": replicates,
            "seed": seed,
            "regions_included": sorted({
                region for region_map in active_plan.values()
                for region in region_map
            }),
            "training_seed_variation_included": False,
        },
        "interpretation": "Negative candidate-minus-reference RMSE favors the candidate.",
        "comparisons": comparisons,
        "truth_free_conditioning": True,
        "sealed_test_used": False,
    }
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = (
        csv_output.expanduser().resolve()
        if csv_output is not None
        else output.with_suffix(".csv")
    )
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair_name, observables in comparisons.items():
        for observable, region_map in observables.items():
            for region, metric_map in region_map.items():
                for metric, values in metric_map.items():
                    reference_estimator, candidate_estimator = next(
                        (reference_name, candidate_name)
                        for reference_name, candidate_name in estimator_pairs
                        if candidate_name == pair_name
                        or f"{candidate_name}_minus_{reference_name}" == pair_name
                    )
                    rows.append({
                        "comparison": pair_name,
                        "reference_model": reference_meta["velocity_backbone"],
                        "candidate_model": candidate_meta["velocity_backbone"],
                        "reference_estimator": reference_estimator,
                        "candidate_estimator": candidate_estimator,
                        "observable": observable,
                        "region": region,
                        "metric": metric,
                        **values,
                    })
    csv_fields = (
        "comparison", "reference_model", "candidate_model", "reference_estimator",
        "candidate_estimator", "observable", "region", "metric", "events",
        "candidate_minus_reference", "bootstrap_95_low", "bootstrap_95_high",
        "bootstrap_probability_candidate_lower", "bootstrap_replicates", "status",
    )
    with csv_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    result["csv_output"] = str(csv_output)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42011)
    parser.add_argument("--bootstrap-regions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-estimator", choices=ESTIMATORS)
    parser.add_argument("--candidate-estimator", choices=ESTIMATORS)
    args = parser.parse_args()
    result = compare(
        args.reference,
        args.candidate,
        output=args.output,
        csv_output=args.csv_output,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
        bootstrap_regions=args.bootstrap_regions,
        reference_estimator=args.reference_estimator,
        candidate_estimator=args.candidate_estimator,
    )
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output.expanduser().resolve()),
        "csv_output": result["csv_output"],
        "events": result["events"],
        "estimators": result["estimators"],
        "bootstrap_replicates": result["bootstrap"]["replicates"],
        "source_bank_sha256": result["source_bank_sha256"],
        "training_seed_variation_included": result["bootstrap"]["training_seed_variation_included"],
        "sealed_test_used": result["sealed_test_used"],
    }, indent=2))


if __name__ == "__main__":
    main()
