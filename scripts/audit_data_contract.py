#!/usr/bin/env python3
"""Audit the real canonical inputs retained by one compact JER experiment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from joint_event_reconstruction.configuration import load_experiment


QUANTILES = (0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0)


def _scaler_contract(transform) -> tuple[float | None, float | None, bool]:
    if isinstance(transform, list):
        if len(transform) != 3 or transform[0] != "minmax":
            raise ValueError(f"Unsupported public transform: {transform}")
        return float(transform[1]), float(transform[2]), False
    text = str(transform).lower()
    if text == "log":
        return None, None, True
    return None, None, False


def _new_stats(transform) -> dict:
    lower, upper, positive = _scaler_contract(transform)
    return {
        "transform": transform,
        "lower": lower,
        "upper": upper,
        "positive": positive,
        "count": 0,
        "finite": 0,
        "nan": 0,
        "inf": 0,
        "zero": 0,
        "outside": 0,
        "minimum": math.inf,
        "maximum": -math.inf,
        "sum": 0.0,
        "sum2": 0.0,
        "sample": [],
    }


def _update(stats: dict, values: np.ndarray, sample_stride: int) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    stats["count"] += len(values)
    nan = np.isnan(values)
    inf = np.isinf(values)
    stats["nan"] += int(nan.sum())
    stats["inf"] += int(inf.sum())
    finite = np.isfinite(values)
    stats["finite"] += int(finite.sum())
    values = values[finite]
    if not len(values):
        return
    stats["zero"] += int((values == 0).sum())
    if stats["positive"]:
        stats["outside"] += int((values <= 0).sum())
    elif stats["lower"] is not None:
        stats["outside"] += int(
            ((values < stats["lower"]) | (values > stats["upper"])).sum()
        )
    stats["minimum"] = min(stats["minimum"], float(values.min()))
    stats["maximum"] = max(stats["maximum"], float(values.max()))
    stats["sum"] += float(values.sum(dtype=np.float64))
    stats["sum2"] += float(np.square(values).sum(dtype=np.float64))
    stats["sample"].append(values[::sample_stride])


def _finish(stats: dict) -> dict:
    count = stats.pop("count")
    finite = stats.pop("finite")
    nan = stats.pop("nan")
    inf = stats.pop("inf")
    sample = np.concatenate(stats.pop("sample")) if finite else np.empty(0)
    mean = stats.pop("sum") / finite if finite else None
    sum2 = stats.pop("sum2")
    variance = max(sum2 / finite - mean * mean, 0.0) if finite else None
    minimum = stats.pop("minimum")
    maximum = stats.pop("maximum")
    result = {
        "count": count,
        "finite_fraction": finite / count if count else None,
        "nan": nan,
        "inf": inf,
        "nonfinite": count - finite,
        "minimum": minimum if finite else None,
        "maximum": maximum if finite else None,
        "mean": mean,
        "variance": variance,
        "zero_fraction": stats.pop("zero") / finite if finite else None,
        "outside_scaler_count": stats.pop("outside"),
        "quantiles": {
            str(value): float(np.quantile(sample, value)) for value in QUANTILES
        },
        **stats,
    }
    result["outside_scaler_fraction"] = (
        result["outside_scaler_count"] / finite if finite else None
    )
    return result


def audit(experiment_path: Path, chunk_size: int) -> dict:
    experiment = load_experiment(experiment_path)
    data = OmegaConf.to_container(experiment.input_definition, resolve=True)
    source_path = Path(data["input"])

    forbidden = ("truth", "label", "target")
    active = {}
    configured = data["collections"]
    for object_name, values in configured.items():
        variables = {name: name for name in values["features"]}
        if any(any(word in name.lower() for word in forbidden) for name in variables):
            raise ValueError(f"Truth-derived feature name in {object_name}: {list(variables)}")
        active[object_name] = {
            name: {**_new_stats(data["transforms"][name]), "source": source}
            for name, source in variables.items()
        }

    with h5py.File(source_path, "r") as source:
        total = len(source["EVENT_ID"])
        sample_stride = max(1, math.ceil(total / 250_000))
        retained_jets = int(data["collections"]["JET"]["count"])
        njet_over = 0
        electron_outside = 0
        muon_outside = 0
        either_outside = 0
        reconstructable = 0
        by_njet: dict[int, dict[str, int]] = {}

        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            njet = source["INPUTS/GLOBAL"][start:stop, 0]["njet"].astype(int)
            labels = source["LABELS/JET"][start:stop]
            positions = np.arange(labels.shape[1])[None, :]
            electron = labels == 1
            muon = labels == 2
            in_retained = positions < retained_jets
            event_reconstructable = (electron & in_retained).any(axis=1) & (
                muon & in_retained
            ).any(axis=1)
            electron_event_outside = (electron & ~in_retained).any(axis=1)
            muon_event_outside = (muon & ~in_retained).any(axis=1)
            event_outside = electron_event_outside | muon_event_outside
            njet_over += int((njet > retained_jets).sum())
            electron_outside += int(electron_event_outside.sum())
            muon_outside += int(muon_event_outside.sum())
            either_outside += int(event_outside.sum())
            reconstructable += int(event_reconstructable.sum())
            for value in np.unique(njet):
                mask = njet == value
                row = by_njet.setdefault(
                    int(value),
                    {
                        "events": 0,
                        "reconstructable": 0,
                        "electron_outside": 0,
                        "muon_outside": 0,
                        "truth_outside": 0,
                    },
                )
                row["events"] += int(mask.sum())
                row["reconstructable"] += int(event_reconstructable[mask].sum())
                row["truth_outside"] += int(event_outside[mask].sum())
                row["electron_outside"] += int(electron_event_outside[mask].sum())
                row["muon_outside"] += int(muon_event_outside[mask].sum())

            for object_name, variables in active.items():
                values = source[f"INPUTS/{object_name}"][start:stop]
                if object_name == "JET":
                    count = int(data["collections"][object_name]["count"])
                    values = values[:, :count]
                    mask = np.arange(count)[None, :] < np.minimum(njet, count)[:, None]
                else:
                    mask = np.ones(values.shape, dtype=bool)
                for _, stats in variables.items():
                    _update(stats, values[stats["source"]][mask], sample_stride)

    for row in by_njet.values():
        row["reconstructable_fraction"] = row["reconstructable"] / row["events"]
        row["electron_outside_fraction"] = row["electron_outside"] / row["events"]
        row["muon_outside_fraction"] = row["muon_outside"] / row["events"]
        row["truth_outside_fraction"] = row["truth_outside"] / row["events"]
    return {
        "experiment": str(experiment_path),
        "canonical_file": str(source_path),
        "events": total,
        "retained_jets": retained_jets,
        "njet_above_retained": njet_over,
        "njet_above_retained_fraction": njet_over / total,
        "electron_side_b_outside_retained": electron_outside,
        "electron_side_b_outside_retained_fraction": electron_outside / total,
        "muon_side_b_outside_retained": muon_outside,
        "muon_side_b_outside_retained_fraction": muon_outside / total,
        "truth_role_b_outside_retained": either_outside,
        "truth_role_b_outside_retained_fraction": either_outside / total,
        "reconstructable": reconstructable,
        "reconstructable_fraction": reconstructable / total,
        "by_njet": {str(key): value for key, value in sorted(by_njet.items())},
        "active_inputs": {
            object_name: {
                variable: _finish(stats) for variable, stats in variables.items()
            }
            for object_name, variables in active.items()
        },
        "truth_derived_model_inputs": [],
        "quantiles_are_deterministic_stride_sample": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    experiment = args.experiment.expanduser().resolve()
    result = audit(experiment, args.chunk_size)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
