#!/usr/bin/env python3
"""Measure constituent-to-parton residual structure on accepted R2 validation rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
PRODUCTION_ROOT = PROJECT_ROOT / (
    "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/"
    "realnu_top_antitop_20260927_release2"
)
EVENT_TABLE = PRODUCTION_ROOT / "evaluation_r2/core/event_metrics.h5"
OUTPUT_ROOT = PRODUCTION_ROOT / "evaluation_v3_diagnostics/residual_parent"
MODELS = ("J0", "J_A", "J_AN")
P4_NAMES = ("E", "px", "py", "pz")


def _metrics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "n": 0, "mean": np.nan, "std": np.nan, "sigma68": np.nan,
            "mae": np.nan, "q95_abs": np.nan, "q99_abs": np.nan,
        }
    q16, q84 = np.quantile(values, [0.16, 0.84])
    absolute = np.abs(values)
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "sigma68": float((q84 - q16) / 2.0),
        "mae": float(absolute.mean()),
        "q95_abs": float(np.quantile(absolute, 0.95)),
        "q99_abs": float(np.quantile(absolute, 0.99)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty diagnostic table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(event_table: Path = EVENT_TABLE, output_root: Path = OUTPUT_ROOT) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    with h5py.File(event_table, "r") as source:
        if source.attrs.get("schema") != "realnu_event_metrics_v1":
            raise ValueError(f"Unexpected R2 event table schema: {event_table}")
        if bool(source.attrs.get("sealed_test_used", True)):
            raise ValueError("R2 event table does not establish sealed_test_used=false")
        if json.loads(str(source.attrs["parent_role_order"])) != ["top", "antitop"]:
            raise ValueError("R2 parent role order differs from [top, antitop]")
        ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        if len(ids) != 100_342 or len(np.unique(ids)) != len(ids):
            raise ValueError("Expected 100,342 unique validation EVENT_IDs")
        truth = np.asarray(source["truth/parents"][:], dtype=np.float64)
        truth_mttbar = np.asarray(source["truth/mttbar"][:], dtype=np.float64)
        source_njet = np.asarray(source["source_njet"][:], dtype=np.int64)
        parent_support = np.asarray(source["support/parent_roles"][:], dtype=bool).all(axis=1)
        if truth.shape != (len(ids), 2, 4) or not parent_support.any():
            raise ValueError("After-FSR truth parent table has an invalid shape or no support")
        models = {}
        for model in MODELS:
            constituent = np.asarray(source[f"models/{model}/constituent_parent"][:], dtype=np.float64)
            valid = np.asarray(source[f"models/{model}/constituent_parent_valid"][:], dtype=bool)
            correct = np.asarray(source[f"models/{model}/assignment_complete_correct"][:], dtype=bool)
            if constituent.shape != truth.shape or valid.shape != (len(ids),):
                raise ValueError(f"{model}: constituent parents or support are misaligned")
            models[model] = (constituent, valid & parent_support, correct)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10})

    truth_pt = np.hypot(truth[:, 0, 1], truth[:, 0, 2])
    strata = {
        "inclusive": np.ones(len(ids), dtype=bool),
        "truth_mttbar_<345": truth_mttbar < 345,
        "truth_mttbar_345_400": (truth_mttbar >= 345) & (truth_mttbar < 400),
        "truth_mttbar_400_500": (truth_mttbar >= 400) & (truth_mttbar < 500),
        "truth_mttbar_500_700": (truth_mttbar >= 500) & (truth_mttbar < 700),
        "truth_mttbar_700_1000": (truth_mttbar >= 700) & (truth_mttbar <= 1000),
        "truth_mttbar_>1000": truth_mttbar > 1000,
        "top_pt_<100": truth_pt < 100,
        "top_pt_100_200": (truth_pt >= 100) & (truth_pt < 200),
        "top_pt_200_400": (truth_pt >= 200) & (truth_pt < 400),
        "top_pt_>400": truth_pt >= 400,
        "source_njet_0_3": source_njet < 4,
        "source_njet_4_5": (source_njet >= 4) & (source_njet < 6),
        "source_njet_6_7": (source_njet >= 6) & (source_njet < 8),
        "source_njet_8plus": source_njet >= 8,
    }
    rows = []
    complexity = []
    correlation = {}
    for model, (constituent, valid, assignment_correct) in models.items():
        residual = truth - constituent  # Required sign: parton parent minus constituent parent.
        residual[~valid] = np.nan
        figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.3), constrained_layout=True)
        for role_index, role in enumerate(("top", "antitop")):
            for component_index, component in enumerate(P4_NAMES):
                axis = axes[role_index, component_index]
                values = residual[valid, role_index, component_index]
                axis.hist(values, bins=100, range=(float(values.min()), float(values.max())),
                          histtype="step", color="#315a8c", linewidth=1.4, log=True)
                summary = _metrics(values)
                axis.axvline(0.0, color="#bb3e3e", linewidth=0.9, linestyle="--")
                axis.set_title(f"{role}: Δ {component}")
                axis.set_xlabel("parton − constituent [GeV]")
                axis.set_ylabel("events")
                axis.text(
                    0.98, 0.96,
                    f"N={summary['n']:,}\nσ={summary['std']:.1f} GeV\n"
                    f"σ68={summary['sigma68']:.1f} GeV\n|Δ|99={summary['q99_abs']:.1f} GeV",
                    transform=axis.transAxes, ha="right", va="top", fontsize=7,
                )
        figure.suptitle(f"{model}: constituent-to-parton parent residuals, full finite range")
        figure.savefig(output_root / f"{model}_residual_components.png", dpi=180)
        figure.savefig(output_root / f"{model}_residual_components.pdf")
        plt.close(figure)
        for stratum, selected in strata.items():
            support = selected & valid
            for role_index, role in enumerate(("top", "antitop")):
                for component_index, component in enumerate(P4_NAMES):
                    measured = _metrics(residual[support, role_index, component_index])
                    target = _metrics(truth[support, role_index, component_index])
                    rows.append({
                        "model": model,
                        "stratum": stratum,
                        "role": role,
                        "component": component,
                        **measured,
                    })
                    complexity.append({
                        "model": model,
                        "stratum": stratum,
                        "role": role,
                        "component": component,
                        "n": measured["n"],
                        "parton_truth_std": target["std"],
                        "residual_std": measured["std"],
                        "residual_to_truth_std": (
                            float(measured["std"] / target["std"])
                            if target["std"] and np.isfinite(target["std"]) else np.nan
                        ),
                        "parton_truth_sigma68": target["sigma68"],
                        "residual_sigma68": measured["sigma68"],
                        "residual_to_truth_sigma68": (
                            float(measured["sigma68"] / target["sigma68"])
                            if target["sigma68"] and np.isfinite(target["sigma68"]) else np.nan
                        ),
                    })
            if stratum == "inclusive":
                flat = residual[valid].reshape(int(valid.sum()), 8)
                correlation[model] = np.corrcoef(flat, rowvar=False).tolist()
            for assignment_class, choose in (
                ("correct", assignment_correct),
                ("incorrect", ~assignment_correct),
            ):
                current = selected & valid & choose
                if stratum != "inclusive":
                    continue
                for role_index, role in enumerate(("top", "antitop")):
                    for component_index, component in enumerate(P4_NAMES):
                        measured = _metrics(residual[current, role_index, component_index])
                        rows.append({
                            "model": model,
                            "stratum": f"assignment_{assignment_class}",
                            "role": role,
                            "component": component,
                            **measured,
                        })

    _write_csv(output_root / "residual_metrics.csv", rows)
    _write_csv(output_root / "target_complexity.csv", complexity)
    correlation_path = output_root / "residual_correlations.json"
    correlation_path.write_text(json.dumps({
        "status": "passed",
        "split": "validation",
        "event_count": int(len(ids)),
        "sealed_test_used": False,
        "residual_definition": "after-FSR parton parent four-vector minus reconstructed constituent parent four-vector",
        "role_order": ["top", "antitop"],
        "component_order": list(P4_NAMES),
        "correlation_component_order": [f"{role}_{component}" for role in ("top", "antitop") for component in P4_NAMES],
        "correlation": correlation,
    }, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "passed",
        "event_table": str(event_table.resolve()),
        "output_root": str(output_root.resolve()),
        "event_count": int(len(ids)),
        "event_id_unique": True,
        "sealed_test_used": False,
        "models": list(MODELS),
        "residual_definition": "after-FSR parton parent four-vector minus reconstructed constituent parent four-vector",
        "strata": list(strata) + ["assignment_correct", "assignment_incorrect"],
        "files": [
            "residual_metrics.csv", "target_complexity.csv", "residual_correlations.json",
            *[f"{mode}_residual_components.{extension}" for mode in MODELS for extension in ("png", "pdf")],
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-table", type=Path, default=EVENT_TABLE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    print(json.dumps(run(args.event_table.expanduser().resolve(), args.output_root.expanduser().resolve()), indent=2))


if __name__ == "__main__":
    main()
