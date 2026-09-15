#!/usr/bin/env python3
"""Build the validation-only human review for Study 02."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.studies import load_study


def _mean_std(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
    }


def _metrics(document: dict) -> dict[str, float | int | None]:
    invisible = document["invisible"]
    posterior = invisible["truncated_assignment_weighted_mixture"]
    result: dict[str, float | int | None] = {
        "events": int(document["events"]),
        "energy_score": float(posterior["energy_score_joint"]),
        "posterior_samples": int(posterior["samples_per_event"]),
        "sample_equivalent_nfe_per_event": int(
            document["sample_equivalent_nfe_per_event"]
        ),
    }
    for role, role_values in posterior["roles"].items():
        result[f"{role}.energy_score"] = float(role_values["energy_score"])
        for component in ("px", "py", "pz"):
            component_values = role_values[component]
            for name in ("bias", "rmse", "central_68_width"):
                result[f"{role}.{component}.{name}"] = float(component_values[name])
            for level in (68, 90, 95):
                result[f"{role}.{component}.coverage_{level}"] = float(
                    component_values[f"coverage_{level}"]
                )

    met = document.get("physics", {}).get("met_closure", {})
    for name in ("transverse_rmse", "px_bias", "py_bias", "px_rmse", "py_rmse"):
        if name in met:
            result[f"met_closure.{name}"] = float(met[name])
    for object_name in ("top_e", "top_mu"):
        distribution = document.get("physics", {}).get(object_name, {}).get(
            "top1_assignment_posterior_distribution", {}
        )
        for observable in ("mass", "pt", "eta"):
            values = distribution.get(observable, {})
            for name in ("bias", "rmse", "central_68_width"):
                if name in values:
                    result[f"{object_name}.{observable}.{name}"] = float(values[name])
    root_system = document.get("physics", {}).get("root_system", {})
    for observable in ("system_mass", "system_pt"):
        values = root_system.get(observable, {})
        for name in ("bias", "rmse", "central_68_width"):
            if name in values:
                result[f"ttbar.{observable}.{name}"] = float(values[name])
    return result


def _coverage_summary(row: dict[str, float | int | None]) -> dict[str, float | None]:
    for level in (68, 90, 95):
        values = [
            float(value)
            for key, value in row.items()
            if key.endswith(f".coverage_{level}") and value is not None
        ]
        summary = _mean_std(values)
        row[f"coverage_{level}_mean"] = summary["mean"]
    return {
        f"coverage_{level}": row.get(f"coverage_{level}_mean") for level in (68, 90, 95)
    }


def build_review(output_root: Path, manifest: Path) -> tuple[Path, Path]:
    study = load_study(manifest)
    study_root = output_root / study.name
    records: list[dict] = []
    for variant in study.variants:
        if variant.reporting == "setup":
            continue
        if len(variant.predictions) != 1:
            raise ValueError(
                f"Study 02 review expects one canonical prediction per variant: {variant.name}"
            )
        prediction = variant.predictions[0]
        for seed in study.seeds:
            path = (
                study_root / "ttbar2L" / variant.name / f"seed_{seed}"
                / "evaluation" / f"{prediction.name}.json"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Study 02 review is missing {path}")
            document = json.loads(path.read_text())
            metrics = _metrics(document)
            _coverage_summary(metrics)
            metrics["selection_split"] = study.split
            metrics["test_split_used"] = False
            records.append(
                {
                    "variant": variant.name,
                    "variant_label": variant.label,
                    "seed": seed,
                    "prediction": prediction.name,
                    "evaluation": str(path),
                    "metrics": metrics,
                }
            )

    by_variant: dict[str, list[dict]] = {}
    for record in records:
        by_variant.setdefault(record["variant"], []).append(record)
    aggregate = {}
    metric_names = sorted(records[0]["metrics"])
    for variant, values in sorted(by_variant.items()):
        flat = {}
        for name in metric_names:
            numbers = [
                float(record["metrics"][name])
                for record in values
                if isinstance(record["metrics"].get(name), (int, float))
                and not isinstance(record["metrics"].get(name), bool)
            ]
            if numbers:
                flat[name] = _mean_std(numbers)
        flat["coverage"] = {}
        for level in (68, 90, 95):
            numbers = [
                float(record["metrics"][f"coverage_{level}_mean"])
                for record in values
                if f"coverage_{level}_mean" in record["metrics"]
            ]
            flat["coverage"][str(level)] = _mean_std(numbers)
        aggregate[variant] = {"seeds": values, "aggregate": flat}

    paired = {}
    variants = sorted(by_variant)
    for first_index, first in enumerate(variants):
        for second in variants[first_index + 1 :]:
            first_by_seed = {item["seed"]: item for item in by_variant[first]}
            second_by_seed = {item["seed"]: item for item in by_variant[second]}
            deltas = {
                str(seed): second_by_seed[seed]["metrics"]["energy_score"]
                - first_by_seed[seed]["metrics"]["energy_score"]
                for seed in study.seeds
            }
            paired[f"{second} - {first}"] = {
                "definition": "paired same-model-seed delta; positive means second has higher (worse) energy score",
                "energy_score": deltas,
                "summary": _mean_std(list(deltas.values())),
            }

    review = {
        "study": study.name,
        "topology": "ttbar2L",
        "selection_split": study.split,
        "test_split_used": False,
        "metric_definitions": {
            "energy_score": "evaluation_joint._posterior_metrics energy_score_joint for the truncated assignment-weighted mixture; lower is better",
            "coverage": "evaluation._scalar_posterior_metrics central interval coverage for each momentum component",
            "bias_rmse": "posterior mean residual and root mean squared residual against truth",
            "paired_deltas": "same model seeds, not an event-level paired uncertainty because the maintained JSONs store aggregates",
            "compatibility": "physics quantities are diagnostics; no hard W/top/MET constraint is applied",
        },
        "records": records,
        "aggregate": aggregate,
        "paired_energy_score_deltas": paired,
        "generator_selection": {
            "status": "review_required",
            "rule": "primary posterior quality with calibration, physics, seed-stability, and compute checks; no arbitrary weighted score",
        },
    }
    json_path = study_root / "study02_human_review.json"
    md_path = study_root / "study02_human_review.md"
    json_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Study 02 human review — generator formulations",
        "",
        f"Selection split: `{study.split}`. Test split used: `False`.",
        "",
        "Primary posterior metric (lower energy score is better):",
        "",
        "| variant | seed 42 | seed 43 | seed 44 | mean | sample std |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, value in sorted(aggregate.items()):
        energies = [
            item["metrics"]["energy_score"] for item in value["seeds"]
        ]
        summary = value["aggregate"]["energy_score"]
        lines.append(
            f"| {variant} | {energies[0]:.6f} | {energies[1]:.6f} | {energies[2]:.6f} | "
            f"{summary['mean']:.6f} | {summary['sample_std']:.6f} |"
        )
    lines += [
        "",
        "Mean component coverage and MET closure:",
        "",
        "| variant | 68% | 90% | 95% | MET transverse RMSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, value in sorted(aggregate.items()):
        coverage = value["aggregate"]["coverage"]
        met = value["aggregate"].get("met_closure.transverse_rmse", {}).get("mean")
        lines.append(
            f"| {variant} | {coverage['68']['mean']:.6f} | {coverage['90']['mean']:.6f} | "
            f"{coverage['95']['mean']:.6f} | {met:.6f} |"
        )
    lines += [
        "",
        "Pairwise energy-score deltas are recorded in the JSON review. Positive values mean the second named candidate is worse.",
        "The result is not auto-promoted by this script; the scientific gate applies the stated no-material-regression rule.",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/studies"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/studies/02_generator_formulations.yaml")
    )
    args = parser.parse_args()
    paths = build_review(args.output_root.expanduser().resolve(), args.manifest)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
