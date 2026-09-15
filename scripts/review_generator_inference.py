#!/usr/bin/env python3
"""Build the validation-only quality-versus-inference-cost review for Study 08."""

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


def build_review(output_root: Path, manifest: Path) -> tuple[Path, Path]:
    study = load_study(manifest)
    study_root = output_root / study.name
    records: list[dict] = []
    for variant in study.variants:
        if variant.reporting == "setup":
            continue
        for prediction in variant.predictions:
            for seed in study.seeds:
                path = (
                    study_root / "ttbar2L" / variant.name / f"seed_{seed}"
                    / "evaluation" / f"{prediction.name}.json"
                )
                if not path.is_file():
                    raise FileNotFoundError(f"Study 08 review is missing {path}")
                document = json.loads(path.read_text())
                posterior = document["invisible"][
                    "truncated_assignment_weighted_mixture"
                ]
                met = document.get("physics", {}).get("met_closure", {})
                records.append(
                    {
                        "variant": variant.name,
                        "variant_label": variant.label,
                        "prediction": prediction.name,
                        "seed": seed,
                        "evaluation": str(path),
                        "nfe_per_sample": int(document["nfe_per_invisible_sample"]),
                        "sample_equivalent_nfe_per_event": int(
                            document["sample_equivalent_nfe_per_event"]
                        ),
                        "energy_score": float(posterior["energy_score_joint"]),
                        "met_transverse_rmse": float(met["transverse_rmse"]),
                        "selection_split": study.split,
                        "test_split_used": False,
                    }
                )

    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["variant"], record["prediction"]), []).append(record)
    aggregate = []
    for (variant, prediction), values in sorted(groups.items()):
        aggregate.append(
            {
                "variant": variant,
                "prediction": prediction,
                "nfe_per_sample": values[0]["nfe_per_sample"],
                "sample_equivalent_nfe_per_event": values[0][
                    "sample_equivalent_nfe_per_event"
                ],
                "energy_score": _mean_std([item["energy_score"] for item in values]),
                "met_transverse_rmse": _mean_std(
                    [item["met_transverse_rmse"] for item in values]
                ),
                "seeds": [item["seed"] for item in values],
            }
        )

    review = {
        "study": study.name,
        "topology": "ttbar2L",
        "selection_split": study.split,
        "test_split_used": False,
        "metric_definitions": {
            "energy_score": "the maintained invisible posterior energy_score_joint; lower is better",
            "cost": "sample_equivalent_nfe_per_event, equal to posterior samples times NFE per sample for this factorized scan",
            "interpretation": "efficiency support for the selected generator, not a new architecture gate",
        },
        "records": records,
        "aggregate": aggregate,
        "recommended_budget": {
            "status": "review_required",
            "rule": "choose the lowest cost on the quality plateau; do not trade calibration or physics regressions for a central energy-score gain alone",
        },
    }
    json_path = study_root / "study08_human_review.json"
    md_path = study_root / "study08_human_review.md"
    plot_path = study_root / "study08_quality_vs_cost.png"
    json_path.write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Study 08 human review — inference cost",
        "",
        f"Selection split: `{study.split}`. Test split used: `False`.",
        "",
        "| variant | prediction | NFE/sample | sample-equivalent NFE/event | energy mean | energy sample std | MET transverse RMSE mean |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in aggregate:
        lines.append(
            f"| {item['variant']} | {item['prediction']} | {item['nfe_per_sample']} | "
            f"{item['sample_equivalent_nfe_per_event']} | {item['energy_score']['mean']:.6f} | "
            f"{item['energy_score']['sample_std']:.6f} | "
            f"{item['met_transverse_rmse']['mean']:.6f} |"
        )
    lines += [
        "",
        "The budget recommendation remains a quality-plateau review, with calibration and derived-physics checks retained as constraints.",
    ]
    md_path.write_text("\n".join(lines) + "\n")

    try:
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(7.0, 4.5))
        for variant in sorted({item["variant"] for item in aggregate}):
            points = [item for item in aggregate if item["variant"] == variant]
            points.sort(key=lambda item: item["sample_equivalent_nfe_per_event"])
            axis.plot(
                [item["sample_equivalent_nfe_per_event"] for item in points],
                [item["energy_score"]["mean"] for item in points],
                marker="o",
                label=variant,
            )
        axis.set_xlabel("Sample-equivalent NFE/event")
        axis.set_ylabel("Posterior energy score (lower is better)")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
    except ImportError:
        plot_path = None
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("results/studies"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("configs/studies/08_generator_inference.yaml")
    )
    args = parser.parse_args()
    for path in build_review(args.output_root.expanduser().resolve(), args.manifest):
        print(path)


if __name__ == "__main__":
    main()
