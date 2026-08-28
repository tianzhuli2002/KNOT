#!/usr/bin/env python3
"""Plan and run reproducible JER study manifests using the maintained CLI.

This script owns orchestration only. Model construction, training, prediction, evaluation
and plotting remain in the maintained package APIs.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import h5py
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions
from joint_event_reconstruction.plotting import plot_comparison, plot_prediction
from joint_event_reconstruction.studies import PredictionSpec, StudySpec, VariantSpec, load_study
from joint_event_reconstruction.study_metrics import (
    assert_same_events,
    common_non_null_truth_assignment,
)


INFERENCE_SEED = 12345
SUMMARY_FIELDS = (
    "study",
    "topology",
    "variant",
    "variant_label",
    "prediction",
    "prediction_label",
    "reporting",
    "seed",
    "events",
    "assignment_support",
    "assignment_accuracy",
    "assignment_accuracy_standard_error",
    "common_non_null_events",
    "common_non_null_assignment_accuracy",
    "common_non_null_assignment_standard_error",
    "compatibility_assignment_only_top1",
    "compatibility_combined_top1",
    "energy_score",
    "sample_equivalent_nfe_per_event",
    "prediction_file",
)


def _run_dir(
    output_root: Path, study: StudySpec, topology: str, variant: str, seed: int
) -> Path:
    return output_root / study.name / topology / variant / f"seed_{seed}"


def _checkpoint(
    output_root: Path, study: StudySpec, topology: str, variant: str, seed: int
) -> Path:
    return (
        _run_dir(output_root, study, topology, variant, seed)
        / "train/checkpoints/best-loss.ckpt"
    )


def _prediction_path(
    output_root: Path,
    study: StudySpec,
    topology: str,
    variant: str,
    seed: int,
    prediction: str,
) -> Path:
    return (
        _run_dir(output_root, study, topology, variant, seed)
        / "predictions"
        / f"{prediction}.h5"
    )


def _evaluation_path(
    output_root: Path,
    study: StudySpec,
    topology: str,
    variant: str,
    seed: int,
    prediction: str,
) -> Path:
    return (
        _run_dir(output_root, study, topology, variant, seed)
        / "evaluation"
        / f"{prediction}.json"
    )


def _selected(value: str, allowed: tuple[str, ...], label: str) -> tuple[str, ...]:
    if value == "all":
        return allowed
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; allowed={list(allowed)}")
    return selected


def _variant_overrides(
    variant: VariantSpec,
    *,
    seed: int,
    train_output: Path,
    global_overrides: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        *variant.train_overrides,
        *global_overrides,
        f"experiment.seed={seed}",
        f"training.output={train_output}",
    )


def _resolved(
    study: StudySpec,
    topology: str,
    variant: VariantSpec,
    seed: int,
    output_root: Path,
    global_overrides: tuple[str, ...],
    prediction: PredictionSpec | None = None,
):
    run_dir = _run_dir(output_root, study, topology, variant.name, seed)
    experiment = study.experiment_path(ROOT, topology)
    overrides = list(
        _variant_overrides(
            variant,
            seed=seed,
            train_output=run_dir / "train",
            global_overrides=global_overrides,
        )
    )
    if prediction is not None:
        overrides.extend(prediction.overrides)
    return experiment, load_experiment(experiment, overrides), tuple(overrides)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _write_run_metadata(
    run_dir: Path,
    *,
    study: StudySpec,
    topology: str,
    variant: VariantSpec,
    seed: int,
    experiment: Path,
    overrides: tuple[str, ...],
    initial_checkpoint: Path | None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study": study.name,
        "study_manifest": str(study.path),
        "description": study.description,
        "topology": topology,
        "variant": variant.name,
        "variant_label": variant.label,
        "reporting": variant.reporting,
        "seed": seed,
        "experiment": str(experiment),
        "train_overrides": list(overrides),
        "initial_checkpoint": (
            None if initial_checkpoint is None else str(initial_checkpoint)
        ),
        "initial_components": list(variant.initialise_components),
        "inference_seed": INFERENCE_SEED,
        "git_commit": _git_commit(),
    }
    (run_dir / "study_run.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def _execute(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _train_one(
    study: StudySpec,
    topology: str,
    variant: VariantSpec,
    seed: int,
    *,
    output_root: Path,
    global_overrides: tuple[str, ...],
    device: str,
    force: bool,
    dry_run: bool,
) -> None:
    run_dir = _run_dir(output_root, study, topology, variant.name, seed)
    checkpoint = _checkpoint(output_root, study, topology, variant.name, seed)
    if checkpoint.is_file() and not force:
        print(f"SKIP train: {checkpoint}")
        return
    if force and run_dir.joinpath("train").exists() and not dry_run:
        shutil.rmtree(run_dir / "train")

    experiment, config, overrides = _resolved(
        study,
        topology,
        variant,
        seed,
        output_root,
        global_overrides,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        (run_dir / "resolved_train.yaml").write_text(
            OmegaConf.to_yaml(config, resolve=True)
        )

    initial = None
    if variant.initialise_from is not None:
        initial = _checkpoint(
            output_root, study, topology, variant.initialise_from, seed
        )
        if not initial.is_file() and not dry_run:
            raise FileNotFoundError(
                f"Variant {variant.name!r} requires initial checkpoint {initial}"
            )
    if not dry_run:
        _write_run_metadata(
            run_dir,
            study=study,
            topology=topology,
            variant=variant,
            seed=seed,
            experiment=experiment,
            overrides=overrides,
            initial_checkpoint=initial,
        )

    command = [
        sys.executable,
        "-m",
        "joint_event_reconstruction.cli.run",
        "train",
        "--experiment",
        str(experiment),
        "--device",
        device,
    ]
    for override in overrides:
        command += ["--set", override]
    if initial is not None:
        command += ["--initial-checkpoint", str(initial), "--initial-components"]
        command += list(variant.initialise_components)
    _execute(command, dry_run=dry_run)


def _predict_one(
    study: StudySpec,
    topology: str,
    variant: VariantSpec,
    prediction: PredictionSpec,
    seed: int,
    *,
    output_root: Path,
    global_overrides: tuple[str, ...],
    device: str,
    force: bool,
    dry_run: bool,
) -> None:
    checkpoint = _checkpoint(output_root, study, topology, variant.name, seed)
    if not checkpoint.is_file() and not dry_run:
        raise FileNotFoundError(f"Prediction requires checkpoint {checkpoint}")
    target = _prediction_path(
        output_root, study, topology, variant.name, seed, prediction.name
    )
    if target.is_file() and not force:
        print(f"SKIP predict: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)

    experiment, config, overrides = _resolved(
        study,
        topology,
        variant,
        seed,
        output_root,
        global_overrides,
        prediction,
    )
    prediction_config = (
        _run_dir(output_root, study, topology, variant.name, seed)
        / "resolved_predictions"
        / f"{prediction.name}.yaml"
    )
    if not dry_run:
        prediction_config.parent.mkdir(parents=True, exist_ok=True)
        prediction_config.write_text(OmegaConf.to_yaml(config, resolve=True))
        if target.exists():
            target.unlink()

    command = [
        sys.executable,
        "-m",
        "joint_event_reconstruction.cli.run",
        "predict",
        "--experiment",
        str(experiment),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(target),
        "--split",
        study.split,
        "--device",
        device,
    ]
    for override in overrides:
        command += ["--set", override]
    if prediction.recurrent_steps is not None:
        command += ["--recurrent-steps", str(prediction.recurrent_steps)]
    _execute(command, dry_run=dry_run)

    if not dry_run:
        with h5py.File(target, "r+") as output:
            output.attrs["study_name"] = study.name
            output.attrs["study_variant"] = variant.name
            output.attrs["study_prediction"] = prediction.name
            output.attrs["study_seed"] = seed
            output.attrs["study_inference_seed"] = INFERENCE_SEED


def _evaluate_one(
    study: StudySpec,
    topology: str,
    variant: VariantSpec,
    prediction: PredictionSpec,
    seed: int,
    *,
    output_root: Path,
    force: bool,
    dry_run: bool,
) -> None:
    source = _prediction_path(
        output_root, study, topology, variant.name, seed, prediction.name
    )
    target = _evaluation_path(
        output_root, study, topology, variant.name, seed, prediction.name
    )
    if target.is_file() and not force:
        print(f"SKIP evaluate: {target}")
        return
    if not source.is_file() and not dry_run:
        raise FileNotFoundError(f"Evaluation requires prediction {source}")
    print(f"evaluate {source} -> {target}")
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_any_predictions(source)
    metrics["common_non_null_truth_assignment"] = common_non_null_truth_assignment(
        source
    )
    metrics["study"] = {
        "name": study.name,
        "topology": topology,
        "variant": variant.name,
        "variant_label": variant.label,
        "prediction": prediction.name,
        "prediction_label": prediction.label,
        "reporting": variant.reporting,
        "seed": seed,
        "inference_seed": INFERENCE_SEED,
    }
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _plot_one(
    study: StudySpec,
    topology: str,
    variant: VariantSpec,
    prediction: PredictionSpec,
    seed: int,
    *,
    output_root: Path,
    formats: tuple[str, ...],
    dry_run: bool,
) -> None:
    source = _prediction_path(
        output_root, study, topology, variant.name, seed, prediction.name
    )
    target = (
        _run_dir(output_root, study, topology, variant.name, seed)
        / "plots"
        / prediction.name
    )
    print(f"plot {source} -> {target}")
    if dry_run:
        return
    if not source.is_file():
        raise FileNotFoundError(f"Plotting requires prediction {source}")
    plot_prediction(source, target, formats=formats)


def _metric_row(
    study: StudySpec,
    topology: str,
    variant: VariantSpec,
    prediction: PredictionSpec,
    seed: int,
    metrics: dict,
) -> dict:
    if metrics["assignment_formulation"] == "discriminative":
        complete = metrics["visible"]["complete"]
        assignment_accuracy = complete["top1_efficiency"]
        assignment_error = complete["top1_efficiency_standard_error"]
    else:
        assignment = metrics["assignment"]
        assignment_accuracy = assignment["empirical_mode_accuracy"]
        assignment_error = assignment["empirical_mode_accuracy_standard_error"]

    invisible = metrics.get("invisible")
    energy = None
    if invisible is not None:
        if metrics["model_mode"] == "factorized":
            mixture = invisible.get("truncated_assignment_weighted_mixture")
            if mixture is not None:
                energy = mixture.get("energy_score_joint")
            elif "top1_assignment_conditioned" in invisible:
                energy = invisible["top1_assignment_conditioned"].get(
                    "energy_score_joint"
                )
        else:
            energy = invisible.get("energy_score_joint")

    nfe = metrics.get("sample_equivalent_nfe_per_event")
    if nfe is None and metrics["assignment_formulation"] == "flow_matching":
        nfe = metrics.get("network_calls_per_event")
    common = metrics.get("common_non_null_truth_assignment", {})
    compatibility = metrics.get("compatibility", {})
    return {
        "study": study.name,
        "topology": topology,
        "variant": variant.name,
        "variant_label": variant.label,
        "prediction": prediction.name,
        "prediction_label": prediction.label,
        "reporting": variant.reporting,
        "seed": seed,
        "events": metrics["events"],
        "assignment_support": metrics["assignment_support"]["name"],
        "assignment_accuracy": assignment_accuracy,
        "assignment_accuracy_standard_error": assignment_error,
        "common_non_null_events": common.get("events"),
        "common_non_null_assignment_accuracy": common.get("accuracy"),
        "common_non_null_assignment_standard_error": common.get("standard_error"),
        "compatibility_assignment_only_top1": compatibility.get(
            "assignment_only_top1"
        ),
        "compatibility_combined_top1": compatibility.get("combined_top1"),
        "energy_score": energy,
        "sample_equivalent_nfe_per_event": nfe,
        "prediction_file": metrics["prediction_file"],
    }


def _summarize(
    study: StudySpec,
    topologies: tuple[str, ...],
    variants: tuple[VariantSpec, ...],
    seeds: tuple[int, ...],
    *,
    output_root: Path,
    dry_run: bool,
) -> None:
    rows = []
    for topology in topologies:
        for variant in variants:
            if variant.reporting == "setup":
                continue
            for seed in seeds:
                for prediction in variant.predictions:
                    path = _evaluation_path(
                        output_root,
                        study,
                        topology,
                        variant.name,
                        seed,
                        prediction.name,
                    )
                    if not path.is_file():
                        if dry_run:
                            continue
                        raise FileNotFoundError(f"Study summary requires {path}")
                    metrics = json.loads(path.read_text())
                    rows.append(
                        _metric_row(
                            study, topology, variant, prediction, seed, metrics
                        )
                    )

    root = output_root / study.name
    print(f"summarize {len(rows)} prediction results -> {root}")
    if dry_run:
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (root / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["topology"],
            row["variant"],
            row["prediction"],
            row["reporting"],
            row["assignment_support"],
        )
        groups[key].append(row)
    aggregate = []
    aggregate_metrics = (
        "assignment_accuracy",
        "common_non_null_assignment_accuracy",
        "compatibility_assignment_only_top1",
        "compatibility_combined_top1",
        "energy_score",
        "sample_equivalent_nfe_per_event",
    )
    for key, values in sorted(groups.items()):
        item = {
            "topology": key[0],
            "variant": key[1],
            "prediction": key[2],
            "reporting": key[3],
            "assignment_support": key[4],
            "seeds": len(values),
        }
        for metric in aggregate_metrics:
            numbers = [
                float(value[metric])
                for value in values
                if value[metric] is not None
            ]
            item[f"{metric}_mean"] = statistics.mean(numbers) if numbers else None
            item[f"{metric}_seed_std"] = (
                statistics.stdev(numbers) if len(numbers) > 1 else None
            )
        aggregate.append(item)
    (root / "summary_aggregate.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n"
    )


def _compare(
    study: StudySpec,
    topology: str,
    variants: tuple[VariantSpec, ...],
    seed: int,
    *,
    output_root: Path,
    formats: tuple[str, ...],
    include_diagnostics: bool,
    dry_run: bool,
) -> None:
    paths = []
    labels = []
    for variant in variants:
        if variant.reporting == "setup" or (
            variant.reporting != "primary" and not include_diagnostics
        ):
            continue
        for prediction in variant.predictions:
            path = _prediction_path(
                output_root,
                study,
                topology,
                variant.name,
                seed,
                prediction.name,
            )
            if path.is_file() or dry_run:
                paths.append(path)
                labels.append(f"{variant.label}: {prediction.label}")
    if len(paths) < 2:
        print(
            f"SKIP comparison {topology} seed={seed}: fewer than two prediction files"
        )
        return
    target = output_root / study.name / topology / "comparisons" / f"seed_{seed}"
    print(f"compare {len(paths)} predictions -> {target}")
    if not dry_run:
        assert_same_events(paths)
        plot_comparison(paths, target, labels=labels, formats=formats)


def _iter_variants(
    study: StudySpec, selected: tuple[str, ...]
) -> tuple[VariantSpec, ...]:
    allowed = set(selected)
    return tuple(
        variant for variant in study.dependency_order() if variant.name in allowed
    )


def _plan(
    study: StudySpec,
    topologies: tuple[str, ...],
    variants: tuple[VariantSpec, ...],
    seeds: tuple[int, ...],
    output_root: Path,
    global_overrides: tuple[str, ...],
) -> None:
    print(f"study: {study.name}")
    print(f"description: {study.description}")
    print(f"topologies: {', '.join(topologies)}")
    print(f"seeds: {', '.join(map(str, seeds))}")
    print(f"output: {output_root / study.name}")
    print("variants:")
    for variant in variants:
        dependency = variant.initialise_from or "-"
        print(
            f"  {variant.name:24s} {variant.reporting:10s} "
            f"init={dependency:20s} predictions={len(variant.predictions)}"
        )
    for topology in topologies:
        for variant in variants:
            for seed in seeds:
                _resolved(
                    study,
                    topology,
                    variant,
                    seed,
                    output_root,
                    global_overrides,
                )
                for prediction in variant.predictions:
                    _resolved(
                        study,
                        topology,
                        variant,
                        seed,
                        output_root,
                        global_overrides,
                        prediction,
                    )
    print("All selected training and prediction configurations validate.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a validated JER study manifest")
    parser.add_argument("study", type=Path)
    parser.add_argument(
        "stage",
        choices=(
            "plan",
            "train",
            "predict",
            "evaluate",
            "plot",
            "compare",
            "summary",
            "all",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results/studies"))
    parser.add_argument("--topologies", default="all", help="all or comma-separated")
    parser.add_argument("--variants", default="all", help="all or comma-separated")
    parser.add_argument("--seeds", default="all", help="all or comma-separated integers")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--set", dest="global_overrides", action="append", default=[])
    parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf")
    )
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    study = load_study(args.study)
    topologies = _selected(args.topologies, study.topologies, "topologies")
    variant_names = _selected(
        args.variants, tuple(variant.name for variant in study.variants), "variants"
    )
    variants = _iter_variants(study, variant_names)
    if args.seeds == "all":
        seeds = study.seeds
    else:
        seeds = tuple(
            int(value.strip()) for value in args.seeds.split(",") if value.strip()
        )
        unknown = sorted(set(seeds) - set(study.seeds))
        if unknown:
            raise ValueError(f"Seeds {unknown} are not declared by the study manifest")
    output_root = args.output_root.expanduser().resolve()
    global_overrides = tuple(args.global_overrides)
    formats = tuple(args.formats)

    _plan(
        study,
        topologies,
        variants,
        seeds,
        output_root,
        global_overrides,
    )
    if args.stage == "plan":
        return

    stages = (
        ("train", "predict", "evaluate", "plot", "compare", "summary")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        if stage == "train":
            for topology in topologies:
                for variant in variants:
                    for seed in seeds:
                        _train_one(
                            study,
                            topology,
                            variant,
                            seed,
                            output_root=output_root,
                            global_overrides=global_overrides,
                            device=args.device,
                            force=args.force,
                            dry_run=args.dry_run,
                        )
        elif stage in {"predict", "evaluate", "plot"}:
            for topology in topologies:
                for variant in variants:
                    if variant.reporting == "setup":
                        continue
                    for seed in seeds:
                        for prediction in variant.predictions:
                            if stage == "predict":
                                _predict_one(
                                    study,
                                    topology,
                                    variant,
                                    prediction,
                                    seed,
                                    output_root=output_root,
                                    global_overrides=global_overrides,
                                    device=args.device,
                                    force=args.force,
                                    dry_run=args.dry_run,
                                )
                            elif stage == "evaluate":
                                _evaluate_one(
                                    study,
                                    topology,
                                    variant,
                                    prediction,
                                    seed,
                                    output_root=output_root,
                                    force=args.force,
                                    dry_run=args.dry_run,
                                )
                            else:
                                _plot_one(
                                    study,
                                    topology,
                                    variant,
                                    prediction,
                                    seed,
                                    output_root=output_root,
                                    formats=formats,
                                    dry_run=args.dry_run,
                                )
        elif stage == "compare":
            for topology in topologies:
                for seed in seeds:
                    _compare(
                        study,
                        topology,
                        variants,
                        seed,
                        output_root=output_root,
                        formats=formats,
                        include_diagnostics=args.include_diagnostics,
                        dry_run=args.dry_run,
                    )
        elif stage == "summary":
            _summarize(
                study,
                topologies,
                variants,
                seeds,
                output_root=output_root,
                dry_run=args.dry_run,
            )
        else:
            raise RuntimeError(f"Unhandled study stage {stage!r}")


if __name__ == "__main__":
    main()
