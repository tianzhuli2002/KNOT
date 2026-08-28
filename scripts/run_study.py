#!/usr/bin/env python3
"""Run one reproducible JER study manifest through the maintained CLI."""

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

from joint_event_reconstruction.configuration import validate_experiment
from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions
from joint_event_reconstruction.plotting import plot_comparison, plot_prediction
from joint_event_reconstruction.public_config import resolve_public_experiment
from joint_event_reconstruction.studies import PredictionSpec, StudySpec, VariantSpec, load_study
from joint_event_reconstruction.study_metrics import (
    assert_same_events,
    common_non_null_truth_assignment,
)


SUMMARY_FIELDS = (
    "study", "topology", "variant", "variant_label", "prediction",
    "prediction_label", "reporting", "seed", "events", "assignment_support",
    "assignment_accuracy", "assignment_accuracy_standard_error",
    "common_non_null_events", "common_non_null_assignment_accuracy",
    "common_non_null_assignment_standard_error", "compatibility_assignment_only_top1",
    "compatibility_combined_top1", "energy_score", "sample_equivalent_nfe_per_event",
    "prediction_seed", "prediction_file",
)


def _run_dir(root: Path, study: StudySpec, topology: str, variant: str, seed: int) -> Path:
    return root / study.name / topology / variant / f"seed_{seed}"


def _checkpoint(root: Path, study: StudySpec, topology: str, variant: str, seed: int) -> Path:
    return _run_dir(root, study, topology, variant, seed) / "train/checkpoints/best-loss.ckpt"


def _prediction_path(root, study, topology, variant, seed, prediction) -> Path:
    return _run_dir(root, study, topology, variant, seed) / "predictions" / f"{prediction}.h5"


def _evaluation_path(root, study, topology, variant, seed, prediction) -> Path:
    return _run_dir(root, study, topology, variant, seed) / "evaluation" / f"{prediction}.json"


def _selected(value: str, allowed: tuple, label: str) -> tuple:
    if value == "all":
        return allowed
    selected = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}; allowed={list(allowed)}")
    return selected


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _public_run_config(
    study: StudySpec,
    topology: str,
    variant: VariantSpec,
    seed: int,
    output_root: Path,
    global_overrides: tuple[str, ...],
    prediction: PredictionSpec | None = None,
):
    experiment = study.experiment_path(ROOT, topology)
    public = OmegaConf.load(experiment)
    public = OmegaConf.merge(public, OmegaConf.create(variant.patch))
    if global_overrides:
        public = OmegaConf.merge(public, OmegaConf.from_dotlist(list(global_overrides)))
    public = OmegaConf.merge(
        public,
        {
            "experiment": {"seed": seed},
            "training": {
                "output": str(_run_dir(output_root, study, topology, variant.name, seed) / "train")
            },
        },
    )
    if prediction is not None:
        public = OmegaConf.merge(public, OmegaConf.create(prediction.patch))
    resolved = resolve_public_experiment(public, source_path=experiment)
    validate_experiment(resolved)
    return experiment, public, resolved


def _write_yaml(path: Path, config) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(config, resolve=True))


def _execute(command: list[str], *, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _train_one(study, topology, variant, seed, *, root, global_overrides, device, force, dry_run):
    run_dir = _run_dir(root, study, topology, variant.name, seed)
    checkpoint = _checkpoint(root, study, topology, variant.name, seed)
    if checkpoint.is_file() and not force:
        print(f"SKIP train: {checkpoint}")
        return
    if force and (run_dir / "train").exists() and not dry_run:
        shutil.rmtree(run_dir / "train")

    experiment, public, resolved = _public_run_config(
        study, topology, variant, seed, root, global_overrides
    )
    public_path = run_dir / "experiment.yaml"
    if not dry_run:
        _write_yaml(public_path, public)
        _write_yaml(run_dir / "resolved.yaml", resolved)

    initial = None
    if variant.initialise_from is not None:
        initial = _checkpoint(root, study, topology, variant.initialise_from, seed)
        if not initial.is_file() and not dry_run:
            raise FileNotFoundError(
                f"Variant {variant.name!r} requires checkpoint {initial}"
            )
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "study_run.json").write_text(
            json.dumps(
                {
                    "study": study.name,
                    "study_manifest": str(study.path),
                    "topology": topology,
                    "variant": variant.name,
                    "variant_label": variant.label,
                    "reporting": variant.reporting,
                    "seed": seed,
                    "base_experiment": str(experiment),
                    "variant_patch": variant.patch,
                    "initial_checkpoint": None if initial is None else str(initial),
                    "initial_components": list(variant.initialise_components),
                    "prediction_seed": int(resolved.inference.seed),
                    "git_commit": _git_commit(),
                },
                indent=2,
                sort_keys=True,
            ) + "\n"
        )

    command = [
        sys.executable, "-m", "joint_event_reconstruction.cli.run", "train",
        "--experiment", str(public_path), "--device", device,
    ]
    if initial is not None:
        command += ["--initial-checkpoint", str(initial), "--initial-components", *variant.initialise_components]
    _execute(command, dry_run=dry_run)


def _predict_one(study, topology, variant, prediction, seed, *, root, global_overrides, device, force, dry_run):
    checkpoint = _checkpoint(root, study, topology, variant.name, seed)
    if not checkpoint.is_file() and not dry_run:
        raise FileNotFoundError(f"Prediction requires checkpoint {checkpoint}")
    target = _prediction_path(root, study, topology, variant.name, seed, prediction.name)
    if target.is_file() and not force:
        print(f"SKIP predict: {target}")
        return

    _, public, resolved = _public_run_config(
        study, topology, variant, seed, root, global_overrides, prediction
    )
    config_path = _run_dir(root, study, topology, variant.name, seed) / "predictions" / f"{prediction.name}.yaml"
    if not dry_run:
        _write_yaml(config_path, public)
        _write_yaml(config_path.with_suffix(".resolved.yaml"), resolved)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
    command = [
        sys.executable, "-m", "joint_event_reconstruction.cli.run", "predict",
        "--experiment", str(config_path), "--checkpoint", str(checkpoint),
        "--output", str(target), "--split", study.split, "--device", device,
    ]
    if prediction.recurrent_steps is not None:
        command += ["--recurrent-steps", str(prediction.recurrent_steps)]
    _execute(command, dry_run=dry_run)
    if not dry_run:
        with h5py.File(target, "r+") as output:
            output.attrs["study_name"] = study.name
            output.attrs["study_variant"] = variant.name
            output.attrs["study_prediction"] = prediction.name
            output.attrs["study_seed"] = seed


def _evaluate_one(study, topology, variant, prediction, seed, *, root, force, dry_run):
    source = _prediction_path(root, study, topology, variant.name, seed, prediction.name)
    target = _evaluation_path(root, study, topology, variant.name, seed, prediction.name)
    if target.is_file() and not force:
        print(f"SKIP evaluate: {target}")
        return
    if dry_run:
        print(f"evaluate {source} -> {target}")
        return
    metrics = evaluate_any_predictions(source)
    metrics["common_non_null_truth_assignment"] = common_non_null_truth_assignment(source)
    with h5py.File(source, "r") as handle:
        prediction_seed = int(handle.attrs.get("prediction_seed", -1))
    metrics["study"] = {
        "name": study.name,
        "topology": topology,
        "variant": variant.name,
        "variant_label": variant.label,
        "prediction": prediction.name,
        "prediction_label": prediction.label,
        "reporting": variant.reporting,
        "seed": seed,
        "prediction_seed": prediction_seed,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


def _plot_one(study, topology, variant, prediction, seed, *, root, formats, dry_run):
    source = _prediction_path(root, study, topology, variant.name, seed, prediction.name)
    target = _run_dir(root, study, topology, variant.name, seed) / "plots" / prediction.name
    print(f"plot {source} -> {target}")
    if not dry_run:
        plot_prediction(source, target, formats=formats)


def _metric_row(study, topology, variant, prediction, seed, metrics):
    if metrics["assignment_formulation"] == "discriminative":
        assignment = metrics["visible"]["complete"]
        accuracy = assignment["top1_efficiency"]
        accuracy_error = assignment["top1_efficiency_standard_error"]
    else:
        assignment = metrics["assignment"]
        accuracy = assignment["empirical_mode_accuracy"]
        accuracy_error = assignment["empirical_mode_accuracy_standard_error"]

    invisible = metrics.get("invisible", {})
    energy = None
    if metrics["model_mode"] == "factorized":
        posterior = invisible.get("truncated_assignment_weighted_mixture") or invisible.get("top1_assignment_conditioned")
        if posterior:
            energy = posterior.get("energy_score_joint")
    else:
        energy = invisible.get("energy_score_joint")
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
        "assignment_accuracy": accuracy,
        "assignment_accuracy_standard_error": accuracy_error,
        "common_non_null_events": common.get("events"),
        "common_non_null_assignment_accuracy": common.get("accuracy"),
        "common_non_null_assignment_standard_error": common.get("standard_error"),
        "compatibility_assignment_only_top1": compatibility.get("assignment_only_top1"),
        "compatibility_combined_top1": compatibility.get("combined_top1"),
        "energy_score": energy,
        "sample_equivalent_nfe_per_event": metrics.get("sample_equivalent_nfe_per_event") or metrics.get("network_calls_per_event"),
        "prediction_seed": metrics.get("study", {}).get("prediction_seed"),
        "prediction_file": metrics["prediction_file"],
    }


def _summary(study, topologies, variants, seeds, *, root, dry_run):
    rows = []
    for topology in topologies:
        for variant in variants:
            if variant.reporting == "setup":
                continue
            for seed in seeds:
                for prediction in variant.predictions:
                    path = _evaluation_path(root, study, topology, variant.name, seed, prediction.name)
                    if not path.is_file():
                        if dry_run:
                            continue
                        raise FileNotFoundError(path)
                    rows.append(_metric_row(study, topology, variant, prediction, seed, json.loads(path.read_text())))
    destination = root / study.name
    print(f"summarize {len(rows)} results -> {destination}")
    if dry_run:
        return
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (destination / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    groups = defaultdict(list)
    for row in rows:
        groups[(row["topology"], row["variant"], row["prediction"], row["reporting"], row["assignment_support"])].append(row)
    aggregate = []
    for key, values in sorted(groups.items()):
        item = {
            "topology": key[0], "variant": key[1], "prediction": key[2],
            "reporting": key[3], "assignment_support": key[4], "seeds": len(values),
        }
        for metric in ("assignment_accuracy", "common_non_null_assignment_accuracy", "energy_score", "sample_equivalent_nfe_per_event"):
            numbers = [float(value[metric]) for value in values if value[metric] is not None]
            item[f"{metric}_mean"] = statistics.mean(numbers) if numbers else None
            item[f"{metric}_seed_std"] = statistics.stdev(numbers) if len(numbers) > 1 else None
        aggregate.append(item)
    (destination / "summary_aggregate.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")


def _compare(study, topology, variants, seed, *, root, formats, include_diagnostics, dry_run):
    paths, labels = [], []
    for variant in variants:
        if variant.reporting == "setup" or (variant.reporting != "primary" and not include_diagnostics):
            continue
        for prediction in variant.predictions:
            path = _prediction_path(root, study, topology, variant.name, seed, prediction.name)
            if path.is_file() or dry_run:
                paths.append(path)
                labels.append(f"{variant.label}: {prediction.label}")
    if len(paths) < 2:
        return
    if not dry_run:
        assert_same_events(paths)
        plot_comparison(paths, root / study.name / topology / "comparisons" / f"seed_{seed}", labels=labels, formats=formats)


def _plan(study, topologies, variants, seeds, root):
    print(f"study: {study.name}\ntopologies: {', '.join(topologies)}\nseeds: {', '.join(map(str, seeds))}\noutput: {root / study.name}")
    for variant in variants:
        print(f"  {variant.name:24s} {variant.reporting:10s} init={variant.initialise_from or '-':20s} predictions={len(variant.predictions)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a validated JER study manifest")
    parser.add_argument("study", type=Path)
    parser.add_argument("stage", choices=("plan", "train", "predict", "evaluate", "plot", "compare", "summary", "all"))
    parser.add_argument("--output-root", type=Path, default=Path("results/studies"))
    parser.add_argument("--topologies", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--seeds", default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--set", dest="global_overrides", action="append", default=[])
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf"))
    parser.add_argument("--include-diagnostics", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    study = load_study(args.study)
    topologies = _selected(args.topologies, study.topologies, "topologies")
    requested = _selected(args.variants, tuple(v.name for v in study.variants), "variants")
    variants = study.dependency_order() if args.variants == "all" else study.dependency_closure(requested)
    seeds = study.seeds if args.seeds == "all" else tuple(int(v) for v in args.seeds.split(",") if v)
    unknown_seeds = sorted(set(seeds) - set(study.seeds))
    if unknown_seeds:
        raise ValueError(f"Seeds {unknown_seeds} are not declared by the study")
    root = args.output_root.expanduser().resolve()
    global_overrides = tuple(args.global_overrides)
    formats = tuple(args.formats)
    _plan(study, topologies, variants, seeds, root)

    if args.stage == "plan":
        for topology in topologies:
            for variant in variants:
                for seed in seeds:
                    _public_run_config(study, topology, variant, seed, root, global_overrides)
                    for prediction in variant.predictions:
                        _public_run_config(study, topology, variant, seed, root, global_overrides, prediction)
        print("All selected compact training and prediction configurations validate.")
        return

    stages = ("train", "predict", "evaluate", "plot", "compare", "summary") if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "train":
            for topology in topologies:
                for variant in variants:
                    for seed in seeds:
                        _train_one(study, topology, variant, seed, root=root, global_overrides=global_overrides, device=args.device, force=args.force, dry_run=args.dry_run)
        elif stage in {"predict", "evaluate", "plot"}:
            for topology in topologies:
                for variant in variants:
                    if variant.reporting == "setup":
                        continue
                    for seed in seeds:
                        for prediction in variant.predictions:
                            if stage == "predict":
                                _predict_one(study, topology, variant, prediction, seed, root=root, global_overrides=global_overrides, device=args.device, force=args.force, dry_run=args.dry_run)
                            elif stage == "evaluate":
                                _evaluate_one(study, topology, variant, prediction, seed, root=root, force=args.force, dry_run=args.dry_run)
                            else:
                                _plot_one(study, topology, variant, prediction, seed, root=root, formats=formats, dry_run=args.dry_run)
        elif stage == "compare":
            for topology in topologies:
                for seed in seeds:
                    _compare(study, topology, variants, seed, root=root, formats=formats, include_diagnostics=args.include_diagnostics, dry_run=args.dry_run)
        elif stage == "summary":
            _summary(study, topologies, variants, seeds, root=root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
