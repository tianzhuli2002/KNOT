#!/usr/bin/env python3
"""Summarize matched KNOT C/J seed evaluations on one event population."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import analyse_knot_production_posterior as production
import h5py
import numpy as np

EVALUATOR_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = Path("/net/scratch/w00238tl/knot")
EVALDEV_ROOT = Path("/net/scratch/w00238tl/knot")
DEFAULT_EVALUATION_ROOT = (
    EVALDEV_ROOT / "results/experiments/knot/matched-seed-replication-v1/validation-2000"
)
DEFAULT_OUTPUT = EVALUATOR_ROOT / "results/experiments/knot/multiseed-v1"
SEEDS = (42011, 42012, 42013)
VARIANTS = ("C", "J")
BOOTSTRAP_REPLICATES = 10_000
METRIC_SHUFFLE_SEED = 42013

POSTERIOR_FIELDS = (
    "EVENT_ID",
    "invisible",
    "parents",
    "TRUTH/neutrinos",
    "TRUTH/parents",
    "TRUTH/assignment_state",
    "TRUTH/assignment_support",
    "diagnostics/assignment_MAP_state",
    "diagnostics/assignment_MAP_node_correct",
    "diagnostics/assignment_truth_NLL",
    "measured/leptons",
    "measured_b/sample_four_vectors",
    "measured_b/truth_four_vectors",
)
TRUTH_FIELDS = (
    "TRUTH/neutrinos",
    "TRUTH/parents",
    "TRUTH/assignment_state",
    "TRUTH/assignment_support",
    "measured/leptons",
    "measured_b/truth_four_vectors",
)
TRAINING_SETTINGS = (
    "batch_size",
    "epochs",
    "precision",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "scheduler",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded(value):
    return value.decode("utf-8") if isinstance(value, bytes) else value


def run_manifest_path(seed: int, variant: str, training_root: Path) -> Path:
    if seed == 42011:
        return FROZEN_ROOT / "results/experiments/knot/matched-production-v1" / variant / "run_manifest.json"
    return training_root / f"{variant}_seed{seed}" / "run_manifest.json"


def load_one(seed: int, variant: str, evaluation_root: Path, training_root: Path):
    run_path = run_manifest_path(seed, variant, training_root)
    evaluation_dir = evaluation_root / f"{variant}_seed{seed}"
    evaluation_path = evaluation_dir / "evaluation_manifest.json"
    metrics_path = evaluation_dir / "metrics.json"
    posterior_path = evaluation_dir / "posterior.h5"
    for path in (run_path, evaluation_path, metrics_path, posterior_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing seed-replication input: {path}")

    run = json.loads(run_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    if run.get("training_status") != "completed" or run.get("sealed_test_used") is not False:
        raise ValueError(f"Training manifest is incomplete or used the sealed test: {run_path}")
    if run.get("variant") != variant or run.get("seed") != seed:
        raise ValueError(f"Training variant/seed mismatch in {run_path}")
    if evaluation.get("variant") != variant or evaluation.get("sealed_test_used") is not False:
        raise ValueError(f"Evaluation variant/test status mismatch in {evaluation_path}")
    if evaluation.get("events") != 2000 or evaluation.get("samples_per_event") != 32:
        raise ValueError(f"Unexpected event/sample count in {evaluation_path}")
    if evaluation.get("evaluation_seed") != 42011:
        raise ValueError(f"Unexpected posterior sampling seed in {evaluation_path}")
    if evaluation.get("inference_steps") != 8 or evaluation.get("inference_solver") != "heun":
        raise ValueError(f"Unexpected posterior solver in {evaluation_path}")
    if evaluation.get("inference_override") is not False:
        raise ValueError(f"Evaluation overrides the frozen inference settings: {evaluation_path}")
    if metrics.get("variant") != variant or metrics.get("events") != 2000:
        raise ValueError(f"Metric variant/event count mismatch in {metrics_path}")
    expected_semantics = {"C": "ancestral", "J": "joint"}[variant]
    if metrics.get("posterior_semantics") != expected_semantics:
        raise ValueError(f"Unexpected posterior semantics in {metrics_path}")
    if metrics.get("samples_per_event") != 32 or metrics.get("sealed_test_used") is not False:
        raise ValueError(f"Metric sample/test status mismatch in {metrics_path}")
    if metrics.get("inference") != {"steps": 8, "solver": "heun"}:
        raise ValueError(f"Metric solver differs from the frozen inference setting in {metrics_path}")
    if evaluation.get("training_source_commit") != run.get("source_commit"):
        raise ValueError(f"Evaluation and training source commits disagree for {variant} seed {seed}")

    with h5py.File(posterior_path, "r") as source:
        missing = [field for field in POSTERIOR_FIELDS if field not in source]
        if missing:
            raise ValueError(f"{posterior_path} lacks required posterior fields: {missing}")
        values = {field: source[field][:] for field in POSTERIOR_FIELDS}
        attrs = {key: decoded(value) for key, value in source.attrs.items()}
    if attrs.get("sealed_test_used") not in (False, 0):
        raise ValueError(f"Posterior is not marked sealed_test_used=false: {posterior_path}")
    if attrs.get("factorisation") != variant:
        raise ValueError(f"Posterior factorisation does not match {variant}: {posterior_path}")
    if attrs.get("posterior_semantics") != expected_semantics:
        raise ValueError(f"Posterior semantics do not match {variant}: {posterior_path}")
    for field, expected in (
        ("parent_roles", ["top", "antitop"]),
        ("invisible_roles", ["top.nu", "antitop.nu"]),
        ("assignment_nodes", ["antitop", "top"]),
    ):
        actual = attrs.get(field)
        if not isinstance(actual, str) or json.loads(actual) != expected:
            raise ValueError(f"Unexpected KNOT role ordering for {field} in {posterior_path}")
    if values["invisible"].shape != (2000, 32, 2, 4) or values["parents"].shape != (2000, 32, 2, 4):
        raise ValueError(f"Unexpected posterior vector dimensions: {posterior_path}")
    if values["TRUTH/neutrinos"].shape != (2000, 2, 4) or values["TRUTH/parents"].shape != (2000, 2, 4):
        raise ValueError(f"Unexpected truth vector dimensions: {posterior_path}")
    event_ids = values["EVENT_ID"]
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError(f"Duplicate EVENT_ID in {posterior_path}")
    event_hash = production.sha256_ids(event_ids)
    if event_hash != evaluation.get("validation_event_id_sha256"):
        raise ValueError(f"Ordered EVENT_ID hash mismatch in {posterior_path}")
    if event_hash != metrics.get("event_id_sha256_ordered_little_endian_int64"):
        raise ValueError(f"Metric EVENT_ID hash mismatch in {metrics_path}")
    for field in POSTERIOR_FIELDS:
        value = values[field]
        if np.issubdtype(value.dtype, np.floating):
            if field == "diagnostics/assignment_truth_NLL":
                support = values["TRUTH/assignment_support"].astype(bool)
                finite = np.isfinite(value[support])
            else:
                finite = np.isfinite(value)
            if not finite.all():
                raise ValueError(f"Non-finite {field} values in {posterior_path}")
    values["attrs"] = {
        "assignment_nodes": attrs.get("assignment_nodes"),
    }
    posterior = values

    truth_n = values["TRUTH/neutrinos"]
    truth_p = values["TRUTH/parents"]
    pack = production.metric_pack(posterior, metrics, truth_n, truth_p, METRIC_SHUFFLE_SEED)
    arrays = pack["diagnostic_arrays"]
    mtt_draws = arrays["draws"]["mttbar"]
    mtt_truth = arrays["truth"]["mttbar"]
    mtt_low, mtt_high = np.quantile(mtt_draws, [0.16, 0.84], axis=1)
    event_values = {
        "mtt_squared_error": np.square(arrays["points"]["mttbar"] - mtt_truth),
        "mtt_covered_68": ((mtt_truth >= mtt_low) & (mtt_truth <= mtt_high)).astype(np.float64),
        "assignment_correct": arrays["map_correct"].astype(np.float64),
        "assignment_support": arrays["support"].astype(bool),
        "joint_energy": arrays["joint"]["coherent_energy_score_by_event"],
        "joint_shuffle_penalty": arrays["joint"]["shuffle_minus_coherent_by_event"],
        "decay_pt_squared": np.square(
            arrays["decay"]["rmse_by_event"]["coherent"]["delta_pt"]
        ),
    }
    estimates = {
        "assignment_accuracy_supported": pack["assignment"]["complete_MAP_accuracy_supported"],
        "mtt_RMSE_GeV": pack["reconstruction"]["mttbar"]["RMSE"],
        "joint_energy_score_GeV": pack["joint"]["coherent_energy_score_GeV"],
        "N_P_shuffle_penalty_GeV": pack["joint"]["shuffle_minus_coherent_GeV"],
        "decay_chain_delta_pt_RMSE_GeV": pack["decay_chain"]["coherent"]["delta_pt"]["RMSE"],
        "mtt_68_coverage": pack["posterior_calibration"]["mttbar"]["coverage"]["0.68"],
    }
    record = {
        "seed": seed,
        "variant": variant,
        "estimates": estimates,
        "events": len(event_ids),
        "supported_assignment_events": int(event_values["assignment_support"].sum()),
        "ordered_event_id_sha256": event_hash,
        "training": run,
        "evaluation": evaluation,
        "paths": {
            "run_manifest": str(run_path),
            "evaluation_manifest": str(evaluation_path),
            "metrics": str(metrics_path),
            "posterior": str(posterior_path),
        },
        "input_hashes": {
            "run_manifest": file_sha256(run_path),
            "evaluation_manifest": file_sha256(evaluation_path),
            "metrics": file_sha256(metrics_path),
            "posterior": file_sha256(posterior_path),
        },
    }
    return record, event_ids, {name: np.asarray(value) for name, value in event_values.items()}


def pair_eligibility(c_record, j_record):
    c, j = c_record["training"], j_record["training"]
    reasons = []
    for field in ("source_commit", "run_script_sha256"):
        if c.get(field) != j.get(field):
            reasons.append(f"training {field} differs")
    if c.get("seed") != j.get("seed") or c.get("variant_initialization_seed") != j.get("variant_initialization_seed"):
        reasons.append("training or variant-initialization seeds differ")
    if c.get("membership") != j.get("membership"):
        reasons.append("train/validation EVENT_ID membership differs")
    if c.get("split_metadata", {}).get("source_contract") != j.get("split_metadata", {}).get("source_contract"):
        reasons.append("input source contract differs")
    if c.get("split_metadata", {}).get("event_id_hashes") != j.get("split_metadata", {}).get("event_id_hashes"):
        reasons.append("train/validation split EVENT_ID hashes differ")
    if c.get("initialization", {}).get("shared_component_sha256") != j.get("initialization", {}).get("shared_component_sha256"):
        reasons.append("shared model initialization differs")
    for field in TRAINING_SETTINGS:
        if c.get(field) != j.get(field):
            reasons.append(f"training setting {field} differs")
    if c.get("optimizer_execution") != j.get("optimizer_execution"):
        reasons.append("optimizer execution details differ")
    if c.get("source_worktree_dirty") != j.get("source_worktree_dirty"):
        reasons.append("source worktree dirty status differs")
    for field in ("evaluation_script_sha256", "evaluation_source_commit", "validation_event_id_sha256"):
        if c_record["evaluation"].get(field) != j_record["evaluation"].get(field):
            reasons.append(f"evaluation {field} differs")
    if c_record["ordered_event_id_sha256"] != j_record["ordered_event_id_sha256"]:
        reasons.append("ordered evaluation EVENT_ID hash differs")
    return {"status": "strict_matched_pair" if not reasons else "provenance_limited", "reasons": reasons}


def paired_bootstrap(c_values, j_values, seed: int):
    n_events = len(c_values["mtt_squared_error"])
    if len(j_values["mtt_squared_error"]) != n_events:
        raise ValueError("Paired C/J event populations have different lengths")
    support_c = c_values["assignment_support"]
    support_j = j_values["assignment_support"]
    if not np.array_equal(support_c, support_j):
        raise ValueError("C/J assignment support differs on the common event population")
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, n_events, size=(BOOTSTRAP_REPLICATES, n_events), dtype=np.int32)
    supported = np.flatnonzero(support_c)
    support_rows = rng.integers(
        0, len(supported), size=(BOOTSTRAP_REPLICATES, len(supported)), dtype=np.int32
    )
    support_rows = supported[support_rows]

    definitions = {
        "assignment_accuracy_supported": ("assignment_correct", "mean", True),
        "mtt_RMSE_GeV": ("mtt_squared_error", "sqrt", False),
        "joint_energy_score_GeV": ("joint_energy", "mean", False),
        "N_P_shuffle_penalty_GeV": ("joint_shuffle_penalty", "mean", False),
        "decay_chain_delta_pt_RMSE_GeV": ("decay_pt_squared", "sqrt", False),
        "mtt_68_coverage": ("mtt_covered_68", "mean", False),
    }
    output = {}
    for label, (field, transform, use_support) in definitions.items():
        index = support_rows if use_support else rows
        c_sample, j_sample = c_values[field][index], j_values[field][index]
        if transform == "sqrt":
            c_boot = np.sqrt(c_sample.mean(axis=1))
            j_boot = np.sqrt(j_sample.mean(axis=1))
            c_value = float(np.sqrt(c_values[field].mean()))
            j_value = float(np.sqrt(j_values[field].mean()))
        else:
            c_boot, j_boot = c_sample.mean(axis=1), j_sample.mean(axis=1)
            c_value, j_value = float(c_values[field].mean()), float(j_values[field].mean())
        delta_boot = c_boot - j_boot
        output[label] = {
            "C_minus_J": c_value - j_value,
            "C": c_value,
            "J": j_value,
            "event_bootstrap_95pct_interval_C_minus_J": [
                float(value) for value in np.quantile(delta_boot, [0.025, 0.975])
            ],
            "bootstrap_resamples": BOOTSTRAP_REPLICATES,
            "bootstrap_unit": "events",
        }
    return output


def summarize(records, event_arrays):
    per_seed = {}
    for seed in SEEDS:
        c_key, j_key = f"C_seed{seed}", f"J_seed{seed}"
        c_record, j_record = records[c_key], records[j_key]
        eligibility = pair_eligibility(c_record, j_record)
        per_seed[str(seed)] = {
            "pair_status": eligibility,
            "C": c_record["estimates"],
            "J": j_record["estimates"],
            "C_minus_J_event_bootstrap": paired_bootstrap(
                event_arrays[c_key], event_arrays[j_key], seed + 31_009
            ),
            "source_commit": c_record["training"].get("source_commit"),
            "run_script_sha256": c_record["training"].get("run_script_sha256"),
        }

    eligible = [
        seed for seed in SEEDS
        if per_seed[str(seed)]["pair_status"]["status"] == "strict_matched_pair"
    ]
    if not eligible:
        raise ValueError("No C/J seed pair passes matched training and evaluation provenance checks")

    metric_names = tuple(records[f"C_seed{SEEDS[0]}"]["estimates"])
    between_seed = {}
    for variant in VARIANTS:
        between_seed[variant] = {}
        for metric in metric_names:
            values = np.asarray(
                [records[f"{variant}_seed{seed}"]["estimates"][metric] for seed in eligible],
                dtype=np.float64,
            )
            between_seed[variant][metric] = {
                "mean": float(values.mean()),
                "between_seed_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
                "min": float(values.min()),
                "max": float(values.max()),
                "matched_seed_pairs": eligible,
            }

    difference_by_seed = {
        seed: {
            metric: values["C_minus_J"]
            for metric, values in per_seed[str(seed)]["C_minus_J_event_bootstrap"].items()
        }
        for seed in eligible
    }
    higher_is_better = {"assignment_accuracy_supported", "mtt_68_coverage"}
    ranking = {}
    for metric in metric_names:
        deltas = np.asarray([difference_by_seed[seed][metric] for seed in eligible])
        j_better = deltas < 0 if metric in higher_is_better else deltas > 0
        ranking[metric] = {
            "better_direction_for_J": "higher" if metric in higher_is_better else "lower",
            "C_minus_J_by_seed": {str(seed): float(difference_by_seed[seed][metric]) for seed in eligible},
            "J_better_seed_count": int(j_better.sum()),
            "eligible_seed_pair_count": len(eligible),
            "direction_consistent": bool(j_better.all() or (~j_better).all()),
            "J_better_on_every_pair": bool(j_better.all()),
        }

    source_fingerprints = {
        f"{variant}_seed{seed}": {
            "source_commit": records[f"{variant}_seed{seed}"]["training"].get("source_commit"),
            "run_script_sha256": records[f"{variant}_seed{seed}"]["training"].get("run_script_sha256"),
        }
        for seed in eligible
        for variant in VARIANTS
    }
    unique_sources = {
        (item["source_commit"], item["run_script_sha256"])
        for item in source_fingerprints.values()
    }
    return {
        "evaluation_population": {
            "events": 2000,
            "posterior_samples_per_event": 32,
            "evaluation_seed": 42011,
            "ordered_event_id_sha256": records[f"C_seed{SEEDS[0]}"]["ordered_event_id_sha256"],
        },
        "per_seed": per_seed,
        "matched_pairs_used_for_between_seed_statistics": eligible,
        "between_seed_statistics": between_seed,
        "J_ranking_consistency": ranking,
        "provenance_note": (
            "Between-seed spread is descriptive. It is not a pure stochastic-seed estimate when eligible pairs "
            "were trained from different source commits or run-script hashes."
        ),
        "eligible_pair_source_fingerprints": source_fingerprints,
        "eligible_pairs_share_one_source_fingerprint": len(unique_sources) == 1,
    }


def write_summary(metrics: dict, output: Path):
    lines = [
        "# Matched KNOT C/J seed replication",
        "",
        "All metrics use the same ordered 2,000 validation events, 32 posterior draws per event, and event-level paired bootstrap intervals.",
        "The 4D joint energy score, N-P shuffle penalty, and decay-chain consistency are recomputed with the established production metric definitions.",
        "",
        "## Per-seed results",
        "",
        "| Seed | Pair provenance | C mtt RMSE | J mtt RMSE | C−J RMSE | C joint energy | J joint energy | C−J energy | C N-P penalty | J N-P penalty | C−J penalty | C decay ΔpT RMSE | J decay ΔpT RMSE | C−J decay | C 68% coverage | J 68% coverage | C−J coverage |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        entry = metrics["per_seed"][str(seed)]
        c, j = entry["C"], entry["J"]
        delta = entry["C_minus_J_event_bootstrap"]
        status = entry["pair_status"]["status"]
        lines.append(
            f"| {seed} | {status} | {c['mtt_RMSE_GeV']:.3f} | {j['mtt_RMSE_GeV']:.3f} | {delta['mtt_RMSE_GeV']['C_minus_J']:+.3f} | "
            f"{c['joint_energy_score_GeV']:.3f} | {j['joint_energy_score_GeV']:.3f} | {delta['joint_energy_score_GeV']['C_minus_J']:+.3f} | "
            f"{c['N_P_shuffle_penalty_GeV']:+.3f} | {j['N_P_shuffle_penalty_GeV']:+.3f} | {delta['N_P_shuffle_penalty_GeV']['C_minus_J']:+.3f} | "
            f"{c['decay_chain_delta_pt_RMSE_GeV']:.3f} | {j['decay_chain_delta_pt_RMSE_GeV']:.3f} | {delta['decay_chain_delta_pt_RMSE_GeV']['C_minus_J']:+.3f} | "
            f"{c['mtt_68_coverage']:.3f} | {j['mtt_68_coverage']:.3f} | {delta['mtt_68_coverage']['C_minus_J']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "C−J differences are paired event-bootstrap estimates. For RMSE, energy, shuffle penalty and decay RMSE, negative favors J; for assignment accuracy and coverage, positive favors J.",
            "The assignment accuracy and its bootstrap interval use only events with explicit truth assignment support.",
            "",
            "## C−J paired event-bootstrap intervals",
            "",
            "Intervals are percentile 95% intervals; the resampling unit is the event, never the posterior draw.",
            "",
            "| Seed | Metric | C−J | 95% interval |",
            "|---:|---|---:|---:|",
        ]
    )
    for seed in SEEDS:
        entry = metrics["per_seed"][str(seed)]
        for metric, value in entry["C_minus_J_event_bootstrap"].items():
            low, high = value["event_bootstrap_95pct_interval_C_minus_J"]
            lines.append(f"| {seed} | {metric} | {value['C_minus_J']:+.4f} | [{low:+.4f}, {high:+.4f}] |")
    lines.extend(["", "## Between-seed descriptive summary", "", f"Strict matched pairs used: {metrics['matched_pairs_used_for_between_seed_statistics']}.", ""])
    if not metrics["eligible_pairs_share_one_source_fingerprint"]:
        lines.append("Eligible pairs span more than one training source fingerprint; the spread below is descriptive and does not isolate seed randomness.")
    lines.extend(["", "| Model | Metric | Mean | Between-seed SD | Range |", "|---|---|---:|---:|---:|"])
    for variant in VARIANTS:
        for metric, value in metrics["between_seed_statistics"][variant].items():
            spread = "n/a" if value["between_seed_sd"] is None else f"{value['between_seed_sd']:.4f}"
            lines.append(f"| {variant} | {metric} | {value['mean']:.4f} | {spread} | [{value['min']:.4f}, {value['max']:.4f}] |")
    lines.extend(["", "## Direction consistency", "", "| Metric | Pairs where J is better | Consistent direction | J better on every pair |", "|---|---:|---|---|"])
    for metric, value in metrics["J_ranking_consistency"].items():
        lines.append(
            f"| {metric} | {value['J_better_seed_count']}/{value['eligible_seed_pair_count']} | "
            f"{value['direction_consistent']} | {value['J_better_on_every_pair']} |"
        )
    lines.extend(["", "## Provenance limitations", ""])
    for seed in SEEDS:
        status = metrics["per_seed"][str(seed)]["pair_status"]
        if status["reasons"]:
            lines.append(f"- Seed {seed}: " + "; ".join(status["reasons"]) + ".")
        else:
            lines.append(f"- Seed {seed}: C and J match on recorded source, membership, shared initialization, training settings, and evaluation provenance.")
    lines.append("")
    (output / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument(
        "--training-root",
        type=Path,
        default=EVALDEV_ROOT / "results/experiments/knot/matched-seed-replication-v1",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    records, event_arrays = {}, {}
    reference_ids = None
    for seed in SEEDS:
        for variant in VARIANTS:
            key = f"{variant}_seed{seed}"
            record, event_ids, arrays = load_one(seed, variant, args.evaluation_root, args.training_root)
            if reference_ids is None:
                reference_ids = event_ids
            elif not np.array_equal(event_ids, reference_ids):
                raise ValueError(f"Ordered validation EVENT_IDs differ for {key}")
            records[key] = record
            event_arrays[key] = arrays

    for field in TRUTH_FIELDS:
        reference = None
        for seed in SEEDS:
            for variant in VARIANTS:
                key = f"{variant}_seed{seed}"
                with h5py.File(records[key]["paths"]["posterior"], "r") as source:
                    current = source[field][:]
                if reference is None:
                    reference = current
                elif not np.array_equal(current, reference):
                    raise ValueError(f"Truth or measured reference field {field} differs in {key}")

    metrics = summarize(records, event_arrays)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite seed summary: {args.output}")
    args.output.mkdir(parents=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    manifest = {
        "analysis": "matched C/J seed replication on common validation events",
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "metric_definition_script": str(Path(production.__file__).resolve()),
        "metric_definition_script_sha256": file_sha256(Path(production.__file__).resolve()),
        "metric_shuffle_seed": METRIC_SHUFFLE_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_unit": "events",
        "training_and_evaluation_inputs": records,
        "sealed_test_used": False,
        "output_files": ["metrics.json", "summary.md", "manifest.json"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    write_summary(metrics, args.output)
    print(json.dumps({"output": str(args.output), "strict_pairs": metrics["matched_pairs_used_for_between_seed_statistics"]}, indent=2))


if __name__ == "__main__":
    main()
