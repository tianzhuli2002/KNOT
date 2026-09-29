#!/usr/bin/env python3
"""Write the paired parent-CFM tables and evidence-gated comparison report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

MODELS = ("PCFM_noQ", "PCFM_Q", "PCFM_geom")
ESTIMATORS = ("single_draw", "observable_mean", "observable_q50", "joint_medoid")
CORE_OBSERVABLES = (
    "mttbar", "top_pt", "antitop_pt", "top_rapidity", "antitop_rapidity",
    "top_phi", "antitop_phi", "ptttbar", "yttbar",
)
METRIC_FIELDS = (
    "bias", "std", "sigma68", "mae", "rmse", "median_absolute_error",
    "fractional_bias", "fractional_sigma68", "fractional_rmse",
    "posterior_width_68_median", "posterior_width_68_mean",
    "posterior_width_90_median", "posterior_width_90_mean",
    "truth_coverage_68", "truth_coverage_90",
)


def _assignments(values: list[str], name: str) -> dict[str, Path]:
    result = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"{name} values must use LABEL=PATH syntax: {item!r}")
        label, raw_path = item.split("=", 1)
        if not label or label in result:
            raise ValueError(f"Empty or duplicate {name} label: {label!r}")
        result[label] = Path(raw_path).expanduser().resolve()
    return result


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_summaries(paths: dict[str, Path]) -> dict[str, dict]:
    records = {}
    for label, path in paths.items():
        if label not in MODELS:
            raise ValueError(f"Unknown model label {label!r}; expected {MODELS}")
        if not path.is_file():
            raise FileNotFoundError(f"Parent reduction summary is missing: {path}")
        record = _load_json(path)
        if record.get("schema") != "realnu_parent_posterior_reduction_v1":
            raise ValueError(f"Unsupported parent reduction summary: {path}")
        if record.get("sealed_test_used") is not False:
            raise ValueError(f"Parent summary does not establish sealed_test_used=false: {path}")
        records[label] = record
    if records:
        reference = next(iter(records.values()))
        for label, record in records.items():
            for key in ("source_bank_sha256", "event_id_sha256", "case", "solver", "steps", "samples"):
                if record.get(key) != reference.get(key):
                    raise ValueError(f"{label} summary differs in paired field {key}")
    return records


def _metrics_csv(records: dict[str, dict], path: Path) -> None:
    fields = (
        "model", "velocity_backbone", "conditioning", "checkpoint_sha256", "case",
        "solver", "steps", "nfe_per_sample", "samples", "events", "estimator",
        "observable", "region", *METRIC_FIELDS,
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, record in records.items():
            for estimator, observables in record["metrics"].items():
                for observable, regions in observables.items():
                    for region, metrics in regions.items():
                        writer.writerow({
                            "model": label,
                            "velocity_backbone": record["velocity_backbone"],
                            "conditioning": record["conditioning"],
                            "checkpoint_sha256": record["checkpoint_sha256"],
                            "case": record["case"],
                            "solver": record["solver"],
                            "steps": record["steps"],
                            "nfe_per_sample": record["nfe_per_sample"],
                            "samples": record["samples"],
                            "events": metrics["events"],
                            "estimator": estimator,
                            "observable": observable,
                            "region": region,
                            **{name: metrics.get(name) for name in METRIC_FIELDS},
                        })


def _bootstrap_csv(paths: dict[str, Path], csv_path: Path, copied_dir: Path) -> list[dict]:
    fields = (
        "comparison", "reference_model", "candidate_model", "reference_estimator",
        "candidate_estimator", "observable", "region", "metric", "events",
        "candidate_minus_reference", "bootstrap_95_low", "bootstrap_95_high",
        "bootstrap_probability_candidate_lower", "bootstrap_replicates",
        "training_seed_variation_included", "status",
    )
    rows = []
    copied_dir.mkdir(parents=True, exist_ok=True)
    for label, path in paths.items():
        record = _load_json(path)
        if record.get("schema") != "realnu_parent_paired_comparison_v1":
            raise ValueError(f"Unsupported paired comparison result: {path}")
        if record.get("sealed_test_used") is not False:
            raise ValueError(f"Paired comparison does not establish sealed_test_used=false: {path}")
        target = copied_dir / f"{label}.json"
        if path.expanduser().resolve() != target.resolve():
            shutil.copyfile(path, target)
        estimator_pairs = record["estimators"]
        comparison_map = record["comparisons"]
        for pair_name, observables in comparison_map.items():
            matching = next(
                item for item in estimator_pairs
                if item["candidate"] == pair_name
                or f"{item['candidate']}_minus_{item['reference']}" == pair_name
            )
            for observable, region_map in observables.items():
                for region, metric_map in region_map.items():
                    for metric, value in metric_map.items():
                        rows.append({
                            "comparison": label,
                            "reference_model": record["reference"]["model"],
                            "candidate_model": record["candidate"]["model"],
                            "reference_estimator": matching["reference"],
                            "candidate_estimator": matching["candidate"],
                            "observable": observable,
                            "region": region,
                            "metric": metric,
                            **value,
                            "training_seed_variation_included": record["bootstrap"][
                                "training_seed_variation_included"
                            ],
                        })
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _metric(record: dict | None, estimator: str, observable: str, region: str, key: str):
    if record is None:
        return None
    return record.get("metrics", {}).get(estimator, {}).get(observable, {}).get(region, {}).get(key)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "pending"
    return f"{float(value):.{digits}g}"


def _rotation_rows(paths: dict[str, Path], output_dir: Path) -> list[dict]:
    rows = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for label, path in paths.items():
        record = _load_json(path)
        if record.get("sealed_test_used") is not False or record.get("truth_free_conditioning") is not True:
            raise ValueError(f"Rotation summary is not validation-only and truth-free: {path}")
        destination = output_dir / f"{label}_rotation.json"
        if path.expanduser().resolve() != destination.resolve():
            shutil.copyfile(path, destination)
        model_status = record.get("diagnostic_model_status", "trained checkpoint")
        for angle_key, angle in sorted(record["per_angle"].items(), key=lambda item: float(item[0])):
            fixed = angle["input_rotation_same_source"]
            covariance = angle["formal_co_rotated_source_covariance"]
            geom_rotation = record.get("geometric_context_rotation", {}).get(angle_key)
            source_action = next(
                action for action in record["source_prior_rotation_action"]
                if abs(float(action["angle_rad"]) - float(angle_key)) < 1e-12
            )

            def rms(summary: dict | None) -> float | None:
                return None if summary is None else float(summary["rms"])

            rows.append({
                "model": label,
                "model_status": model_status,
                "angle_rad": float(angle_key),
                "sample_events": int(record["sample_events"]),
                "fixed_expected_phi_response_rad": fixed.get("expected_phi_response_rad"),
                "fixed_top_phi_response_mean_rad": fixed["top_phi_shift_rad"]["mean"],
                "fixed_top_phi_response_rms_rad": fixed["top_phi_shift_rad"]["rms"],
                "fixed_top_phi_response_error_rms_rad": rms(fixed.get("top_phi_response_error_rad")),
                "fixed_antitop_phi_response_mean_rad": fixed["antitop_phi_shift_rad"]["mean"],
                "fixed_antitop_phi_response_rms_rad": fixed["antitop_phi_shift_rad"]["rms"],
                "fixed_antitop_phi_response_error_rms_rad": rms(fixed.get("antitop_phi_response_error_rad")),
                "fixed_top_pt_change_rms_gev": rms(fixed.get("top_pt_change_gev")),
                "fixed_antitop_pt_change_rms_gev": rms(fixed.get("antitop_pt_change_gev")),
                "fixed_top_mass_change_rms_gev": rms(fixed.get("top_mass_change_gev")),
                "fixed_antitop_mass_change_rms_gev": rms(fixed.get("antitop_mass_change_gev")),
                "fixed_mttbar_change_rms_gev": rms(fixed.get("mttbar_change_gev")),
                "formal_top_phi_covariance_rms_rad": covariance["top_phi_residual_rad"]["rms"],
                "formal_antitop_phi_covariance_rms_rad": covariance["antitop_phi_residual_rad"]["rms"],
                "formal_top_pt_covariance_rms_gev": rms(covariance.get("top_pt_residual_gev")),
                "formal_antitop_pt_covariance_rms_gev": rms(covariance.get("antitop_pt_residual_gev")),
                "formal_top_mass_covariance_rms_gev": rms(covariance.get("top_mass_residual_gev")),
                "formal_antitop_mass_covariance_rms_gev": rms(covariance.get("antitop_mass_residual_gev")),
                "formal_mttbar_covariance_rms_gev": covariance["mttbar_residual_gev"]["rms"],
                "geometric_token_rotation_rms": None if geom_rotation is None else geom_rotation["rms"],
                "scaler_relative_transverse_std_difference": record["scaler_transverse"]["relative_std_difference"],
                "source_prior_max_covariance_deviation": max(
                    role["max_abs_covariance_minus_identity"]
                    for role in source_action["roles"]
                ),
            })
    return rows


def write_report(
    *,
    output_root: Path,
    summary_paths: dict[str, Path],
    bootstrap_paths: dict[str, Path],
    rotation_paths: dict[str, Path],
    source_manifest_path: Path,
    fairness_path: Path | None,
    parameter_counts_path: Path | None,
    geometry_cpu_validation_path: Path | None,
    solver_analysis_path: Path | None,
    runtime_profile_path: Path | None,
    solver_selection_path: Path | None,
    job_states: dict[str, str],
) -> dict:
    output_root = output_root.expanduser().resolve()
    metrics_dir = output_root / "metrics"
    bootstrap_dir = output_root / "bootstrap"
    rotation_dir = output_root / "rotation"
    report_dir = output_root / "report"
    for directory in (metrics_dir, bootstrap_dir, rotation_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    summaries = _read_summaries(summary_paths)
    metrics_path = metrics_dir / "parent_metrics.csv"
    _metrics_csv(summaries, metrics_path)
    bootstrap_path = bootstrap_dir / "paired_differences.csv"
    bootstrap_rows = _bootstrap_csv(bootstrap_paths, bootstrap_path, bootstrap_dir)
    rotation_rows = _rotation_rows(rotation_paths, rotation_dir)
    rotation_csv = rotation_dir / "rotation_diagnostics.csv"
    rotation_fields = (
        "model", "model_status", "angle_rad", "sample_events",
        "fixed_expected_phi_response_rad",
        "fixed_top_phi_response_mean_rad", "fixed_top_phi_response_rms_rad",
        "fixed_top_phi_response_error_rms_rad",
        "fixed_antitop_phi_response_mean_rad", "fixed_antitop_phi_response_rms_rad",
        "fixed_antitop_phi_response_error_rms_rad",
        "fixed_top_pt_change_rms_gev", "fixed_antitop_pt_change_rms_gev",
        "fixed_top_mass_change_rms_gev", "fixed_antitop_mass_change_rms_gev",
        "fixed_mttbar_change_rms_gev",
        "formal_top_phi_covariance_rms_rad", "formal_antitop_phi_covariance_rms_rad",
        "formal_top_pt_covariance_rms_gev", "formal_antitop_pt_covariance_rms_gev",
        "formal_top_mass_covariance_rms_gev", "formal_antitop_mass_covariance_rms_gev",
        "formal_mttbar_covariance_rms_gev", "geometric_token_rotation_rms",
        "scaler_relative_transverse_std_difference", "source_prior_max_covariance_deviation",
    )
    with rotation_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rotation_fields)
        writer.writeheader()
        writer.writerows(rotation_rows)
    source_manifest = _load_json(source_manifest_path.expanduser().resolve())
    fairness = _load_json(fairness_path.expanduser().resolve()) if fairness_path else None
    parameter_counts = _load_json(parameter_counts_path.expanduser().resolve()) if parameter_counts_path else None
    geometry_validation = _load_json(geometry_cpu_validation_path.expanduser().resolve()) if geometry_cpu_validation_path else None
    solver = _load_json(solver_analysis_path.expanduser().resolve()) if solver_analysis_path else None
    runtime = _load_json(runtime_profile_path.expanduser().resolve()) if runtime_profile_path else None
    selection = _load_json(solver_selection_path.expanduser().resolve()) if solver_selection_path else None

    full = all(name in summaries for name in MODELS) and selection is not None
    lines = [
        "# Parent CFM comparison",
        "",
        f"**Status:** {'complete' if full else 'partial; required scans or model checkpoints remain pending'}.",
        "",
        "## Scope and provenance",
        "",
        f"- Validation source bank: `{source_manifest.get('file_sha256', 'missing')}`; "
        f"seed {source_manifest.get('seed', 'unknown')}; "
        f"{source_manifest.get('validation_event_count', source_manifest.get('events', 'unknown')):,} events.",
        f"- Source-bank EVENT_ID SHA256: `{source_manifest.get('event_id_sha256', 'unknown')}`.",
        "- Predictions are truth-free at inference and the sealed test split is unused.",
        "- Paired bootstrap resamples validation EVENT_IDs; event-bootstrap uncertainty does not include training-seed variation.",
        "",
        "## Training fairness and model size",
        "",
    ]
    if fairness:
        lines.append(
            f"The resolved PCFM_Q/PCFM_geom configs differ in {fairness['differing_leaf_count']} leaves: "
            + ", ".join(f"`{name}`" for name in fairness["differences"])
            + ". Frozen scientific and training fields match."
        )
    else:
        lines.append("Resolved training-config comparison is pending.")
    if parameter_counts:
        lines.extend(["", "| Model | All parameters | Trainable parameters | Parent CFM | Backbone / query input |", "|---|---:|---:|---:|---|"])
        for label in MODELS:
            item = parameter_counts.get("models", {}).get(label)
            if item:
                lines.append(
                    f"| {label} | {item['all_parameters']:,} | {item['trainable_parameters']:,} | "
                    f"{item['parent_cfm_parameters']:,} | {item['velocity_backbone']} / "
                    f"{'queries' if item['use_topology_queries'] else 'learned roles'} |"
                )
        lines.append("")
        lines.append("Parameter counts are not capacity matching; PCFM_geom has fewer parameters than PCFM_Q.")
    lines.extend([
        "",
        "## L-GATr representation contract",
        "",
        "The visible encoder returns scalar tokens and the full L-GATr multivector tensor. "
        "Scalar object tokens omit the learned event token; EventModel prepends its pooled "
        "scalar event token, while `geometric_tokens` already retain the L-GATr event token. "
        "Both therefore align as `[B,N+1]`, with the event token unmasked. `H_geom` has "
        "shape `[B,N+1,1,16]` and the boolean padding mask has shape `[B,N+1]`.",
        "",
        "For each parent sample, the geometric CFM maps the scaled top/antitop state back "
        "to physical four-vectors, embeds the two vectors as multivectors, and conditions "
        "on measured-event `H_geom` plus the retained P0 scalar role/global context. "
        "The velocity output is mapped back to the exact existing scaled `(px,py,pz,log1p(m))` target coordinates.",
    ])
    if geometry_validation:
        architecture = geometry_validation["architecture"]
        representation = architecture["representation_rotation"]
        dynamic = architecture["dynamic_parent_token_rotation"]
        max_hgeom_rms = max(
            item["h_geom_covariance_residual"]["rms"]
            for item in representation.values()
        )
        max_parent_p4 = max(
            item["physical_parent_p4_residual_gev"]["max_abs"]
            for item in dynamic.values()
        )
        max_parent_token = max(
            item["geometric_parent_token_residual"]["max_abs"]
            for item in dynamic.values()
        )
        gradients = geometry_validation["forward_backward"]["gradient_norms"]
        lines.extend([
            "",
            f"The local CPU gate passed on a 2-event fixture: `H_geom` shape `{architecture['geometric_tokens_shape']}`, "
            f"maximum H_geom rotation RMS {max_hgeom_rms:.3g}, dynamic parent p4 maximum residual "
            f"{max_parent_p4:.3g} GeV, and parent-token maximum residual {max_parent_token:.3g}.",
            f"Visible/query/geometry gradient norms were {gradients['visible_encoder']:.4g}, "
            f"{gradients['topology_queries']:.4g}, and {gradients['geometry_velocity']:.4g}; "
            f"tiny-fixture loss changed from {geometry_validation['tiny_overfit_losses'][0]:.4g} "
            f"to {geometry_validation['tiny_overfit_losses'][-1]:.4g}. Checkpoint reload was "
            f"exact: {geometry_validation['save_reload_fixed_source_equal']}; truth-free "
            f"prediction passed for {geometry_validation['truth_free_prediction']['events']} events.",
            "These checks validate the representation and implementation contract; they are not physics performance results.",
        ])
    lines.extend(["", "## Inference policy", ""])
    if selection:
        lines.append(
            f"Selected `{selection['solver']}` with {selection['steps']} steps and "
            f"M={selection['samples']} ({selection['nfe_per_sample']} NFE/sample, "
            f"{selection['aggregate_nfe_per_event']} NFE×sample work units per event)."
        )
        if selection.get("reason"):
            lines.append(selection["reason"])
    else:
        lines.append("FINAL_SOLVER / FINAL_STEPS / FINAL_M are pending the PCFM_Q solver scan.")
    if solver:
        lines.extend([
            "",
            "### Equal-NFE solver/sample tradeoff",
            "",
            "| Solver | Steps | M | NFE/sample | NFE×M/event | N | Mean mttbar RMSE [GeV] | q50 mttbar RMSE [GeV] | Medoid mttbar RMSE [GeV] |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for case in solver.get("equal_nfe_cases", []):
            metrics = case["metrics"]
            lines.append(
                f"| {case['solver']} | {case['steps']} | {case['samples']} | "
                f"{case['nfe_per_sample']} | {case['aggregate_nfe_per_event']} | "
                f"{metrics['observable_mean']['mttbar']['inclusive']['events']:,} | "
                f"{_fmt(metrics['observable_mean']['mttbar']['inclusive']['rmse'])} | "
                f"{_fmt(metrics['observable_q50']['mttbar']['inclusive']['rmse'])} | "
                f"{_fmt(metrics['joint_medoid']['mttbar']['inclusive']['rmse'])} |"
            )
        lines.extend([
            "",
            "### Nested sample-count convergence",
            "",
            "The scan JSON/CSV and `plots/convergence` report bias, residual std, sigma68, MAE, and RMSE versus M for observable mean, q50, and joint medoid. Solver changes are compared with Heun32 using identical source states; the plots show the RMS prediction change divided by Heun32 reconstruction RMSE for mttbar and both parent azimuths.",
        ])
    if runtime:
        lines.extend([
            "",
            f"Measured CPU timing used {runtime['events']:,} validation events, batch size {runtime['batch_size']}, and {runtime['repeats']} repeats. Timing covers only parent CFM sampling calls; it excludes data loading, model loading, and output serialization.",
        ])
        if selection:
            chosen = [
                item for item in runtime["cases"].values()
                if item["steps"] == selection["steps"]
            ]
            lines.extend([
                "",
                "| M | NFE×M/event | CPU seconds/event | Events/s |",
                "|---:|---:|---:|---:|",
            ])
            for item in sorted(chosen, key=lambda value: value["samples"]):
                lines.append(
                    f"| {item['samples']} | {item['aggregate_nfe_per_event']} | "
                    f"{item['median_seconds_per_event']:.6g} | {item['events_per_second']:.6g} |"
                )
        else:
            equal_nfe = ((32, 1), (16, 2), (8, 4), (4, 8))
            measured = [
                runtime["cases"][f"heun_steps_{steps}_samples_{samples}"]
                for steps, samples in equal_nfe
                if f"heun_steps_{steps}_samples_{samples}" in runtime["cases"]
            ]
            if measured:
                lines.extend([
                    "",
                    "### Measured equal-NFE CPU costs",
                    "",
                    "These timings compare the same total Heun NFE×M work per event. Accuracy is not shown here; policy selection remains pending the paired solver scan.",
                    "",
                    "| Solver | Steps | M | NFE×M/event | CPU seconds/event | Events/s |",
                    "|---|---:|---:|---:|---:|---:|",
                ])
                for item in measured:
                    lines.append(
                        f"| Heun | {item['steps']} | {item['samples']} | "
                        f"{item['aggregate_nfe_per_event']} | "
                        f"{item['median_seconds_per_event']:.6g} | "
                        f"{item['events_per_second']:.6g} |"
                    )
    lines.extend(["", "## Point-estimator and posterior metrics", ""])
    if summaries:
        lines.extend([
            "`metrics/parent_metrics.csv` contains all reducer observables, estimators, "
            "regions, posterior widths, coverage, and fractional metrics. RMSE is reported "
            "as RMSE, not called resolution.",
            "",
            "| Model | Estimator | mttbar RMSE [GeV] | 345–400 bias [GeV] | 345–400 RMSE [GeV] | >1000 RMSE [GeV] | top φ MAE [rad] | antitop φ MAE [rad] | Joint energy score [GeV] |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for model in MODELS:
            record = summaries.get(model)
            if record is None:
                continue
            for estimator in ESTIMATORS:
                lines.append(
                    f"| {model} | {estimator} | "
                    f"{_fmt(_metric(record, estimator, 'mttbar', 'inclusive', 'rmse'))} | "
                    f"{_fmt(_metric(record, estimator, 'mttbar', 'truth_mttbar/345-400', 'bias'))} | "
                    f"{_fmt(_metric(record, estimator, 'mttbar', 'truth_mttbar/345-400', 'rmse'))} | "
                    f"{_fmt(_metric(record, estimator, 'mttbar', 'truth_mttbar/>1000', 'rmse'))} | "
                    f"{_fmt(_metric(record, estimator, 'top_phi', 'inclusive', 'mae'))} | "
                    f"{_fmt(_metric(record, estimator, 'antitop_phi', 'inclusive', 'mae'))} | "
                    f"{_fmt(record.get('joint_energy_score_parent_p4_gev'))} |"
                )
            lines.append(
                f"Low-stat/high-mass support for {model}: "
                f"{record['region_counts'].get('truth_mttbar/>1000', 0):,} validation events above 1000 GeV."
            )
    else:
        lines.append("No full model posterior reduction is available yet.")
    lines.extend(["", "## Paired bootstrap differences", ""])
    if bootstrap_rows:
        lines.extend([
            "All rows are candidate minus reference; negative RMSE/MAE favors the candidate. "
            "Estimator comparisons are within a model; architecture comparisons use the same estimator.",
            "",
            "| Comparison | Estimator | Observable / region | Metric | Δ | 95% interval | P(candidate lower) | N |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ])
        for row in bootstrap_rows:
            delta = row.get("candidate_minus_reference")
            interval = (
                f"[{row['bootstrap_95_low']:.4g}, {row['bootstrap_95_high']:.4g}]"
                if "bootstrap_95_low" in row else "empty"
            )
            p_lower = _fmt(row.get("bootstrap_probability_candidate_lower"), 3)
            lines.append(
                f"| {row['comparison']} | {row['candidate_estimator']} − {row['reference_estimator']} | "
                f"{row['observable']} / {row['region']} | {row['metric']} | {_fmt(delta, 4)} | "
                f"{interval} | {p_lower} | {row.get('events', 0):,} |"
            )
    else:
        lines.append("Paired architecture and estimator bootstrap comparisons are pending.")
    lines.extend(["", "## Rotation and absolute direction", ""])
    if rotation_rows:
        lines.extend([
            "The fixed-source test rotates the measured event while keeping the latent source fixed. Its response error is the wrapped observed φ shift minus the applied angle. The formal test co-rotates both the event and source, then compares with the rotated baseline output.",
            "",
            "### Fixed-source orientation response",
            "",
            "| Model / status | α [rad] | N | Role | observed mean Δφ [rad] | expected Δφ [rad] | response-error RMS [rad] | pT-change RMS [GeV] | mass-change RMS [GeV] | mttbar-change RMS [GeV] |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rotation_rows:
            for role, phi_key, error_key, pt_key, mass_key in (
                ("top", "top", "fixed_top_phi_response_error_rms_rad", "fixed_top_pt_change_rms_gev", "fixed_top_mass_change_rms_gev"),
                ("antitop", "antitop", "fixed_antitop_phi_response_error_rms_rad", "fixed_antitop_pt_change_rms_gev", "fixed_antitop_mass_change_rms_gev"),
            ):
                lines.append(
                    f"| {row['model']} ({row['model_status']}) | {row['angle_rad']:.4f} | "
                    f"{row['sample_events']:,} | {role} | "
                    f"{_fmt(row[f'fixed_{phi_key}_phi_response_mean_rad'], 4)} | "
                    f"{_fmt(row['fixed_expected_phi_response_rad'], 4)} | "
                    f"{_fmt(row[error_key], 4)} | {_fmt(row[pt_key], 4)} | "
                    f"{_fmt(row[mass_key], 4)} | {_fmt(row['fixed_mttbar_change_rms_gev'], 4)} |"
                )
        lines.extend([
            "",
            "### Formal co-rotated-source covariance",
            "",
            "| Model / status | α [rad] | N | top/antitop φ residual RMS [rad] | top/antitop pT residual RMS [GeV] | top/antitop mass residual RMS [GeV] | mttbar residual RMS [GeV] | H_geom RMS |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in rotation_rows:
            lines.append(
                f"| {row['model']} ({row['model_status']}) | {row['angle_rad']:.4f} | "
                f"{row['sample_events']:,} | "
                f"{_fmt(row['formal_top_phi_covariance_rms_rad'], 4)} / "
                f"{_fmt(row['formal_antitop_phi_covariance_rms_rad'], 4)} | "
                f"{_fmt(row['formal_top_pt_covariance_rms_gev'], 4)} / "
                f"{_fmt(row['formal_antitop_pt_covariance_rms_gev'], 4)} | "
                f"{_fmt(row['formal_top_mass_covariance_rms_gev'], 4)} / "
                f"{_fmt(row['formal_antitop_mass_covariance_rms_gev'], 4)} | "
                f"{_fmt(row['formal_mttbar_covariance_rms_gev'], 4)} | "
                f"{_fmt(row['geometric_token_rotation_rms'], 4)} |"
            )
        max_scaler = max(row["scaler_relative_transverse_std_difference"] for row in rotation_rows)
        max_prior = max(row["source_prior_max_covariance_deviation"] for row in rotation_rows)
        lines.append(
            f"The largest measured relative px/py scaler-width mismatch is {max_scaler:.4%}; "
            f"the largest source-prior covariance deviation from identity is {max_prior:.4%}. "
            "The scaler is approximately, not exactly, transverse-rotation compatible."
        )
        if any("untrained" in row["model_status"].lower() for row in rotation_rows):
            lines.append("Untrained geometry rotation results describe representation and sampler structure only; they are not evidence of physics improvement.")
    else:
        lines.append("Model rotation diagnostics are pending.")
    lines.extend(["", "## Evidence-gated answers", ""])
    q_rotation_candidates = [row for row in rotation_rows if row["model"] == "PCFM_Q"]
    q_rotation = min(
        q_rotation_candidates,
        key=lambda row: abs(row["angle_rad"] - 0.5 * 3.141592653589793),
        default=None,
    )
    trained_rotation_rows = [
        row for row in rotation_rows if "untrained" not in row["model_status"].lower()
    ]
    orientation_answer = (
        "pending rotation diagnostics"
        if q_rotation is None
        else (
            "PCFM_Q has no measurable fixed-source orientation response at π/2: "
            f"top/antitop Δφ RMS {q_rotation['fixed_top_phi_response_rms_rad']:.3g}/"
            f"{q_rotation['fixed_antitop_phi_response_rms_rad']:.3g} rad"
        )
    )
    covariance_answer = "pending"
    if trained_rotation_rows:
        q_covariance_candidates = [row for row in trained_rotation_rows if row["model"] == "PCFM_Q"]
        q_covariance = min(
            q_covariance_candidates,
            key=lambda row: abs(row["angle_rad"] - 0.5 * 3.141592653589793),
            default=None,
        )
        if q_covariance is not None:
            covariance_answer = (
                "PCFM_Q does not satisfy exact co-rotated-source covariance: "
                f"top/antitop φ residual RMS {q_covariance['formal_top_phi_covariance_rms_rad']:.3g}/"
                f"{q_covariance['formal_antitop_phi_covariance_rms_rad']:.3g} rad and "
                f"mttbar RMS {q_covariance['formal_mttbar_covariance_rms_gev']:.3g} GeV; "
                "compare the other trained-model rows where available"
            )
        else:
            covariance_answer = "see formal residual rows for the trained models listed"
    answers = {
        "Conventional scalar-query path sensitivity to measured-event orientation": orientation_answer,
        "Co-rotated-source covariance by model": covariance_answer,
        "Production scaler transverse-rotation compatibility": (
            "approximately compatible; transverse width mismatch and source-prior covariance deviations are quantified above" if rotation_rows else "pending"
        ),
        "PCFM_noQ versus PCFM_Q": "pending a full paired prediction and bootstrap" if "PCFM_noQ" not in summaries else "see paired bootstrap table",
        "PCFM_geom absolute-direction improvement": "pending geometry training and matched paired evaluation" if "PCFM_geom" not in summaries else "see paired bootstrap and rotation tables",
        "PCFM_geom mttbar and 345–400 GeV improvement": "pending geometry training and matched paired evaluation" if "PCFM_geom" not in summaries else "see paired bootstrap table",
        "Architecture versus estimator effects": (
            "paired architecture comparisons remain pending; available within-model estimator comparisons are shown above"
            if not all(name in summaries for name in MODELS)
            else "reported separately by architecture and estimator comparison labels above"
        ),
        "Preferred estimator": "no globally preferred estimator is selected; report all four estimators because their effects vary by observable and truth-mttbar region",
        "Posterior samples needed": "pending the Q M=1,2,4,8,16,32 convergence study" if solver is None else "see solver convergence analysis",
        "Compute cost": "exact NFE×M is recorded; measured CPU time-vs-M awaits the runtime profile" if runtime is None else "see NFE work and measured runtime profile",
        "Next baseline": "defer until all available checkpoints share the same inference policy and paired validation IDs",
        "Training-seed ambiguity": "one training seed per architecture; event bootstrap does not include seed variation",
    }
    for question, answer in answers.items():
        lines.append(f"- **{question}:** {answer}.")
    lines.extend(["", "## Jobs and remaining issues", ""])
    lines.extend([
        "- PCFM_noQ job 21527436 was inspected once before this continuation and was not re-queried or duplicated.",
        "- Its live diagnostics source remains `/net/scratch/w00238tl/JointEventReconstruction_topology_queries_diagnostics`, branch `research/realnu-cfm-query-diagnostics-v2`, clean at `314bfc465180990a399e29cf63c48be4fb6ecd68`.",
    ])
    for label, state in job_states.items():
        lines.append(f"- {label}: {state}.")
    if solver is None:
        lines.append("- The 10k-event PCFM_Q solver/sample scan has not completed; no final policy has been selected.")
    if runtime is None:
        lines.append("- The measured CPU runtime-versus-M profile has not completed.")
    if not all(name in summaries for name in MODELS):
        lines.append("- Full three-model evaluation and architecture recommendation remain incomplete.")
    if fairness:
        lines.extend(["", "## Training configuration diff", "", "```json", json.dumps(fairness, indent=2), "```"])
    if parameter_counts:
        lines.extend(["", "## Parameter-count evidence", "", "```json", json.dumps(parameter_counts, indent=2), "```"])
    report_path = report_dir / "PARENT_COMPARISON.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "status": "complete" if full else "partial",
        "report": str(report_path),
        "metrics_csv": str(metrics_path),
        "paired_csv": str(bootstrap_path),
        "rotation_csv": str(rotation_csv),
        "model_summaries": sorted(summaries),
        "paired_comparisons": list(bootstrap_paths),
        "rotation_models": sorted({row["model"] for row in rotation_rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-bank-manifest", type=Path, required=True)
    parser.add_argument("--summary", action="append", default=[], metavar="MODEL=JSON")
    parser.add_argument("--paired", action="append", default=[], metavar="LABEL=JSON")
    parser.add_argument("--rotation", action="append", default=[], metavar="MODEL=JSON")
    parser.add_argument("--fairness-diff", type=Path)
    parser.add_argument("--parameter-counts", type=Path)
    parser.add_argument("--geometry-cpu-validation", type=Path)
    parser.add_argument("--solver-analysis", type=Path)
    parser.add_argument("--runtime-profile", type=Path)
    parser.add_argument("--solver-selection", type=Path)
    parser.add_argument("--job-state", action="append", default=[], metavar="JOB=STATE")
    args = parser.parse_args()
    job_states = {}
    for item in args.job_state:
        if "=" not in item:
            raise ValueError(f"--job-state must use JOB=STATE syntax: {item!r}")
        job, state = item.split("=", 1)
        job_states[job] = state
    result = write_report(
        output_root=args.output_root,
        summary_paths=_assignments(args.summary, "summary"),
        bootstrap_paths=_assignments(args.paired, "paired"),
        rotation_paths=_assignments(args.rotation, "rotation"),
        source_manifest_path=args.source_bank_manifest,
        fairness_path=args.fairness_diff,
        parameter_counts_path=args.parameter_counts,
        geometry_cpu_validation_path=args.geometry_cpu_validation,
        solver_analysis_path=args.solver_analysis,
        runtime_profile_path=args.runtime_profile,
        solver_selection_path=args.solver_selection,
        job_states=job_states,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
