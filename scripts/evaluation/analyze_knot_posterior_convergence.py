"""Quantify KNOT-J posterior sample-count convergence and draw variability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "results/experiments/knot/matched-production-v1/evaluation-v1/full-10k"
DEFAULT_K32 = EVALUATION / "J/posterior.h5"
DEFAULT_K64 = EVALUATION / "J-K64/posterior.h5"
DEFAULT_OUTPUT = EVALUATION / "posterior-usage-plotting-v1"
DRAW_METRICS = (
    "wasserstein_1",
    "ks_distance",
    "iqr_ratio",
    "central_68_width_ratio",
    "q90_tail_fraction_ratio",
    "q95_tail_fraction_ratio",
    "q99_tail_fraction_ratio",
)
DISPLAY_METRICS = (
    ("wasserstein_1", "Wasserstein-1 [GeV]"),
    ("ks_distance", "KS distance"),
    ("iqr_ratio", "IQR ratio"),
    ("q95_tail_fraction_ratio", "q95 tail ratio"),
)
POSTERIOR_MEAN_DEFINITION = (
    "For nonlinear observable O, evaluate O on each coherent posterior sample and then "
    "average across posterior samples."
)

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_knot_posterior_usage as usage

from joint_event_reconstruction.plotting_style import (
    COLORS,
    controlled_ratio_ylim,
    decorate_axis,
    format_legend,
    plot_histogram_ratio,
    ratio_axes,
    save_figure,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    stripped = payload.rstrip(b"\x00\r\n\t ")
    if b"\x00" in stripped:
        raise ValueError(f"{path}: NUL byte occurs before the end of JSON content")
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _observable_values(data: dict[str, Any]) -> dict[str, np.ndarray]:
    parents = usage._parent_observables(data["parent_samples"], str(data["path"]))
    neutrinos = usage._neutrino_observables(data["neutrino_samples"], str(data["path"]))
    return {**parents, **neutrinos}


def _coverage_by_level(
    samples: np.ndarray, truth: np.ndarray, name: str
) -> dict[float, dict[str, float]]:
    rows = usage._coverage_rows(samples, truth, observable=name)
    requested = (0.50, 0.68, 0.90)
    coverage = {
        float(row["nominal_credibility"]): {
            "empirical_coverage": float(row["empirical_coverage"]),
            "finite_draw_rank_expectation": float(row["finite_draw_rank_expectation"]),
        }
        for row in rows
        if float(row["nominal_credibility"]) in requested
    }
    if set(coverage) != set(requested):
        raise ValueError(f"{name}: missing requested empirical coverage levels")
    return coverage


def _migration_comparison(
    name: str,
    truth: np.ndarray,
    samples32: np.ndarray,
    samples64: np.ndarray,
    edges: np.ndarray,
) -> dict[str, Any]:
    matrix32 = usage._migration_matrix(
        truth,
        samples32,
        edges,
        representation="posterior_distribution",
        label=f"{name} K32 posterior migration",
    )
    matrix64 = usage._migration_matrix(
        truth,
        samples64,
        edges,
        representation="posterior_distribution",
        label=f"{name} K64 posterior migration",
    )
    difference = matrix64["matrix_percent"] - matrix32["matrix_percent"]
    truth_bin_tv = 0.5 * np.sum(np.abs(difference), axis=0) / 100.0
    return {
        "bin_edges": np.asarray(edges, dtype=np.float64).tolist(),
        "matrix_orientation": "rows are reconstructed bins; columns are truth bins",
        "normalization": "each truth-bin column sums to 100 percent",
        "max_abs_cell_change_percentage_points": float(np.max(np.abs(difference))),
        "mean_abs_cell_change_percentage_points": float(np.mean(np.abs(difference))),
        "frobenius_norm_percentage_points": float(np.linalg.norm(difference)),
        "truth_bin_total_variation_distance": truth_bin_tv.tolist(),
        "truth_bin_total_variation_mean": float(np.mean(truth_bin_tv)),
        "truth_bin_total_variation_max": float(np.max(truth_bin_tv)),
        "max_column_normalization_error_K32_percentage_points": float(
            matrix32["max_column_normalization_error"]
        ),
        "max_column_normalization_error_K64_percentage_points": float(
            matrix64["max_column_normalization_error"]
        ),
        "event_weight_closure_error_K32": float(matrix32["max_event_weight_closure_error"]),
        "event_weight_closure_error_K64": float(matrix64["max_event_weight_closure_error"]),
        "total_event_weight_K32": float(matrix32["total_weight"]),
        "total_event_weight_K64": float(matrix64["total_weight"]),
    }


def _plot_spectrum_convergence(
    output: Path,
    name: str,
    spec: dict[str, Any],
    truth: np.ndarray,
    samples32: np.ndarray,
    samples64: np.ndarray,
) -> list[Path]:
    edges = np.asarray(spec["edges"], dtype=np.float64)
    flat32, weights32 = usage._flatten_population(
        samples32, truth, representation="posterior_distribution"
    )
    flat64, weights64 = usage._flatten_population(
        samples64, truth, representation="posterior_distribution"
    )
    counts32, density32 = usage._histogram(flat32, weights32, edges, label=f"{name} K32")
    counts64, density64 = usage._histogram(flat64, weights64, edges, label=f"{name} K64")

    fig, main, lower = ratio_axes(ylabel="K32 / K64")
    main.stairs(
        density32,
        edges,
        color=COLORS["J"],
        linewidth=2.2,
        label="KNOT-J full posterior, K=32",
    )
    main.stairs(
        density64,
        edges,
        color=COLORS["posterior"],
        linewidth=2.2,
        label="KNOT-J full posterior, K=64",
    )
    ratio, supported = plot_histogram_ratio(
        lower,
        counts32,
        counts64,
        edges,
        method="J",
        min_reference_events=usage.MIN_RATIO_SUPPORT,
        label="K32 / K64",
    )
    del supported
    centers = 0.5 * (edges[:-1] + edges[1:])
    controlled_ratio_ylim(lower, [ratio], centers, colors=[COLORS["J"]])
    main.set_ylabel(spec["density_label"])
    main.set_title(f"KNOT-J posterior sample-count convergence: {name}")
    lower.set_xlabel(spec["label"])
    format_legend(main, ncol=1, loc="upper right", fontsize=9.5)
    decorate_axis(main, grid="y")
    decorate_axis(lower, grid="y")
    stem = output / "plots/convergence" / f"{name}_k32_k64_posterior"
    return save_figure(fig, stem)


def _plot_one_draw_metrics(
    output: Path,
    values: dict[str, np.ndarray],
    headline: dict[str, float],
    percentiles: dict[str, float],
) -> list[Path]:
    fig, axes = plt.subplots(1, len(DISPLAY_METRICS), figsize=(12.5, 3.8))
    for axis, (metric, label) in zip(axes, DISPLAY_METRICS, strict=True):
        axis.hist(values[metric], bins=14, color=COLORS["draw"], alpha=0.75, edgecolor="white")
        axis.axvline(headline[metric], color=COLORS["J"], linewidth=2.0, linestyle="--")
        axis.text(
            0.97,
            0.95,
            f"displayed draw: p{percentiles[metric]:.0f}",
            ha="right",
            va="top",
            transform=axis.transAxes,
            fontsize=9,
        )
        axis.set_xlabel(label)
        axis.set_ylabel("Pseudo-datasets")
        decorate_axis(axis, grid="y")
    fig.suptitle("mttbar metrics across event-keyed one-draw pseudo-datasets", y=1.03)
    return save_figure(fig, output / "plots/core/mttbar_one_draw_metric_ensemble")


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _scorecard(output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    point_rows = {
        row["estimator"]: row
        for row in _read_csv_rows(output / "tables/point_metrics.csv")
        if row["observable"] == "mttbar"
    }
    spectrum_rows = {
        row["representation"]: row
        for row in _read_csv_rows(output / "tables/spectrum_metrics.csv")
        if row["observable"] == "mttbar" and row["family"] == "posterior_usage"
    }
    if set(point_rows) != {"one_draw", "posterior_mean"} or set(spectrum_rows) != {
        "one_draw",
        "posterior_mean",
        "posterior_distribution",
    }:
        raise ValueError("K64 mttbar point or population metrics do not contain the expected representations")

    rows: list[dict[str, Any]] = []
    for representation, point_key in (
        ("one_draw", "one_draw"),
        ("posterior_mean", "posterior_mean"),
        ("full_posterior", "posterior_distribution"),
    ):
        point: dict[str, str] | None = point_rows.get(point_key)
        spectrum = spectrum_rows[point_key]
        rows.append(
            {
                "representation": representation,
                "point_bias": point["bias"] if point else "n/a",
                "point_RMSE": point["rmse"] if point else "n/a",
                "point_MAE": point["mae"] if point else "n/a",
                "population_W1": spectrum["wasserstein_1"],
                "population_KS": spectrum["ks_distance"],
                "IQR_ratio": spectrum["iqr_ratio"],
                "central_68_width_ratio": spectrum["central_68_width_ratio"],
                "q95_tail_ratio": spectrum["q95_tail_fraction_ratio"],
            }
        )

    calibration = [
        row
        for row in _read_csv_rows(output / "tables/calibration_metrics.csv")
        if row["diagnostic"] == "coverage" and row["observable"] == "mttbar"
    ]
    coverage = {
        str(round(float(row["nominal_credibility"]) * 100)): float(
            row["empirical_coverage"]
        )
        for row in calibration
        if float(row["nominal_credibility"]) in (0.5, 0.68, 0.9)
    }
    if set(coverage) != {"50", "68", "90"}:
        raise ValueError("K64 mttbar coverage table is missing the 50/68/90 percent levels")
    rows[-1].update(
        {
            "posterior_coverage_50": coverage["50"],
            "posterior_coverage_68": coverage["68"],
            "posterior_coverage_90": coverage["90"],
        }
    )
    rank = _read_json(output / "metadata/calibration_rank_summary.json")
    proper_score_rows = [
        row
        for row in _read_csv_rows(output / "tables/calibration_metrics.csv")
        if row["diagnostic"] == "proper_score"
        and row["observable"] == "mttbar"
        and row["score"] == "CRPS"
    ]
    if len(proper_score_rows) != 1:
        raise ValueError("K64 mttbar CRPS summary is missing or duplicated")
    spin_coverage = {
        row["observable"]: float(row["empirical_coverage"])
        for row in _read_csv_rows(output / "tables/calibration_metrics.csv")
        if row["diagnostic"] == "coverage"
        and row["observable"] in {"spin/cos_theta_plus", "spin/cos_theta_minus"}
        and float(row["nominal_credibility"]) == 0.68
    }
    if set(spin_coverage) != {"spin/cos_theta_plus", "spin/cos_theta_minus"}:
        raise ValueError("K64 spin helicity coverage diagnostics are missing")
    posterior_quality = {
        "coverage": coverage,
        "spin_helicity_68_coverage": spin_coverage,
        "mttbar_CRPS": float(proper_score_rows[0]["score_value"]),
        "rank_summary": rank,
    }
    return rows, posterior_quality


def _summary_markdown(
    output: Path,
    scorecard: list[dict[str, Any]],
    posterior_quality: dict[str, Any],
    convergence_rows: list[dict[str, Any]],
    migration: dict[str, Any],
    draw_summary: dict[str, Any],
    mean_of_observable_check: dict[str, float],
    k64_is_publication_default: bool,
) -> None:
    lines = [
        "# KNOT posterior usage summary",
        "",
        "The primary posterior is frozen KNOT-J with K=64. The matched M/C/J factorisation comparison remains at K=32 for every model.",
        "",
        "A one draw is one deterministic, event-keyed coherent posterior sample. The posterior mean for a nonlinear observable is computed as E[O|x] = (1/K) sum_k O(y^(k)); each observable is evaluated on each coherent sample before averaging. The full posterior uses all K samples with event-normalised weight 1/K and is excluded from ordinary event residuals.",
        "",
        "## mttbar point reconstruction",
        "",
        "| Estimator | Bias [GeV] | RMSE [GeV] | MAE [GeV] |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in scorecard[:2]:
        lines.append(
            f"| {row['representation']} | {float(row['point_bias']):.3f} | "
            f"{float(row['point_RMSE']):.3f} | {float(row['point_MAE']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## mttbar population reconstruction",
            "",
            "The full posterior has no point RMSE; it is reported as `n/a` for point metrics.",
            "",
            "| Representation | W1 [GeV] | KS | IQR ratio | 68% width ratio | q95 tail ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in scorecard:
        lines.append(
            f"| {row['representation']} | {float(row['population_W1']):.3f} | "
            f"{float(row['population_KS']):.4f} | {float(row['IQR_ratio']):.4f} | "
            f"{float(row['central_68_width_ratio']):.4f} | {float(row['q95_tail_ratio']):.4f} |"
        )
    coverage = posterior_quality["coverage"]
    rank = posterior_quality["rank_summary"]
    lines.extend(
        [
            "",
            "## Conditional posterior quality",
            "",
            f"mttbar empirical coverage is 50%: {coverage['50']:.3f}, 68%: {coverage['68']:.3f}, and 90%: {coverage['90']:.3f}.",
            f"The K=64 mttbar rank/PIT summary has maximum absolute rank-fraction deviation {rank['rank_fraction_max_abs_deviation_from_uniform']:.5f}, RMS deviation {rank['rank_fraction_rms_deviation_from_uniform']:.5f}, and chi-square p-value {rank['p_value']:.3g}; CRPS is {posterior_quality['mttbar_CRPS']:.3f} GeV.",
            (
                "mttbar coverage is slightly below nominal. The larger under-dispersion is in the spin helicity cosines: "
                "the K=64 suite gives 68% coverage of "
                f"{posterior_quality['spin_helicity_68_coverage']['spin/cos_theta_plus']:.3f} and "
                f"{posterior_quality['spin_helicity_68_coverage']['spin/cos_theta_minus']:.3f} "
                "for the positive and negative lepton helicity cosines. Population-spectrum closure does not establish "
                "eventwise posterior calibration."
            ),
            "",
            "## K32 to K64 convergence",
            "",
            "Each row compares the same event population and frozen J checkpoint. K32 and K64 were generated on CUDA and CPU respectively, so their posterior sample values are separate, non-nested Monte Carlo draws; no inference was rerun for this comparison.",
            "",
            "| Observable | W1(K32,K64) | KS(K32,K64) | Δ truth W1 | Δ truth KS | Δ IQR ratio | Δ 68% width | Δ q95 tail |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in convergence_rows:
        lines.append(
            f"| {row['observable']} | {float(row['K32_K64_W1']):.4g} | "
            f"{float(row['K32_K64_KS']):.4g} | {float(row['delta_truth_W1_K64_minus_K32']):+.4g} | "
            f"{float(row['delta_truth_KS_K64_minus_K32']):+.4g} | "
            f"{float(row['delta_IQR_ratio_K64_minus_K32']):+.4g} | "
            f"{float(row['delta_central_68_width_ratio_K64_minus_K32']):+.4g} | "
            f"{float(row['delta_q95_tail_ratio_K64_minus_K32']):+.4g} |"
        )
    lines.append("")
    lines.append(
        "mttbar and top-pT posterior migration matrix changes are small at the matrix level: "
        + "; ".join(
            f"{name} max cell change {record['max_abs_cell_change_percentage_points']:.3f} percentage points, "
            f"mean truth-bin TV {record['truth_bin_total_variation_mean']:.4f}"
            for name, record in migration.items()
        )
        + "."
    )
    lines.append(
        "For the 68% coverage stability check, the finite-draw rank expectations are "
        f"{float(convergence_rows[0]['coverage_68_finite_rank_expectation_K32']):.4f} at K=32 and "
        f"{float(convergence_rows[0]['coverage_68_finite_rank_expectation_K64']):.4f} at K=64; "
        "coverage is compared with its finite-K expectation because the central order-statistic ranks differ with K."
    )
    lines.append(
        "The audited distributional summaries retain the same physical interpretation at K=32 and K=64; K=64 is the publication default for the primary J posterior. This is a finite-sample stability check, not a requirement that the empirical distributions be identical."
        if k64_is_publication_default
        else "At least one audited distributional conclusion changes between K=32 and K=64; K=64 is not designated as a publication default by this comparison."
    )
    lines.extend(
        [
            "",
            "## One-draw pseudo-dataset variability",
            "",
            f"The {draw_summary['pseudo_dataset_count']} existing event-keyed pseudo-datasets use seeds {draw_summary['seed_min']}–{draw_summary['seed_max']}. The displayed one-draw selection uses seed {draw_summary['headline_seed']}; its empirical percentile is reported against that ensemble.",
            "",
            "| Metric | Mean | Median | SD | 16–84% | 2.5–97.5% | Displayed draw | Ensemble percentile |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in ("wasserstein_1", "ks_distance", "iqr_ratio", "q95_tail_fraction_ratio"):
        row = draw_summary["metrics"][metric]
        label = {
            "wasserstein_1": "W1 [GeV]",
            "ks_distance": "KS",
            "iqr_ratio": "IQR ratio",
            "q95_tail_fraction_ratio": "q95 tail ratio",
        }[metric]
        lines.append(
            f"| {label} | {row['mean']:.4g} | {row['median']:.4g} | {row['standard_deviation']:.4g} | "
            f"[{row['p16']:.4g}, {row['p84']:.4g}] | [{row['p2_5']:.4g}, {row['p97_5']:.4g}] | "
            f"{row['headline_value']:.4g} | {row['headline_percentile']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "The deterministic draw is positioned by its empirical mid-rank among the 128 pseudo-dataset metric values. The ensemble demonstrates the spread expected from choosing one coherent sample/event; it does not change the separate point-estimator comparison.",
            "",
            "## Nonlinear observable check",
            "",
            f"For mttbar, the mean absolute eventwise difference between E[mttbar|x] and mttbar(E[parent four-vector|x]) is {mean_of_observable_check['mean_absolute_difference_GeV']:.4g} GeV (mean signed difference {mean_of_observable_check['mean_signed_difference_GeV']:.4g} GeV). The analysis uses E[O], as required.",
            "",
            "## Principal plots",
            "",
            "- `plots/core/mttbar_shape.png` and `plots/core/mttbar_residual.png`",
            "- `plots/convergence/mttbar_k32_k64_posterior.png` and `plots/convergence/top_pt_k32_k64_posterior.png`",
            "- `plots/core/mttbar_one_draw_metric_ensemble.png`",
            "- `plots/migration/mttbar_j_migration_comparison.png`",
            "",
            "Numerical details are in `tables/posterior_usage_scorecard.csv`, `tables/k32_k64_convergence.csv`, `tables/one_draw_metric_ensemble.csv`, and the corresponding `metadata/*.json` files.",
        ]
    )
    (output / "POSTERIOR_USAGE_SUMMARY.md").write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    k32_path = args.k32.resolve()
    k64_path = args.k64.resolve()
    output = args.output.resolve()
    if not output.is_dir():
        raise FileNotFoundError(f"Expected existing K64 posterior-usage output directory: {output}")
    for path in (k32_path, k64_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required KNOT posterior is missing: {path}")

    k32 = usage._load_posterior(k32_path, expected_model="J", include_spin=True)
    k64 = usage._load_posterior(k64_path, expected_model="J", include_spin=True)
    if k32["sample_count"] != 32 or k64["sample_count"] != 64:
        raise ValueError(
            f"Expected J K32/K64 posteriors; found K={k32['sample_count']}/{k64['sample_count']}"
        )
    if not np.array_equal(k32["event_ids"], k64["event_ids"]):
        raise ValueError("K32/K64 EVENT_ID order differs")
    if not np.array_equal(k32["truth_parents"], k64["truth_parents"]):
        raise ValueError("K32/K64 parent truth differs on aligned EVENT_ID rows")
    neutrino_truth_max_difference = usage._assert_neutrino_truth_aligned(
        k64, k32, k32_path
    )
    for data in (k32, k64):
        if bool(data["attrs"].get("sealed_test_used", False)):
            raise ValueError(f"{data['path']}: sealed-test use is declared")
        if str(data["attrs"].get("sample_semantics", "")) != "coherent_event_hypothesis":
            raise ValueError(f"{data['path']}: samples are not coherent event hypotheses")

    manifest32 = _read_json(k32_path.parent / "evaluation_manifest.json")
    manifest64 = _read_json(k64_path.parent / "evaluation_manifest.json")
    for expected_k, manifest in ((32, manifest32), (64, manifest64)):
        if manifest.get("events") != len(k64["event_ids"]):
            raise ValueError(f"K{expected_k} evaluation manifest has a different event count")
        if manifest.get("samples_per_event") != expected_k:
            raise ValueError(f"K{expected_k} evaluation manifest reports the wrong sample count")
        if manifest.get("sealed_test_used") is not False:
            raise ValueError(f"K{expected_k} evaluation manifest does not certify sealed_test_used=false")
    if manifest32.get("checkpoint") != manifest64.get("checkpoint"):
        raise ValueError("K32 and K64 do not name the same frozen J checkpoint")

    output_input_manifest = _read_json(output / "metadata/input_manifest.json")
    output_draw_manifest = _read_json(output / "metadata/draw_selection.json")
    closure_manifest = _read_json(output / "metadata/one_draw_ensemble_closure.json")
    if output_input_manifest.get("posterior_samples_per_event") != 64:
        raise ValueError("Canonical posterior-usage output is not based on K64")
    if output_draw_manifest.get("posterior_samples_per_event") != 64:
        raise ValueError("Canonical draw metadata is not based on K64")
    event_ids = k64["event_ids"]
    n_events = len(event_ids)
    draw_seed = int(output_draw_manifest["draw_seed"])
    pseudo_seeds = [int(value) for value in output_draw_manifest["pseudo_dataset_seeds"]]
    if len(pseudo_seeds) != int(output_draw_manifest["pseudo_dataset_count"]):
        raise ValueError("Existing pseudo-dataset seed list does not match its recorded ensemble count")
    if closure_manifest.get("pseudo_dataset_seeds") != pseudo_seeds:
        raise ValueError("One-draw closure and draw metadata contain different pseudo-dataset seeds")
    if int(closure_manifest.get("pseudo_datasets", -1)) != len(pseudo_seeds):
        raise ValueError("One-draw closure and draw metadata contain different ensemble sizes")

    primary_j = output_input_manifest["models"]["J"]
    if Path(primary_j["path"]).resolve() != k64_path:
        raise ValueError("Canonical K64 plotting manifest points to a different J posterior")
    expected_manifests = [
        ("primary", model, record)
        for model, record in output_input_manifest["models"].items()
    ] + [
        ("factorisation", model, record)
        for model, record in output_input_manifest["factorisation_models"].items()
    ]
    for family, model, record in expected_manifests:
        path = Path(record["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Posterior-usage input for model {model} is missing: {path}")
        if path.stat().st_size != int(record["size_bytes"]) or _sha256(path) != record["sha256"]:
            raise ValueError(f"Posterior-usage input hash changed for {model}: {path}")
        expected_samples = 64 if family == "primary" and model == "J" else 32
        if int(record["posterior_samples_per_event"]) != expected_samples:
            raise ValueError(
                f"Posterior-usage input {family}/{model} does not use K={expected_samples}"
            )

    choices = usage.event_keyed_indices(event_ids, 64, seed=draw_seed, repeats=1)[0]
    draw_hash = usage._draw_choice_hash(event_ids, choices)
    if draw_hash != output_draw_manifest["draw_selection_sha256_order_independent_event_id_index_pairs"]:
        raise ValueError("K64 deterministic one-draw hash does not reproduce the saved selection")

    parent_values32 = usage._parent_observables(k32["parent_samples"], "KNOT-J K32")
    parent_values64 = usage._parent_observables(k64["parent_samples"], "KNOT-J K64")
    neutrino_values32 = usage._neutrino_observables(k32["neutrino_samples"], "KNOT-J K32")
    neutrino_values64 = usage._neutrino_observables(k64["neutrino_samples"], "KNOT-J K64")
    truth_parents = usage._parent_observables(k64["truth_parents"], "KNOT-J K64 truth")
    truth_neutrinos = usage._neutrino_observables(
        k64["truth_neutrinos"], "KNOT-J K64 truth"
    )
    observables = {
        **usage.PARENT_OBSERVABLES,
        **usage.NEUTRINO_OBSERVABLES,
    }
    values32 = {**parent_values32, **neutrino_values32}
    values64 = {**parent_values64, **neutrino_values64}
    truth_values = {**truth_parents, **truth_neutrinos}

    rows: list[dict[str, Any]] = []
    convergence: dict[str, Any] = {}
    migration_details: dict[str, Any] = {}
    for name in (
        "mttbar",
        "ptttbar",
        "yttbar",
        "top_pt",
        "neutrino_pt",
        "neutrino_pz",
    ):
        truth = truth_values[name]
        samples32, samples64 = values32[name], values64[name]
        metrics32 = usage._spectrum_metrics(
            truth, samples32, representation="posterior_distribution"
        )
        metrics64 = usage._spectrum_metrics(
            truth, samples64, representation="posterior_distribution"
        )
        flat32, weights32 = usage._flatten_population(
            samples32, truth, representation="posterior_distribution"
        )
        flat64, weights64 = usage._flatten_population(
            samples64, truth, representation="posterior_distribution"
        )
        cover32 = _coverage_by_level(samples32, truth, name)
        cover64 = _coverage_by_level(samples64, truth, name)
        record: dict[str, Any] = {
            "observable": name,
            "unit": observables[name]["unit"],
            "K32_K64_W1": float(
                usage.wasserstein_distance(
                    flat32, flat64, u_weights=weights32, v_weights=weights64
                )
            ),
            "K32_K64_KS": usage._weighted_ks_distance(
                flat32, weights32, flat64, weights64
            ),
            "truth_W1_K32": metrics32["wasserstein_1"],
            "truth_W1_K64": metrics64["wasserstein_1"],
            "delta_truth_W1_K64_minus_K32": metrics64["wasserstein_1"]
            - metrics32["wasserstein_1"],
            "truth_KS_K32": metrics32["ks_distance"],
            "truth_KS_K64": metrics64["ks_distance"],
            "delta_truth_KS_K64_minus_K32": metrics64["ks_distance"]
            - metrics32["ks_distance"],
        }
        for metric, output_name in (
            ("iqr_ratio", "IQR_ratio"),
            ("central_68_width_ratio", "central_68_width_ratio"),
            ("q90_tail_fraction_ratio", "q90_tail_ratio"),
            ("q95_tail_fraction_ratio", "q95_tail_ratio"),
            ("q99_tail_fraction_ratio", "q99_tail_ratio"),
        ):
            record[f"{output_name}_K32"] = metrics32[metric]
            record[f"{output_name}_K64"] = metrics64[metric]
            record[f"delta_{output_name}_K64_minus_K32"] = metrics64[metric] - metrics32[metric]
        for level, label in ((0.50, "50"), (0.68, "68"), (0.90, "90")):
            coverage32 = cover32[level]["empirical_coverage"]
            coverage64 = cover64[level]["empirical_coverage"]
            record[f"coverage_{label}_K32"] = coverage32
            record[f"coverage_{label}_K64"] = coverage64
            record[f"delta_coverage_{label}_K64_minus_K32"] = coverage64 - coverage32
            if level == 0.68:
                rank_expectation32 = cover32[level]["finite_draw_rank_expectation"]
                rank_expectation64 = cover64[level]["finite_draw_rank_expectation"]
                record["coverage_68_finite_rank_expectation_K32"] = rank_expectation32
                record["coverage_68_finite_rank_expectation_K64"] = rank_expectation64
                record["coverage_68_minus_finite_rank_expectation_K32"] = (
                    coverage32 - rank_expectation32
                )
                record["coverage_68_minus_finite_rank_expectation_K64"] = (
                    coverage64 - rank_expectation64
                )
        migration: dict[str, Any] | None = None
        if name in {"mttbar", "top_pt"}:
            migration = _migration_comparison(
                name,
                truth,
                samples32,
                samples64,
                np.asarray(observables[name]["migration_edges"], dtype=np.float64),
            )
            migration_details[name] = migration
            record.update(
                {
                    "migration_max_abs_cell_change_percentage_points": migration[
                        "max_abs_cell_change_percentage_points"
                    ],
                    "migration_mean_abs_cell_change_percentage_points": migration[
                        "mean_abs_cell_change_percentage_points"
                    ],
                    "migration_frobenius_norm_percentage_points": migration[
                        "frobenius_norm_percentage_points"
                    ],
                    "migration_mean_truth_bin_TV": migration[
                        "truth_bin_total_variation_mean"
                    ],
                    "migration_max_truth_bin_TV": migration[
                        "truth_bin_total_variation_max"
                    ],
                }
            )
        else:
            for column in (
                "migration_max_abs_cell_change_percentage_points",
                "migration_mean_abs_cell_change_percentage_points",
                "migration_frobenius_norm_percentage_points",
                "migration_mean_truth_bin_TV",
                "migration_max_truth_bin_TV",
            ):
                record[column] = ""
        rows.append(record)
        convergence[name] = {
            "full_posterior_metrics_vs_truth_K32": metrics32,
            "full_posterior_metrics_vs_truth_K64": metrics64,
            "full_posterior_K32_vs_K64_W1": record["K32_K64_W1"],
            "full_posterior_K32_vs_K64_KS": record["K32_K64_KS"],
            "empirical_coverage_K32": {str(key): value for key, value in cover32.items()},
            "empirical_coverage_K64": {str(key): value for key, value in cover64.items()},
            "migration": migration,
        }

    _write_csv(output / "tables/k32_k64_convergence.csv", rows)

    plot_paths: list[Path] = []
    for observable in ("mttbar", "top_pt"):
        group = usage.PARENT_OBSERVABLES[observable]
        plot_paths.extend(
            _plot_spectrum_convergence(
                output,
                observable,
                group,
                truth_values[observable],
                values32[observable],
                values64[observable],
            )
        )

    mttbar_truth = truth_values["mttbar"]
    mttbar_samples = values64["mttbar"]
    headline_draw = mttbar_samples[np.arange(n_events), choices]
    headline_metrics = usage._spectrum_metrics(
        mttbar_truth, headline_draw, representation="posterior_mean"
    )
    pseudo_values: dict[str, list[float]] = {metric: [] for metric in DRAW_METRICS}
    pseudo_rows: list[dict[str, Any]] = []
    for index, seed in enumerate(pseudo_seeds):
        pseudo_choices = usage.event_keyed_indices(event_ids, 64, seed=seed, repeats=1)[0]
        pseudo_draw = mttbar_samples[np.arange(n_events), pseudo_choices]
        metrics = usage._spectrum_metrics(
            mttbar_truth, pseudo_draw, representation="posterior_mean"
        )
        for metric in DRAW_METRICS:
            value = float(metrics[metric])
            pseudo_values[metric].append(value)
            pseudo_rows.append(
                {
                    "record_type": "pseudo_dataset",
                    "dataset_index": index,
                    "seed": seed,
                    "metric": metric,
                    "value": value,
                }
            )

    headline_percentiles: dict[str, float] = {}
    draw_summary: dict[str, Any] = {
        "events": n_events,
        "posterior_samples_per_event": 64,
        "pseudo_dataset_count": len(pseudo_seeds),
        "pseudo_dataset_seeds": pseudo_seeds,
        "seed_min": min(pseudo_seeds),
        "seed_max": max(pseudo_seeds),
        "headline_seed": draw_seed,
        "headline_draw_selection_sha256": draw_hash,
        "headline_percentile_method": "empirical mid-rank among the repeated pseudo-dataset metric values",
        "metrics": {},
    }
    summary_rows = []
    for metric in DRAW_METRICS:
        values = np.asarray(pseudo_values[metric], dtype=np.float64)
        headline = float(headline_metrics[metric])
        percentile = 100.0 * (
            np.mean(values < headline) + 0.5 * np.mean(values == headline)
        )
        headline_percentiles[metric] = float(percentile)
        summary = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "p16": float(np.percentile(values, 16)),
            "p84": float(np.percentile(values, 84)),
            "p2_5": float(np.percentile(values, 2.5)),
            "p97_5": float(np.percentile(values, 97.5)),
            "headline_value": headline,
            "headline_percentile": float(percentile),
        }
        draw_summary["metrics"][metric] = summary
        summary_rows.append(
            {
                "record_type": "ensemble_summary",
                "dataset_index": "",
                "seed": "",
                "metric": metric,
                "value": "",
                **summary,
            }
        )
        pseudo_rows.append(
            {
                "record_type": "deterministic_headline",
                "dataset_index": "",
                "seed": draw_seed,
                "metric": metric,
                "value": headline,
                "headline_percentile": float(percentile),
            }
        )
    _write_csv(
        output / "tables/one_draw_metric_ensemble.csv", pseudo_rows + summary_rows
    )
    _write_json(output / "metadata/one_draw_metric_ensemble.json", draw_summary)
    plot_paths.extend(
        _plot_one_draw_metrics(
            output, {key: np.asarray(value) for key, value in pseudo_values.items()},
            headline_metrics, headline_percentiles,
        )
    )

    scorecard_rows, posterior_quality = _scorecard(output)
    _write_csv(output / "tables/posterior_usage_scorecard.csv", scorecard_rows)

    sample_prefix_exact = bool(
        np.array_equal(k32["sample_index"], k64["sample_index"][:, :32])
        and np.array_equal(k32["parent_samples"], k64["parent_samples"][:, :32])
        and np.array_equal(k32["neutrino_samples"], k64["neutrino_samples"][:, :32])
    )
    sample_index_prefix_matches = bool(
        np.array_equal(k32["sample_index"], k64["sample_index"][:, :32])
    )
    parent_sample_prefix_exact = bool(
        np.array_equal(k32["parent_samples"], k64["parent_samples"][:, :32])
    )
    neutrino_sample_prefix_exact = bool(
        np.array_equal(k32["neutrino_samples"], k64["neutrino_samples"][:, :32])
    )
    mean_observable = np.mean(parent_values64["mttbar"], axis=1, dtype=np.float64)
    parent_mean = np.mean(k64["parent_samples"], axis=1, dtype=np.float64)
    observable_of_parent_mean = usage._parent_observables(
        parent_mean, "KNOT-J K64 posterior mean parent four-vector"
    )["mttbar"]
    observable_difference = mean_observable - observable_of_parent_mean
    mean_of_observable_check = {
        "mean_signed_difference_GeV": float(np.mean(observable_difference)),
        "mean_absolute_difference_GeV": float(np.mean(np.abs(observable_difference))),
        "rms_difference_GeV": float(np.sqrt(np.mean(np.square(observable_difference)))),
        "definition": "per-event E[mttbar|x] minus mttbar(E[parent four-vector|x])",
    }

    qualitative_checks = {}
    for row in rows:
        name = row["observable"]
        qualitative_checks[name] = {
            "central_68_width_side_of_truth_unchanged": bool(
                np.sign(float(row["central_68_width_ratio_K32"]) - 1.0)
                == np.sign(float(row["central_68_width_ratio_K64"]) - 1.0)
            ),
            "q95_tail_side_of_truth_unchanged": bool(
                np.sign(float(row["q95_tail_ratio_K32"]) - 1.0)
                == np.sign(float(row["q95_tail_ratio_K64"]) - 1.0)
            ),
            "coverage_68_side_of_finite_draw_expectation_unchanged": bool(
                np.sign(float(row["coverage_68_minus_finite_rank_expectation_K32"]))
                == np.sign(float(row["coverage_68_minus_finite_rank_expectation_K64"]))
            ),
        }
    k64_is_publication_default = all(
        all(checks.values()) for checks in qualitative_checks.values()
    )
    convergence_metadata = {
        "comparison": "KNOT-J K32 versus K64 using the same frozen checkpoint and aligned validation event population",
        "events": n_events,
        "posterior_samples_per_event": {"K32": 32, "K64": 64},
        "checkpoint": manifest64["checkpoint"],
        "evaluation_seeds": {
            "K32": manifest32.get("evaluation_seed"),
            "K64": manifest64.get("evaluation_seed"),
        },
        "inference_devices": {
            "K32": manifest32.get("device"),
            "K64": manifest64.get("device"),
        },
        "sealed_test_used": False,
        "event_id_sha256_ordered_little_endian_int64": usage._event_id_hash(event_ids),
        "K32_K64_parent_truth_exact": True,
        "K32_K64_neutrino_truth_max_abs_difference_GeV": neutrino_truth_max_difference,
        "K32_is_exact_prefix_of_K64_samples": sample_prefix_exact,
        "sample_index_ordinal_prefix_matches": sample_index_prefix_matches,
        "parent_sample_values_exact_prefix": parent_sample_prefix_exact,
        "neutrino_sample_values_exact_prefix": neutrino_sample_prefix_exact,
        "sample_sequence_relationship": (
            "K32 and K64 sample values are non-nested CPU/GPU Monte Carlo draws. The sample_index arrays share ordinal indexes but do not identify equal posterior hypotheses across devices."
        ),
        "posterior_mean_definition": POSTERIOR_MEAN_DEFINITION,
        "full_posterior_weights": "w_ik = w_i/K; total posterior weight is one per event",
        "observable_results": convergence,
        "qualitative_checks": qualitative_checks,
        "publication_default_decision": {
            "K64_publication_default": k64_is_publication_default,
            "criterion": "The side of truth for central-68 width and q95 tail ratio, and the side of the finite-draw rank expectation for 68 percent coverage, are unchanged for every audited observable.",
            "interpretation": "Empirical K32 and K64 distributions are not expected to be exactly identical; distances and finite-sample metric shifts are reported explicitly.",
        },
        "input_files": {
            "K32": {"path": str(k32_path), "size_bytes": k32_path.stat().st_size, "sha256": _sha256(k32_path)},
            "K64": {"path": str(k64_path), "size_bytes": k64_path.stat().st_size, "sha256": _sha256(k64_path)},
        },
        "plots": [str(path) for path in plot_paths[:4]],
        "migration_metrics": migration_details,
    }
    _write_json(output / "metadata/k32_k64_convergence.json", convergence_metadata)

    plotting_specification = _read_json(output / "metadata/plotting_specification.json")
    plotting_specification["posterior_mean_definition"] = POSTERIOR_MEAN_DEFINITION
    plotting_specification["posterior_mean_detail"] = (
        "E[O|x] = (1/K) sum_k O(y^(k)); nonlinear observables are evaluated on each coherent posterior sample, not on E[y|x]."
    )
    plotting_specification["posterior_sample_count_convergence"] = {
        "table": "tables/k32_k64_convergence.csv",
        "metadata": "metadata/k32_k64_convergence.json",
        "publication_default": "K64" if k64_is_publication_default else "not established",
    }
    plotting_specification["one_draw_metric_ensemble"] = {
        "table": "tables/one_draw_metric_ensemble.csv",
        "metadata": "metadata/one_draw_metric_ensemble.json",
        "pseudo_dataset_count": len(pseudo_seeds),
        "draw_seed": draw_seed,
    }
    all_plot_paths = list(plotting_specification.get("plot_files", []))
    all_plot_paths.extend(str(path) for path in plot_paths)
    plotting_specification["plot_files"] = list(dict.fromkeys(all_plot_paths))
    _write_json(output / "metadata/plotting_specification.json", plotting_specification)

    _summary_markdown(
        output,
        scorecard_rows,
        posterior_quality,
        rows,
        migration_details,
        draw_summary,
        mean_of_observable_check,
        k64_is_publication_default,
    )

    return {
        "output": str(output),
        "events": n_events,
        "K32": 32,
        "K64": 64,
        "pseudo_datasets": len(pseudo_seeds),
        "draw_selection_sha256": draw_hash,
        "K64_publication_default": k64_is_publication_default,
        "outputs": [
            "tables/k32_k64_convergence.csv",
            "metadata/k32_k64_convergence.json",
            "tables/one_draw_metric_ensemble.csv",
            "metadata/one_draw_metric_ensemble.json",
            "tables/posterior_usage_scorecard.csv",
            "POSTERIOR_USAGE_SUMMARY.md",
        ],
        "mttbar_one_draw_metrics": draw_summary["metrics"],
        "mttbar_EO_minus_OEymean": mean_of_observable_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k32", type=Path, default=DEFAULT_K32)
    parser.add_argument("--k64", type=Path, default=DEFAULT_K64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
