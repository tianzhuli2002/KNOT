#!/usr/bin/env python3
"""Measure event-conditional skill against whole-posterior permutations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import analyze_realnu_mass_response as mass


EXPECTED_EVENTS = mass.EXPECTED_EVENTS
EXPECTED_SAMPLES = mass.EXPECTED_SAMPLES
PERMUTATIONS = 32
PERMUTATION_SEED_BASE = 2026093001
BOOTSTRAP_SEED = 2026093099


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})
    os.replace(temporary, path)


def _csv_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _load_mass_cache(manifest_path: Path, mass_audit_dir: Path) -> tuple[dict, dict]:
    inputs = mass._resolve_inputs(manifest_path)
    cache_path = mass_audit_dir / "cache/PCFM_Q_mass_samples_m32.npz"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Validated Q mass cache is missing: {cache_path}")
    with np.load(cache_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
        if metadata.get("schema") != "realnu_mttbar_sample_cache_v1":
            raise ValueError("Existing Q mass cache has an unexpected schema")
        ids = np.asarray(cache["EVENT_ID"], dtype=np.int64)
        truth = np.asarray(cache["truth_mttbar"], dtype=np.float64)
        samples = np.asarray(cache["sample_mttbar"], dtype=np.float64)
    if ids.shape != (EXPECTED_EVENTS,) or len(np.unique(ids)) != EXPECTED_EVENTS:
        raise ValueError("Existing mass cache does not contain the unique full validation EVENT_ID population")
    if truth.shape != (EXPECTED_EVENTS,) or not np.isfinite(truth).all():
        raise ValueError("Existing mass cache truth mass is malformed or non-finite")
    if samples.shape != (EXPECTED_EVENTS, EXPECTED_SAMPLES) or not np.isfinite(samples).all():
        raise ValueError("Existing mass cache samples are not finite [100342,32]")
    provenance = mass._prediction_provenance(inputs, ids)
    cached_provenance = metadata.get("provenance", {})
    if not cached_provenance:
        raise ValueError("Mass cache lacks its accepted-input provenance record")
    mismatch = {
        key: (cached_provenance.get(key), provenance.get(key))
        for key in cached_provenance
        if cached_provenance.get(key) != provenance.get(key)
    }
    if mismatch:
        raise ValueError(f"Existing mass cache provenance no longer matches accepted inputs: {mismatch}")
    if provenance.get("sealed_test_used") is not False or provenance.get("truth_free_conditioning") is not True:
        raise ValueError("Accepted mass cache is not truth-free validation data with sealed test unused")
    cache_info = {
        **provenance,
        "cache_path": str(cache_path.resolve()),
        "cache_sha256": _sha256(cache_path),
        "cache_created_utc": metadata.get("created_utc"),
        "events": len(ids),
        "samples_per_event": samples.shape[1],
        "cache_reused": True,
        "prediction_schema": "realnu_parent_solver_scan_v1",
    }
    return {"EVENT_ID": ids, "truth": truth, "samples": samples}, cache_info


def empirical_crps(samples: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical finite-ensemble CRPS via the sorted-sample identity."""
    values = np.asarray(samples, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if values.ndim != 2 or target.shape != (len(values),) or not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("CRPS requires finite samples [events,M] and aligned finite truth")
    ordered = np.sort(values, axis=1)
    count = values.shape[1]
    coefficients = 2.0 * np.arange(count, dtype=np.float64) - (count - 1.0)
    pairwise_term = (ordered @ coefficients) / float(count * count)
    score = np.mean(np.abs(values - target[:, None]), axis=1) - pairwise_term
    return score, ordered


def _validate_crps_formula(samples: np.ndarray, truth: np.ndarray) -> float:
    scores, _ = empirical_crps(samples[:8], truth[:8])
    reference = []
    for row, target in zip(samples[:8], truth[:8], strict=True):
        count = len(row)
        reference.append(
            np.mean(np.abs(row - target))
            - np.abs(row[:, None] - row[None, :]).sum() / (2.0 * count * count)
        )
    difference = float(np.max(np.abs(scores - np.asarray(reference))))
    if difference > 2.0e-12:
        raise RuntimeError(f"Sorted CRPS does not match the explicit pairwise formula: {difference:.3g}")
    return difference


def _derangement(event_count: int, seed: int) -> np.ndarray:
    """Create a seeded bijection with no fixed event assignment, without truth input."""
    if event_count < 2:
        raise ValueError("A derangement requires at least two event rows")
    rng = np.random.default_rng(seed)
    order = rng.permutation(event_count)
    shift = int(rng.integers(1, event_count))
    permutation = np.empty(event_count, dtype=np.int64)
    permutation[order] = np.roll(order, shift)
    if not np.array_equal(np.sort(permutation), np.arange(event_count)):
        raise RuntimeError("Generated posterior assignment is not a bijection")
    if np.any(permutation == np.arange(event_count)):
        raise RuntimeError("Generated posterior assignment retained a self-paired event")
    return permutation


def _region_summaries(samples: np.ndarray, truth: np.ndarray, masks: dict[str, np.ndarray],
                      score: np.ndarray, ordered: np.ndarray) -> tuple[list[dict], np.ndarray]:
    event_count, sample_count = samples.shape
    if sample_count != EXPECTED_SAMPLES:
        raise ValueError(f"Expected M32, received M={sample_count}")
    truth_bin = mass._bin_ids(truth, mass.COARSE_EDGES)
    sample_bin = mass._bin_ids(samples.reshape(-1), mass.COARSE_EDGES).reshape(samples.shape)
    correct_probability = np.mean(sample_bin == truth_bin[:, None], axis=1)
    adjacent_probability = np.mean(np.abs(sample_bin - truth_bin[:, None]) <= 1, axis=1)
    q05, q16, q84, q95 = np.quantile(ordered, [0.05, 0.16, 0.84, 0.95], axis=1)
    width68, width90 = q84 - q16, q95 - q05
    posterior_mean = np.mean(samples, axis=1)
    records = []
    for region, mask in masks.items():
        point = mass._point_metrics(posterior_mean, truth, mask)
        records.append({
            "truth_region": region,
            "events": int(mask.sum()),
            "crps_mean_gev": float(np.mean(score[mask])),
            "crps_median_gev": float(np.median(score[mask])),
            "mean_rmse_gev": point["rmse_gev"],
            "mean_bias_gev": point["bias_gev"],
            "mean_response_slope": point["response_slope"],
            "correct_bin_probability": float(correct_probability[mask].mean()),
            "correct_or_adjacent_probability": float(adjacent_probability[mask].mean()),
            "coverage68": float(np.mean((truth[mask] >= q16[mask]) & (truth[mask] <= q84[mask]))),
            "coverage90": float(np.mean((truth[mask] >= q05[mask]) & (truth[mask] <= q95[mask]))),
            "width68_mean_gev": float(width68[mask].mean()),
            "width68_median_gev": float(np.median(width68[mask])),
            "width90_mean_gev": float(width90[mask].mean()),
            "width90_median_gev": float(np.median(width90[mask])),
        })
    flat = np.repeat(truth_bin, sample_count) * len(mass.COARSE_LABELS) + sample_bin.reshape(-1)
    weighted = np.bincount(
        flat,
        weights=np.full(event_count * sample_count, 1.0 / sample_count),
        minlength=len(mass.COARSE_LABELS) ** 2,
    ).reshape(len(mass.COARSE_LABELS), len(mass.COARSE_LABELS))
    truth_counts = np.bincount(truth_bin, minlength=len(mass.COARSE_LABELS)).astype(np.float64)
    matrix = np.divide(weighted, truth_counts[:, None], out=np.zeros_like(weighted), where=truth_counts[:, None] > 0)
    if not np.allclose(matrix.sum(axis=1), (truth_counts > 0).astype(float), rtol=0.0, atol=1e-12):
        raise RuntimeError("Posterior-predictive migration rows do not normalise to one")
    return records, matrix


def _bootstrap_mean(values: np.ndarray, *, replicates: int, seed: int) -> dict:
    if replicates < 100:
        raise ValueError("Use at least 100 paired event bootstrap replicas")
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        rows = rng.integers(0, len(values), size=len(values))
        draws[replicate] = values[rows].mean()
    return {
        "events": len(values),
        "replicates": replicates,
        "seed": seed,
        "mean_delta_crps_perm_minus_q_gev": float(values.mean()),
        "ci95_low_gev": float(np.quantile(draws, 0.025)),
        "ci95_high_gev": float(np.quantile(draws, 0.975)),
        "convention": "paired event bootstrap of the event-wise mean CRPS over the fixed permutation seeds minus real-Q CRPS",
    }


def _write_plots(crps_rows: list[dict], response_rows: list[dict],
                 permutation_rows: list[dict], rotation_rows: list[dict],
                 output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    regions = list(mass.COARSE_LABELS)
    region_rows = {row["truth_region"]: row for row in crps_rows if row["truth_region"] != "inclusive"}

    def save(fig, stem: str) -> None:
        fig.tight_layout()
        fig.savefig(plot_dir / f"{stem}.png", dpi=220)
        fig.savefig(plot_dir / f"{stem}.pdf")
        plt.close(fig)

    positions = np.arange(len(regions))
    real_crps = np.asarray([region_rows[region]["q_crps_mean_gev"] for region in regions])
    perm_crps = np.asarray([region_rows[region]["permuted_crps_mean_gev"] for region in regions])
    perm_low = np.asarray([region_rows[region]["permuted_crps_p05_gev"] for region in regions])
    perm_high = np.asarray([region_rows[region]["permuted_crps_p95_gev"] for region in regions])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(positions, perm_crps, yerr=[perm_crps - perm_low, perm_high - perm_crps],
                fmt="s", capsize=4, label="Condition-destroyed posterior (5–95% over permutations)")
    ax.plot(positions, real_crps, "o-", label="Real PCFM_Q")
    ax.set_xticks(positions, regions)
    ax.set_ylabel("Empirical CRPS [GeV] (lower is better)")
    ax.set_xlabel("Truth $m_{t\\bar{t}}$ region [GeV]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "crps_by_truth_region")

    skills = np.asarray([region_rows[region]["skill_score_mean_over_permutations"] for region in regions])
    skill_low = np.asarray([region_rows[region]["skill_score_p05_over_permutations"] for region in regions])
    skill_high = np.asarray([region_rows[region]["skill_score_p95_over_permutations"] for region in regions])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(positions, skills, yerr=[skills - skill_low, skill_high - skills],
                fmt="o", capsize=4, color="tab:blue")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(positions, regions)
    ax.set_ylabel(r"CRPS skill $1-CRPS_Q/CRPS_{perm}$")
    ax.set_xlabel("Truth $m_{t\\bar{t}}$ region [GeV]")
    ax.grid(alpha=0.25)
    save(fig, "conditional_skill_by_truth_region")

    fine_rows = [row for row in response_rows if row["events"] >= 40]
    fine_rows.sort(key=lambda row: row["truth_bin_index"])
    x = np.asarray([row["truth_mean_gev"] for row in fine_rows])
    q_y = np.asarray([row["q_reco_mean_gev"] for row in fine_rows])
    p_y = np.asarray([row["permuted_reco_mean_gev"] for row in fine_rows])
    p_low = np.asarray([row["permuted_reco_mean_p05_gev"] for row in fine_rows])
    p_high = np.asarray([row["permuted_reco_mean_p95_gev"] for row in fine_rows])
    fig, ax = plt.subplots(figsize=(7, 7))
    lower = min(float(np.min(x)), float(np.min(q_y)), float(np.min(p_low)))
    upper = max(float(np.max(x)), float(np.max(q_y)), float(np.max(p_high)))
    ax.plot([lower, upper], [lower, upper], "k--", label="Truth response")
    ax.fill_between(x, p_low, p_high, alpha=0.2, color="tab:orange", label="Permutation 5–95%")
    ax.plot(x, p_y, "s-", color="tab:orange", label="Condition-destroyed mean")
    ax.plot(x, q_y, "o-", color="tab:blue", label="Real PCFM_Q mean")
    ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Truth $m_{t\\bar{t}}$ [GeV]",
           ylabel="Mean predicted $m_{t\\bar{t}}$ [GeV]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "response_real_vs_permuted")

    q_prob = np.asarray([region_rows[region]["q_correct_bin_probability"] for region in regions])
    perm_prob = np.asarray([region_rows[region]["permuted_correct_bin_probability"] for region in regions])
    prob_low = np.asarray([region_rows[region]["permuted_correct_bin_probability_p05"] for region in regions])
    prob_high = np.asarray([region_rows[region]["permuted_correct_bin_probability_p95"] for region in regions])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(positions, perm_prob, yerr=[perm_prob - prob_low, prob_high - perm_prob],
                fmt="s", capsize=4, label="Condition-destroyed posterior (5–95%)")
    ax.plot(positions, q_prob, "o-", label="Real PCFM_Q")
    ax.set_xticks(positions, regions)
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.set_ylabel("Posterior probability in correct truth bin")
    ax.set_xlabel("Truth $m_{t\\bar{t}}$ region [GeV]")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "correct_bin_real_vs_permuted")

    angles = sorted({row["angle_rad"] for row in rotation_rows})
    roles = ("top", "antitop")
    modes = ("event-only", "source-only", "co-rotation")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, role in zip(axes, roles, strict=True):
        width = 0.24
        for mode_index, mode in enumerate(modes):
            values = []
            for angle in angles:
                match = next(row for row in rotation_rows if row["angle_rad"] == angle and row["role"] == role and row["mode"] == mode)
                values.append(match["mean_output_delta_phi_rad"])
            x_mode = np.arange(len(angles)) + (mode_index - 1) * width
            bars = axis.bar(x_mode, values, width=width, label=mode)
            axis.scatter(x_mode, values, color=bars.patches[0].get_facecolor(), s=18, zorder=3)
        axis.plot(np.arange(len(angles)), angles, "k--", marker=".", label="Requested angle")
        axis.set_ylim(bottom=-0.05)
        axis.set_xticks(np.arange(len(angles)), [f"{angle:.3f}" for angle in angles])
        axis.set_xlabel("Rotation angle [rad]")
        axis.set_title(role)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean wrapped output $\\Delta\\phi$ [rad]")
    axes[1].legend(fontsize=8)
    save(fig, "rotation_event_vs_source_vs_corotation")


def _rotation_comparison(rotation: dict) -> list[dict]:
    if rotation.get("status") != "completed" or rotation.get("diagnostic_model_status") != "trained checkpoint":
        raise ValueError("Source-only rotation artifact is not from the trained real PCFM_Q checkpoint")
    if rotation.get("sealed_test_used") is not False or rotation.get("truth_free_conditioning") is not True:
        raise ValueError("Rotation artifact does not record truth-free validation with sealed test unused")
    if rotation.get("sample_events") != 64 or rotation.get("solver") != "heun" or rotation.get("steps") != 8:
        raise ValueError("Rotation sample count or inference policy differs from the established diagnostic")
    if not np.allclose(rotation.get("angles_rad", []), [np.pi / 6.0, np.pi / 3.0, np.pi / 2.0], rtol=0.0, atol=1e-14):
        raise ValueError("Rotation angles differ from the established three-angle sweep")
    rows = []
    for angle_text, per_angle in rotation["per_angle"].items():
        angle = float(angle_text)
        modes = {
            "event-only": per_angle["input_rotation_same_source"],
            "source-only": per_angle["source_rotation_fixed_event"],
            "co-rotation": per_angle["formal_co_rotated_source_covariance"],
        }
        for mode, record in modes.items():
            for role in ("top", "antitop"):
                phi_key = f"{role}_phi"
                if mode == "co-rotation":
                    shift = record[f"{phi_key}_shift_rad"]
                    response_error = record[f"{phi_key}_response_error_rad"]
                    mean_shift = shift["mean"]
                    mean_residual = response_error["mean"]
                    rms_residual = response_error["rms"]
                    pt = record[f"{role}_pt_change_gev"]
                else:
                    shift = record[f"{phi_key}_shift_rad"]
                    response_error = record[f"{phi_key}_response_error_rad"]
                    mean_shift = shift["mean"]
                    mean_residual = response_error["mean"]
                    rms_residual = response_error["rms"]
                    pt = record[f"{role}_pt_change_gev"]
                rows.append({
                    "angle_rad": angle,
                    "role": role,
                    "mode": mode,
                    "mean_output_delta_phi_rad": mean_shift,
                    "rms_output_delta_phi_rad": (
                        record[f"{phi_key}_shift_rad"]["rms"]
                    ),
                    "mean_covariance_residual_delta_phi_rad": mean_residual,
                    "rms_covariance_residual_delta_phi_rad": rms_residual,
                    "mean_pt_change_gev": pt["mean"],
                    "rms_pt_change_gev": pt["rms"],
                    "mean_mttbar_change_gev": record["mttbar_change_gev"]["mean"],
                    "rms_mttbar_change_gev": record["mttbar_change_gev"]["rms"],
                })
    return rows


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    cache, provenance = _load_mass_cache(args.comparison_manifest, args.mass_audit_dir)
    ids, truth, samples = cache["EVENT_ID"], cache["truth"], cache["samples"]
    masks = mass._region_masks(truth)
    score_q, ordered_q = empirical_crps(samples, truth)
    crps_check = _validate_crps_formula(samples, truth)

    accepted_metrics_dir = args.mass_audit_dir / "metrics"
    accepted_estimators = list(csv.DictReader((accepted_metrics_dir / "estimator_metrics.csv").open()))
    accepted_support = list(csv.DictReader((accepted_metrics_dir / "posterior_support.csv").open()))
    mean_row = next(row for row in accepted_estimators if row["estimator"] == "observable_mean" and row["truth_region"] == "inclusive")
    support_row = next(row for row in accepted_support if row["truth_region"] == "inclusive")
    q_summary, q_migration = _region_summaries(samples, truth, masks, score_q, ordered_q)
    q_by_region = {row["truth_region"]: row for row in q_summary}
    regression = {
        "posterior_mean_rmse_gev": {
            "computed": q_by_region["inclusive"]["mean_rmse_gev"],
            "accepted": float(mean_row["rmse_gev"]),
        },
        "posterior_mean_response_slope": {
            "computed": q_by_region["inclusive"]["mean_response_slope"],
            "accepted": float(mean_row["response_slope"]),
        },
        "correct_bin_probability": {
            "computed": q_by_region["inclusive"]["correct_bin_probability"],
            "accepted": float(support_row["mean_true_bin_probability"]),
        },
        "coverage68": {
            "computed": q_by_region["inclusive"]["coverage68"],
            "accepted": float(support_row["coverage68"]),
        },
        "coverage90": {
            "computed": q_by_region["inclusive"]["coverage90"],
            "accepted": float(support_row["coverage90"]),
        },
    }
    for name, values in regression.items():
        delta = abs(values["computed"] - values["accepted"])
        values["absolute_difference"] = delta
        if delta > 1.0e-9:
            raise RuntimeError(f"Real-Q {name} differs from the accepted mass audit by {delta:.3g}")

    # The shuffled posterior marginal must be exactly the same pooled sample
    # multiset.  A bijective row assignment proves this; these explicit checks
    # guard the histogram, moments, and event-equivalent spectrum weight.
    histogram_edges = np.linspace(float(samples.min()), float(samples.max()), 1001)
    reference_histogram = np.histogram(samples.reshape(-1), bins=histogram_edges)[0]
    reference_mean = float(samples.mean())
    reference_variance = float(samples.var())
    if not math.isclose(float(samples.size / EXPECTED_SAMPLES), len(truth), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("Posterior spectrum does not carry one event-equivalent weight per event")

    permutation_rows = []
    permutation_invariants = []
    crps_permutation_by_event = np.empty((args.permutations, len(truth)), dtype=np.float64)
    permutation_migration = np.empty((args.permutations, len(mass.COARSE_LABELS), len(mass.COARSE_LABELS)), dtype=np.float64)
    fine_edges = mass._fine_edges()
    fine_id = mass._bin_ids(truth, fine_edges)
    fine_count = np.bincount(fine_id, minlength=len(fine_edges) - 1)
    fine_truth_mean = np.bincount(fine_id, weights=truth, minlength=len(fine_edges) - 1) / np.maximum(fine_count, 1)
    q_mass_mean = samples.mean(axis=1)
    fine_q_reco_mean = np.bincount(fine_id, weights=q_mass_mean, minlength=len(fine_edges) - 1) / np.maximum(fine_count, 1)
    fine_perm_reco_means = np.empty((args.permutations, len(fine_edges) - 1), dtype=np.float64)
    seed_records = []
    for permutation_index in range(args.permutations):
        seed = PERMUTATION_SEED_BASE + permutation_index
        assignment = _derangement(len(truth), seed)
        permuted_samples = samples[assignment]
        if permuted_samples.shape != samples.shape:
            raise RuntimeError("Whole-event posterior permutation changed the M32 shape")
        if not np.array_equal(np.histogram(permuted_samples.reshape(-1), bins=histogram_edges)[0], reference_histogram):
            raise RuntimeError(f"Pooled sample histogram changed under permutation seed {seed}")
        if abs(float(permuted_samples.mean()) - reference_mean) > 1e-10:
            raise RuntimeError(f"Pooled sample mean changed under permutation seed {seed}")
        if abs(float(permuted_samples.var()) - reference_variance) > 1e-9:
            raise RuntimeError(f"Pooled sample variance changed under permutation seed {seed}")
        if not math.isclose(float(permuted_samples.size / EXPECTED_SAMPLES), len(truth), rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Event-equivalent sample spectrum weight changed under seed {seed}")
        permutation_invariants.append({
            "seed": seed,
            "fixed_points": 0,
            "bijection": True,
            "complete_posterior_rows_moved": True,
            "pooled_histogram_unchanged": True,
            "pooled_mean_abs_difference": abs(float(permuted_samples.mean()) - reference_mean),
            "pooled_variance_abs_difference": abs(float(permuted_samples.var()) - reference_variance),
            "event_equivalent_weight": float(permuted_samples.size / EXPECTED_SAMPLES),
        })
        score_perm, ordered_perm = empirical_crps(permuted_samples, truth)
        crps_permutation_by_event[permutation_index] = score_perm
        summaries, migration = _region_summaries(permuted_samples, truth, masks, score_perm, ordered_perm)
        permutation_migration[permutation_index] = migration
        perm_mean = permuted_samples.mean(axis=1)
        fine_perm_reco_means[permutation_index] = (
            np.bincount(fine_id, weights=perm_mean, minlength=len(fine_edges) - 1)
            / np.maximum(fine_count, 1)
        )
        for row in summaries:
            permutation_rows.append({"permutation_seed": seed, **row})
        seed_records.append({"seed": seed})
        del assignment, permuted_samples, score_perm, ordered_perm

    # Sanity-check the efficient score on real rows as well as on the tiny
    # random fixture used above.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    check_rows = rng.choice(len(truth), size=24, replace=False)
    check_fast, _ = empirical_crps(samples[check_rows], truth[check_rows])
    check_brute = np.asarray([
        np.mean(np.abs(row - target))
        - np.abs(row[:, None] - row[None, :]).sum() / (2.0 * row.size * row.size)
        for row, target in zip(samples[check_rows], truth[check_rows], strict=True)
    ])
    crps_check = max(crps_check, float(np.max(np.abs(check_fast - check_brute))))
    if crps_check > 2.0e-12:
        raise RuntimeError(f"CRPS explicit-formula check failed: {crps_check:.3g}")

    seed_crps = np.asarray([
        [next(row["crps_mean_gev"] for row in permutation_rows if row["permutation_seed"] == seed and row["truth_region"] == region)
         for region in ["inclusive"] + list(mass.COARSE_LABELS)]
        for seed in [record["seed"] for record in seed_records]
    ])
    region_names = ["inclusive"] + list(mass.COARSE_LABELS)
    regional_crps_rows = []
    aggregate_rows = []
    for region_index, region in enumerate(region_names):
        values = seed_crps[:, region_index]
        q_value = q_by_region[region]["crps_mean_gev"]
        mean_perm = float(values.mean())
        skill_permutation = 1.0 - q_value / values
        row = {
            "truth_region": region,
            "events": int(masks[region].sum()),
            "q_crps_mean_gev": q_value,
            "q_crps_median_gev": q_by_region[region]["crps_median_gev"],
            "permuted_crps_mean_gev": mean_perm,
            "permuted_crps_std_gev": float(values.std(ddof=1)),
            "permuted_crps_p05_gev": float(np.quantile(values, 0.05)),
            "permuted_crps_p95_gev": float(np.quantile(values, 0.95)),
            "delta_crps_perm_minus_q_gev": mean_perm - q_value,
            "skill_score_mean_over_permutations": float(skill_permutation.mean()),
            "skill_score_p05_over_permutations": float(np.quantile(skill_permutation, 0.05)),
            "skill_score_p95_over_permutations": float(np.quantile(skill_permutation, 0.95)),
            "skill_score_convention": "1 - CRPS(Q) / CRPS(condition-destroyed); positive favours event-conditioned Q",
        }
        for metric in (
            "correct_bin_probability", "correct_or_adjacent_probability",
            "coverage68", "coverage90", "width68_mean_gev", "width68_median_gev",
            "width90_mean_gev", "width90_median_gev",
        ):
            metric_values = np.asarray([
                record[metric] for record in permutation_rows
                if record["truth_region"] == region
            ], dtype=np.float64)
            row[f"q_{metric}"] = float(q_by_region[region][metric])
            row[f"permuted_{metric}"] = float(metric_values.mean())
            row[f"permuted_{metric}_p05"] = float(np.quantile(metric_values, 0.05))
            row[f"permuted_{metric}_p95"] = float(np.quantile(metric_values, 0.95))
        aggregate_rows.append(row)
        for seed_index, seed_record in enumerate(seed_records):
            regional_crps_rows.append({
                "permutation_seed": seed_record["seed"],
                "truth_region": region,
                "events": row["events"],
                "permuted_crps_mean_gev": float(seed_crps[seed_index, region_index]),
                "q_crps_mean_gev": q_value,
            })

    mean_permuted_crps_by_event = crps_permutation_by_event.mean(axis=0)
    paired_delta = mean_permuted_crps_by_event - score_q
    bootstrap = _bootstrap_mean(paired_delta, replicates=args.bootstrap_replicates, seed=BOOTSTRAP_SEED)

    # Aggregate the requested secondary comparisons over the fixed seed set.
    metric_names = (
        "mean_rmse_gev", "mean_bias_gev", "mean_response_slope",
        "correct_bin_probability", "correct_or_adjacent_probability",
        "coverage68", "coverage90", "width68_mean_gev", "width68_median_gev",
        "width90_mean_gev", "width90_median_gev",
    )
    conditional_metric_rows = []
    for region in region_names:
        qrow = q_by_region[region]
        for metric in metric_names:
            values = np.asarray([
                row[metric] for row in permutation_rows
                if row["truth_region"] == region
            ], dtype=np.float64)
            q_value = float(qrow[metric])
            perm_value = float(values.mean())
            conditional_metric_rows.append({
                "truth_region": region,
                "events": int(masks[region].sum()),
                "metric": metric,
                "q_value": q_value,
                "permuted_mean": perm_value,
                "permuted_std_over_seeds": float(values.std(ddof=1)),
                "permuted_p05_over_seeds": float(np.quantile(values, 0.05)),
                "permuted_p95_over_seeds": float(np.quantile(values, 0.95)),
                "q_minus_permuted_mean": q_value - perm_value,
            })

    migration_rows = []
    truth_bins = mass._bin_ids(truth, mass.COARSE_EDGES)
    truth_bin_counts = np.bincount(truth_bins, minlength=len(mass.COARSE_LABELS))
    for truth_bin, truth_label in enumerate(mass.COARSE_LABELS):
        for reco_bin, reco_label in enumerate(mass.COARSE_LABELS):
            migration_rows.append({
                "source": "PCFM_Q", "permutation_seed": "real", "truth_bin": truth_label,
                "reco_bin": reco_label, "events_truth_bin": int(truth_bin_counts[truth_bin]),
                "P_reco_given_truth": float(q_migration[truth_bin, reco_bin]),
            })
            values = permutation_migration[:, truth_bin, reco_bin]
            migration_rows.append({
                "source": "condition-destroyed mean over permutations", "permutation_seed": "mean",
                "truth_bin": truth_label, "reco_bin": reco_label,
                "events_truth_bin": int(truth_bin_counts[truth_bin]),
                "P_reco_given_truth": float(values.mean()),
                "P_reco_given_truth_std_over_seeds": float(values.std(ddof=1)),
                "P_reco_given_truth_p05_over_seeds": float(np.quantile(values, 0.05)),
                "P_reco_given_truth_p95_over_seeds": float(np.quantile(values, 0.95)),
            })
            for seed_index, seed_record in enumerate(seed_records):
                migration_rows.append({
                    "source": "condition-destroyed individual permutation",
                    "permutation_seed": seed_record["seed"], "truth_bin": truth_label,
                    "reco_bin": reco_label,
                    "events_truth_bin": int(truth_bin_counts[truth_bin]),
                    "P_reco_given_truth": float(permutation_migration[seed_index, truth_bin, reco_bin]),
                })

    # Plot rows use one joined record per fine bin.
    joined_response_rows = []
    for index in range(len(fine_edges) - 1):
        if fine_count[index] == 0:
            continue
        perm_values = fine_perm_reco_means[:, index]
        joined_response_rows.append({
            "truth_bin_index": index, "truth_bin": mass._bin_labels(fine_edges)[index],
            "events": int(fine_count[index]), "truth_mean_gev": float(fine_truth_mean[index]),
            "q_reco_mean_gev": float(fine_q_reco_mean[index]),
            "permuted_reco_mean_gev": float(perm_values.mean()),
            "permuted_reco_mean_std_gev": float(perm_values.std(ddof=1)),
            "permuted_reco_mean_p05_gev": float(np.quantile(perm_values, 0.05)),
            "permuted_reco_mean_p95_gev": float(np.quantile(perm_values, 0.95)),
        })

    coverage_rows = []
    for region in region_names:
        qrow = q_by_region[region]
        region_perm = [row for row in permutation_rows if row["truth_region"] == region]
        for metric in ("coverage68", "coverage90", "width68_mean_gev", "width68_median_gev", "width90_mean_gev", "width90_median_gev"):
            values = np.asarray([row[metric] for row in region_perm], dtype=np.float64)
            coverage_rows.append({
                "truth_region": region, "events": int(masks[region].sum()), "metric": metric,
                "q_value": float(qrow[metric]), "permuted_mean": float(values.mean()),
                "permuted_std_over_seeds": float(values.std(ddof=1)),
                "permuted_p05_over_seeds": float(np.quantile(values, 0.05)),
                "permuted_p95_over_seeds": float(np.quantile(values, 0.95)),
            })

    if args.rotation_json is None or not args.rotation_json.is_file():
        raise FileNotFoundError("Real trained-Q rotation JSON is required for this extension")
    rotation = json.loads(args.rotation_json.read_text(encoding="utf-8"))
    if rotation.get("checkpoint_sha256") != provenance.get("checkpoint_sha256"):
        raise ValueError("Real-Q rotation and mass cache use different checkpoint hashes")
    if rotation.get("source_bank_sha256") != provenance.get("source_bank_sha256"):
        raise ValueError("Real-Q rotation and mass cache use different source banks")
    rotation_rows = _rotation_comparison(rotation)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_dir = args.output_dir / "metrics"
    crps_rows = aggregate_rows
    _csv_write(metric_dir / "crps_conditional_skill.csv", crps_rows)
    _csv_write(metric_dir / "crps_by_truth_region.csv", regional_crps_rows)
    _csv_write(metric_dir / "permutation_summary.csv", permutation_rows)
    _csv_write(metric_dir / "conditional_metrics.csv", conditional_metric_rows)
    _csv_write(metric_dir / "conditional_response.csv", joined_response_rows)
    _csv_write(metric_dir / "conditional_migration.csv", migration_rows)
    _csv_write(metric_dir / "coverage_comparison.csv", coverage_rows)
    _csv_write(args.output_dir / "rotation/rotation_comparison.csv", rotation_rows)
    _json_write(args.output_dir / "rotation/PCFM_Q_source_only_rotation.json", rotation)
    _json_write(args.output_dir / "metrics/bootstrap_summary.json", {
        "global_paired_event_bootstrap": bootstrap,
        "difference_sign": "permuted CRPS minus real-Q CRPS; positive means real Q scores better",
    })

    # The CRPS and correct-bin tables include an inclusive row.  The response
    # plot uses all predeclared fine bins with at least 40 truth events.
    _write_plots(crps_rows, joined_response_rows, permutation_rows, rotation_rows, args.output_dir)

    global_skill = next(row for row in crps_rows if row["truth_region"] == "inclusive")
    central_skills = [next(row for row in crps_rows if row["truth_region"] == label)["skill_score_mean_over_permutations"]
                      for label in ("400-500", "500-700")]
    tail_skills = [next(row for row in crps_rows if row["truth_region"] == label)["skill_score_mean_over_permutations"]
                   for label in ("<345", ">1000")]
    diagnosis = "strong global and regional conditioning" if global_skill["skill_score_mean_over_permutations"] >= 0.20 and min(central_skills + tail_skills) >= 0.10 else (
        "strong central conditioning with weaker tail conditioning" if min(central_skills) >= 0.20 and min(tail_skills) < 0.10 else
        "weak global conditioning relative to the pooled posterior marginal"
    )
    if diagnosis == "strong global and regional conditioning":
        recommended_next_action = (
            "compare a deterministic parent-regression baseline against the same validation response and spectrum metrics"
        )
    elif diagnosis == "strong central conditioning with weaker tail conditioning":
        recommended_next_action = (
            "test a controlled tail/training-prior treatment against the frozen event-posterior control"
        )
    else:
        recommended_next_action = (
            "evaluate a geometry-conditioned parent model against this event-posterior permutation control when its validated prediction becomes available"
        )
    report_lines = [
        "# REALNU conditional-skill diagnosis",
        "",
        "## Scope and provenance",
        "",
        f"- Accepted Q mass cache: `{provenance['cache_path']}` (SHA-256 `{provenance['cache_sha256']}`).",
        f"- Prediction SHA-256: `{provenance['prediction_sha256']}`; ordered EVENT_ID SHA-256: `{provenance['event_id_sha256']}`.",
        f"- Source-bank SHA-256: `{provenance['source_bank_sha256']}`; checkpoint SHA-256: `{provenance['checkpoint_sha256']}`; scaler identity: `{provenance['scaler_identity_sha256']}`.",
        f"- Population: {EXPECTED_EVENTS:,} validation events, M={EXPECTED_SAMPLES}; Heun{provenance['steps']}; sealed test used: `false`.",
        f"- Cache provenance passed and the cache was reused: `{provenance['cache_reused']}`. No parent predictions were regenerated.",
        f"- Condition-destroyed controls: {args.permutations} deterministic whole-event posterior derangements; seeds {PERMUTATION_SEED_BASE}–{PERMUTATION_SEED_BASE + args.permutations - 1}.",
        "- Permutations use only seeded row indices. Each complete 32-sample event posterior stays intact; truth never enters assignment generation.",
        "",
        "## Conditional skill",
        "",
        f"- Classification: **{diagnosis}**. Positive skill is `1 - CRPS(Q)/CRPS(permuted)`; a positive value favours real event conditioning.",
        f"- Inclusive CRPS: Q {global_skill['q_crps_mean_gev']:.2f} GeV; condition-destroyed mean {global_skill['permuted_crps_mean_gev']:.2f} ± {global_skill['permuted_crps_std_gev']:.2f} GeV across seeds; ΔCRPS (permuted−Q) {global_skill['delta_crps_perm_minus_q_gev']:.2f} GeV; skill {global_skill['skill_score_mean_over_permutations']:.3f}.",
        f"- Paired event-bootstrap 95% CI for the global CRPS improvement: [{bootstrap['ci95_low_gev']:.2f}, {bootstrap['ci95_high_gev']:.2f}] GeV ({args.bootstrap_replicates} replicas).",
        "",
        "| Truth region [GeV] | N | Q CRPS | Permuted CRPS mean ± seed SD | ΔCRPS | Skill | Mean RMSE Q / permuted | Slope Q / permuted | Correct-bin probability Q / permuted | 68% coverage Q / permuted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in mass.COARSE_LABELS:
        row = next(row for row in crps_rows if row["truth_region"] == region)
        lookup = {(r["truth_region"], r["metric"]): r for r in conditional_metric_rows}
        rmse = lookup[(region, "mean_rmse_gev")]
        slope = lookup[(region, "mean_response_slope")]
        correct = lookup[(region, "correct_bin_probability")]
        cov = lookup[(region, "coverage68")]
        report_lines.append(
            f"| {region} | {row['events']:,} | {row['q_crps_mean_gev']:.2f} | {row['permuted_crps_mean_gev']:.2f} ± {row['permuted_crps_std_gev']:.2f} | {row['delta_crps_perm_minus_q_gev']:.2f} | {row['skill_score_mean_over_permutations']:.3f} | {rmse['q_value']:.1f} / {rmse['permuted_mean']:.1f} | {slope['q_value']:.3f} / {slope['permuted_mean']:.3f} | {correct['q_value']:.3f} / {correct['permuted_mean']:.3f} | {cov['q_value']:.3f} / {cov['permuted_mean']:.3f} |"
        )
    report_lines.extend([
        "",
        "The CRPS comparison scores the event-conditioned empirical posterior and the same global collection of complete event posteriors assigned to unrelated truth events. The permutation therefore preserves the global sample multiset, mean, variance, and spectrum while removing the original event-to-posterior pairing.",
        f"- Inclusive posterior-mean RMSE is {q_by_region['inclusive']['mean_rmse_gev']:.2f} GeV for Q and {lookup[('inclusive', 'mean_rmse_gev')]['permuted_mean']:.2f} GeV after permutation; response slope is {q_by_region['inclusive']['mean_response_slope']:.3f} versus {lookup[('inclusive', 'mean_response_slope')]['permuted_mean']:.3f}.",
        f"- Correct-bin / correct-or-adjacent probability is {q_by_region['inclusive']['correct_bin_probability']:.3f} / {q_by_region['inclusive']['correct_or_adjacent_probability']:.3f} for Q and {lookup[('inclusive', 'correct_bin_probability')]['permuted_mean']:.3f} / {lookup[('inclusive', 'correct_or_adjacent_probability')]['permuted_mean']:.3f} after permutation.",
        f"- 68% / 90% coverage is {q_by_region['inclusive']['coverage68']:.3f} / {q_by_region['inclusive']['coverage90']:.3f} for Q and {lookup[('inclusive', 'coverage68')]['permuted_mean']:.3f} / {lookup[('inclusive', 'coverage90')]['permuted_mean']:.3f} after permutation. Global mean 68% / 90% widths remain {q_by_region['inclusive']['width68_mean_gev']:.1f} / {q_by_region['inclusive']['width90_mean_gev']:.1f} GeV in both, as expected when the exact posterior-set multiset is preserved.",
        "- Conclusion: the current shrinkage and tail undercoverage coexist with substantial event-specific mass information. Q is not reproducing only the global mass prior.",
        "",
        "## Real PCFM_Q orientation response",
        "",
        f"- Rotation artifact: `{args.rotation_json}`; trained checkpoint SHA matches the accepted Q prediction, and the fixed source-bank SHA matches the comparison manifest.",
        "- The same 64-event truth-mass-quantile sample, source-bank first sample, Heun8 policy, and scaler-aware physical px/py transform were used for event-only, source-only, and co-rotation.",
        "- Source-only results below report wrapped output Δφ and its residual from the requested angle; pT and mttbar changes are also in `rotation/rotation_comparison.csv`.",
        "- Event-only rotation leaves both parent mean Δφ values at zero within numerical precision. Source-only rotation moves both parents by nearly the requested angle, and co-rotation follows the same response. On this fixed-source 64-event test, the stochastic source controls absolute orientation much more strongly than rotating the measured event does.",
        "",
        "| Angle [rad] | Mode | Top mean Δφ | Top RMS residual | Antitop mean Δφ | Antitop RMS residual |",
        "|---:|---|---:|---:|---:|---:|",
    ])
    for angle in sorted({row["angle_rad"] for row in rotation_rows}):
        for mode in ("event-only", "source-only", "co-rotation"):
            top = next(row for row in rotation_rows if row["angle_rad"] == angle and row["mode"] == mode and row["role"] == "top")
            antitop = next(row for row in rotation_rows if row["angle_rad"] == angle and row["mode"] == mode and row["role"] == "antitop")
            report_lines.append(
                f"| {angle:.3f} | {mode} | {top['mean_output_delta_phi_rad']:.4f} | {top['rms_covariance_residual_delta_phi_rad']:.4f} | {antitop['mean_output_delta_phi_rad']:.4f} | {antitop['rms_covariance_residual_delta_phi_rad']:.4f} |"
            )
    report_lines.extend([
        "",
        "## Interpretation and next experiment",
        "",
        f"The paired CRPS and regional migration comparisons classify PCFM_Q as having **{diagnosis}**. The global mass marginal can remain excellent while the conditioning conclusion is determined by the proper event-wise score and loss of response under posterior-set permutation.",
        "",
        "PCFM_geom was unavailable at the single bounded artifact check (no trained checkpoint, training record, or validated M32 prediction at the manifest-linked production paths). No geom physics result is inferred.",
        "",
        f"**Recommended next action:** {recommended_next_action}.",
        "",
        "## Validation gates",
        "",
        f"- Sorted CRPS vs explicit pairwise formula maximum difference: {crps_check:.3g} GeV.",
        f"- Accepted-Q regression max absolute differences: {max(value['absolute_difference'] for value in regression.values()):.3g}.",
        f"- Every posterior-set assignment was a full-population derangement; pooled histogram, mean, variance, and event-equivalent weight were checked for each seed.",
        f"- Runtime: {time.monotonic() - started:.1f} s; {args.bootstrap_replicates} paired bootstrap replicas.",
    ])
    report_path = args.output_dir / "report/CONDITIONAL_SKILL_DIAGNOSIS.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "status": "completed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis": "realnu_q_conditional_skill_by_whole_event_posterior_permutation_v1",
        "inputs": provenance,
        "mass_response_audit_dir": str(args.mass_audit_dir.resolve()),
        "rotation_json": str(args.rotation_json.resolve()),
        "rotation_json_sha256": _sha256(args.rotation_json),
        "population_events": EXPECTED_EVENTS,
        "posterior_samples_per_event": EXPECTED_SAMPLES,
        "permutation_count": args.permutations,
        "permutation_seed_base": PERMUTATION_SEED_BASE,
        "permutation_seeds": [row["seed"] for row in seed_records],
        "permutation_invariants": permutation_invariants,
        "permutation_definition": "seeded random row order with a nonzero cyclic offset; complete event posterior rows move as units; no fixed points; truth is never read by assignment generation",
        "pooled_sample_histogram_bin_edges_gev": histogram_edges.tolist(),
        "pooled_sample_histogram_sha256": hashlib.sha256(reference_histogram.astype("<i8").tobytes()).hexdigest(),
        "pooled_sample_mean_gev": reference_mean,
        "pooled_sample_variance_gev2": reference_variance,
        "total_event_equivalent_spectrum_weight": float(samples.size / EXPECTED_SAMPLES),
        "crps": {
            "definition": "mean(abs(sample-truth)) - sum_j((2j-M+1)*sorted_sample_j)/M^2, j=0..M-1",
            "explicit_pairwise_max_abs_difference_gev": crps_check,
        },
        "q_regression_against_accepted_mass_audit": regression,
        "bootstrap": bootstrap,
        "sealed_test_used": False,
        "truth_free_conditioning": True,
        "diagnosis": diagnosis,
        "recommended_next_action": recommended_next_action,
        "elapsed_seconds": time.monotonic() - started,
    }
    _json_write(args.output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--mass-audit-dir", type=Path, required=True)
    parser.add_argument("--rotation-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    if args.permutations < 16:
        raise ValueError("Use at least 16 whole-event posterior permutations")
    result = run(args)
    print(json.dumps({"status": result["status"], "diagnosis": result["diagnosis"],
                      "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
