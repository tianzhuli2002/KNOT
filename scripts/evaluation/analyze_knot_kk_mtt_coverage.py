#!/usr/bin/env python3
"""Measure event-level mtt posterior coverage on an existing KK transfer sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import wasserstein_distance


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def invariant_mass(p4: np.ndarray, name: str) -> np.ndarray:
    mass2 = p4[..., 0] ** 2 - np.square(p4[..., 1:]).sum(axis=-1)
    if not np.isfinite(mass2).all() or np.any(mass2 < -1e-5):
        raise ValueError(f"{name} has a non-finite or spacelike four-vector")
    return np.sqrt(np.maximum(mass2, 0.0))


def bootstrap_interval(values: np.ndarray, seed: int, resamples: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    n = len(values)
    for index in range(resamples):
        rows = rng.integers(0, n, size=n)
        samples[index] = values[rows].mean()
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def model_coverage(
    name: str,
    posterior_path: Path,
    truth_path: Path,
    expected_ids: np.ndarray | None,
    seed: int,
    resamples: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    manifest_path = posterior_path.parent / "evaluation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing inference manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("variant") != name:
        raise ValueError(f"{manifest_path} says variant {manifest.get('variant')!r}, expected {name}")
    if manifest.get("truth_used_during_inference") is not False:
        raise ValueError(f"Truth-independence check failed for {manifest_path}")

    with h5py.File(posterior_path, "r") as posterior, h5py.File(truth_path, "r") as truth:
        for path in ("EVENT_ID", "parents", "derived/m_parent_pair"):
            if path not in posterior:
                raise KeyError(f"{posterior_path} lacks {path}")
        for path in ("EVENT_ID", "TRUTH/parents"):
            if path not in truth:
                raise KeyError(f"{truth_path} lacks {path}")
        ids = np.asarray(posterior["EVENT_ID"][:], dtype=np.int64)
        truth_ids = np.asarray(truth["EVENT_ID"][:], dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError(f"{posterior_path} EVENT_ID is not unique and one-dimensional")
        if not np.array_equal(ids, truth_ids):
            raise ValueError(f"EVENT_ID order differs between {posterior_path} and {truth_path}")
        if truth.attrs.get("units") != "GeV" or truth.attrs.get("four_vector_order") != "E,px,py,pz":
            raise ValueError(f"Truth units or four-vector order are not the maintained GeV contract: {truth_path}")
        if expected_ids is not None and not np.array_equal(ids, expected_ids):
            raise ValueError("M/C/J posterior EVENT_ID arrays differ")
        roles = json.loads(posterior.attrs["parent_roles"])
        if roles != ["top", "antitop"]:
            raise ValueError(f"Unexpected parent role order in {posterior_path}: {roles}")
        draws = np.asarray(posterior["derived/m_parent_pair"][:], dtype=np.float64)
        parents = np.asarray(posterior["parents"][:], dtype=np.float64)
        truth_parents = np.asarray(truth["TRUTH/parents"][:], dtype=np.float64)

    if draws.ndim != 2 or draws.shape[0] != len(ids) or draws.shape[1] < 2:
        raise ValueError(f"Invalid mtt posterior draw shape: {draws.shape}")
    if parents.shape != (len(ids), draws.shape[1], 2, 4):
        raise ValueError(f"Unexpected posterior parent shape: {parents.shape}")
    if truth_parents.shape != (len(ids), 2, 4):
        raise ValueError(f"Unexpected truth parent shape: {truth_parents.shape}")
    if not np.isfinite(draws).all() or not np.isfinite(parents).all():
        raise ValueError(f"Non-finite posterior values in {posterior_path}")
    if not np.isfinite(truth_parents).all():
        raise ValueError(f"Non-finite truth parent values in {truth_path}")

    recomputed_draws = invariant_mass(parents.sum(axis=2), f"{name} posterior parents")
    if not np.allclose(draws, recomputed_draws, rtol=1e-4, atol=0.1):
        raise ValueError(f"Stored derived mtt disagrees with parent four-vectors for {name}")
    truth_mtt = invariant_mass(truth_parents.sum(axis=1), "after-FSR truth parents")
    point_mtt = draws.mean(axis=1)
    coverages = {}
    hit_arrays = {}
    for level, lower_q, upper_q in ((0.68, 0.16, 0.84), (0.90, 0.05, 0.95)):
        low, high = np.quantile(draws, [lower_q, upper_q], axis=1)
        hit = (truth_mtt >= low) & (truth_mtt <= high)
        key = str(int(level * 100))
        hit_arrays[key] = hit
        coverages[key] = {
            "nominal": level,
            "equal_tail_quantiles": [lower_q, upper_q],
            "empirical": float(hit.mean()),
            "event_bootstrap_95pct_interval": bootstrap_interval(hit.astype(np.float64), seed, resamples),
        }

    predictive = draws.reshape(-1)
    truth_quantiles = np.quantile(truth_mtt, [0.90, 0.99])
    predictive_quantiles = np.quantile(predictive, [0.90, 0.99])
    response_residual = point_mtt - truth_mtt
    centered_truth = truth_mtt - truth_mtt.mean()
    slope = float(np.mean(centered_truth * (point_mtt - point_mtt.mean())) / np.mean(centered_truth**2))
    result = {
        "variant": name,
        "posterior_path": str(posterior_path),
        "posterior_sha256": sha256_file(posterior_path),
        "inference_manifest_sha256": sha256_file(manifest_path),
        "events": len(ids),
        "posterior_draws_per_event": int(draws.shape[1]),
        "event_id_ordered_sha256_little_endian_int64": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "truth_used_during_inference": False,
        "mttbar_point_response": {
            "bias_GeV": float(response_residual.mean()),
            "RMSE_GeV": float(np.sqrt(np.mean(response_residual**2))),
            "truth_to_posterior_mean_slope": slope,
        },
        "equal_tail_coverage": coverages,
        "posterior_predictive_tail": {
            "truth_q90_q99_GeV": [float(x) for x in truth_quantiles],
            "posterior_predictive_q90_q99_GeV": [float(x) for x in predictive_quantiles],
            "predictive_over_truth_q90_q99": [float(x) for x in predictive_quantiles / truth_quantiles],
            "wasserstein1_GeV": float(wasserstein_distance(truth_mtt, predictive)),
        },
    }
    return result, hit_arrays["68"], hit_arrays["90"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--posterior", action="append", required=True, metavar="MODEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42011)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    if args.bootstrap_resamples < 100:
        parser.error("At least 100 event bootstrap resamples are required")
    if not args.truth.is_file() or not args.contract.is_file():
        parser.error("Truth sidecar and conversion contract must exist")

    contract = json.loads(args.contract.read_text())
    parsed = {}
    for item in args.posterior:
        model, separator, value = item.partition("=")
        if not separator or model not in {"M", "C", "J"} or model in parsed:
            parser.error(f"Invalid or duplicate posterior specification: {item}")
        parsed[model] = Path(value)
    if set(parsed) != {"M", "C", "J"}:
        parser.error("Supply exactly one saved posterior for each of M, C, and J")

    results = {}
    common_ids = None
    hit_arrays = {}
    for offset, model in enumerate(("M", "C", "J")):
        result, hit68, hit90 = model_coverage(
            model,
            parsed[model],
            args.truth,
            common_ids,
            args.seed + offset,
            args.bootstrap_resamples,
        )
        results[model] = result
        hit_arrays[model] = {"68": hit68, "90": hit90}
        with h5py.File(parsed[model], "r") as posterior:
            ids = np.asarray(posterior["EVENT_ID"][:], dtype=np.int64)
        common_ids = ids if common_ids is None else common_ids

    paired = {}
    for first, second in (("J", "C"), ("J", "M"), ("C", "M")):
        paired[f"{first}_minus_{second}"] = {}
        for level in ("68", "90"):
            delta = hit_arrays[first][level].astype(np.float64) - hit_arrays[second][level].astype(np.float64)
            paired[f"{first}_minus_{second}"][level] = {
                "coverage_difference": float(delta.mean()),
                "event_bootstrap_95pct_interval": bootstrap_interval(delta, args.seed + 100 + int(level) + ord(first), args.bootstrap_resamples),
            }

    data = {
        "sample": {
            "DSID": contract["DSID"],
            "production_campaign": contract["production_campaign"],
            "production_tag": contract["production_tag"],
            "candidate_mass_gev": contract.get("physics_parameters", {}).get("mass_gev"),
            "candidate_coupling": contract.get("physics_parameters", {}).get("coupling"),
            "production_lineage_status": contract.get("physics_parameters", {}).get("mapping_status"),
        },
        "truth_sidecar": str(args.truth),
        "truth_sidecar_sha256": sha256_file(args.truth),
        "conversion_contract": str(args.contract),
        "conversion_contract_sha256": sha256_file(args.contract),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "coverage_definition": "Per-event equal-tail posterior interval of derived mtt; event is the bootstrap unit, posterior draws are not independent bootstrap units.",
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "models": results,
        "paired_coverage_differences": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=args.output.name + ".", suffix=".tmp", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, args.output)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    print(json.dumps({"output": str(args.output), "DSID": contract["DSID"], "events": results["J"]["events"]}, indent=2))


if __name__ == "__main__":
    main()
