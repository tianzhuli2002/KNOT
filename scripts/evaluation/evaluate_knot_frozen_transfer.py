#!/usr/bin/env python3
"""Run a frozen KNOT M/C/J checkpoint on a truth-free external measured input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.invisible import InvisibleScaler
from joint_event_reconstruction.data.partonic import PartonicScaler
from joint_event_reconstruction.knot_physics import (
    joint_energy_score,
    joint_metric,
    mass,
    reconstruction_metrics,
)
from scripts.evaluation.evaluate_knot_production_checkpoint import (
    as_numpy,
    concatenate,
    physical_calibration_features,
    sha256_file,
)
from scripts.run_knot import (
    FREEZE_PATH,
    make_module,
    move_to_device,
    truth_free_batch,
    validate_frozen_configuration,
)

ROLE_ORDER = ("top", "antitop")
ROOT = Path(__file__).resolve().parents[2]


def sha256_ids(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def validate_transfer_files(measured_path: Path, truth_path: Path) -> np.ndarray:
    with h5py.File(measured_path, "r") as measured:
        if not bool(measured.attrs.get("measured_only", False)):
            raise ValueError(f"Transfer input is not marked measured-only: {measured_path}")
        if any(name in measured for name in ("TRUTH", "LABELS", "SUPERVISION")):
            raise ValueError(f"Truth-like group is present in measured transfer input: {measured_path}")
        required = ("EVENT_ID", "SPLIT", "INPUTS/JET", "INPUTS/LEPTON", "INPUTS/MET", "INPUTS/GLOBAL")
        missing = [name for name in required if name not in measured]
        if missing:
            raise KeyError(f"Transfer input {measured_path} is missing {missing}")
        ids = np.asarray(measured["EVENT_ID"][:], dtype=np.int64)
        split = np.asarray(measured["SPLIT"][:])
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Transfer EVENT_ID must be unique one-dimensional values: {measured_path}")
        if split.shape != ids.shape or not np.all(split == 1):
            raise ValueError(f"Every external transfer event must have SPLIT=1: {measured_path}")
        expected_shapes = {
            "INPUTS/JET": (len(ids), 8),
            "INPUTS/LEPTON": (len(ids), 2),
            "INPUTS/MET": (len(ids), 1),
            "INPUTS/GLOBAL": (len(ids),),
        }
        for name, shape in expected_shapes.items():
            if measured[name].shape != shape:
                raise ValueError(f"Transfer input {measured_path}:{name} has shape {measured[name].shape}, expected {shape}")
        jets = measured["INPUTS/JET"][:]
        njet = measured["INPUTS/GLOBAL"]["njet"]
        if not np.isfinite(jets["pt"][:, :].astype(np.float64)[np.arange(8)[None, :] < njet[:, None]]).all():
            raise ValueError(f"Valid jet slots contain non-finite pT values: {measured_path}")
        if not np.all(np.isnan(jets["pt"][np.arange(8)[None, :] >= njet[:, None]])):
            raise ValueError(f"Padded jet slots must be NaN: {measured_path}")

    with h5py.File(truth_path, "r") as truth:
        required = ("EVENT_ID", "TRUTH/parents", "TRUTH/neutrinos", "TRUTH/w_daughter_pdg_id")
        missing = [name for name in required if name not in truth]
        if missing:
            raise KeyError(f"Transfer truth sidecar {truth_path} is missing {missing}")
        truth_ids = np.asarray(truth["EVENT_ID"][:], dtype=np.int64)
        if not np.array_equal(ids, truth_ids):
            raise ValueError(f"Measured input and separate truth sidecar EVENT_ID order differs: {measured_path}, {truth_path}")
        if truth["TRUTH/parents"].shape != (len(ids), 2, 4):
            raise ValueError(f"Invalid after-FSR parents shape in {truth_path}")
        if truth["TRUTH/neutrinos"].shape != (len(ids), 2, 3):
            raise ValueError(f"Invalid direct-neutrino p3 shape in {truth_path}")
    return ids


def posterior_mtt(parents: np.ndarray) -> np.ndarray:
    return mass(parents.sum(axis=2))


def empirical_wasserstein_1d(left: np.ndarray, right: np.ndarray) -> float:
    """Exact W1 distance between equally weighted empirical 1D samples."""
    x = np.sort(np.asarray(left, dtype=np.float64))
    y = np.sort(np.asarray(right, dtype=np.float64))
    if not len(x) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Wasserstein inputs must be nonempty finite samples")
    i = j = 0
    left_edge = 0.0
    distance = 0.0
    while i < len(x) and j < len(y):
        right_edge = min((i + 1) / len(x), (j + 1) / len(y))
        distance += (right_edge - left_edge) * abs(x[i] - y[j])
        left_edge = right_edge
        left_cross, right_cross = (i + 1) * len(y), (j + 1) * len(x)
        if left_cross <= right_cross:
            i += 1
        if right_cross <= left_cross:
            j += 1
    return float(distance)


def tail_closure(truth_mtt: np.ndarray, posterior_mtt_values: np.ndarray, seed: int) -> dict:
    quantiles = (0.90, 0.99)
    truth_q = np.quantile(truth_mtt, quantiles)
    predictive_q = np.quantile(posterior_mtt_values, quantiles)
    ratios = predictive_q / truth_q
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((300, len(quantiles)), dtype=np.float64)
    for index in range(len(bootstrap)):
        rows = rng.integers(0, len(truth_mtt), size=len(truth_mtt))
        bootstrap[index] = (
            np.quantile(posterior_mtt_values[rows].reshape(-1), quantiles)
            / np.quantile(truth_mtt[rows], quantiles)
        )
    wasserstein = empirical_wasserstein_1d(truth_mtt, posterior_mtt_values.reshape(-1))
    return {
        "truth_quantiles_GeV": {str(q): float(value) for q, value in zip(quantiles, truth_q)},
        "posterior_predictive_quantiles_GeV": {
            str(q): float(value) for q, value in zip(quantiles, predictive_q)
        },
        "predictive_over_truth_quantile_ratio": {
            str(q): float(value) for q, value in zip(quantiles, ratios)
        },
        "event_bootstrap_95pct_interval_for_ratio": {
            str(q): [float(value) for value in interval]
            for q, interval in zip(quantiles, np.quantile(bootstrap, [0.025, 0.975], axis=0).T)
        },
        "mttbar_wasserstein_distance_GeV": float(wasserstein),
        "posterior_predictive_draws_per_event": int(posterior_mtt_values.shape[1]),
        "bootstrap_resamples": len(bootstrap),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Truth-free measured HDF5")
    parser.add_argument("--truth", type=Path, required=True, help="Separate keyed truth sidecar")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-events", type=int, default=None, help="Deterministic validation subset for preflight")
    parser.add_argument("--seed", type=int, default=42011)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.samples < 2 or args.batch_size <= 0 or (args.max_events is not None and args.max_events <= 0):
        parser.error("Require at least two samples and positive batch/event counts")
    for path in (args.config, args.research, args.checkpoint, args.input, args.truth):
        if not path.is_file():
            parser.error(f"Required transfer input does not exist: {path}")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite KNOT transfer output: {output}")

    torch.set_num_threads(1)
    started_utc = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    config = load_experiment(args.config)
    research = OmegaConf.to_container(OmegaConf.load(args.research), resolve=True)
    variant = str(research["variant"])
    if variant not in {"M", "C", "J"}:
        raise ValueError(f"Frozen transfer requires KNOT M/C/J, found {variant!r}")
    validate_frozen_configuration(
        argparse.Namespace(variant=variant, config=args.config, research=args.research),
        config,
        research,
        json.loads(FREEZE_PATH.read_text()),
    )
    if not config.data.input or not config.data.tensor_cache:
        raise ValueError("Frozen training input and cache identity must remain present during validation")

    ids = validate_transfer_files(args.input, args.truth)
    config.data.inference_input = str(args.input.expanduser().resolve())
    # An external event sample has no production tensor cache. Rebuild its
    # measured tokens while retaining the frozen transforms and checkpoint.
    config.data.tensor_cache = None
    visible = load_visible_config(config)
    visible.dataset.cache.enabled = False
    visible.dataset.cache.path = None
    data = build_data(config, visible, include_truth=False, device="cpu", complete_assignments=True)
    data.setup("predict")
    all_ids = data._selected_ids("validation", np.arange(len(data.val_dataset), dtype=np.int64))
    if not np.array_equal(all_ids, ids):
        raise ValueError("KNOT measured-input validation split or EVENT_ID order differs from the transfer contract")
    if args.max_events is None:
        positions = np.arange(len(data.val_dataset), dtype=np.int64)
    else:
        from joint_event_reconstruction.data.subsets import (
            deterministic_subset_positions,
        )

        positions = deterministic_subset_positions(all_ids, args.max_events, args.seed + 200)
    selected_ids = all_ids[positions]
    loader = DataLoader(
        Subset(data.val_dataset, positions.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data.structured_collator,
        num_workers=0,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    invisible_scaler = checkpoint.get("invisible_scaler")
    if invisible_scaler is not None:
        data.invisible_scaler = InvisibleScaler.from_dict(invisible_scaler)
        data.structured_collator.invisible_scaler = data.invisible_scaler
    partonic_scaler = checkpoint.get("partonic_scaler")
    if partonic_scaler is not None:
        data.partonic_scaler = PartonicScaler.from_dict(partonic_scaler)
        data.structured_collator.partonic_scaler = data.partonic_scaler
    module = make_module(config, visible, research, data)
    module.validate_checkpoint_metadata(checkpoint)
    module.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA transfer was requested but CUDA is unavailable")
    module.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    predictions = []
    for batch_index, host_batch in enumerate(loader):
        clean = truth_free_batch(move_to_device(host_batch, device))
        predictions.append(module.posterior(clean, samples=args.samples, generator=generator))
        if batch_index % 20 == 0:
            print(f"KNOT-{variant} transfer batch {batch_index + 1}/{len(loader)}", flush=True)
    posterior = concatenate(predictions)
    event_ids = as_numpy(posterior.event_id).astype(np.int64)
    if not np.array_equal(event_ids, selected_ids):
        raise ValueError("KNOT posterior event order differs from the selected transfer EVENT_ID values")

    # Join truth only after the truth-free prediction pass has completed.
    with h5py.File(args.truth, "r") as truth_file:
        truth_ids = np.asarray(truth_file["EVENT_ID"][:], dtype=np.int64)
        if args.max_events is None:
            truth_rows = np.arange(len(truth_ids), dtype=np.int64)
        else:
            row_for_id = {int(event_id): row for row, event_id in enumerate(truth_ids)}
            truth_rows = np.asarray([row_for_id[int(event_id)] for event_id in event_ids], dtype=np.int64)
        truth_p = np.asarray(truth_file["TRUTH/parents"][:], dtype=np.float64)[truth_rows]
        truth_n3 = np.asarray(truth_file["TRUTH/neutrinos"][:], dtype=np.float64)[truth_rows]
    if not np.array_equal(truth_ids[truth_rows], event_ids):
        raise ValueError("Truth sidecar EVENT_ID join did not reproduce posterior event order")
    truth_n = np.concatenate(
        (np.linalg.norm(truth_n3, axis=-1, keepdims=True), truth_n3), axis=-1
    )
    neutrinos = as_numpy(posterior.invisible).astype(np.float64)
    parents = as_numpy(posterior.parents).astype(np.float64)
    if not np.isfinite(truth_p).all() or not np.isfinite(truth_n).all():
        raise ValueError("Joined after-FSR parent or direct-neutrino truth contains non-finite values")
    metrics = reconstruction_metrics(neutrinos, parents, truth_n, truth_p)
    calibration = physical_calibration_features(neutrinos, parents, truth_n, truth_p)
    metrics["posterior_calibration"] = calibration
    metrics["joint_posterior"] = joint_metric(neutrinos, parents, seed=args.seed + 1)
    metrics["joint_posterior"]["energy_score"] = joint_energy_score(
        neutrinos, parents, truth_n, truth_p, seed=args.seed + 1
    )
    truth_mtt = mass(truth_p.sum(axis=1))
    pred_mtt = posterior_mtt(parents)
    metrics["mttbar_tail_closure"] = tail_closure(truth_mtt, pred_mtt, args.seed + 2)
    top_decile = truth_mtt >= np.quantile(truth_mtt, 0.90)
    metrics["truth_mttbar_top_decile"] = {
        "events": int(top_decile.sum()),
        "truth_mttbar_threshold_GeV": float(np.quantile(truth_mtt, 0.90)),
        "parent_mttbar_mean_bias_GeV": float((pred_mtt.mean(axis=1)[top_decile] - truth_mtt[top_decile]).mean()),
        "parent_mttbar_RMSE_GeV": float(np.sqrt(np.mean((pred_mtt.mean(axis=1)[top_decile] - truth_mtt[top_decile]) ** 2))),
    }

    with h5py.File(args.input, "r") as measured_file:
        measured_root_sha = str(measured_file.attrs["source_root_sha256"])
        event_weights = np.asarray(measured_file["META/weight_mc_NOSYS"][:], dtype=np.float64)
    if args.max_events is not None:
        weight_lookup = {int(event_id): float(weight) for event_id, weight in zip(ids, event_weights)}
        event_weights = np.asarray([weight_lookup[int(event_id)] for event_id in event_ids])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        posterior.write(staging / "posterior.h5")
        metrics_payload = {
            "variant": variant,
            "posterior_semantics": {
                "M": "independent_invisible_and_parent_posteriors",
                "C": "invisible_posterior_then_assignment_and_neutrino_conditioned_parent_posterior",
                "J": "joint_invisible_and_parent_posterior",
            }[variant],
            "events": len(event_ids),
            "samples_per_event": int(args.samples),
            "inference": {"steps": int(research["steps"]), "solver": str(research["solver"])},
            "event_id_sha256_ordered_little_endian_int64": sha256_ids(event_ids),
            "truth_event_id_set_sha256_sorted_little_endian_int64": sha256_ids(np.sort(event_ids)),
            "sealed_test_used": False,
            "assignment_accuracy": "scored separately only when explicit after-FSR jet labels and support are available",
            "reconstruction": metrics,
            "unweighted_event_metrics": True,
            "generator_weights": {
                "branch": "META/weight_mc_NOSYS",
                "sumw": float(event_weights.sum()),
                "sum_absw": float(np.abs(event_weights).sum()),
                "negative_weight_events": int((event_weights < 0).sum()),
                "absolute_weight_effective_events": float(
                    np.abs(event_weights).sum() ** 2 / np.square(event_weights).sum()
                ),
                "used_for_reported_metrics": False,
            },
            "runtime_seconds": float(time.perf_counter() - started),
        }
        (staging / "metrics.json").write_text(json.dumps(metrics_payload, indent=2, allow_nan=False) + "\n")
        manifest = {
            "variant": variant,
            "training_source_commit": json.loads(FREEZE_PATH.read_text())["source_commit"],
            "evaluation_source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "evaluation_worktree_root": str(ROOT),
            "evaluation_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT, text=True).strip()),
            "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
            "measured_input": str(args.input.resolve()),
            "measured_input_sha256": sha256_file(args.input.resolve()),
            "truth_sidecar": str(args.truth.resolve()),
            "truth_sidecar_sha256": sha256_file(args.truth.resolve()),
            "source_root_sha256": measured_root_sha,
            "events": len(event_ids),
            "samples_per_event": int(args.samples),
            "seed": int(args.seed),
            "sealed_test_used": False,
            "truth_used_during_inference": False,
            "started_utc": started_utc,
            "finished_utc": datetime.now(UTC).isoformat(),
            "wall_seconds": float(time.perf_counter() - started),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "device": str(device),
        }
        (staging / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with h5py.File(staging / "posterior.h5", "r") as saved:
            if not np.array_equal(saved["EVENT_ID"][:], event_ids):
                raise ValueError("Staged posterior EVENT_ID validation failed")
            for name in ("invisible", "parents"):
                if not np.isfinite(saved[name][:]).all():
                    raise ValueError(f"Staged posterior {name} contains non-finite values")
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps({"variant": variant, "output": str(output), "events": len(event_ids), "samples": args.samples}, indent=2))


if __name__ == "__main__":
    main()
