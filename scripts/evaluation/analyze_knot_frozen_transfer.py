#!/usr/bin/env python3
"""Add after-FSR response and, when labeled, exact assignment metrics to a saved transfer posterior."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def invariant_mass(p4: np.ndarray, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    mass2 = p4[..., 0] ** 2 - np.square(p4[..., 1:]).sum(axis=-1)
    if not np.isfinite(mass2).all():
        raise ValueError(f"{name} contains non-finite invariant mass squared")
    if np.any(mass2 < -1e-5):
        first = np.argwhere(mass2 < -1e-5)[0].tolist()
        raise ValueError(f"{name} contains a spacelike four-vector at index {first}")
    return np.sqrt(np.maximum(mass2, 0.0)), mass2


def eta_from_p4(p4: np.ndarray, *, name: str) -> np.ndarray:
    pt = np.hypot(p4[..., 1], p4[..., 2])
    if not np.isfinite(pt).all() or np.any(pt <= 0):
        raise ValueError(f"{name} contains a non-finite or zero-pT four-vector")
    return np.arcsinh(p4[..., 3] / pt)


def paired_bootstrap(values: dict[str, np.ndarray], seed: int, n_resamples: int = 1000) -> dict:
    lengths = {len(value) for value in values.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) < 2:
        raise ValueError("Paired bootstrap inputs must have the same length >= 2")
    n_events = next(iter(lengths))
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(n_resamples, dtype=np.float64) for name in values}
    for index in range(n_resamples):
        rows = rng.integers(0, n_events, size=n_events)
        for name, value in values.items():
            samples[name][index] = value[rows].mean()
    return {
        name: [float(value) for value in np.quantile(sample, [0.025, 0.975])]
        for name, sample in samples.items()
    }


def regression(truth: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    centered = truth - truth.mean()
    variance = np.mean(centered**2)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Truth has no finite variance for response regression")
    slope = float(np.mean(centered * (estimate - estimate.mean())) / variance)
    intercept = float(estimate.mean() - slope * truth.mean())
    return slope, intercept


def response_diagnostics(posterior_path: Path, truth_path: Path, seed: int) -> dict:
    with h5py.File(posterior_path, "r") as posterior, h5py.File(truth_path, "r") as truth:
        required_posterior = ("EVENT_ID", "parents")
        required_truth = ("EVENT_ID", "TRUTH/parents")
        missing = [name for name in required_posterior if name not in posterior]
        missing += [name for name in required_truth if name not in truth]
        if missing:
            raise KeyError(f"Missing transfer posterior/truth fields: {missing}")
        ids = np.asarray(posterior["EVENT_ID"][:], dtype=np.int64)
        truth_ids = np.asarray(truth["EVENT_ID"][:], dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError("Posterior EVENT_ID must be one-dimensional and unique")
        if not np.array_equal(ids, truth_ids):
            raise ValueError("Saved posterior and after-FSR truth sidecar EVENT_ID order differs")
        parents = np.asarray(posterior["parents"][:], dtype=np.float64)
        truth_parent = np.asarray(truth["TRUTH/parents"][:], dtype=np.float64)
        roles = json.loads(posterior.attrs["parent_roles"])
        if roles != ["top", "antitop"]:
            raise ValueError(f"Transfer posterior parent role order is not top,antitop: {roles}")
        if parents.ndim != 4 or parents.shape[0] != len(ids) or parents.shape[2:] != (2, 4):
            raise ValueError(f"Invalid posterior parents shape: {parents.shape}")
        if truth_parent.shape != (len(ids), 2, 4):
            raise ValueError(f"Invalid after-FSR truth parent shape: {truth_parent.shape}")
        if not np.isfinite(parents).all() or not np.isfinite(truth_parent).all():
            raise ValueError("Parent posterior or after-FSR truth contains non-finite values")

    truth_mtt, truth_mtt2 = invariant_mass(truth_parent.sum(axis=1), name="after-FSR truth mtt")
    prediction_mtt, prediction_mtt2 = invariant_mass(
        parents.sum(axis=2), name="posterior mtt draws"
    )
    point_mtt = prediction_mtt.mean(axis=1)
    residual = point_mtt - truth_mtt
    slope, intercept = regression(truth_mtt, point_mtt)

    # Bootstrap RMSE and response slope/intercept from the same event resamples.
    rng = np.random.default_rng(seed)
    n_resamples = 1000
    boot_bias = np.empty(n_resamples)
    boot_rmse = np.empty(n_resamples)
    boot_slope = np.empty(n_resamples)
    boot_intercept = np.empty(n_resamples)
    for index in range(n_resamples):
        rows = rng.integers(0, len(ids), size=len(ids))
        x, y = truth_mtt[rows], point_mtt[rows]
        boot_bias[index] = np.mean(y - x)
        boot_rmse[index] = np.sqrt(np.mean((y - x) ** 2))
        boot_slope[index], boot_intercept[index] = regression(x, y)

    parent_rows = {}
    for side_index, side in enumerate(("top", "antitop")):
        pred_p4 = parents[:, :, side_index]
        truth_p4 = truth_parent[:, side_index]
        pred_pt = np.hypot(pred_p4[..., 1], pred_p4[..., 2]).mean(axis=1)
        truth_pt = np.hypot(truth_p4[..., 1], truth_p4[..., 2])
        pred_eta = eta_from_p4(pred_p4, name=f"{side} posterior parents").mean(axis=1)
        truth_eta = eta_from_p4(truth_p4, name=f"{side} after-FSR truth parents")
        pt_residual = pred_pt - truth_pt
        eta_residual = pred_eta - truth_eta
        boot = paired_bootstrap(
            {
                "pt_bias_GeV": pt_residual,
                "pt_squared_error_GeV2": pt_residual**2,
                "eta_bias": eta_residual,
                "eta_squared_error": eta_residual**2,
            },
            seed + side_index + 1,
            n_resamples,
        )
        parent_rows[side] = {
            "parent_pT_posterior_mean_vs_truth": {
                "bias_GeV": float(pt_residual.mean()),
                "RMSE_GeV": float(np.sqrt(np.mean(pt_residual**2))),
                "event_bootstrap_95pct_interval_for_bias_GeV": boot["pt_bias_GeV"],
                "event_bootstrap_95pct_interval_for_RMSE_GeV": [
                    float(np.sqrt(max(value, 0.0)))
                    for value in boot["pt_squared_error_GeV2"]
                ],
            },
            "parent_eta_posterior_mean_vs_truth": {
                "bias": float(eta_residual.mean()),
                "RMSE": float(np.sqrt(np.mean(eta_residual**2))),
                "event_bootstrap_95pct_interval_for_bias": boot["eta_bias"],
                "event_bootstrap_95pct_interval_for_RMSE": [
                    float(np.sqrt(max(value, 0.0)))
                    for value in boot["eta_squared_error"]
                ],
            },
        }

    fixed_edges = np.asarray(
        [0, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, np.inf],
        dtype=np.float64,
    )
    differential = []
    for bin_index, (low, high) in enumerate(pairwise(fixed_edges)):
        rows = np.flatnonzero((truth_mtt >= low) & (truth_mtt < high))
        if not len(rows):
            differential.append({"bin": bin_index, "truth_mtt_range_GeV": [float(low), None if np.isinf(high) else float(high)], "events": 0})
            continue
        bin_error = residual[rows]
        rng_bin = np.random.default_rng(seed + 100 + bin_index)
        boot_bias_bin = np.empty(n_resamples)
        boot_rmse_bin = np.empty(n_resamples)
        for index in range(n_resamples):
            sample_rows = rows[rng_bin.integers(0, len(rows), size=len(rows))]
            errors = residual[sample_rows]
            boot_bias_bin[index] = errors.mean()
            boot_rmse_bin[index] = np.sqrt(np.mean(errors**2))
        differential.append(
            {
                "bin": bin_index,
                "truth_mtt_range_GeV": [float(low), None if np.isinf(high) else float(high)],
                "events": len(rows),
                "mean_truth_mtt_GeV": float(truth_mtt[rows].mean()),
                "posterior_mean_mtt_bias_GeV": float(bin_error.mean()),
                "posterior_mean_mtt_RMSE_GeV": float(np.sqrt(np.mean(bin_error**2))),
                "event_bootstrap_95pct_interval_for_bias_GeV": [float(x) for x in np.quantile(boot_bias_bin, [0.025, 0.975])],
                "event_bootstrap_95pct_interval_for_RMSE_GeV": [float(x) for x in np.quantile(boot_rmse_bin, [0.025, 0.975])],
            }
        )

    tail = {}
    for q in (0.90, 0.99):
        truth_q = float(np.quantile(truth_mtt, q))
        predictive_q = float(np.quantile(prediction_mtt.reshape(-1), q))
        boot_ratio = np.empty(n_resamples)
        rng_tail = np.random.default_rng(seed + int(q * 1000))
        for index in range(n_resamples):
            rows = rng_tail.integers(0, len(ids), size=len(ids))
            boot_ratio[index] = (
                np.quantile(prediction_mtt[rows].reshape(-1), q)
                / np.quantile(truth_mtt[rows], q)
            )
        tail[str(q)] = {
            "truth_quantile_GeV": truth_q,
            "posterior_predictive_quantile_GeV": predictive_q,
            "predictive_over_truth_ratio": predictive_q / truth_q,
            "event_bootstrap_95pct_interval_for_ratio": [float(x) for x in np.quantile(boot_ratio, [0.025, 0.975])],
        }

    return {
        "events": len(ids),
        "samples_per_event": int(parents.shape[1]),
        "event_id_ordered_sha256": hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
        "truth_used_during_inference": False,
        "point_estimator": "event-wise posterior mean of mtt computed from posterior parent draws",
        "mttbar_posterior_mean_response": {
            "bias_GeV": float(residual.mean()),
            "RMSE_GeV": float(np.sqrt(np.mean(residual**2))),
            "OLS_estimate_vs_truth_slope": slope,
            "OLS_estimate_vs_truth_intercept_GeV": intercept,
            "event_bootstrap_95pct_interval": {
                "bias_GeV": [float(x) for x in np.quantile(boot_bias, [0.025, 0.975])],
                "RMSE_GeV": [float(x) for x in np.quantile(boot_rmse, [0.025, 0.975])],
                "OLS_slope": [float(x) for x in np.quantile(boot_slope, [0.025, 0.975])],
                "OLS_intercept_GeV": [float(x) for x in np.quantile(boot_intercept, [0.025, 0.975])],
            },
            "differential_fixed_truth_mtt_bins": differential,
            "posterior_predictive_tail_closure": tail,
            "spacelike_truth_vectors": int(np.count_nonzero(truth_mtt2 < 0)),
            "spacelike_posterior_draws": int(np.count_nonzero(prediction_mtt2 < 0)),
        },
        "parents": parent_rows,
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": int(seed),
    }


def assignment_diagnostics(args: argparse.Namespace) -> dict:
    with h5py.File(args.truth, "r") as truth:
        has_labels = "LABELS/JET" in truth and "SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE" in truth
    if not has_labels:
        return {
            "status": "unavailable",
            "reason": (
                f"{args.truth} has no explicit LABELS/JET and reconstructable-support contract; "
                "no assignment was inferred from diagnostic candidate branches"
            ),
        }

    import torch
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader, Subset

    from joint_event_reconstruction.cli.run import build_data, load_visible_config
    from joint_event_reconstruction.configuration import load_experiment
    from joint_event_reconstruction.data.invisible import InvisibleScaler
    from joint_event_reconstruction.data.partonic import PartonicScaler
    from joint_event_reconstruction.data.structured import VisibleTruthStore
    from scripts.run_knot import (
        FREEZE_PATH,
        make_module,
        move_to_device,
        truth_free_batch,
        validate_frozen_configuration,
    )

    for path in (args.config, args.research, args.checkpoint, args.measured, args.truth):
        if path is None or not path.is_file():
            raise FileNotFoundError(f"Required exact-assignment input is absent: {path}")
    torch.set_num_threads(1)
    config = load_experiment(args.config)
    research = OmegaConf.to_container(OmegaConf.load(args.research), resolve=True)
    variant = str(research["variant"])
    if variant not in {"M", "C", "J"}:
        raise ValueError(f"Exact transfer assignment scoring requires KNOT M/C/J, found {variant!r}")
    validate_frozen_configuration(
        argparse.Namespace(variant=variant, config=args.config, research=args.research),
        config,
        research,
        json.loads(FREEZE_PATH.read_text()),
    )
    config.data.inference_input = str(args.measured.resolve())
    config.data.tensor_cache = None
    visible = load_visible_config(config)
    visible.dataset.cache.enabled = False
    visible.dataset.cache.path = None
    data = build_data(config, visible, include_truth=False, device="cpu", complete_assignments=True)
    data.setup("predict")
    all_ids = data._selected_ids("validation", np.arange(len(data.val_dataset), dtype=np.int64))
    with h5py.File(args.measured, "r") as measured, h5py.File(args.truth, "r") as truth:
        measured_ids = np.asarray(measured["EVENT_ID"][:], dtype=np.int64)
        truth_ids = np.asarray(truth["EVENT_ID"][:], dtype=np.int64)
    if not np.array_equal(all_ids, measured_ids) or not np.array_equal(measured_ids, truth_ids):
        raise ValueError("Exact assignment scorer input, visible loader and truth EVENT_ID orders differ")
    source_names = tuple(sorted({role.source for role in data.structured_collator.spec.visible_leaf_roles}))
    data.structured_collator.visible_store = VisibleTruthStore(args.truth, source_names)
    loader = DataLoader(
        Subset(data.val_dataset, np.arange(len(data.val_dataset), dtype=np.int64).tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data.structured_collator,
        num_workers=0,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("invisible_scaler") is not None:
        data.invisible_scaler = InvisibleScaler.from_dict(checkpoint["invisible_scaler"])
        data.structured_collator.invisible_scaler = data.invisible_scaler
    if checkpoint.get("partonic_scaler") is not None:
        data.partonic_scaler = PartonicScaler.from_dict(checkpoint["partonic_scaler"])
        data.structured_collator.partonic_scaler = data.partonic_scaler
    module = make_module(config, visible, research, data)
    module.validate_checkpoint_metadata(checkpoint)
    module.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA exact assignment scoring was requested but CUDA is unavailable")
    module.to(device).eval()

    scores, targets, supports, role_scores = [], [], [], []
    with torch.inference_mode():
        for batch_index, host_batch in enumerate(loader):
            target = host_batch["structured"]["assignments"]["truth_state"].detach().cpu().numpy()
            support = host_batch["visible_assignment_support"].detach().cpu().numpy().astype(bool)
            clean = truth_free_batch(move_to_device(host_batch, device))
            encoded = module.event_model.encode_visible(clean)
            assignment, space = module.assignment(clean, encoded)
            probability = space.state_probabilities(assignment["assignment_logits"])
            predicted = probability.argmax(dim=1)
            safe_target = torch.as_tensor(target, device=device).clamp_min(0)
            nll = -probability.gather(1, safe_target[:, None]).squeeze(1).clamp_min(1e-30).log()
            predicted_candidates = space.node_indices(predicted)
            truth_candidates = space.node_indices(safe_target)
            scores.append(predicted.detach().cpu().numpy())
            targets.append(target)
            supports.append(support)
            role_scores.append(
                (predicted_candidates == truth_candidates).detach().cpu().numpy()
            )
            scores[-1] = (scores[-1], nll.detach().cpu().numpy())
            if batch_index % 20 == 0:
                print(f"Exact transfer assignment batch {batch_index + 1}/{len(loader)}", flush=True)

    predicted = np.concatenate([row[0] for row in scores])
    nll = np.concatenate([row[1] for row in scores])
    target = np.concatenate(targets)
    support = np.concatenate(supports)
    role_correct = np.concatenate(role_scores)
    if not support.any():
        raise ValueError("Explicit visible truth sidecar has zero assignment-supported events")
    selected = np.flatnonzero(support)
    correct = (predicted[selected] == target[selected]).astype(np.float64)
    selected_nll = nll[selected]
    rng = np.random.default_rng(args.seed + 300)
    boot_acc = np.empty(1000)
    boot_nll = np.empty(1000)
    for index in range(len(boot_acc)):
        rows = rng.integers(0, len(selected), size=len(selected))
        boot_acc[index] = correct[rows].mean()
        boot_nll[index] = selected_nll[rows].mean()
    node_names = list(space.nodes)
    if role_correct.shape[1] != len(node_names):
        raise ValueError("Assignment node diagnostics have an inconsistent role axis")
    return {
        "status": "scored_from_explicit_after_matching_labels",
        "events_total": len(target),
        "supported_events": int(support.sum()),
        "unsupported_events": int((~support).sum()),
        "support_fraction": float(support.mean()),
        "complete_MAP_accuracy": float(correct.mean()),
        "truth_assignment_NLL_nats": float(selected_nll.mean()),
        "complete_MAP_accuracy_event_bootstrap_95pct_interval": [float(x) for x in np.quantile(boot_acc, [0.025, 0.975])],
        "truth_assignment_NLL_event_bootstrap_95pct_interval_nats": [float(x) for x in np.quantile(boot_nll, [0.025, 0.975])],
        "per_node_MAP_accuracy": {
            name: float(role_correct[selected, index].mean())
            for index, name in enumerate(node_names)
        },
        "labels_used_only_after_truth_free_forward": True,
        "bootstrap_resamples": len(boot_acc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--research", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42011)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    for path in (args.posterior, args.measured, args.truth):
        if not path.is_file():
            parser.error(f"Required saved transfer input does not exist: {path}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing transfer diagnostics: {args.output}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    started = time.perf_counter()
    response = response_diagnostics(args.posterior, args.truth, args.seed)
    with h5py.File(args.measured, "r") as measured:
        measured_ids = np.asarray(measured["EVENT_ID"][:], dtype=np.int64)
        with h5py.File(args.posterior, "r") as posterior:
            posterior_ids = np.asarray(posterior["EVENT_ID"][:], dtype=np.int64)
        if not np.array_equal(measured_ids, posterior_ids):
            raise ValueError("Measured and saved posterior EVENT_ID order differs")
        root_hash = str(measured.attrs["source_root_sha256"])
    response["assignment"] = assignment_diagnostics(args)
    response["source_root_sha256"] = root_hash
    response["source_files_sha256"] = {
        "posterior": sha256_file(args.posterior),
        "measured": sha256_file(args.measured),
        "truth": sha256_file(args.truth),
    }
    response["analysis_script_sha256"] = sha256_file(Path(__file__).resolve())
    response["runtime_seconds"] = float(time.perf_counter() - started)
    response["evaluation_worktree_head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(response, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "events": response["events"]}, indent=2))


if __name__ == "__main__":
    main()
