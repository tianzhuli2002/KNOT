#!/usr/bin/env python3
"""Test transverse-rotation response with fixed parent source noise."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
SOURCE_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = PROJECT_ROOT / (
    "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/"
    "realnu_top_antitop_20260927_release2"
)
EVENT_TABLE = PRODUCTION_ROOT / "evaluation_r2/core/event_metrics.h5"
OUTPUT_PATH = PRODUCTION_ROOT / "evaluation_v3_diagnostics/rotation_tests/rotation_covariance.json"
MODES = ("PCFM_Q", "J0", "J_A", "J_AN")
ROTATION_ANGLES = (np.pi / 6.0, np.pi / 3.0, np.pi / 2.0)
CONTEXT_EVENTS = 2048
SAMPLE_EVENTS = 64


def _sha256_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _rotation_matrix(angle: float, *, device, dtype) -> torch.Tensor:
    cosine, sine = np.cos(angle), np.sin(angle)
    return torch.tensor([[cosine, -sine], [sine, cosine]], device=device, dtype=dtype)


def _ensure_met_xy(batch: dict, met_type_id: int) -> dict:
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    type_ids = batch["type_ids"]
    if type_ids.dtype not in integer_dtypes:
        raise TypeError(f"Rotation fixture type_ids must be integer, got {type_ids.dtype}")
    if "met_xy" in batch and type_ids.dtype == torch.int64:
        return batch
    vectors = batch["physics_vectors"]
    type_ids = type_ids.to(dtype=torch.int64)
    mask = type_ids == int(met_type_id)
    counts = mask.sum(dim=1)
    if not torch.equal(counts, torch.ones_like(counts)):
        raise ValueError("Rotation fixture requires exactly one measured MET four-vector per event")
    result = dict(batch)
    result["type_ids"] = type_ids
    result["met_xy"] = (vectors[..., 1:3] * mask.unsqueeze(-1)).sum(dim=1)
    return result


def _rotate_batch(batch: dict, angle: float) -> dict:
    rotated = dict(batch)
    vectors = batch["physics_vectors"].clone()
    matrix = _rotation_matrix(angle, device=vectors.device, dtype=vectors.dtype)
    vectors[..., 1:3] = vectors[..., 1:3] @ matrix.T
    rotated["physics_vectors"] = vectors
    if "met_xy" not in batch:
        raise KeyError("Transverse rotation requires measured MET in met_xy")
    rotated["met_xy"] = batch["met_xy"] @ matrix.T
    # The accepted physical-encoder contract has no coordinate-dependent
    # scalar kinematics: p4 and MET are the rotated measured-vector channels.
    rotated["tokens"] = batch["tokens"]
    return rotated


def _slice_batch(value, length: int, full_length: int):
    if torch.is_tensor(value) and value.ndim and value.shape[0] == full_length:
        return value[:length]
    if isinstance(value, dict):
        return {key: _slice_batch(item, length, full_length) for key, item in value.items()}
    if isinstance(value, list):
        return [_slice_batch(item, length, full_length) for item in value]
    if isinstance(value, tuple):
        return tuple(_slice_batch(item, length, full_length) for item in value)
    return value


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("Rotation diagnostic produced no finite comparison values")
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "rms": float(np.sqrt(np.mean(values * values))),
        "median": float(np.median(values)),
        "q16": float(np.quantile(values, 0.16)),
        "q84": float(np.quantile(values, 0.84)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _observables(parents: torch.Tensor) -> dict[str, torch.Tensor]:
    pair = parents[:, 0] + parents[:, 1]
    return {
        "top_phi": torch.atan2(parents[:, 0, 2], parents[:, 0, 1]),
        "antitop_phi": torch.atan2(parents[:, 1, 2], parents[:, 1, 1]),
        "top_pt": torch.linalg.vector_norm(parents[:, 0, 1:3], dim=-1),
        "antitop_pt": torch.linalg.vector_norm(parents[:, 1, 1:3], dim=-1),
        "top_mass": (parents[:, 0, 0].square() - parents[:, 0, 1:].square().sum(-1)).clamp_min(0).sqrt(),
        "antitop_mass": (parents[:, 1, 0].square() - parents[:, 1, 1:].square().sum(-1)).clamp_min(0).sqrt(),
        "mttbar": (pair[:, 0].square() - pair[:, 1:].square().sum(-1)).clamp_min(0).sqrt(),
        "ttbar_pt": torch.linalg.vector_norm(pair[:, 1:3], dim=-1),
    }


def _worker_mode(mode: str, selected_indices: np.ndarray, expected_ids: np.ndarray) -> tuple[dict, dict]:
    os.environ["JER_PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["JER_REALNU_PROJECT_ROOT"] = str(PROJECT_ROOT)
    os.environ["JER_REALNU_SOURCE_ROOT"] = str(SOURCE_ROOT)
    sys.path.insert(0, str(SOURCE_ROOT / "src"))
    sys.path.insert(0, str(SOURCE_ROOT / "scripts/production"))
    from joint_event_reconstruction.cli.run import build_data, load_trained, load_visible_config
    import run_realnu_production as runner

    config, _, _ = runner.load_and_validate_config(
        mode, runner.MODE_CONFIGS[mode], PRODUCTION_ROOT / mode
    )
    visible_config = load_visible_config(config)
    visible_config.general.mode = "predict"
    visible_config.dataset.val_batch_size = 256
    visible_config.dataset.val_num_workers = 0
    visible_config.dataset.pin_memory = False
    visible_config.dataset.persistent_workers = False
    coordinate_scalars = {"e", "pt", "eta", "phi", "sin_phi", "cos_phi"}
    configured_scalars = set(config.model.visible.inputs.object_features) | set(
        config.model.visible.inputs.event_features
    )
    if configured_scalars & coordinate_scalars:
        raise ValueError(
            f"{mode}: rotation fixture must rotate coordinate-dependent scalar features too: "
            f"{sorted(configured_scalars & coordinate_scalars)}"
        )
    data = build_data(config, visible_config, include_truth=False, device="cpu")
    data.setup("predict")
    selected_dataset = Subset(data.val_dataset, selected_indices.tolist())
    loader = DataLoader(
        selected_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        collate_fn=data.structured_collator,
    )
    met_type_id = int(data.schema.source_type_ids["MET"])
    record = json.loads((PRODUCTION_ROOT / mode / "production_training.json").read_text())
    checkpoint = Path(record["checkpoint"]).resolve()
    module = load_trained(config, visible_config, data, str(checkpoint)).cpu().eval()
    event_model = module.event_model

    context_ids = []
    context_comparisons = {key: [] for key in ("encoder_pooled", "tokens", "query_states", "query_context", "parent_role_context", "invisible_role_context")}
    first_batch = None
    first_visible = None
    with torch.no_grad():
        for batch in loader:
            batch = _ensure_met_xy(batch, met_type_id)
            ids = batch["event_index"].cpu().numpy().astype(np.int64, copy=False)
            context_ids.extend(ids.tolist())
            rotated = _rotate_batch(batch, float(ROTATION_ANGLES[0]))
            original_visible = event_model.encode_visible(batch)
            rotated_visible = event_model.encode_visible(rotated)
            for key in context_comparisons:
                left = original_visible.get(key)
                right = rotated_visible.get(key)
                if left is None and right is None:
                    continue
                if left is None or right is None or left.shape != right.shape:
                    raise ValueError(f"{mode}: visible tensor {key!r} changed shape under rotation")
                context_comparisons[key].append((left - right).detach().cpu().numpy().reshape(-1))
            if first_batch is None:
                first_batch = batch
                first_visible = original_visible
    context_ids = np.asarray(context_ids, dtype=np.int64)
    if not np.array_equal(context_ids, expected_ids):
        raise ValueError(f"{mode}: validation EVENT_ID selection/order differs from fixed fixture")
    context_summary = {
        name: {
            "max_abs": float(np.max(np.abs(np.concatenate(chunks)))) if chunks else 0.0,
            "rms": float(np.sqrt(np.mean(np.square(np.concatenate(chunks))))) if chunks else 0.0,
        }
        for name, chunks in context_comparisons.items()
    }

    if first_batch is None or first_visible is None:
        raise RuntimeError(f"{mode}: selected validation subset is empty")
    full_batch_length = len(first_batch["tokens"])
    sample_batch = _slice_batch(first_batch, SAMPLE_EVENTS, full_batch_length)
    parent_source = torch.randn(
        (SAMPLE_EVENTS, 1, 8), generator=torch.Generator(device="cpu").manual_seed(20260929)
    )
    angle_records = {}
    with torch.no_grad():
        original = event_model.sample_partonic(
            sample_batch,
            samples=1,
            steps=8,
            solver="heun",
            generator=torch.Generator(device="cpu").manual_seed(918273),
            parent_source_state=parent_source,
        )
        original_parents = original["parents"][:, 0]
        original_observables = _observables(original_parents)
        for angle in ROTATION_ANGLES:
            rotated_batch = _rotate_batch(sample_batch, float(angle))
            rotated = event_model.sample_partonic(
                rotated_batch,
                samples=1,
                steps=8,
                solver="heun",
                generator=torch.Generator(device="cpu").manual_seed(918273),
                parent_source_state=parent_source,
            )
            rotated_parents = rotated["parents"][:, 0]
            rotated_observables = _observables(rotated_parents)
            checks = {}
            for role in ("top", "antitop"):
                covariance_error = (
                    rotated_observables[f"{role}_phi"]
                    - original_observables[f"{role}_phi"]
                    - float(angle)
                )
                covariance_error = torch.remainder(covariance_error + np.pi, 2.0 * np.pi) - np.pi
                checks[f"{role}_phi_covariance_error_rad"] = _stats(covariance_error.cpu().numpy())
                checks[f"{role}_pt_invariance_error_gev"] = _stats(
                    (rotated_observables[f"{role}_pt"] - original_observables[f"{role}_pt"]).cpu().numpy()
                )
                checks[f"{role}_mass_invariance_error_gev"] = _stats(
                    (rotated_observables[f"{role}_mass"] - original_observables[f"{role}_mass"]).cpu().numpy()
                )
            checks["mttbar_invariance_error_gev"] = _stats(
                (rotated_observables["mttbar"] - original_observables["mttbar"]).cpu().numpy()
            )
            checks["ttbar_pt_invariance_error_gev"] = _stats(
                (rotated_observables["ttbar_pt"] - original_observables["ttbar_pt"]).cpu().numpy()
            )
            angle_records[str(angle)] = checks
    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "n_context_events": int(len(context_ids)),
        "n_sample_events": SAMPLE_EVENTS,
        "context_EVENT_ID_sha256": _sha256_ids(context_ids),
        "source_state": "same explicit [B,1,8] FP32 CPU source for original and rotated input",
        "rotation_angles_rad": list(ROTATION_ANGLES),
        "parent_sampling": {"solver": "heun", "steps": 8, "nfe": 16},
        "context_invariance": context_summary,
        "sampled_parent_covariance": angle_records,
    }
    del module, data
    return metadata, {"event_ids": context_ids}


def run(output_path: Path = OUTPUT_PATH) -> dict:
    torch.set_num_threads(4)
    if not EVENT_TABLE.is_file():
        raise FileNotFoundError(f"Accepted R2 event table is missing: {EVENT_TABLE}")
    with h5py.File(EVENT_TABLE, "r") as source:
        if source.attrs.get("schema") != "realnu_event_metrics_v1":
            raise ValueError("Accepted R2 event table has an unexpected schema")
        if bool(source.attrs.get("sealed_test_used", True)):
            raise ValueError("Accepted R2 event table does not prove sealed_test_used=false")
        all_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        truth_mttbar = np.asarray(source["truth/mttbar"][:], dtype=np.float64)
        if len(all_ids) != 100_342 or len(np.unique(all_ids)) != len(all_ids):
            raise ValueError("Expected 100,342 unique validation EVENT_IDs")
    # Systematic truth-mass quantiles give a deterministic subset spanning the
    # full finite truth range without selecting by model response.
    finite = np.isfinite(truth_mttbar)
    order = np.flatnonzero(finite)[np.argsort(truth_mttbar[finite], kind="stable")]
    positions = np.linspace(0, len(order) - 1, CONTEXT_EVENTS, dtype=np.int64)
    selected_indices = order[positions]
    expected_ids = all_ids[selected_indices]
    per_mode = {}
    all_sample_ids = None
    for mode in MODES:
        metadata, products = _worker_mode(mode, selected_indices, expected_ids)
        per_mode[mode] = metadata
        if all_sample_ids is None:
            all_sample_ids = products["event_ids"][:SAMPLE_EVENTS]
        elif not np.array_equal(all_sample_ids, products["event_ids"][:SAMPLE_EVENTS]):
            raise ValueError("Cross-model sample subset EVENT_IDs differ")
    result = {
        "status": "passed",
        "diagnostic": "transverse_rotation_covariance",
        "validation_events": CONTEXT_EVENTS,
        "sampled_events_per_model": SAMPLE_EVENTS,
        "selection": "2,048 deterministic validation EVENT_IDs at uniform truth-mttbar quantiles; first 64 selected IDs for fixed-source parent sampling",
        "event_id_sha256": _sha256_ids(expected_ids),
        "sample_event_id_sha256": _sha256_ids(np.asarray(all_sample_ids)),
        "parent_roles": ["top", "antitop"],
        "sealed_test_used": False,
        "models": per_mode,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    print(json.dumps(run(args.output.expanduser().resolve()), indent=2))


if __name__ == "__main__":
    main()
