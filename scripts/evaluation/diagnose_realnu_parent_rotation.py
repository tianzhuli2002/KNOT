#!/usr/bin/env python3
"""Check PCFM_Q transverse covariance with the source state co-rotated."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
PRODUCTION_ROOT = PROJECT_ROOT / (
    "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/"
    "realnu_top_antitop_20260927_release2"
)
EVENT_TABLE = PRODUCTION_ROOT / "evaluation_r2/core/event_metrics.h5"
SOURCE_BANK = PRODUCTION_ROOT / (
    "evaluation_v4_parent_comparison/source_bank/parent_source_bank.npz"
)
RECORD = PRODUCTION_ROOT / "PCFM_Q/production_training.json"
ANGLES = (np.pi / 6.0, np.pi / 3.0, np.pi / 2.0)
CONTEXT_EVENTS = 2048
SAMPLE_EVENTS = 64


def _sha256_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rotation_matrix(angle: float, *, device, dtype) -> torch.Tensor:
    cosine, sine = np.cos(angle), np.sin(angle)
    return torch.tensor(
        [[cosine, -sine], [sine, cosine]], device=device, dtype=dtype
    )


def _multivector_rotation_matrix(angle: float) -> np.ndarray:
    """SO(2) action in L-GATr's grade-grouped 16-component blade basis."""
    cosine, sine = np.cos(angle), np.sin(angle)
    vector_action = np.eye(4, dtype=np.float64)
    vector_action[1:3, 1:3] = [[cosine, -sine], [sine, cosine]]
    blades = [()]
    for grade in range(1, 5):
        blades.extend(itertools.combinations(range(4), grade))
    if len(blades) != 16:
        raise RuntimeError("Unexpected L-GATr multivector blade basis")
    action = np.zeros((16, 16), dtype=np.float64)
    for output_index, output_blade in enumerate(blades):
        for input_index, input_blade in enumerate(blades):
            if len(output_blade) != len(input_blade):
                continue
            if not input_blade:
                action[output_index, input_index] = 1.0
            else:
                action[output_index, input_index] = np.linalg.det(
                    vector_action[np.ix_(output_blade, input_blade)]
                )
    return action


def _rotate_batch(batch: dict, angle: float) -> dict:
    rotated = dict(batch)
    vectors = batch["physics_vectors"].clone()
    matrix = _rotation_matrix(angle, device=vectors.device, dtype=vectors.dtype)
    vectors[..., 1:3] = vectors[..., 1:3] @ matrix.T
    rotated["physics_vectors"] = vectors
    if "met_xy" not in batch:
        raise KeyError("Transverse rotation requires measured MET in met_xy")
    rotated["met_xy"] = batch["met_xy"] @ matrix.T
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


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Rotation covariance diagnostic requires finite values")
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "rms": float(np.sqrt(np.mean(values * values))),
        "max_abs": float(np.max(np.abs(values))),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _angle_difference(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.remainder(left - right + np.pi, 2.0 * np.pi) - np.pi


def _rotate_parents(parents: torch.Tensor, angle: float) -> torch.Tensor:
    expected = parents.clone()
    matrix = _rotation_matrix(angle, device=parents.device, dtype=parents.dtype)
    expected[..., 1:3] = parents[..., 1:3] @ matrix.T
    return expected


def _parent_observables(parents: torch.Tensor) -> dict[str, torch.Tensor]:
    if parents.ndim != 3 or parents.shape[1:] != (2, 4):
        raise ValueError(f"Parent four-vectors must have shape [B,2,4], got {tuple(parents.shape)}")
    if not torch.isfinite(parents).all():
        raise ValueError("Parent rotation diagnostic received non-finite four-vectors")
    spatial2 = parents[..., 1:4].square().sum(dim=-1)
    mass = (parents[..., 0].square() - spatial2).clamp_min(0.0).sqrt()
    pt = torch.linalg.vector_norm(parents[..., 1:3], dim=-1)
    phi = torch.atan2(parents[..., 2], parents[..., 1])
    pair = parents.sum(dim=1)
    mttbar = (
        pair[:, 0].square() - pair[:, 1:4].square().sum(dim=-1)
    ).clamp_min(0.0).sqrt()
    return {
        "top_phi": phi[:, 0],
        "antitop_phi": phi[:, 1],
        "top_pt": pt[:, 0],
        "antitop_pt": pt[:, 1],
        "top_mass": mass[:, 0],
        "antitop_mass": mass[:, 1],
        "mttbar": mttbar,
    }


def _observable_differences(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, torch.Tensor]:
    reference_observables = _parent_observables(reference)
    candidate_observables = _parent_observables(candidate)
    result = {
        "top_phi": _angle_difference(
            candidate_observables["top_phi"], reference_observables["top_phi"]
        ),
        "antitop_phi": _angle_difference(
            candidate_observables["antitop_phi"], reference_observables["antitop_phi"]
        ),
    }
    for name in (
        "top_pt", "antitop_pt", "top_mass", "antitop_mass", "mttbar"
    ):
        result[name] = candidate_observables[name] - reference_observables[name]
    return result


def _rotate_source(
    source: torch.Tensor,
    angle: float,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Apply the physical px/py rotation conjugated by the production scaler."""
    if source.ndim != 3 or source.shape[-1] != 8:
        raise ValueError("Parent source must be [B,M,8]")
    scaled = source.reshape(*source.shape[:2], 2, 4)
    physical = scaled * std.to(source).reshape(1, 1, 2, 4) + mean.to(source).reshape(1, 1, 2, 4)
    matrix = _rotation_matrix(angle, device=source.device, dtype=source.dtype)
    physical[..., :2] = physical[..., :2] @ matrix.T
    return ((physical - mean.to(source).reshape(1, 1, 2, 4)) / std.to(source).reshape(1, 1, 2, 4)).reshape_as(source)


def _source_prior_action(
    angle: float, mean: np.ndarray, std: np.ndarray
) -> dict:
    matrix = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float64,
    )
    role_records = []
    for role in range(2):
        mu = np.asarray(mean[role, :2], dtype=np.float64)
        sigma = np.asarray(std[role, :2], dtype=np.float64)
        action = np.diag(1.0 / sigma) @ matrix @ np.diag(sigma)
        offset = np.diag(1.0 / sigma) @ (matrix @ mu - mu)
        covariance = action @ action.T
        role_records.append(
            {
                "role": ("top", "antitop")[role],
                "mean_shift": offset.tolist(),
                "max_abs_mean_shift": float(np.max(np.abs(offset))),
                "covariance_after_rotation": covariance.tolist(),
                "max_abs_covariance_minus_identity": float(
                    np.max(np.abs(covariance - np.eye(2)))
                ),
                "covariance_eigenvalues": np.linalg.eigvalsh(covariance).tolist(),
            }
        )
    return {"angle_rad": float(angle), "roles": role_records}


def run(
    *,
    output: Path,
    config_path: Path | None = None,
    checkpoint: Path | None = None,
    record_path: Path | None = None,
    scaler_checkpoint: Path | None = None,
    context_events: int = CONTEXT_EVENTS,
    sample_events: int = SAMPLE_EVENTS,
    steps: int = 8,
    initialize_from_shared_checkpoint: bool = False,
) -> dict:
    torch.set_num_threads(4)
    if context_events <= 0 or sample_events <= 0 or sample_events > 256:
        raise ValueError("Require 0 < sample_events <= 256")
    if context_events < sample_events or context_events > 100_342:
        raise ValueError("Rotation context selection must cover the requested samples")
    if steps <= 0:
        raise ValueError("Rotation sampler steps must be positive")
    record_path = (
        RECORD
        if record_path is None and not initialize_from_shared_checkpoint
        else record_path.expanduser().resolve()
        if record_path is not None
        else None
    )
    required_paths = [
        (EVENT_TABLE, "accepted validation event table"),
        (SOURCE_BANK, "fixed parent source bank"),
        (SOURCE_BANK.with_name("manifest.json"), "fixed parent source-bank manifest"),
    ]
    if record_path is not None:
        required_paths.append((record_path, "model training record"))
    for path, label in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    record = (
        json.loads(record_path.read_text(encoding="utf-8"))
        if record_path is not None else {}
    )
    if initialize_from_shared_checkpoint and (config_path is None or checkpoint is None):
        raise ValueError(
            "Untrained geometry rotation requires explicit --config and shared --checkpoint"
        )
    config_path = (
        Path(record.get("config_path", record.get("config", "")))
        if config_path is None
        else config_path
    ).expanduser().resolve()
    checkpoint = (
        Path(record["checkpoint"])
        if checkpoint is None
        else checkpoint
    ).expanduser().resolve()
    if not config_path.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(
            f"Model diagnostic config/checkpoint is missing: config={config_path}, checkpoint={checkpoint}"
        )
    checkpoint_sha = str(record.get("checkpoint_sha256", ""))
    if not checkpoint_sha:
        checkpoint_sha = _sha256(checkpoint)
    if initialize_from_shared_checkpoint:
        checkpoint_sha = _sha256(checkpoint)
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        scaler_checkpoint = (
            Path(record["checkpoint"])
            if scaler_checkpoint is None
            else scaler_checkpoint.expanduser().resolve()
        )
        if not scaler_checkpoint.is_file():
            raise FileNotFoundError(
                f"Production scaler checkpoint is missing: {scaler_checkpoint}"
            )
        record.update(
            {
                "config_path": str(config_path),
                "config_sha256": _sha256(config_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "diagnostic_model_status": "untrained geometry model with shared visible/query initialization",
                "scaler_checkpoint": str(scaler_checkpoint),
                "scaler_checkpoint_sha256": _sha256(scaler_checkpoint),
            }
        )
    if _sha256(checkpoint) != checkpoint_sha:
        raise ValueError("Model checkpoint SHA differs from the training record")

    with h5py.File(EVENT_TABLE, "r") as table:
        if table.attrs.get("schema") != "realnu_event_metrics_v1":
            raise ValueError("Accepted validation event table has an unexpected schema")
        if bool(table.attrs.get("sealed_test_used", True)):
            raise ValueError("Rotation diagnostic requires sealed_test_used=false")
        all_ids = np.asarray(table["EVENT_ID"][:], dtype=np.int64)
        truth_mass = np.asarray(table["truth/mttbar"][:], dtype=np.float64)
    if len(all_ids) != 100_342 or len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("Accepted validation table must contain 100,342 unique event IDs")
    finite = np.isfinite(truth_mass)
    mass_order = np.flatnonzero(finite)[np.argsort(truth_mass[finite], kind="stable")]
    positions = np.linspace(0, len(mass_order) - 1, context_events, dtype=np.int64)
    selected = mass_order[positions]
    expected_ids = all_ids[selected]

    with np.load(SOURCE_BANK, allow_pickle=False) as bank:
        bank_ids = np.asarray(bank["EVENT_ID"], dtype=np.int64)
        bank_source = np.asarray(bank["source_state"], dtype=np.float32)
    if not np.array_equal(bank_ids, all_ids):
        raise ValueError("Source bank EVENT_ID order differs from the accepted event table")
    if bank_source.shape != (len(all_ids), 32, 8) or not np.isfinite(bank_source).all():
        raise ValueError("Source bank must be finite [validation events,32,8]")

    sys.path.insert(0, str(ROOT / "scripts/production"))
    from joint_event_reconstruction.cli.run import (
        _load_initial_components,
        build_data,
        load_trained,
        load_visible_config,
    )
    from joint_event_reconstruction.configuration import load_experiment

    config = load_experiment(config_path)
    visible_config = load_visible_config(config)
    visible_config.general.mode = "predict"
    visible_config.dataset.val_batch_size = 256
    visible_config.dataset.val_num_workers = 0
    visible_config.dataset.pin_memory = False
    visible_config.dataset.persistent_workers = False
    data = build_data(config, visible_config, include_truth=False, device="cpu")
    data.setup("predict")
    data.val_dataset = Subset(data.val_dataset, selected.tolist())
    loader = DataLoader(
        data.val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        collate_fn=data.structured_collator,
    )
    if initialize_from_shared_checkpoint:
        from joint_event_reconstruction.data.partonic import PartonicScaler
        from joint_event_reconstruction.training.joint import JointEventModule

        scaler_payload = torch.load(
            scaler_checkpoint, map_location="cpu", weights_only=False
        )
        partonic_scaler = PartonicScaler.from_dict(
            scaler_payload.get("partonic_scaler")
        )
        data.partonic_scaler = partonic_scaler
        data.structured_collator.partonic_scaler = partonic_scaler
        torch.manual_seed(int(config.experiment.seed))
        module = JointEventModule(
            config, visible_config, partonic_scaler=partonic_scaler
        )
        module.hparams["partonic_scaler_fit"] = record["partonic_scaler_fit"]
        module.hparams["invisible_scaler_fit"] = record.get("invisible_scaler_fit")
        _load_initial_components(module, str(checkpoint), ("visible", "queries"))
        module = module.cpu().eval()
    else:
        module = load_trained(config, visible_config, data, str(checkpoint)).cpu().eval()
    event_model = module.event_model
    if event_model.parent_cfm is None or event_model.parent_cfm.conditioning != "P0":
        raise ValueError("Transverse parent rotation requires a P0 parent CFM")
    source_manifest = json.loads(
        SOURCE_BANK.with_name("manifest.json").read_text(encoding="utf-8")
    )
    scaler_fits = {
        "invisible": module.hparams.get("invisible_scaler_fit"),
        "partonic": module.hparams.get("partonic_scaler_fit"),
    }
    scaler_values = {
        "invisible": module.scaler_metadata,
        "partonic": module.partonic_scaler_metadata,
    }
    if (
        _json_sha256({"scaler": scaler_values["partonic"], "fit": scaler_fits["partonic"]})
        != source_manifest["target_scaler_identity"]["partonic_scaler_sha256"]
        or _json_sha256({"scalers": scaler_values, "fit": scaler_fits})
        != source_manifest["target_scaler_identity"]["scaler_identity_sha256"]
    ):
        raise ValueError("Model parent scaler differs from the common-source-bank scaler")
    if data.structured_collator.partonic_store is not None:
        raise RuntimeError("Rotation inference opened a partonic truth store")

    scalar_features = set(config.model.visible.inputs.object_features) | set(
        config.model.visible.inputs.event_features
    )
    coordinate_scalars = {"e", "pt", "eta", "phi", "sin_phi", "cos_phi"}
    if scalar_features & coordinate_scalars:
        raise ValueError(
            "Rotation input fixture has coordinate-dependent scalar channels that need "
            f"their own transformation: {sorted(scalar_features & coordinate_scalars)}"
        )

    context_differences: dict[str, list[np.ndarray]] = {}
    geometric_rotation_differences: dict[str, list[np.ndarray]] = {
        str(float(angle)): [] for angle in ANGLES
    }
    selected_batch = None
    selected_ids = []
    context_id_chunks = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["event_index"].detach().cpu().numpy().astype(np.int64, copy=False)
            context_id_chunks.append(ids)
            original = event_model.encode_visible(batch)
            rotated = event_model.encode_visible(_rotate_batch(batch, ANGLES[0]))
            for name in ("pooled", "tokens", "query_states", "query_context", "parent_role_context"):
                left, right = original.get(name), rotated.get(name)
                if left is None and right is None:
                    continue
                if left is None or right is None or left.shape != right.shape:
                    raise ValueError(f"Visible tensor {name!r} changed shape under rotation")
                context_differences.setdefault(name, []).append(
                    (left - right).detach().cpu().numpy().reshape(-1)
                )
            if selected_batch is None:
                selected_batch = _slice_batch(batch, sample_events, len(ids))
                selected_ids = ids[:sample_events].tolist()
    if selected_batch is None:
        raise RuntimeError("Rotation validation subset is empty")
    if len(context_id_chunks) != (context_events + 255) // 256:
        raise RuntimeError("Rotation diagnostic read an unexpected number of validation batches")
    expected_sample_ids = expected_ids[:sample_events]
    observed_ids = np.concatenate(context_id_chunks)
    if not np.array_equal(observed_ids, expected_ids):
        raise ValueError("Selected measured EVENT_IDs differ from the validation selection")
    if not np.array_equal(np.asarray(selected_ids, dtype=np.int64), expected_sample_ids):
        raise ValueError("Selected sample EVENT_IDs do not match the fixed validation subset")

    with torch.no_grad():
        base_representation = event_model.encode_visible(selected_batch)
        geometric_reference = base_representation.get("geometric_tokens")
        if geometric_reference is None or geometric_reference.shape[-2:] != (1, 16):
            raise ValueError("Visible L-GATr did not expose [B,N,1,16] geometric tokens")
        for angle in ANGLES:
            rotated_representation = event_model.encode_visible(
                _rotate_batch(selected_batch, angle)
            )
            geometric_rotated = rotated_representation.get("geometric_tokens")
            if geometric_rotated is None or geometric_rotated.shape != geometric_reference.shape:
                raise ValueError("H_geom changed shape under a transverse event rotation")
            action = torch.as_tensor(
                _multivector_rotation_matrix(angle),
                dtype=geometric_reference.dtype,
                device=geometric_reference.device,
            )
            expected_geometric = geometric_reference @ action.T
            geometric_rotation_differences[str(float(angle))].append(
                (geometric_rotated - expected_geometric).detach().cpu().numpy().reshape(-1)
            )

    source_by_id = {int(event_id): i for i, event_id in enumerate(bank_ids)}
    source_rows = np.asarray([source_by_id[int(event_id)] for event_id in expected_sample_ids])
    base_source = torch.as_tensor(bank_source[source_rows, :1], dtype=torch.float32)
    cfm = event_model.parent_cfm
    mean = cfm.target_mean.detach().cpu().numpy().reshape(2, 4)
    std = cfm.target_std.detach().cpu().numpy().reshape(2, 4)
    source_action = [_source_prior_action(angle, mean, std) for angle in ANGLES]

    with torch.no_grad():
        baseline = event_model.sample_partonic(
            selected_batch,
            samples=1,
            steps=steps,
            solver="heun",
            parent_source_state=base_source,
        )
        base_parents = baseline["parents"][:, 0]
        angle_results = {}
        for angle in ANGLES:
            rotated_batch = _rotate_batch(selected_batch, angle)
            fixed = event_model.sample_partonic(
                rotated_batch,
                samples=1,
                steps=steps,
                solver="heun",
                parent_source_state=base_source,
            )["parents"][:, 0]
            co_rotated_source = _rotate_source(base_source, angle, cfm.target_mean, cfm.target_std)
            co_rotated = event_model.sample_partonic(
                rotated_batch,
                samples=1,
                steps=steps,
                solver="heun",
                parent_source_state=co_rotated_source,
            )["parents"][:, 0]
            expected_parent = _rotate_parents(base_parents, angle)
            fixed_changes = _observable_differences(base_parents, fixed)
            covariance_changes = _observable_differences(expected_parent, co_rotated)
            expected_response = torch.full_like(
                fixed_changes["top_phi"], float(angle)
            )
            sensitivity = {
                "input_rotation_same_source": {
                    "expected_phi_response_rad": float(angle),
                    "top_phi_shift_rad": _summary(
                        fixed_changes["top_phi"].cpu().numpy()
                    ),
                    "antitop_phi_shift_rad": _summary(
                        fixed_changes["antitop_phi"].cpu().numpy()
                    ),
                    "top_phi_response_error_rad": _summary(
                        _angle_difference(
                            fixed_changes["top_phi"], expected_response
                        ).cpu().numpy()
                    ),
                    "antitop_phi_response_error_rad": _summary(
                        _angle_difference(
                            fixed_changes["antitop_phi"], expected_response
                        ).cpu().numpy()
                    ),
                    "top_pt_change_gev": _summary(
                        fixed_changes["top_pt"].cpu().numpy()
                    ),
                    "antitop_pt_change_gev": _summary(
                        fixed_changes["antitop_pt"].cpu().numpy()
                    ),
                    "top_mass_change_gev": _summary(
                        fixed_changes["top_mass"].cpu().numpy()
                    ),
                    "antitop_mass_change_gev": _summary(
                        fixed_changes["antitop_mass"].cpu().numpy()
                    ),
                    "mttbar_change_gev": _summary(
                        fixed_changes["mttbar"].cpu().numpy()
                    ),
                },
                "formal_co_rotated_source_covariance": {
                    "parent_four_vector_residual_gev": _summary(
                        (co_rotated - expected_parent).cpu().numpy()
                    ),
                    "top_pt_residual_gev": _summary(
                        covariance_changes["top_pt"].cpu().numpy()
                    ),
                    "antitop_pt_residual_gev": _summary(
                        covariance_changes["antitop_pt"].cpu().numpy()
                    ),
                    "top_mass_residual_gev": _summary(
                        covariance_changes["top_mass"].cpu().numpy()
                    ),
                    "antitop_mass_residual_gev": _summary(
                        covariance_changes["antitop_mass"].cpu().numpy()
                    ),
                    "top_phi_residual_rad": _summary(
                        covariance_changes["top_phi"].cpu().numpy()
                    ),
                    "antitop_phi_residual_rad": _summary(
                        covariance_changes["antitop_phi"].cpu().numpy()
                    ),
                    "mttbar_residual_gev": _summary(
                        covariance_changes["mttbar"].cpu().numpy()
                    ),
                    "source_action": _source_prior_action(angle, mean, std),
                },
            }
            angle_results[str(float(angle))] = sensitivity

    invariance = {}
    for name, chunks in context_differences.items():
        difference = np.concatenate(chunks)
        invariance[name] = {
            "max_abs": float(np.max(np.abs(difference))),
            "rms": float(np.sqrt(np.mean(difference * difference))),
        }
    geometric_rotation = {}
    for angle, chunks in geometric_rotation_differences.items():
        residual = np.concatenate(chunks)
        geometric_rotation[angle] = {
            **_summary(residual),
            "representation_shape": list(geometric_reference.shape),
            "expected_action": "grade-wise exterior-power representation of the transverse SO(2) rotation",
        }
    backbone = str(event_model.parent_cfm.velocity_backbone)
    result = {
        "status": "completed",
        "diagnostic": "realnu_parent_transverse_rotation_with_co_rotated_source",
        "assessment": (
            "The fixed-source test shows orientation sensitivity separately from the formal "
            "co-rotated-source test. The source prior is only approximately rotation invariant "
            "under the production scaler. The formal residual reports model covariance error."
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "diagnostic_model_status": record.get("diagnostic_model_status", "trained checkpoint"),
        "config": str(config_path),
        "config_sha256": str(record.get("config_sha256", _sha256(config_path))),
        "velocity_backbone": backbone,
        "validation_context_events": context_events,
        "sample_events": sample_events,
        "event_id_sha256": _sha256_ids(expected_ids),
        "sample_event_id_sha256": _sha256_ids(expected_sample_ids),
        "source_bank": str(SOURCE_BANK),
        "source_bank_sha256": _sha256(SOURCE_BANK),
        "source_seed": 20260929,
        "selection": "truth-mttbar-quantiles on validation only; samples join by exact EVENT_ID",
        "angles_rad": list(ANGLES),
        "solver": "heun",
        "steps": steps,
        "nfe": 2 * steps,
        "scalar_context_rotation": invariance,
        "geometric_context_rotation": geometric_rotation,
        "scaler_transverse": {
            "mean_px_py_gev": mean[:, :2].tolist(),
            "std_px_py_gev": std[:, :2].tolist(),
            "relative_std_difference": float(
                np.max(np.abs(std[:, 0] / std[:, 1] - 1.0))
            ),
            "scaler_identity_sha256": record["scaler_identity_sha256"],
        },
        "source_prior_rotation_action": source_action,
        "per_angle": angle_results,
        "sealed_test_used": False,
        "truth_free_conditioning": True,
        "parent_role_order": ["top", "antitop"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PRODUCTION_ROOT / "evaluation_v4_parent_comparison/audit/PCFM_Q_rotation_covariance.json",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--scaler-checkpoint", type=Path)
    parser.add_argument("--context-events", type=int, default=CONTEXT_EVENTS)
    parser.add_argument("--sample-events", type=int, default=SAMPLE_EVENTS)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--initialize-from-shared-checkpoint",
        action="store_true",
        help="Build an untrained model and load only visible/query components from the supplied initial checkpoint.",
    )
    args = parser.parse_args()
    result = run(
        output=args.output.expanduser().resolve(),
        config_path=args.config,
        checkpoint=args.checkpoint,
        record_path=args.record,
        scaler_checkpoint=args.scaler_checkpoint,
        context_events=args.context_events,
        sample_events=args.sample_events,
        steps=args.steps,
        initialize_from_shared_checkpoint=args.initialize_from_shared_checkpoint,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
