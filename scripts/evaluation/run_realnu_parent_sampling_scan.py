#!/usr/bin/env python3
"""Run paired, truth-free parent solver/sample-count scans on validation IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
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
SOURCE_MANIFEST = SOURCE_BANK.with_name("manifest.json")
PCFM_Q_RECORD = PRODUCTION_ROOT / "PCFM_Q/production_training.json"
EXPECTED_VALIDATION_EVENTS = 100_342
MAX_SOURCE_SAMPLES = 32
SOLVER_STEPS = (4, 8, 16, 32)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _json_sha256(value: dict) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_selection(event_count: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(EVENT_TABLE, "r") as table:
        if table.attrs.get("schema") != "realnu_event_metrics_v1":
            raise ValueError("Accepted validation event table has an unexpected schema")
        if bool(table.attrs.get("sealed_test_used", True)):
            raise ValueError("Parent scan requires sealed_test_used=false")
        ids = np.asarray(table["EVENT_ID"][:], dtype=np.int64)
        mass = np.asarray(table["truth/mttbar"][:], dtype=np.float64)
    if ids.shape != (EXPECTED_VALIDATION_EVENTS,) or len(np.unique(ids)) != len(ids):
        raise ValueError("Accepted event table must contain 100,342 unique validation EVENT_IDs")
    if event_count <= 0 or event_count > len(ids):
        raise ValueError(f"event_count must be in [1,{len(ids)}]")
    finite = np.isfinite(mass)
    ordered = np.flatnonzero(finite)[np.argsort(mass[finite], kind="stable")]
    if event_count == 1:
        selected = ordered[[len(ordered) // 2]]
    else:
        selected = ordered[np.linspace(0, len(ordered) - 1, event_count, dtype=np.int64)]
    return selected, ids[selected]


def _create_case(group: h5py.Group, event_count: int, sample_count: int) -> dict[str, h5py.Dataset]:
    return {
        "parents": group.create_dataset(
            "parents", shape=(event_count, sample_count, 2, 4), dtype=np.float32,
            chunks=(min(64, event_count), sample_count, 2, 4),
        ),
        "scaled_states": group.create_dataset(
            "scaled_states", shape=(event_count, sample_count, 8), dtype=np.float32,
            chunks=(min(64, event_count), sample_count, 8),
        ),
    }


def run(
    *,
    config_path: Path,
    checkpoint: Path,
    output: Path,
    event_count: int = 10_000,
    batch_size: int = 64,
    samples: int = MAX_SOURCE_SAMPLES,
    steps: tuple[int, ...] = SOLVER_STEPS,
    device: str = "cpu",
) -> dict:
    torch.set_num_threads(4)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested parent scan CUDA device is unavailable")
    if batch_size <= 0 or samples <= 0 or samples > MAX_SOURCE_SAMPLES:
        raise ValueError(f"Require batch_size>0 and samples in [1,{MAX_SOURCE_SAMPLES}]")
    if not steps or any(value not in SOLVER_STEPS for value in steps):
        raise ValueError(f"Parent solver scan steps must be selected from {SOLVER_STEPS}")
    config_path = config_path.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    output = output.expanduser().resolve()
    for path, label in (
        (EVENT_TABLE, "accepted validation event table"),
        (SOURCE_BANK, "paired parent source bank"),
        (SOURCE_MANIFEST, "parent source-bank manifest"),
        (config_path, "public experiment config"),
        (checkpoint, "trained model checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing parent scan: {output}")

    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source_bank_sha = _sha256(SOURCE_BANK)
    if source_manifest.get("file_sha256") != source_bank_sha:
        raise ValueError("Parent source bank SHA does not match its manifest")
    selection, expected_ids = _read_selection(event_count)
    with h5py.File(EVENT_TABLE, "r") as table:
        validation_ids = np.asarray(table["EVENT_ID"][:], dtype=np.int64)
    with np.load(SOURCE_BANK, allow_pickle=False) as bank:
        bank_ids = np.asarray(bank["EVENT_ID"], dtype=np.int64)
        source_bank = np.asarray(bank["source_state"], dtype=np.float32)
    if not np.array_equal(bank_ids, validation_ids):
        raise ValueError("Source-bank EVENT_ID order differs from the accepted validation table")
    if source_bank.shape != (EXPECTED_VALIDATION_EVENTS, MAX_SOURCE_SAMPLES, 8):
        raise ValueError(f"Malformed parent source bank shape: {source_bank.shape}")
    if source_manifest.get("target_scaler_identity", {}).get("partonic_scaler_sha256") is None:
        raise ValueError("Parent source-bank manifest omits the train-fitted scaler SHA")

    from joint_event_reconstruction.cli.run import (
        build_data,
        load_trained,
        load_visible_config,
    )
    from joint_event_reconstruction.configuration import load_experiment

    config = load_experiment(config_path)
    visible_config = load_visible_config(config)
    visible_config.general.mode = "predict"
    visible_config.dataset.val_batch_size = batch_size
    visible_config.dataset.val_num_workers = 0
    visible_config.dataset.pin_memory = device.type == "cuda"
    visible_config.dataset.persistent_workers = False
    data = build_data(config, visible_config, include_truth=False, device=str(device))
    data.setup("predict")
    data.val_dataset = Subset(data.val_dataset, selection.tolist())
    loader = DataLoader(
        data.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=data.structured_collator,
    )
    module = load_trained(config, visible_config, data, str(checkpoint)).to(device).eval()
    if data.structured_collator.partonic_store is not None:
        raise RuntimeError("Truth-free scan opened the partonic truth sidecar")
    cfm = module.event_model.parent_cfm
    if cfm is None:
        raise ValueError("Selected checkpoint has no parent CFM")
    scaler_fits = {
        "invisible": module.hparams.get("invisible_scaler_fit"),
        "partonic": module.hparams.get("partonic_scaler_fit"),
    }
    scaler_values = {
        "invisible": module.scaler_metadata,
        "partonic": module.partonic_scaler_metadata,
    }
    partonic_scaler_sha = _json_sha256(
        {"scaler": scaler_values["partonic"], "fit": scaler_fits["partonic"]}
    )
    scaler_identity_sha = _json_sha256(
        {"scalers": scaler_values, "fit": scaler_fits}
    )
    if (
        partonic_scaler_sha
        != source_manifest["target_scaler_identity"]["partonic_scaler_sha256"]
        or scaler_identity_sha
        != source_manifest["target_scaler_identity"]["scaler_identity_sha256"]
    ):
        raise ValueError("Selected checkpoint uses a different production parent scaler")
    if source_bank.shape[1] < samples:
        raise ValueError("Source bank does not contain the requested source prefix")

    id_to_row = {int(event_id): index for index, event_id in enumerate(bank_ids)}
    measured_path = Path(data.h5_path).expanduser().resolve()
    split_identity = json.dumps(data.split_metadata, sort_keys=True, default=str)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    cursor = 0
    try:
        with h5py.File(temporary, "w") as target:
            target.attrs["prediction_schema"] = "realnu_parent_solver_scan_v1"
            target.attrs["truth_free_input"] = True
            target.attrs["sealed_test_used"] = False
            target.attrs["split"] = "validation"
            target.attrs["event_count"] = int(event_count)
            target.attrs["event_id_sha256"] = _sha256_ids(expected_ids)
            target.attrs["parent_role_order"] = json.dumps(["top", "antitop"])
            target.attrs["source_bank_path"] = str(SOURCE_BANK)
            target.attrs["source_bank_sha256"] = source_bank_sha
            target.attrs["source_bank_seed"] = int(source_manifest["seed"])
            target.attrs["source_sample_prefix"] = int(samples)
            target.attrs["partonic_scaler_sha256"] = str(
                source_manifest["target_scaler_identity"]["partonic_scaler_sha256"]
            )
            target.attrs["scaler_identity_sha256"] = str(
                source_manifest["target_scaler_identity"]["scaler_identity_sha256"]
            )
            target.attrs["config_path"] = str(config_path)
            target.attrs["config_sha256"] = _sha256(config_path)
            target.attrs["checkpoint_path"] = str(checkpoint)
            target.attrs["checkpoint_sha256"] = _sha256(checkpoint)
            target.attrs["measured_input_path"] = str(measured_path)
            target.attrs["measured_input_sha256"] = _sha256(measured_path)
            target.attrs["split_identity"] = split_identity
            target.attrs["selection_definition"] = (
                "uniform truth-mttbar quantiles on the accepted validation EVENT_ID table; "
                "truth is used only to choose the deterministic scan subset"
            )
            target.attrs["velocity_backbone"] = str(cfm.velocity_backbone)
            target.attrs["conditioning"] = str(cfm.conditioning)
            target.attrs["solver"] = "heun"
            target.create_dataset("EVENT_ID", data=expected_ids, dtype=np.int64)
            cases = target.create_group("CASES")
            writers = {}
            for step_count in steps:
                group = cases.create_group(f"heun_steps_{step_count}")
                group.attrs["solver"] = "heun"
                group.attrs["steps"] = int(step_count)
                group.attrs["nfe_per_sample"] = 2 * int(step_count)
                group.attrs["samples"] = int(samples)
                writers[step_count] = _create_case(group, event_count, samples)

            for batch in loader:
                moved = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                ids = moved["event_index"].detach().cpu().numpy().astype(np.int64, copy=False)
                expected_slice = expected_ids[cursor : cursor + len(ids)]
                if not np.array_equal(ids, expected_slice):
                    raise ValueError("Validation batch EVENT_ID order differs from the fixed scan")
                source_rows = np.asarray([id_to_row[int(event_id)] for event_id in ids])
                source = torch.as_tensor(
                    source_bank[source_rows, :samples], dtype=torch.float32, device=device
                )
                for step_count in steps:
                    result = module.event_model.sample_partonic(
                        moved,
                        samples=samples,
                        steps=step_count,
                        solver="heun",
                        parent_source_state=source,
                    )
                    parents = result["parents"].detach().float().cpu().numpy()
                    scaled = result["scaled"].reshape(len(ids), samples, 8).detach().float().cpu().numpy()
                    actual_nfe = int(result["nfe"])
                    expected_nfe = 2 * int(step_count)
                    if actual_nfe != expected_nfe:
                        raise ValueError(
                            f"Heun{step_count} reported {actual_nfe} NFE, expected {expected_nfe}"
                        )
                    if parents.shape != (len(ids), samples, 2, 4):
                        raise ValueError(f"Malformed parent output shape: {parents.shape}")
                    if not np.isfinite(parents).all() or not np.isfinite(scaled).all():
                        raise ValueError(f"Non-finite parent prediction at Heun{step_count}")
                    writers[step_count]["parents"][cursor : cursor + len(ids)] = parents
                    writers[step_count]["scaled_states"][cursor : cursor + len(ids)] = scaled
                cursor += len(ids)
            if cursor != event_count:
                raise RuntimeError(f"Wrote {cursor} scan events, expected {event_count}")
            target.flush()

        with h5py.File(temporary, "r") as check:
            if bool(check.attrs.get("truth_free_input", False)) is not True:
                raise RuntimeError("Parent scan lost its truth-free provenance")
            if "TRUTH" in check or "TRUTH_SIDE_CAR" in check:
                raise RuntimeError("Parent scan output unexpectedly contains truth")
            if not np.array_equal(check["EVENT_ID"][:], expected_ids):
                raise RuntimeError("Parent scan output EVENT_ID order changed")
            for step_count in steps:
                if check[f"CASES/heun_steps_{step_count}/parents"].shape != (
                    event_count, samples, 2, 4
                ):
                    raise RuntimeError("Parent scan output shape validation failed")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "status": "passed",
        "output": str(output),
        "events": event_count,
        "event_id_sha256": _sha256_ids(expected_ids),
        "samples": samples,
        "heun_steps": list(steps),
        "nfe_per_sample": {str(value): 2 * value for value in steps},
        "velocity_backbone": str(cfm.velocity_backbone),
        "conditioning": str(cfm.conditioning),
        "source_bank_sha256": source_bank_sha,
        "checkpoint_sha256": _sha256(checkpoint),
        "truth_free_input": True,
        "sealed_test_used": False,
    }


def main() -> None:
    default_record = json.loads(PCFM_Q_RECORD.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(default_record["config_path"]))
    parser.add_argument("--checkpoint", type=Path, default=Path(default_record["checkpoint"]))
    parser.add_argument(
        "--output",
        type=Path,
        default=PRODUCTION_ROOT / "evaluation_v4_parent_comparison/scan/PCFM_Q_solver_scan.h5",
    )
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=MAX_SOURCE_SAMPLES)
    parser.add_argument("--steps", type=int, nargs="+", default=list(SOLVER_STEPS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    result = run(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output=args.output,
        event_count=args.events,
        batch_size=args.batch_size,
        samples=args.samples,
        steps=tuple(args.steps),
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
