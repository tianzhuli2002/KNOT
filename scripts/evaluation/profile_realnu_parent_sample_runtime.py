#!/usr/bin/env python3
"""Measure truth-free PCFM_Q CPU inference time as the posterior sample count grows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_realnu_parent_sampling_scan import (  # noqa: E402
    EVENT_TABLE,
    EXPECTED_VALIDATION_EVENTS,
    MAX_SOURCE_SAMPLES,
    PRODUCTION_ROOT,
    SOURCE_BANK,
    SOURCE_MANIFEST,
    _json_sha256,
    _read_selection,
    _sha256,
    _sha256_ids,
)

PCFM_Q_RECORD = PRODUCTION_ROOT / "PCFM_Q/production_training.json"
SAMPLE_COUNTS = (1, 2, 4, 8, 16, 32)


def profile(
    *,
    config_path: Path,
    checkpoint: Path,
    output: Path,
    event_count: int = 512,
    batch_size: int = 64,
    steps: tuple[int, ...] = (4, 8, 16, 32),
    repeats: int = 2,
    device: str = "cpu",
) -> dict:
    if device != "cpu":
        raise ValueError("This profile is intended for the CPU multicore queue")
    if event_count <= 0 or batch_size <= 0 or repeats <= 0:
        raise ValueError("Require positive events, batch size, and repeats")
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    config_path, checkpoint, output = (
        path.expanduser().resolve() for path in (config_path, checkpoint, output)
    )
    for path, label in (
        (EVENT_TABLE, "accepted validation event table"),
        (SOURCE_BANK, "paired parent source bank"),
        (SOURCE_MANIFEST, "source bank manifest"),
        (config_path, "PCFM_Q public config"),
        (checkpoint, "PCFM_Q checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")

    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    bank_sha = _sha256(SOURCE_BANK)
    if manifest.get("file_sha256") != bank_sha:
        raise ValueError("Parent source bank SHA differs from its manifest")
    selected_rows, expected_ids = _read_selection(event_count)
    with h5py.File(EVENT_TABLE, "r") as table:
        validation_ids = np.asarray(table["EVENT_ID"][:], dtype=np.int64)
    if validation_ids.shape != (EXPECTED_VALIDATION_EVENTS,):
        raise ValueError("Accepted validation table has an unexpected EVENT_ID count")
    with np.load(SOURCE_BANK, allow_pickle=False) as bank:
        bank_ids = np.asarray(bank["EVENT_ID"], dtype=np.int64)
        source_bank = np.asarray(bank["source_state"], dtype=np.float32)
    if not np.array_equal(bank_ids, validation_ids):
        raise ValueError("Source-bank EVENT_ID order differs from validation")
    if source_bank.shape != (EXPECTED_VALIDATION_EVENTS, MAX_SOURCE_SAMPLES, 8):
        raise ValueError(f"Malformed parent source-bank shape: {source_bank.shape}")

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
    visible_config.dataset.pin_memory = False
    visible_config.dataset.persistent_workers = False
    data = build_data(config, visible_config, include_truth=False, device="cpu")
    data.setup("predict")
    data.val_dataset = Subset(data.val_dataset, selected_rows.tolist())
    loader = DataLoader(
        data.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=data.structured_collator,
    )
    module = load_trained(config, visible_config, data, str(checkpoint)).cpu().eval()
    if data.structured_collator.partonic_store is not None:
        raise RuntimeError("Truth-free runtime profile opened the partonic truth sidecar")
    cfm = module.event_model.parent_cfm
    if cfm is None or cfm.velocity_backbone != "conventional":
        raise ValueError("Runtime profile requires the trained conventional PCFM_Q")

    scaler_values = {
        "invisible": module.scaler_metadata,
        "partonic": module.partonic_scaler_metadata,
    }
    scaler_fits = {
        "invisible": module.hparams.get("invisible_scaler_fit"),
        "partonic": module.hparams.get("partonic_scaler_fit"),
    }
    expected_scalers = manifest["target_scaler_identity"]
    if _json_sha256({"scaler": scaler_values["partonic"], "fit": scaler_fits["partonic"]}) != expected_scalers[
        "partonic_scaler_sha256"
    ]:
        raise ValueError("PCFM_Q and parent source bank use different parent scalers")
    if _json_sha256({"scalers": scaler_values, "fit": scaler_fits}) != expected_scalers[
        "scaler_identity_sha256"
    ]:
        raise ValueError("PCFM_Q scaler identity differs from the paired source bank")

    id_to_row = {int(event_id): index for index, event_id in enumerate(bank_ids)}
    if not steps or any(value not in (4, 8, 16, 32) for value in steps):
        raise ValueError("Profile Heun steps must be selected from (4, 8, 16, 32)")
    timing: dict[tuple[int, int], list[float]] = {
        (step_count, sample_count): []
        for step_count in steps
        for sample_count in SAMPLE_COUNTS
    }
    for repeat in range(repeats):
        for step_count in steps:
            for sample_count in SAMPLE_COUNTS:
                elapsed = 0.0
                cursor = 0
                for batch_index, batch in enumerate(loader):
                    moved = {
                        key: value.cpu() if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                    ids = moved["event_index"].detach().cpu().numpy().astype(np.int64, copy=False)
                    expected = expected_ids[cursor : cursor + len(ids)]
                    if not np.array_equal(ids, expected):
                        raise ValueError("Runtime-profile EVENT_ID order differs from its fixed subset")
                    source_rows = np.asarray([id_to_row[int(event_id)] for event_id in ids])
                    source = torch.as_tensor(
                        source_bank[source_rows, :sample_count], dtype=torch.float32
                    )
                    # One untimed batch warms the model kernels and CPU allocator per (steps, M).
                    if batch_index == 0:
                        module.event_model.sample_partonic(
                            moved,
                            samples=sample_count,
                            steps=step_count,
                            solver="heun",
                            parent_source_state=source,
                        )
                    started = time.perf_counter()
                    result = module.event_model.sample_partonic(
                        moved,
                        samples=sample_count,
                        steps=step_count,
                        solver="heun",
                        parent_source_state=source,
                    )
                    elapsed += time.perf_counter() - started
                    parents = result["parents"]
                    actual_nfe = int(result["nfe"])
                    if actual_nfe != 2 * step_count:
                        raise ValueError(
                            f"Heun{step_count} reported {actual_nfe} NFE, "
                            f"expected {2 * step_count}"
                        )
                    if parents.shape != (len(ids), sample_count, 2, 4):
                        raise ValueError(f"Malformed parent sample shape: {tuple(parents.shape)}")
                    if not torch.isfinite(parents).all():
                        raise ValueError(f"Non-finite parent output at M={sample_count}")
                    cursor += len(ids)
                if cursor != event_count:
                    raise RuntimeError(f"Profiled {cursor} events, expected {event_count}")
                timing[(step_count, sample_count)].append(elapsed)

    cases = {}
    for (step_count, sample_count), values in timing.items():
        median = float(np.median(values))
        cases[f"heun_steps_{step_count}_samples_{sample_count}"] = {
            "steps": step_count,
            "samples": sample_count,
            "nfe_per_sample": 2 * step_count,
            "aggregate_nfe_per_event": 2 * step_count * sample_count,
            "repeat_seconds": values,
            "median_seconds": median,
            "median_seconds_per_event": median / event_count,
            "events_per_second": event_count / median,
        }

    result = {
        "status": "passed",
        "schema": "realnu_parent_runtime_profile_v1",
        "model": "PCFM_Q",
        "velocity_backbone": "conventional",
        "conditioning": str(cfm.conditioning),
        "solver": "heun",
        "steps": list(steps),
        "events": event_count,
        "event_id_sha256": _sha256_ids(expected_ids),
        "source_bank_sha256": bank_sha,
        "checkpoint_sha256": _sha256(checkpoint),
        "config_sha256": _sha256(config_path),
        "batch_size": batch_size,
        "repeats": repeats,
        "warmup": "one untimed first batch for each sample count per repeat",
        "timed_scope": "parent CFM sampling calls; excludes data loading, model loading, and output serialization",
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
        },
        "truth_free_input": True,
        "sealed_test_used": False,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to replace runtime profile: {output}")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def main() -> None:
    record = json.loads(PCFM_Q_RECORD.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(record["config_path"]))
    parser.add_argument("--checkpoint", type=Path, default=Path(record["checkpoint"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    print(json.dumps(profile(
        config_path=args.config,
        checkpoint=args.checkpoint,
        output=args.output,
        event_count=args.events,
        batch_size=args.batch_size,
        steps=tuple(args.steps),
        repeats=args.repeats,
        device=args.device,
    ), indent=2))


if __name__ == "__main__":
    main()
