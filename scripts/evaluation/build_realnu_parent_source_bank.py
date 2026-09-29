#!/usr/bin/env python3
"""Freeze paired, nested standard-normal parent CFM source states."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
PRODUCTION_ROOT = PROJECT_ROOT / (
    "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/"
    "realnu_top_antitop_20260927_release2"
)
EVENT_TABLE = PRODUCTION_ROOT / "evaluation_r2/core/event_metrics.h5"
PRODUCTION_RECORD = PRODUCTION_ROOT / "PCFM_Q/production_training.json"
OUTPUT_ROOT = PRODUCTION_ROOT / "evaluation_v4_parent_comparison/source_bank"
OUTPUT_PATH = OUTPUT_ROOT / "parent_source_bank.npz"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.json"
MAX_SAMPLES = 32
SEED = 20260929
SOURCE_DIMENSION = 8
EXPECTED_EVENTS = 100_342


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _event_id_sha256(event_ids: np.ndarray) -> str:
    canonical = np.asarray(event_ids, dtype="<i8")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def build() -> dict:
    if not EVENT_TABLE.is_file():
        raise FileNotFoundError(f"Accepted validation event table is missing: {EVENT_TABLE}")
    if not PRODUCTION_RECORD.is_file():
        raise FileNotFoundError(f"Accepted PCFM_Q production record is missing: {PRODUCTION_RECORD}")
    if OUTPUT_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError(
            f"Refusing to replace an existing parent source bank: {OUTPUT_ROOT}"
        )
    record = json.loads(PRODUCTION_RECORD.read_text(encoding="utf-8"))
    scaler_sha = record.get("partonic_scaler_sha256")
    scaler_identity_sha = record.get("scaler_identity_sha256")
    if not scaler_sha or not scaler_identity_sha:
        raise ValueError("PCFM_Q production record lacks the frozen parent scaler identity")

    with h5py.File(EVENT_TABLE, "r") as source:
        schema = source.attrs.get("schema")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        if schema != "realnu_event_metrics_v1":
            raise ValueError(f"Unexpected accepted evaluation schema: {schema!r}")
        if bool(source.attrs.get("sealed_test_used", True)):
            raise ValueError("Accepted event table does not prove sealed_test_used=false")
        if "EVENT_ID" not in source:
            raise KeyError(f"{EVENT_TABLE}: missing EVENT_ID")
        event_ids = np.asarray(source["EVENT_ID"][:])
    if (
        event_ids.shape != (EXPECTED_EVENTS,)
        or not np.issubdtype(event_ids.dtype, np.integer)
        or len(np.unique(event_ids)) != EXPECTED_EVENTS
    ):
        raise ValueError(
            f"Expected {EXPECTED_EVENTS} unique integer validation EVENT_IDs; "
            f"found shape={event_ids.shape}, dtype={event_ids.dtype}"
        )
    event_ids = event_ids.astype(np.int64, copy=False)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    source_state = torch.randn(
        (EXPECTED_EVENTS, MAX_SAMPLES, SOURCE_DIMENSION),
        dtype=torch.float32,
        generator=generator,
    ).numpy()
    if source_state.shape != (EXPECTED_EVENTS, MAX_SAMPLES, SOURCE_DIMENSION):
        raise RuntimeError("Generated parent source bank has the wrong dimensions")
    if source_state.dtype != np.float32 or not np.isfinite(source_state).all():
        raise RuntimeError("Generated parent source bank is not finite float32")
    source_mean = float(source_state.mean(dtype=np.float64))
    source_std = float(source_state.std(dtype=np.float64))
    if abs(source_mean) > 0.01 or abs(source_std - 1.0) > 0.01:
        raise RuntimeError(
            f"Generated source bank fails standard-normal sanity checks: "
            f"mean={source_mean}, std={source_std}"
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    temporary = OUTPUT_ROOT / ".parent_source_bank.npz.tmp"
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, EVENT_ID=event_ids, source_state=source_state)
        with np.load(temporary, allow_pickle=False) as archive:
            if set(archive.files) != {"EVENT_ID", "source_state"}:
                raise RuntimeError("Source bank contains unexpected or truth-bearing arrays")
            check_ids = archive["EVENT_ID"]
            check_source = archive["source_state"]
            if not np.array_equal(check_ids, event_ids):
                raise RuntimeError("Source bank EVENT_ID order changed during serialization")
            if check_source.shape != source_state.shape or check_source.dtype != np.float32:
                raise RuntimeError("Serialized source-state shape/dtype validation failed")
            if not np.array_equal(check_source, source_state):
                raise RuntimeError("Serialized source states changed during round-trip")
        os.replace(temporary, OUTPUT_PATH)
        manifest = {
            "schema": "realnu_parent_source_bank_v1",
            "file": str(OUTPUT_PATH),
            "file_sha256": _sha256(OUTPUT_PATH),
            "validation_event_count": EXPECTED_EVENTS,
            "event_id_sha256": _event_id_sha256(event_ids),
            "parent_role_order": ["top", "antitop"],
            "source_dimension": SOURCE_DIMENSION,
            "source_dtype": "float32",
            "maximum_samples": MAX_SAMPLES,
            "sample_indices": "implicit contiguous 0..31; sample-count scans use exact prefixes",
            "rng_algorithm": "torch.Generator CPU default with torch.randn",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "seed": SEED,
            "source_distribution": (
                "Independent standard normal in the production-scaled joint parent target "
                "coordinates used by IndependentCoupling"
            ),
            "target_scaler_identity": {
                "partonic_scaler_sha256": scaler_sha,
                "scaler_identity_sha256": scaler_identity_sha,
            },
            "source_state_mean": source_mean,
            "source_state_std": source_std,
            "sealed_test_used": False,
            "truth_stored": False,
        }
        temporary_manifest = OUTPUT_ROOT / ".manifest.json.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_manifest, MANIFEST_PATH)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
