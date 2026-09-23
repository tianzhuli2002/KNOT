#!/usr/bin/env python3
"""Recheck the immutable common event, input, truth and parent contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _equal_sample(left, right, name: str) -> None:
    if left.dtype.names:
        if left.dtype != right.dtype or left.dtype.names != right.dtype.names:
            raise AssertionError(f"Sample contract dtype mismatch at {name}")
        for field in left.dtype.names:
            if not np.array_equal(left[field], right[field], equal_nan=True):
                raise AssertionError(f"Sample contract mismatch at {name}.{field}")
    elif not np.array_equal(left, right, equal_nan=True):
        raise AssertionError(f"Sample contract mismatch at {name}")


def verify(root: Path, *, sample_rows: tuple[int, ...]) -> dict:
    contract = root / "data_contract"
    manifest_path = contract / "matched_event_manifest.h5"
    canonical_path = root.parent.parent.parent / "data" / "real" / "ttbar2L_canonical.h5"
    # The experiment root lives below the repository; resolve the explicit
    # canonical path from manifest metadata rather than guessing if relocated.
    with h5py.File(manifest_path, "r") as manifest:
        event_ids = np.asarray(manifest["EVENT_ID"][:], dtype=np.uint64)
        canonical_rows = np.asarray(manifest["canonical_row"][:], dtype=np.uint64)
        split = np.asarray(manifest["split"][:]).astype("U")
        truth_support = np.asarray(manifest["truth_support"][:], dtype=np.uint8)
        canonical_path = Path(str(manifest.attrs["canonical_source"])).expanduser().resolve()
    if len(np.unique(event_ids)) != len(event_ids) or not np.all(event_ids[1:] > event_ids[:-1]):
        raise AssertionError("Manifest EVENT_ID is not unique and ordered")
    if not np.array_equal(canonical_rows, np.arange(len(event_ids), dtype=np.uint64)):
        raise AssertionError("Manifest canonical rows do not identify the canonical order")

    positions = {name: np.flatnonzero(split == name) for name in ("train", "validation", "test")}
    concatenated = np.concatenate(tuple(positions.values()))
    if len(np.unique(concatenated)) != len(event_ids) or not np.array_equal(np.sort(concatenated), canonical_rows.astype(np.int64)):
        raise AssertionError("Train/validation/test membership is not a disjoint partition")
    if len(positions["train"]) != 1_910_215 or len(positions["validation"]) != 50_000:
        raise AssertionError(f"Unexpected matched even counts: { {k: len(v) for k,v in positions.items()} }")

    view_records = {}
    with h5py.File(canonical_path, "r") as canonical:
        for name, rows in positions.items():
            for view_dir in (contract / "vyper_views", contract / "vyper_common_views_jet8"):
                view_path = view_dir / f"{name}.h5"
                with h5py.File(view_path, "r") as view:
                    observed_ids = np.asarray(view["EVENT_ID"][:], dtype=np.uint64)
                    expected_ids = event_ids[rows]
                    if not np.array_equal(observed_ids, expected_ids):
                        raise AssertionError(f"{view_path}: EVENT_ID order differs from manifest")
                    for row in sample_rows:
                        if row >= len(rows):
                            continue
                        canonical_row = int(rows[row])
                        for dataset in ("INPUTS/ELECTRON", "INPUTS/MUON", "INPUTS/MET", "INPUTS/GLOBAL", "LABELS/ELECTRON", "LABELS/MUON", "LABELS/NEUTRINO", "LABELS/GLOBAL"):
                            _equal_sample(view[dataset][row], canonical[dataset][canonical_row], f"{view_path}:{dataset}:{row}")
                        if view_dir.name == "vyper_common_views_jet8":
                            _equal_sample(view["INPUTS/JET"][row], canonical["INPUTS/JET"][canonical_row, :8], f"{view_path}:INPUTS/JET:{row}")
                            _equal_sample(view["LABELS/JET"][row], canonical["LABELS/JET"][canonical_row, :8], f"{view_path}:LABELS/JET:{row}")
                    view_records[f"{view_dir.name}/{name}"] = {
                        "events": int(len(observed_ids)),
                        "event_id_sha256": _digest(observed_ids),
                    }

        outside_jet8 = 0
        for start in range(0, len(event_ids), 100_000):
            stop = min(len(event_ids), start + 100_000)
            labels = np.asarray(canonical["LABELS/JET"][start:stop], dtype=np.float32)
            has_truth = ((labels == 1).sum(axis=1) == 1) & ((labels == 2).sum(axis=1) == 1)
            has_retained = ((labels[:, :8] == 1).sum(axis=1) == 1) & ((labels[:, :8] == 2).sum(axis=1) == 1)
            outside_jet8 += int(np.sum(has_truth & ~has_retained))

    parent_path = contract / "parent_truth" / "ttbar2L_parent_hybrid_truth.h5"
    with h5py.File(parent_path, "r") as parents:
        parent_ids = np.asarray(parents["EVENT_ID"][:], dtype=np.uint64)
        if not np.array_equal(parent_ids, event_ids):
            raise AssertionError("Parent truth sidecar EVENT_ID order differs from manifest")
        parent_valid = True
        for role in ("top_e", "top_mu"):
            values = np.asarray(parents[f"PARENTS/{role}"][:], dtype=np.float32)
            mass2 = values[:, 0] ** 2 - np.square(values[:, 1:]).sum(axis=1)
            parent_valid &= bool(np.isfinite(values).all() and (values[:, 0] >= 0).all() and (mass2 >= -1.0e-3).all())
    if not parent_valid:
        raise AssertionError("Parent truth sidecar contains invalid physical four-vectors")

    result = {
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "canonical": str(canonical_path),
        "counts": {name: int(len(rows)) for name, rows in positions.items()},
        "event_id_sha256": {name: _digest(event_ids[rows]) for name, rows in positions.items()},
        "truth_support_events": int(truth_support.sum()),
        "jet8_truth_outside_retained_support": int(outside_jet8),
        "view_records": view_records,
        "parent_truth": {"valid": True, "roles": ["top_e", "top_mu"], "semantics": "hybrid_visible_reconstructed_plus_truth_invisible_v1"},
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.experiment_root.expanduser().resolve(), sample_rows=(0, 1, 17, 1000, 49999))
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
