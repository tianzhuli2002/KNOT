#!/usr/bin/env python3
"""Materialise small, split-specific eight-jet VyPER views.

The files are deliberately separate for train, validation and development
test.  VyPER's GraphDB cache is keyed by the input path and its preprocessing
semantics differ between training and prediction, so one path must not be
reused for all three phases of the smoke test.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _copy_group(source, target, group_name: str, rows: np.ndarray, jet_count: int) -> None:
    group = target.create_group(group_name)
    for name, dataset in source[group_name].items():
        values = dataset[rows]
        if group_name in {"INPUTS", "LABELS"} and name == "JET":
            values = values[:, :jet_count]
        group.create_dataset(name, data=values)


def build(source_path: Path, output_dir: Path, jet_count: int) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite tiny VyPER views: {output_dir}")
    output_dir.mkdir(parents=True)
    with h5py.File(source_path, "r") as source:
        split = np.asarray(source["SPLIT"][:], dtype=np.uint8)
        event_id = np.asarray(source["EVENT_ID"][:], dtype=np.uint64)
        if np.any(source["SOURCE/partition"][:] != 0):
            raise ValueError("Tiny VyPER views must be even-source development events")
        for split_code, name in ((0, "train"), (1, "validation"), (2, "test")):
            rows = np.flatnonzero(split == split_code).astype(np.int64)
            output_path = output_dir / f"{name}.h5"
            with h5py.File(output_path, "w") as target:
                target.create_dataset("EVENT_ID", data=event_id[rows])
                _copy_group(source, target, "INPUTS", rows, jet_count)
                _copy_group(source, target, "LABELS", rows, jet_count)
                target.attrs.update(
                    {
                        "schema_version": "matched_vyper_jer_tiny_view_v1",
                        "source_fixture": str(source_path.resolve()),
                        "split": name,
                        "jet_count": jet_count,
                        "source_partition": "even-only development fixture",
                        "event_id_order": "copied from fixture without shuffling",
                    }
                )
            print(f"wrote {output_path} rows={len(rows)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jet-count", type=int, default=8)
    args = parser.parse_args()
    if args.jet_count <= 0:
        raise ValueError("jet-count must be positive")
    build(args.source.expanduser().resolve(), args.output_dir.expanduser().resolve(), args.jet_count)


if __name__ == "__main__":
    main()
