#!/usr/bin/env python3
"""Chunked validation and concise physics summary of canonical joint data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


SCHEMA_VERSION = "joint_event_reconstruction_canonical_v1"


def splitmix64(values: np.ndarray) -> np.ndarray:
    """Reproduce the converter's deterministic even-partition split hash."""
    with np.errstate(over="ignore"):
        values = values.astype(np.uint64, copy=True) + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def finite_labels(values: np.ndarray) -> Counter:
    selected = values[np.isfinite(values)].astype(np.int64, copy=False)
    unique, counts = np.unique(selected, return_counts=True)
    return Counter(dict(zip(unique.tolist(), counts.tolist())))


def validate(path: Path, *, chunk_size: int = 100_000) -> dict:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Canonical data file does not exist: {path}")
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: unsupported schema {source.attrs.get('schema_version')!r}")
        topology = str(source.attrs["topology"])
        expected_roles = 1 if topology == "ttbar1L" else 2 if topology == "ttbar2L" else None
        if expected_roles is None:
            raise ValueError(f"{path}: unsupported canonical topology {topology!r}")
        count = len(source["EVENT_ID"])
        required = ["EVENT_ID", "SOURCE/partition", "SOURCE/row", "SPLIT", "LABELS/NEUTRINO"]
        required += [f"INPUTS/{name}" for name in source["INPUTS"]]
        for name in required:
            if name not in source:
                raise KeyError(f"{path}: missing {name}")
        neutrino = source["LABELS/NEUTRINO"]
        if neutrino.shape != (count, expected_roles) or neutrino.dtype.names != ("px", "py", "pz"):
            raise ValueError(f"{path}: malformed invisible truth {neutrino.shape}, {neutrino.dtype}")

        split_counts = np.zeros(3, dtype=np.int64)
        partition_counts = np.zeros(2, dtype=np.int64)
        labels: dict[str, Counter] = {name: Counter() for name in source["LABELS"] if name != "NEUTRINO"}
        finite_neutrino = np.zeros(expected_roles, dtype=np.int64)
        previous = None
        emu_presence = 0
        for start in range(0, count, chunk_size):
            stop = min(count, start + chunk_size)
            event_id = source["EVENT_ID"][start:stop]
            partition = source["SOURCE/partition"][start:stop]
            row = source["SOURCE/row"][start:stop]
            split = source["SPLIT"][start:stop]
            if not np.isin(partition, (0, 1)).all():
                raise ValueError(f"{path}: invalid source partition at row {start}")
            if not np.isin(split, (0, 1, 2)).all():
                raise ValueError(f"{path}: invalid split value at row {start}")
            if not np.array_equal(event_id, partition.astype(np.uint64) * np.uint64(1 << 32) + row):
                raise ValueError(f"{path}: EVENT_ID provenance closure failed at row {start}")
            if previous is not None and event_id[0] <= previous:
                raise ValueError(f"{path}: EVENT_ID is not unique/ordered at row {start}")
            if len(event_id) > 1 and not np.all(event_id[1:] > event_id[:-1]):
                raise ValueError(f"{path}: EVENT_ID duplicates within chunk starting {start}")
            previous = event_id[-1]
            expected_split = np.full(len(event_id), 2, dtype=np.uint8)
            even = partition == 0
            expected_split[even] = np.where(
                splitmix64(event_id[even] ^ np.uint64(42)) % np.uint64(20) == 0,
                1,
                0,
            ).astype(np.uint8)
            if not np.array_equal(split, expected_split):
                raise ValueError(f"{path}: split policy mismatch at row {start}")
            split_counts += np.bincount(split, minlength=3)
            partition_counts += np.bincount(partition, minlength=2)
            truth = neutrino[start:stop]
            for role in range(expected_roles):
                finite = np.ones(stop - start, dtype=bool)
                for component in ("px", "py", "pz"):
                    finite &= np.isfinite(truth[component][:, role])
                finite_neutrino[role] += finite.sum()
            for name in labels:
                labels[name].update(finite_labels(source[f"LABELS/{name}"][start:stop]))
            if topology == "ttbar2L":
                electrons = source["LABELS/ELECTRON"][start:stop]
                muons = source["LABELS/MUON"][start:stop]
                present = (electrons == 1).sum(axis=1) == 1
                present &= (muons == 1).sum(axis=1) == 1
                emu_presence += present.sum()

        if not np.all(finite_neutrino == count):
            raise ValueError(f"{path}: non-finite neutrino roles: {finite_neutrino.tolist()} / {count}")
        if not split_counts.all() or split_counts.sum() != count:
            raise ValueError(f"{path}: invalid split counts {split_counts.tolist()}")
        if topology == "ttbar2L" and emu_presence != count:
            raise ValueError(f"{path}: {count - emu_presence} events lack canonical e-mu support")

        result = {
            "path": str(path), "schema_version": SCHEMA_VERSION, "topology": topology,
            "events": count, "partition_counts": partition_counts.tolist(),
            "split_counts": {name: int(split_counts[index]) for index, name in enumerate(("train", "validation", "test"))},
            "invisible_roles": json.loads(source.attrs["invisible_roles"]),
            "finite_neutrinos": finite_neutrino.tolist(),
            "label_frequencies": {name: dict(sorted(counter.items())) for name, counter in labels.items()},
            "split_policy": str(source.attrs["split_policy"]),
            "source_files": json.loads(source.attrs["source_files"]),
        }
        if topology == "ttbar2L":
            result["canonical_emu_events"] = int(emu_presence)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk size must be positive")
    print(json.dumps(validate(Path(args.path), chunk_size=args.chunk_size), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
