#!/usr/bin/env python3
"""Materialise a small real-event JER smoke fixture from an existing canonical file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def copy_tree(source, output, limit):
    for group_name in ("INPUTS", "LABELS"):
        group = output.create_group(group_name)
        for name, dataset in source[group_name].items():
            values = dataset[:limit]
            group.create_dataset(name, data=values)
    for name in ("EVENT_ID",):
        if name in source:
            output.create_dataset(name, data=source[name][:limit])
    for group_name in ("SOURCE",):
        if group_name in source:
            group = output.create_group(group_name)
            for name, dataset in source[group_name].items():
                group.create_dataset(name, data=dataset[:limit])
    if "LABELS/GLOBAL" not in output:
        output["LABELS"].create_dataset("GLOBAL", data=np.zeros(limit, dtype=np.float32))
    split = np.full(limit, 2, dtype=np.uint8)
    split[: max(1, int(0.8 * limit))] = 0
    split[max(1, int(0.8 * limit)): max(2, int(0.9 * limit))] = 1
    output.create_dataset("SPLIT", data=split)
    for key, value in source.attrs.items():
        output.attrs[key] = value
    output.attrs["schema_version"] = "joint_event_reconstruction_tiny_real_fixture_v1"
    output.attrs["source_fixture"] = str(Path(source.filename).resolve())
    output.attrs["classification_labels"] = "all-zero smoke labels; not a physics target"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", type=int, default=128)
    args = parser.parse_args()
    if args.events <= 0:
        raise ValueError("events must be positive")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite tiny fixture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, "r") as source:
        total = len(source["EVENT_ID"])
        if args.events > total:
            raise ValueError(f"Requested {args.events} events from source with {total}")
        with h5py.File(output, "w") as target:
            copy_tree(source, target, args.events)
    print(f"wrote {output} with {args.events} real canonical events")


if __name__ == "__main__":
    main()
