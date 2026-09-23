#!/usr/bin/env python3
"""Create no-copy VyPER views with the selected JER retained-jet contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


DATASETS = (
    "EVENT_ID",
    "INPUTS/ELECTRON", "INPUTS/GLOBAL", "INPUTS/JET", "INPUTS/MET", "INPUTS/MUON",
    "LABELS/ELECTRON", "LABELS/GLOBAL", "LABELS/JET", "LABELS/MET", "LABELS/MUON", "LABELS/NEUTRINO",
)


def runs(indices):
    if not len(indices):
        return
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(indices)]))
    for begin, end in zip(starts, stops):
        yield indices[begin:end], int(begin), int(end)


def build(args):
    canonical = Path(args.canonical).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing view directory: {output_dir}")
    output_dir.mkdir(parents=True)

    with h5py.File(manifest, "r") as source:
        split = np.asarray(source["split"][:]).astype("U")
        manifest_hash = json.loads(manifest.with_suffix(".json").read_text())["sha256"]
    with h5py.File(canonical, "r") as source:
        total = len(source["EVENT_ID"])
        for name in ("train", "validation", "test"):
            rows = np.flatnonzero(split == name).astype(np.int64)
            output = output_dir / f"{name}.h5"
            with h5py.File(output, "w", libver="latest") as view:
                for path in DATASETS:
                    if path not in source:
                        raise KeyError(f"Canonical source is missing {path}: {canonical}")
                    source_dataset = source[path]
                    if path in {"INPUTS/JET", "LABELS/JET"}:
                        if args.jet_count <= 0 or args.jet_count > source_dataset.shape[1]:
                            raise ValueError(f"jet-count {args.jet_count} is invalid for {path} shape {source_dataset.shape}")
                        tail = (args.jet_count,)
                    else:
                        tail = source_dataset.shape[1:]
                    layout = h5py.VirtualLayout(shape=(len(rows),) + tail, dtype=source_dataset.dtype)
                    virtual_source = h5py.VirtualSource(str(canonical), path, shape=source_dataset.shape)
                    for selected, begin, end in runs(rows):
                        source_slice = virtual_source[selected[0]:selected[-1] + 1]
                        if path in {"INPUTS/JET", "LABELS/JET"}:
                            source_slice = virtual_source[selected[0]:selected[-1] + 1, :args.jet_count]
                        layout[begin:end] = source_slice
                    view.create_virtual_dataset(path, layout)
                view.attrs.update({
                    "schema_version": "matched_vyper_jer_view_v2",
                    "manifest_sha256": manifest_hash,
                    "canonical_source": str(canonical),
                    "split": name,
                    "jet_count": int(args.jet_count),
                    "event_id_semantics": "partition*2^32+source_row",
                })
            print(f"wrote {output} rows={len(rows)} jet_count={args.jet_count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jet-count", type=int, default=8)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
