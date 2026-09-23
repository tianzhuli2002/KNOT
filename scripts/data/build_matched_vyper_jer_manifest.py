#!/usr/bin/env python3
"""Build the immutable event contract for the matched VyPER/JER study.

The manifest is deliberately derived from the already validated canonical JER
file.  The raw even/odd files are rescanned with the converter's selection
predicate and compared against the canonical SOURCE mapping before anything is
written.  VyPER inputs are HDF5 virtual views over the canonical rows; no raw
or canonical event payload is copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from convert_vyper_joint import EVENT_OFFSET, expected_paths, selected_emu, validate_sources, splitmix64
from joint_event_reconstruction.m0.data.splits import save_split, source_identity


ROLE_NAMES = ("top_e.nu", "top_mu.nu")
VIEW_DATASETS = (
    "EVENT_ID",
    "INPUTS/ELECTRON", "INPUTS/GLOBAL", "INPUTS/JET", "INPUTS/MET", "INPUTS/MUON",
    "LABELS/ELECTRON", "LABELS/GLOBAL", "LABELS/JET", "LABELS/MET", "LABELS/MUON", "LABELS/NEUTRINO",
)


def _atomic_path(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    return Path(temporary)


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _selected_source_rows(path: Path, chunk_size: int) -> np.ndarray:
    rows = []
    with h5py.File(path, "r") as source:
        total = len(source["LABELS/NEUTRINO"])
        for start in range(0, total, chunk_size):
            stop = min(total, start + chunk_size)
            selected, _, _ = selected_emu(source, start, stop)
            rows.append(np.flatnonzero(selected).astype(np.uint64) + np.uint64(start))
    return np.concatenate(rows) if rows else np.empty(0, dtype=np.uint64)


def _contiguous_runs(indices: np.ndarray):
    if len(indices) == 0:
        return
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(indices)]))
    for begin, end in zip(starts, stops):
        yield indices[begin:end], int(begin), int(end)


def _make_virtual_view(canonical: Path, output: Path, rows: np.ndarray, metadata: dict) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing VyPER view: {output}")
    temporary = _atomic_path(output)
    try:
        with h5py.File(canonical, "r") as source, h5py.File(temporary, "w", libver="latest") as view:
            for name in VIEW_DATASETS:
                if name not in source:
                    raise KeyError(f"Canonical source is missing {name}: {canonical}")
                source_dataset = source[name]
                tail = source_dataset.shape[1:]
                layout = h5py.VirtualLayout(
                    shape=(len(rows),) + tail,
                    dtype=source_dataset.dtype,
                )
                virtual_source = h5py.VirtualSource(
                    str(canonical), name, shape=source_dataset.shape
                )
                for selected_rows, begin, end in _contiguous_runs(rows):
                    layout[begin:end] = virtual_source[selected_rows[0]:selected_rows[-1] + 1]
                parent, _, leaf = name.rpartition("/")
                if parent:
                    view.require_group(parent)
                view.create_virtual_dataset(name, layout)
            view.attrs.update(metadata)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_manifest(path: Path, canonical: Path, source_paths: tuple[Path, Path], arrays: dict, metadata: dict) -> str:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    temporary = _atomic_path(path)
    try:
        with h5py.File(temporary, "w") as output:
            output.attrs.update(metadata)
            for name, values in arrays.items():
                output.create_dataset(name, data=values, compression="gzip", shuffle=True)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest metadata: {sidecar}")
    sidecar.write_text(json.dumps({
        "manifest": str(path),
        "sha256": digest,
        "canonical_source": str(canonical),
        "source_files": [str(item) for item in source_paths],
        "metadata": metadata,
    }, indent=2, sort_keys=True) + "\n")
    return digest


def build(args: argparse.Namespace) -> None:
    canonical = Path(args.canonical).expanduser().resolve()
    even = Path(args.even).expanduser().resolve()
    odd = Path(args.odd).expanduser().resolve()
    root = Path(args.output_root).expanduser().resolve()
    manifest_path = root / "data_contract" / "matched_event_manifest.h5"
    split_path = root / "data_contract" / "jer_split"
    view_dir = root / "data_contract" / "vyper_views"
    for item in (manifest_path, manifest_path.with_suffix(".json"), split_path / "split_indices.npz", view_dir):
        if item.exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment artifact: {item}")

    validate_sources("ttbar2L", (even, odd))
    with h5py.File(canonical, "r") as source:
        if source.attrs.get("schema_version") != "joint_event_reconstruction_canonical_v1":
            raise ValueError(f"Unexpected canonical schema in {canonical}")
        n_events = len(source["EVENT_ID"])
        canonical_partition = np.asarray(source["SOURCE/partition"][:], dtype=np.uint8)
        canonical_row = np.asarray(source["SOURCE/row"][:], dtype=np.uint64)
        event_id = np.asarray(source["EVENT_ID"][:], dtype=np.uint64)
        split_existing = np.asarray(source["SPLIT"][:], dtype=np.uint8)
        if not np.array_equal(event_id, canonical_partition.astype(np.uint64) * EVENT_OFFSET + canonical_row):
            raise ValueError("Canonical EVENT_ID/SOURCE mapping does not close")
        if len(np.unique(event_id)) != len(event_id) or not np.all(event_id[1:] > event_id[:-1]):
            raise ValueError("Canonical EVENT_ID is not unique and strictly ordered")

        raw_even_rows = _selected_source_rows(even, args.chunk_size)
        raw_odd_rows = _selected_source_rows(odd, args.chunk_size)
        for partition, raw_rows in enumerate((raw_even_rows, raw_odd_rows)):
            observed = canonical_row[canonical_partition == partition]
            if not np.array_equal(observed, raw_rows):
                raise ValueError(
                    f"Canonical selection differs from raw partition {partition}: "
                    f"canonical={len(observed)}, raw={len(raw_rows)}"
                )

        even_positions = np.flatnonzero(canonical_partition == 0).astype(np.int64)
        odd_positions = np.flatnonzero(canonical_partition == 1).astype(np.int64)
        if args.validation_events <= 0 or args.validation_events >= len(even_positions):
            raise ValueError(
                f"validation-events must lie in [1,{len(even_positions)-1}], found {args.validation_events}"
            )
        ranking = splitmix64(event_id[even_positions] ^ np.uint64(args.seed))
        order = np.lexsort((event_id[even_positions], ranking))
        validation = np.sort(even_positions[order[:args.validation_events]])
        validation_set = np.zeros(n_events, dtype=bool)
        validation_set[validation] = True
        train = even_positions[~validation_set[even_positions]]
        test = odd_positions
        selected = np.arange(n_events, dtype=np.int64)
        arrays = {
            "selected": selected,
            "train": train,
            "validation": validation,
            "test": test,
        }
        for name, values in arrays.items():
            if len(np.unique(values)) != len(values):
                raise ValueError(f"{name} contains repeated canonical rows")
        if np.array_equal(split_existing, np.full(n_events, 9, dtype=np.uint8)):
            raise ValueError("Canonical split is unexpectedly unassigned")
        if not np.array_equal(np.sort(np.concatenate((train, validation, test))), selected):
            raise ValueError("Matched train/validation/test rows do not close")

        jet = source["INPUTS/JET"][:]
        jet_labels = source["LABELS/JET"][:]
        electron_labels = source["LABELS/ELECTRON"][:]
        muon_labels = source["LABELS/MUON"][:]
        neutrino = source["LABELS/NEUTRINO"][:]
        njet = np.isfinite(jet["pt"]).sum(axis=1).astype(np.int16)
        support = (
            (np.isfinite(electron_labels).sum(axis=1) == 1)
            & (np.isfinite(muon_labels).sum(axis=1) == 1)
            & ((jet_labels == 1).sum(axis=1) == 1)
            & ((jet_labels == 2).sum(axis=1) == 1)
            & np.isfinite(neutrino["px"]).all(axis=1)
            & np.isfinite(neutrino["py"]).all(axis=1)
            & np.isfinite(neutrino["pz"]).all(axis=1)
        )
        split = np.full(n_events, "unassigned", dtype="S10")
        split[train] = b"train"
        split[validation] = b"validation"
        split[test] = b"test"
        manifest_arrays = {
            "canonical_row": np.arange(n_events, dtype=np.uint64),
            "source_partition": canonical_partition,
            "source_row": canonical_row,
            "EVENT_ID": event_id,
            "split": split,
            "njet": njet,
            "truth_support": support.astype(np.uint8),
        }
        metadata = {
            "schema_version": "matched_vyper_jer_manifest_v1",
            "canonical_schema": str(source.attrs["schema_version"]),
            "canonical_source": str(canonical),
            "source_files": json.dumps([str(even), str(odd)]),
            "source_partition_semantics": "0=even,1=odd",
            "event_id_semantics": "partition*2^32+source_row",
            "selection": str(source.attrs.get("ttbar2L_selection", "")),
            "invisible_roles": json.dumps(ROLE_NAMES),
            "momentum_units": "GeV",
            "validation_policy": "lowest splitmix64(EVENT_ID XOR seed), tie-break EVENT_ID, even partition only",
            "validation_events": int(len(validation)),
            "seed": int(args.seed),
            "counts": json.dumps({"selected": len(selected), "train": len(train), "validation": len(validation), "test": len(test)}),
            "source_counts": json.dumps({"even": len(raw_even_rows), "odd": len(raw_odd_rows)}),
        }

    digest = _write_manifest(manifest_path, canonical, (even, odd), manifest_arrays, metadata)
    split_metadata = source_identity(canonical, n_events)
    split_metadata = {
        "source": split_metadata,
        "seed": int(args.seed),
        "limit_events": None,
        "test_size": 0.2,
        "policy": "matched_even_train_all_but_50000_validation_odd_test",
        "validation_events": int(len(validation)),
    }
    save_split(split_path, arrays, split_metadata)

    view_metadata = {
        "manifest_sha256": digest,
        "canonical_source": str(canonical),
        "schema_version": "matched_vyper_jer_view_v1",
        "event_id_semantics": "partition*2^32+source_row",
    }
    view_dir.mkdir(parents=True, exist_ok=False)
    for name, rows in (("train", train), ("validation", validation), ("test", test)):
        _make_virtual_view(canonical, view_dir / f"{name}.h5", rows.astype(np.int64), {**view_metadata, "split": name})

    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": digest,
        "split_indices": str(split_path / "split_indices.npz"),
        "vyper_views": str(view_dir),
        "counts": {"selected": len(selected), "train": len(train), "validation": len(validation), "test": len(test)},
        "raw_selected": {"even": len(raw_even_rows), "odd": len(raw_odd_rows)},
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--even", required=True)
    parser.add_argument("--odd", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--validation-events", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    build(args)


if __name__ == "__main__":
    main()
