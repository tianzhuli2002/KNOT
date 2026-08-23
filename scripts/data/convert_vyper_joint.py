#!/usr/bin/env python3
"""Stream authoritative VyPER event records into the canonical joint HDF5 schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np


RAW = Path("/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/VyPER_ONNX/VyPER_run3/raw")
SOURCES = {
    "ttbar1L": (RAW / "ttbar_SL_FM_8M_even.h5", RAW / "ttbar_SL_FM_8M_odd.h5"),
    "ttbar2L": (RAW / "ttbar_dilep_30_4_run3_even.h5", RAW / "ttbar_dilep_30_4_run3_odd.h5"),
}
ROLE_NAMES = {"ttbar1L": ("tlep.nu",), "ttbar2L": ("top_e.nu", "top_mu.nu")}
EVENT_OFFSET = np.uint64(1 << 32)
SCHEMA_VERSION = "joint_event_reconstruction_canonical_v1"
SPLIT_NAMES = {0: "train", 1: "validation", 2: "test"}


def splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        values = values.astype(np.uint64, copy=True) + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def expected_paths(topology: str) -> tuple[str, ...]:
    if topology == "ttbar1L":
        return (
            "INPUTS/GLOBAL", "INPUTS/JET", "INPUTS/LEPTON",
            "LABELS/JET", "LABELS/LEPTON", "LABELS/NEUTRINO",
        )
    return (
        "INPUTS/ELECTRON", "INPUTS/GLOBAL", "INPUTS/JET", "INPUTS/MET", "INPUTS/MUON",
        "LABELS/ELECTRON", "LABELS/JET", "LABELS/MET", "LABELS/MUON", "LABELS/NEUTRINO",
    )


def validate_sources(topology: str, paths: tuple[Path, Path]) -> None:
    signatures = []
    for partition, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(f"VyPER source does not exist: {path}")
        with h5py.File(path, "r") as source:
            if bool(source.attrs.get("save_even")) != (partition == 0) or not bool(source.attrs.get("split_mode")):
                raise ValueError(f"Unexpected even/odd provenance attributes in {path}")
            signature = []
            for name in expected_paths(topology):
                if name not in source:
                    raise KeyError(f"{path} is missing {name}")
                dataset = source[name]
                signature.append((name, dataset.shape[1:], dataset.dtype.descr if dataset.dtype.names else dataset.dtype.str))
            signatures.append(signature)
            expected_roles = 1 if topology == "ttbar1L" else 2
            neutrino = source["LABELS/NEUTRINO"]
            if neutrino.shape[1] != expected_roles or neutrino.dtype.names != ("px", "py", "pz"):
                raise ValueError(f"Unexpected neutrino schema in {path}: {neutrino.shape}, {neutrino.dtype}")
    if signatures[0] != signatures[1]:
        raise ValueError("Even and odd source schemas differ")


def selected_emu(source: h5py.File, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    electron = source["LABELS/ELECTRON"][start:stop]
    muon = source["LABELS/MUON"][start:stop]
    electron_count = np.isfinite(electron).sum(axis=1)
    muon_count = np.isfinite(muon).sum(axis=1)
    electron_role = np.max(np.where(np.isfinite(electron), electron, -1), axis=1).astype(np.int8)
    muon_role = np.max(np.where(np.isfinite(muon), muon, -1), axis=1).astype(np.int8)
    selected = (electron_count == 1) & (muon_count == 1)
    selected &= ((electron_role == 1) & (muon_role == 2)) | ((electron_role == 2) & (muon_role == 1))
    jets = source["LABELS/JET"][start:stop]
    selected &= (jets == 1).sum(axis=1) == 1
    selected &= (jets == 2).sum(axis=1) == 1
    return selected, electron_role, muon_role


def canonical_met(global_rows: np.ndarray, jet_dtype: np.dtype) -> np.ndarray:
    result = np.zeros((len(global_rows), 1), dtype=jet_dtype)
    names = set(jet_dtype.names or ())
    pt = global_rows["met_met"][:, 0]
    sin_phi = global_rows["met_sin_phi"][:, 0]
    cos_phi = global_rows["met_cos_phi"][:, 0]
    values = {
        "id": np.full(len(result), 4, dtype=np.float32), "e": pt, "eta": np.zeros(len(result)),
        "phi": np.arctan2(sin_phi, cos_phi), "cos_phi": cos_phi, "sin_phi": sin_phi,
        "pt": pt, "btag": np.zeros(len(result)), "charge": np.zeros(len(result)),
    }
    for name, value in values.items():
        if name in names:
            result[name][:, 0] = value
    return result


def create_output(output: h5py.File, topology: str, sources: tuple[Path, Path], chunk_size: int) -> dict[str, h5py.Dataset]:
    datasets = {}
    output.create_group("SOURCE")
    output.create_group("INPUTS")
    output.create_group("LABELS")
    datasets["EVENT_ID"] = output.create_dataset("EVENT_ID", (0,), maxshape=(None,), dtype="<u8", chunks=(chunk_size,))
    datasets["SOURCE/partition"] = output.create_dataset("SOURCE/partition", (0,), maxshape=(None,), dtype="u1", chunks=(chunk_size,))
    datasets["SOURCE/row"] = output.create_dataset("SOURCE/row", (0,), maxshape=(None,), dtype="<u8", chunks=(chunk_size,))
    datasets["SPLIT"] = output.create_dataset("SPLIT", (0,), maxshape=(None,), dtype="u1", chunks=(chunk_size,))
    datasets["LABELS/GLOBAL"] = output.create_dataset(
        "LABELS/GLOBAL", (0,), maxshape=(None,), dtype="<f4", chunks=(chunk_size,),
    )
    with h5py.File(sources[0], "r") as reference:
        for name in expected_paths(topology):
            source_dataset = reference[name]
            tail = source_dataset.shape[1:]
            if topology == "ttbar2L" and name in {
                "INPUTS/ELECTRON", "INPUTS/MUON", "LABELS/ELECTRON", "LABELS/MUON",
            }:
                tail = (1,)
            shape = (0,) + tail
            maximum = (None,) + tail
            chunks = (min(chunk_size, 50_000),) + tail
            datasets[name] = output.create_dataset(name, shape, maxshape=maximum, dtype=source_dataset.dtype, chunks=chunks)
        if topology == "ttbar1L":
            shape = (0, 1)
            datasets["INPUTS/MET"] = output.create_dataset(
                "INPUTS/MET", shape, maxshape=(None, 1), dtype=reference["INPUTS/JET"].dtype,
                chunks=(min(chunk_size, 50_000), 1),
            )
    return datasets


def append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    old = len(dataset)
    dataset.resize(old + len(values), axis=0)
    dataset[old:] = values


def convert(topology: str, output_path: Path, chunk_size: int, max_source_events: int | None) -> None:
    sources = SOURCES[topology]
    validate_sources(topology, sources)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    counts = []
    with h5py.File(temporary, "w") as output:
        datasets = create_output(output, topology, sources, chunk_size)
        for partition, path in enumerate(sources):
            saved = 0
            with h5py.File(path, "r") as source:
                total = len(source["LABELS/NEUTRINO"])
                if max_source_events is not None:
                    total = min(total, max_source_events)
                for start in range(0, total, chunk_size):
                    stop = min(total, start + chunk_size)
                    selection = np.ones(stop - start, dtype=bool)
                    electron_role = muon_role = None
                    if topology == "ttbar2L":
                        selection, electron_role, muon_role = selected_emu(source, start, stop)
                    rows = np.arange(start, stop, dtype=np.uint64)[selection]
                    event_id = np.uint64(partition) * EVENT_OFFSET + rows
                    split = np.full(len(rows), 2 if partition else 0, dtype=np.uint8)
                    if partition == 0:
                        split[splitmix64(event_id ^ np.uint64(42)) % np.uint64(20) == 0] = 1
                    append(datasets["EVENT_ID"], event_id)
                    append(datasets["SOURCE/partition"], np.full(len(rows), partition, dtype=np.uint8))
                    append(datasets["SOURCE/row"], rows)
                    append(datasets["SPLIT"], split)
                    append(datasets["LABELS/GLOBAL"], np.zeros(len(rows), dtype=np.float32))
                    for name in expected_paths(topology):
                        values = source[name][start:stop][selection]
                        if topology == "ttbar2L" and name == "LABELS/JET":
                            source_labels = values.copy()
                            values = np.where(np.isnan(source_labels), np.nan, 0).astype(np.float32)
                            erole = electron_role[selection]
                            mrole = muon_role[selection]
                            values[source_labels == erole[:, None]] = 1
                            values[source_labels == mrole[:, None]] = 2
                        elif topology == "ttbar2L" and name in {"LABELS/ELECTRON", "LABELS/MUON"}:
                            finite = np.isfinite(values)
                            if not (finite.sum(axis=1) == 1).all():
                                raise RuntimeError(f"Accepted e-mu event has malformed {name}")
                            values = np.ones((len(values), 1), dtype=np.float32)
                        elif topology == "ttbar2L" and name in {"INPUTS/ELECTRON", "INPUTS/MUON"}:
                            label_name = name.replace("INPUTS/", "LABELS/")
                            labels = source[label_name][start:stop][selection]
                            finite = np.isfinite(labels)
                            slot = finite.argmax(axis=1)
                            values = values[np.arange(len(values)), slot][:, None]
                        append(datasets[name], values)
                    if topology == "ttbar1L":
                        append(datasets["INPUTS/MET"], canonical_met(
                            source["INPUTS/GLOBAL"][start:stop], source["INPUTS/JET"].dtype,
                        ))
                    saved += len(rows)
                    if stop == total or stop % 1_000_000 == 0:
                        print(f"partition={partition} source_rows={stop}/{total} canonical_rows={saved}", flush=True)
            counts.append(saved)
        commit = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
        output.attrs.update({
            "schema_version": SCHEMA_VERSION, "topology": topology, "synthetic": False,
            "source_files": json.dumps([str(path) for path in sources]),
            "source_event_counts": json.dumps(counts), "source_partition_semantics": "0=even,1=odd",
            "event_id_semantics": "partition*2^32+source_row", "momentum_units": "GeV",
            "invisible_roles": json.dumps(ROLE_NAMES[topology]),
            "source_neutrino_semantics": (
                "single physical semileptonic neutrino" if topology == "ttbar1L"
                else "slot0=electron neutrino,slot1=muon neutrino"
            ),
            "split_policy": "even: deterministic 95% train/5% validation by EVENT_ID; odd: test",
            "split_names": json.dumps(SPLIT_NAMES), "conversion_commit": commit,
            "ttbar2L_selection": "one labelled electron, one labelled muon, opposite source roles, one b and one bbar" if topology == "ttbar2L" else "all rows",
            "ttbar2L_canonical_labels": "JET1=electron-branch b,JET2=muon-branch b; ELECTRON1/MUON1=selected support" if topology == "ttbar2L" else "unchanged",
        })
    validate_output(temporary, topology)
    temporary.replace(output_path)
    write_split_product(output_path, topology)
    print(f"Wrote canonical {topology} data to {output_path}", flush=True)


def write_split_product(output_path: Path, topology: str) -> None:
    output_path = output_path.expanduser().resolve()
    with h5py.File(output_path, "r") as canonical:
        if canonical.attrs.get("topology") != topology:
            raise ValueError(f"{output_path} does not contain topology {topology}")
        split = canonical["SPLIT"][:]
    split_dir = output_path.parent / "splits" / topology
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / "split_indices.npz"
    arrays = {
        "selected": np.arange(len(split), dtype=np.int64),
        "train": np.flatnonzero(split == 0),
        "validation": np.flatnonzero(split == 1),
        "test": np.flatnonzero(split == 2),
    }
    np.savez(
        split_path, **arrays,
    )
    stat = output_path.stat()
    identity = {
        "path": str(output_path), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "total_events": len(split),
    }
    identity["fingerprint"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    metadata = {
        "source": identity, "seed": 42, "limit_events": None, "test_size": 0.2,
        "canonical_policy": "even: deterministic 95% train/5% validation; odd: test",
        "index_hashes": {
            name: hashlib.sha256(np.ascontiguousarray(values, dtype=np.int64).tobytes()).hexdigest()
            for name, values in arrays.items()
        },
    }
    (split_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote persistent split indices to {split_path}", flush=True)


def validate_output(path: Path, topology: str) -> None:
    with h5py.File(path, "r") as source:
        count = len(source["EVENT_ID"])
        if count == 0 or source.attrs.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Canonical output {path} is empty or has wrong schema")
        for name in ("SOURCE/partition", "SOURCE/row", "SPLIT", "LABELS/GLOBAL", *expected_paths(topology)):
            if name not in source or len(source[name]) != count:
                raise ValueError(f"Canonical output {path} has inconsistent {name}")
        roles = 1 if topology == "ttbar1L" else 2
        if source["LABELS/NEUTRINO"].shape != (count, roles):
            raise ValueError("Canonical invisible role shape is wrong")
        previous = None
        split_counts = np.zeros(3, dtype=np.int64)
        for start in range(0, count, 100_000):
            stop = min(count, start + 100_000)
            event_id = source["EVENT_ID"][start:stop]
            partition = source["SOURCE/partition"][start:stop]
            row = source["SOURCE/row"][start:stop]
            if not np.array_equal(event_id, partition.astype(np.uint64) * EVENT_OFFSET + row):
                raise ValueError("EVENT_ID provenance closure failed")
            if previous is not None and event_id[0] <= previous:
                raise ValueError("EVENT_ID is not unique and monotonically ordered")
            if len(event_id) > 1 and not np.all(event_id[1:] > event_id[:-1]):
                raise ValueError("EVENT_ID contains duplicates")
            previous = event_id[-1]
            split_counts += np.bincount(source["SPLIT"][start:stop], minlength=3)
            neutrino = source["LABELS/NEUTRINO"][start:stop]
            if not all(np.isfinite(neutrino[field]).all() for field in ("px", "py", "pz")):
                raise ValueError("Canonical neutrino truth contains non-finite values")
        if not split_counts.all():
            raise ValueError(f"Canonical split contains an empty partition: {split_counts.tolist()}")
        print(f"Validated {count} events; split_counts={split_counts.tolist()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", choices=("ttbar1L", "ttbar2L"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--max-source-events", type=int)
    parser.add_argument("--splits-only", action="store_true")
    args = parser.parse_args()
    if args.chunk_size <= 0 or (args.max_source_events is not None and args.max_source_events <= 0):
        raise ValueError("Chunk sizes and event limits must be positive")
    if args.splits_only:
        write_split_product(Path(args.output), args.topology)
    else:
        convert(args.topology, Path(args.output), args.chunk_size, args.max_source_events)


if __name__ == "__main__":
    main()
