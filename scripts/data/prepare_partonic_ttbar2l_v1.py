"""Bounded source audit and stable event split for the 2M after-FSR sample.

This command reads the event identity columns in chunks and reads physics
branches only for a deterministic bounded sample from train/development rows.
It never loads detector arrays for the sealed test partition.
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


DEFAULT_SOURCE = Path(
    "/net/scratch/w00238tl/datasets/ttbar2L/combined_partonic_afterFSR_v1/raw/"
    "TtbarDilepton_2M_even.h5"
)
DEFAULT_OUTPUT = Path("results/experiments/partonic_ttbar2L_2M_v1")
EXPECTED_DATASETS = (
    "/INPUTS/JET", "/INPUTS/ELECTRON", "/INPUTS/MUON", "/INPUTS/GLOBAL",
    "/LABELS/JET", "/LABELS/ELECTRON", "/LABELS/MUON", "/LABELS/NEUTRINO",
    "/TARGETS/TOP", "/TARGETS/TRUTH", "/META/eventNumber",
    "/META/source_entry_index",
)
SPLIT_VERSION = "event_number_splitmix64_90_5_5_v1"


def _json_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attrs(obj) -> dict:
    return {str(key): _json_value(value) for key, value in obj.attrs.items()}


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(values, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    """Vectorised SplitMix64; stable across Python and NumPy versions."""
    with np.errstate(over="ignore"):
        x = values.astype(np.uint64, copy=False) ^ np.uint64(seed)
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _read_identity(source: h5py.File, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(source["/META/eventNumber"].shape[0])
    event_ids = np.empty(count, dtype=np.int64)
    source_entries = np.empty(count, dtype=np.int64)
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        event_ids[start:stop] = source["/META/eventNumber"][start:stop]
        source_entries[start:stop] = source["/META/source_entry_index"][start:stop]
    return event_ids, source_entries


def _physical_top(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    masses = []
    for prefix in ("top", "antitop"):
        pt = targets[f"{prefix}_pt"].astype(np.float64)
        eta = targets[f"{prefix}_eta"].astype(np.float64)
        phi = targets[f"{prefix}_phi"].astype(np.float64)
        mass = targets[f"{prefix}_m"].astype(np.float64)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2)
        vectors.append(np.stack((energy, px, py, pz), axis=-1))
        masses.append(mass)
    return np.stack(vectors, axis=1), np.stack(masses, axis=1)


def inspect(source_path: Path, output_dir: Path, *, chunk_size: int, sample_size: int, seed: int) -> dict:
    source_path = source_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Source HDF5 does not exist: {source_path}")
    if chunk_size <= 0 or sample_size <= 0:
        raise ValueError("chunk_size and sample_size must be positive")

    with h5py.File(source_path, "r") as source:
        missing = [name for name in EXPECTED_DATASETS if name not in source]
        if missing:
            raise KeyError(f"Source HDF5 is missing required datasets: {missing}")
        count = int(source["/META/eventNumber"].shape[0])
        mismatched = {
            name: tuple(source[name].shape)
            for name in EXPECTED_DATASETS
            if not source[name].shape or source[name].shape[0] != count
        }
        if mismatched:
            raise ValueError(f"Event-axis length mismatch: expected={count}, found={mismatched}")
        event_ids, source_entries = _read_identity(source, chunk_size)
        if not np.all(event_ids % 2 == 0):
            bad = event_ids[event_ids % 2 != 0][:10].tolist()
            raise ValueError(f"Source eventNumber includes odd values: {bad}")
        unique_ids = np.unique(event_ids)
        unique_entries = np.unique(source_entries)
        if len(unique_ids) != count:
            raise ValueError(f"META/eventNumber is not unique: {len(unique_ids)}/{count}")
        if len(unique_entries) != count:
            raise ValueError(f"META/source_entry_index is not unique: {len(unique_entries)}/{count}")

        hash_values = _splitmix64(event_ids, seed)
        buckets = (hash_values % np.uint64(100)).astype(np.uint8)
        split = np.where(buckets < 90, 0, np.where(buckets < 95, 1, 2)).astype(np.uint8)
        rows = np.arange(count, dtype=np.int64)
        train_rows = rows[split == 0]
        validation_rows = rows[split == 1]
        test_rows = rows[split == 2]
        if min(len(train_rows), len(validation_rows), len(test_rows)) == 0:
            raise ValueError(f"Stable split has an empty partition: {np.bincount(split, minlength=3)}")

        rng = np.random.default_rng(seed ^ 0x4A455231)
        n_train_subset = min(256, len(train_rows))
        n_val_subset = min(128, len(validation_rows))
        train_subset_rows = np.sort(rng.choice(train_rows, n_train_subset, replace=False))
        val_subset_rows = np.sort(rng.choice(validation_rows, n_val_subset, replace=False))
        selected_rows = np.sort(np.concatenate((train_subset_rows, val_subset_rows)))
        # A bounded, deterministic scan from the permitted development population.
        scan_pool = np.concatenate((train_rows, validation_rows))
        n_scan = min(max(sample_size, 2048), len(scan_pool))
        scan_rows = np.sort(rng.choice(scan_pool, n_scan, replace=False))

        globals_sample = source["/INPUTS/GLOBAL"][scan_rows]
        electrons = source["/INPUTS/ELECTRON"][scan_rows]
        muons = source["/INPUTS/MUON"][scan_rows]
        jets = source["/INPUTS/JET"][selected_rows]
        jet_labels = source["/LABELS/JET"][selected_rows]
        electron_labels = source["/LABELS/ELECTRON"][selected_rows]
        muon_labels = source["/LABELS/MUON"][selected_rows]
        electron_selected = source["/INPUTS/ELECTRON"][selected_rows]
        muon_selected = source["/INPUTS/MUON"][selected_rows]
        top = source["/TARGETS/TOP"][selected_rows]
        truth = source["/TARGETS/TRUTH"][selected_rows]

        jet_present = np.isfinite(jets["pt"]) & np.isfinite(jets["eta"]) & np.isfinite(jets["phi"]) & np.isfinite(jets["e"])
        electron_present = np.isfinite(electron_selected["pt"]) & np.isfinite(electron_selected["eta"]) & np.isfinite(electron_selected["phi"]) & np.isfinite(electron_selected["e"])
        muon_present = np.isfinite(muon_selected["pt"]) & np.isfinite(muon_selected["eta"]) & np.isfinite(muon_selected["phi"]) & np.isfinite(muon_selected["e"])
        padding_consistent = (
            np.array_equal(~jet_present, np.isnan(jet_labels))
            and np.array_equal(~electron_present, np.isnan(electron_labels))
            and np.array_equal(~muon_present, np.isnan(muon_labels))
        )
        if not padding_consistent:
            raise ValueError("Sampled input padding and NaN label sentinels do not align")
        physical_top, top_masses = _physical_top(top)
        truth_vectors = np.stack(
            [
                np.stack((truth["truth_top_E"], truth["truth_top_px"], truth["truth_top_py"], truth["truth_top_pz"]), axis=-1),
                np.stack((truth["truth_antitop_E"], truth["truth_antitop_px"], truth["truth_antitop_py"], truth["truth_antitop_pz"]), axis=-1),
            ],
            axis=1,
        ).astype(np.float64)
        four_vector_difference = np.abs(physical_top - truth_vectors)
        max_four_vector_difference = float(four_vector_difference.max(initial=0.0))
        top_mass_squared = physical_top[..., 0] ** 2 - (physical_top[..., 1:] ** 2).sum(axis=-1)
        target_valid = (
            np.isfinite(physical_top).all(axis=(1, 2))
            & np.isfinite(top_masses).all(axis=1)
            & (top_masses >= 0).all(axis=1)
            & (top_mass_squared >= -1.0e-2).all(axis=1)
        )
        truth_valid = np.isfinite(truth_vectors).all(axis=(1, 2))
        if max_four_vector_difference > 5.0e-3:
            raise ValueError(
                "TARGETS/TOP and redundant TARGETS/TRUTH four-vectors disagree; "
                f"sample maximum absolute difference={max_four_vector_difference:.8g} GeV"
            )

        charge_counts = {}
        for name, values, labels in (
            ("electron", electron_selected, electron_labels),
            ("muon", muon_selected, muon_labels),
        ):
            present = np.isfinite(values["pt"]) & np.isfinite(values["charge"])
            charge = values["charge"]
            role_label = labels
            charge_counts[name] = {
                "present_slots": int(present.sum()),
                "positive": int((present & (charge > 0)).sum()),
                "negative": int((present & (charge < 0)).sum()),
                "zero_or_invalid_charge": int((present & (charge == 0)).sum()),
                "label_1": int((role_label == 1).sum()),
                "label_2": int((role_label == 2).sum()),
            }
        sampled_njet = np.asarray(globals_sample["njet"])
        sampled_ne = np.asarray(globals_sample["nelectron"])
        sampled_nmu = np.asarray(globals_sample["nmuon"])
        combinations = {}
        for ne, nmu in zip(sampled_ne, sampled_nmu, strict=True):
            key = f"e{int(ne)}_mu{int(nmu)}" if np.isfinite(ne) and np.isfinite(nmu) else "invalid"
            combinations[key] = combinations.get(key, 0) + 1

        expected_attr = {
            "n_saved_events": int(source.attrs.get("n_saved_events", -1)),
            "split_name": str(_json_value(source.attrs.get("split_name", ""))),
            "split_key": str(_json_value(source.attrs.get("split_key", ""))),
            "target_source": str(_json_value(source.attrs.get("target_source", ""))),
            "fully_matched_applied": bool(source.attrs.get("fully_matched_applied", False)),
            "truth_energy_scale_applied": float(source.attrs.get("truth_energy_scale_applied", np.nan)),
            "reco_energy_scale_applied": float(source.attrs.get("reco_energy_scale_applied", np.nan)),
            "neutrino_energy_scale_applied": float(source.attrs.get("neutrino_energy_scale_applied", np.nan)),
        }
        if expected_attr["n_saved_events"] != count:
            raise ValueError(f"n_saved_events attr disagrees with datasets: {expected_attr['n_saved_events']} != {count}")
        if expected_attr["target_source"] != "partonic_after_fsr":
            raise ValueError(f"Unexpected target_source attribute: {expected_attr['target_source']!r}")
        neutrino_metadata = _attrs(source["/LABELS/NEUTRINO"])
        if neutrino_metadata.get("order") != "electron-flavour neutrino, muon-flavour neutrino":
            raise ValueError(f"Unexpected LABELS/NEUTRINO slot order: {neutrino_metadata}")

        split_path = output_dir / "split_membership_v1.npz"
        _atomic_npz(
            split_path,
            event_id=event_ids,
            source_entry_index=source_entries,
            split=split,
            source_row=rows,
        )
        dev_path = output_dir / "cpu_development_ids_v1.npz"
        _atomic_npz(
            dev_path,
            train_event_id=event_ids[train_subset_rows],
            train_source_row=train_subset_rows,
            validation_event_id=event_ids[val_subset_rows],
            validation_source_row=val_subset_rows,
        )

        source_stat = source_path.stat()
        manifest = {
            "manifest_version": "partonic_ttbar2l_source_v1",
            "source": {
                "path": str(source_path),
                "size_bytes": int(source_stat.st_size),
                "mtime_ns": int(source_stat.st_mtime_ns),
                "event_count": count,
                "root_attributes": _attrs(source),
                "dataset_shapes": {name: list(source[name].shape) for name in EXPECTED_DATASETS},
                "dataset_dtypes": {name: str(source[name].dtype) for name in EXPECTED_DATASETS},
                "dataset_attributes": {
                    name: _attrs(source[name]) for name in EXPECTED_DATASETS if source[name].attrs
                },
                "identity": {
                    "event_number_unique": True,
                    "source_entry_index_unique": True,
                    "all_event_numbers_even": True,
                    "event_number_min": int(event_ids.min()),
                    "event_number_max": int(event_ids.max()),
                    "source_entry_min": int(source_entries.min()),
                    "source_entry_max": int(source_entries.max()),
                },
            },
            "split": {
                "version": SPLIT_VERSION,
                "identity_field": "/META/eventNumber",
                "secondary_identity_field": "/META/source_entry_index",
                "seed": int(seed),
                "allocation": {"train": "hash_bucket < 90", "validation": "90 <= hash_bucket < 95", "test": "hash_bucket >= 95"},
                "counts": {"train": int(len(train_rows)), "validation": int(len(validation_rows)), "reserved_test": int(len(test_rows))},
                "membership_path": str(split_path.relative_to(output_dir)),
                "membership_sha256": _sha256(split_path),
                "cpu_development_path": str(dev_path.relative_to(output_dir)),
                "cpu_development_sha256": _sha256(dev_path),
                "cpu_development_counts": {"train": int(len(train_subset_rows)), "validation": int(len(val_subset_rows))},
            },
            "bounded_sample": {
                "seed": int(seed),
                "rows_read": int(len(scan_rows) + len(selected_rows)),
                "training_and_validation_only": True,
                "jet_multiplicity": {
                    "minimum": float(np.nanmin(sampled_njet)),
                    "maximum": float(np.nanmax(sampled_njet)),
                    "observed_values": sorted(int(value) for value in np.unique(sampled_njet[np.isfinite(sampled_njet)])),
                },
                "lepton_flavour_combinations": combinations,
                "charge_counts": charge_counts,
                "padding_alignment_verified": bool(padding_consistent),
                "partonic_supported_in_sample": int(target_valid.sum()),
                "partonic_invalid_in_sample": int((~target_valid | ~truth_valid).sum()),
                "max_TOP_vs_TRUTH_abs_difference_GeV": max_four_vector_difference,
                "partonic_target_units": "GeV",
                "partonic_target_order": ["top", "antitop"],
                "partonic_truth_definition": expected_attr["target_source"],
                "neutrino_verified_order": neutrino_metadata.get("order"),
                "neutrino_verified_units": neutrino_metadata.get("units"),
                "neutrino_side_mapping_verified": False,
            },
            "physics_contract": {
                "lepton_label_semantics": "LABELS/ELECTRON and LABELS/MUON attrs declare 1=positive lepton and 2=negative lepton",
                "partonic_roles": "TARGETS/TOP top and antitop are separate after-FSR generator targets",
                "neutrino_slots": "flavour ordered (electron, muon), px/py/pz GeV",
                "neutrino_charge_or_top_side_association": "unverified; real-data neutrino supervision disabled",
                "same_flavour_pairing": "not verified by source metadata",
                "inference_inputs": "INPUTS and META only; never LABELS or TARGETS",
            },
        }
    _atomic_json(output_dir / "source_manifest_v1.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--sample-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    values = inspect(
        args.input,
        args.output_dir,
        chunk_size=args.chunk_size,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    print(json.dumps(values, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
