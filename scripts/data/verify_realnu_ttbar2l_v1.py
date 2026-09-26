"""Verify the new real-neutrino ttbar2l source against the prior source.

The scan is bounded by HDF5 chunks.  Full-row comparisons are source-contract
checks; model selection and metrics never read the reserved-test partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

DEFAULT_NEW = Path(
    "/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/datasets/ttbar2L/"
    "combined_partonic_afterFSR_v1/raw/"
    "TtbarDilepton_TopRegression_partonic_truth_nu_2M_even.h5"
)
DEFAULT_OLD = Path(
    "/net/scratch/w00238tl/datasets/ttbar2L/combined_partonic_afterFSR_v1/raw/"
    "TtbarDilepton_2M_even.h5"
)
DEFAULT_OUT = Path("results/experiments/partonic_ttbar2L_2M_realnu_v1")
SPLIT_VERSION = "event_number_splitmix64_90_5_5_v1"
SPLIT_SEED = 20260924
SPLIT_NAME = {0: "train", 1: "validation", 2: "reserved_test"}
COMMON = (
    "INPUTS/JET", "INPUTS/ELECTRON", "INPUTS/MUON", "INPUTS/GLOBAL",
    "LABELS/JET", "LABELS/ELECTRON", "LABELS/MUON", "TARGETS/TOP",
    "TARGETS/TRUTH", "META/eventNumber", "META/source_entry_index",
)


def _jsonable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _attrs(obj) -> dict:
    return {str(key): _jsonable(value) for key, value in obj.attrs.items()}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        value = values.astype(np.uint64, copy=False) ^ np.uint64(seed)
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return value ^ (value >> np.uint64(31))


def _schema(source: h5py.File) -> dict:
    datasets = {}

    def walk(group: h5py.Group, prefix: str = ""):
        for name in sorted(group.keys()):
            item = group[name]
            path = f"{prefix}/{name}"
            if isinstance(item, h5py.Group):
                walk(item, path)
            else:
                datasets[path.lstrip("/")] = {
                    "shape": list(item.shape),
                    "dtype": str(item.dtype),
                    "fields": list(item.dtype.names or ()),
                    "chunks": list(item.chunks) if item.chunks else None,
                    "compression": item.compression,
                    "attrs": _attrs(item),
                }

    walk(source)
    return {"root_attrs": _attrs(source), "datasets": datasets}


def _field_equal(left: np.ndarray, right: np.ndarray) -> tuple[int, float]:
    if left.dtype.names or right.dtype.names:
        if left.dtype.names != right.dtype.names:
            return int(left.size), float("inf")
        mismatches = 0
        max_abs = 0.0
        for name in left.dtype.names or ():
            bad, delta = _field_equal(left[name], right[name])
            mismatches += bad
            max_abs = max(max_abs, delta)
        return mismatches, max_abs
    if left.shape != right.shape:
        return int(max(left.size, right.size)), float("inf")
    if np.issubdtype(left.dtype, np.floating) or np.issubdtype(right.dtype, np.floating):
        a = np.asarray(left)
        b = np.asarray(right)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        finite = np.isfinite(a) & np.isfinite(b)
        delta = float(np.max(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)), initial=0.0))
        return int((~same).sum()), delta
    return int((left != right).sum()), 0.0


def _records_to_dense(values: np.ndarray) -> np.ndarray:
    return np.stack([values[name] for name in ("px", "py", "pz")], axis=-1).astype(np.float64)


def _lepton_p4(values: np.ndarray) -> np.ndarray:
    pt = values["pt"].astype(np.float64)
    eta = values["eta"].astype(np.float64)
    phi = values["phi"].astype(np.float64)
    energy = values["e"].astype(np.float64)
    return np.stack((energy, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)), axis=-1)


def _mass(vectors: np.ndarray) -> np.ndarray:
    mass_squared = vectors[..., 0] ** 2 - np.square(vectors[..., 1:]).sum(axis=-1)
    return np.sqrt(np.maximum(mass_squared, 0.0))


def _update_range(stats: dict, name: str, values: np.ndarray, mask: np.ndarray) -> None:
    array = np.asarray(values)
    valid_rows = np.broadcast_to(np.asarray(mask, dtype=bool), array.shape)
    finite = np.isfinite(array)
    current = stats.setdefault(name, {"finite_count": 0, "nonfinite_present_count": 0, "min": None, "max": None})
    selected = array[valid_rows & finite]
    current["finite_count"] += int(selected.size)
    current["nonfinite_present_count"] += int((valid_rows & ~finite).sum())
    if selected.size:
        low = float(selected.min())
        high = float(selected.max())
        current["min"] = low if current["min"] is None else min(current["min"], low)
        current["max"] = high if current["max"] is None else max(current["max"], high)


def _parent_vectors(top: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    records = []
    masses = []
    for role in ("top", "antitop"):
        pt = top[f"{role}_pt"].astype(np.float64)
        eta = top[f"{role}_eta"].astype(np.float64)
        phi = top[f"{role}_phi"].astype(np.float64)
        mass = top[f"{role}_m"].astype(np.float64)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2)
        records.append(np.stack((energy, px, py, pz), axis=-1))
        masses.append(mass)
    return np.stack(records, axis=1), np.stack(masses, axis=1)


def verify(new_path: Path, old_path: Path, out_dir: Path, chunk_size: int, prior_split_path: Path | None = None) -> dict:
    new_path = new_path.expanduser().resolve()
    old_path = old_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    if chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    for path in (new_path, old_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    counts = {
        "common_field_mismatches": {key: 0 for key in COMMON},
        "common_field_max_abs_difference": {key: 0.0 for key in COMMON},
        "channels": {name: 0 for name in ("emu", "ee", "mumu", "other")},
        "lepton_padding_alignment_failures": 0,
        "jet_padding_alignment_failures": 0,
        "charge_label_failures": 0,
        "neutrino_nonfinite_events": 0,
        "neutrino_zero_pair_events": 0,
        "neutrino_source_pair_match_failures": {name: 0 for name in ("emu", "ee", "mumu")},
        "neutrino_source_pair_order": {name: {"same_order": 0, "reversed_order": 0} for name in ("emu", "ee", "mumu")},
        "emu_charge_flavour_role_crosscheck_failures": 0,
        "parent_invalid_events": 0,
        "parent_vs_truth_max_abs_difference_GeV": 0.0,
        "top_pair_closure_max_abs_difference_GeV": 0.0,
        "source_identity_mismatch_chunks": 0,
        "source_entry_mismatch_chunks": 0,
    }
    by_split_channel: dict[str, dict[str, int]] = {
        split: {f"{channel}_{metric}": 0 for channel in ("emu", "ee", "mumu", "other") for metric in ("events", "neutrino_supported")}
        for split in ("train", "validation")
    }
    jet_counts = {str(limit): {"train_validation_events": 0, "reconstructable": 0, "outside_support": 0} for limit in (8, 10, 12, 16, 20)}
    role_lepton_counts = {"positive_charge_label_1": 0, "negative_charge_label_2": 0, "present_leptons": 0}
    source_ids = np.empty(0, dtype=np.int64)
    source_entries = np.empty(0, dtype=np.int64)
    source_schema = None
    old_schema = None
    first_bad = {}
    max_njet = 0
    max_neutrino_pair_diff = {name: 0.0 for name in ("emu", "ee", "mumu")}
    neutrino_pair_close_events = {name: 0 for name in ("emu", "ee", "mumu")}
    neutrino_compared_events = {name: 0 for name in ("emu", "ee", "mumu")}
    w_side_checks = {name: {"events": 0, "top_side_preferred": 0, "tie": 0, "top_w_within_20_gev": 0, "antitop_w_within_20_gev": 0} for name in ("emu", "ee", "mumu")}
    parent_valid_by_channel = {name: 0 for name in ("emu", "ee", "mumu", "other")}
    invisible_support_by_channel = {name: 0 for name in ("emu", "ee", "mumu", "other")}
    lepton_charge_populations = {
        name: {"events": 0, "neutrino_supported": 0}
        for name in ("positive_electron", "negative_electron", "positive_muon", "negative_muon")
    }
    input_feature_ranges = {}
    jet_multiplicity_counts: Counter[int] = Counter()
    label_health = {"invalid_jet_label_values": 0, "invalid_lepton_label_values": 0, "not_one_top_and_antitop_b_label": 0}
    input_p4_quality = {"objects": 0, "nonpositive_energy": 0, "spacelike_beyond_tolerance": 0, "minimum_mass_squared_GeV2": None}
    met_quality = {"events": 0, "invalid_magnitude_events": 0, "max_cos_sin_abs_difference": 0.0}
    assignment_neutrino_intersection = 0
    assignment_parent_intersection = 0
    neutrino_parent_intersection = 0
    all_three_intersection = 0
    legacy_conflict_samples: list[dict[str, int]] = []
    tv_max = 0.0
    ttbar_max = 0.0
    split = None
    role_support = None
    rows = None

    with h5py.File(new_path, "r") as new, h5py.File(old_path, "r") as old:
        source_schema = _schema(new)
        old_schema = _schema(old)
        required_new = (*COMMON, "TARGETS/NEUTRINO")
        missing = [name for name in required_new if name not in new]
        missing_old = [name for name in (*COMMON, "LABELS/NEUTRINO") if name not in old]
        if missing or missing_old:
            raise KeyError(f"Dataset schema missing required paths: new={missing}, old={missing_old}")
        count = int(new["META/eventNumber"].shape[0])
        if count != int(new.attrs.get("n_saved_events", -1)):
            raise ValueError("New-file n_saved_events disagrees with the event axis")
        for key in required_new:
            if not new[key].shape or int(new[key].shape[0]) != count:
                raise ValueError(f"New dataset event axis mismatch: {key}, shape={new[key].shape}, expected={count}")
        for key in (*COMMON, "LABELS/NEUTRINO"):
            if not old[key].shape or int(old[key].shape[0]) != count:
                raise ValueError(f"Old dataset event axis mismatch: {key}, shape={old[key].shape}, expected={count}")
        if int(old["META/eventNumber"].shape[0]) != count:
            raise ValueError("Old and new event counts differ")
        if tuple(new["TARGETS/NEUTRINO"].dtype.names or ()) != ("px", "py", "pz"):
            raise TypeError("TARGETS/NEUTRINO fields must be exactly px,py,pz")
        nu_attrs = _attrs(new["TARGETS/NEUTRINO"])
        expected_nu_attrs = {
            "components": "px,py,pz",
            "order": "top, antitop",
            "slot_0": "neutrino from top (positive-lepton side)",
            "slot_1": "antineutrino from antitop (negative-lepton side)",
            "units": "GeV",
        }
        mismatched_nu_attrs = {
            key: (value, nu_attrs.get(key))
            for key, value in expected_nu_attrs.items()
            if nu_attrs.get(key) != value
        }
        if mismatched_nu_attrs:
            raise ValueError(f"New TARGETS/NEUTRINO role contract is missing or inconsistent: {mismatched_nu_attrs}")
        label_convention = _attrs(new["LABELS/ELECTRON"]).get("label_convention", "")
        if "1=top b or positive charged lepton" not in label_convention or "2=antitop bbar or negative charged lepton" not in label_convention:
            raise ValueError(f"Source lepton label convention does not establish charge-side roles: {label_convention!r}")
        if _attrs(new["TARGETS/TOP"]).get("target_source") != "partonic_after_fsr":
            raise ValueError("TARGETS/TOP is not declared as partonic_after_fsr")
        old_nu_attrs = _attrs(old["LABELS/NEUTRINO"])
        if old_nu_attrs.get("units") != "GeV" or old_nu_attrs.get("components") != "px,py,pz":
            raise ValueError(f"Prior neutrino source component/unit convention is unexpected: {old_nu_attrs}")

        source_ids = np.empty(count, dtype=np.int64)
        source_entries = np.empty(count, dtype=np.int64)
        split = np.empty(count, dtype=np.uint8)
        role_support = np.zeros((count, 2), dtype=np.bool_)
        rows = np.arange(count, dtype=np.int64)
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            size = stop - start
            ids = np.asarray(new["META/eventNumber"][start:stop], dtype=np.int64)
            old_ids = np.asarray(old["META/eventNumber"][start:stop], dtype=np.int64)
            entries = np.asarray(new["META/source_entry_index"][start:stop], dtype=np.int64)
            old_entries = np.asarray(old["META/source_entry_index"][start:stop], dtype=np.int64)
            source_ids[start:stop] = ids
            source_entries[start:stop] = entries
            bucket = (_splitmix64(ids, SPLIT_SEED) % np.uint64(100)).astype(np.uint8)
            split_chunk = np.where(bucket < 90, 0, np.where(bucket < 95, 1, 2)).astype(np.uint8)
            split[start:stop] = split_chunk
            if not np.array_equal(ids, old_ids):
                counts["source_identity_mismatch_chunks"] += 1
                if "event_id" not in first_bad:
                    local = int(np.flatnonzero(ids != old_ids)[0])
                    first_bad["event_id"] = int(start + local)
            if not np.array_equal(entries, old_entries):
                counts["source_entry_mismatch_chunks"] += 1
                if "source_entry_index" not in first_bad:
                    local = int(np.flatnonzero(entries != old_entries)[0])
                    first_bad["source_entry_index"] = int(start + local)
            for key in COMMON:
                if key in ("META/eventNumber", "META/source_entry_index"):
                    continue
                left = new[key][start:stop]
                right = old[key][start:stop]
                bad, max_abs = _field_equal(left, right)
                counts["common_field_mismatches"][key] += bad
                counts["common_field_max_abs_difference"][key] = max(
                    counts["common_field_max_abs_difference"][key], max_abs
                )
                if bad and key not in first_bad:
                    first_bad[key] = {"chunk_start": start, "mismatches": bad}

            jets = new["INPUTS/JET"][start:stop]
            jet_labels = np.asarray(new["LABELS/JET"][start:stop])
            electrons = new["INPUTS/ELECTRON"][start:stop]
            muons = new["INPUTS/MUON"][start:stop]
            e_labels = np.asarray(new["LABELS/ELECTRON"][start:stop])
            m_labels = np.asarray(new["LABELS/MUON"][start:stop])
            global_rows = new["INPUTS/GLOBAL"][start:stop]
            old_nu = _records_to_dense(old["LABELS/NEUTRINO"][start:stop])
            new_nu = _records_to_dense(new["TARGETS/NEUTRINO"][start:stop])
            top = new["TARGETS/TOP"][start:stop]
            truth = new["TARGETS/TRUTH"][start:stop]
            for role in ("top", "antitop"):
                for field in ("pt", "eta", "phi", "m"):
                    _update_range(input_feature_ranges, f"TARGETS.TOP.{role}_{field}", top[f"{role}_{field}"], np.ones(size, dtype=bool))
            for component in ("px", "py", "pz"):
                _update_range(input_feature_ranges, f"TARGETS.NEUTRINO.{component}", new["TARGETS/NEUTRINO"][start:stop][component], np.ones((size, 2), dtype=bool))

            jet_present = np.isfinite(jets["e"]) & np.isfinite(jets["pt"]) & np.isfinite(jets["eta"]) & np.isfinite(jets["phi"])
            e_present = np.isfinite(electrons["e"]) & np.isfinite(electrons["pt"]) & np.isfinite(electrons["eta"]) & np.isfinite(electrons["phi"])
            m_present = np.isfinite(muons["e"]) & np.isfinite(muons["pt"]) & np.isfinite(muons["eta"]) & np.isfinite(muons["phi"])
            bad_jet_padding = (~jet_present) != np.isnan(jet_labels)
            bad_lepton_padding = ((~e_present) != np.isnan(e_labels)).any(axis=1) | ((~m_present) != np.isnan(m_labels)).any(axis=1)
            counts["jet_padding_alignment_failures"] += int(bad_jet_padding.any(axis=1).sum())
            counts["lepton_padding_alignment_failures"] += int(bad_lepton_padding.sum())
            if bad_jet_padding.any() or bad_lepton_padding.any():
                raise ValueError(f"Input padding and truth-label sentinels disagree in rows {start}:{stop}")

            for source_name, values, present, fields in (
                ("JET", jets, jet_present, ("e", "eta", "phi", "pt", "btag", "id")),
                ("ELECTRON", electrons, e_present, ("e", "eta", "phi", "pt", "btag", "charge", "id")),
                ("MUON", muons, m_present, ("e", "eta", "phi", "pt", "btag", "charge", "id")),
            ):
                for field in fields:
                    _update_range(input_feature_ranges, f"{source_name}.{field}", values[field], present)
                    if ((~np.isfinite(values[field])) & present).any():
                        raise ValueError(f"Non-finite measured input {source_name}.{field} on a present object in rows {start}:{stop}")
            for field in global_rows.dtype.names or ():
                _update_range(input_feature_ranges, f"GLOBAL.{field}", global_rows[field], np.ones(size, dtype=bool))
            all_globals = np.stack([global_rows[field] for field in global_rows.dtype.names or ()], axis=-1)
            if not np.isfinite(all_globals).all():
                raise ValueError(f"INPUTS/GLOBAL has non-finite model-conditioning fields in rows {start}:{stop}")
            visible_p4 = np.concatenate((_lepton_p4(jets), _lepton_p4(electrons), _lepton_p4(muons)), axis=1)
            visible_present = np.concatenate((jet_present, e_present, m_present), axis=1)
            visible_m2 = visible_p4[..., 0] ** 2 - np.square(visible_p4[..., 1:]).sum(axis=-1)
            visible_m2_present = visible_m2[visible_present]
            input_p4_quality["objects"] += int(visible_present.sum())
            input_p4_quality["nonpositive_energy"] += int(((visible_p4[..., 0] <= 0.0) & visible_present).sum())
            input_p4_quality["spacelike_beyond_tolerance"] += int(((visible_m2 < -1.0e-2) & visible_present).sum())
            if visible_m2_present.size:
                low_m2 = float(visible_m2_present.min())
                current_low_m2 = input_p4_quality["minimum_mass_squared_GeV2"]
                input_p4_quality["minimum_mass_squared_GeV2"] = low_m2 if current_low_m2 is None else min(current_low_m2, low_m2)

            n_e = e_present.sum(axis=1)
            n_mu = m_present.sum(axis=1)
            emu = (n_e == 1) & (n_mu == 1)
            ee = (n_e == 2) & (n_mu == 0)
            mumu = (n_e == 0) & (n_mu == 2)
            channels = {"emu": emu, "ee": ee, "mumu": mumu, "other": ~(emu | ee | mumu)}
            for name, mask in channels.items():
                counts["channels"][name] += int(mask.sum())
            if channels["other"].any():
                example = ids[np.flatnonzero(channels["other"])[:5]].tolist()
                raise ValueError(f"Unsupported lepton multiplicity in new source; event IDs={example}")

            valid_jet_labels = jet_present & np.isfinite(jet_labels)
            valid_lepton_objects = np.concatenate((e_present, m_present), axis=1)
            all_lepton_labels = np.concatenate((e_labels, m_labels), axis=1)
            label_health["invalid_jet_label_values"] += int((valid_jet_labels & ~np.isin(jet_labels, (0.0, 1.0, 2.0))).sum())
            label_health["invalid_lepton_label_values"] += int((valid_lepton_objects & ~np.isin(all_lepton_labels, (1.0, 2.0))).sum())
            top_b_labels = np.sum(jet_present & (jet_labels == 1.0), axis=1)
            antitop_b_labels = np.sum(jet_present & (jet_labels == 2.0), axis=1)
            label_health["not_one_top_and_antitop_b_label"] += int(((top_b_labels != 1) | (antitop_b_labels != 1)).sum())

            if not np.array_equal(np.asarray(global_rows["nelectron"], dtype=np.int64), n_e):
                raise ValueError(f"INPUTS/GLOBAL nelectron disagrees with object presence in rows {start}:{stop}")
            if not np.array_equal(np.asarray(global_rows["nmuon"], dtype=np.int64), n_mu):
                raise ValueError(f"INPUTS/GLOBAL nmuon disagrees with object presence in rows {start}:{stop}")
            if not np.array_equal(np.asarray(global_rows["njet"], dtype=np.int64), jet_present.sum(axis=1)):
                raise ValueError(f"INPUTS/GLOBAL njet disagrees with stored jet presence in rows {start}:{stop}")
            met_fields = ("met_met", "met_phi", "met_cosphi", "met_sinphi")
            if any(field not in (global_rows.dtype.names or ()) for field in met_fields):
                raise KeyError(f"INPUTS/GLOBAL must contain measured MET fields {met_fields}")
            met_pt = np.asarray(global_rows["met_met"], dtype=np.float64)
            met_phi = np.asarray(global_rows["met_phi"], dtype=np.float64)
            met_cos = np.asarray(global_rows["met_cosphi"], dtype=np.float64)
            met_sin = np.asarray(global_rows["met_sinphi"], dtype=np.float64)
            met_delta = np.maximum(np.abs(met_cos - np.cos(met_phi)), np.abs(met_sin - np.sin(met_phi)))
            met_quality["events"] += size
            met_quality["invalid_magnitude_events"] += int((met_pt < 0.0).sum())
            met_quality["max_cos_sin_abs_difference"] = max(met_quality["max_cos_sin_abs_difference"], float(met_delta.max(initial=0.0)))
            if (met_pt < 0.0).any() or (met_delta > 5.0e-4).any():
                raise ValueError(f"Measured MET magnitude/direction is invalid or inconsistent in rows {start}:{stop}")
            # Count real lepton labels from both fixed-width source arrays.
            pos_count = ((e_present & (electrons["charge"] > 0) & (e_labels == 1)).sum(axis=1)
                         + (m_present & (muons["charge"] > 0) & (m_labels == 1)).sum(axis=1))
            neg_count = ((e_present & (electrons["charge"] < 0) & (e_labels == 2)).sum(axis=1)
                         + (m_present & (muons["charge"] < 0) & (m_labels == 2)).sum(axis=1))
            invalid_charge = (~np.isfinite(electrons["charge"]) & e_present).any(axis=1) | (~np.isfinite(muons["charge"]) & m_present).any(axis=1)
            failures = (pos_count != 1) | (neg_count != 1) | invalid_charge
            counts["charge_label_failures"] += int(failures.sum())
            if failures.any():
                bad_ids = ids[np.flatnonzero(failures)[:10]].tolist()
                raise ValueError(f"Charge and source lepton side labels disagree; event IDs={bad_ids}")
            role_lepton_counts["present_leptons"] += int(n_e.sum() + n_mu.sum())
            role_lepton_counts["positive_charge_label_1"] += int(pos_count.sum())
            role_lepton_counts["negative_charge_label_2"] += int(neg_count.sum())

            nu_finite = np.isfinite(new_nu).all(axis=(1, 2))
            counts["neutrino_nonfinite_events"] += int((~nu_finite).sum())
            if (~nu_finite).any():
                raise ValueError(f"New neutrino truth has non-finite p components in rows {start}:{stop}")
            counts["neutrino_zero_pair_events"] += int((np.linalg.norm(new_nu, axis=-1) == 0.0).any(axis=1).sum())
            for name, mask in (("emu", emu), ("ee", ee), ("mumu", mumu)):
                # The old file calls its slots electron/muon flavour ordered,
                # but that label is not a reliable charge-side association for
                # this population. Compare the two-vector payload without
                # assigning either old slot a top/antitop identity.
                direct = np.max(np.abs(old_nu - new_nu), axis=(1, 2))
                reverse = np.max(np.abs(old_nu[:, ::-1, :] - new_nu), axis=(1, 2))
                direct_close = direct <= 5.0e-3
                reverse_close = reverse <= 5.0e-3
                event_close = direct_close | reverse_close
                counts["neutrino_source_pair_match_failures"][name] += int((mask & ~event_close).sum())
                counts["neutrino_source_pair_order"][name]["same_order"] += int((mask & direct_close).sum())
                counts["neutrino_source_pair_order"][name]["reversed_order"] += int((mask & reverse_close & ~direct_close).sum())
                neutrino_pair_close_events[name] += int((mask & event_close).sum())
                neutrino_compared_events[name] += int(mask.sum())
                if mask.any():
                    max_neutrino_pair_diff[name] = max(
                        max_neutrino_pair_diff[name],
                        float(np.max(np.minimum(direct[mask], reverse[mask]), initial=0.0)),
                    )

            # Keep the historical e/mu flavor-plus-charge comparison as a
            # diagnostic. TARGETS/NEUTRINO's producer-level top/antitop
            # contract is authoritative, so these conflicts never change
            # role assignment or support.
            e_slot = np.argmax(e_present, axis=1)
            e_top = e_labels[np.arange(size), e_slot] == 1.0
            old_top_emu = np.where(e_top[:, None], old_nu[:, 0, :], old_nu[:, 1, :])
            old_antitop_emu = np.where(e_top[:, None], old_nu[:, 1, :], old_nu[:, 0, :])
            legacy_flavor_side_matches = (
                (np.max(np.abs(old_top_emu - new_nu[:, 0, :]), axis=1) <= 5.0e-3)
                & (np.max(np.abs(old_antitop_emu - new_nu[:, 1, :]), axis=1) <= 5.0e-3)
            )
            legacy_conflicts = emu & ~legacy_flavor_side_matches
            counts["emu_charge_flavour_role_crosscheck_failures"] += int(legacy_conflicts.sum())
            if len(legacy_conflict_samples) < 8:
                for local_index in np.flatnonzero(legacy_conflicts & (split_chunk != 2)):
                    legacy_conflict_samples.append(
                        {"event_id": int(ids[local_index]), "source_row": int(start + local_index)}
                    )
                    if len(legacy_conflict_samples) == 8:
                        break

            # Independent charge-side check: each new massless neutrino is
            # paired with the positive/negative measured lepton according to
            # the declared top/antitop order. The original source's flavor-slot
            # metadata is deliberately not used to choose these assignments.
            measured_leptons = np.concatenate((_lepton_p4(electrons), _lepton_p4(muons)), axis=1)
            measured_labels = np.concatenate((e_labels, m_labels), axis=1)
            measured_present = np.concatenate((e_present, m_present), axis=1)
            role_p4 = []
            for label in (1.0, 2.0):
                selected = measured_present & (measured_labels == label)
                index = np.argmax(selected, axis=1)
                role_p4.append(measured_leptons[np.arange(size), index])
            positive_lepton, negative_lepton = role_p4
            nu_energy = np.linalg.norm(new_nu, axis=-1)
            nu_p4 = np.concatenate((nu_energy[..., None], new_nu), axis=-1)
            w_top = _mass(positive_lepton + nu_p4[:, 0])
            w_antitop = _mass(negative_lepton + nu_p4[:, 1])
            w_top_swapped = _mass(positive_lepton + nu_p4[:, 1])
            w_antitop_swapped = _mass(negative_lepton + nu_p4[:, 0])
            identity_cost = np.abs(w_top - 80.379) + np.abs(w_antitop - 80.379)
            swapped_cost = np.abs(w_top_swapped - 80.379) + np.abs(w_antitop_swapped - 80.379)
            train_validation = split_chunk != 2
            for channel, channel_mask in channels.items():
                if channel == "other":
                    continue
                selected = train_validation & channel_mask
                if not selected.any():
                    continue
                w_side_checks[channel]["events"] += int(selected.sum())
                w_side_checks[channel]["top_side_preferred"] += int((selected & (identity_cost < swapped_cost)).sum())
                w_side_checks[channel]["tie"] += int((selected & (identity_cost == swapped_cost)).sum())
                w_side_checks[channel]["top_w_within_20_gev"] += int((selected & (np.abs(w_top - 80.379) < 20.0)).sum())
                w_side_checks[channel]["antitop_w_within_20_gev"] += int((selected & (np.abs(w_antitop - 80.379) < 20.0)).sum())

            # After-FSR parent p4 and redundant truth p4 closure.
            physical, masses = _parent_vectors(top)
            truth_vectors = np.stack(
                (
                    np.stack((truth["truth_top_E"], truth["truth_top_px"], truth["truth_top_py"], truth["truth_top_pz"]), axis=-1),
                    np.stack((truth["truth_antitop_E"], truth["truth_antitop_px"], truth["truth_antitop_py"], truth["truth_antitop_pz"]), axis=-1),
                ),
                axis=1,
            ).astype(np.float64)
            finite_parent = np.isfinite(physical).all(axis=(1, 2)) & np.isfinite(masses).all(axis=1)
            pt_nonnegative = np.stack(
                (top["top_pt"] >= 0.0, top["antitop_pt"] >= 0.0), axis=1
            ).all(axis=1)
            valid_parent = finite_parent & (masses >= 0.0).all(axis=1) & pt_nonnegative & (physical[:, :, 0] >= 0.0).all(axis=1)
            counts["parent_invalid_events"] += int((~valid_parent).sum())
            if (~valid_parent).any():
                bad_ids = ids[np.flatnonzero(~valid_parent)[:10]].tolist()
                raise ValueError(f"Invalid partonic parent truth in source; event IDs={bad_ids}")
            parent_delta = np.abs(physical - truth_vectors)
            parent_error = float(np.max(parent_delta, initial=0.0))
            tv_max = max(tv_max, parent_error)
            counts["parent_vs_truth_max_abs_difference_GeV"] = tv_max
            if parent_error > 5.0e-3:
                raise ValueError(f"TARGETS/TOP and TARGETS/TRUTH differ by {parent_error} GeV")
            pair = physical[:, 0, :] + physical[:, 1, :]
            stored_pair = np.stack((truth["truth_ttbar_E"], truth["truth_ttbar_px"], truth["truth_ttbar_py"], truth["truth_ttbar_pz"]), axis=-1).astype(np.float64)
            pair_error = float(np.max(np.abs(pair - stored_pair), initial=0.0))
            ttbar_max = max(ttbar_max, pair_error)
            counts["top_pair_closure_max_abs_difference_GeV"] = ttbar_max
            if pair_error > 1.0e-2:
                raise ValueError(f"Parent sum and stored ttbar truth differ by {pair_error} GeV")

            # The new source has no separate neutrino validity flag. Event
            # identity and role presence were validated above; finite px,py,pz
            # in each authoritative role is the support rule. A zero vector
            # remains supported if it occurs.
            valid_nu = nu_finite
            role_support[start:stop] = valid_nu[:, None]
            njet_field = np.asarray(global_rows["njet"], dtype=np.int64)
            max_njet = max(max_njet, int(njet_field.max(initial=0)))
            jet_multiplicity_counts.update(int(value) for value in njet_field)
            tv = split_chunk != 2
            for split_id, split_name in ((0, "train"), (1, "validation")):
                in_split = split_chunk == split_id
                for channel, mask in channels.items():
                    key = f"{channel}_events"
                    by_split_channel[split_name][key] += int((in_split & mask).sum())
                    by_split_channel[split_name][f"{channel}_neutrino_supported"] += int((in_split & mask & valid_nu).sum())
            for name, mask in channels.items():
                parent_valid_by_channel[name] += int((mask & valid_parent & (split_chunk != 2)).sum())
                invisible_support_by_channel[name] += int((mask & valid_nu & (split_chunk != 2)).sum())
            charge_groups = {
                "positive_electron": (e_present & (electrons["charge"] > 0)).any(axis=1),
                "negative_electron": (e_present & (electrons["charge"] < 0)).any(axis=1),
                "positive_muon": (m_present & (muons["charge"] > 0)).any(axis=1),
                "negative_muon": (m_present & (muons["charge"] < 0)).any(axis=1),
            }
            for name, mask in charge_groups.items():
                lepton_charge_populations[name]["events"] += int((mask & (split_chunk != 2)).sum())
                lepton_charge_populations[name]["neutrino_supported"] += int((mask & valid_nu & (split_chunk != 2)).sum())

            # Full train + development scan of the b-jet truth support.  The
            # sealed test partition is excluded from support selection.
            in_tv = tv
            labels_in_tv = jet_labels[in_tv]
            n_tv = int(in_tv.sum())
            if n_tv:
                for limit in (8, 10, 12, 16, 20):
                    inside = (np.sum(labels_in_tv[:, :limit] == 1.0, axis=1) == 1) & (np.sum(labels_in_tv[:, :limit] == 2.0, axis=1) == 1)
                    jet_counts[str(limit)]["train_validation_events"] += n_tv
                    jet_counts[str(limit)]["reconstructable"] += int(inside.sum())
                    jet_counts[str(limit)]["outside_support"] += int((~inside).sum())
                inside8 = (np.sum(labels_in_tv[:, :8] == 1.0, axis=1) == 1) & (np.sum(labels_in_tv[:, :8] == 2.0, axis=1) == 1)
                parent_mask = valid_parent[in_tv]
                neutrino_mask = valid_nu[in_tv]
                assignment_neutrino_intersection += int((inside8 & neutrino_mask).sum())
                assignment_parent_intersection += int((inside8 & parent_mask).sum())
                neutrino_parent_intersection += int((neutrino_mask & parent_mask).sum())
                all_three_intersection += int((inside8 & neutrino_mask & parent_mask).sum())

        if counts["source_identity_mismatch_chunks"] or counts["source_entry_mismatch_chunks"]:
            raise ValueError(f"Old/new source identity mismatch: {first_bad}")
        if any(counts["common_field_mismatches"].values()):
            raise ValueError(f"Common old/new input or truth fields changed: {first_bad}")
        if not np.all(source_ids % 2 == 0):
            raise ValueError("New source contains eventNumber values that are not even")
        if len(np.unique(source_ids)) != len(source_ids):
            raise ValueError("New source META/eventNumber is not unique")
        if len(np.unique(source_entries)) != len(source_entries):
            raise ValueError("New source META/source_entry_index is not unique")
        if counts["neutrino_nonfinite_events"]:
            raise ValueError("New source has non-finite neutrino targets")
        if label_health["invalid_jet_label_values"] or label_health["invalid_lepton_label_values"]:
            raise ValueError(f"Source contains invalid visible truth labels: {label_health}")
        if input_p4_quality["nonpositive_energy"]:
            raise ValueError(f"Measured present object has nonpositive energy: {input_p4_quality}")

    # For the source-pair validation, each populated channel must reproduce
    # the new two-vector truth payload, allowing only slot reordering.
    if any(counts["neutrino_source_pair_match_failures"].values()):
        raise ValueError(
            "New neutrino targets do not reproduce the previous source two-vector payload; "
            f"failures={counts['neutrino_source_pair_match_failures']}, max_abs={max_neutrino_pair_diff}"
        )

    split = np.asarray(split, dtype=np.uint8)
    split_counts = {name: int(np.count_nonzero(split == key)) for key, name in SPLIT_NAME.items()}
    membership_path = out_dir / "splits" / "split_membership_realnu_v1.npz"
    prior_split_details = {"exact_prior_membership_reused": False}
    if prior_split_path is not None:
        prior_split_path = prior_split_path.expanduser().resolve()
        if not prior_split_path.is_file():
            raise FileNotFoundError(f"The prior split required for exact reuse is missing: {prior_split_path}")
        with np.load(prior_split_path, allow_pickle=False) as prior:
            expected = {
                "event_id": source_ids,
                "source_entry_index": source_entries,
                "split": split,
                "source_row": rows,
            }
            bad = [key for key, value in expected.items() if key not in prior or not np.array_equal(prior[key], value)]
            if bad:
                raise ValueError(
                    "New source does not have the same stable population and exact frozen split as the prior source: "
                    f"{bad}"
                )
            prior_split_details = {
                "exact_prior_membership_reused": True,
                "prior_membership_path": str(prior_split_path),
                "prior_membership_sha256": _sha256(prior_split_path),
            }
    _atomic_npz(
        membership_path,
        source_row=rows,
        event_id=source_ids,
        source_entry_index=source_entries,
        split=split,
    )
    _atomic_copy(membership_path, out_dir / "split_membership_v1.npz")
    split_hash = _sha256(membership_path)
    reserved_path = out_dir / "splits" / "reserved_test_event_ids_realnu_v1.npz"
    reserved_mask = split == 2
    _atomic_npz(
        reserved_path,
        event_id=source_ids[reserved_mask],
        source_entry_index=source_entries[reserved_mask],
        source_row=rows[reserved_mask],
    )
    reserved_hash = _sha256(reserved_path)
    support_path = out_dir / "splits" / "neutrino_role_support_train_validation_realnu_v1.npz"
    train_validation = split != 2
    _atomic_npz(
        support_path,
        source_row=rows[train_validation],
        event_id=source_ids[train_validation],
        support=role_support[train_validation],
    )
    support_hash = _sha256(support_path)
    unique_event_hash = hashlib.sha256(source_ids.astype("<i8", copy=False).tobytes()).hexdigest()
    source_entry_hash = hashlib.sha256(source_entries.astype("<i8", copy=False).tobytes()).hexdigest()
    new_stat, old_stat = new_path.stat(), old_path.stat()
    manifest = {
        "manifest_version": "partonic_ttbar2l_realnu_source_v1",
        "new_source": {
            "path": str(new_path), "size_bytes": new_stat.st_size, "sha256": _sha256(new_path),
            "event_count": len(source_ids), "schema": source_schema,
        },
        "comparison_source": {
            "path": str(old_path), "size_bytes": old_stat.st_size, "sha256": _sha256(old_path),
            "event_count": len(source_ids), "schema": old_schema,
        },
        "source_pair_checks": {
            "event_number_and_source_entry_index_exactly_aligned": True,
            "all_common_measured_inputs_labels_partonic_targets_and_truth_exactly_equal": True,
            "full_row_source_fields": counts,
            "first_mismatch": first_bad,
        },
        "identity": {
            "eventNumber_unique": True, "source_entry_index_unique": True,
            "all_event_numbers_even": True,
            "eventNumber_ordered_sequence_sha256": unique_event_hash,
            "source_entry_index_ordered_sequence_sha256": source_entry_hash,
            "event_number_min": int(source_ids.min()), "event_number_max": int(source_ids.max()),
            "source_entry_min": int(source_entries.min()), "source_entry_max": int(source_entries.max()),
        },
        "split": {
            "version": SPLIT_VERSION, "seed": SPLIT_SEED, "identity": "META/eventNumber",
            "allocation": {"train": "SplitMix64(eventNumber XOR seed) mod 100 < 90", "validation": "90..94", "reserved_test": "95..99"},
            "counts": split_counts, "membership_path": str(membership_path.relative_to(out_dir)),
            "membership_sha256": split_hash,
            "prior_membership_verification": prior_split_details,
            "reserved_event_identity_path": str(reserved_path.relative_to(out_dir)),
            "reserved_event_identity_sha256": reserved_hash,
            "neutrino_role_support_path": str(support_path.relative_to(out_dir)),
            "neutrino_role_support_sha256": support_hash,
        },
        "truth_contract": {
            "parent_target": "TARGETS/TOP: top then antitop, partonic_after_fsr, pt_eta_phi_m in GeV",
            "neutrino_target": "TARGETS/NEUTRINO: top then antitop, px_py_pz in GeV",
            "new_source_branch_mapping": _attrs_from_schema(source_schema, "TARGETS/NEUTRINO"),
            "old_neutrino_source": _attrs_from_schema(old_schema, "LABELS/NEUTRINO"),
            "new_side_vs_old_source_max_abs_GeV_up_to_pair_permutation": max_neutrino_pair_diff,
            "new_side_vs_old_source_close_events": neutrino_pair_close_events,
            "new_side_vs_old_source_compared_events": neutrino_compared_events,
            "old_slot_order_vs_new_role_order": counts["neutrino_source_pair_order"],
            "charge_side_w_mass_check_train_validation_only": {
                channel: {
                    **values,
                    "top_side_preferred_fraction": values["top_side_preferred"] / max(values["events"], 1),
                    "top_w_within_20_gev_fraction": values["top_w_within_20_gev"] / max(values["events"], 1),
                    "antitop_w_within_20_gev_fraction": values["antitop_w_within_20_gev"] / max(values["events"], 1),
                }
                for channel, values in w_side_checks.items()
            },
            "charge_side_label_semantics": _attrs_from_schema(source_schema, "LABELS/ELECTRON").get("label_convention"),
            "invisible_role_convention": "top_antitop",
            "invisible_role_order": ["top.nu", "antitop.nu"],
            "neutrino_truth_definition": "generator top-side neutrino and generator antitop-side neutrino",
            "units": "GeV",
            "synthetic": False,
            "same_flavour_slot_interpretation": "The prior source's same-flavour slots do not define top/antitop identity. TARGETS/NEUTRINO producer metadata defines the role order; prior-source pair matching and charge-side W-mass checks are diagnostics only.",
            "producer_source_or_documentation": "The authoritative producer-level HDF5 contract declares TARGETS/NEUTRINO order=top, antitop, slot_0=neutrino from top, and slot_1=antineutrino from antitop, with px,py,pz in GeV. Prior flavor-slot and charge-side comparisons are retained only as diagnostics.",
            "train_validation_event_count": int((split != 2).sum()),
            "unsupported_neutrino_events": int((~role_support[split != 2].all(axis=1)).sum()),
            "supported_event_count_by_role": {"top.nu": int(sum(invisible_support_by_channel.values())), "antitop.nu": int(sum(invisible_support_by_channel.values()))},
            "support_policy": "train/validation event identity is valid and each producer-declared TARGETS/NEUTRINO role exists with finite px,py,pz; no explicit producer validity flag is present; no flavor/charge inference is applied",
            "e_mu_old_flavour_charge_mapping_conflicts": counts["emu_charge_flavour_role_crosscheck_failures"],
            "e_mu_old_flavour_charge_mapping_conflict_sample_train_validation": legacy_conflict_samples,
        },
        "support": {
            "channel_counts": counts["channels"],
            "by_split_channel_counts": by_split_channel,
            "parent_valid_by_channel": parent_valid_by_channel,
            "neutrino_supported_by_channel": invisible_support_by_channel,
            "neutrino_supported_by_role": {
                role: {"count": int(sum(invisible_support_by_channel.values())), "fraction": float(sum(invisible_support_by_channel.values()) / max(int((split != 2).sum()), 1))}
                for role in ("top.nu", "antitop.nu")
            },
            "neutrino_supported_by_charge_population": lepton_charge_populations,
            "jet_limit_reconstructability_train_validation_only": {
                limit: {**values, "fraction": values["reconstructable"] / max(values["train_validation_events"], 1)}
                for limit, values in jet_counts.items()
            },
            "max_input_njet": max_njet,
            "input_njet_counts": {str(key): int(value) for key, value in sorted(jet_multiplicity_counts.items())},
            "intersections_train_validation_at_8_jets": {
                "assignment_and_neutrino": assignment_neutrino_intersection,
                "assignment_and_partonic": assignment_parent_intersection,
                "neutrino_and_partonic": neutrino_parent_intersection,
                "all_three": all_three_intersection,
            },
        },
        "physics_closure": {
            "parent_invalid_events": counts["parent_invalid_events"],
            "parent_target_vs_redundant_truth_max_abs_GeV": tv_max,
            "top_antitop_sum_vs_stored_pair_max_abs_GeV": ttbar_max,
            "lepton_charge_role_failures": counts["charge_label_failures"],
            "input_truth_padding_alignment_failures": {
                "jet": counts["jet_padding_alignment_failures"],
                "lepton": counts["lepton_padding_alignment_failures"],
            },
            "visible_label_health": label_health,
            "measured_input_feature_ranges_GeV_or_native": input_feature_ranges,
            "measured_input_four_vector_quality": input_p4_quality,
            "measured_MET_quality": met_quality,
        },
    }
    manifest_path = out_dir / "manifests" / "source_manifest_realnu_v1.json"
    _atomic_json(manifest_path, manifest)
    _atomic_json(out_dir / "manifests" / "source_manifest.json", manifest)
    _atomic_json(out_dir / "source_manifest_v1.json", manifest)
    contract_path = out_dir / "audit" / "source_contract_realnu_v1.json"
    _atomic_json(contract_path, manifest)
    summary_lines = [
        "# Real-neutrino source contract",
        "",
        f"- New source: `{new_path}` ({len(source_ids):,} events; SHA256 `{manifest['new_source']['sha256']}`).",
        f"- Prior source: `{old_path}` ({len(source_ids):,} events; SHA256 `{manifest['comparison_source']['sha256']}`).",
        "- Event IDs and source-entry IDs are unique, even where required, and align row by row.",
        f"- Stable split: `{SPLIT_VERSION}`, seed `{SPLIT_SEED}`, counts `{split_counts}`, membership SHA256 `{split_hash}`.",
        f"- Channels: `{counts['channels']}`; parent-target invalid rows `{counts['parent_invalid_events']}`.",
        f"- Neutrino pair matches old source up to slot permutation: `{counts['neutrino_source_pair_match_failures']}` failures; maximum pairwise differences `{max_neutrino_pair_diff}` GeV.",
        f"- Old slot permutation counts by channel: `{counts['neutrino_source_pair_order']}`. Old flavor-slot metadata is not used to assign top/antitop roles.",
        f"- Legacy e-mu flavor-plus-charge disagreement diagnostic: `{counts['emu_charge_flavour_role_crosscheck_failures']}` rows across the source; this count does not alter neutrino role support or reorder targets.",
        f"- Train/validation charge-side W-mass pairing preferred the declared new top/antitop order in `{ {k: (v['top_side_preferred'], v['events']) for k,v in w_side_checks.items()} }` events.",
        f"- Top/antitop versus redundant parent truth maximum absolute difference: `{tv_max}` GeV; parent sum versus stored pair: `{ttbar_max}` GeV.",
        "- The new source attributes declare `truth_nu1_*` as top-side and `truth_nu2_*` as antitop-side. The previous source contains the same two neutrino vectors for each event, but their slot order changes between these role slots; the old electron/muon slot names therefore do not establish charge-side order. The new role order is independently supported by W-mass compatibility with the charge-labeled leptons on train and validation rows.",
        f"- Authoritative real-neutrino support uses train/validation event identity and finite producer-declared top/antitop px,py,pz only: {sum(invisible_support_by_channel.values()):,}/{manifest['truth_contract']['train_validation_event_count']:,} events; unsupported event count `{manifest['truth_contract']['unsupported_neutrino_events']}`. Role-level counts: `{manifest['truth_contract']['supported_event_count_by_role']}`.",
        "- The producer-level HDF5 role metadata is authoritative. The legacy flavor/charge mismatch, old-source pair comparison, and W-mass checks are diagnostics only; none changes support or target order.",
        "- Reserved-test rows are excluded from support counts, supervision, model selection, and evaluation.",
        "",
        "## Train/validation jet support",
        "",
        "| Jet limit | Events | Reconstructable | Fraction |",
        "|---:|---:|---:|---:|",
    ]
    for limit, value in jet_counts.items():
        summary_lines.append(f"| {limit} | {value['train_validation_events']:,} | {value['reconstructable']:,} | {value['reconstructable'] / max(value['train_validation_events'], 1):.6%} |")
    summary_lines.extend(("", "## Neutrino role support by channel", "", "| Channel | Events | Joint role support | Fraction |", "|---|---:|---:|---:|"))
    for channel, total in counts["channels"].items():
        supported = invisible_support_by_channel[channel]
        summary_lines.append(f"| {channel} | {total:,} | {supported:,} | {supported / max(total, 1):.6%} |")
    summary_lines.extend(("", f"Machine-readable manifest: `{manifest_path}`.", f"Detailed JSON: `{contract_path}`.", ""))
    report_path = out_dir / "audit" / "source_contract_realnu_v1.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(summary_lines)
    report_path.write_text(report_text, encoding="utf-8")
    (out_dir / "audit" / "source_contract.md").write_text(report_text, encoding="utf-8")
    return {"manifest": str(manifest_path), "report": str(report_path), "split_hash": split_hash, "counts": counts, "support": manifest["support"]}


def _attrs_from_schema(schema: dict, path: str) -> dict:
    return schema["datasets"].get(path, {}).get("attrs", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-source", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--old-source", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument(
        "--prior-split",
        type=Path,
        default=Path("results/experiments/partonic_ttbar2L_2M_v1/split_membership_v1.npz"),
    )
    args = parser.parse_args()
    print(json.dumps(verify(args.new_source, args.old_source, args.output_dir, args.chunk_size, args.prior_split), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
