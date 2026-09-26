"""Select reproducible bounded CPU fixtures from frozen train and validation."""

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
    "/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/datasets/ttbar2L/"
    "combined_partonic_afterFSR_v1/raw/"
    "TtbarDilepton_TopRegression_partonic_truth_nu_2M_even.h5"
)
DEFAULT_MANIFEST_DIR = Path("results/experiments/partonic_ttbar2L_2M_realnu_v1")
DEFAULT_OUTPUT = DEFAULT_MANIFEST_DIR / "cpu_development_ids_v1.npz"


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        value = values.astype(np.uint64, copy=False) ^ np.uint64(seed)
        value = value + np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return value ^ (value >> np.uint64(31))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(values, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _required_category_keys(event: dict) -> tuple[str, ...]:
    keys = [f"channel:{event['channel']}", f"jet_bin:{event['jet_bin']}"]
    keys.append(f"visible_support:{'supported' if event['visible_support'] else 'unsupported'}")
    keys.append(f"neutrino_support:{'supported' if event['neutrino_support'] else 'unsupported'}")
    keys.append(f"mttbar_bin:{event['mttbar_bin']}")
    keys.append(f"parent_pt:{'high' if event['top_pt'] >= 500.0 else 'standard'}")
    if event["channel"] == "emu":
        keys.append(f"charge_order:{event['charge_order']}")
    else:
        keys.append(f"same_flavour_positive_slot:{event['positive_slot']}")
    return tuple(keys)


def _channel_and_charge(electron, muon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_present = np.isfinite(electron["pt"]) & (electron["pt"] > 0.0)
    m_present = np.isfinite(muon["pt"]) & (muon["pt"] > 0.0)
    n_e = e_present.sum(axis=1)
    n_mu = m_present.sum(axis=1)
    channel = np.full(len(electron), "other", dtype="U4")
    channel[(n_e == 1) & (n_mu == 1)] = "emu"
    channel[(n_e == 2) & (n_mu == 0)] = "ee"
    channel[(n_e == 0) & (n_mu == 2)] = "mumu"
    e_positive = e_present & (electron["charge"] > 0.0)
    m_positive = m_present & (muon["charge"] > 0.0)
    charge_order = np.full(len(electron), "na", dtype="U24")
    charge_order[(channel == "emu") & e_positive.any(axis=1)] = "eplus_muonminus"
    charge_order[(channel == "emu") & ~e_positive.any(axis=1)] = "eminus_muonplus"
    positive_slot = np.full(len(electron), -1, dtype=np.int8)
    ee_mask = channel == "ee"
    mm_mask = channel == "mumu"
    positive_slot[ee_mask] = e_positive[ee_mask].argmax(axis=1)
    positive_slot[mm_mask] = m_positive[mm_mask].argmax(axis=1)
    return channel, charge_order, positive_slot


def build(
    source_path: Path,
    manifest_dir: Path,
    output_path: Path,
    *,
    train_count: int = 512,
    validation_count: int = 256,
    seed: int = 20260925,
    chunk_size: int = 8192,
) -> dict:
    source_path = source_path.expanduser().resolve()
    manifest_dir = manifest_dir.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if train_count <= 0 or validation_count <= 0 or chunk_size <= 0:
        raise ValueError("Subset counts and chunk size must be positive")
    split_path = manifest_dir / "split_membership_v1.npz"
    support_path = manifest_dir / "splits" / "neutrino_role_support_train_validation_realnu_v1.npz"
    if not source_path.is_file() or not split_path.is_file() or not support_path.is_file():
        raise FileNotFoundError(f"Missing source, split or role-support input: {source_path}, {split_path}, {support_path}")
    with np.load(split_path, allow_pickle=False) as split_file:
        rows = split_file["source_row"].astype(np.int64, copy=True)
        event_ids = split_file["event_id"].astype(np.int64, copy=True)
        split_values = split_file["split"].astype(np.uint8, copy=True)
    train_rows = rows[split_values == 0]
    validation_rows = rows[split_values == 1]
    with np.load(support_path, allow_pickle=False) as support_file:
        support_rows = support_file["source_row"].astype(np.int64, copy=True)
        support_ids = support_file["event_id"].astype(np.int64, copy=True)
        support_values = support_file["support"].astype(np.bool_, copy=True)
    if support_values.shape != (len(support_rows), 2):
        raise ValueError("Neutrino role-support product must have shape [train+validation,2]")
    tv_mask = split_values != 2
    if not np.array_equal(support_rows, rows[tv_mask]) or not np.array_equal(support_ids, event_ids[tv_mask]):
        raise ValueError("Neutrino role support does not exactly match train/validation event identity")

    # Keep one deterministic representative for each required coverage label.
    best: dict[int, dict[str, tuple[int, int]]] = {0: {}, 1: {}}
    trainval_rows = np.sort(np.concatenate((train_rows, validation_rows)))
    with h5py.File(source_path, "r") as source:
        if len(source["META/eventNumber"]) != len(rows):
            raise ValueError("Source event axis differs from frozen split")
        for start in range(0, len(trainval_rows), chunk_size):
            selected_rows = trainval_rows[start : start + chunk_size]
            if not len(selected_rows):
                continue
            ids = np.asarray(source["META/eventNumber"][selected_rows], dtype=np.int64)
            expected_ids = event_ids[selected_rows]
            if not np.array_equal(ids, expected_ids):
                raise ValueError("Development sample source rows do not match frozen EVENT_ID values")
            split_chunk = split_values[selected_rows]
            e = source["INPUTS/ELECTRON"][selected_rows]
            m = source["INPUTS/MUON"][selected_rows]
            global_rows = source["INPUTS/GLOBAL"][selected_rows]
            jets = source["LABELS/JET"][selected_rows, :8]
            neutrino_bins = source["TARGETS/TRUTH"][selected_rows]["truth_mttbar_bin"].astype(np.int64)
            top_pt = source["TARGETS/TOP"][selected_rows]["top_pt"].astype(np.float64)
            channel, charge_order, positive_slot = _channel_and_charge(e, m)
            njet = np.asarray(global_rows["njet"], dtype=np.int64)
            jet_bin = np.where(njet <= 3, "low", np.where(njet <= 6, "mid", "high"))
            visible_support = (np.sum(jets == 1.0, axis=1) == 1) & (np.sum(jets == 2.0, axis=1) == 1)
            support_positions = np.searchsorted(support_rows, selected_rows)
            if (support_positions >= len(support_rows)).any() or not np.array_equal(support_rows[support_positions], selected_rows):
                raise ValueError("Development selection found a source row missing from role-support manifest")
            if not np.array_equal(support_ids[support_positions], ids):
                raise ValueError("Development selection support EVENT_ID join mismatch")
            neutrino_supported = support_values[support_positions].all(axis=1)
            for local, row in enumerate(selected_rows):
                split_id = int(split_chunk[local])
                event = {
                    "channel": str(channel[local]),
                    "charge_order": str(charge_order[local]),
                    "positive_slot": int(positive_slot[local]),
                    "jet_bin": str(jet_bin[local]),
                    "visible_support": bool(visible_support[local]),
                    "neutrino_support": bool(neutrino_supported[local]),
                    "mttbar_bin": int(neutrino_bins[local]),
                    "top_pt": float(top_pt[local]),
                }
                hash_value = int(_splitmix64(np.asarray([ids[local]], dtype=np.int64), seed ^ (split_id << 16))[0])
                for key in _required_category_keys(event):
                    previous = best[split_id].get(key)
                    candidate = (hash_value, int(row))
                    if previous is None or candidate < previous:
                        best[split_id][key] = candidate

    selected_by_split = {}
    categories_by_split = {}
    for split_id, target_count, split_rows in (
        (0, train_count, train_rows),
        (1, validation_count, validation_rows),
    ):
        if target_count > len(split_rows):
            raise ValueError(f"Requested {target_count} events exceeds split {split_id} population {len(split_rows)}")
        mandatory_rows = np.asarray(sorted({value[1] for value in best[split_id].values()}), dtype=np.int64)
        if len(mandatory_rows) > target_count:
            raise ValueError(f"Coverage contract needs {len(mandatory_rows)} mandatory rows, more than requested {target_count}")
        available = np.setdiff1d(split_rows, mandatory_rows, assume_unique=True)
        rng = np.random.default_rng(seed + 1009 * (split_id + 1))
        fill = rng.choice(available, target_count - len(mandatory_rows), replace=False)
        selected = np.sort(np.concatenate((mandatory_rows, fill))).astype(np.int64)
        selected_by_split[split_id] = selected
        categories_by_split[split_id] = sorted(best[split_id])

    train_selected = selected_by_split[0]
    validation_selected = selected_by_split[1]
    train_ids = event_ids[train_selected]
    validation_ids = event_ids[validation_selected]
    if np.intersect1d(train_ids, validation_ids).size:
        raise ValueError("CPU development train and validation fixtures overlap")
    _atomic_npz(
        output_path,
        train_event_id=train_ids,
        train_source_row=train_selected,
        validation_event_id=validation_ids,
        validation_source_row=validation_selected,
    )
    report = {
        "version": "realnu_cpu_development_ids_v1",
        "source_path": str(source_path),
        "source_sha256": json.loads((manifest_dir / "source_manifest_v1.json").read_text())["new_source"]["sha256"],
        "split_membership_sha256": _sha256(split_path),
        "role_support_sha256": _sha256(support_path),
        "selection_seed": seed,
        "train_count": len(train_selected),
        "validation_count": len(validation_selected),
        "train_event_id_sha256": hashlib.sha256(np.sort(train_ids).astype("<i8").tobytes()).hexdigest(),
        "validation_event_id_sha256": hashlib.sha256(np.sort(validation_ids).astype("<i8").tobytes()).hexdigest(),
        "coverage_representatives": {
            "train": categories_by_split[0],
            "validation": categories_by_split[1],
        },
        "reserved_test_rows_read": 0,
        "output_path": str(output_path),
        "output_sha256": _sha256(output_path),
    }
    _atomic_json(output_path.with_suffix(".json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-events", type=int, default=512)
    parser.add_argument("--validation-events", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args()
    result = build(
        args.source,
        args.manifest_dir,
        args.output,
        train_count=args.train_events,
        validation_count=args.validation_events,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
