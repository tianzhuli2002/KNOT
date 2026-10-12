#!/usr/bin/env python3
"""Make a no-copy JER training input from one exact measured/supervision pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np

MEASURED_DATASETS = (
    "EVENT_ID",
    "SPLIT",
    "INPUTS/GLOBAL",
    "INPUTS/JET",
    "INPUTS/LEPTON",
    "INPUTS/MET",
)
TRUTH_DATASETS = (
    "LABELS/JET",
    "LABELS/LEPTON",
    "LABELS/NEUTRINO",
)
TTBAR_GLOBAL_CLASS = np.float32(0.0)
TTBAR_GLOBAL_CLASS_CONTRACT = (
    "all rows are from the homogeneous ttbar sample; value 0 follows the maintained "
    "legacy ttbar converter. This run has no global-classification objective; the P5 "
    "loader consumes this field for split/cache bookkeeping."
)
MET_EXISTENCE_LABEL = np.float32(1.0)
MET_EXISTENCE_LABEL_CONTRACT = (
    "one observed MET token per event; label 1 follows the maintained legacy converter's "
    "existing-object convention"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def build(measured_path: Path, truth_path: Path, output: Path) -> dict:
    measured_path = measured_path.expanduser().resolve()
    truth_path = truth_path.expanduser().resolve()
    output = output.expanduser().resolve()
    for path in (measured_path, truth_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required after-FSR JER input does not exist: {path}")
    if output.exists():
        raise FileExistsError(f"Refusing to replace the JER after-FSR VDS: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary VDS path already exists: {temporary}")

    with h5py.File(measured_path, "r") as measured, h5py.File(truth_path, "r") as truth:
        missing_measured = [name for name in MEASURED_DATASETS if name not in measured]
        missing_truth = [name for name in TRUTH_DATASETS if name not in truth]
        if missing_measured or missing_truth:
            raise KeyError(
                f"Missing measured={missing_measured}, supervision={missing_truth}; "
                f"measured={measured_path}, truth={truth_path}"
            )
        measured_ids = np.asarray(measured["EVENT_ID"][:], dtype=np.int64)
        truth_ids = np.asarray(truth["EVENT_ID"][:], dtype=np.int64)
        if measured_ids.ndim != 1 or len(np.unique(measured_ids)) != len(measured_ids):
            raise ValueError(f"{measured_path}: EVENT_ID must be unique and one-dimensional")
        if not np.array_equal(measured_ids, truth_ids):
            raise ValueError(
                "Measured and after-FSR supervision EVENT_ID order differs; refusing a row join: "
                f"measured={measured_path}, truth={truth_path}"
            )
        count = len(measured_ids)
        split = np.asarray(measured["SPLIT"][:])
        if split.shape != (count,) or split.dtype.kind not in "iu" or not np.isin(split, (0, 1)).all():
            raise ValueError(f"{measured_path}: expected only frozen train=0/validation=1 SPLIT rows")
        if int((split == 0).sum()) != 1_799_753 or int((split == 1).sum()) != 100_342:
            raise ValueError(
                f"Current frozen KNOT membership changed: train={(split == 0).sum()}, "
                f"validation={(split == 1).sum()}"
            )
        if measured["INPUTS/JET"].shape != (count, 8):
            raise ValueError(f"{measured_path}: JER P5 requires exactly eight retained jet slots")
        if measured["INPUTS/LEPTON"].shape != (count, 2):
            raise ValueError(f"{measured_path}: charge-side topology requires exactly two lepton slots")
        if measured["INPUTS/MET"].shape != (count, 1):
            raise ValueError(f"{measured_path}: charge-side topology requires one MET token")
        met_dataset = measured["INPUTS/MET"]
        met_fields = set(met_dataset.dtype.names or ())
        required_met_fields = {"id", "pt", "phi"}
        if required_met_fields - met_fields:
            raise TypeError(
                f"{measured_path}:INPUTS/MET is missing fields "
                f"{sorted(required_met_fields - met_fields)}"
            )
        for start in range(0, count, 100_000):
            met = met_dataset[start : start + 100_000, 0]
            met_present = (
                np.isfinite(met["id"])
                & (met["id"] == 4)
                & np.isfinite(met["pt"])
                & np.isfinite(met["phi"])
            )
            if not met_present.all():
                bad = start + int(np.flatnonzero(~met_present)[0])
                raise ValueError(
                    f"{measured_path}:INPUTS/MET event row {bad} is not one finite MET token "
                    "with id=4; cannot derive the canonical object-existence label"
                )
        support_path = "SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"
        if support_path not in truth:
            raise KeyError(f"{truth_path} is missing explicit assignment support {support_path}")
        assignment_support = np.asarray(truth[support_path][:])
        if assignment_support.shape != (count,) or assignment_support.dtype != np.bool_:
            raise TypeError(f"{truth_path}:{support_path} must be bool [{count}]")
        neutrinos = np.asarray(truth["LABELS/NEUTRINO"][:])
        if neutrinos.dtype.names != ("px", "py", "pz"):
            raise TypeError(f"{truth_path}:LABELS/NEUTRINO fields must be exactly px,py,pz")
        nu_values = np.stack([neutrinos[name] for name in ("px", "py", "pz")], axis=-1)
        if nu_values.shape != (count, 2, 3) or not np.isfinite(nu_values).all():
            raise ValueError(f"{truth_path}: after-FSR charge-side neutrinos are missing/non-finite")
        for name in TRUTH_DATASETS[:2]:
            labels = np.asarray(truth[name][:])
            finite = labels[np.isfinite(labels)]
            if not np.equal(finite, np.rint(finite)).all() or not np.isin(finite, (0, 1, 2)).all():
                raise ValueError(f"{truth_path}:{name} has labels outside 0,1,2,NaN padding")
        measured_sha = sha256_file(measured_path)
        truth_sha = sha256_file(truth_path)
        split_hash = str(measured.attrs.get("split_membership_sha256", ""))
        if not split_hash:
            raise ValueError(f"{measured_path} lacks frozen split_membership_sha256")

        with h5py.File(temporary, "w", libver="latest") as target:
            target.attrs.update(
                {
                    "schema_version": "jer_charge_side_after_fsr_training_v3",
                    "measured_source": str(measured_path),
                    "supervision_source": str(truth_path),
                    "measured_source_sha256": measured_sha,
                    "supervision_source_sha256": truth_sha,
                    "split_membership_sha256": split_hash,
                    "event_id_ordered_sha256": sha256_ids(measured_ids),
                    "event_count": count,
                    "train_count": int((split == 0).sum()),
                    "validation_count": int((split == 1).sum()),
                    "test_count": 0,
                    "truth_contract": "charge-side direct after-FSR W-daughter neutrinos and explicit matched b-jet labels",
                    "parent_truth_contract": "external PARTONIC/top and antitop sidecar; not an input to the P5 model",
                    "assignment_support_count": int(assignment_support.sum()),
                    "assignment_unsupported_count": int((~assignment_support).sum()),
                    "lepton_order": "positive-charge top side, negative-charge antitop side",
                    "invisible_role_order": "top,antitop",
                    "sealed_test_used": False,
                }
            )
            for name in MEASURED_DATASETS:
                source = measured[name]
                layout = h5py.VirtualLayout(shape=source.shape, dtype=source.dtype)
                layout[...] = h5py.VirtualSource(str(measured_path), name, shape=source.shape)
                target.create_virtual_dataset(name, layout)
            for name in TRUTH_DATASETS:
                source = truth[name]
                layout = h5py.VirtualLayout(shape=source.shape, dtype=source.dtype)
                layout[...] = h5py.VirtualSource(str(truth_path), name, shape=source.shape)
                target.create_virtual_dataset(name, layout)
            global_class = target.create_dataset(
                "LABELS/GLOBAL",
                data=np.full(count, TTBAR_GLOBAL_CLASS, dtype="<f4"),
                dtype="<f4",
            )
            global_class.attrs["label_contract"] = TTBAR_GLOBAL_CLASS_CONTRACT
            met_label = target.create_dataset(
                "LABELS/MET",
                data=np.full((count, 1), MET_EXISTENCE_LABEL, dtype="<f4"),
                dtype="<f4",
            )
            met_label.attrs["label_contract"] = MET_EXISTENCE_LABEL_CONTRACT
            target["LABELS/NEUTRINO"].attrs["role_order"] = '["top.nu", "antitop.nu"]'
            target["LABELS/NEUTRINO"].attrs["truth_definition"] = "direct after-FSR W daughters"

    os.replace(temporary, output)
    with h5py.File(output, "r") as check, h5py.File(truth_path, "r") as truth_check:
        if not np.array_equal(check["EVENT_ID"][:], measured_ids):
            raise ValueError(f"Staged JER VDS EVENT_ID verification failed: {output}")
        if not np.array_equal(check["SPLIT"][:], split):
            raise ValueError(f"Staged JER VDS SPLIT verification failed: {output}")
        if check["LABELS/NEUTRINO"].shape != (count, 2):
            raise ValueError(f"Staged JER VDS neutrino-label verification failed: {output}")
        if check["LABELS/GLOBAL"].shape != (count,) or not np.equal(
            check["LABELS/GLOBAL"][:], TTBAR_GLOBAL_CLASS
        ).all():
            raise ValueError(f"Staged JER VDS global class-label verification failed: {output}")
        if check["LABELS/MET"].shape != (count, 1) or not np.equal(
            check["LABELS/MET"][:], MET_EXISTENCE_LABEL
        ).all():
            raise ValueError(f"Staged JER VDS MET existence-label verification failed: {output}")
        if not np.array_equal(
            check["LABELS/JET"][:], truth_check["LABELS/JET"][:], equal_nan=True
        ):
            raise ValueError(f"Staged JER VDS visible-label verification failed: {output}")
    result = {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "measured_source": str(measured_path),
        "measured_source_sha256": measured_sha,
        "supervision_source": str(truth_path),
        "supervision_source_sha256": truth_sha,
        "events": count,
        "train_events": int((split == 0).sum()),
        "validation_events": int((split == 1).sum()),
        "test_events": 0,
        "event_id_ordered_sha256": sha256_ids(measured_ids),
        "split_membership_sha256": split_hash,
        "assignment_supported_events": int(assignment_support.sum()),
        "assignment_unsupported_events": int((~assignment_support).sum()),
        "global_class_label_value": float(TTBAR_GLOBAL_CLASS),
        "global_class_label_contract": TTBAR_GLOBAL_CLASS_CONTRACT,
        "met_existence_label_value": float(MET_EXISTENCE_LABEL),
        "met_existence_label_contract": MET_EXISTENCE_LABEL_CONTRACT,
        "truth_order": "top,antitop charge-side after-FSR",
        "join": "exact equal EVENT_ID arrays; no row-order assumption",
    }
    manifest = output.with_suffix(output.suffix + ".json")
    manifest.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.measured, args.truth, args.output), indent=2))


if __name__ == "__main__":
    main()
