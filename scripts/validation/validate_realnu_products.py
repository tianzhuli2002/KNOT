"""Validate the train/validation-only real-neutrino products in bounded reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.data.invisible import (
    InvisibleScaler,
    InvisibleTruthStore,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "results/experiments/partonic_ttbar2L_2M_realnu_v1"
SOURCE = Path(
    "/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/datasets/ttbar2L/"
    "combined_partonic_afterFSR_v1/raw/"
    "TtbarDilepton_TopRegression_partonic_truth_nu_2M_even.h5"
)
STEM = "partonic_ttbar2l_train_validation_maxjets8_v1"


def _sha256_bytes(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_record_equal(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if actual.dtype.names != expected.dtype.names:
        raise AssertionError(f"{label} field names differ: {actual.dtype.names} != {expected.dtype.names}")
    for name in actual.dtype.names or ():
        if not np.array_equal(actual[name], expected[name], equal_nan=True):
            raise AssertionError(f"{label}.{name} differs")


def _wrong_neutrino_role_order_rejected(truth_path: Path) -> bool:
    from joint_event_reconstruction.data.invisible import InvisibleTruthStore

    with tempfile.TemporaryDirectory(prefix="jer-realnu-wrong-role-order-") as temporary:
        fixture = Path(temporary) / "wrong_role_order.h5"
        with h5py.File(truth_path, "r") as source, h5py.File(fixture, "w") as target:
            target.attrs.update(dict(source.attrs))
            target.attrs["invisible_roles"] = json.dumps(["antitop.nu", "top.nu"])
            target.attrs["invisible_role_order"] = json.dumps(["antitop.nu", "top.nu"])
            target.create_dataset("EVENT_ID", data=source["EVENT_ID"][:1])
            labels = target.create_group("LABELS")
            neutrinos = labels.create_dataset(
                "NEUTRINO", data=source["LABELS/NEUTRINO"][:1]
            )
            neutrinos.attrs.update(dict(source["LABELS/NEUTRINO"].attrs))
            neutrinos.attrs["role_order"] = json.dumps(["antitop.nu", "top.nu"])
            support_group = target.create_group("SUPPORT")
            support = support_group.create_dataset(
                "INVISIBLE_ROLE", data=source["SUPPORT/INVISIBLE_ROLE"][:1]
            )
            support.attrs.update(dict(source["SUPPORT/INVISIBLE_ROLE"].attrs))
            support.attrs["role_order"] = json.dumps(["antitop.nu", "top.nu"])
        try:
            InvisibleTruthStore(
                fixture, ("top.nu", "antitop.nu"), allow_synthetic=False
            )
        except ValueError:
            return True
        raise AssertionError("InvisibleTruthStore accepted reversed top/antitop role metadata")


def validate(
    output_path: Path,
    *,
    experiment_root: Path = EXPERIMENT_ROOT,
    measured_path: Path | None = None,
    truth_path: Path | None = None,
    cache_path: Path | None = None,
    scaler_output_path: Path | None = None,
) -> dict:
    experiment_root = experiment_root.expanduser().resolve()
    split_path = experiment_root / "split_membership_v1.npz"
    source_manifest_path = experiment_root / "source_manifest_v1.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest_sha = hashlib.sha256(source_manifest_path.read_bytes()).hexdigest()
    split_sha = hashlib.sha256(split_path.read_bytes()).hexdigest()
    measured_path = (
        measured_path.expanduser().resolve()
        if measured_path is not None
        else experiment_root / "adapted/train_validation" / f"{STEM}_measured.h5"
    )
    truth_path = (
        truth_path.expanduser().resolve()
        if truth_path is not None
        else experiment_root / "adapted/train_validation" / f"{STEM}_supervision.h5"
    )
    cache_path = (
        cache_path.expanduser().resolve()
        if cache_path is not None
        else experiment_root / "cache/train_validation_measured_maxjets8_v1.pt"
    )
    scaler_output_path = (
        scaler_output_path.expanduser().resolve()
        if scaler_output_path is not None
        else experiment_root / "scalers/invisible_train_only_top_antitop_20260926.json"
    )
    cache_manifest_path = cache_path.with_suffix(".json")

    with np.load(split_path, allow_pickle=False) as membership:
        source_rows = membership["source_row"].astype(np.int64, copy=False)
        expected_ids = membership["event_id"].astype(np.int64, copy=False)
        split_codes = membership["split"].astype(np.uint8, copy=False)
    keep = split_codes < 2
    expected_rows = source_rows[keep]
    expected_event_ids = expected_ids[keep]
    expected_splits = split_codes[keep]
    reserved_ids = expected_ids[split_codes == 2]
    if np.any(np.isin(expected_event_ids, reserved_ids)):
        raise AssertionError("Train/validation product overlaps reserved-test event identities")
    if not np.isin(expected_splits, (0, 1)).all():
        raise AssertionError("Selected population contains a split other than train/validation")

    with h5py.File(measured_path, "r") as measured, h5py.File(truth_path, "r") as truth:
        count = len(expected_rows)
        if count != 1_900_095:
            raise AssertionError(f"Unexpected train/validation event count: {count}")
        measured_ids = np.asarray(measured["EVENT_ID"][:], dtype=np.int64)
        measured_split = np.asarray(measured["SPLIT"][:], dtype=np.uint8)
        truth_ids = np.asarray(truth["EVENT_ID"][:], dtype=np.int64)
        if not np.array_equal(measured_ids, expected_event_ids):
            raise AssertionError("Measured adapter EVENT_ID order differs from frozen membership")
        if not np.array_equal(truth_ids, expected_event_ids):
            raise AssertionError("Truth sidecar EVENT_ID order differs from frozen membership")
        if not np.array_equal(measured_split, expected_splits):
            raise AssertionError("Measured adapter SPLIT values differ from frozen membership")
        if "LABELS" in measured or "TARGETS" in measured:
            raise AssertionError("Inference measured input contains LABELS or TARGETS")
        for name in (
            "INPUTS/JET", "INPUTS/LEPTON", "INPUTS/MET", "INPUTS/GLOBAL",
            "META/eventNumber", "META/source_entry_index",
        ):
            if len(measured[name]) != count:
                raise AssertionError(f"Measured product event axis mismatch at {name}")
        if not np.array_equal(np.asarray(measured["META/eventNumber"][:]), measured_ids):
            raise AssertionError("META/eventNumber is not aligned with EVENT_ID")
        if int(measured.attrs["max_jets"]) != 8:
            raise AssertionError("Measured product does not declare the selected eight-jet limit")
        for name in ("source_manifest_sha256", "split_membership_sha256"):
            expected = source_manifest_sha if name == "source_manifest_sha256" else split_sha
            if str(measured.attrs[name]) != expected or str(truth.attrs[name]) != expected:
                raise AssertionError(f"Product {name} does not match its frozen manifest")
        if not bool(measured.attrs["measured_inputs_only"]):
            raise AssertionError("Measured adapter is missing measured_inputs_only=true")

        visible_support = np.asarray(truth["SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"][:], dtype=np.bool_)
        invisible_support = np.asarray(truth["SUPPORT/INVISIBLE_ROLE"][:], dtype=np.bool_)
        parent_support = np.asarray(truth["PARTONIC/SUPPORT"][:], dtype=np.bool_)
        if invisible_support.shape != (count, 2) or parent_support.shape != (count, 2):
            raise AssertionError("Task role support arrays do not align with the selected population")
        if int(visible_support.sum()) != 1_899_839:
            raise AssertionError("Eight-jet visible assignment support count changed")
        if not np.all(invisible_support):
            raise AssertionError("Finite authoritative top/antitop targets lost neutrino support")
        if int(invisible_support.all(axis=1).sum()) != count:
            raise AssertionError("Corrected joint neutrino support must cover the frozen train/validation population")
        if (
            str(truth.attrs.get("invisible_role_convention", "")) != "top_antitop"
            or json.loads(str(truth.attrs.get("invisible_role_order", "[]")))
            != ["top.nu", "antitop.nu"]
            or str(truth.attrs.get("neutrino_truth_definition", ""))
            != "generator top-side neutrino and generator antitop-side neutrino"
            or str(truth.attrs.get("units", "")) != "GeV"
            or bool(truth.attrs.get("synthetic", True))
        ):
            raise AssertionError("Corrected supervision metadata does not declare the real top/antitop contract")
        neutrino_metadata = truth["LABELS/NEUTRINO"].attrs
        if (
            str(neutrino_metadata.get("invisible_role_convention", "")) != "top_antitop"
            or json.loads(str(neutrino_metadata.get("role_order", "[]")))
            != ["top.nu", "antitop.nu"]
            or str(neutrino_metadata.get("units", "")) != "GeV"
            or str(neutrino_metadata.get("truth_definition", ""))
            != "generator top-side neutrino and generator antitop-side neutrino"
        ):
            raise AssertionError("Neutrino target dataset metadata is incomplete")
        if not parent_support.all():
            raise AssertionError("Valid partonic top/antitop targets unexpectedly lost support")

        rng = np.random.default_rng(20260926)
        sample_positions = set(rng.choice(count, size=8, replace=False).astype(int).tolist())
        conflict_samples = source_manifest["truth_contract"].get(
            "e_mu_old_flavour_charge_mapping_conflict_sample_train_validation", []
        )
        if not conflict_samples:
            raise AssertionError("No train/validation legacy mismatch sample was retained as provenance")
        conflict_row = int(conflict_samples[0]["source_row"])
        conflict_position = int(np.searchsorted(expected_rows, conflict_row))
        if (
            conflict_position >= len(expected_rows)
            or int(expected_rows[conflict_position]) != conflict_row
        ):
            raise AssertionError("Legacy flavor/charge mismatch sample is outside frozen train/validation")
        sample_positions.add(conflict_position)
        sample_positions = np.asarray(sorted(sample_positions), dtype=np.int64)
        sample_rows = expected_rows[sample_positions]
        sample_record = {}
        with h5py.File(SOURCE, "r") as source:
            for local, (position, source_row) in enumerate(zip(sample_positions, sample_rows, strict=True)):
                _assert_record_equal(
                    measured["INPUTS/JET"][position],
                    source["INPUTS/JET"][source_row, :8],
                    f"raw-vs-adapter jet row {source_row}",
                )
                _assert_record_equal(
                    measured["INPUTS/GLOBAL"][position],
                    source["INPUTS/GLOBAL"][source_row],
                    f"raw-vs-adapter global row {source_row}",
                )
                event_id = int(measured_ids[position])
                if int(source["META/eventNumber"][source_row]) != event_id:
                    raise AssertionError(f"Raw event identity differs at source row {source_row}")
                if int(measured["META/source_entry_index"][position]) != int(
                    source["META/source_entry_index"][source_row]
                ):
                    raise AssertionError(f"source_entry_index differs at source row {source_row}")

                electrons = source["INPUTS/ELECTRON"][source_row]
                muons = source["INPUTS/MUON"][source_row]
                objects = np.concatenate((electrons, muons))
                source_type = np.asarray((1, 1, 2, 2), dtype=np.int8)
                source_index = np.asarray((0, 1, 0, 1), dtype=np.int8)
                present = np.isfinite(objects["pt"]) & (objects["pt"] > 0.0)
                positive = np.flatnonzero(present & (objects["charge"] > 0.0))
                negative = np.flatnonzero(present & (objects["charge"] < 0.0))
                if len(positive) != 1 or len(negative) != 1:
                    raise AssertionError(f"Raw source lepton charge contract failed at event {event_id}")
                order = [int(positive[0]), int(negative[0])]
                adapted = measured["INPUTS/LEPTON"][position]
                for slot, object_index in enumerate(order):
                    for field in ("e", "eta", "phi", "pt", "btag", "charge"):
                        if not np.isclose(adapted[field][slot], objects[field][object_index], atol=0.0, rtol=0.0):
                            raise AssertionError(f"Charge-side lepton field {field} differs at event {event_id}")
                    if int(adapted["source_type"][slot]) != int(source_type[object_index]) or int(adapted["source_index"][slot]) != int(source_index[object_index]):
                        raise AssertionError(f"Charge-side lepton source identity differs at event {event_id}")

                raw_jet_labels = np.asarray(source["LABELS/JET"][source_row, :8], dtype=np.float32)
                if not np.array_equal(truth["LABELS/JET"][position], raw_jet_labels, equal_nan=True):
                    raise AssertionError(f"Visible jet labels are misaligned at event {event_id}")
                raw_e_labels = np.asarray(source["LABELS/ELECTRON"][source_row]).reshape(-1)
                raw_mu_labels = np.asarray(source["LABELS/MUON"][source_row]).reshape(-1)
                raw_lepton_labels = np.concatenate((raw_e_labels, raw_mu_labels))
                if not np.array_equal(truth["LABELS/LEPTON"][position], raw_lepton_labels[order]):
                    raise AssertionError(f"Charge-side lepton labels are misaligned at event {event_id}")

                raw_parent = source["TARGETS/TOP"][source_row]
                expected_parent = []
                for role in ("top", "antitop"):
                    pt = float(raw_parent[f"{role}_pt"])
                    eta = float(raw_parent[f"{role}_eta"])
                    phi = float(raw_parent[f"{role}_phi"])
                    mass = float(raw_parent[f"{role}_m"])
                    expected_parent.append(
                        np.asarray(
                            [
                                np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2),
                                pt * np.cos(phi),
                                pt * np.sin(phi),
                                pt * np.sinh(eta),
                            ],
                            dtype=np.float32,
                        )
                    )
                observed_parent = np.stack(
                    [truth[f"PARTONIC/{role}"][position] for role in ("top", "antitop")]
                )
                if not np.allclose(observed_parent, expected_parent, atol=1.0e-3, rtol=0.0):
                    raise AssertionError(f"Partonic parent truth differs from TARGETS/TOP at event {event_id}")
                support = invisible_support[position]
                if not support.all():
                    raise AssertionError(f"Authoritative real-neutrino roles lost support at event {event_id}")
                raw_nu = source["TARGETS/NEUTRINO"][source_row]
                components = ("px", "py", "pz")
                for role_index in range(2):
                    observed = np.asarray(
                        [truth["LABELS/NEUTRINO"][position][role_index][name] for name in components],
                        dtype=np.float32,
                    )
                    expected = np.asarray(
                        [raw_nu[role_index][name] for name in components], dtype=np.float32
                    )
                    if not np.array_equal(observed, expected):
                        raise AssertionError(
                            f"Authoritative neutrino role {role_index} differs from raw slot at event {event_id}"
                        )
                sample_record[str(local)] = {
                    "event_id": event_id,
                    "source_row": int(source_row),
                    "legacy_flavour_charge_conflict": bool(source_row == conflict_row),
                }

    train_event_ids = expected_event_ids[expected_splits == 0]
    prior_product_path = EXPERIMENT_ROOT / "cpu_validation/full_train_validation_products.json"
    prior_product = json.loads(prior_product_path.read_text())
    if (
        prior_product.get("source_sha256") != source_manifest["new_source"]["sha256"]
        or prior_product.get("split_membership_sha256") != split_sha
        or prior_product.get("train_only_scalers", {}).get("train_event_id_sha256")
        != _sha256_bytes(np.sort(train_event_ids))
    ):
        raise AssertionError("Existing parent scaler provenance does not match the unchanged source and frozen train split")
    invisible_store = InvisibleTruthStore(
        truth_path, ("top.nu", "antitop.nu"), allow_synthetic=False
    )
    train_neutrinos, train_neutrino_support = invisible_store.lookup_with_support(train_event_ids)
    invisible_scaler = InvisibleScaler.fit(
        train_neutrinos,
        ("top.nu", "antitop.nu"),
        support_mask=train_neutrino_support,
    )
    scaler_report = {
        "population": "frozen_train_split_only",
        "train_event_count": len(train_event_ids),
        "train_event_id_sha256": _sha256_bytes(np.sort(train_event_ids)),
        "partonic": prior_product["train_only_scalers"]["partonic"],
        "partonic_support_by_role": prior_product["train_only_scalers"]["partonic_support_by_role"],
        "partonic_scaler_reused_from": str(prior_product_path.resolve()),
        "invisible": invisible_scaler.as_dict(),
        "invisible_support_by_role": train_neutrino_support.sum(dim=0).tolist(),
        "invisible_truth_source_sha256": source_manifest["new_source"]["sha256"],
        "invisible_truth_manifest_sha256": source_manifest_sha,
        "invisible_truth_split_membership_sha256": split_sha,
        "invisible_truth_support_sha256": source_manifest["split"]["neutrino_role_support_sha256"],
        "invisible_truth_store_sha256": _sha256_file(truth_path),
        "invisible_role_convention": "top_antitop",
        "invisible_role_order": ["top.nu", "antitop.nu"],
    }
    scaler_output_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_output_path.write_text(
        json.dumps(scaler_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    cache_manifest = json.loads(cache_manifest_path.read_text())
    if (
        cache_manifest.get("cache_version") != 9
        or cache_manifest.get("n_events") != len(expected_event_ids)
        or cache_manifest.get("source_identity", {}).get("path") != str(measured_path.resolve())
        or cache_manifest.get("physics_vectors_shape") != [len(expected_event_ids), 12, 4]
        or not cache_manifest.get("source_identity", {}).get("event_id_fingerprint")
    ):
        raise AssertionError("Measured tensor cache has incompatible source, event, or physical-vector metadata")

    result = {
        "status": "passed",
        "source_file": str(SOURCE),
        "source_sha256": source_manifest.get("new_source", {}).get("sha256"),
        "source_manifest_sha256": source_manifest_sha,
        "split_membership_sha256": split_sha,
        "invisible_role_convention": "top_antitop",
        "invisible_role_order": ["top.nu", "antitop.nu"],
        "legacy_flavour_charge_mismatch_diagnostic": source_manifest["truth_contract"][
            "e_mu_old_flavour_charge_mapping_conflicts"
        ],
        "wrong_neutrino_role_order_rejected": _wrong_neutrino_role_order_rejected(truth_path),
        "population": "frozen_train_plus_validation_only",
        "event_count": len(expected_event_ids),
        "train_count": int((expected_splits == 0).sum()),
        "validation_count": int((expected_splits == 1).sum()),
        "reserved_test_rows_materialised": 0,
        "support": {
            "visible_assignment": int(visible_support.sum()),
            "neutrino_joint": int(invisible_support.all(axis=1).sum()),
            "neutrino_unsupported": int((~invisible_support.all(axis=1)).sum()),
            "partonic_joint": int(parent_support.all(axis=1).sum()),
            "all_three": int((visible_support & invisible_support.all(axis=1) & parent_support.all(axis=1)).sum()),
        },
        "bounded_raw_adapter_samples": sample_record,
        "train_only_scalers": scaler_report,
        "invisible_scaler_metadata_path": str(scaler_output_path),
        "invisible_scaler_metadata_sha256": _sha256_file(scaler_output_path),
        "truth_store_identity": {
            "path": str(truth_path),
            "size_bytes": int(truth_path.stat().st_size),
            "sha256": _sha256_file(truth_path),
            "source_file_sha256": source_manifest["new_source"]["sha256"],
            "source_manifest_sha256": source_manifest_sha,
            "split_membership_sha256": split_sha,
            "support_sha256": source_manifest["split"]["neutrino_role_support_sha256"],
        },
        "tensor_cache": {
            "path": str(cache_path),
            "bytes": int(cache_path.stat().st_size),
            "cache_version": int(cache_manifest["cache_version"]),
            "n_events": int(cache_manifest["n_events"]),
            "physics_vectors_shape": cache_manifest["physics_vectors_shape"],
            "source_identity": cache_manifest["source_identity"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "cpu_validation/full_train_validation_products.json",
    )
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--measured", type=Path)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--scaler-output", type=Path)
    args = parser.parse_args()
    result = validate(
        args.output.expanduser().resolve(),
        experiment_root=args.experiment_root,
        measured_path=args.measured,
        truth_path=args.truth,
        cache_path=args.cache,
        scaler_output_path=args.scaler_output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
