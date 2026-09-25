"""Write bounded canonical measured inputs and separate truth sidecars.

The measured-input path reads only INPUTS and META.  Assignment labels and
partonic after-FSR targets are read by a separate pass into supervision.h5.
training_view.h5 uses HDF5 external links, so the two payloads are joined by
event order without copying detector arrays or exposing truth to prediction.
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


SOURCE = Path(
    "/net/scratch/w00238tl/datasets/ttbar2L/combined_partonic_afterFSR_v1/raw/"
    "TtbarDilepton_2M_even.h5"
)
EXPERIMENT_DIR = Path("results/experiments/partonic_ttbar2L_2M_v1")
SPLIT_NAME = "event_number_splitmix64_90_5_5_v1"
TRUTH_BINDINGS = {
    "top": "partonic_after_fsr_top",
    "antitop": "partonic_after_fsr_antitop",
}

LEPTON_DTYPE = np.dtype(
    [
        ("e", "<f4"), ("eta", "<f4"), ("phi", "<f4"),
        ("cos_phi", "<f4"), ("sin_phi", "<f4"), ("pt", "<f4"),
        ("btag", "<f4"), ("charge", "<f4"), ("id", "<f4"),
        ("is_el", "<f4"), ("is_mu", "<f4"),
        ("source_type", "<f4"), ("source_index", "<f4"),
    ]
)
MET_DTYPE = np.dtype(
    [
        ("e", "<f4"), ("eta", "<f4"), ("phi", "<f4"),
        ("cos_phi", "<f4"), ("sin_phi", "<f4"), ("pt", "<f4"),
        ("btag", "<f4"), ("charge", "<f4"), ("id", "<f4"),
    ]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_h5(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as output:
            writer(output)
            output.flush()
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _selection(manifest_dir: Path, population: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    membership_path = manifest_dir / "split_membership_v1.npz"
    with np.load(membership_path, allow_pickle=False) as split:
        rows = split["source_row"].astype(np.int64, copy=False)
        event_ids = split["event_id"].astype(np.int64, copy=False)
        assignment = split["split"].astype(np.uint8, copy=False)
    if population == "development":
        path = manifest_dir / "cpu_development_ids_v1.npz"
        with np.load(path, allow_pickle=False) as dev:
            selected_rows = np.sort(
                np.concatenate((dev["train_source_row"], dev["validation_source_row"]))
            ).astype(np.int64, copy=False)
            selected_ids = np.sort(
                np.concatenate((dev["train_event_id"], dev["validation_event_id"]))
            ).astype(np.int64, copy=False)
        # Recover the frozen train/validation assignment by exact source row.
        selected_split = assignment[selected_rows]
        if not np.isin(selected_split, (0, 1)).all():
            raise ValueError("CPU development subset contains a reserved-test row")
        if not np.array_equal(np.sort(event_ids[selected_rows]), selected_ids):
            raise ValueError("CPU development event IDs do not match the frozen split membership")
        return selected_rows, event_ids[selected_rows], selected_split
    desired = {"train_validation": (0, 1), "test": (2,)}[population]
    selected_mask = np.isin(assignment, desired)
    selected_rows = rows[selected_mask]
    return selected_rows, event_ids[selected_mask], assignment[selected_mask]


def _empty_records(dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    result = np.empty(shape, dtype=dtype)
    for name in dtype.names or ():
        result[name] = np.nan
    return result


def measured_event_arrays(electrons, muons, global_rows) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalise only measured objects, retaining charge-side source identity."""
    count = len(global_rows)
    leptons = _empty_records(LEPTON_DTYPE, (count, 2))
    met = _empty_records(MET_DTYPE, (count, 1))
    for event in range(count):
        objects = []
        for source_type, source_name, rows in (
            (1, "ELECTRON", electrons[event]),
            (2, "MUON", muons[event]),
        ):
            for source_index, row in enumerate(rows):
                present = (
                    np.isfinite(row["pt"])
                    and row["pt"] > 0.0
                    and np.isfinite(row["eta"])
                    and np.isfinite(row["phi"])
                    and np.isfinite(row["e"])
                )
                if present:
                    charge = float(row["charge"])
                    if not np.isfinite(charge) or charge == 0.0:
                        raise ValueError(
                            f"Invalid measured lepton charge: source={source_name}, "
                            f"event_in_chunk={event}, source_index={source_index}"
                        )
                    objects.append((charge, source_type, source_name, source_index, row))
        if len(objects) != 2 or sum(item[0] > 0 for item in objects) != 1 or sum(item[0] < 0 for item in objects) != 1:
            raise ValueError(
                f"Expected one positive and one negative measured lepton; "
                f"event_in_chunk={event}, found={len(objects)}"
            )
        ordered = sorted(objects, key=lambda item: 0 if item[0] > 0 else 1)
        for slot, (charge, source_type, _source_name, source_index, row) in enumerate(ordered):
            for field in ("e", "eta", "phi", "cos_phi", "sin_phi", "pt", "btag"):
                if field in (row.dtype.names or ()) and np.isfinite(row[field]):
                    leptons[field][event, slot] = row[field]
            if not np.isfinite(leptons["cos_phi"][event, slot]):
                leptons["cos_phi"][event, slot] = np.cos(float(row["phi"]))
            if not np.isfinite(leptons["sin_phi"][event, slot]):
                leptons["sin_phi"][event, slot] = np.sin(float(row["phi"]))
            leptons["charge"][event, slot] = charge
            leptons["id"][event, slot] = 2.0 if source_type == 1 else 3.0
            leptons["is_el"][event, slot] = float(source_type == 1)
            leptons["is_mu"][event, slot] = float(source_type == 2)
            leptons["source_type"][event, slot] = source_type
            leptons["source_index"][event, slot] = source_index

        global_row = global_rows[event]
        met_pt = float(global_row["met_met"])
        met_phi = float(global_row["met_phi"])
        if not np.isfinite(met_pt) or met_pt < 0.0 or not np.isfinite(met_phi):
            raise ValueError(f"Invalid measured MET magnitude/direction; event_in_chunk={event}")
        met["e"][event, 0] = met_pt
        met["pt"][event, 0] = met_pt
        met["eta"][event, 0] = 0.0
        met["phi"][event, 0] = met_phi
        met["cos_phi"][event, 0] = np.cos(met_phi)
        met["sin_phi"][event, 0] = np.sin(met_phi)
        met["btag"][event, 0] = 0.0
        met["charge"][event, 0] = 0.0
        met["id"][event, 0] = 4.0
    return leptons, met


def _prepare_measured(source_path, output_path, rows, event_ids, splits, *, max_jets, chunk_size, source_manifest_hash, split_hash):
    with h5py.File(source_path, "r") as source:
        count = len(rows)
        source_ids = np.asarray(source["/META/eventNumber"][rows], dtype=np.int64)
        if not np.array_equal(source_ids, event_ids):
            raise ValueError("Selected eventNumber values do not match the frozen manifest")
        jet_width = min(int(max_jets), int(source["/INPUTS/JET"].shape[1]))
        if jet_width <= 0:
            raise ValueError("max_jets must be positive")

        def writer(output):
            inputs = output.create_group("INPUTS")
            meta = output.create_group("META")
            datasets = {
                "JET": inputs.create_dataset(
                    "JET", (count, jet_width), dtype=source["/INPUTS/JET"].dtype,
                    chunks=(min(chunk_size, max(count, 1)), jet_width), compression="lzf",
                ),
                "LEPTON": inputs.create_dataset(
                    "LEPTON", (count, 2), dtype=LEPTON_DTYPE,
                    chunks=(min(chunk_size, max(count, 1)), 2), compression="lzf",
                ),
                "MET": inputs.create_dataset(
                    "MET", (count, 1), dtype=MET_DTYPE,
                    chunks=(min(chunk_size, max(count, 1)), 1), compression="lzf",
                ),
                "GLOBAL": inputs.create_dataset(
                    "GLOBAL", (count,), dtype=source["/INPUTS/GLOBAL"].dtype,
                    chunks=(min(chunk_size, max(count, 1)),), compression="lzf",
                ),
            }
            event_out = output.create_dataset("EVENT_ID", (count,), dtype="<i8")
            split_out = output.create_dataset("SPLIT", (count,), dtype="u1")
            event_number = meta.create_dataset("eventNumber", (count,), dtype="<i8")
            source_entry = meta.create_dataset("source_entry_index", (count,), dtype="<i8")
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                selected_rows = rows[start:stop]
                selected_rows = np.asarray(selected_rows, dtype=np.int64)
                jets = source["/INPUTS/JET"][selected_rows, :jet_width]
                electrons = source["/INPUTS/ELECTRON"][selected_rows]
                muons = source["/INPUTS/MUON"][selected_rows]
                globals_chunk = source["/INPUTS/GLOBAL"][selected_rows]
                leptons, met = measured_event_arrays(electrons, muons, globals_chunk)
                datasets["JET"][start:stop] = jets
                datasets["LEPTON"][start:stop] = leptons
                datasets["MET"][start:stop] = met
                datasets["GLOBAL"][start:stop] = globals_chunk
                event_out[start:stop] = event_ids[start:stop]
                event_number[start:stop] = event_ids[start:stop]
                source_entry[start:stop] = source["/META/source_entry_index"][selected_rows]
                split_out[start:stop] = splits[start:stop]
            output.attrs["adapter_version"] = "partonic_ttbar2l_measured_v1"
            output.attrs["source_path"] = str(source_path.resolve())
            output.attrs["source_event_count"] = int(source["/META/eventNumber"].shape[0])
            output.attrs["selected_event_count"] = count
            output.attrs["max_jets"] = jet_width
            output.attrs["event_identity"] = "META/eventNumber"
            output.attrs["secondary_identity"] = "META/source_entry_index"
            output.attrs["split_version"] = SPLIT_NAME
            output.attrs["split_membership_sha256"] = split_hash
            output.attrs["source_manifest_sha256"] = source_manifest_hash
            output.attrs["measured_inputs_only"] = True
            output.attrs["lepton_slot_order"] = "positive charge (top), negative charge (antitop)"
            output.attrs["lepton_source_identity_fields"] = "source_type,source_index,id/is_el/is_mu"
            output.attrs["met_convention"] = "transverse pseudovector: E=MET,px=MET*cos(phi),py=MET*sin(phi),pz=0; not a neutrino p4"
            output.attrs["kinematic_units"] = "GeV; (E,px,py,pz)"
            inputs["GLOBAL"].attrs["source_contract"] = "copied measured global fields; no engineered observables"

        _atomic_h5(output_path, writer)


def _parent_four_vectors(top_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = len(top_rows)
    values = np.zeros((count, 2, 4), dtype=np.float32)
    support = np.zeros((count, 2), dtype=np.bool_)
    for role_index, prefix in enumerate(("top", "antitop")):
        pt = np.asarray(top_rows[f"{prefix}_pt"], dtype=np.float64)
        eta = np.asarray(top_rows[f"{prefix}_eta"], dtype=np.float64)
        phi = np.asarray(top_rows[f"{prefix}_phi"], dtype=np.float64)
        mass = np.asarray(top_rows[f"{prefix}_m"], dtype=np.float64)
        valid = (
            np.isfinite(np.stack((pt, eta, phi, mass), axis=-1)).all(axis=-1)
            & (pt >= 0.0) & (mass >= 0.0)
        )
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt((pt * np.cosh(eta)) ** 2 + mass**2)
        candidate = np.stack((energy, px, py, pz), axis=-1)
        valid &= np.isfinite(candidate).all(axis=-1)
        values[valid, role_index] = candidate[valid].astype(np.float32)
        support[:, role_index] = valid
    return values, support


def _prepare_supervision(source_path, output_path, measured_path, rows, event_ids, *, max_jets, chunk_size, source_manifest_hash, split_hash):
    with h5py.File(source_path, "r") as source, h5py.File(measured_path, "r") as measured:
        count = len(rows)
        jet_width = int(measured.attrs["max_jets"])

        def writer(output):
            labels = output.create_group("LABELS")
            partonic = output.create_group("PARTONIC")
            support_group = output.create_group("SUPPORT")
            event_dataset = output.create_dataset("EVENT_ID", (count,), dtype="<i8")
            jet_labels = labels.create_dataset(
                "JET", (count, jet_width), dtype="<f4",
                chunks=(min(chunk_size, max(count, 1)), jet_width), compression="lzf",
            )
            lepton_labels = labels.create_dataset(
                "LEPTON", (count, 2), dtype="<f4",
                chunks=(min(chunk_size, max(count, 1)), 2), compression="lzf",
            )
            charge_side = support_group.create_dataset(
                "LEPTON_CHARGE_SIDE", (count, 2), dtype="u1"
            )
            assignment_support = support_group.create_dataset(
                "VISIBLE_ASSIGNMENT_RECONSTRUCTABLE", (count,), dtype=np.bool_
            )
            invisible_support = support_group.create_dataset(
                "INVISIBLE_ROLE", (count, 2), dtype=np.bool_, fillvalue=False
            )
            invisible_support.attrs["role_order"] = json.dumps(
                ["top.nu", "antitop.nu"]
            )
            invisible_support.attrs["definition"] = (
                "false for every event: source slots are electron/muon flavour ordered, "
                "but charge-side neutrino association is unverified"
            )
            target_datasets = {
                role: partonic.create_dataset(
                    role, (count, 4), dtype="<f4",
                    chunks=(min(chunk_size, max(count, 1)), 4), compression="lzf",
                )
                for role in ("top", "antitop")
            }
            parent_support = partonic.create_dataset(
                "SUPPORT", (count, 2), dtype=np.bool_
            )
            for start in range(0, count, chunk_size):
                stop = min(start + chunk_size, count)
                selected_rows = np.asarray(rows[start:stop], dtype=np.int64)
                event_chunk = np.asarray(event_ids[start:stop], dtype=np.int64)
                e_input = source["/INPUTS/ELECTRON"][selected_rows]
                m_input = source["/INPUTS/MUON"][selected_rows]
                e_labels = source["/LABELS/ELECTRON"][selected_rows]
                m_labels = source["/LABELS/MUON"][selected_rows]
                all_jet_labels = source["/LABELS/JET"][selected_rows]
                top_rows = source["/TARGETS/TOP"][selected_rows]
                adapted_leptons = np.empty((stop - start, 2), dtype=np.float32)
                adapted_charge = np.empty((stop - start, 2), dtype=np.uint8)
                reconstructable = np.zeros(stop - start, dtype=np.bool_)
                for event in range(stop - start):
                    charge_slots = []
                    label_slots = []
                    for collection, source_labels, source_name in (
                        (e_input[event], e_labels[event], "ELECTRON"),
                        (m_input[event], m_labels[event], "MUON"),
                    ):
                        for slot, row in enumerate(collection):
                            if np.isfinite(row["pt"]) and row["pt"] > 0.0:
                                charge = float(row["charge"])
                                if not np.isfinite(charge) or charge == 0.0:
                                    raise ValueError(
                                        f"Invalid lepton charge in truth alignment: file={source_path}, "
                                        f"event_id={event_chunk[event]}, source={source_name}, slot={slot}"
                                    )
                                charge_slots.append(charge)
                                label_slots.append(float(source_labels[slot]))
                    order = np.argsort(np.asarray(charge_slots) < 0.0, kind="stable")
                    if len(order) != 2 or not np.array_equal(order, np.asarray([0, 1])) and not np.array_equal(order, np.asarray([1, 0])):
                        raise ValueError(f"Cannot establish charge-side order for event_id={event_chunk[event]}")
                    ordered_labels = np.asarray(label_slots, dtype=np.float64)[order]
                    ordered_charges = np.asarray(charge_slots, dtype=np.float64)[order]
                    charge_expected = np.asarray([1.0, 2.0])
                    if not np.array_equal(ordered_labels, charge_expected):
                        raise ValueError(
                            "Source matched lepton labels disagree with charge-side mapping: "
                            f"file={source_path}, event_id={event_chunk[event]}, "
                            f"labels={ordered_labels.tolist()}, charges={ordered_charges.tolist()}"
                        )
                    adapted_leptons[event, :] = ordered_labels.astype(np.float32)
                    adapted_charge[event, :] = charge_expected.astype(np.uint8)
                    selected_jets = all_jet_labels[event, :jet_width]
                    reconstructable[event] = (
                        np.count_nonzero(selected_jets == 1.0) == 1
                        and np.count_nonzero(selected_jets == 2.0) == 1
                    )
                physical, target_support = _parent_four_vectors(top_rows)
                target_datasets["top"][start:stop] = physical[:, 0]
                target_datasets["antitop"][start:stop] = physical[:, 1]
                parent_support[start:stop] = target_support
                event_dataset[start:stop] = event_chunk
                jet_labels[start:stop] = all_jet_labels[:, :jet_width]
                lepton_labels[start:stop] = adapted_leptons
                charge_side[start:stop] = adapted_charge
                assignment_support[start:stop] = reconstructable
            labels["JET"].attrs["convention"] = "source truth labels; 1=top b, 2=antitop bbar, 0=background, NaN=padding"
            labels["LEPTON"].attrs["convention"] = "source matched labels reordered by measured charge; validated 1=positive/top side, 2=negative/antitop side"
            support_group["LEPTON_CHARGE_SIDE"].attrs["convention"] = "charge-derived labels, 1=positive/top side, 2=negative/antitop side; distinct from LABELS/LEPTON source truth"
            support_group["VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"].attrs["definition"] = "both source-labelled b jets occur within configured max_jets; false is represented by maintained NULL assignment objective"
            for target in target_datasets.values():
                target.attrs["physical_order"] = "E,px,py,pz"
                target.attrs["units"] = "GeV"
                target.attrs["truth_definition"] = "partonic_after_fsr"
                target.attrs["feature_transform"] = "[px,py,pz,log1p(m)] then training-only per-role component standardisation"
                target.attrs["inverse"] = "m=expm1(raw_log_mass); E=sqrt(px^2+py^2+pz^2+m^2)"
            output.attrs["partonic_roles"] = json.dumps(["top", "antitop"])
            output.attrs["partonic_truth_bindings"] = json.dumps(TRUTH_BINDINGS, sort_keys=True)
            output.attrs["partonic_truth_definition"] = "partonic_after_fsr"
            output.attrs["partonic_units"] = "GeV"
            output.attrs["partonic_target_representation"] = "partonic_px_py_pz_log1p_mass_v1"
            output.attrs["partonic_feature_transform"] = "[px,py,pz,log1p(m)] then training-only per-role component standardisation"
            output.attrs["partonic_inverse_transform"] = "m=expm1(raw_log_mass), E=sqrt(px^2+py^2+pz^2+m^2)"
            output.attrs["sidecar_version"] = "partonic_ttbar2l_supervision_v1"
            output.attrs["source_path"] = str(source_path.resolve())
            output.attrs["selected_event_count"] = count
            output.attrs["max_jets"] = jet_width
            output.attrs["split_version"] = SPLIT_NAME
            output.attrs["split_membership_sha256"] = split_hash
            output.attrs["source_manifest_sha256"] = source_manifest_hash
            output.attrs["neutrino_supervision"] = "disabled: flavour order is known, neutrino charge/top side and same-flavour pairing are not verified"
            output.attrs["lepton_truth_contract"] = "charge-side labels and source matched labels are stored separately; they were required to agree in this adapter"
            output.attrs["synthetic"] = False

        _atomic_h5(output_path, writer)


def _write_training_view(path: Path, measured_path: Path, supervision_path: Path) -> None:
    def writer(output):
        with h5py.File(measured_path, "r") as measured:
            for key, value in measured.attrs.items():
                output.attrs[key] = value
        output["INPUTS"] = h5py.ExternalLink(str(measured_path.resolve()), "/INPUTS")
        output["EVENT_ID"] = h5py.ExternalLink(str(measured_path.resolve()), "/EVENT_ID")
        output["SPLIT"] = h5py.ExternalLink(str(measured_path.resolve()), "/SPLIT")
        output["META"] = h5py.ExternalLink(str(measured_path.resolve()), "/META")
        output["LABELS"] = h5py.ExternalLink(str(supervision_path.resolve()), "/LABELS")
        output.attrs["training_view_version"] = "partonic_ttbar2l_external_join_v1"
        output.attrs["measured_event_file"] = str(measured_path.resolve())
        output.attrs["visible_truth_sidecar"] = str(supervision_path.resolve())
        output.attrs["join_key"] = "EVENT_ID"

    _atomic_h5(path, writer)


def _support_summary(visible, invisible_role, partonic, split_values) -> dict:
    visible = np.asarray(visible, dtype=bool)
    invisible_role = np.asarray(invisible_role, dtype=bool)
    partonic = np.asarray(partonic, dtype=bool)
    split_values = np.asarray(split_values, dtype=np.uint8)
    if (
        visible.ndim != 1
        or invisible_role.ndim != 2
        or partonic.ndim != 2
        or len(visible) != len(invisible_role)
        or len(visible) != len(partonic)
        or split_values.shape != visible.shape
    ):
        raise ValueError("Objective-support arrays do not align with the selected event population")
    neutrino = invisible_role.all(axis=1)
    parent = partonic.all(axis=1)

    def counts(mask: np.ndarray) -> dict:
        count = int(mask.sum())
        return {"count": count, "fraction": float(count / max(len(mask), 1))}

    def summarize(rows: np.ndarray) -> dict:
        assignment_rows = visible[rows]
        neutrino_rows = neutrino[rows]
        parent_rows = parent[rows]
        return {
            "events": int(rows.sum()),
            "visible_assignment": counts(assignment_rows),
            "neutrino_cfm_joint": counts(neutrino_rows),
            "partonic_cfm_joint": counts(parent_rows),
            "partonic_role": [counts(partonic[rows, role]) for role in range(partonic.shape[1])],
            "neutrino_role": [counts(invisible_role[rows, role]) for role in range(invisible_role.shape[1])],
            "intersections": {
                "visible_and_neutrino": counts(assignment_rows & neutrino_rows),
                "visible_and_partonic": counts(assignment_rows & parent_rows),
                "neutrino_and_partonic": counts(neutrino_rows & parent_rows),
                "all_three": counts(assignment_rows & neutrino_rows & parent_rows),
            },
        }

    result = {"all_selected": summarize(np.ones(len(visible), dtype=bool))}
    for split_id, split_name in ((0, "train"), (1, "validation"), (2, "reserved_test")):
        selected = split_values == split_id
        if selected.any():
            result[split_name] = summarize(selected)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--manifest-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "adapted")
    parser.add_argument("--population", choices=("development", "train_validation", "test"), default="development")
    parser.add_argument("--max-jets", type=int, choices=(8, 20), default=8)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    manifest_dir = args.manifest_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if not source.is_file():
        raise FileNotFoundError(f"Raw source does not exist: {source}")
    source_manifest = manifest_dir / "source_manifest_v1.json"
    membership = manifest_dir / "split_membership_v1.npz"
    if not source_manifest.is_file() or not membership.is_file():
        raise FileNotFoundError("Source manifest and frozen split membership are required")
    source_manifest_hash = _sha256(source_manifest)
    split_hash = _sha256(membership)
    rows, event_ids, splits = _selection(manifest_dir, args.population)
    if len(np.unique(rows)) != len(rows) or np.any(np.diff(rows) <= 0):
        raise ValueError("Adapter rows must be strictly increasing and unique")
    if args.population != "test" and np.isin(splits, (2,)).any():
        raise ValueError("Reserved test rows cannot enter a training/development adapter")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"partonic_ttbar2l_{args.population}_maxjets{args.max_jets}_v1"
    measured_path = output_dir / f"{stem}_measured.h5"
    supervision_path = output_dir / f"{stem}_supervision.h5"
    training_path = output_dir / f"{stem}_training_view.h5"
    _prepare_measured(
        source, measured_path, rows, event_ids, splits,
        max_jets=args.max_jets, chunk_size=args.chunk_size,
        source_manifest_hash=source_manifest_hash, split_hash=split_hash,
    )
    _prepare_supervision(
        source, supervision_path, measured_path, rows, event_ids,
        max_jets=args.max_jets, chunk_size=args.chunk_size,
        source_manifest_hash=source_manifest_hash, split_hash=split_hash,
    )
    _write_training_view(training_path, measured_path, supervision_path)
    with h5py.File(measured_path, "r") as measured, h5py.File(supervision_path, "r") as truth:
        report = {
            "population": args.population,
            "events": len(rows),
            "split_counts": {str(int(key)): int(value) for key, value in zip(*np.unique(splits, return_counts=True), strict=True)},
            "max_jets": int(args.max_jets),
            "measured_input_path": str(measured_path),
            "training_view_path": str(training_path),
            "supervision_sidecar_path": str(supervision_path),
            "measured_input_has_labels_or_targets": any(name in measured for name in ("LABELS", "TARGETS")),
            "visible_assignment_reconstructable": int(truth["SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"][:].sum()),
            "partonic_role_support": truth["PARTONIC/SUPPORT"][:].sum(axis=0).astype(int).tolist(),
            "objective_support": _support_summary(
                truth["SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"][:],
                truth["SUPPORT/INVISIBLE_ROLE"][:],
                truth["PARTONIC/SUPPORT"][:],
                measured["SPLIT"][:],
            ),
            "neutrino_supervision": str(truth.attrs["neutrino_supervision"]),
            "source_manifest_sha256": source_manifest_hash,
            "split_membership_sha256": split_hash,
        }
    report_path = output_dir / f"{stem}_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
