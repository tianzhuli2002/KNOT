"""Run bounded real-data and synthetic edge checks for the charge-side path."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from joint_event_reconstruction.cli.run import (
    _validate_checkpoint_data_contract,
    build_data,
    load_visible_config,
)
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.partonic import PartonicScaler
from joint_event_reconstruction.models.partonic import parent_pair_observables

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/data"))
from adapt_partonic_ttbar2l_v1 import measured_event_arrays

EXPERIMENTS = ROOT / "configs/experiments/realnu_v1"
REALNU_ROOT = ROOT / "results/experiments/partonic_ttbar2L_2M_realnu_v1"
SOURCE = Path(
    "/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/datasets/ttbar2L/"
    "combined_partonic_afterFSR_v1/raw/"
    "TtbarDilepton_TopRegression_partonic_truth_nu_2M_even.h5"
)


def _overrides(max_jets: int, *, workers: int, batch_size: int) -> list[str]:
    if max_jets == 8:
        adapter = REALNU_ROOT / "adapted/development"
        stem = "partonic_ttbar2l_development_maxjets8_v1"
    else:
        adapter = REALNU_ROOT / "adapted/development"
        stem = "partonic_ttbar2l_development_maxjets20_v1"
    measured = adapter / f"{stem}_measured.h5"
    supervision = adapter / f"{stem}_supervision.h5"
    values = [
        f"data.input={measured}",
        f"data.inference_input={measured}",
        f"data.visible_truth={supervision}",
        f"data.invisible_truth={supervision}",
        "data.cache=null",
        "data.events.train=7",
        "data.events.validation=5",
        "data.events.test=null",
        f"inputs.JET.count={max_jets}",
        f"training.batch_size={batch_size}",
        f"training.workers={workers}",
        "data.loader.pin_memory=false",
        "data.loader.persistent_workers=false",
        "data.loader.prefetch_factor=2",
    ]
    return values


def _measured_adapter_edges() -> dict:
    dtype = np.dtype(
        [
            ("e", "<f4"), ("eta", "<f4"), ("phi", "<f4"),
            ("pt", "<f4"), ("charge", "<f4"), ("btag", "<f4"),
            ("cos_phi", "<f4"), ("sin_phi", "<f4"),
        ]
    )
    electrons = np.full((1, 2), np.nan, dtype=dtype)
    muons = np.full((1, 2), np.nan, dtype=dtype)
    electrons["pt"][0, 0] = 52.0
    electrons["eta"][0, 0] = 0.3
    electrons["phi"][0, 0] = np.pi - 1.0e-7
    electrons["e"][0, 0] = 55.0
    electrons["charge"][0, 0] = 1.0
    electrons["btag"][0, 0] = 0.0
    muons["pt"][0, 0] = 47.0
    muons["eta"][0, 0] = -0.4
    muons["phi"][0, 0] = -np.pi + 1.0e-7
    muons["e"][0, 0] = 51.0
    muons["charge"][0, 0] = -1.0
    muons["btag"][0, 0] = 0.0
    global_dtype = np.dtype([("met_met", "<f4"), ("met_phi", "<f4")])
    global_rows = np.zeros(1, dtype=global_dtype)
    global_rows["met_met"] = 0.0
    global_rows["met_phi"] = -np.pi + 1.0e-7

    leptons, met = measured_event_arrays(electrons, muons, global_rows)
    if leptons["charge"].tolist() != [[1.0, -1.0]]:
        raise AssertionError("Positive/negative charge-side order changed")
    if leptons["source_type"].tolist() != [[1.0, 2.0]] or leptons["source_index"].tolist() != [[0.0, 0.0]]:
        raise AssertionError("Charge-side adapter lost source flavour/index identities")
    if float(met["pt"][0, 0]) != 0.0 or not np.isfinite(met["cos_phi"]).all() or not np.isfinite(met["sin_phi"]).all():
        raise AssertionError("Zero-MET or wrapped-phi adapter handling failed")

    # Exercise the opposite e/mu charge ordering as well; role order remains charge-defined.
    electrons["charge"][0, 0] = -1.0
    muons["charge"][0, 0] = 1.0
    reversed_leptons, _ = measured_event_arrays(electrons, muons, global_rows)
    if reversed_leptons["source_type"].tolist() != [[2.0, 1.0]]:
        raise AssertionError("Opposite e/mu charge ordering did not preserve source identity")
    return {
        "status": "passed",
        "zero_met": True,
        "phi_near_pm_pi": True,
        "charge_orderings": [[1, 2], [2, 1]],
        "original_source_indices_preserved": True,
    }


def _train_loader_edges() -> dict:
    config = load_experiment(EXPERIMENTS / "q1.public.yaml", _overrides(8, workers=2, batch_size=4))
    visible = load_visible_config(config)
    data = build_data(config, visible, device="cpu")
    data.setup("fit")
    item_a = data.train_dataset[1]
    item_b = data.train_dataset[0]
    requested = data.train_dataset.__getitems__([1, 0])
    batch = data.structured_collator(requested)
    expected = [int(item_a["event_index"]), int(item_b["event_index"])]
    if batch["event_index"].tolist() != expected:
        raise AssertionError("Batched __getitems__ did not preserve requested event order")
    loader = DataLoader(
        data.train_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        collate_fn=data.structured_collator,
    )
    observed = []
    batch_sizes = []
    for row in loader:
        observed.extend(int(value) for value in row["event_index"].tolist())
        batch_sizes.append(len(row["event_index"]))
    if len(observed) != 7 or len(set(observed)) != 7 or batch_sizes != [4, 3]:
        raise AssertionError(f"Shuffled/multiworker short-batch contract failed: {batch_sizes}")
    single = data.structured_collator([data.train_dataset[0]])
    if single["tokens"].shape[0] != 1:
        raise AssertionError("Batch-size-one collation failed")

    # Alter only the in-memory checkpoint metadata to confirm split mismatch rejection.
    bad_checkpoint = {
        "hyper_parameters": {
            "split_identity": json.loads(json.dumps(data.split_metadata)),
        }
    }
    bad_checkpoint["hyper_parameters"]["split_identity"]["event_id_hashes"]["train"] = "wrong"
    try:
        _validate_checkpoint_data_contract(bad_checkpoint, config, data)
    except ValueError as exc:
        if "event/split identity" not in str(exc):
            raise
    else:
        raise AssertionError("Checkpoint with a different frozen split was accepted")
    return {
        "status": "passed",
        "batched_order": expected,
        "multiworker_batch_sizes": batch_sizes,
        "batch_size_one": True,
        "checkpoint_split_mismatch_rejected": True,
    }


def _jet20_edge() -> dict:
    config = load_experiment(EXPERIMENTS / "q1.public.yaml", _overrides(20, workers=0, batch_size=2))
    visible = load_visible_config(config)
    data = build_data(config, visible, device="cpu")
    data.setup("fit")
    batch = next(iter(data.train_dataloader()))
    if batch["type_ids"].shape[1] < 23:
        raise AssertionError(f"20-jet event has too few tokens: {batch['type_ids'].shape}")
    if not torch.isfinite(batch["physics_vectors"]).all():
        raise AssertionError("20-jet physical-vector batch contains non-finite values")
    for role in ("top", "antitop"):
        pool = batch["structured"]["pools"][role]
        if pool["candidate_mask"].shape[1] > 100_000:
            raise AssertionError("20-jet candidate state count exceeded configured limit")
    return {
        "status": "passed",
        "event_count": len(batch["event_index"]),
        "token_width": int(batch["type_ids"].shape[1]),
        "candidate_counts": {
            role: int(batch["structured"]["pools"][role]["candidate_mask"].shape[1])
            for role in ("top", "antitop")
        },
        "padding_vectors_are_zero": bool(
            torch.equal(
                batch["physics_vectors"][batch["padding_mask"]],
                torch.zeros_like(batch["physics_vectors"][batch["padding_mask"]]),
            )
        ),
    }


def _outside_jet_limit_and_ambiguity() -> dict:
    membership_path = REALNU_ROOT / "split_membership_v1.npz"
    with np.load(membership_path, allow_pickle=False) as membership:
        rows = membership["source_row"].astype(np.int64, copy=False)
        event_ids = membership["event_id"].astype(np.int64, copy=False)
        split = membership["split"].astype(np.uint8, copy=False)
    training_rows = np.flatnonzero(split < 2)
    selected_row = selected_event_id = None
    with h5py.File(SOURCE, "r") as source:
        for offset in range(0, len(training_rows), 8192):
            chosen = training_rows[offset : offset + 8192]
            labels = np.asarray(source["/LABELS/JET"][rows[chosen]])
            complete = ((labels == 1.0).sum(axis=1) == 1) & ((labels == 2.0).sum(axis=1) == 1)
            retained = ((labels[:, :8] == 1.0).sum(axis=1) == 1) & ((labels[:, :8] == 2.0).sum(axis=1) == 1)
            outside = np.flatnonzero(complete & ~retained)
            if len(outside):
                position = int(chosen[int(outside[0])])
                selected_row = int(rows[position])
                selected_event_id = int(event_ids[position])
                break
        if selected_row is None:
            raise AssertionError("No train/validation event with a truth b jet beyond jet limit 8 was found")
        raw_jets = np.asarray(source["/LABELS/JET"][selected_row])[:8].astype(np.float32)
        raw_input_jets = source["/INPUTS/JET"][selected_row]
        electrons = source["/INPUTS/ELECTRON"][selected_row]
        muons = source["/INPUTS/MUON"][selected_row]
        electron_labels = np.asarray(source["/LABELS/ELECTRON"][selected_row])
        muon_labels = np.asarray(source["/LABELS/MUON"][selected_row])

    input_config = load_experiment(
        EXPERIMENTS / "q1.public.yaml", _overrides(8, workers=0, batch_size=2)
    )
    visible = load_visible_config(input_config)
    data = build_data(input_config, visible, device="cpu")
    data.setup("fit")
    collator = data.structured_collator
    item = copy.deepcopy(data.train_dataset[0])
    item["event_index"] = torch.tensor(selected_event_id, dtype=torch.long)
    jet_positions = (item["type_ids"] == collator.source_type_ids["JET"]).nonzero().flatten()
    if len(jet_positions) != 8:
        raise AssertionError("Fixture token layout does not contain exactly eight retained jets")
    present = np.isfinite(raw_input_jets["pt"]) & (raw_input_jets["pt"] > 0.0)
    item["padding_mask"][jet_positions] = torch.from_numpy(~present[:8])

    objects = np.concatenate((electrons.reshape(1, -1), muons.reshape(1, -1)), axis=1)[0]
    labels = np.concatenate((electron_labels.reshape(-1), muon_labels.reshape(-1)))
    physical = np.isfinite(objects["pt"]) & (objects["pt"] > 0.0)
    charge = objects["charge"]
    if int(physical.sum()) != 2 or int((physical & (charge > 0)).sum()) != 1:
        raise AssertionError("Out-of-limit source fixture has invalid measured charge-side leptons")
    lepton_labels = np.asarray(
        [labels[np.flatnonzero(physical & (charge > 0))[0]], labels[np.flatnonzero(physical & (charge < 0))[0]]],
        dtype=np.float32,
    )

    class OneEventTruth:
        def __init__(self) -> None:
            self.event_ids = np.asarray([selected_event_id], dtype=np.int64)
            self.labels = {
                "JET": raw_jets.reshape(1, -1),
                "LEPTON": lepton_labels.reshape(1, 2),
            }
            self.support = np.asarray([False], dtype=np.bool_)

        def lookup(self, requested):
            values = requested.detach().cpu().numpy() if torch.is_tensor(requested) else np.asarray(requested)
            if not np.array_equal(values, self.event_ids):
                raise KeyError(f"Fixture truth does not contain requested event IDs {values.tolist()}")
            return (
                {name: torch.from_numpy(value.copy()) for name, value in self.labels.items()},
                torch.from_numpy(self.support.copy()),
            )

    original_store = collator.visible_store
    original_invisible_store = collator.invisible_store
    collator.visible_store = OneEventTruth()
    collator.invisible_store = None
    outside_batch = collator([item])
    collator.visible_store = original_store
    collator.invisible_store = original_invisible_store
    outside_roles = {
        role: {
            "reconstructable": bool(outside_batch["structured"]["pools"][role]["reconstructable"][0]),
            "truth_index": int(outside_batch["structured"]["pools"][role]["truth_index"][0]),
        }
        for role in ("top", "antitop")
    }
    missing_role = "top" if not np.any(raw_jets == 1.0) else "antitop"
    retained_role = "antitop" if missing_role == "top" else "top"
    if (
        outside_roles[missing_role]["reconstructable"]
        or outside_roles[missing_role]["truth_index"] != -1
        or not outside_roles[retained_role]["reconstructable"]
        or outside_roles[retained_role]["truth_index"] < 0
        or bool(outside_batch["visible_assignment_support"][0])
    ):
        raise AssertionError(
            "Out-of-limit b truth did not produce the expected role-specific NULL and "
            f"event support state: missing={missing_role}, roles={outside_roles}"
        )

    # A duplicate top-b label must fail as ambiguous rather than selecting one candidate.
    event_id = int(data.train_dataset[0]["event_index"])
    store = collator.visible_store
    store_position = int(np.searchsorted(store.event_ids, event_id))
    original_labels = store.labels["JET"][store_position].copy()
    top_b = np.flatnonzero(original_labels == 1.0)
    antitop_b = np.flatnonzero(original_labels == 2.0)
    if len(top_b) != 1 or len(antitop_b) != 1:
        raise AssertionError("Ambiguity fixture source event lacks one truth b jet per role")
    store.labels["JET"][store_position, antitop_b[0]] = 1.0
    try:
        collator([data.train_dataset[0]])
    except RuntimeError as exc:
        if "multiple truth candidates" not in str(exc):
            raise
    else:
        raise AssertionError("Duplicate b-jet truth labels were silently accepted")
    finally:
        store.labels["JET"][store_position] = original_labels
    return {
        "status": "passed",
        "outside_jet8_event_id": selected_event_id,
        "outside_jet8_missing_role": missing_role,
        "outside_jet8_truth_index_by_role": outside_roles,
        "duplicate_truth_labels_rejected": True,
        "reserved_test_rows_read": 0,
    }


def _parent_extremes() -> dict:
    spatial = torch.tensor(
        [
            [[3.0, 1.0, 2.0], [-2.0, 4.0, -3.0]],
            [[30.0, -80.0, 40.0], [-20.0, 70.0, -10.0]],
            [[100.0, -20.0, 1.0e7], [-90.0, 50.0, -9.0e6]],
        ],
        dtype=torch.float64,
    )
    mass = torch.tensor([[1.0e-4, 2.0e-3], [172.5, 171.8], [1.0e5, 9.0e4]], dtype=torch.float64)
    energy = torch.sqrt(spatial.square().sum(dim=-1) + mass.square())
    truth = torch.cat((energy.unsqueeze(-1), spatial), dim=-1)
    support = torch.ones((3, 2), dtype=torch.bool)
    scaler = PartonicScaler.fit(truth, support, ("top", "antitop"))
    reconstructed = scaler.inverse(scaler.transform(truth))
    if not torch.isfinite(reconstructed).all() or (reconstructed[..., 0] < 0).any():
        raise AssertionError("Extreme parent transform produced non-physical four-vectors")
    pair = parent_pair_observables(reconstructed)
    if not all(torch.isfinite(value).all() for value in pair.values()):
        raise AssertionError("Extreme parent pair observables are not finite")
    if not torch.equal(pair["ttbar"], reconstructed.sum(dim=-2)):
        raise AssertionError("Extreme parent pair is not the exact parent four-vector sum")
    return {
        "status": "passed",
        "mass_scales_GeV": [1.0e-4, 172.5, 1.0e5],
        "max_abs_roundtrip_GeV": float((reconstructed - truth).abs().max()),
        "large_pz_GeV": 1.0e7,
        "derived_observables_finite": True,
        "pair_is_exact_sum": True,
    }


def _cache_mismatch() -> dict:
    cache_path = REALNU_ROOT / "cache/development_measured_maxjets8_v1.pt"
    manifest_path = cache_path.with_suffix(".json")
    with tempfile.TemporaryDirectory(prefix="jer-realnu-stale-cache-") as temporary:
        copied_cache = Path(temporary) / cache_path.name
        copied_manifest = copied_cache.with_suffix(".json")
        shutil.copy2(cache_path, copied_cache)
        manifest = json.loads(manifest_path.read_text())
        manifest["runtime_input_h5_path"] = str(Path(temporary) / "different-source.h5")
        copied_manifest.write_text(json.dumps(manifest))
        bad_config = load_experiment(
            EXPERIMENTS / "q1.public.yaml",
            [f"data.cache={copied_cache}"],
        )
        bad_visible = load_visible_config(bad_config)
        data = build_data(bad_config, bad_visible, device="cpu")
        try:
            data.setup("fit")
        except ValueError as exc:
            if not any(
                marker in str(exc)
                for marker in ("cache manifest mismatch", "cache_path mismatch", "incompatible with the current input configuration")
            ):
                raise
        else:
            raise AssertionError("Cache from a different measured-source identity was accepted")
    return {"status": "passed", "wrong_source_identity_rejected": True}


def run() -> dict:
    torch.set_num_threads(1)
    return {
        "measured_adapter_edges": _measured_adapter_edges(),
        "loader_and_split_edges": _train_loader_edges(),
        "twenty_jet_edge": _jet20_edge(),
        "outside_jet_limit_and_ambiguity": _outside_jet_limit_and_ambiguity(),
        "parent_transform_extremes": _parent_extremes(),
        "stale_cache_identity": _cache_mismatch(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REALNU_ROOT / "cpu_validation/edge_cases.json",
    )
    args = parser.parse_args()
    result = run()
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
