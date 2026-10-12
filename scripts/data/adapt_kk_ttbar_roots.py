#!/usr/bin/env python3
"""Write measured KNOT inputs and a separate after-FSR truth sidecar from KK ROOT files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot

JET_WIDTH = 8
MEV_TO_GEV = 1e-3
KNOWN_DSIDS = {574801, 601230}
JET_DTYPE = np.dtype(
    [
        ("e", "<f4"), ("eta", "<f4"), ("phi", "<f4"),
        ("cos_phi", "<f4"), ("sin_phi", "<f4"), ("pt", "<f4"),
        ("btag", "<f4"), ("charge", "<f4"), ("id", "<f4"),
    ]
)
LEPTON_DTYPE = np.dtype(
    JET_DTYPE.descr
    + [("is_el", "<f4"), ("is_mu", "<f4"),
       ("source_type", "<f4"), ("source_index", "<f4")]
)
MET_DTYPE = JET_DTYPE
GLOBAL_DTYPE = np.dtype(
    [(name, "<f4") for name in (
        "njet", "nelectron", "nmuon", "met_met", "met_phi", "met_cosphi",
        "met_sinphi", "nBJets_90", "nBJets_85", "nBJets_77", "nBJets_70",
        "nBJets_65",
    )]
)

WP_BRANCHES = {
    "nBJets_90": "jet_select_GN2v01_FixedCutBEff_90",
    "nBJets_85": "jet_select_GN2v01_FixedCutBEff_85",
    "nBJets_77": "jet_select_GN2v01_FixedCutBEff_77",
    "nBJets_70": "jet_select_GN2v01_FixedCutBEff_70",
    "nBJets_65": "jet_select_GN2v01_FixedCutBEff_65",
}
WEIGHT_BRANCHES = (
    "weight_mc_NOSYS",
    "weight_pileup_NOSYS",
    "weight_leptonSF_tight_NOSYS",
    "weight_jvt_effSF_NOSYS",
    "weight_ftag_effSF_GN2v01_Continuous_NOSYS",
    "weight_max_NOSYS",
    "TtbarDiLepton_NW_weight_max_NOSYS",
)
JET_MATCH_BRANCHES = (
    "jet_dRmatched_to_Ttbar_MC_b_afterFSR_from_t",
    "jet_dRmatched_to_Ttbar_MC_bbar_afterFSR_from_tbar",
    "n_jet_matched_to_Ttbar_MC_b_afterFSR_from_t",
    "n_jet_matched_to_Ttbar_MC_bbar_afterFSR_from_tbar",
    "Ttbar_MC_b_afterFSR_from_t_pdgId",
    "Ttbar_MC_bbar_afterFSR_from_tbar_pdgId",
)
JET_MATCH_CORE_BRANCHES = JET_MATCH_BRANCHES[:4]
JET_MATCH_PDG_BRANCHES = JET_MATCH_BRANCHES[4:]
REQUIRED_BRANCHES = {
    "eventNumber", "runNumber", "mcChannelNumber",
    "el_pt_NOSYS", "el_eta", "el_phi", "el_e_NOSYS", "el_charge",
    "mu_pt_NOSYS", "mu_eta", "mu_phi", "mu_e_NOSYS", "mu_charge",
    "jet_pt_NOSYS", "jet_eta", "jet_phi", "jet_e_NOSYS",
    "jet_GN2v01_Continuous_quantile", "jet_select_baselineJvt_NOSYS",
    "met_met_NOSYS", "met_phi_NOSYS",
    "Ttbar_MC_t_afterFSR_pt", "Ttbar_MC_t_afterFSR_eta",
    "Ttbar_MC_t_afterFSR_phi", "Ttbar_MC_t_afterFSR_m",
    "Ttbar_MC_tbar_afterFSR_pt", "Ttbar_MC_tbar_afterFSR_eta",
    "Ttbar_MC_tbar_afterFSR_phi", "Ttbar_MC_tbar_afterFSR_m",
    "Ttbar_MC_Wdecay1_afterFSR_from_t_pdgId",
    "Ttbar_MC_Wdecay1_afterFSR_from_tbar_pdgId",
    "Ttbar_MC_Wdecay2_afterFSR_from_t_pdgId",
    "Ttbar_MC_Wdecay2_afterFSR_from_tbar_pdgId",
    "Ttbar_MC_Wdecay2_afterFSR_from_t_pt",
    "Ttbar_MC_Wdecay2_afterFSR_from_t_eta",
    "Ttbar_MC_Wdecay2_afterFSR_from_t_phi",
    "Ttbar_MC_Wdecay2_afterFSR_from_tbar_pt",
    "Ttbar_MC_Wdecay2_afterFSR_from_tbar_eta",
    "Ttbar_MC_Wdecay2_afterFSR_from_tbar_phi",
    "weight_mc_NOSYS",
    *WP_BRANCHES.values(),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(arrays, name: str, dtype=None) -> np.ndarray:
    values = np.asarray(arrays[name])
    if values.ndim != 1:
        raise ValueError(f"Expected scalar ROOT branch {name}, found shape {values.shape}")
    return values.astype(dtype, copy=False) if dtype is not None else values


def vector_rows(arrays, name: str) -> list[list]:
    values = arrays[name]
    if values.ndim != 2:
        raise ValueError(f"Expected one jagged ROOT vector per event in {name}")
    return ak.to_list(values)


def p4_from_pt_eta_phi_m(pt_gev, eta, phi, mass_gev) -> np.ndarray:
    px = pt_gev * np.cos(phi)
    py = pt_gev * np.sin(phi)
    pz = pt_gev * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass_gev * mass_gev)
    return np.stack((energy, px, py, pz), axis=-1)


def root_metadata(root_file, dsid: int) -> tuple[list[str], dict]:
    labels = [str(value) for value in root_file["metadata"].axis().labels()]
    if len(labels) < 4 or labels[2] != str(dsid):
        raise ValueError(f"ROOT metadata labels do not identify DSID {dsid}: {labels}")
    matches = [
        key for key in root_file.keys(cycle=False)
        if key.startswith(f"CutBookkeeper_{dsid}_") and key.endswith("_NOSYS")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one DSID {dsid} CutBookkeeper, found {matches}")
    histogram = root_file[matches[0]]
    return labels, {
        "key": matches[0],
        "labels": [str(value) for value in histogram.axis().labels()],
        "values": [float(value) for value in histogram.values()],
    }


def output_leptons(selected_rows: np.ndarray) -> np.ndarray:
    out = np.full((len(selected_rows), 2), np.nan, dtype=LEPTON_DTYPE)
    for output_row, source_row in enumerate(selected_rows):
        objects = []
        for flavor, prefix, source_type, object_id in (
            ("electron", "el", 1, 2), ("muon", "mu", 2, 3),
        ):
            values = {}
            for field, branch in (
                ("pt", f"{prefix}_pt_NOSYS"),
                ("eta", f"{prefix}_eta"),
                ("phi", f"{prefix}_phi"),
                ("e", f"{prefix}_e_NOSYS"),
                ("charge", f"{prefix}_charge"),
            ):
                row = vector_rows_cache[branch][source_row]
                if len(row) != 1:
                    raise ValueError(
                        f"Selected ROOT entry {source_row} must have exactly one {prefix} in {branch}, "
                        f"found {len(row)}"
                    )
                values[field] = float(row[0])
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"Selected ROOT entry {source_row} has non-finite {prefix} kinematics")
            if values["pt"] <= 0 or values["e"] <= 0:
                raise ValueError(f"Selected ROOT entry {source_row} has nonphysical {prefix} pT or energy")
            objects.append(
                {
                    "pt": values["pt"] * MEV_TO_GEV,
                    "eta": values["eta"],
                    "phi": values["phi"],
                    "e": values["e"] * MEV_TO_GEV,
                    "charge": values["charge"],
                    "source_type": source_type,
                    "id": object_id,
                    "is_el": float(flavor == "electron"),
                    "is_mu": float(flavor == "muon"),
                }
            )
        if any(abs(abs(float(obj["charge"])) - 1.0) > 1e-6 for obj in objects):
            raise ValueError(f"Selected event row {source_row} has lepton charge other than +/-1")
        if sum(float(obj["charge"]) > 0 for obj in objects) != 1 or sum(
            float(obj["charge"]) < 0 for obj in objects
        ) != 1:
            raise ValueError(f"Selected event row {source_row} lacks one positive and one negative lepton")
        objects.sort(key=lambda item: -float(item["charge"]))
        for slot, obj in enumerate(objects):
            for name in ("e", "eta", "phi", "pt", "charge", "id", "is_el", "is_mu", "source_type"):
                out[name][output_row, slot] = obj[name]
            out["source_index"][output_row, slot] = 0.0
            out["btag"][output_row, slot] = 0.0
            out["cos_phi"][output_row, slot] = np.cos(obj["phi"])
            out["sin_phi"][output_row, slot] = np.sin(obj["phi"])
    return out


# Filled only while adapting one input file; keeps the per-event object mapping
# readable without duplicating the ROOT branch conversion logic.
vector_rows_cache: dict[str, list[list]] = {}


def build_measured(arrays, selected_rows: np.ndarray) -> dict[str, np.ndarray]:
    global vector_rows_cache
    vector_names = (
        "el_pt_NOSYS", "el_eta", "el_phi", "el_e_NOSYS", "el_charge",
        "mu_pt_NOSYS", "mu_eta", "mu_phi", "mu_e_NOSYS", "mu_charge",
        "jet_pt_NOSYS", "jet_eta", "jet_phi", "jet_e_NOSYS",
        "jet_GN2v01_Continuous_quantile", "jet_select_baselineJvt_NOSYS",
        *WP_BRANCHES.values(),
    )
    vector_rows_cache = {name: vector_rows(arrays, name) for name in vector_names}
    count = len(selected_rows)
    jets = np.full((count, JET_WIDTH), np.nan, dtype=JET_DTYPE)
    source_indices = np.full((count, JET_WIDTH), -1, dtype=np.int16)
    leptons = output_leptons(selected_rows)
    met = np.zeros((count, 1), dtype=MET_DTYPE)
    global_values = np.zeros(count, dtype=GLOBAL_DTYPE)
    wp_counts = {name: np.zeros(count, dtype=np.float32) for name in WP_BRANCHES}

    met_pt = scalar(arrays, "met_met_NOSYS", np.float64)[selected_rows] * MEV_TO_GEV
    met_phi = scalar(arrays, "met_phi_NOSYS", np.float64)[selected_rows]
    for row, source_row in enumerate(selected_rows):
        pt = np.asarray(vector_rows_cache["jet_pt_NOSYS"][source_row], dtype=np.float64) * MEV_TO_GEV
        eta = np.asarray(vector_rows_cache["jet_eta"][source_row], dtype=np.float64)
        phi = np.asarray(vector_rows_cache["jet_phi"][source_row], dtype=np.float64)
        energy = np.asarray(vector_rows_cache["jet_e_NOSYS"][source_row], dtype=np.float64) * MEV_TO_GEV
        quantile = np.asarray(vector_rows_cache["jet_GN2v01_Continuous_quantile"][source_row])
        baseline_jvt = np.asarray(vector_rows_cache["jet_select_baselineJvt_NOSYS"][source_row])
        if not (len(pt) == len(eta) == len(phi) == len(energy) == len(quantile) == len(baseline_jvt)):
            raise ValueError(f"Jet branches have different lengths at ROOT entry {source_row}")
        if not np.isin(baseline_jvt, (0, 1)).all() or not np.all(baseline_jvt == 1):
            raise ValueError(
                f"Selected ROOT entry {source_row} contains a jet failing baseline JVT; "
                "the audited KK input contract assumes all reco jets pass"
            )
        if len(pt) < 2:
            raise ValueError(f"Selected eμ event has fewer than two jets at ROOT entry {source_row}")
        if not (
            np.isfinite(pt).all() and np.isfinite(eta).all() and np.isfinite(phi).all()
            and np.isfinite(energy).all() and (pt > 0).all() and (energy > 0).all()
        ):
            raise ValueError(f"Non-finite or nonphysical jet input at ROOT entry {source_row}")
        if not np.equal(quantile, np.rint(quantile)).all() or not np.isin(quantile, np.arange(1, 7)).all():
            raise ValueError(f"GN2 continuous quantile must be an integer in [1,6] at ROOT entry {source_row}")
        order = np.argsort(-pt, kind="stable")[:JET_WIDTH]
        for slot, index in enumerate(order):
            source_indices[row, slot] = int(index)
            jets["e"][row, slot] = energy[index]
            jets["eta"][row, slot] = eta[index]
            jets["phi"][row, slot] = phi[index]
            jets["cos_phi"][row, slot] = np.cos(phi[index])
            jets["sin_phi"][row, slot] = np.sin(phi[index])
            jets["pt"][row, slot] = pt[index]
            jets["btag"][row, slot] = quantile[index]
            jets["charge"][row, slot] = 0.0
            jets["id"][row, slot] = 1.0
        for name, branch in WP_BRANCHES.items():
            flags = np.asarray(vector_rows_cache[branch][source_row])
            if len(flags) != len(pt) or not np.isin(flags, (0, 1)).all():
                raise ValueError(f"Malformed {branch} flags at ROOT entry {source_row}")
            wp_counts[name][row] = float(flags.sum())
        global_values["njet"][row] = len(pt)
        global_values["nelectron"][row] = 1.0
        global_values["nmuon"][row] = 1.0
        global_values["met_met"][row] = met_pt[row]
        global_values["met_phi"][row] = met_phi[row]
        global_values["met_cosphi"][row] = np.cos(met_phi[row])
        global_values["met_sinphi"][row] = np.sin(met_phi[row])
        for name in WP_BRANCHES:
            global_values[name][row] = wp_counts[name][row]

    if not np.isfinite(met_pt).all() or (met_pt < 0).any() or not np.isfinite(met_phi).all():
        raise ValueError("MET branch contains a non-finite or negative selected-event value")
    met["e"][:, 0] = met_pt
    met["pt"][:, 0] = met_pt
    met["eta"][:, 0] = 0.0
    met["phi"][:, 0] = met_phi
    met["cos_phi"][:, 0] = np.cos(met_phi)
    met["sin_phi"][:, 0] = np.sin(met_phi)
    met["btag"][:, 0] = 0.0
    met["charge"][:, 0] = 0.0
    met["id"][:, 0] = 4.0
    return {
        "JET": jets,
        "LEPTON": leptons,
        "MET": met,
        "GLOBAL": global_values,
        "jet_source_indices": source_indices,
    }


def build_visible_assignment(arrays, selected_rows, measured) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map the unique ROOT reco-to-b matches into charge-side sorted jet slots."""
    labels = np.full((len(selected_rows), JET_WIDTH), np.nan, dtype=np.float32)
    support = np.zeros(len(selected_rows), dtype=np.bool_)
    top_index = scalar(arrays, JET_MATCH_BRANCHES[0], np.int64)[selected_rows]
    antitop_index = scalar(arrays, JET_MATCH_BRANCHES[1], np.int64)[selected_rows]
    top_count = scalar(arrays, JET_MATCH_BRANCHES[2], np.int64)[selected_rows]
    antitop_count = scalar(arrays, JET_MATCH_BRANCHES[3], np.int64)[selected_rows]
    if set(JET_MATCH_PDG_BRANCHES).issubset(arrays.fields):
        top_pdg = scalar(arrays, JET_MATCH_PDG_BRANCHES[0], np.int32)[selected_rows]
        antitop_pdg = scalar(arrays, JET_MATCH_PDG_BRANCHES[1], np.int32)[selected_rows]
    else:
        top_pdg = antitop_pdg = None
    reasons = {
        "unique_charge_side_matches_in_retained_jets": 0,
        "no_unique_match_for_one_or_both_b_partons": 0,
        "same_reconstructed_jet_matches_b_and_bbar": 0,
        "one_or_both_truth_b_jets_outside_max_jets_8": 0,
    }
    source_indices = measured["jet_source_indices"]
    njet = measured["GLOBAL"]["njet"].astype(np.int64)
    for row in range(len(selected_rows)):
        if top_pdg is not None and (top_pdg[row] != 5 or antitop_pdg[row] != -5):
            raise ValueError(
                "After-FSR b/bbar PDG roles are inconsistent with top/antitop: "
                f"ROOT entry={int(selected_rows[row])}, pdg=({int(top_pdg[row])},{int(antitop_pdg[row])})"
            )
        for count, index, side in (
            (top_count[row], top_index[row], "top b"),
            (antitop_count[row], antitop_index[row], "antitop bbar"),
        ):
            if count < 0 or (count == 0 and index != -1) or (
                count > 0 and not 0 <= index < njet[row]
            ):
                raise ValueError(
                    f"Inconsistent ROOT {side} match count/index at entry {int(selected_rows[row])}: "
                    f"count={int(count)}, index={int(index)}, njet={int(njet[row])}"
                )
        real_width = min(int(njet[row]), JET_WIDTH)
        labels[row, :real_width] = 0.0
        if top_count[row] != 1 or antitop_count[row] != 1:
            reasons["no_unique_match_for_one_or_both_b_partons"] += 1
            continue
        if top_index[row] == antitop_index[row]:
            reasons["same_reconstructed_jet_matches_b_and_bbar"] += 1
            continue
        top_slots = np.flatnonzero(source_indices[row, :real_width] == top_index[row])
        antitop_slots = np.flatnonzero(source_indices[row, :real_width] == antitop_index[row])
        if len(top_slots) != 1 or len(antitop_slots) != 1:
            reasons["one_or_both_truth_b_jets_outside_max_jets_8"] += 1
            continue
        labels[row, int(top_slots[0])] = 1.0
        labels[row, int(antitop_slots[0])] = 2.0
        support[row] = True
        reasons["unique_charge_side_matches_in_retained_jets"] += 1
    if int(support.sum()) != reasons["unique_charge_side_matches_in_retained_jets"]:
        raise RuntimeError("Visible assignment support counter did not close")
    return labels, support, reasons


def build_truth(arrays, selected_rows: np.ndarray) -> dict[str, np.ndarray]:
    roles = ("t", "tbar")
    parents = np.empty((len(selected_rows), 2, 4), dtype=np.float32)
    neutrinos = np.empty((len(selected_rows), 2, 3), dtype=np.float32)
    daughter_ids = np.empty((len(selected_rows), 4), dtype=np.int32)
    for side_index, side in enumerate(roles):
        prefix = f"Ttbar_MC_{side}"
        pt = scalar(arrays, f"{prefix}_afterFSR_pt", np.float64)[selected_rows] * MEV_TO_GEV
        eta = scalar(arrays, f"{prefix}_afterFSR_eta", np.float64)[selected_rows]
        phi = scalar(arrays, f"{prefix}_afterFSR_phi", np.float64)[selected_rows]
        mass = scalar(arrays, f"{prefix}_afterFSR_m", np.float64)[selected_rows] * MEV_TO_GEV
        if not all(np.isfinite(value).all() for value in (pt, eta, phi, mass)) or (pt < 0).any() or (mass < 0).any():
            raise ValueError(f"Invalid after-FSR parent components for role {side}")
        parents[:, side_index] = p4_from_pt_eta_phi_m(pt, eta, phi, mass).astype(np.float32)
        nu_pt = scalar(arrays, f"Ttbar_MC_Wdecay2_afterFSR_from_{side}_pt", np.float64)[selected_rows] * MEV_TO_GEV
        nu_eta = scalar(arrays, f"Ttbar_MC_Wdecay2_afterFSR_from_{side}_eta", np.float64)[selected_rows]
        nu_phi = scalar(arrays, f"Ttbar_MC_Wdecay2_afterFSR_from_{side}_phi", np.float64)[selected_rows]
        if not (
            np.isfinite(nu_pt).all() and np.isfinite(nu_eta).all() and np.isfinite(nu_phi).all()
        ) or (nu_pt < 0).any():
            raise ValueError(f"Invalid direct-neutrino components for role {side}")
        nu_px = nu_pt * np.cos(nu_phi)
        nu_py = nu_pt * np.sin(nu_phi)
        nu_pz = nu_pt * np.sinh(nu_eta)
        neutrinos[:, side_index] = np.stack((nu_px, nu_py, nu_pz), axis=-1).astype(np.float32)
        daughter_ids[:, 2 * side_index] = scalar(
            arrays, f"Ttbar_MC_Wdecay1_afterFSR_from_{side}_pdgId", np.int32
        )[selected_rows]
        daughter_ids[:, 2 * side_index + 1] = scalar(
            arrays, f"Ttbar_MC_Wdecay2_afterFSR_from_{side}_pdgId", np.int32
        )[selected_rows]
    if not np.isfinite(parents).all() or not np.isfinite(neutrinos).all():
        raise FloatingPointError("After-FSR parent or direct-neutrino sidecar contains non-finite values")
    return {"parents": parents, "neutrinos": neutrinos, "w_daughter_pdg_id": daughter_ids}


def deterministic_split(event_ids: np.ndarray) -> np.ndarray:
    """Stable 90/5/5 split from the complete composite event identity."""
    value = np.asarray(event_ids, dtype=np.uint64).copy()
    value += np.uint64(0x9E3779B97F4A7C15)
    value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    value ^= value >> np.uint64(31)
    uniform = (value >> np.uint64(11)).astype(np.float64) * (1.0 / 2**53)
    return np.where(uniform < 0.90, 0, np.where(uniform < 0.95, 1, 2)).astype(np.uint8)


def weight_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Weight summary requires finite one-dimensional values")
    sumw = float(values.sum(dtype=np.float64))
    sumabsw = float(np.abs(values).sum(dtype=np.float64))
    sumw2 = float(np.square(values).sum(dtype=np.float64))
    return {
        "count": len(values),
        "sumw": sumw,
        "sumabsw": sumabsw,
        "sumw2": sumw2,
        "negative_count": int((values < 0).sum()),
        "negative_fraction": float((values < 0).mean()) if len(values) else None,
        "effective_count_signed": float(sumw * sumw / sumw2) if sumw2 > 0 else None,
        "effective_count_absolute": float(sumabsw * sumabsw / sumw2) if sumw2 > 0 else None,
    }


def write_hdf5(path: Path, arrays: dict, *, attrs: dict, dataset_attrs: dict | None = None) -> None:
    with h5py.File(path, "w") as output:
        for name, values in arrays.items():
            parent, _, leaf = name.rpartition("/")
            group = output.require_group(parent) if parent else output
            dataset = group.create_dataset(
                leaf or name,
                data=values,
                compression="lzf" if np.asarray(values).ndim > 1 else None,
            )
            for key, value in (dataset_attrs or {}).get(name, {}).items():
                dataset.attrs[key] = value
        for name, value in attrs.items():
            output.attrs[name] = value
        output.flush()


def convert(input_path: Path, output_path: Path) -> dict:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"KK ROOT input does not exist: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite KK transfer output: {output_path}")
    source_sha = sha256_file(input_path)
    with uproot.open(input_path) as root_file:
        if "reco" not in root_file or root_file["reco"].classname != "TTree":
            raise TypeError(f"KK input must contain a TTree named reco: {input_path}")
        tree = root_file["reco"]
        branch_names = set(tree.keys())
        missing = sorted(REQUIRED_BRANCHES - branch_names)
        if missing:
            raise KeyError(f"{input_path}:reco is missing required branches {missing}")
        matched_core_set = set(JET_MATCH_CORE_BRANCHES) & branch_names
        matched_pdg_set = set(JET_MATCH_PDG_BRANCHES) & branch_names
        if matched_core_set and matched_core_set != set(JET_MATCH_CORE_BRANCHES):
            raise KeyError(
                f"{input_path}:reco has only part of the explicit visible-match contract: "
                f"present={sorted(matched_core_set)}, missing={sorted(set(JET_MATCH_CORE_BRANCHES) - matched_core_set)}"
            )
        if matched_pdg_set and matched_pdg_set != set(JET_MATCH_PDG_BRANCHES):
            raise KeyError(
                f"{input_path}:reco has only part of the optional b-role PDG contract: "
                f"present={sorted(matched_pdg_set)}, missing={sorted(set(JET_MATCH_PDG_BRANCHES) - matched_pdg_set)}"
            )
        assignment_available = matched_core_set == set(JET_MATCH_CORE_BRANCHES)
        available_match_branches = sorted(matched_core_set | matched_pdg_set)
        present_weight_branches = sorted(set(WEIGHT_BRANCHES) & branch_names)
        arrays = tree.arrays(
            sorted(
                REQUIRED_BRANCHES
                | set(present_weight_branches)
                | (set(available_match_branches) if assignment_available else set())
            ),
            library="ak",
        )
        dsids = scalar(arrays, "mcChannelNumber", np.uint32)
        unique_dsids = np.unique(dsids)
        if len(unique_dsids) != 1 or int(unique_dsids[0]) not in KNOWN_DSIDS:
            raise ValueError(f"Expected one audited KK DSID in {sorted(KNOWN_DSIDS)}, found {unique_dsids.tolist()}")
        dsid = int(unique_dsids[0])
        metadata_labels, cutbookkeeper = root_metadata(root_file, dsid)
        event_number = scalar(arrays, "eventNumber", np.uint64)
        if event_number.max(initial=0) >= 2**32:
            raise ValueError(f"eventNumber cannot be encoded with the DSID prefix: {input_path}")
        if len(np.unique(event_number)) != len(event_number):
            raise ValueError(f"Duplicate eventNumber values in {input_path}:reco")

        n_el = ak.to_numpy(ak.num(arrays["el_pt_NOSYS"]))
        n_mu = ak.to_numpy(ak.num(arrays["mu_pt_NOSYS"]))
        el_charge = ak.to_numpy(ak.fill_none(ak.firsts(arrays["el_charge"]), 0.0)).astype(np.float64)
        mu_charge = ak.to_numpy(ak.fill_none(ak.firsts(arrays["mu_charge"]), 0.0)).astype(np.float64)
        top_lepton = scalar(arrays, "Ttbar_MC_Wdecay1_afterFSR_from_t_pdgId", np.int32)
        antitop_lepton = scalar(arrays, "Ttbar_MC_Wdecay1_afterFSR_from_tbar_pdgId", np.int32)
        top_neutrino = scalar(arrays, "Ttbar_MC_Wdecay2_afterFSR_from_t_pdgId", np.int32)
        antitop_neutrino = scalar(arrays, "Ttbar_MC_Wdecay2_afterFSR_from_tbar_pdgId", np.int32)
        one_each = (n_el == 1) & (n_mu == 1)
        opposite_sign = one_each & (el_charge * mu_charge < 0)
        charge_valid = np.isclose(np.abs(el_charge), 1.0, atol=1e-6) & np.isclose(
            np.abs(mu_charge), 1.0, atol=1e-6
        )
        one_positive = (el_charge > 0) ^ (mu_charge > 0)
        positive_abs_pdg = np.where(el_charge > 0, 11, 13)
        negative_abs_pdg = np.where(el_charge < 0, 11, 13)
        decay_match = (
            (top_lepton == -positive_abs_pdg)
            & (antitop_lepton == negative_abs_pdg)
            & (top_neutrino == positive_abs_pdg + 1)
            & (antitop_neutrino == -(negative_abs_pdg + 1))
        )
        selected = opposite_sign & charge_valid & one_positive & decay_match
        selected_rows = np.flatnonzero(selected)
        if len(selected_rows) == 0:
            raise ValueError(f"No direct, charge-matched OS eμ events in {input_path}")
        if not ((n_el[selected_rows] == 1) & (n_mu[selected_rows] == 1)).all():
            raise RuntimeError("Internal eμ selection failed its exact multiplicity check")

        measured = build_measured(arrays, selected_rows)
        truth = build_truth(arrays, selected_rows)
        if assignment_available:
            jet_labels, visible_support, assignment_reasons = build_visible_assignment(
                arrays, selected_rows, measured
            )
        else:
            jet_labels = visible_support = assignment_reasons = None
        selected_event_number = event_number[selected_rows]
        event_id = ((np.uint64(dsid) << np.uint64(32)) | selected_event_number).astype(np.int64)
        if len(np.unique(event_id)) != len(event_id):
            raise ValueError("Composite DSID/eventNumber identities are not unique")

        run_number = scalar(arrays, "runNumber", np.uint32)[selected_rows]
        raw_weights = {}
        for branch in present_weight_branches:
            values = scalar(arrays, branch, np.float64)[selected_rows]
            if not np.isfinite(values).all():
                raise ValueError(f"{input_path}:reco:{branch} contains non-finite selected values")
            raw_weights[branch] = values
        if "weight_mc_NOSYS" not in raw_weights:
            raise KeyError(f"{input_path}:reco:weight_mc_NOSYS is required to define w_gen")
        mc_weight = raw_weights["weight_mc_NOSYS"]
        w_analysis = np.full(len(event_id), np.nan, dtype=np.float64)
        w_train = np.full(len(event_id), np.nan, dtype=np.float64)
        canonical_split = deterministic_split(event_id)
        source_entry = selected_rows.astype(np.uint64)
        measured_output = {
            "EVENT_ID": event_id,
            # This view is explicitly an external validation population for
            # frozen-checkpoint inference. The canonical file retains the
            # deterministic 90/5/5 split below.
            "SPLIT": np.ones(len(event_id), dtype=np.uint8),
            "INPUTS/JET": measured["JET"],
            "INPUTS/LEPTON": measured["LEPTON"],
            "INPUTS/MET": measured["MET"],
            "INPUTS/GLOBAL": measured["GLOBAL"],
            "META/eventNumber": selected_event_number,
            "META/runNumber": run_number,
            "META/mcChannelNumber": np.full(len(event_id), dsid, dtype=np.uint32),
            "META/source_entry_index": source_entry,
            "META/w_gen": mc_weight,
            "META/w_analysis": w_analysis,
            "META/w_train": w_train,
            "META/weight_available_w_gen": np.ones(len(event_id), dtype=np.bool_),
            "META/weight_available_w_analysis": np.zeros(len(event_id), dtype=np.bool_),
            "META/weight_available_w_train": np.zeros(len(event_id), dtype=np.bool_),
        }
        for branch, values in raw_weights.items():
            measured_output[f"META/{branch}"] = values
        neutrino_labels = np.empty(
            (len(event_id), 2), dtype=np.dtype([("px", "<f4"), ("py", "<f4"), ("pz", "<f4")])
        )
        for component, index in (("px", 0), ("py", 1), ("pz", 2)):
            neutrino_labels[component] = truth["neutrinos"][:, :, index]
        truth_output = {
            "EVENT_ID": event_id,
            "LABELS/LEPTON": np.broadcast_to(
                np.asarray((1.0, 2.0), dtype=np.float32), (len(event_id), 2)
            ).copy(),
            "LABELS/NEUTRINO": neutrino_labels,
            "SUPPORT/INVISIBLE_ROLE": np.ones((len(event_id), 2), dtype=np.bool_),
            "PARTONIC/top": truth["parents"][:, 0],
            "PARTONIC/antitop": truth["parents"][:, 1],
            "PARTONIC/SUPPORT": np.ones((len(event_id), 2), dtype=np.bool_),
            "TRUTH/parents": truth["parents"],
            "TRUTH/neutrinos": truth["neutrinos"],
            "TRUTH/w_daughter_pdg_id": truth["w_daughter_pdg_id"],
        }
        if assignment_available:
            truth_output["LABELS/JET"] = jet_labels
            truth_output["SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"] = visible_support
        sample_class = {
            574801: "unclassified_signal_candidate",
            601230: "sm_ttbar",
        }[dsid]
        source_sample = (
            "mc23_13p6TeV.601230.PhPy8EG_A14_ttbar_hdamp258p75_dil"
            if dsid == 601230 else None
        )
        sbi_output = {
            "SBI/mass_gev": np.full(len(event_id), np.nan, dtype=np.float32),
            "SBI/mass_valid": np.zeros(len(event_id), dtype=np.bool_),
            "SBI/coupling": np.full(len(event_id), np.nan, dtype=np.float32),
            "SBI/coupling_valid": np.zeros(len(event_id), dtype=np.bool_),
            "SBI/width_gev": np.full(len(event_id), np.nan, dtype=np.float32),
            "SBI/width_valid": np.zeros(len(event_id), dtype=np.bool_),
        }
        canonical_output = {
            **measured_output,
            **truth_output,
            **sbi_output,
            "SPLIT": canonical_split,
        }
        weight_summaries = {
            "w_gen": weight_summary(mc_weight),
            "w_analysis": None,
            "w_train": None,
            "components": {
                branch: weight_summary(values) for branch, values in raw_weights.items()
            },
        }
        source_metadata = {
            "sample_class": sample_class,
            "source_sample_name": source_sample,
            "simulation_type": metadata_labels[0],
            "production_campaign": metadata_labels[1],
            "mcChannelNumber": dsid,
            "production_tag": metadata_labels[3],
            "mass_gev": None,
            "coupling": None,
            "width_gev": None,
            "interference_semantics": "unknown; no source JO or metadata in delivered ROOT file",
            "cross_section_pb": None,
            "filter_efficiency": None,
            "k_factor": None,
            "generator_process": (
                "PhPy8EG_A14_ttbar_hdamp258p75_dil (MC23a DSID catalog identity)"
                if dsid == 601230 else None
            ),
        }
        measured_attrs = {
            "source_root_file": str(input_path),
            "source_root_sha256": source_sha,
            "source_tree": "reco",
            "sample_metadata_labels": json.dumps(metadata_labels),
            "mcChannelNumber": dsid,
            "event_id_semantics": "(mcChannelNumber << 32) | eventNumber",
            "kinematic_units": "GeV",
            "four_vector_order": "E,px,py,pz",
            "lepton_slot_order": "positive charge (top), negative charge (antitop)",
            "measured_only": True,
            "split_meaning": "all rows are external transfer validation rows; no row was used in training",
            "split_version": "external_transfer_validation_all_events_v1",
            "max_jets": JET_WIDTH,
            "weight_definitions": json.dumps({
                "w_gen": "weight_mc_NOSYS",
                "w_analysis": None,
                "w_train": None,
                "unavailable_policy": "NaN with explicit availability=false; never substitute unity",
            }, sort_keys=True),
            "weight_component_branches": json.dumps(present_weight_branches),
            "jet_btag_source": "jet_GN2v01_Continuous_quantile; values retained as integer categories 1..6",
            "met_convention": "E=MET,px=MET*cos(phi),py=MET*sin(phi),pz=0; not a neutrino four-vector",
        }
        truth_attrs = {
            "source_root_file": str(input_path),
            "source_root_sha256": source_sha,
            "source_tree": "reco",
            "mcChannelNumber": dsid,
            "parent_truth_definition": "Ttbar_MC_t_afterFSR and Ttbar_MC_tbar_afterFSR",
            "kinematic_units": "GeV",
            "four_vector_order": "E,px,py,pz",
            "neutrino_component_order": "px,py,pz",
            "partonic_roles": json.dumps(["top", "antitop"]),
            "partonic_truth_bindings": json.dumps({
                "top": "Ttbar_MC_t_afterFSR",
                "antitop": "Ttbar_MC_tbar_afterFSR",
            }, sort_keys=True),
            "partonic_truth_definition": "partonic_after_fsr",
            "partonic_units": "GeV",
            "partonic_target_representation": "partonic_px_py_pz_log1p_mass_v1",
            "partonic_feature_transform": "[px,py,pz,log1p(m)] then training-only per-role component standardisation",
            "partonic_inverse_transform": "m=expm1(raw_log_mass); E=sqrt(px^2+py^2+pz^2+m^2)",
            "sidecar_version": "kk_ttbar_roots_truth_v2",
            "max_jets": JET_WIDTH,
            "invisible_roles": json.dumps(["top.nu", "antitop.nu"]),
            "invisible_role_order": json.dumps(["top.nu", "antitop.nu"]),
            "invisible_role_convention": "top_antitop",
            "units": "GeV",
            "neutrino_truth_definition": "generator top-side neutrino and generator antitop-side neutrino",
            "synthetic": False,
        }
        canonical_attrs = {
            **measured_attrs,
            "measured_only": False,
            "canonical_contract_version": "kk_ttbar_roots_v2",
            "sample_class": sample_class,
            "source_sample_name": source_sample or "unknown_from_delivered_metadata",
            "production_campaign": metadata_labels[1],
            "simulation_type": metadata_labels[0],
            "production_tag": metadata_labels[3],
            "sample_parameters": json.dumps({
                "mass_gev": None,
                "coupling": None,
                "width_gev": None,
                "interference_semantics": "unknown; no source JO or metadata in delivered ROOT file",
            }, sort_keys=True),
            "normalization_metadata": json.dumps({
                "cross_section_pb": None,
                "filter_efficiency": None,
                "k_factor": None,
                "source": "not present in delivered ROOT metadata; no JO/AMI record resolved",
            }, sort_keys=True),
            "split_version": "event_id_splitmix64_90_5_5_v1",
            "split_meaning": "0=train, 1=validation, 2=test; deterministic SplitMix64 of EVENT_ID",
            "partonic_roles": json.dumps(["top", "antitop"]),
            "partonic_truth_bindings": json.dumps({
                "top": "Ttbar_MC_t_afterFSR",
                "antitop": "Ttbar_MC_tbar_afterFSR",
            }, sort_keys=True),
            "partonic_truth_definition": "partonic_after_fsr",
            "partonic_units": "GeV",
            "partonic_target_representation": "partonic_px_py_pz_log1p_mass_v1",
            "partonic_feature_transform": "[px,py,pz,log1p(m)] then training-only per-role component standardisation",
            "partonic_inverse_transform": "m=expm1(raw_log_mass); E=sqrt(px^2+py^2+pz^2+m^2)",
            "invisible_roles": json.dumps(["top.nu", "antitop.nu"]),
            "invisible_role_order": json.dumps(["top.nu", "antitop.nu"]),
            "invisible_role_convention": "top_antitop",
            "neutrino_truth_definition": "generator top-side neutrino and generator antitop-side neutrino",
            "synthetic": False,
            "source_sample_metadata": json.dumps(source_metadata, sort_keys=True),
            "weight_summaries": json.dumps(weight_summaries, sort_keys=True),
            "assignment_label_convention": (
                "1=top b, 2=antitop bbar, 0=other real jet, NaN=padding"
                if assignment_available else "unavailable"
            ),
            "visible_assignment_source": (
                "explicit unique reco-jet indices from after-FSR b and bbar match branches"
                if assignment_available else "unavailable; no explicit charge-side reco↔truth b match branches"
            ),
        }
        report = {
            "input_root_file": str(input_path),
            "input_root_sha256": source_sha,
            "source_tree": "reco",
            "source_entries": int(tree.num_entries),
            "output_directory": str(output_path),
            "sample_metadata_labels": metadata_labels,
            "DSID": dsid,
            "sample_class": sample_class,
            "source_sample_name": source_sample,
            "production_campaign": metadata_labels[1],
            "simulation_type": metadata_labels[0],
            "production_tag": metadata_labels[3],
            "physics_parameters": {
                "mass_gev": None,
                "coupling": None,
                "width_gev": None,
                "interference_semantics": "unknown; source JO or authoritative sample metadata not resolved",
            },
            "normalization": {
                "cross_section_pb": None,
                "filter_efficiency": None,
                "k_factor": None,
                "source": "not present in delivered ROOT metadata; no JO/AMI record resolved",
            },
            "cutbookkeeper": cutbookkeeper,
            "selection": {
                "exactly_one_reconstructed_electron_and_muon": True,
                "opposite_reconstructed_charge": True,
                "top_and_antitop_W_daughters_are_direct_e_or_mu": True,
                "generator_lepton_flavour_and_charge_match_reconstructed_charge_side": True,
                "neutrino_pdg_id_matches_direct_W_decay_flavour": True,
                "additional_analysis_cuts": "none; use the delivered reco tree population",
            },
            "events": {
                "one_e_one_mu": int(one_each.sum()),
                "OS_e_mu": int(opposite_sign.sum()),
                "direct_charge_matched_e_mu_written": len(event_id),
                "njet_min_written": int(measured["GLOBAL"]["njet"].min()),
                "njet_max_written": int(measured["GLOBAL"]["njet"].max()),
                "events_with_more_than_8_jets": int((measured["GLOBAL"]["njet"] > JET_WIDTH).sum()),
            },
            "canonical_split": {
                "definition": "SplitMix64(EVENT_ID), 90% train / 5% validation / 5% test thresholds",
                "counts": {
                    "train_0": int((canonical_split == 0).sum()),
                    "validation_1": int((canonical_split == 1).sum()),
                    "test_2": int((canonical_split == 2).sum()),
                },
                "membership_sha256": hashlib.sha256(
                    np.column_stack((event_id, canonical_split)).astype("<i8").tobytes()
                ).hexdigest(),
            },
            "visible_assignment": {
                "available": assignment_available,
                "source": (
                    "explicit reco↔truth b match branches; original jet indices remapped after stable pT sort"
                    if assignment_available else None
                ),
                "supported_events": int(visible_support.sum()) if assignment_available else None,
                "unsupported_events": int((~visible_support).sum()) if assignment_available else None,
                "reasons": assignment_reasons,
                "status": (
                    "labels and reconstructability support written"
                    if assignment_available
                    else "unavailable; delivered tree lacks explicit charge-side reco↔truth b match branches"
                ),
            },
            "weights": {
                **weight_summaries,
                "definitions": {
                    "w_gen": "weight_mc_NOSYS",
                    "w_analysis": None,
                    "w_train": None,
                    "unavailable_policy": "stored as NaN with availability=false; unity is never substituted",
                },
                "available_component_branches": present_weight_branches,
            },
            "branch_mapping": {
                "jet_kinematics": ["jet_e_NOSYS", "jet_pt_NOSYS", "jet_eta", "jet_phi"],
                "jet_btag_category": "jet_GN2v01_Continuous_quantile",
                "jet_quality": "jet_select_baselineJvt_NOSYS (all selected jet flags must equal 1)",
                "jet_working_points": WP_BRANCHES,
                "lepton_kinematics": ["el_*_NOSYS / el_eta / el_phi / el_charge", "mu_*_NOSYS / mu_eta / mu_phi / mu_charge"],
                "met": ["met_met_NOSYS", "met_phi_NOSYS"],
                "parents": ["Ttbar_MC_t_afterFSR_*", "Ttbar_MC_tbar_afterFSR_*"],
                "W_decay_pdgs": [
                    "Ttbar_MC_Wdecay1_afterFSR_from_t_pdgId",
                    "Ttbar_MC_Wdecay2_afterFSR_from_t_pdgId",
                    "Ttbar_MC_Wdecay1_afterFSR_from_tbar_pdgId",
                    "Ttbar_MC_Wdecay2_afterFSR_from_tbar_pdgId",
                ],
                "visible_assignment": available_match_branches if assignment_available else [],
                "b_role_pdg_diagnostics": sorted(matched_pdg_set),
                "uninterpreted_candidate_matching_branches": [
                    name for name in ("event_jet_truth_candidates", "event_jet_truth_idx", "dilep_assigner_jet_indices_NOSYS")
                    if name in branch_names
                ] if not assignment_available else [],
                "weight_components": present_weight_branches,
            },
            "outputs": {
                "canonical_event_file": "canonical.h5",
                "measured_input": "measured.h5",
                "truth_sidecar": "truth.h5",
                "ordered_EVENT_ID_sha256": hashlib.sha256(event_id.astype("<i8").tobytes()).hexdigest(),
                "measured_parent_truth_rows": False,
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=output_path.parent))
    try:
        neutrino_dataset_attrs = {
            "invisible_role_convention": "top_antitop",
            "units": "GeV",
            "truth_definition": "generator top-side neutrino and generator antitop-side neutrino",
            "component_order": "px,py,pz",
        }
        write_hdf5(temporary / "measured.h5", measured_output, attrs=measured_attrs)
        write_hdf5(
            temporary / "truth.h5",
            truth_output,
            attrs=truth_attrs,
            dataset_attrs={
                "LABELS/JET": {
                    "convention": "1=top b, 2=antitop bbar, 0=other real jet, NaN=padding",
                    "source": "explicit reco↔truth b match branches",
                },
                "LABELS/LEPTON": {
                    "convention": "1=positive charge (top side), 2=negative charge (antitop side)",
                },
                "LABELS/NEUTRINO": neutrino_dataset_attrs,
                "PARTONIC/top": {"units": "GeV", "physical_order": "E,px,py,pz"},
                "PARTONIC/antitop": {"units": "GeV", "physical_order": "E,px,py,pz"},
            },
        )
        write_hdf5(
            temporary / "canonical.h5",
            canonical_output,
            attrs=canonical_attrs,
            dataset_attrs={
                "LABELS/JET": {
                    "convention": "1=top b, 2=antitop bbar, 0=other real jet, NaN=padding",
                    "source": "explicit reco↔truth b match branches",
                },
                "LABELS/LEPTON": {
                    "convention": "1=positive charge (top side), 2=negative charge (antitop side)",
                },
                "LABELS/NEUTRINO": neutrino_dataset_attrs,
                "PARTONIC/top": {"units": "GeV", "physical_order": "E,px,py,pz"},
                "PARTONIC/antitop": {"units": "GeV", "physical_order": "E,px,py,pz"},
            },
        )
        (temporary / "contract.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        with h5py.File(temporary / "measured.h5", "r") as measured_check, h5py.File(
            temporary / "truth.h5", "r"
        ) as truth_check:
            ids = measured_check["EVENT_ID"][:]
            if not np.array_equal(ids, truth_check["EVENT_ID"][:]):
                raise ValueError("Measured input and truth sidecar EVENT_ID order differs")
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Output EVENT_ID values are not unique")
            if measured_check["INPUTS/JET"].shape != (len(ids), JET_WIDTH):
                raise ValueError("Output JET shape differs from the frozen eight-jet input contract")
            for dataset_name in (
                "INPUTS/LEPTON", "INPUTS/MET", "INPUTS/GLOBAL",
            ):
                if len(measured_check[dataset_name]) != len(ids):
                    raise ValueError(f"Output {dataset_name} event count differs from EVENT_ID")
            if truth_check["TRUTH/parents"].shape != (len(ids), 2, 4):
                raise ValueError("Output after-FSR parent sidecar has invalid shape")
            if truth_check["TRUTH/neutrinos"].shape != (len(ids), 2, 3):
                raise ValueError("Output direct-neutrino sidecar has invalid shape")
            for dataset_name in ("TRUTH/parents", "TRUTH/neutrinos"):
                if not np.isfinite(truth_check[dataset_name][:]).all():
                    raise ValueError(f"Output {dataset_name} contains non-finite values")
            if assignment_available:
                labels = truth_check["LABELS/JET"][:]
                support = truth_check["SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"][:]
                if support.dtype != np.bool_ or support.shape != (len(ids),):
                    raise TypeError("Visible assignment support must be boolean per event")
                njet = measured_check["INPUTS/GLOBAL"]["njet"].astype(np.int64)
                for row, count in enumerate(njet):
                    real = min(int(count), JET_WIDTH)
                    if not np.isfinite(labels[row, :real]).all() or not np.isnan(labels[row, real:]).all():
                        raise ValueError(f"Visible labels do not close over real/padded jets at EVENT_ID={int(ids[row])}")
                    if support[row] and (
                        int((labels[row, :real] == 1).sum()) != 1
                        or int((labels[row, :real] == 2).sum()) != 1
                    ):
                        raise ValueError(f"Supported visible assignment is not unique at EVENT_ID={int(ids[row])}")
                    if not support[row] and np.any(np.isin(labels[row, :real], (1, 2))):
                        raise ValueError(f"Unsupported visible assignment has partial labels at EVENT_ID={int(ids[row])}")
            with h5py.File(temporary / "canonical.h5", "r") as canonical_check:
                if not np.array_equal(canonical_check["EVENT_ID"][:], ids):
                    raise ValueError("Canonical event file and measured transfer view EVENT_ID order differs")
                if not np.array_equal(canonical_check["SPLIT"][:], canonical_split):
                    raise ValueError("Canonical deterministic split differs from the recorded split")
                if assignment_available and not np.array_equal(
                    canonical_check["LABELS/JET"][:], labels, equal_nan=True
                ):
                    raise ValueError("Canonical and truth-sidecar visible labels differ")
                if not np.isfinite(canonical_check["META/w_gen"][:]).all():
                    raise ValueError("Canonical w_gen contains non-finite selected values")
                if not np.isnan(canonical_check["META/w_analysis"][:]).all() or not np.isnan(
                    canonical_check["META/w_train"][:]
                ).all():
                    raise ValueError("Unavailable analysis/train weights must remain explicitly NaN")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            import shutil

            shutil.rmtree(temporary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = convert(args.input, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
