"""Validate and compare stored VyPER assignment/neutrino outputs on KK events."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import awkward as ak
import h5py
import numpy as np
import uproot

EXPECTED_TOPOLOGY = "5cbc52bdf24eaf5d208151f3f0e506a4846150e666e2a1697c2e3b2282db348e"
BOOTSTRAP_SEED = 42011
BOOTSTRAP_RESAMPLES = 1000
MAX_PARENT_CLOSURE_GEV = 0.005
MAX_COMPONENT_PT_DIFFERENCE_GEV = 1.0e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_digest(event_ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(event_ids, dtype="<i8").tobytes()).hexdigest()


def required_branch(tree, name: str):
    if name not in tree:
        raise KeyError(f"Required VyPER ROOT branch is absent: {name}")
    return tree[name]


def scalar_rows(tree, name: str, rows: np.ndarray) -> np.ndarray:
    return np.asarray(required_branch(tree, name).array(library="np"))[rows]


def vector_rows(tree, name: str, rows: np.ndarray) -> list[np.ndarray]:
    values = ak.to_list(required_branch(tree, name).array(library="ak"))
    return [np.asarray(values[int(row)]) for row in rows]


def root_four_vector(tree, branch: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coordinates = {}
    for coordinate in ("fPt", "fEta", "fPhi", "fM"):
        path = f"{branch}/fCoordinates/fCoordinates.{coordinate}"
        coordinates[coordinate] = np.asarray(
            required_branch(tree, path).array(library="np"), dtype=np.float64
        )[rows]
    pt = coordinates["fPt"] * 1.0e-3
    eta = coordinates["fEta"]
    phi = coordinates["fPhi"]
    mass = coordinates["fM"] * 1.0e-3
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(mass**2 + (pt * np.cosh(eta)) ** 2)
    vector = np.stack((energy, px, py, pz), axis=-1)
    if not np.isfinite(vector).all():
        raise FloatingPointError(f"VyPER vector {branch} has non-finite components")
    return vector, pt


def bootstrap_mean_interval(
    event_values: np.ndarray, bootstrap_indices: np.ndarray
) -> list[float]:
    sampled_means = event_values[bootstrap_indices].mean(axis=1)
    return [float(value) for value in np.quantile(sampled_means, [0.025, 0.975])]


def paired_rmse_difference(
    first_event_mse: np.ndarray,
    second_event_mse: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict:
    difference = np.sqrt(first_event_mse[bootstrap_indices].mean(axis=1)) - np.sqrt(
        second_event_mse[bootstrap_indices].mean(axis=1)
    )
    return {
        "first_minus_second_RMSE_GeV": float(
            np.sqrt(first_event_mse.mean()) - np.sqrt(second_event_mse.mean())
        ),
        "event_bootstrap_95pct_interval_GeV": [
            float(value) for value in np.quantile(difference, [0.025, 0.975])
        ],
    }


def load_hdf5_contract(measured_path: Path, truth_path: Path) -> dict:
    with h5py.File(measured_path, "r") as measured, h5py.File(
        truth_path, "r"
    ) as truth:
        for path in (
            "EVENT_ID",
            "INPUTS/JET",
            "INPUTS/LEPTON",
        ):
            if path not in measured:
                raise KeyError(f"Measured input {measured_path} lacks {path}")
        for path in (
            "EVENT_ID",
            "LABELS/JET",
            "SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE",
            "TRUTH/neutrinos",
        ):
            if path not in truth:
                raise KeyError(f"Truth contract {truth_path} lacks {path}")
        event_ids = np.asarray(measured["EVENT_ID"][:], dtype=np.int64)
        truth_ids = np.asarray(truth["EVENT_ID"][:], dtype=np.int64)
        if event_ids.ndim != 1 or len(np.unique(event_ids)) != len(event_ids):
            raise ValueError(f"Measured EVENT_ID is not unique: {measured_path}")
        if not np.array_equal(event_ids, truth_ids):
            raise ValueError("Measured and truth EVENT_ID order differs")
        jets = np.asarray(measured["INPUTS/JET"][:])
        leptons = np.asarray(measured["INPUTS/LEPTON"][:])
        labels = np.asarray(truth["LABELS/JET"][:], dtype=np.float64)
        assignment_support = np.asarray(
            truth["SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"][:], dtype=bool
        )
        neutrinos = np.asarray(truth["TRUTH/neutrinos"][:], dtype=np.float64)
        role_order = json.loads(truth.attrs["invisible_role_order"])
        label_convention = str(truth["LABELS/JET"].attrs["convention"])
        event_id_semantics = str(measured.attrs["event_id_semantics"])
        root_sha256 = str(measured.attrs["source_root_sha256"])
    if labels.shape != jets.shape[:2] or assignment_support.shape != (len(event_ids),):
        raise ValueError("Truth assignment labels/support have incompatible shapes")
    if neutrinos.shape != (len(event_ids), 2, 3) or not np.isfinite(neutrinos).all():
        raise ValueError("Charge-side neutrino truth must be finite [events,2,3]")
    if role_order != ["top.nu", "antitop.nu"]:
        raise ValueError(f"Unexpected neutrino role order metadata: {role_order!r}")
    if label_convention != "1=top b, 2=antitop bbar, 0=other real jet, NaN=padding":
        raise ValueError(f"Unexpected visible assignment convention: {label_convention!r}")
    n_jets = np.isfinite(jets["pt"]).sum(axis=1).astype(np.int64)
    n_leptons = np.isfinite(leptons["pt"]).sum(axis=1).astype(np.int64)
    if np.any(n_jets <= 0) or not np.all(n_leptons == 2):
        raise ValueError("Transfer input must have real jets and exactly two measured leptons")
    truth_jet_indices = np.full((len(event_ids), 2), -1, dtype=np.int64)
    for role_index, label in enumerate((1.0, 2.0)):
        for row in np.flatnonzero(assignment_support):
            candidates = np.flatnonzero(labels[row] == label)
            if len(candidates) != 1:
                raise ValueError(
                    f"Truth assignment label {int(label)} is not unique: "
                    f"event={int(event_ids[row])}, candidates={candidates.tolist()}"
                )
            truth_jet_indices[row, role_index] = candidates[0]
    return {
        "event_ids": event_ids,
        "jets": jets,
        "n_jets": n_jets,
        "max_jets": labels.shape[1],
        "n_leptons": n_leptons,
        "assignment_support": assignment_support,
        "truth_jet_indices": truth_jet_indices,
        "truth_neutrinos": neutrinos,
        "root_sha256": root_sha256,
        "event_id_semantics": event_id_semantics,
    }


def load_vyper(root_path: Path, data: dict) -> dict:
    event_ids = data["event_ids"]
    with uproot.open(root_path) as source:
        if "reco" not in source:
            raise KeyError(f"VyPER ROOT file lacks required reco tree: {root_path}")
        tree = source["reco"]
        event_number = np.asarray(
            required_branch(tree, "eventNumber").array(library="np"), dtype=np.int64
        )
        channel = np.asarray(
            required_branch(tree, "mcChannelNumber").array(library="np"),
            dtype=np.int64,
        )
        if len(np.unique(event_number)) != len(event_number):
            raise ValueError(f"ROOT eventNumber is not unique in {root_path}")
        dsids = np.unique(channel)
        if len(dsids) != 1 or int(dsids[0]) != 574801:
            raise ValueError(f"Expected only DSID 574801 in {root_path}; found {dsids}")
        root_ids = (channel << np.int64(32)) | event_number
        if len(np.unique(root_ids)) != len(root_ids):
            raise ValueError(f"Stable ROOT EVENT_ID is not unique in {root_path}")
        row_by_id = {int(value): index for index, value in enumerate(root_ids)}
        missing = [int(value) for value in event_ids if int(value) not in row_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)}/{len(event_ids)} canonical IDs are absent from ROOT; "
                f"first missing IDs={missing[:5]}"
            )
        rows = np.asarray([row_by_id[int(value)] for value in event_ids], dtype=np.int64)

        top_parent, _ = root_four_vector(
            tree, "TtbarDiLepton_VyPER_Top_NOSYS", rows
        )
        antitop_parent, _ = root_four_vector(
            tree, "TtbarDiLepton_VyPER_Tbar_NOSYS", rows
        )
        top_b, top_b_pt = root_four_vector(
            tree, "TtbarDiLepton_VyPER_Top_b_p4_NOSYS", rows
        )
        top_lepton, top_lepton_pt = root_four_vector(
            tree, "TtbarDiLepton_VyPER_Top_lep_p4_NOSYS", rows
        )
        top_neutrino, _ = root_four_vector(
            tree, "TtbarDiLepton_VyPER_Top_nu_p4_NOSYS", rows
        )
        antitop_b, antitop_b_pt = root_four_vector(
            tree, "TtbarDiLepton_VyPER_Topbar_bbar_p4_NOSYS", rows
        )
        antitop_lepton, antitop_lepton_pt = root_four_vector(
            tree, "TtbarDiLepton_VyPER_Topbar_lepbar_p4_NOSYS", rows
        )
        antitop_neutrino, _ = root_four_vector(
            tree, "TtbarDiLepton_VyPER_Topbar_nubar_p4_NOSYS", rows
        )
        top_ids = vector_rows(tree, "TtbarDiLepton_VyPER_Top_IDs_NOSYS", rows)
        antitop_ids = vector_rows(
            tree, "TtbarDiLepton_VyPER_AntiTop_IDs_NOSYS", rows
        )
        top_indices = vector_rows(
            tree, "TtbarDiLepton_VyPER_Top_Indices_NOSYS", rows
        )
        antitop_indices = vector_rows(
            tree, "TtbarDiLepton_VyPER_AntiTop_Indices_NOSYS", rows
        )
        jet_pt = vector_rows(tree, "jet_pt_NOSYS", rows)
        electron_pt = vector_rows(tree, "el_pt_NOSYS", rows)
        muon_pt = vector_rows(tree, "mu_pt_NOSYS", rows)
        electron_charge = vector_rows(tree, "el_charge", rows)
        muon_charge = vector_rows(tree, "mu_charge", rows)
        truth_top_jet = scalar_rows(
            tree, "jet_dRmatched_to_Ttbar_MC_b_afterFSR_from_t", rows
        ).astype(np.int64)
        truth_antitop_jet = scalar_rows(
            tree, "jet_dRmatched_to_Ttbar_MC_bbar_afterFSR_from_tbar", rows
        ).astype(np.int64)
        n_truth_top_jet = scalar_rows(
            tree, "n_jet_matched_to_Ttbar_MC_b_afterFSR_from_t", rows
        ).astype(np.int64)
        n_truth_antitop_jet = scalar_rows(
            tree, "n_jet_matched_to_Ttbar_MC_bbar_afterFSR_from_tbar", rows
        ).astype(np.int64)

    top_rank = np.full(len(event_ids), -1, dtype=np.int64)
    antitop_rank = np.full(len(event_ids), -1, dtype=np.int64)
    for row in range(len(event_ids)):
        order = np.argsort(-jet_pt[row], kind="stable")
        if 0 <= truth_top_jet[row] < len(order):
            top_rank[row] = int(np.flatnonzero(order == truth_top_jet[row])[0])
        if 0 <= truth_antitop_jet[row] < len(order):
            antitop_rank[row] = int(np.flatnonzero(order == truth_antitop_jet[row])[0])
    root_support = (
        (n_truth_top_jet == 1)
        & (n_truth_antitop_jet == 1)
        & (truth_top_jet >= 0)
        & (truth_antitop_jet >= 0)
        & (truth_top_jet != truth_antitop_jet)
        & (top_rank < data["max_jets"])
        & (antitop_rank < data["max_jets"])
    )
    if not np.array_equal(data["assignment_support"], root_support):
        raise ValueError("ROOT unique-match/support rules differ from the canonical truth contract")
    for row in np.flatnonzero(data["assignment_support"]):
        if (
            top_rank[row] != data["truth_jet_indices"][row, 0]
            or antitop_rank[row] != data["truth_jet_indices"][row, 1]
        ):
            raise ValueError(
                f"ROOT b-match indices disagree with canonical labels: "
                f"event={int(event_ids[row])}"
            )

    predicted_top_jet = np.empty(len(event_ids), dtype=np.int64)
    predicted_antitop_jet = np.empty(len(event_ids), dtype=np.int64)
    lepton_charge = np.empty((len(event_ids), 2), dtype=np.float64)
    lepton_pt_difference = np.empty((len(event_ids), 2), dtype=np.float64)
    b_pt_difference = np.empty((len(event_ids), 2), dtype=np.float64)
    for row in range(len(event_ids)):
        top_id, antitop_id = top_ids[row], antitop_ids[row]
        top_index, antitop_index = top_indices[row], antitop_indices[row]
        if (
            len(top_id) != 2
            or len(antitop_id) != 2
            or len(top_index) != 2
            or len(antitop_index) != 2
            or top_id[0] != 0
            or antitop_id[0] != 0
            or top_id[1] not in (1, 2)
            or antitop_id[1] not in (1, 2)
        ):
            raise ValueError(f"VyPER ID/index layout is invalid: event={int(event_ids[row])}")
        top_jet, antitop_jet = int(top_index[0]), int(antitop_index[0])
        if (
            not 0 <= top_jet < len(jet_pt[row])
            or not 0 <= antitop_jet < len(jet_pt[row])
            or top_jet == antitop_jet
        ):
            raise ValueError(f"VyPER selected b-jet indices are invalid: event={int(event_ids[row])}")
        predicted_top_jet[row] = top_jet
        predicted_antitop_jet[row] = antitop_jet
        b_pt_difference[row] = (
            top_b_pt[row] - jet_pt[row][top_jet] * 1.0e-3,
            antitop_b_pt[row] - jet_pt[row][antitop_jet] * 1.0e-3,
        )
        for role_index, (ident, object_index, pt_rows, charge_rows, branch_pt) in enumerate(
            (
                (
                    int(top_id[1]),
                    int(top_index[1]),
                    electron_pt if top_id[1] == 1 else muon_pt,
                    electron_charge if top_id[1] == 1 else muon_charge,
                    top_lepton_pt[row],
                ),
                (
                    int(antitop_id[1]),
                    int(antitop_index[1]),
                    electron_pt if antitop_id[1] == 1 else muon_pt,
                    electron_charge if antitop_id[1] == 1 else muon_charge,
                    antitop_lepton_pt[row],
                ),
            )
        ):
            if not 0 <= object_index < len(pt_rows[row]):
                raise ValueError(
                    f"VyPER lepton index is invalid: event={int(event_ids[row])}, role={role_index}"
                )
            lepton_charge[row, role_index] = charge_rows[row][object_index]
            lepton_pt_difference[row, role_index] = (
                branch_pt - pt_rows[row][object_index] * 1.0e-3
            )

    if not np.all(lepton_charge[:, 0] == 1.0) or not np.all(
        lepton_charge[:, 1] == -1.0
    ):
        raise ValueError("VyPER Top/AntiTop lepton branches do not follow charge-side roles")
    if np.max(np.abs(lepton_pt_difference)) > MAX_COMPONENT_PT_DIFFERENCE_GEV:
        raise ValueError("VyPER lepton ID/index branches do not resolve to stored constituents")
    if np.max(np.abs(b_pt_difference)) > MAX_COMPONENT_PT_DIFFERENCE_GEV:
        raise ValueError("VyPER b-jet indices do not resolve to stored constituents")

    top_closure = top_parent - top_b - top_lepton - top_neutrino
    antitop_closure = antitop_parent - antitop_b - antitop_lepton - antitop_neutrino
    closure_max = float(
        max(np.max(np.abs(top_closure)), np.max(np.abs(antitop_closure)))
    )
    if closure_max > MAX_PARENT_CLOSURE_GEV:
        raise ValueError(
            f"VyPER parent constituents do not close: max component residual="
            f"{closure_max:.6g} GeV"
        )

    support = data["assignment_support"]
    correct_top = predicted_top_jet == truth_top_jet
    correct_antitop = predicted_antitop_jet == truth_antitop_jet
    correct_complete = correct_top & correct_antitop
    if not support.any():
        raise ValueError("VyPER has no events with reconstructable truth b assignments")

    truth_nu = data["truth_neutrinos"]
    predicted_nu = np.stack((top_neutrino[:, 1:], antitop_neutrino[:, 1:]), axis=1)
    if not np.isfinite(predicted_nu).all():
        raise FloatingPointError("VyPER charge-side neutrino predictions are non-finite")
    nu_delta = predicted_nu - truth_nu
    event_neutrino_mse = np.square(nu_delta).sum(axis=-1).mean(axis=1)
    event_neutrino_pt_mse = np.square(
        np.hypot(predicted_nu[:, :, 0], predicted_nu[:, :, 1])
        - np.hypot(truth_nu[:, :, 0], truth_nu[:, :, 1])
    ).mean(axis=1)
    event_neutrino_pz_mse = np.square(nu_delta[:, :, 2]).mean(axis=1)
    return {
        "root_rows": len(root_ids),
        "matched_rows": len(event_ids),
        "matched_root_rows_sha256": hashlib.sha256(
            np.asarray(rows, dtype="<i8").tobytes()
        ).hexdigest(),
        "assignment": {
            "support_events": int(support.sum()),
            "top_accuracy": float(correct_top[support].mean()),
            "antitop_accuracy": float(correct_antitop[support].mean()),
            "complete_accuracy": float(correct_complete[support].mean()),
            "top_correct_events": int(correct_top[support].sum()),
            "antitop_correct_events": int(correct_antitop[support].sum()),
            "complete_correct_events": int(correct_complete[support].sum()),
            "charge_side_role_validation": {
                "top_positive_lepton_events": int(np.count_nonzero(lepton_charge[:, 0] == 1.0)),
                "antitop_negative_lepton_events": int(np.count_nonzero(lepton_charge[:, 1] == -1.0)),
                "max_abs_b_pt_vs_selected_root_jet_GeV": float(np.max(np.abs(b_pt_difference))),
                "max_abs_lepton_pt_vs_selected_root_lepton_GeV": float(np.max(np.abs(lepton_pt_difference))),
                "max_parent_constituent_closure_component_GeV": closure_max,
                "truth_root_b_matches_reproduce_canonical_labels": True,
            },
            "labels_used_only_for_post_inference_scoring": True,
        },
        "neutrino": {
            "role_order": ["top.nu", "antitop.nu"],
            "event_weighting": "unweighted; each event contributes total weight one",
            "3vector_RMSE_GeV": float(np.sqrt(event_neutrino_mse.mean())),
            "3vector_event_bootstrap_95pct_interval_GeV": None,
            "pT_RMSE_GeV": float(np.sqrt(event_neutrino_pt_mse.mean())),
            "pz_RMSE_GeV": float(np.sqrt(event_neutrino_pz_mse.mean())),
            "event_mse_3vector": event_neutrino_mse,
        },
        "correct_top": correct_top,
        "correct_antitop": correct_antitop,
        "correct_complete": correct_complete,
        "event_neutrino_mse": event_neutrino_mse,
        "event_neutrino_pt_mse": event_neutrino_pt_mse,
        "event_neutrino_pz_mse": event_neutrino_pz_mse,
    }


def load_knot_posterior(path: Path, event_ids: np.ndarray, n_jets: np.ndarray) -> dict:
    with h5py.File(path, "r") as posterior:
        required = (
            "EVENT_ID",
            "assignment_candidates",
            "assignment_support",
            "invisible",
        )
        missing = [name for name in required if name not in posterior]
        if missing:
            raise KeyError(f"KNOT posterior {path} lacks required fields: {missing}")
        observed_ids = np.asarray(posterior["EVENT_ID"][:], dtype=np.int64)
        candidates = np.asarray(posterior["assignment_candidates"][:], dtype=np.int64)
        assignment_support = np.asarray(posterior["assignment_support"][:], dtype=bool)
        invisible = np.asarray(posterior["invisible"][:], dtype=np.float64)
        assignment_nodes = json.loads(posterior.attrs["assignment_nodes"])
        invisible_roles = json.loads(posterior.attrs["invisible_roles"])
        topology = str(posterior.attrs["topology_identity"])
        sample_semantics = str(posterior.attrs["sample_semantics"])
    if not np.array_equal(observed_ids, event_ids):
        raise ValueError(f"KNOT posterior EVENT_ID order differs: {path}")
    if assignment_nodes != ["antitop", "top"]:
        raise ValueError(f"KNOT assignment node order differs: {path}: {assignment_nodes}")
    if invisible_roles != ["top.nu", "antitop.nu"]:
        raise ValueError(f"KNOT neutrino role order differs: {path}: {invisible_roles}")
    if topology != EXPECTED_TOPOLOGY:
        raise ValueError(f"KNOT topology differs from frozen charge-side topology: {path}")
    if candidates.ndim != 3 or candidates.shape[0] != len(event_ids) or candidates.shape[2] != 2:
        raise ValueError(f"KNOT assignment-candidate shape is invalid: {path}: {candidates.shape}")
    if assignment_support.shape != candidates.shape or not assignment_support.all():
        raise ValueError(f"KNOT assignment posterior has unsupported candidate roles: {path}")
    if invisible.shape != (len(event_ids), candidates.shape[1], 2, 4):
        raise ValueError(f"KNOT invisible posterior shape is invalid: {path}: {invisible.shape}")
    if not np.isfinite(invisible).all():
        raise FloatingPointError(f"KNOT invisible posterior is non-finite: {path}")
    if np.any(invisible[..., 0] < 0.0):
        raise ValueError(f"KNOT invisible energy is negative: {path}")
    if np.any(candidates < 0) or np.any(candidates >= n_jets[:, None, None] * 2):
        raise ValueError(f"KNOT assignment candidate is outside the per-event topology pool: {path}")
    # Frozen topology order is b-jet outer and the two measured lepton slots inner.
    candidate_jet_indices = candidates // 2
    massless_energy = np.sqrt(np.square(invisible[..., 1:4]).sum(axis=-1))
    mass_shell_residual = np.abs(invisible[..., 0] - massless_energy)
    if float(np.max(mass_shell_residual)) > 0.01:
        raise ValueError(
            f"KNOT neutrino samples fail the massless four-vector check: "
            f"max residual={float(np.max(mass_shell_residual)):.6g} GeV"
        )
    return {
        "candidates": candidates,
        "candidate_jet_indices": candidate_jet_indices,
        "invisible": invisible,
        "sample_semantics": sample_semantics,
        "massless_energy_max_residual_gev": float(np.max(mass_shell_residual)),
    }


def load_assignment_diagnostics(
    path: Path,
    label: str,
    event_ids: np.ndarray,
    assignment_support: np.ndarray,
    source_hashes: dict[str, str],
) -> dict:
    with path.open() as stream:
        diagnostics = json.load(stream)
    assignment = diagnostics.get("assignment", {})
    expected_support = int(assignment_support.sum())
    if diagnostics.get("events") != len(event_ids):
        raise ValueError(f"{label} assignment diagnostic event count differs: {path}")
    if diagnostics.get("event_id_ordered_sha256") != event_digest(event_ids):
        raise ValueError(f"{label} assignment diagnostic event order differs: {path}")
    if diagnostics.get("truth_used_during_inference") is not False:
        raise ValueError(f"{label} assignment diagnostic is not truth-free: {path}")
    if diagnostics.get("source_files_sha256") != source_hashes:
        raise ValueError(f"{label} assignment diagnostic source hashes differ: {path}")
    if (
        assignment.get("status") != "scored_from_explicit_after_matching_labels"
        or assignment.get("events_total") != len(event_ids)
        or assignment.get("supported_events") != expected_support
        or assignment.get("unsupported_events") != len(event_ids) - expected_support
        or assignment.get("labels_used_only_after_truth_free_forward") is not True
    ):
        raise ValueError(f"{label} assignment diagnostic support/label contract differs: {path}")
    accuracy = float(assignment["complete_MAP_accuracy"])
    interval = [
        float(value)
        for value in assignment["complete_MAP_accuracy_event_bootstrap_95pct_interval"]
    ]
    nll = float(assignment["truth_assignment_NLL_nats"])
    nll_interval = [
        float(value)
        for value in assignment["truth_assignment_NLL_event_bootstrap_95pct_interval_nats"]
    ]
    if (
        not np.isfinite([accuracy, *interval, nll, *nll_interval]).all()
        or not 0.0 <= accuracy <= 1.0
        or not 0.0 <= interval[0] <= accuracy <= interval[1] <= 1.0
    ):
        raise ValueError(f"{label} assignment diagnostic metrics are invalid: {path}")
    return {
        "diagnostics_file": str(path.resolve()),
        "complete_accuracy": accuracy,
        "event_bootstrap_95pct_interval": interval,
        "truth_assignment_NLL_nats": nll,
        "truth_assignment_NLL_event_bootstrap_95pct_interval_nats": nll_interval,
        "per_node_MAP_accuracy": assignment["per_node_MAP_accuracy"],
        "supported_events": expected_support,
        "bootstrap_resamples": int(assignment["bootstrap_resamples"]),
        "labels_used_only_after_truth_free_forward": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-file", type=Path, required=True)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument(
        "--posterior",
        action="append",
        required=True,
        metavar="M|C|J=PATH",
        help="supply the frozen transfer posterior for M, C, and/or J",
    )
    parser.add_argument(
        "--assignment-diagnostics",
        action="append",
        required=True,
        metavar="M|C|J=PATH",
        help="supply the exact-MAP, truth-free assignment score for each posterior",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    args = parser.parse_args()
    for path in (args.root_file, args.measured, args.truth):
        if not path.is_file():
            parser.error(f"Required source file does not exist: {path}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing KK/VyPER comparison: {args.output}")
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")

    posterior_paths = {}
    for value in args.posterior:
        label, separator, filename = value.partition("=")
        if not separator or label not in {"M", "C", "J"} or label in posterior_paths:
            parser.error(f"Invalid or duplicate --posterior: {value}")
        posterior_path = Path(filename).resolve()
        if not posterior_path.is_file():
            parser.error(f"KNOT posterior does not exist: {posterior_path}")
        posterior_paths[label] = posterior_path
    if not posterior_paths:
        parser.error("At least one KNOT posterior is required")
    diagnostic_paths = {}
    for value in args.assignment_diagnostics:
        label, separator, filename = value.partition("=")
        if not separator or label not in {"M", "C", "J"} or label in diagnostic_paths:
            parser.error(f"Invalid or duplicate --assignment-diagnostics: {value}")
        diagnostic_path = Path(filename).resolve()
        if not diagnostic_path.is_file():
            parser.error(f"KNOT assignment diagnostic does not exist: {diagnostic_path}")
        diagnostic_paths[label] = diagnostic_path
    if set(diagnostic_paths) != set(posterior_paths):
        parser.error("Provide exactly one assignment diagnostic for each KNOT posterior")

    data = load_hdf5_contract(args.measured.resolve(), args.truth.resolve())
    source_hashes = {
        "measured": sha256_file(args.measured.resolve()),
        "truth": sha256_file(args.truth.resolve()),
    }
    root_sha256 = sha256_file(args.root_file.resolve())
    if root_sha256 != data["root_sha256"]:
        raise ValueError(
            "The ROOT content hash differs from the source hash recorded in measured.h5"
        )
    vyper = load_vyper(args.root_file.resolve(), data)
    support_indices = np.flatnonzero(data["assignment_support"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    assignment_bootstrap = rng.integers(
        0, len(support_indices), size=(args.bootstrap_resamples, len(support_indices))
    )
    all_event_bootstrap = np.random.default_rng(BOOTSTRAP_SEED + 1).integers(
        0,
        len(data["event_ids"]),
        size=(args.bootstrap_resamples, len(data["event_ids"])),
    )
    vyper_event_correct = {
        "top": vyper["correct_top"][data["assignment_support"]].astype(np.float64),
        "antitop": vyper["correct_antitop"][data["assignment_support"]].astype(np.float64),
        "complete": vyper["correct_complete"][data["assignment_support"]].astype(np.float64),
    }
    vyper_nu_mse = vyper["event_neutrino_mse"]
    vyper_nu_pt_mse = vyper["event_neutrino_pt_mse"]
    vyper_nu_pz_mse = vyper["event_neutrino_pz_mse"]

    results = {}
    for label, posterior_path in posterior_paths.items():
        model_source_hashes = {
            **source_hashes,
            "posterior": sha256_file(posterior_path),
        }
        assignment_map = load_assignment_diagnostics(
            diagnostic_paths[label],
            label,
            data["event_ids"],
            data["assignment_support"],
            model_source_hashes,
        )
        posterior = load_knot_posterior(
            posterior_path, data["event_ids"], data["n_jets"]
        )
        candidates = posterior["candidate_jet_indices"]
        # Assignment nodes are [antitop, top]; truth labels are [top, antitop].
        role_correct = np.stack(
            (
                candidates[:, :, 1] == data["truth_jet_indices"][:, 0, None],
                candidates[:, :, 0] == data["truth_jet_indices"][:, 1, None],
            ),
            axis=-1,
        )
        role_correct &= data["assignment_support"][:, None, None]
        event_correct = {
            "top": role_correct[:, :, 0].mean(axis=1)[data["assignment_support"]],
            "antitop": role_correct[:, :, 1].mean(axis=1)[data["assignment_support"]],
            "complete": role_correct.all(axis=-1).mean(axis=1)[
                data["assignment_support"]
            ],
        }
        invisible = posterior["invisible"]
        predicted_neutrino = invisible[:, :, :, 1:4].mean(axis=1)
        neutrino_delta = predicted_neutrino - data["truth_neutrinos"]
        knot_nu_mse = np.square(neutrino_delta).sum(axis=-1).mean(axis=1)
        knot_nu_pt = np.hypot(predicted_neutrino[:, :, 0], predicted_neutrino[:, :, 1])
        truth_nu_pt = np.hypot(
            data["truth_neutrinos"][:, :, 0], data["truth_neutrinos"][:, :, 1]
        )
        knot_nu_pt_mse = np.square(knot_nu_pt - truth_nu_pt).mean(axis=1)
        knot_nu_pz_mse = np.square(neutrino_delta[:, :, 2]).mean(axis=1)

        assignment_difference = {
            role: event_correct[role] - vyper_event_correct[role]
            for role in ("top", "antitop", "complete")
        }
        results[label] = {
            "sample_semantics": posterior["sample_semantics"],
            "posterior_draws_per_event": int(candidates.shape[1]),
            "assignment_measure": (
                "per-event fraction of posterior assignment draws matching truth; "
                "event bootstrap, not a MAP classification"
            ),
            "assignment_MAP": assignment_map,
            "assignment_posterior_expected_accuracy": {
                role: float(event_correct[role].mean())
                for role in ("top", "antitop", "complete")
            },
            "assignment_complete_event_bootstrap_95pct_interval": bootstrap_mean_interval(
                event_correct["complete"], assignment_bootstrap
            ),
            "assignment_minus_VyPER_accuracy": {
                role: {
                    "difference": float(assignment_difference[role].mean()),
                    "event_bootstrap_95pct_interval": bootstrap_mean_interval(
                        assignment_difference[role], assignment_bootstrap
                    ),
                }
                for role in ("top", "antitop", "complete")
            },
            "neutrino_point_estimator": (
                "posterior mean of draw-level px,py,pz; each event equally weighted"
            ),
            "neutrino_3vector_RMSE_GeV": float(np.sqrt(knot_nu_mse.mean())),
            "neutrino_3vector_event_bootstrap_95pct_interval_GeV": [
                float(value)
                for value in np.quantile(
                    np.sqrt(knot_nu_mse[all_event_bootstrap].mean(axis=1)),
                    [0.025, 0.975],
                )
            ],
            "neutrino_3vector_RMSE_minus_VyPER_GeV": float(
                np.sqrt(knot_nu_mse.mean()) - np.sqrt(vyper_nu_mse.mean())
            ),
            "paired_event_bootstrap_neutrino_3vector_RMSE_difference_minus_VyPER": paired_rmse_difference(
                knot_nu_mse, vyper_nu_mse, all_event_bootstrap
            ),
            "neutrino_pT_RMSE_GeV": float(np.sqrt(knot_nu_pt_mse.mean())),
            "neutrino_pT_event_bootstrap_95pct_interval_GeV": [
                float(value)
                for value in np.quantile(
                    np.sqrt(knot_nu_pt_mse[all_event_bootstrap].mean(axis=1)),
                    [0.025, 0.975],
                )
            ],
            "neutrino_pT_RMSE_minus_VyPER_GeV": float(
                np.sqrt(knot_nu_pt_mse.mean()) - np.sqrt(vyper_nu_pt_mse.mean())
            ),
            "paired_event_bootstrap_neutrino_pT_RMSE_difference_minus_VyPER": paired_rmse_difference(
                knot_nu_pt_mse, vyper_nu_pt_mse, all_event_bootstrap
            ),
            "neutrino_pz_RMSE_GeV": float(np.sqrt(knot_nu_pz_mse.mean())),
            "neutrino_pz_event_bootstrap_95pct_interval_GeV": [
                float(value)
                for value in np.quantile(
                    np.sqrt(knot_nu_pz_mse[all_event_bootstrap].mean(axis=1)),
                    [0.025, 0.975],
                )
            ],
            "neutrino_pz_RMSE_minus_VyPER_GeV": float(
                np.sqrt(knot_nu_pz_mse.mean()) - np.sqrt(vyper_nu_pz_mse.mean())
            ),
            "paired_event_bootstrap_neutrino_pz_RMSE_difference_minus_VyPER": paired_rmse_difference(
                knot_nu_pz_mse, vyper_nu_pz_mse, all_event_bootstrap
            ),
            "max_massless_energy_residual_GeV": posterior[
                "massless_energy_max_residual_gev"
            ],
        }

    result = {
        "status": "completed",
        "dsid": 574801,
        "root_file": str(args.root_file.resolve()),
        "root_file_sha256": root_sha256,
        "measured_file": str(args.measured.resolve()),
        "truth_file": str(args.truth.resolve()),
        "event_count": len(data["event_ids"]),
        "truth_assignment_supported_events": int(data["assignment_support"].sum()),
        "ordered_EVENT_ID_sha256": event_digest(data["event_ids"]),
        "same_exact_ROOT_event_join": vyper["matched_rows"] == len(data["event_ids"]),
        "root_rows": vyper["root_rows"],
        "matched_root_rows_sha256": vyper["matched_root_rows_sha256"],
        "event_id_semantics": data["event_id_semantics"],
        "truth_contract": "partonic after-FSR top/antitop and explicit top/antitop neutrino roles",
        "topology_identity": EXPECTED_TOPOLOGY,
        "assignment_candidate_encoding": (
            "frozen charge-side topology; candidate index enumerates b-jet slots outside "
            "the two measured lepton slots, so candidate_index // 2 is the b-jet slot"
        ),
        "bootstrap": {
            "unit": "event",
            "resamples": args.bootstrap_resamples,
            "seed": BOOTSTRAP_SEED,
        },
        "VyPER": {
            "assignment": {
                role: {
                    "accuracy": float(vyper_event_correct[role].mean()),
                    "event_bootstrap_95pct_interval": bootstrap_mean_interval(
                        vyper_event_correct[role], assignment_bootstrap
                    ),
                }
                for role in ("top", "antitop", "complete")
            },
            "neutrino_3vector_RMSE_GeV": float(np.sqrt(vyper_nu_mse.mean())),
            "neutrino_3vector_event_bootstrap_95pct_interval_GeV": [
                float(value)
                for value in np.quantile(
                    np.sqrt(vyper_nu_mse[all_event_bootstrap].mean(axis=1)),
                    [0.025, 0.975],
                )
            ],
            "neutrino_pT_RMSE_GeV": float(np.sqrt(vyper_nu_pt_mse.mean())),
            "neutrino_pT_event_bootstrap_95pct_interval_GeV": [
                float(value)
                for value in np.quantile(
                    np.sqrt(vyper_nu_pt_mse[all_event_bootstrap].mean(axis=1)),
                    [0.025, 0.975],
                )
            ],
            "neutrino_pz_RMSE_GeV": float(np.sqrt(vyper_nu_pz_mse.mean())),
            "neutrino_pz_event_bootstrap_95pct_interval_GeV": [
                float(value)
                for value in np.quantile(
                    np.sqrt(vyper_nu_pz_mse[all_event_bootstrap].mean(axis=1)),
                    [0.025, 0.975],
                )
            ],
            "role_validation": vyper["assignment"]["charge_side_role_validation"],
        },
        "KNOT": results,
        "source_sha256": {
            **source_hashes,
            **{
                f"posterior_{label}": sha256_file(path)
                for label, path in posterior_paths.items()
            },
            **{
                f"assignment_diagnostics_{label}": sha256_file(path)
                for label, path in diagnostic_paths.items()
            },
        },
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "checked_at_utc": datetime.now(UTC).isoformat(),
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "event_count": result["event_count"],
                "supported_events": result["truth_assignment_supported_events"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
