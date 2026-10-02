"""Audit charge-side truth and assignment support on a bounded real JER sample."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

SOURCE = Path(
    "/mnt/iusers01/fatpou01/phy01/w00238tl/scratch/datasets/ttbar2L/"
    "combined_partonic_afterFSR_v1/raw/"
    "TtbarDilepton_TopRegression_partonic_truth_nu_2M_even.h5"
)
SPLIT_SEED = 20260924
SELECTION_SEED = 20261002
SPLIT_NAME = "event_number_splitmix64_90_5_5_v1"
QUOTAS = {0: 32, 1: 16}
CHANNELS = ("emu", "ee", "mumu")
MAX_JETS = 8
W_MASS = 80.379


def splitmix64(values, seed):
    with np.errstate(over="ignore"):
        value = values.astype(np.uint64, copy=False) ^ np.uint64(seed)
        value += np.uint64(0x9E3779B97F4A7C15)
        value = (value ^ (value >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        value = (value ^ (value >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return value ^ (value >> np.uint64(31))


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def p4(objects):
    pt = objects["pt"].astype(np.float64)
    eta = objects["eta"].astype(np.float64)
    phi = objects["phi"].astype(np.float64)
    energy = objects["e"].astype(np.float64)
    return np.stack(
        (energy, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)), axis=-1
    )


def mass(vectors):
    mass2 = vectors[..., 0] ** 2 - np.sum(vectors[..., 1:] ** 2, axis=-1)
    return np.sqrt(np.maximum(mass2, 0.0))


def choose_rows(source, event_ids, split, counts):
    selected = {split_id: {channel: [] for channel in CHANNELS} for split_id in counts}
    for split_id, quotas in counts.items():
        rows = np.flatnonzero(split == split_id)
        priority = splitmix64(event_ids[rows], SELECTION_SEED ^ split_id)
        rows = rows[np.argsort(priority, kind="stable")]
        remaining = {channel: quotas for channel in CHANNELS}
        for start in range(0, len(rows), 2048):
            candidates = rows[start : start + 2048]
            # h5py requires increasing row indices for fancy indexing. The
            # within-block source order is deterministic; block order follows
            # the event-ID hash priority.
            candidates = np.sort(candidates)
            electron = source["INPUTS/ELECTRON"][candidates]
            muon = source["INPUTS/MUON"][candidates]
            e_present = np.isfinite(electron["pt"]) & (electron["pt"] > 0)
            m_present = np.isfinite(muon["pt"]) & (muon["pt"] > 0)
            n_e, n_mu = e_present.sum(axis=1), m_present.sum(axis=1)
            channel = np.full(len(candidates), "", dtype="U4")
            channel[(n_e == 1) & (n_mu == 1)] = "emu"
            channel[(n_e == 2) & (n_mu == 0)] = "ee"
            channel[(n_e == 0) & (n_mu == 2)] = "mumu"
            for name in CHANNELS:
                if remaining[name] <= 0:
                    continue
                matches = candidates[channel == name][: remaining[name]]
                selected[split_id][name].extend(matches.tolist())
                remaining[name] -= len(matches)
            if all(value == 0 for value in remaining.values()):
                break
        if any(value for value in remaining.values()):
            raise ValueError(f"Could not fill {split_id=} channel quotas: {remaining}")
    return {
        split_id: np.asarray(sorted(row for rows in by_channel.values() for row in rows), dtype=np.int64)
        for split_id, by_channel in selected.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/knot/scientific-closure-run1/real_sample"),
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Required real JER truth source is missing: {SOURCE}")
    source_sha = sha256(SOURCE)
    with h5py.File(SOURCE, "r") as source:
        required = (
            "META/eventNumber",
            "INPUTS/ELECTRON",
            "INPUTS/MUON",
            "INPUTS/JET",
            "LABELS/ELECTRON",
            "LABELS/MUON",
            "LABELS/JET",
            "TARGETS/NEUTRINO",
            "TARGETS/TOP",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"{SOURCE}: missing required truth-contract inputs {missing}")
        event_ids = np.asarray(source["META/eventNumber"][:], dtype=np.int64)
        split_bucket = (splitmix64(event_ids, SPLIT_SEED) % 100).astype(np.uint8)
        split = np.where(split_bucket < 90, 0, np.where(split_bucket < 95, 1, 2)).astype(np.uint8)
        selected_rows = choose_rows(source, event_ids, split, QUOTAS)

        nu_ds = source["TARGETS/NEUTRINO"]
        top_ds = source["TARGETS/TOP"]
        nu_attrs = {key: text(value) for key, value in nu_ds.attrs.items()}
        top_attrs = {key: text(value) for key, value in top_ds.attrs.items()}
        expected_nu = {
            "order": "top, antitop",
            "slot_0": "neutrino from top (positive-lepton side)",
            "slot_1": "antineutrino from antitop (negative-lepton side)",
            "units": "GeV",
        }
        for key, expected in expected_nu.items():
            if nu_attrs.get(key) != expected:
                raise ValueError(f"{SOURCE}: TARGETS/NEUTRINO {key}={nu_attrs.get(key)!r}, expected {expected!r}")
        if top_attrs.get("target_source") != "partonic_after_fsr":
            raise ValueError(f"{SOURCE}: TARGETS/TOP target source is not partonic after-FSR")
        branch_map = json.loads(top_attrs["source_branch_map"])
        if branch_map.get("top_pt") != "t_afterFSR_pt_NOSYS" or branch_map.get(
            "antitop_pt"
        ) != "tbar_afterFSR_pt_NOSYS":
            raise ValueError(f"{SOURCE}: TARGETS/TOP branch map is not top/antitop authoritative")
        lepton_label_convention = text(source["LABELS/ELECTRON"].attrs["label_convention"])
        if "1=top b or positive charged lepton" not in lepton_label_convention or "2=antitop bbar or negative charged lepton" not in lepton_label_convention:
            raise ValueError(f"{SOURCE}: source lepton-label convention is ambiguous")

        selected_ids = {}
        selected_split = {}
        details = {}
        all_rows = []
        for split_id, rows in selected_rows.items():
            ids = event_ids[rows]
            selected_ids[split_id] = ids
            selected_split[split_id] = np.full(len(rows), split_id, dtype=np.uint8)
            all_rows.extend(rows.tolist())
            electrons = source["INPUTS/ELECTRON"][rows]
            muons = source["INPUTS/MUON"][rows]
            e_labels = source["LABELS/ELECTRON"][rows]
            m_labels = source["LABELS/MUON"][rows]
            jets = source["INPUTS/JET"][rows, :MAX_JETS]
            jet_labels = source["LABELS/JET"][rows, :MAX_JETS]
            all_jet_labels = source["LABELS/JET"][rows]
            neutrinos = source["TARGETS/NEUTRINO"][rows]
            parents = source["TARGETS/TOP"][rows]
            e_present = np.isfinite(electrons["pt"]) & (electrons["pt"] > 0)
            m_present = np.isfinite(muons["pt"]) & (muons["pt"] > 0)
            present = np.concatenate((e_present, m_present), axis=1)
            charge = np.concatenate((electrons["charge"], muons["charge"]), axis=1)
            labels = np.concatenate((e_labels, m_labels), axis=1)
            positive = present & (charge > 0)
            negative = present & (charge < 0)
            charge_label_ok = (
                (positive.sum(axis=1) == 1)
                & (negative.sum(axis=1) == 1)
                & np.all(labels[positive] == 1.0)
                & np.all(labels[negative] == 2.0)
            )
            if not charge_label_ok.all():
                bad = ids[np.flatnonzero(~charge_label_ok)[:5]].tolist()
                raise ValueError(f"Charge and source truth labels disagree: EVENT_ID={bad}")
            if not np.isfinite(np.stack([neutrinos[name] for name in ("px", "py", "pz")], axis=-1)).all():
                raise ValueError("Selected producer-declared top/antitop neutrino truth is non-finite")
            if not np.isfinite(np.stack([parents[name] for name in ("top_pt", "top_eta", "top_phi", "top_m", "antitop_pt", "antitop_eta", "antitop_phi", "antitop_m")], axis=-1)).all():
                raise ValueError("Selected partonic top/antitop targets are non-finite")

            lepton_p4 = np.concatenate((p4(electrons), p4(muons)), axis=1)
            role_vectors = []
            for role_label in (1.0, 2.0):
                role = present & (labels == role_label)
                index = role.argmax(axis=1)
                role_vectors.append(lepton_p4[np.arange(len(rows)), index])
            positive_lepton, negative_lepton = role_vectors
            nu_momentum = np.stack([neutrinos[name] for name in ("px", "py", "pz")], axis=-1).astype(np.float64)
            nu_energy = np.linalg.norm(nu_momentum, axis=-1)
            nu_p4 = np.concatenate((nu_energy[..., None], nu_momentum), axis=-1)
            w_direct = np.stack((mass(positive_lepton + nu_p4[:, 0]), mass(negative_lepton + nu_p4[:, 1])), axis=1)
            w_swapped = np.stack((mass(positive_lepton + nu_p4[:, 1]), mass(negative_lepton + nu_p4[:, 0])), axis=1)
            direct_cost = np.abs(w_direct - W_MASS).sum(axis=1)
            swapped_cost = np.abs(w_swapped - W_MASS).sum(axis=1)

            count_all = np.stack(
                ((all_jet_labels == 1.0).sum(axis=1), (all_jet_labels == 2.0).sum(axis=1)), axis=1
            )
            count_kept = np.stack(
                ((jet_labels == 1.0).sum(axis=1), (jet_labels == 2.0).sum(axis=1)), axis=1
            )
            support = np.all(count_kept == 1, axis=1)
            causes = {
                "candidate_truncation": int((~support & np.all(count_all == 1, axis=1)).sum()),
                "NULL_or_missing_role": int((~support & ~np.all(count_all == 1, axis=1)).sum()),
                "jet_multiplicity": int((~support & (np.isfinite(jets["pt"]).sum(axis=1) < 2)).sum()),
                "other": 0,
            }
            cause_sum = causes["candidate_truncation"] + causes["NULL_or_missing_role"]
            causes["other"] = int((~support).sum()) - cause_sum

            channel_values = np.full(len(rows), "", dtype="U4")
            n_e, n_mu = e_present.sum(axis=1), m_present.sum(axis=1)
            channel_values[(n_e == 1) & (n_mu == 1)] = "emu"
            channel_values[(n_e == 2) & (n_mu == 0)] = "ee"
            channel_values[(n_e == 0) & (n_mu == 2)] = "mumu"
            details[str(split_id)] = {}
            for name in CHANNELS:
                mask = channel_values == name
                details[str(split_id)][name] = {
                    "events": int(mask.sum()),
                    "charge_side_labels_valid": bool(charge_label_ok[mask].all()),
                    "nu_top_slot_finite": bool(np.isfinite(nu_momentum[mask, 0]).all()),
                    "antinu_antitop_slot_finite": bool(np.isfinite(nu_momentum[mask, 1]).all()),
                    "W_mass_positive_top_side_median_GeV": float(np.median(w_direct[mask, 0])),
                    "W_mass_negative_antitop_side_median_GeV": float(np.median(w_direct[mask, 1])),
                    "direct_charge_side_pairing_lower_W_cost_fraction": float(np.mean(direct_cost[mask] < swapped_cost[mask])),
                    "truth_assignment_supported": int(support[mask].sum()),
                    "truth_assignment_unsupported": int((~support[mask]).sum()),
                }
                # Recount causes in this channel to preserve the support breakdown.
                channel_causes = {
                    "candidate_truncation": int((~support[mask] & np.all(count_all[mask] == 1, axis=1)).sum()),
                    "NULL_or_missing_role": int((~support[mask] & ~np.all(count_all[mask] == 1, axis=1)).sum()),
                    "jet_multiplicity": int((~support[mask] & (np.isfinite(jets["pt"][mask]).sum(axis=1) < 2)).sum()),
                }
                channel_causes["other"] = int((~support[mask]).sum()) - channel_causes["candidate_truncation"] - channel_causes["NULL_or_missing_role"]
                details[str(split_id)][name]["unsupported_causes"] = channel_causes
            details[str(split_id)]["support"] = {
                "events": len(rows),
                "supported": int(support.sum()),
                "unsupported": int((~support).sum()),
                "fraction": float(support.mean()),
                "unsupported_causes": causes,
            }

        rows_all = np.asarray(sorted(all_rows), dtype=np.int64)
        ids_all = event_ids[rows_all]
        split_all = split[rows_all]
        truth_neutrinos = source["TARGETS/NEUTRINO"][rows_all]
        finite_support = np.isfinite(
            np.stack([truth_neutrinos[name] for name in ("px", "py", "pz")], axis=-1)
        )
        role_support = finite_support.all(axis=-1)

    # The files in this directory are a bounded diagnostic slice, not a production manifest.
    source_manifest = {
        "kind": "bounded_scientific_closure_input",
        "new_source": {"path": str(SOURCE), "sha256": source_sha},
        "split": {"version": SPLIT_NAME, "seed": SPLIT_SEED, "reserved_test_materialized": False},
        "selection": {"seed": SELECTION_SEED, "per_split_per_channel": QUOTAS},
    }
    (output_dir / "source_manifest_v1.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    np.savez_compressed(
        output_dir / "split_membership_v1.npz",
        source_row=rows_all,
        event_id=ids_all,
        split=split_all,
    )
    (output_dir / "splits").mkdir()
    np.savez_compressed(
        output_dir / "splits/neutrino_role_support_train_validation_realnu_v1.npz",
        source_row=rows_all,
        event_id=ids_all,
        support=role_support,
    )
    np.savez_compressed(
        output_dir / "cpu_development_ids_v1.npz",
        train_source_row=selected_rows[0],
        train_event_id=selected_ids[0],
        validation_source_row=selected_rows[1],
        validation_event_id=selected_ids[1],
    )
    total_events = len(rows_all)
    total_supported = sum(int(details[str(split_id)]["support"]["supported"]) for split_id in QUOTAS)
    result = {
        "source": str(SOURCE),
        "source_sha256": source_sha,
        "source_events": len(event_ids),
        "source_neutrino_role_contract": expected_nu,
        "source_neutrino_branches": nu_attrs.get("source_branches"),
        "source_parent_branch_map": branch_map,
        "source_parent_truth_definition": top_attrs.get("target_source"),
        "source_lepton_label_convention": lepton_label_convention,
        "split": SPLIT_NAME,
        "split_seed": SPLIT_SEED,
        "events_checked": total_events,
        "train_events_checked": len(selected_rows[0]),
        "validation_events_checked": len(selected_rows[1]),
        "reserved_test_used": False,
        "truth_assignment_supported": total_supported,
        "truth_assignment_unsupported": total_events - total_supported,
        "truth_assignment_support_fraction": total_supported / total_events,
        "by_split_and_channel": details,
        "output_dir": str(output_dir),
    }
    (output_dir / "truth_contract.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
