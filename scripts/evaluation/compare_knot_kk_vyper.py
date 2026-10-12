#!/usr/bin/env python3
"""Compare frozen KNOT parents with stored VyPER top/tbar vectors on exact KK IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import uproot

SIDES = ("top", "antitop")
BOOTSTRAPS = 1000
BOOTSTRAP_SEED = 42011
ROOT_PARENTS = {
    "top": "TtbarDiLepton_VyPER_Top_NOSYS",
    "antitop": "TtbarDiLepton_VyPER_Tbar_NOSYS",
}


def mass(p4: np.ndarray, label: str) -> np.ndarray:
    mass2 = p4[..., 0] ** 2 - np.square(p4[..., 1:]).sum(axis=-1)
    if not np.isfinite(mass2).all():
        raise FloatingPointError(f"{label} has non-finite invariant mass squared")
    if np.any(mass2 < -1.0e-5):
        first = np.argwhere(mass2 < -1.0e-5)[0].tolist()
        raise ValueError(f"{label} has a spacelike vector at index {first}")
    return np.sqrt(np.maximum(mass2, 0.0))


def eta(p4: np.ndarray, label: str) -> np.ndarray:
    pt = np.hypot(p4[..., 1], p4[..., 2])
    if not np.isfinite(pt).all() or np.any(pt <= 0):
        raise ValueError(f"{label} has non-finite or zero pT")
    return np.arcsinh(p4[..., 3] / pt)


def event_digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def root_four_vector(tree, branch: str, entries: np.ndarray) -> np.ndarray:
    coordinates = {}
    for coordinate in ("fPt", "fEta", "fPhi", "fM"):
        path = f"{branch}/fCoordinates/fCoordinates.{coordinate}"
        if path not in tree:
            raise KeyError(f"Stored VyPER p4 branch is absent: {path}")
        coordinates[coordinate] = np.asarray(
            tree[path].array(entry_start=0, entry_stop=tree.num_entries, library="np"),
            dtype=np.float64,
        )[entries]
    pt = coordinates["fPt"] * 1.0e-3
    pseudorapidity = coordinates["fEta"]
    phi = coordinates["fPhi"]
    mass_gev = coordinates["fM"] * 1.0e-3
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    pz = pt * np.sinh(pseudorapidity)
    energy = np.sqrt(np.square(mass_gev) + np.square(pt * np.cosh(pseudorapidity)))
    p4 = np.stack((energy, px, py, pz), axis=-1)
    if not np.isfinite(p4).all():
        raise FloatingPointError(f"Stored VyPER {branch} contains non-finite four-vectors")
    return p4


def load_vyper(root_file: Path, dsid: int, expected_ids: np.ndarray) -> dict:
    with uproot.open(root_file) as source:
        if "reco" not in source:
            raise KeyError(f"ROOT file lacks required reco tree: {root_file}")
        tree = source["reco"]
        for name in ("eventNumber", "mcChannelNumber"):
            if name not in tree:
                raise KeyError(f"ROOT tree lacks required event identity branch {name}")
        event_number = np.asarray(
            tree["eventNumber"].array(library="np"), dtype=np.int64
        )
        channels = np.asarray(
            tree["mcChannelNumber"].array(library="np"), dtype=np.int64
        )
        if len(np.unique(event_number)) != len(event_number):
            raise ValueError(f"ROOT eventNumber is not unique in {root_file}")
        if not np.all(channels == dsid):
            raise ValueError(f"ROOT mcChannelNumber does not equal requested DSID {dsid}")
        root_ids = (np.int64(dsid) << 32) | event_number
        if len(np.unique(root_ids)) != len(root_ids):
            raise ValueError(f"Stable ROOT EVENT_ID is not unique in {root_file}")
        row_by_id = {int(value): index for index, value in enumerate(root_ids)}
        missing = [int(value) for value in expected_ids if int(value) not in row_by_id]
        if missing:
            raise ValueError(
                f"{len(missing)}/{len(expected_ids)} KNOT transfer IDs are absent from {root_file}; "
                f"first missing IDs={missing[:5]}"
            )
        rows = np.asarray([row_by_id[int(value)] for value in expected_ids], dtype=np.int64)
        root_entries = int(tree.num_entries)
        vectors = {
            side: root_four_vector(tree, branch, rows)
            for side, branch in ROOT_PARENTS.items()
        }
    return {
        "parent_samples": np.stack((vectors["top"], vectors["antitop"]), axis=1)[:, None],
        "root_entries": root_entries,
        "matched_entries": len(rows),
        "matched_ROOT_row_sha256": hashlib.sha256(
            np.asarray(rows, dtype="<i8").tobytes()
        ).hexdigest(),
        "matched_EVENT_ID_sha256": event_digest(expected_ids),
    }


def load_knot(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen KNOT transfer posterior is absent: {path}")
    with h5py.File(path, "r") as source:
        required = ("EVENT_ID", "parents")
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"{path} lacks required posterior fields: {missing}")
        if source.attrs.get("sealed_test_used") not in (None, False, 0):
            raise ValueError(f"{path} is not marked sealed_test_used=false")
        ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        parents = np.asarray(source["parents"][:], dtype=np.float64)
        roles = json.loads(source.attrs.get("parent_roles", "[]"))
    if roles != ["top", "antitop"]:
        raise ValueError(f"{path} parent role order is not top,antitop: {roles}")
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path} EVENT_ID must be one-dimensional and unique")
    if parents.ndim != 4 or parents.shape[0] != len(ids) or parents.shape[2:] != (2, 4):
        raise ValueError(f"{path} posterior parent shape is invalid: {parents.shape}")
    if not np.isfinite(parents).all():
        raise FloatingPointError(f"{path} contains non-finite posterior parent vectors")
    return {
        "label": label,
        "path": path,
        "ids": ids,
        "parents": parents,
    }


def load_truth(path: Path, expected_ids: np.ndarray) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required after-FSR truth sidecar is absent: {path}")
    with h5py.File(path, "r") as source:
        if "EVENT_ID" not in source or "TRUTH/parents" not in source:
            raise KeyError(f"{path} lacks EVENT_ID or TRUTH/parents")
        truth_definition = source.attrs.get("parent_truth_definition", "")
        role_order = json.loads(source.attrs.get("partonic_roles", "[]"))
        if truth_definition != "Ttbar_MC_t_afterFSR and Ttbar_MC_tbar_afterFSR":
            raise ValueError(
                f"{path} does not declare the required after-FSR top/antitop truth: "
                f"{truth_definition!r}"
            )
        if role_order != ["top", "antitop"]:
            raise ValueError(f"{path} truth role order is not top,antitop: {role_order}")
        ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        truth = np.asarray(source["TRUTH/parents"][:], dtype=np.float64)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path} EVENT_ID must be one-dimensional and unique")
    if not np.array_equal(ids, expected_ids):
        raise ValueError(f"{path} truth EVENT_ID order differs from the posterior")
    if truth.shape != (len(ids), 2, 4):
        raise ValueError(f"{path} after-FSR truth parent shape is invalid: {truth.shape}")
    if not np.isfinite(truth).all():
        raise FloatingPointError(f"{path} contains non-finite after-FSR truth vectors")
    return truth


def summarize_prediction(
    samples: np.ndarray, truth: np.ndarray, bootstrap_indices: np.ndarray, label: str
) -> tuple[dict, dict]:
    truth_mtt = mass(truth.sum(axis=1), "after-FSR truth mtt")
    draw_mtt = mass(samples.sum(axis=2), f"{label} parent-pair draws")
    point_mtt = draw_mtt.mean(axis=1)
    mtt_error = point_mtt - truth_mtt
    parent_pt_error = {}
    parent_eta_error = {}
    for side_index, side in enumerate(SIDES):
        sampled_pt = np.hypot(
            samples[:, :, side_index, 1], samples[:, :, side_index, 2]
        ).mean(axis=1)
        truth_pt = np.hypot(truth[:, side_index, 1], truth[:, side_index, 2])
        parent_pt_error[side] = sampled_pt - truth_pt
        sampled_eta = eta(samples[:, :, side_index], f"{label} {side} draws").mean(axis=1)
        truth_eta = eta(truth[:, side_index], f"after-FSR truth {side}")
        parent_eta_error[side] = sampled_eta - truth_eta

    order = np.argsort(truth_mtt, kind="stable")
    differential = []
    for bin_index, rows in enumerate(np.array_split(order, 10)):
        error = mtt_error[rows]
        differential.append(
            {
                "truth_decile": bin_index + 1,
                "events": len(rows),
                "truth_mtt_mean_GeV": float(truth_mtt[rows].mean()),
                "mtt_bias_GeV": float(error.mean()),
                "mtt_RMSE_GeV": float(np.sqrt(np.square(error).mean())),
            }
        )

    sampled_mtt, truth_mtt_boot = mtt_error[bootstrap_indices], truth_mtt[bootstrap_indices]
    slope_boot = np.empty(len(bootstrap_indices), dtype=np.float64)
    for index, (x, y) in enumerate(zip(truth_mtt_boot, sampled_mtt + truth_mtt_boot, strict=True)):
        centered_x, centered_y = x - x.mean(), y - y.mean()
        slope_boot[index] = np.dot(centered_x, centered_y) / np.dot(centered_x, centered_x)

    def intervals(error: np.ndarray) -> dict:
        sampled = error[bootstrap_indices]
        return {
            "bias": [
                float(value)
                for value in np.quantile(sampled.mean(axis=1), [0.025, 0.975])
            ],
            "RMSE": [
                float(value)
                for value in np.quantile(
                    np.sqrt(np.square(sampled).mean(axis=1)), [0.025, 0.975]
                )
            ],
        }

    response_slope = float(np.polyfit(truth_mtt, point_mtt, 1)[0])
    summary = {
        "events": len(truth_mtt),
        "posterior_draws_per_event": int(samples.shape[1]),
        "mtt": {
            "bias_GeV": float(mtt_error.mean()),
            "RMSE_GeV": float(np.sqrt(np.square(mtt_error).mean())),
            "event_bootstrap_95pct_interval": intervals(mtt_error),
            "truth_to_estimate_response_slope": response_slope,
            "event_bootstrap_95pct_interval_for_slope": [
                float(value) for value in np.quantile(slope_boot, [0.025, 0.975])
            ],
            "bias_and_RMSE_by_truth_mtt_decile": differential,
        },
        "parents": {
            side: {
                "pT_bias_GeV": float(parent_pt_error[side].mean()),
                "pT_RMSE_GeV": float(np.sqrt(np.square(parent_pt_error[side]).mean())),
                "pT_event_bootstrap_95pct_interval": intervals(parent_pt_error[side]),
                "eta_bias": float(parent_eta_error[side].mean()),
                "eta_RMSE": float(np.sqrt(np.square(parent_eta_error[side]).mean())),
            }
            for side in SIDES
        },
    }
    errors = {"mtt": mtt_error}
    errors.update({f"parent_pT_{side}": error for side, error in parent_pt_error.items()})
    return summary, errors


def paired_rmse_difference(
    first: np.ndarray, second: np.ndarray, bootstrap_indices: np.ndarray
) -> dict:
    left, right = first[bootstrap_indices], second[bootstrap_indices]
    difference = np.sqrt(np.square(left).mean(axis=1)) - np.sqrt(
        np.square(right).mean(axis=1)
    )
    return {
        "first_minus_second_RMSE_GeV": float(
            np.sqrt(np.square(first).mean()) - np.sqrt(np.square(second).mean())
        ),
        "event_bootstrap_95pct_interval_GeV": [
            float(value) for value in np.quantile(difference, [0.025, 0.975])
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-file", type=Path, required=True)
    parser.add_argument("--dsid", type=int, required=True)
    parser.add_argument("--truth-file", type=Path, required=True)
    parser.add_argument(
        "--posterior",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="one frozen KNOT M/C/J transfer posterior; may be repeated",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAPS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    if args.dsid != 574801:
        parser.error("This audited stored-VyPER ROOT contract is only established for DSID 574801")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite stored-VyPER comparison: {output}")
    records = {}
    for specification in args.posterior:
        label, separator, path = specification.partition("=")
        if not separator or label not in {"M", "C", "J"} or label in records:
            parser.error(f"Invalid or duplicate --posterior specification: {specification}")
        records[label] = load_knot(Path(path).resolve(), label)
    if not records:
        parser.error("At least one frozen KNOT posterior is required")
    first_label = next(iter(records))
    ids = records[first_label]["ids"]
    for label, record in records.items():
        if not np.array_equal(ids, record["ids"]):
            raise ValueError(f"KNOT-{label} EVENT_ID order differs from KNOT-{first_label}")
    truth = load_truth(args.truth_file.resolve(), ids)
    vyper = load_vyper(args.root_file.resolve(), args.dsid, ids)
    bootstrap_indices = np.random.default_rng(args.bootstrap_seed).integers(
        0, len(ids), size=(args.bootstrap_resamples, len(ids))
    )
    root_summary, root_errors = summarize_prediction(
        vyper["parent_samples"], truth, bootstrap_indices, "stored VyPER"
    )
    summaries, event_errors = {}, {}
    for label, record in records.items():
        summaries[label], event_errors[label] = summarize_prediction(
            record["parents"], truth, bootstrap_indices, f"KNOT-{label}"
        )
    summaries["VyPER"] = root_summary
    event_errors["VyPER"] = root_errors

    paired = {}
    for label in records:
        paired[label] = {
            "unit": "paired EVENT_ID bootstrap; posterior draws remain fixed within each event",
            "mtt_RMSE_vs_VyPER": paired_rmse_difference(
                event_errors[label]["mtt"], root_errors["mtt"], bootstrap_indices
            ),
            "parent_pT_RMSE_vs_VyPER": {
                side: paired_rmse_difference(
                    event_errors[label][f"parent_pT_{side}"],
                    root_errors[f"parent_pT_{side}"],
                    bootstrap_indices,
                )
                for side in SIDES
            },
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "completed",
        "dsid": args.dsid,
        "root_file": str(args.root_file.resolve()),
        "root_file_sha256": hashlib.sha256(args.root_file.read_bytes()).hexdigest(),
        "truth_file": str(args.truth_file.resolve()),
        "truth_file_sha256": hashlib.sha256(args.truth_file.read_bytes()).hexdigest(),
        "events": len(ids),
        "ordered_EVENT_ID_sha256": event_digest(ids),
        "same_exact_event_join": True,
        "truth_contract": "partonic after-FSR top/antitop; charge-side truth inherited from the KNOT transfer posterior",
        "stored_VyPER_contract": {
            "parent_vectors": ROOT_PARENTS,
            "units_in_ROOT": "MeV pT and mass; eta and phi dimensionless/radians",
            "units_in_metrics": "GeV",
            "assignment_and_neutrino_comparison": "unavailable in this report; the raw Top1/Top2 neutrino/ID branches were not used as charge-side predictions",
        },
        "bootstrap": {
            "unit": "event",
            "replicates": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
        },
        "metrics": summaries,
        "paired_KNOT_minus_VyPER": paired,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "completed", "events": len(ids), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
