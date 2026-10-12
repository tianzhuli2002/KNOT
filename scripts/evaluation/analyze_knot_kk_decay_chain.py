#!/usr/bin/env python3
"""Measure decay-chain consistency and stratify it by truth mtt after inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from scripts.run_knot import FREEZE_PATH

ROOT = Path(__file__).resolve().parents[2]
ROLES = ("top", "antitop")
MODELS = ("M", "C", "J")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_id_hash(event_ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(event_ids, dtype="<i8").tobytes()).hexdigest()


def wrapped_delta_phi(delta: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(delta), np.cos(delta))


def invariant_mass(p4: np.ndarray, name: str) -> np.ndarray:
    mass2 = p4[..., 0] ** 2 - np.square(p4[..., 1:]).sum(axis=-1)
    if not np.isfinite(mass2).all() or np.any(mass2 < -1e-4):
        first = np.argwhere(~np.isfinite(mass2) | (mass2 < -1e-4))[0].tolist()
        raise ValueError(f"{name} has a non-finite or spacelike vector at index {first}")
    return np.sqrt(np.maximum(mass2, 0.0))


def transverse_momentum(p4: np.ndarray, name: str) -> np.ndarray:
    result = np.hypot(p4[..., 1], p4[..., 2])
    if not np.isfinite(result).all() or np.any(result <= 0):
        first = np.argwhere(~np.isfinite(result) | (result <= 0))[0].tolist()
        raise ValueError(f"{name} has non-finite or zero pT at index {first}")
    return result


def eta(p4: np.ndarray, name: str) -> np.ndarray:
    return np.arcsinh(p4[..., 3] / transverse_momentum(p4, name))


def read_posterior(path: Path, model: str) -> tuple[dict, dict]:
    manifest_path = path.parent / "evaluation_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {model} posterior or evaluation manifest: {path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("variant") != model or manifest.get("truth_used_during_inference") is not False:
        raise ValueError(f"{model} transfer manifest does not verify truth-free inference: {manifest_path}")
    with h5py.File(path, "r") as source:
        required = (
            "EVENT_ID",
            "assignment_candidates",
            "assignment_support",
            "invisible",
            "invisible_support",
            "parents",
            "parent_support",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"{path} is missing posterior fields: {missing}")
        values = {name: source[name][:] for name in required}
        attrs = {
            name: source.attrs[name]
            for name in ("assignment_nodes", "invisible_roles", "parent_roles", "topology_identity")
            if name in source.attrs
        }
    for name in ("assignment_nodes", "invisible_roles", "parent_roles", "topology_identity"):
        if name not in attrs:
            raise KeyError(f"{path} is missing required posterior attribute {name}")
    decoded_attrs = {
        name: json.loads(value) if name.endswith("roles") or name == "assignment_nodes" else str(value)
        for name, value in attrs.items()
    }
    if decoded_attrs["parent_roles"] != list(ROLES):
        raise ValueError(f"Unexpected parent role order in {path}: {decoded_attrs['parent_roles']}")
    if decoded_attrs["invisible_roles"] != ["top.nu", "antitop.nu"]:
        raise ValueError(f"Unexpected neutrino role order in {path}: {decoded_attrs['invisible_roles']}")
    if set(decoded_attrs["assignment_nodes"]) != set(ROLES):
        raise ValueError(f"Unexpected assignment node order in {path}: {decoded_attrs['assignment_nodes']}")
    return values | {"attrs": decoded_attrs}, manifest


def event_bootstrap(values: np.ndarray, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        rows = rng.integers(0, len(values), size=len(values))
        samples[index] = values[rows].mean()
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def event_bootstrap_rmse(event_rmse: np.ndarray, seed: int, replicates: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        rows = rng.integers(0, len(event_rmse), size=len(event_rmse))
        samples[index] = np.sqrt(np.mean(np.square(event_rmse[rows])))
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def residuals(
    b_vectors: np.ndarray,
    neutrinos: np.ndarray,
    parents: np.ndarray,
    leptons: np.ndarray,
    label: str,
) -> dict[str, np.ndarray]:
    chain = b_vectors + neutrinos + leptons[:, None]
    parent_pt = transverse_momentum(parents, f"{label} parent vectors")
    chain_pt = transverse_momentum(chain, f"{label} daughter-chain vectors")
    delta_phi = wrapped_delta_phi(
        np.arctan2(chain[..., 2], chain[..., 1])
        - np.arctan2(parents[..., 2], parents[..., 1])
    )
    delta_r = np.hypot(
        eta(chain, f"{label} daughter-chain vectors")
        - eta(parents, f"{label} parent vectors"),
        delta_phi,
    )
    return {
        "delta_pT_GeV": chain_pt - parent_pt,
        "delta_mass_GeV": invariant_mass(chain, f"{label} daughter-chain vectors")
        - invariant_mass(parents, f"{label} parent vectors"),
        "delta_R": delta_r,
    }


def event_rmse(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=(1, 2)))


def summarize_residuals(values: dict[str, np.ndarray], seed: int, replicates: int) -> dict:
    result = {}
    for offset, (name, value) in enumerate(values.items()):
        per_event = event_rmse(value)
        result[name] = {
            "RMSE": float(np.sqrt(np.mean(np.square(value)))),
            "event_bootstrap_95pct_interval_for_RMSE": event_bootstrap_rmse(
                per_event, seed + offset, replicates
            ),
            "mean_absolute_residual": float(np.mean(np.abs(value))),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Truth-free measured HDF5")
    parser.add_argument("--truth", type=Path, required=True, help="Separate after-FSR truth sidecar")
    parser.add_argument("--posterior", action="append", required=True, metavar="MODEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None, help="Frozen KNOT-J experiment config")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42011)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    if args.batch_size <= 0 or args.bootstrap_replicates < 100:
        parser.error("Require positive batch size and at least 100 event-bootstrap replicates")
    if not args.input.is_file() or not args.truth.is_file():
        parser.error(f"Measured input or truth sidecar does not exist: {args.input}, {args.truth}")

    torch.set_num_threads(1)
    freeze = json.loads(FREEZE_PATH.read_text())
    config_path = args.config or ROOT / freeze["configs"]["J"]["experiment"]
    if not config_path.is_file():
        parser.error(f"Frozen KNOT experiment config does not exist: {config_path}")
    parsed: dict[str, Path] = {}
    for item in args.posterior:
        model, separator, value = item.partition("=")
        if not separator or model not in MODELS or model in parsed:
            parser.error(f"Invalid or duplicate posterior specification: {item}")
        parsed[model] = Path(value)
    if set(parsed) != set(MODELS):
        parser.error("Supply exactly one saved transfer posterior for M, C, and J")

    posterior_data: dict[str, dict] = {}
    manifests: dict[str, dict] = {}
    common_ids: np.ndarray | None = None
    common_candidates = None
    draws = None
    topology_identity = None
    for model in MODELS:
        data, manifest = read_posterior(parsed[model], model)
        ids = np.asarray(data["EVENT_ID"], dtype=np.int64)
        if ids.ndim != 1 or len(np.unique(ids)) != len(ids):
            raise ValueError(f"{model} posterior EVENT_ID must be unique and one-dimensional")
        if common_ids is not None and not np.array_equal(ids, common_ids):
            raise ValueError("M/C/J posterior EVENT_ID order differs")
        if data["assignment_candidates"].ndim != 3 or data["assignment_candidates"].shape[2] != 2:
            raise ValueError(f"{model} assignment candidate shape is invalid")
        if draws is not None and data["assignment_candidates"].shape[1] != draws:
            raise ValueError("M/C/J posterior sample counts differ")
        if common_candidates is not None and data["assignment_candidates"].shape != common_candidates.shape:
            raise ValueError("M/C/J assignment candidate dimensions differ")
        if topology_identity is not None and data["attrs"]["topology_identity"] != topology_identity:
            raise ValueError("M/C/J topology identities differ")
        common_ids = ids
        common_candidates = data["assignment_candidates"] if common_candidates is None else common_candidates
        draws = int(data["assignment_candidates"].shape[1])
        topology_identity = data["attrs"]["topology_identity"]
        for field in ("assignment_support", "invisible_support", "parent_support"):
            if data[field].shape != (len(ids), draws, 2) or not np.asarray(data[field], dtype=bool).all():
                raise ValueError(f"{model} {field} is incomplete for the saved transfer draws")
        for field in ("invisible", "parents"):
            values = np.asarray(data[field])
            if values.shape != (len(ids), draws, 2, 4) or not np.isfinite(values).all():
                raise ValueError(f"{model} {field} has invalid shape or non-finite values")
        posterior_data[model] = data
        manifests[model] = manifest

    with h5py.File(args.input, "r") as measured:
        if not bool(measured.attrs.get("measured_only", False)):
            raise ValueError(f"Transfer input is not marked measured-only: {args.input}")
        if any(name in measured for name in ("TRUTH", "LABELS", "SUPERVISION")):
            raise ValueError(f"Truth-like group appears in measured input: {args.input}")
        if measured.attrs.get("kinematic_units") != "GeV" or measured.attrs.get("four_vector_order") != "E,px,py,pz":
            raise ValueError(f"Measured-input units or four-vector order are invalid: {args.input}")
        if measured.attrs.get("lepton_slot_order") != "positive charge (top), negative charge (antitop)":
            raise ValueError(f"Measured lepton slots do not follow top/antitop charge order: {args.input}")
        measured_ids = np.asarray(measured["EVENT_ID"][:], dtype=np.int64)
        measured_leptons = np.asarray(measured["INPUTS/LEPTON"][:])
        measured_charges = np.asarray(measured_leptons["charge"], dtype=np.float64)
        if measured_charges.shape != (len(measured_ids), 2) or not (
            (measured_charges[:, 0] > 0).all() and (measured_charges[:, 1] < 0).all()
        ):
            raise ValueError("Measured leptons do not follow positive-charge then negative-charge order")
        if not np.array_equal(measured_ids, common_ids):
            raise ValueError("Measured input and M/C/J posterior EVENT_ID order differs")

    config = load_experiment(config_path)
    config.data.inference_input = str(args.input.expanduser().resolve())
    config.data.tensor_cache = None
    visible = load_visible_config(config)
    visible.dataset.cache.enabled = False
    visible.dataset.cache.path = None
    data = build_data(config, visible, include_truth=False, device="cpu", complete_assignments=True)
    data.setup("predict")
    selected_ids = data._selected_ids("validation", np.arange(len(data.val_dataset), dtype=np.int64))
    if not np.array_equal(selected_ids, common_ids):
        raise ValueError("KNOT reconstructed measured-token EVENT_ID order differs from posterior files")
    if data.structured_collator.spec.signature != topology_identity:
        raise ValueError("KNOT measured-token topology differs from the frozen transfer posterior")

    loader = DataLoader(
        Subset(data.val_dataset, list(range(len(data.val_dataset)))),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data.structured_collator,
        num_workers=0,
    )
    leptons = np.empty((len(common_ids), 2, 4), dtype=np.float64)
    b_vectors = {model: np.empty((len(common_ids), draws, 2, 4), dtype=np.float64) for model in MODELS}
    b_token_indices = {
        model: np.empty((len(common_ids), draws, 2), dtype=np.int64) for model in MODELS
    }
    lepton_positions = None
    cursor = 0
    node_orders = {model: posterior_data[model]["attrs"]["assignment_nodes"] for model in MODELS}
    if any(order != node_orders["M"] for order in node_orders.values()):
        raise ValueError("M/C/J assignment-node orders differ")

    for batch_index, batch in enumerate(loader):
        batch_ids = selected_ids[cursor : cursor + len(batch["event_index"])]
        if len(batch_ids) != len(batch["event_index"]):
            raise RuntimeError("Measured-token batch exceeds the saved transfer rows")
        if lepton_positions is None:
            lepton_positions = (
                batch["type_ids"][0] == data.schema.source_type_ids["LEPTON"]
            ).nonzero(as_tuple=False).flatten()
            if len(lepton_positions) != 2:
                raise ValueError(f"Expected two measured lepton tokens, found {len(lepton_positions)}")
        if not torch.all(
            batch["type_ids"][:, lepton_positions] == data.schema.source_type_ids["LEPTON"]
        ):
            raise ValueError("Lepton-token positions change within the measured transfer batch")
        batch_leptons = batch["physics_vectors"][:, lepton_positions].detach().cpu().numpy()
        if not np.isfinite(batch_leptons).all():
            raise ValueError(f"Non-finite measured lepton four-vector near EVENT_ID={int(batch_ids[0])}")
        measured_batch = measured_leptons[cursor : cursor + len(batch_ids)]
        expected_leptons = np.stack(
            (
                measured_batch["e"],
                measured_batch["pt"] * np.cos(measured_batch["phi"]),
                measured_batch["pt"] * np.sin(measured_batch["phi"]),
                measured_batch["pt"] * np.sinh(measured_batch["eta"]),
            ),
            axis=-1,
        )
        if not np.allclose(batch_leptons, expected_leptons, rtol=2e-5, atol=1e-3):
            raise ValueError(f"Measured lepton token vectors/order differ from INPUTS/LEPTON near EVENT_ID={int(batch_ids[0])}")
        leptons[cursor : cursor + len(batch_ids)] = batch_leptons
        physics = batch["physics_vectors"]
        batch_rows = torch.arange(len(batch_ids), dtype=torch.long)

        for model in MODELS:
            candidates = posterior_data[model]["assignment_candidates"][cursor : cursor + len(batch_ids)]
            for role_index, role in enumerate(ROLES):
                node_index = node_orders[model].index(role)
                role_candidates = torch.as_tensor(candidates[:, :, node_index], dtype=torch.long)
                pool = batch["structured"]["pools"][role]
                candidate_table = pool["leaf_indices"][f"{role}.b"]
                if candidate_table.ndim != 2 or candidate_table.shape[1] != 1:
                    raise ValueError(f"{role}.b candidate table has unexpected shape {tuple(candidate_table.shape)}")
                if torch.any(role_candidates < 0) or torch.any(role_candidates >= len(candidate_table)):
                    raise ValueError(f"{model} {role} assignment candidate lies outside the topology table")
                candidate_valid = pool["candidate_mask"][batch_rows[:, None], role_candidates]
                if not bool(candidate_valid.all()):
                    bad = (~candidate_valid).nonzero(as_tuple=False)[0]
                    event_id = int(batch_ids[int(bad[0])])
                    raise ValueError(f"{model} {role} assignment candidate is invalid for EVENT_ID={event_id}")
                token_index = candidate_table[role_candidates, 0]
                if torch.any(token_index < 0) or torch.any(token_index >= physics.shape[1]):
                    raise ValueError(f"{model} {role}.b assignment maps outside measured tokens")
                if not torch.all(
                    batch["type_ids"][batch_rows[:, None], token_index]
                    == data.schema.source_type_ids["JET"]
                ):
                    raise ValueError(f"{model} {role}.b assignment does not map to measured jet tokens")
                selected = physics[batch_rows[:, None], token_index]
                if not torch.isfinite(selected).all():
                    raise ValueError(f"{model} {role}.b has non-finite measured vectors near EVENT_ID={int(batch_ids[0])}")
                b_vectors[model][cursor : cursor + len(batch_ids), :, role_index] = (
                    selected.detach().cpu().numpy()
                )
                b_token_indices[model][cursor : cursor + len(batch_ids), :, role_index] = (
                    token_index.detach().cpu().numpy()
                )
        cursor += len(batch_ids)
        if batch_index % 50 == 0:
            print(f"Mapped measured b candidates for batch {batch_index + 1}/{len(loader)}", flush=True)

    if cursor != len(common_ids):
        raise RuntimeError(f"Mapped {cursor} measured rows for {len(common_ids)} saved posterior events")

    with h5py.File(args.truth, "r") as truth:
        if truth.attrs.get("kinematic_units") != "GeV" or truth.attrs.get("four_vector_order") != "E,px,py,pz":
            raise ValueError(f"Truth units or four-vector order are invalid: {args.truth}")
        truth_ids = np.asarray(truth["EVENT_ID"][:], dtype=np.int64)
        truth_parents = np.asarray(truth["TRUTH/parents"][:], dtype=np.float64)
    if not np.array_equal(truth_ids, common_ids):
        raise ValueError("After-FSR truth sidecar and M/C/J posterior EVENT_ID order differs")
    if truth_parents.shape != (len(common_ids), 2, 4) or not np.isfinite(truth_parents).all():
        raise ValueError("After-FSR truth parents have invalid shape or non-finite values")
    truth_mtt = invariant_mass(truth_parents.sum(axis=1), "after-FSR truth mtt")
    truth_mtt_edges = np.quantile(truth_mtt, np.linspace(0.0, 1.0, 6))
    if np.any(np.diff(truth_mtt_edges) <= 0):
        raise ValueError("After-FSR truth mtt does not support five distinct quantile bins")
    truth_mtt_bin = np.searchsorted(truth_mtt_edges[1:-1], truth_mtt, side="right")

    metrics = {}
    coherent_event_rmse: dict[str, np.ndarray] = {}
    shuffled_event_rmse: dict[str, dict[str, np.ndarray]] = {}
    seed_by_model = {model: args.seed + offset for offset, model in enumerate(MODELS)}
    for model in MODELS:
        if np.any(b_token_indices[model][:, :, 0] == b_token_indices[model][:, :, 1]):
            row, draw = np.argwhere(
                b_token_indices[model][:, :, 0] == b_token_indices[model][:, :, 1]
            )[0]
            raise ValueError(
                f"{model} posterior assigns the same measured b token to both sides: "
                f"EVENT_ID={int(common_ids[row])}, draw={int(draw)}"
            )
        posterior = posterior_data[model]
        b_draws = b_vectors[model]
        neutrinos = np.asarray(posterior["invisible"], dtype=np.float64)
        parents = np.asarray(posterior["parents"], dtype=np.float64)
        coherent = residuals(b_draws, neutrinos, parents, leptons, f"KNOT-{model} coherent")
        rng = np.random.default_rng(seed_by_model[model])
        permutation = np.argsort(rng.random((len(common_ids), draws)), axis=1, kind="stable")
        b_shuffled = np.take_along_axis(b_draws, permutation[:, :, None, None], axis=1)
        nu_shuffled = np.take_along_axis(neutrinos, permutation[:, :, None, None], axis=1)
        assignment_shuffle = residuals(
            b_shuffled, neutrinos, parents, leptons, f"KNOT-{model} assignment-b-shuffled"
        )
        neutrino_shuffle = residuals(
            b_draws, nu_shuffled, parents, leptons, f"KNOT-{model} neutrino-shuffled"
        )

        coherent_event_rmse[model] = event_rmse(coherent["delta_pT_GeV"])
        shuffled_event_rmse[model] = {}
        shuffled_event_rmse[model]["assignment_b"] = event_rmse(
            assignment_shuffle["delta_pT_GeV"]
        )
        shuffled_event_rmse[model]["neutrino"] = event_rmse(
            neutrino_shuffle["delta_pT_GeV"]
        )
        summaries = {
            "coherent": summarize_residuals(coherent, seed_by_model[model] + 10, args.bootstrap_replicates),
            "assignment_b_shuffled": summarize_residuals(
                assignment_shuffle, seed_by_model[model] + 20, args.bootstrap_replicates
            ),
            "neutrino_shuffled": summarize_residuals(
                neutrino_shuffle, seed_by_model[model] + 30, args.bootstrap_replicates
            ),
        }
        differences = {}
        for label, shuffled in (
            ("assignment_b_shuffle_minus_coherent", assignment_shuffle),
            ("neutrino_shuffle_minus_coherent", neutrino_shuffle),
        ):
            shuffled_event = event_rmse(shuffled["delta_pT_GeV"])
            difference = shuffled_event - coherent_event_rmse[model]
            differences[label] = {
                "mean_event_RMSE_difference_GeV": float(difference.mean()),
                "event_bootstrap_95pct_interval_GeV": event_bootstrap(
                    difference, seed_by_model[model] + 40, args.bootstrap_replicates
                ),
            }
        binned = []
        for bin_index, (low, high) in enumerate(pairwise(truth_mtt_edges)):
            rows = np.flatnonzero(truth_mtt_bin == bin_index)
            if not len(rows):
                raise ValueError(f"Truth mtt quantile bin {bin_index} is empty")
            binned_penalties = {}
            for label, key in (
                ("assignment_b_shuffle_minus_coherent", "assignment_b"),
                ("neutrino_shuffle_minus_coherent", "neutrino"),
            ):
                difference = (
                    shuffled_event_rmse[model][key][rows]
                    - coherent_event_rmse[model][rows]
                )
                binned_penalties[label] = {
                    "mean_event_RMSE_difference_GeV": float(difference.mean()),
                    "event_bootstrap_95pct_interval_GeV": event_bootstrap(
                        difference,
                        seed_by_model[model] + 100 + 2 * bin_index,
                        args.bootstrap_replicates,
                    ),
                }
            binned.append(
                {
                    "bin": bin_index,
                    "truth_mtt_range_GeV": [float(low), float(high)],
                    "events": len(rows),
                    "mean_truth_mtt_GeV": float(truth_mtt[rows].mean()),
                    "coherent_delta_pT_RMSE_GeV": float(
                        np.sqrt(np.mean(np.square(coherent_event_rmse[model][rows])))
                    ),
                    "coherent_delta_pT_RMSE_event_bootstrap_95pct_interval_GeV": event_bootstrap_rmse(
                        coherent_event_rmse[model][rows],
                        seed_by_model[model] + 110 + bin_index,
                        args.bootstrap_replicates,
                    ),
                    "delta_pT_shuffle_penalties": binned_penalties,
                }
            )
        metrics[model] = {
            "posterior_semantics": json.loads(
                (parsed[model].parent / "metrics.json").read_text()
            )["posterior_semantics"],
            "coherent_and_shuffled_residuals": summaries,
            "delta_pT_shuffle_penalties": differences,
            "coherent_delta_pT_by_truth_mtt_quantile": binned,
        }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "frozen KK transfer decay-chain consistency",
        "definition": "For every truth-free posterior draw, compare the sampled parent four-vector with its paired measured b candidate, charge-ordered lepton, and sampled neutrino. Shuffled controls permute b candidates or neutrino pairs across draws within each event. After this residual calculation, after-FSR truth mtt is used only to define common event quantile bins.",
        "events": len(common_ids),
        "posterior_draws_per_event": draws,
        "parent_role_order": list(ROLES),
        "invisible_role_order": ["top.nu", "antitop.nu"],
        "assignment_node_order": node_orders["M"],
        "truth_used_for_residuals": False,
        "truth_used_for_stratification": True,
        "truth_sidecar": str(args.truth.resolve()),
        "truth_sidecar_sha256": sha256_file(args.truth.resolve()),
        "truth_mtt_stratification": {
            "binning": "five equal-event quantiles of after-FSR truth mtt, common to M/C/J",
            "edges_GeV": [float(value) for value in truth_mtt_edges],
        },
        "event_id_ordered_sha256_little_endian_int64": ordered_id_hash(common_ids),
        "measured_input": str(args.input.resolve()),
        "measured_input_sha256": sha256_file(args.input.resolve()),
        "measured_topology_identity": topology_identity,
        "seed": int(args.seed),
        "bootstrap_resamples": int(args.bootstrap_replicates),
        "bootstrap_unit": "events",
        "models": metrics,
        "posterior_sources": {
            model: {
                "posterior": str(parsed[model].resolve()),
                "posterior_sha256": sha256_file(parsed[model].resolve()),
                "evaluation_manifest_sha256": sha256_file(
                    parsed[model].parent / "evaluation_manifest.json"
                ),
                "truth_used_during_inference": False,
            }
            for model in MODELS
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    print(json.dumps({"output": str(output), "events": len(common_ids), "models": list(MODELS)}, indent=2))


if __name__ == "__main__":
    main()
