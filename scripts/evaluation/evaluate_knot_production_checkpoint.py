"""Evaluate one frozen KNOT M/C/J checkpoint on validation events only."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import fields
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.subsets import deterministic_subset_positions
from joint_event_reconstruction.knot import EventPosterior
from joint_event_reconstruction.knot_physics import (
    SPIN_DEFINITION,
    joint_energy_score,
    joint_metric,
    mass,
    reconstruction_metrics,
    spin_observables,
)
from scripts.run_knot import (
    FREEZE_PATH,
    make_module,
    move_to_device,
    truth_free_batch,
    validate_frozen_configuration,
)

ROOT = Path(__file__).resolve().parents[2]


def as_numpy(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def concatenate(posteriors):
    result = {}
    for field in fields(EventPosterior):
        values = [getattr(item, field.name) for item in posteriors]
        result[field.name] = (
            torch.cat(values) if torch.is_tensor(values[0]) else values[0]
        )
    return EventPosterior(**result).validate()


def select_batch(value, indices, batch_size):
    if torch.is_tensor(value):
        if value.ndim and value.shape[0] == batch_size:
            return value.index_select(0, indices)
        return value
    if isinstance(value, dict):
        return {
            key: select_batch(item, indices, batch_size)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(select_batch(item, indices, batch_size) for item in value)
    if isinstance(value, list) and len(value) == batch_size:
        return [value[int(index)] for index in indices.cpu().tolist()]
    return value


def wrapped_delta(value):
    return np.angle(np.exp(1j * value))


def point_metrics(samples, truth, *, periodic=False):
    if periodic:
        residual_draws = wrapped_delta(samples - truth[:, None])
        point = truth + np.angle(np.exp(1j * residual_draws).mean(axis=1))
        residual = wrapped_delta(point - truth)
    else:
        point = samples.mean(axis=1)
        residual = point - truth
    return {
        "bias": float(residual.mean()),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "residual_std": float(residual.std()),
    }


def posterior_calibration(samples, truth, *, periodic=False, bins=10):
    draws = wrapped_delta(samples - truth[:, None]) if periodic else samples
    target = np.zeros_like(truth) if periodic else truth
    coverage = {}
    for level in (0.50, 0.68, 0.90):
        tail = (1.0 - level) / 2.0
        lower, upper = np.quantile(draws, [tail, 1.0 - tail], axis=1)
        coverage[str(level)] = float(((target >= lower) & (target <= upper)).mean())
    less = (draws < target[:, None]).sum(axis=1)
    equal = (draws == target[:, None]).sum(axis=1)
    pit = (less + 0.5 * equal + 0.5) / (draws.shape[1] + 1.0)
    rank_hist, _ = np.histogram(pit, bins=np.linspace(0.0, 1.0, bins + 1))
    width = np.quantile(draws, 0.84, axis=1) - np.quantile(draws, 0.16, axis=1)
    median = np.median(draws, axis=1)
    absolute_error = np.abs(median - target)
    order = np.argsort(width, kind="stable")
    width_error = []
    for group in np.array_split(order, min(5, len(order))):
        if len(group):
            width_error.append(
                {
                    "events": len(group),
                    "mean_68_width": float(width[group].mean()),
                    "mean_absolute_median_error": float(absolute_error[group].mean()),
                }
            )
    return {
        "nominal_vs_empirical_coverage": coverage,
        "rank_pit_histogram": rank_hist.astype(int).tolist(),
        "rank_pit_bins": int(bins),
        "width_vs_absolute_error_68": width_error,
    }


def physical_calibration_features(neutrinos, parents, truth_n, truth_p):
    features = {}
    for side_index, side in enumerate(("top", "antitop")):
        n, tn = neutrinos[:, :, side_index], truth_n[:, side_index]
        for name, index in (("px", 1), ("py", 2), ("pz", 3)):
            features[f"neutrino_{side}_{name}"] = (n[..., index], tn[..., index], False)
        p, tp = parents[:, :, side_index], truth_p[:, side_index]
        p_pt = np.hypot(p[..., 1], p[..., 2])
        tp_pt = np.hypot(tp[..., 1], tp[..., 2])
        p_eta = np.arcsinh(p[..., 3] / np.maximum(p_pt, 1e-12))
        tp_eta = np.arcsinh(tp[..., 3] / np.maximum(tp_pt, 1e-12))
        p_phi, tp_phi = np.arctan2(p[..., 2], p[..., 1]), np.arctan2(tp[..., 2], tp[..., 1])
        features[f"parent_{side}_pT"] = (p_pt, tp_pt, False)
        features[f"parent_{side}_eta"] = (p_eta, tp_eta, False)
        features[f"parent_{side}_phi"] = (p_phi, tp_phi, True)
        features[f"parent_{side}_mass"] = (mass(p), mass(tp), False)
    return {
        name: posterior_calibration(samples, truth, periodic=periodic)
        for name, (samples, truth, periodic) in features.items()
    }


def assignment_diagnostics(probabilities, space, batch):
    truth_state = batch["structured"]["assignments"]["truth_state"]
    predicted_state = probabilities.argmax(dim=-1)
    predicted_candidate = space.node_indices(predicted_state)
    support = batch["visible_assignment_support"].bool()
    safe_truth_state = torch.where(support, truth_state, torch.zeros_like(truth_state))
    truth_candidate = space.node_indices(safe_truth_state)
    truth_candidate = torch.where(
        support[:, None], truth_candidate, torch.full_like(truth_candidate, -1)
    )
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(dim=-1)
    nll = -probabilities.gather(1, safe_truth_state[:, None]).squeeze(1).clamp_min(1e-30).log()
    role_correct = predicted_candidate == truth_candidate
    leaves = {}
    for role in batch["structured"]["pools"]:
        if role not in space.nodes:
            continue
        node_index = space.nodes.index(role)
        leaf_tables = batch["structured"]["pools"][role].get("leaf_indices", {})
        for leaf_role, table in leaf_tables.items():
            pred = predicted_candidate[:, node_index]
            target = truth_candidate[:, node_index]
            pred_row = table[pred.clamp_min(0)]
            target_row = table[target.clamp_min(0)]
            pred_row = torch.where(pred[:, None] >= 0, pred_row, -1)
            leaves[leaf_role] = (pred_row == target_row).all(-1)

    return {
        "entropy_nats": as_numpy(entropy),
        "support_mask": as_numpy(support),
        "complete_correct": as_numpy(predicted_state == safe_truth_state),
        "nll": as_numpy(nll),
        "node_correct": as_numpy(role_correct),
        "node_names": list(space.nodes),
        "truth_state": as_numpy(truth_state),
        "truth_candidates": as_numpy(truth_candidate),
        "leaf_correct": {name: as_numpy(value) for name, value in leaves.items()},
    }


def entropy_bins(
    diagnostics,
    neutrinos,
    parents,
    truth_n,
    truth_p,
    complete_correct,
    assignment_nll,
    bins=4,
):
    support = diagnostics["support_mask"].astype(bool)
    eligible = np.flatnonzero(support)
    order = eligible[np.argsort(diagnostics["entropy_nats"][eligible], kind="stable")]
    groups = np.array_split(order, min(bins, len(order)))
    rows = []
    for index, event_indices in enumerate(groups):
        row = {
            "bin": index,
            "events": len(event_indices),
            "mean_assignment_entropy_nats": float(diagnostics["entropy_nats"][event_indices].mean()),
            "complete_assignment_accuracy": float(complete_correct[event_indices].mean()),
            "assignment_NLL": float(assignment_nll[event_indices].mean()),
        }
        row["top_neutrino_pz_RMSE"] = float(
            np.sqrt(np.mean((neutrinos[event_indices, :, 0, 3].mean(axis=1) - truth_n[event_indices, 0, 3]) ** 2))
        )
        row["top_parent_pT_RMSE"] = float(
            np.sqrt(
                np.mean(
                    (
                        np.hypot(parents[event_indices, :, 0, 1], parents[event_indices, :, 0, 2]).mean(axis=1)
                        - np.hypot(truth_p[event_indices, 0, 1], truth_p[event_indices, 0, 2])
                    )
                    ** 2
                )
            )
        )
        row["joint"] = joint_metric(
            neutrinos[event_indices], parents[event_indices]
        )
        rows.append(row)
    return rows


def spin_comparison(parents, leptons, truth_parents, seed):
    paired = spin_observables(parents, leptons[:, None])
    permutation = np.argsort(
        np.random.default_rng(seed).random(parents.shape[:2]), axis=1, kind="stable"
    )
    shuffled_parents = parents.copy()
    shuffled_parents[:, :, 1] = np.take_along_axis(
        parents[:, :, 1], permutation[..., None], axis=1
    )
    shuffled = spin_observables(shuffled_parents, leptons[:, None])
    truth = spin_observables(truth_parents, leptons)
    summary = {}
    for name in truth:
        paired_event = paired[name].mean(axis=1)
        shuffled_event = shuffled[name].mean(axis=1)
        delta = paired_event - shuffled_event
        summary[name] = {
            "truth_mean": float(truth[name].mean()),
            "paired_mean": float(paired[name].mean()),
            "shuffled_mean": float(shuffled[name].mean()),
            "paired_vs_shuffled_event_RMS": float(np.sqrt(np.mean(delta**2))),
            "paired_difference_SE": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        }
    return summary, truth, paired, shuffled


def add_dataset(file, name, value):
    parent, _, leaf = name.rpartition("/")
    group = file.require_group(parent) if parent else file
    group.create_dataset(leaf, data=value)


def draw_plots(output, neutrinos, parents, truth_n, truth_p, spin):
    pair = parents.sum(axis=2)
    truth_pair = truth_p.sum(axis=1)
    def rapidity(vector):
        plus, minus = vector[..., 0] + vector[..., 3], vector[..., 0] - vector[..., 3]
        if np.any(plus <= 0) or np.any(minus <= 0):
            raise ValueError("Nonpositive E±pz encountered while computing rapidity")
        return 0.5 * np.log(plus / minus)

    quantities = [
        ("top pT", np.hypot(parents[:, :, 0, 1], parents[:, :, 0, 2]), np.hypot(truth_p[:, 0, 1], truth_p[:, 0, 2]), None),
        ("antitop pT", np.hypot(parents[:, :, 1, 1], parents[:, :, 1, 2]), np.hypot(truth_p[:, 1, 1], truth_p[:, 1, 2]), None),
        ("mttbar", mass(pair), mass(truth_pair), (250, 1500)),
        ("pTttbar", np.hypot(pair[..., 1], pair[..., 2]), np.hypot(truth_pair[:, 1], truth_pair[:, 2]), (0, 600)),
        ("yttbar", rapidity(pair), rapidity(truth_pair), None),
        ("top neutrino pz", neutrinos[:, :, 0, 3], truth_n[:, 0, 3], None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, (label, samples, truth, limits) in zip(axes.flat, quantities):
        axis.hist(np.broadcast_to(truth[:, None], samples.shape).ravel(), bins=40, range=limits, density=True, histtype="step", label="truth")
        axis.hist(samples.ravel(), bins=40, range=limits, density=True, histtype="step", label="posterior")
        axis.set_xlabel(label)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "physics_distributions.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3))
    for axis, name in zip(axes, spin[0]):
        limits = (0, np.pi) if name == "delta_phi_ll" else (-1, 1)
        axis.hist(spin[1][name], bins=30, range=limits, density=True, histtype="step", label="truth hybrid")
        axis.hist(spin[2][name].ravel(), bins=30, range=limits, density=True, histtype="step", label="paired")
        axis.hist(spin[3][name].ravel(), bins=30, range=limits, density=True, histtype="step", label="parent-pair shuffled")
        axis.set_xlabel(name)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "spin_distributions.png", dpi=140)
    plt.close(fig)


@torch.inference_mode()
def evaluate(args):
    config = load_experiment(args.config)
    research = OmegaConf.to_container(OmegaConf.load(args.research), resolve=True)
    if research["variant"] != args.variant:
        raise ValueError("Research config variant differs from --variant")
    validate_frozen_configuration(args, config, research, json.loads(FREEZE_PATH.read_text()))
    if config.data.input is None:
        raise ValueError("Frozen KNOT validation requires the configured measured input")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite KNOT evaluation output: {output}")
    with h5py.File(config.data.input, "r") as source:
        if "SPLIT" not in source or not np.isin(source["SPLIT"][:], (0, 1)).all():
            raise ValueError("Production evaluator input contains rows outside train+validation")

    visible = load_visible_config(config)
    data = build_data(config, visible, device="cpu", complete_assignments=True)
    data.setup("fit")
    ids = data._selected_ids("validation", np.arange(len(data.val_dataset), dtype=np.int64))
    positions = deterministic_subset_positions(ids, args.events, args.seed + 200)
    selected_ids = ids[positions]
    if len(selected_ids) == 0:
        raise ValueError("KNOT evaluator selected no validation events")
    loader = DataLoader(
        Subset(data.val_dataset, positions.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data.structured_collator,
        num_workers=0,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    module = make_module(config, visible, research, data)
    module.validate_checkpoint_metadata(checkpoint)
    module.load_state_dict(checkpoint["state_dict"])
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false")
    module.to(device).eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)

    predictions, truth_n_rows, truth_p_rows, lepton_rows = [], [], [], []
    assignment_rows = []
    sampled_b_vectors_rows, sampled_b_tokens_rows, truth_b_tokens_rows = [], [], []
    oracle_rows = {
        "truth_assignment_truth_neutrino": [],
        "predicted_assignment_truth_neutrino": [],
    }
    started = time.perf_counter()
    for batch_index, host_batch in enumerate(loader):
        batch = move_to_device(host_batch, device)
        clean = truth_free_batch(batch)
        posterior = module.posterior(clean, samples=args.samples, generator=generator)
        predictions.append(posterior)

        truth_n = module.event_model.invisible.decode_physical_state(
            batch["invisible_truth_scaled"][:, None],
            batch["met_xy"],
            module.event_model.invisible.scaler,
        )["four_vector"][:, 0]
        truth_n_rows.append(as_numpy(truth_n))
        truth_p_rows.append(as_numpy(batch["partonic_truth"]))
        lepton_positions = (
            batch["type_ids"][0] == data.schema.source_type_ids["LEPTON"]
        ).nonzero(as_tuple=False).flatten()
        if len(lepton_positions) != 2:
            raise ValueError("KNOT spin evaluation requires two charge-ordered lepton slots")
        lepton_rows.append(as_numpy(batch["physics_vectors"][:, lepton_positions]))

        encoded = module.event_model.encode_visible(clean)
        assignment, space = module.assignment(clean, encoded)
        probabilities = space.state_probabilities(assignment["assignment_logits"])
        diagnostics = assignment_diagnostics(probabilities, space, batch)
        assignment_rows.append(diagnostics)

        # Save the measured b four-vector paired with each sampled assignment.
        # Candidate states use the exact model-native topology leaf table.
        b_token_indices, truth_b_token_indices = [], []
        for role in ("top", "antitop"):
            node_index = space.nodes.index(role)
            leaf_table = batch["structured"]["pools"][role]["leaf_indices"]["b"]
            sampled_candidate = posterior.assignment_candidates[..., node_index]
            sampled_tokens = leaf_table[sampled_candidate, 0]
            truth_candidate = torch.as_tensor(
                diagnostics["truth_candidates"][:, node_index], device=device
            )
            safe_candidate = truth_candidate.clamp_min(0)
            truth_tokens = leaf_table[safe_candidate, 0]
            truth_tokens = torch.where(
                batch["visible_assignment_support"].bool(),
                truth_tokens,
                torch.full_like(truth_tokens, -1),
            )
            b_token_indices.append(sampled_tokens)
            truth_b_token_indices.append(truth_tokens)
        sampled_tokens = torch.stack(b_token_indices, dim=-1)
        truth_tokens = torch.stack(truth_b_token_indices, dim=-1)
        batch_indices = torch.arange(len(sampled_tokens), device=device)[:, None, None]
        sampled_b_vectors_rows.append(
            as_numpy(batch["physics_vectors"][batch_indices, sampled_tokens])
        )
        sampled_b_tokens_rows.append(as_numpy(sampled_tokens))
        truth_b_tokens_rows.append(as_numpy(truth_tokens))

        if args.variant == "C":
            keep = batch["visible_assignment_support"].bool().nonzero(as_tuple=False).flatten()
            if len(keep):
                oracle_batch = select_batch(batch, keep, len(batch["event_index"]))
                for endpoint, rows in oracle_rows.items():
                    oracle = module.posterior(
                        oracle_batch,
                        samples=args.samples,
                        generator=torch.Generator(device=device).manual_seed(args.seed + batch_index),
                        oracle=endpoint,
                    )
                    rows.append(as_numpy(oracle.parents))

        if batch_index % 20 == 0:
            print(f"{args.variant} validation batch {batch_index + 1}/{len(loader)}", flush=True)

    posterior = concatenate(predictions)
    n, p = as_numpy(posterior.invisible), as_numpy(posterior.parents)
    tn, tp, leptons = map(np.concatenate, (truth_n_rows, truth_p_rows, lepton_rows))
    sampled_b_vectors = np.concatenate(sampled_b_vectors_rows)
    sampled_b_tokens = np.concatenate(sampled_b_tokens_rows)
    truth_b_tokens = np.concatenate(truth_b_tokens_rows)
    support = np.concatenate([row["support_mask"] for row in assignment_rows]).astype(bool)
    entropy = np.concatenate([row["entropy_nats"] for row in assignment_rows])
    truth_states = np.concatenate([row["truth_state"] for row in assignment_rows])
    truth_candidates = np.concatenate([row["truth_candidates"] for row in assignment_rows])
    event_ids = as_numpy(posterior.event_id).astype(np.int64)
    if not np.array_equal(event_ids, selected_ids):
        raise ValueError("Posterior EVENT_ID order differs from deterministic validation selection")

    score_support = np.concatenate([row["support_mask"] for row in assignment_rows]).astype(bool)
    complete_correct = np.concatenate([row["complete_correct"] for row in assignment_rows])
    assignment_nll = np.concatenate([row["nll"] for row in assignment_rows])
    node_correct = np.concatenate([row["node_correct"] for row in assignment_rows])
    node_names = assignment_rows[0]["node_names"]
    if not score_support.any():
        raise ValueError("KNOT production validation has no model-supported truth assignments")
    if not np.array_equal(score_support, support):
        raise ValueError("Assignment support changed during aggregation")
    assignment_result = {
        "supported_events": int(support.sum()),
        "unsupported_events": int((~support).sum()),
        "support_fraction": float(support.mean()),
        "complete_accuracy": float(complete_correct[score_support].mean()),
        "NLL": float(assignment_nll[score_support].mean()),
        "per_node_accuracy": {
            name: float(node_correct[score_support, index].mean())
            for index, name in enumerate(node_names)
        },
        "per_role_accuracy": {
            name: float(
                np.concatenate([row["leaf_correct"][name] for row in assignment_rows])[score_support].mean()
            )
            for name in assignment_rows[0]["leaf_correct"]
        },
    }

    physics = reconstruction_metrics(n, p, tn, tp)
    for side_index, side in enumerate(("top", "antitop")):
        predicted_parent, truth_parent = p[:, :, side_index], tp[:, side_index]
        predicted_pt = np.hypot(predicted_parent[..., 1], predicted_parent[..., 2])
        truth_pt = np.hypot(truth_parent[..., 1], truth_parent[..., 2])
        predicted_eta = np.arcsinh(predicted_parent[..., 3] / np.maximum(predicted_pt, 1e-12))
        truth_eta = np.arcsinh(truth_parent[..., 3] / np.maximum(truth_pt, 1e-12))
        predicted_phi = np.arctan2(predicted_parent[..., 2], predicted_parent[..., 1])
        truth_phi = np.arctan2(truth_parent[..., 2], truth_parent[..., 1])
        physics[side]["parent_eta"] = point_metrics(predicted_eta, truth_eta)
        physics[side]["parent_phi"] = point_metrics(predicted_phi, truth_phi, periodic=True)
        predicted_n, truth_n_side = n[:, :, side_index], tn[:, side_index]
        predicted_n_pt = np.hypot(predicted_n[..., 1], predicted_n[..., 2])
        truth_n_pt = np.hypot(truth_n_side[..., 1], truth_n_side[..., 2])
        predicted_n_eta = np.arcsinh(predicted_n[..., 3] / np.maximum(predicted_n_pt, 1e-12))
        truth_n_eta = np.arcsinh(truth_n_side[..., 3] / np.maximum(truth_n_pt, 1e-12))
        predicted_n_phi = np.arctan2(predicted_n[..., 2], predicted_n[..., 1])
        truth_n_phi = np.arctan2(truth_n_side[..., 2], truth_n_side[..., 1])
        physics[side]["neutrino_eta"] = point_metrics(predicted_n_eta, truth_n_eta)
        physics[side]["neutrino_phi"] = point_metrics(predicted_n_phi, truth_n_phi, periodic=True)

    pair, truth_pair = p.sum(axis=2), tp.sum(axis=1)
    def rapidity(vector):
        plus, minus = vector[..., 0] + vector[..., 3], vector[..., 0] - vector[..., 3]
        if np.any(plus <= 0) or np.any(minus <= 0):
            raise ValueError("Nonpositive E±pz encountered while computing ttbar rapidity")
        return 0.5 * np.log(plus / minus)
    physics["yttbar"] = point_metrics(rapidity(pair), rapidity(truth_pair))
    calibration = physical_calibration_features(n, p, tn, tp)
    joint = joint_metric(n, p)
    joint["energy_score"] = joint_energy_score(n, p, tn, tp)
    assignment_result["entropy_nats"] = {
        "mean": float(entropy.mean()),
        "median": float(np.median(entropy)),
        "p90": float(np.quantile(entropy, 0.90)),
    }
    ambiguity = entropy_bins(
        {"support_mask": support, "entropy_nats": entropy},
        n,
        p,
        tn,
        tp,
        complete_correct,
        assignment_nll,
    )
    spin = spin_comparison(p, leptons, tp, args.seed + 214)

    oracle_summary = None
    if args.variant == "C":
        oracle_truth_n = tn[support]
        oracle_truth_p = tp[support]
        oracle_summary = {}
        for endpoint, values in oracle_rows.items():
            predicted_parents = np.concatenate(values, axis=0)
            oracle_metrics = reconstruction_metrics(
                oracle_truth_n[:, None], predicted_parents, oracle_truth_n, oracle_truth_p
            )
            oracle_summary[endpoint] = {
                side: {name: value for name, value in oracle_metrics[side].items() if name.startswith("parent_")}
                for side in ("top", "antitop")
            }
        deployed_metrics = reconstruction_metrics(n[support], p[support], tn[support], tp[support])
        oracle_summary["deployed_predicted_assignment_neutrino"] = {
            side: {name: value for name, value in deployed_metrics[side].items() if name.startswith("parent_")}
            for side in ("top", "antitop")
        }

    output.mkdir(parents=True)
    posterior.write(output / "posterior.h5")
    with h5py.File(output / "posterior.h5", "a") as saved:
        for name, value in {
            "TRUTH/neutrinos": tn,
            "TRUTH/parents": tp,
            "TRUTH/assignment_state": truth_states,
            "TRUTH/assignment_candidates": truth_candidates,
            "TRUTH/assignment_support": support,
            "measured/leptons": leptons,
            "measured_b/sample_four_vectors": sampled_b_vectors,
            "measured_b/sample_token_indices": sampled_b_tokens,
            "TRUTH/b_token_indices": truth_b_tokens,
            "diagnostics/assignment_entropy_nats": entropy,
        }.items():
            add_dataset(saved, name, value)
        for name, value in (("truth", spin[1]), ("paired", spin[2]), ("shuffled", spin[3])):
            for observable, draws in value.items():
                add_dataset(saved, f"spin/{name}/{observable}", draws)
        if args.variant == "C":
            for endpoint, values in oracle_rows.items():
                add_dataset(saved, f"oracle/{endpoint}/parents", np.concatenate(values, axis=0))
        saved.attrs["spin_definition"] = json.dumps(SPIN_DEFINITION)
        saved.attrs["sealed_test_used"] = False

    digest = hashlib.sha256(event_ids.astype("<i8").tobytes()).hexdigest()
    metrics = {
        "variant": args.variant,
        "posterior_semantics": {"M": "marginal", "C": "ancestral", "J": "joint"}[args.variant],
        "events": len(event_ids),
        "samples_per_event": int(args.samples),
        "event_id_sha256_ordered_little_endian_int64": digest,
        "sealed_test_used": False,
        "reconstruction": physics,
        "posterior_calibration_14d": calibration,
        "assignment": assignment_result,
        "assignment_entropy_quartiles": ambiguity,
        "joint": joint,
        "spin_hybrid": spin[0],
        "spin_definition": SPIN_DEFINITION,
        "oracle_ladder": oracle_summary,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    training_manifest_path = args.checkpoint.parent.parent / "run_manifest.json"
    training_manifest = (
        json.loads(training_manifest_path.read_text())
        if training_manifest_path.is_file()
        else {}
    )
    manifest = {
        "variant": args.variant,
        "training_source_commit": training_manifest.get("source_commit"),
        "evaluation_source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "public_config": str(Path(args.config).resolve()),
        "research_config": str(Path(args.research).resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "events": len(event_ids),
        "samples_per_event": int(args.samples),
        "evaluation_seed": int(args.seed),
        "validation_event_id_sha256": digest,
        "sealed_test_used": False,
        "spin_truth_benchmark": "hybrid: authoritative truth parents with measured charge-ordered leptons",
    }
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    draw_plots(output, n, p, tn, tp, spin)
    print(json.dumps({"output": str(output), "events": len(event_ids), "samples": args.samples, "runtime_seconds": metrics["runtime_seconds"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("M", "C", "J"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=int, default=None, help="deterministic validation subset; default is full validation")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42011)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.samples < 2 or args.batch_size <= 0 or (args.events is not None and args.events <= 0):
        parser.error("at least two samples, positive batch size, and a positive event count are required")
    for path in (args.config, args.research, args.checkpoint):
        if not path.is_file():
            parser.error(f"Required KNOT evaluation input does not exist: {path}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite KNOT evaluation output: {args.output}")
    torch.set_num_threads(1)
    evaluate(args)


if __name__ == "__main__":
    main()
