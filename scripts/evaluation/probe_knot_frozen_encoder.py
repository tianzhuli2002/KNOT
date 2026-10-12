#!/usr/bin/env python3
"""Fit frozen linear probes to KNOT-J encoder and assignment/neutrino contexts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.invisible import InvisibleScaler
from joint_event_reconstruction.data.partonic import PartonicScaler
from joint_event_reconstruction.data.subsets import deterministic_subset_positions
from joint_event_reconstruction.knot import KnotModule
from joint_event_reconstruction.models.partonic import (
    four_vectors_from_partonic_features,
    partonic_features,
)
from joint_event_reconstruction.topology.assignments import AssignmentSpace
from scripts.run_knot import make_module, move_to_device, truth_free_batch


def digest_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


@torch.inference_mode()
def map_conditioned_neutrino_mean(
    module: KnotModule,
    batch: dict,
    visible: dict,
    reconstruction: dict,
    space: AssignmentSpace,
    map_state: torch.Tensor,
    *,
    samples: int,
    generator: torch.Generator,
    steps: int,
) -> torch.Tensor:
    """Sample J's joint flow with a fixed predicted assignment, then average N."""
    batch_size = len(map_state)
    states = map_state[:, None]
    n_context = module.invisible_context(reconstruction, space, states, visible)[:, 0]
    parent_context = module.parent_context(
        reconstruction, space, states, visible, neutrino=None
    )[:, 0]
    context = torch.cat((n_context, parent_context), dim=1)
    memory, mask = module.event_model.invisible.visible_context(visible)
    repeated_memory = memory.repeat_interleave(samples, dim=0)
    repeated_mask = mask.repeat_interleave(samples, dim=0)
    repeated_context = context.repeat_interleave(samples, dim=0)
    dimensions = int(module.continuous.active_mask.sum())
    draws, _ = module.continuous.sample(
        (batch_size * samples, dimensions),
        repeated_memory,
        repeated_mask,
        repeated_context,
        steps=steps,
        solver="heun",
        generator=generator,
    )
    draws = draws.reshape(batch_size, samples, -1)
    neutrino_state = module.unpack_invisible(draws[..., : module.n_invisible])
    physical = module.event_model.invisible.decode_physical_state(
        neutrino_state,
        batch["met_xy"],
        module.event_model.invisible.scaler,
    )["four_vector"]
    return physical.mean(dim=1)


@torch.inference_mode()
def collect_features(
    module: KnotModule,
    data,
    dataset,
    positions: np.ndarray,
    *,
    expected_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
    posterior_samples: int,
    posterior_seed: int,
    posterior_steps: int,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        Subset(dataset, positions.tolist()),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data.structured_collator,
        num_workers=0,
        pin_memory=False,
    )
    result: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "event_id",
            "h",
            "h_A_truth",
            "h_A_map",
            "h_A_map_N_truth",
            "h_A_map_N_pred",
            "target_scaled",
            "target_truth_p4",
            "support",
            "map_state",
            "truth_state",
        )
    }
    generator = torch.Generator(device=device).manual_seed(posterior_seed)
    parent_roles = tuple(module.parent_roles)

    for batch_index, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        clean = truth_free_batch(batch)
        visible = module.event_model.encode_visible(clean)
        reconstruction, predicted_space = module.assignment(clean, visible)
        probabilities = predicted_space.state_probabilities(
            reconstruction["assignment_logits"]
        )
        map_state = probabilities.argmax(dim=1)
        truth_space = AssignmentSpace.from_structured(batch["structured"])
        truth_state = truth_space.truth_state

        if tuple(visible["parent_role_names"]) != parent_roles:
            raise ValueError(
                "KNOT-J parent query order differs from the checkpoint roles: "
                f"queries={visible['parent_role_names']}, roles={parent_roles}"
            )
        base_h = torch.cat(
            (
                visible["pooled"],
                visible["parent_role_context"].flatten(1),
            ),
            dim=1,
        )
        truth_assignment_context = module.hard_context(
            reconstruction,
            truth_space,
            truth_state[:, None],
            visible,
            parent_roles,
        )[:, 0]
        map_assignment_context = module.hard_context(
            reconstruction,
            predicted_space,
            map_state[:, None],
            visible,
            parent_roles,
        )[:, 0]
        invisible = module.event_model.invisible
        truth_neutrinos = invisible.decode_physical_state(
            batch["invisible_truth_scaled"],
            batch["met_xy"],
            invisible.scaler,
        )["four_vector"]
        predicted_neutrinos = map_conditioned_neutrino_mean(
            module,
            batch,
            visible,
            reconstruction,
            predicted_space,
            map_state,
            samples=posterior_samples,
            generator=generator,
            steps=posterior_steps,
        )

        target_features = partonic_features(batch["partonic_truth"])
        target_scaled = (
            target_features - module.parent_mean
        ) / module.parent_std
        required_support = (
            batch["partonic_support"].all(dim=1)
            & batch["invisible_support"].all(dim=1)
        )
        if "visible_assignment_support" not in batch:
            raise KeyError("KNOT probe batch lacks explicit visible-assignment support")
        required_support &= batch["visible_assignment_support"]

        chunks = {
            "event_id": batch["event_index"],
            "h": base_h,
            "h_A_truth": torch.cat(
                (base_h, truth_assignment_context.flatten(1)), dim=1
            ),
            "h_A_map": torch.cat(
                (base_h, map_assignment_context.flatten(1)), dim=1
            ),
            "h_A_map_N_truth": torch.cat(
                (
                    base_h,
                    map_assignment_context.flatten(1),
                    truth_neutrinos.flatten(1),
                ),
                dim=1,
            ),
            "h_A_map_N_pred": torch.cat(
                (
                    base_h,
                    map_assignment_context.flatten(1),
                    predicted_neutrinos.flatten(1),
                ),
                dim=1,
            ),
            "target_scaled": target_scaled.flatten(1),
            "target_truth_p4": batch["partonic_truth"].flatten(1),
            "support": required_support,
            "map_state": map_state,
            "truth_state": truth_state,
        }
        for name, value in chunks.items():
            result[name].append(value.detach().cpu().numpy())
        if (batch_index + 1) % 10 == 0:
            print(
                f"encoded {batch_index + 1} batches, "
                f"{sum(len(part) for part in result['event_id'])} events",
                flush=True,
            )

    merged = {name: np.concatenate(parts, axis=0) for name, parts in result.items()}
    if not np.array_equal(merged["event_id"], expected_ids):
        raise ValueError("KNOT probe loader returned a different ordered EVENT_ID list")
    return merged


def fit_ridge_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict]:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_validation = np.asarray(x_validation, dtype=np.float64)
    if x_train.ndim != 2 or y_train.ndim != 2 or len(x_train) != len(y_train):
        raise ValueError("Frozen-probe training arrays have incompatible shapes")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise FloatingPointError("Frozen-probe training arrays contain non-finite values")
    if not np.isfinite(x_validation).all():
        raise FloatingPointError("Frozen-probe validation features contain non-finite values")

    order = np.random.default_rng(seed).permutation(len(x_train))
    split = int(0.8 * len(order))
    fit_inner, tune_inner = order[:split], order[split:]
    inner_mean = x_train[fit_inner].mean(axis=0)
    inner_std = x_train[fit_inner].std(axis=0)
    inner_std[inner_std < 1.0e-8] = 1.0
    y_inner_mean = y_train[fit_inner].mean(axis=0)
    y_inner_std = y_train[fit_inner].std(axis=0)
    y_inner_std[y_inner_std < 1.0e-8] = 1.0
    xi = (x_train[fit_inner] - inner_mean) / inner_std
    yi = (y_train[fit_inner] - y_inner_mean) / y_inner_std
    xv = (x_train[tune_inner] - inner_mean) / inner_std
    yv = (y_train[tune_inner] - y_inner_mean) / y_inner_std
    gram = (xi.T @ xi) / len(xi)
    cross = (xi.T @ yi) / len(xi)
    identity = np.eye(gram.shape[0], dtype=np.float64)
    candidates = (1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
    scores = {}
    for alpha in candidates:
        coefficient = np.linalg.solve(gram + alpha * identity, cross)
        prediction = xv @ coefficient
        scores[str(alpha)] = float(np.mean(np.square(prediction - yv)))
    chosen = min(candidates, key=lambda alpha: scores[str(alpha)])

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1.0e-8] = 1.0
    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)
    y_std[y_std < 1.0e-8] = 1.0
    x_normalized = (x_train - x_mean) / x_std
    y_normalized = (y_train - y_mean) / y_std
    gram = (x_normalized.T @ x_normalized) / len(x_normalized)
    cross = (x_normalized.T @ y_normalized) / len(x_normalized)
    coefficient = np.linalg.solve(gram + chosen * identity, cross)
    prediction = ((x_validation - x_mean) / x_std) @ coefficient
    prediction = prediction * y_std + y_mean
    return prediction, {
        "ridge_alpha_normalized": chosen,
        "ridge_tuning_seed": int(seed),
        "inner_validation_mse_standardized": scores[str(chosen)],
        "candidate_inner_validation_mse": scores,
        "input_dimensions": int(x_train.shape[1]),
        "target_dimensions": int(y_train.shape[1]),
    }


def fit_mlp_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Fit a small early-stopped MLP, using only the supplied training split."""
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    x_validation = np.asarray(x_validation, dtype=np.float32)
    if x_train.ndim != 2 or y_train.ndim != 2 or len(x_train) != len(y_train):
        raise ValueError("Frozen MLP probe training arrays have incompatible shapes")
    if not np.isfinite(x_train).all() or not np.isfinite(y_train).all():
        raise FloatingPointError("Frozen MLP probe training arrays contain non-finite values")
    if not np.isfinite(x_validation).all():
        raise FloatingPointError("Frozen MLP probe validation features contain non-finite values")

    order = np.random.default_rng(seed).permutation(len(x_train))
    split = int(0.8 * len(order))
    fit_inner, tune_inner = order[:split], order[split:]

    def scaling(x: np.ndarray, y: np.ndarray, rows: np.ndarray):
        x_mean, x_std = x[rows].mean(axis=0), x[rows].std(axis=0)
        y_mean, y_std = y[rows].mean(axis=0), y[rows].std(axis=0)
        x_std[x_std < 1.0e-6] = 1.0
        y_std[y_std < 1.0e-6] = 1.0
        return x_mean, x_std, y_mean, y_std

    def make_model(input_dimensions: int, output_dimensions: int):
        return torch.nn.Sequential(
            torch.nn.Linear(input_dimensions, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, output_dimensions),
        )

    torch.manual_seed(seed)
    inner_mean, inner_std, target_mean, target_std = scaling(
        x_train, y_train, fit_inner
    )
    x_inner = torch.from_numpy((x_train[fit_inner] - inner_mean) / inner_std)
    y_inner = torch.from_numpy((y_train[fit_inner] - target_mean) / target_std)
    x_tune = torch.from_numpy((x_train[tune_inner] - inner_mean) / inner_std)
    y_tune = torch.from_numpy((y_train[tune_inner] - target_mean) / target_std)
    inner_model = make_model(x_train.shape[1], y_train.shape[1])
    optimizer = torch.optim.AdamW(inner_model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    max_epochs = 200
    patience = 20
    batch_size = 256
    for epoch in range(1, max_epochs + 1):
        inner_model.train()
        permutation = torch.randperm(len(x_inner), generator=generator)
        for start in range(0, len(permutation), batch_size):
            batch_rows = permutation[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = inner_model(x_inner[batch_rows])
            loss = torch.mean(torch.square(prediction - y_inner[batch_rows]))
            if not torch.isfinite(loss):
                raise FloatingPointError("Frozen MLP probe training loss is non-finite")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in inner_model.parameters()
            ):
                raise FloatingPointError("Frozen MLP probe has a non-finite gradient")
            optimizer.step()

        inner_model.eval()
        with torch.inference_mode():
            validation_loss = float(
                torch.mean(torch.square(inner_model(x_tune) - y_tune)).item()
            )
        if not np.isfinite(validation_loss):
            raise FloatingPointError("Frozen MLP probe tuning loss is non-finite")
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_epoch <= 0:
        raise RuntimeError("Frozen MLP probe did not produce a finite early-stopping state")

    # Refit at the selected epoch count using all available training rows.
    full_mean, full_std, full_target_mean, full_target_std = scaling(
        x_train, y_train, np.arange(len(x_train))
    )
    x_full = torch.from_numpy((x_train - full_mean) / full_std)
    y_full = torch.from_numpy((y_train - full_target_mean) / full_target_std)
    torch.manual_seed(seed + 2)
    final_model = make_model(x_train.shape[1], y_train.shape[1])
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(), lr=1.0e-3, weight_decay=1.0e-4
    )
    final_generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    for _ in range(best_epoch):
        final_model.train()
        permutation = torch.randperm(len(x_full), generator=final_generator)
        for start in range(0, len(permutation), batch_size):
            batch_rows = permutation[start : start + batch_size]
            final_optimizer.zero_grad(set_to_none=True)
            prediction = final_model(x_full[batch_rows])
            loss = torch.mean(torch.square(prediction - y_full[batch_rows]))
            if not torch.isfinite(loss):
                raise FloatingPointError("Frozen MLP full-fit loss is non-finite")
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in final_model.parameters()
            ):
                raise FloatingPointError("Frozen MLP full-fit gradient is non-finite")
            final_optimizer.step()
    final_model.eval()
    validation_tensor = torch.from_numpy((x_validation - full_mean) / full_std)
    with torch.inference_mode():
        prediction = final_model(validation_tensor).numpy()
    prediction = prediction * full_target_std + full_target_mean
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Frozen MLP probe produced non-finite predictions")
    return prediction.astype(np.float64), {
        "model": "two-hidden-layer MLP",
        "hidden_width": 128,
        "activation": "GELU",
        "optimizer": "AdamW",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
        "inner_training_fraction": 0.8,
        "early_stopping_patience_epochs": patience,
        "selected_epoch_count": int(best_epoch),
        "maximum_epoch_count": max_epochs,
        "inner_validation_mse_standardized": best_loss,
        "early_stopping_seed": int(seed),
        "refit_seed": int(seed + 2),
        "input_dimensions": int(x_train.shape[1]),
        "target_dimensions": int(y_train.shape[1]),
    }


def parent_vectors_from_scaled_features(
    prediction: np.ndarray, parent_mean: np.ndarray, parent_std: np.ndarray
) -> np.ndarray:
    features = prediction.reshape(-1, 2, 4) * parent_std[None] + parent_mean[None]
    vectors = four_vectors_from_partonic_features(torch.from_numpy(features)).numpy()
    if not np.isfinite(vectors).all():
        raise FloatingPointError("Frozen probe produced non-finite parent four-vectors")
    return vectors


def invariant_mass(vectors: np.ndarray) -> np.ndarray:
    total = vectors.sum(axis=-2)
    mass_squared = total[..., 0] ** 2 - np.square(total[..., 1:]).sum(axis=-1)
    if not np.isfinite(mass_squared).all() or (mass_squared < -1.0e-5).any():
        raise FloatingPointError("Frozen probe produced an invalid parent-pair four-vector")
    return np.sqrt(np.maximum(mass_squared, 0.0))


def rapidity(vectors: np.ndarray, name: str) -> np.ndarray:
    energy, pz = vectors[..., 0], vectors[..., 3]
    plus, minus = energy + pz, energy - pz
    if not np.isfinite(plus).all() or not np.isfinite(minus).all():
        raise FloatingPointError(f"{name} contains non-finite rapidity arguments")
    if np.any(plus <= 0) or np.any(minus <= 0):
        raise ValueError(f"{name} contains a non-physical rapidity argument")
    return 0.5 * np.log(plus / minus)


def bootstrap_bias_rmse(error: np.ndarray, bootstrap_indices: np.ndarray) -> dict:
    bootstrap_error = error[bootstrap_indices]
    biases = bootstrap_error.mean(axis=1)
    rmses = np.sqrt(np.square(bootstrap_error).mean(axis=1))
    return {
        "bias": [float(value) for value in np.quantile(biases, [0.025, 0.975])],
        "RMSE": [float(value) for value in np.quantile(rmses, [0.025, 0.975])],
    }


def paired_rmse_difference(
    prediction_error: np.ndarray,
    reference_error: np.ndarray,
    bootstrap_indices: np.ndarray,
    *,
    reference_name: str,
) -> dict:
    left, right = prediction_error[bootstrap_indices], reference_error[bootstrap_indices]
    differences = np.sqrt(np.square(left).mean(axis=1)) - np.sqrt(
        np.square(right).mean(axis=1)
    )
    return {
        f"probe_minus_{reference_name}_RMSE_difference": float(
            np.sqrt(np.square(prediction_error).mean())
            - np.sqrt(np.square(reference_error).mean())
        ),
        "event_bootstrap_95pct_interval_GeV": [
            float(value) for value in np.quantile(differences, [0.025, 0.975])
        ],
    }


def per_parent_metrics(prediction: np.ndarray, truth: np.ndarray, roles: tuple[str, ...]):
    result = {}
    for index, role in enumerate(roles):
        observed = truth[:, index]
        estimated = prediction[:, index]
        truth_pt = np.hypot(observed[:, 1], observed[:, 2])
        pred_pt = np.hypot(estimated[:, 1], estimated[:, 2])
        truth_eta = np.arcsinh(observed[:, 3] / np.maximum(truth_pt, 1.0e-9))
        pred_eta = np.arcsinh(estimated[:, 3] / np.maximum(pred_pt, 1.0e-9))
        result[role] = {
            "parent_pT_bias_GeV": float(np.mean(pred_pt - truth_pt)),
            "parent_pT_RMSE_GeV": float(np.sqrt(np.mean(np.square(pred_pt - truth_pt)))),
            "parent_eta_bias": float(np.mean(pred_eta - truth_eta)),
            "parent_eta_RMSE": float(np.sqrt(np.mean(np.square(pred_eta - truth_eta)))),
            "parent_pz_bias_GeV": float(np.mean(estimated[:, 3] - observed[:, 3])),
            "parent_pz_RMSE_GeV": float(np.sqrt(np.mean(np.square(estimated[:, 3] - observed[:, 3])))),
        }
    return result


def summarize_response(
    prediction: np.ndarray,
    truth: np.ndarray,
    roles: tuple[str, ...],
    bootstrap_indices: np.ndarray,
) -> dict:
    pred_mtt = invariant_mass(prediction)
    true_mtt = invariant_mass(truth)
    residual = pred_mtt - true_mtt
    centered_truth = true_mtt - true_mtt.mean()
    slope = float(np.dot(centered_truth, pred_mtt - pred_mtt.mean()) / np.dot(centered_truth, centered_truth))
    bootstrap_bias = []
    bootstrap_rmse = []
    bootstrap_slope = []
    for indices in bootstrap_indices:
        t = true_mtt[indices]
        p = pred_mtt[indices]
        e = p - t
        tc = t - t.mean()
        bootstrap_bias.append(float(e.mean()))
        bootstrap_rmse.append(float(np.sqrt(np.mean(np.square(e)))))
        bootstrap_slope.append(float(np.dot(tc, p - p.mean()) / np.dot(tc, tc)))
    lohi = lambda values: [float(x) for x in np.quantile(values, [0.025, 0.975])]
    pred_pair = prediction.sum(axis=1)
    true_pair = truth.sum(axis=1)
    pred_pair_pt = np.hypot(pred_pair[:, 1], pred_pair[:, 2])
    true_pair_pt = np.hypot(true_pair[:, 1], true_pair[:, 2])
    pt_error = pred_pair_pt - true_pair_pt
    pred_y = rapidity(pred_pair, "frozen-probe predicted parent pair")
    true_y = rapidity(true_pair, "after-FSR truth parent pair")
    y_error = pred_y - true_y
    return {
        "mtt_posterior_probe_bias_GeV": float(residual.mean()),
        "mtt_posterior_probe_RMSE_GeV": float(np.sqrt(np.mean(np.square(residual)))),
        "mtt_response_slope": slope,
        "event_bootstrap_95pct_interval": {
            "bias_GeV": lohi(bootstrap_bias),
            "RMSE_GeV": lohi(bootstrap_rmse),
            "slope": lohi(bootstrap_slope),
        },
        "parents": per_parent_metrics(prediction, truth, roles),
        "ttbar_system": {
            "pT_bias_GeV": float(pt_error.mean()),
            "pT_RMSE_GeV": float(np.sqrt(np.square(pt_error).mean())),
            "pT_event_bootstrap_95pct_interval": bootstrap_bias_rmse(
                pt_error, bootstrap_indices
            ),
            "rapidity_bias": float(y_error.mean()),
            "rapidity_RMSE": float(np.sqrt(np.square(y_error).mean())),
            "rapidity_event_bootstrap_95pct_interval": bootstrap_bias_rmse(
                y_error, bootstrap_indices
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-posterior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-events", type=int, default=5000)
    parser.add_argument("--validation-events", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--posterior-samples", type=int, default=8)
    parser.add_argument("--posterior-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42011)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    args = parser.parse_args()
    for value, name in (
        (args.train_events, "train-events"),
        (args.validation_events, "validation-events"),
        (args.batch_size, "batch-size"),
        (args.posterior_samples, "posterior-samples"),
        (args.posterior_steps, "posterior-steps"),
        (args.bootstrap_resamples, "bootstrap-resamples"),
    ):
        if value <= 0:
            parser.error(f"--{name} must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite KNOT probe output: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for frozen KNOT probes but is unavailable")
    torch.set_num_threads(1)
    started = time.perf_counter()

    config = load_experiment(args.config)
    research = OmegaConf.to_container(OmegaConf.load(args.research), resolve=True)
    if research.get("variant") != "J":
        raise ValueError("Frozen encoder probes require the trained KNOT-J checkpoint")
    visible = load_visible_config(config)
    data = build_data(config, visible, device="cpu", complete_assignments=True)
    data.setup("fit")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("invisible_scaler") is not None:
        data.invisible_scaler = InvisibleScaler.from_dict(checkpoint["invisible_scaler"])
        data.structured_collator.invisible_scaler = data.invisible_scaler
    if checkpoint.get("partonic_scaler") is not None:
        data.partonic_scaler = PartonicScaler.from_dict(checkpoint["partonic_scaler"])
        data.structured_collator.partonic_scaler = data.partonic_scaler
    module = make_module(config, visible, research, data)
    module.validate_checkpoint_metadata(checkpoint)
    module.load_state_dict(checkpoint["state_dict"])
    module = module.to(device).eval()

    training_ids = data._selected_ids(
        "train", np.arange(len(data.train_dataset), dtype=np.int64)
    )
    validation_ids = data._selected_ids(
        "validation", np.arange(len(data.val_dataset), dtype=np.int64)
    )
    train_positions = deterministic_subset_positions(
        training_ids, args.train_events, args.seed + 100
    )
    validation_positions = deterministic_subset_positions(
        validation_ids, args.validation_events, args.seed + 200
    )
    train_selected = training_ids[train_positions]
    validation_selected = validation_ids[validation_positions]

    with h5py.File(args.validation_posterior, "r") as stored:
        stored_ids = np.asarray(stored["EVENT_ID"][:], dtype=np.int64)
        knot_parent_samples = np.asarray(stored["parents"][:], dtype=np.float64)
        knot_truth_parents = np.asarray(
            stored["TRUTH/parents"][:], dtype=np.float64
        )
    if not np.array_equal(validation_selected, stored_ids):
        raise ValueError(
            "Frozen probe validation EVENT_ID order does not match the 2k KNOT-J "
            "convergence sample"
        )

    train = collect_features(
        module,
        data,
        data.train_dataset,
        train_positions,
        expected_ids=train_selected,
        batch_size=args.batch_size,
        device=device,
        posterior_samples=args.posterior_samples,
        posterior_seed=args.seed,
        posterior_steps=args.posterior_steps,
    )
    validation = collect_features(
        module,
        data,
        data.val_dataset,
        validation_positions,
        expected_ids=validation_selected,
        batch_size=args.batch_size,
        device=device,
        posterior_samples=args.posterior_samples,
        posterior_seed=args.seed + 1,
        posterior_steps=args.posterior_steps,
    )

    train_keep = train["support"].astype(bool)
    validation_keep = validation["support"].astype(bool)
    if train_keep.sum() < max(1000, 2 * args.train_events // 3):
        raise ValueError(
            f"Too few supported training rows for probes: {int(train_keep.sum())}/"
            f"{len(train_keep)}"
        )
    if validation_keep.sum() < max(500, args.validation_events // 2):
        raise ValueError(
            f"Too few supported validation rows for probes: {int(validation_keep.sum())}/"
            f"{len(validation_keep)}"
        )

    training_targets = train["target_scaled"][train_keep]
    validation_truth = validation["target_truth_p4"][validation_keep].reshape(-1, 2, 4)
    role_names = tuple(module.parent_roles)
    parent_mean = module.parent_mean.detach().cpu().numpy()
    parent_std = module.parent_std.detach().cpu().numpy()
    variants = (
        "h",
        "h_A_truth",
        "h_A_map",
        "h_A_map_N_truth",
        "h_A_map_N_pred",
    )
    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, int(validation_keep.sum()),
        size=(args.bootstrap_resamples, int(validation_keep.sum())),
    )
    truth_mtt = invariant_mass(validation_truth)
    if knot_parent_samples.ndim != 4 or knot_parent_samples.shape[0] != len(stored_ids) or knot_parent_samples.shape[2:] != (2, 4):
        raise ValueError(
            "KNOT-J posterior parent samples have an invalid shape: "
            f"{knot_parent_samples.shape}"
        )
    if knot_truth_parents.shape != (len(stored_ids), 2, 4):
        raise ValueError(
            f"KNOT-J posterior truth parents have an invalid shape: {knot_truth_parents.shape}"
        )
    if not np.array_equal(knot_truth_parents[validation_keep], validation_truth):
        raise ValueError("Frozen probe truth parents differ from KNOT-J posterior truth")

    probe_results = {"linear": {}, "mlp": {}}
    predictions = {"linear": {}, "mlp": {}}
    for name in variants:
        for family, fitter in (
            ("linear", fit_ridge_probe),
            ("mlp", fit_mlp_probe),
        ):
            prediction, fit_metadata = fitter(
                train[name][train_keep],
                training_targets,
                validation[name][validation_keep],
                seed=args.seed + 300,
            )
            prediction_vectors = parent_vectors_from_scaled_features(
                prediction, parent_mean, parent_std
            )
            predictions[family][name] = prediction_vectors
            probe_results[family][name] = {
                "frozen_probe": fit_metadata,
                "validation_metrics": summarize_response(
                    prediction_vectors,
                    validation_truth,
                    role_names,
                    bootstrap_indices,
                ),
            }

    paired_vs_h = {}
    paired_vs_knot_j = {}
    knot_draw_mtt = invariant_mass(knot_parent_samples[validation_keep])
    knot_mtt_point = knot_draw_mtt.mean(axis=1)
    knot_mtt_error = knot_mtt_point - truth_mtt
    knot_parent_point_pt = np.hypot(
        knot_parent_samples[validation_keep, :, :, 1],
        knot_parent_samples[validation_keep, :, :, 2],
    ).mean(axis=1)
    knot_truth_parent_pt = np.hypot(
        validation_truth[:, :, 1], validation_truth[:, :, 2]
    )
    knot_parent_pt_error = knot_parent_point_pt - knot_truth_parent_pt
    knot_pair_draws = knot_parent_samples[validation_keep].sum(axis=2)
    knot_pair_pt = np.hypot(knot_pair_draws[..., 1], knot_pair_draws[..., 2]).mean(axis=1)
    knot_pair_y = rapidity(knot_pair_draws, "KNOT-J parent-pair draws").mean(axis=1)
    truth_pair = validation_truth.sum(axis=1)
    knot_pair_pt_error = knot_pair_pt - np.hypot(truth_pair[:, 1], truth_pair[:, 2])
    knot_pair_y_error = knot_pair_y - rapidity(
        truth_pair, "after-FSR truth parent pair"
    )
    knot_reference = {
        "mtt_posterior_mean": {
            "bias_GeV": float(knot_mtt_error.mean()),
            "RMSE_GeV": float(np.sqrt(np.square(knot_mtt_error).mean())),
            "event_bootstrap_95pct_interval": bootstrap_bias_rmse(
                knot_mtt_error, bootstrap_indices
            ),
            "response_slope": float(np.polyfit(truth_mtt, knot_mtt_point, 1)[0]),
        },
        "parent_pT": {
            role: {
                "RMSE_GeV": float(
                    np.sqrt(np.square(knot_parent_pt_error[:, index]).mean())
                ),
                "event_bootstrap_95pct_interval": bootstrap_bias_rmse(
                    knot_parent_pt_error[:, index], bootstrap_indices
                ),
            }
            for index, role in enumerate(role_names)
        },
        "ttbar_system": {
            "pT_RMSE_GeV": float(np.sqrt(np.square(knot_pair_pt_error).mean())),
            "rapidity_RMSE": float(np.sqrt(np.square(knot_pair_y_error).mean())),
        },
    }
    for family in ("linear", "mlp"):
        baseline_mtt = invariant_mass(predictions[family]["h"])
        paired_vs_h[family] = {}
        paired_vs_knot_j[family] = {}
        for name in variants:
            mtt_error = invariant_mass(predictions[family][name]) - truth_mtt
            parent_pt_error = np.hypot(
                predictions[family][name][:, :, 1],
                predictions[family][name][:, :, 2],
            ) - knot_truth_parent_pt
            pair = predictions[family][name].sum(axis=1)
            pair_pt_error = np.hypot(pair[:, 1], pair[:, 2]) - np.hypot(
                truth_pair[:, 1], truth_pair[:, 2]
            )
            pair_y_error = rapidity(pair, f"{family} {name} parent pair") - rapidity(
                truth_pair, "after-FSR truth parent pair"
            )
            paired_vs_knot_j[family][name] = {
                "mtt_RMSE": paired_rmse_difference(
                    mtt_error,
                    knot_mtt_error,
                    bootstrap_indices,
                    reference_name="KNOT_J",
                ),
                "top_parent_pT_RMSE": paired_rmse_difference(
                    parent_pt_error[:, role_names.index("top")],
                    knot_parent_pt_error[:, role_names.index("top")],
                    bootstrap_indices,
                    reference_name="KNOT_J",
                ),
                "antitop_parent_pT_RMSE": paired_rmse_difference(
                    parent_pt_error[:, role_names.index("antitop")],
                    knot_parent_pt_error[:, role_names.index("antitop")],
                    bootstrap_indices,
                    reference_name="KNOT_J",
                ),
                "ttbar_pT_RMSE": paired_rmse_difference(
                    pair_pt_error,
                    knot_pair_pt_error,
                    bootstrap_indices,
                    reference_name="KNOT_J",
                ),
                "ttbar_rapidity_RMSE": paired_rmse_difference(
                    pair_y_error,
                    knot_pair_y_error,
                    bootstrap_indices,
                    reference_name="KNOT_J",
                ),
            }
        for name in variants[1:]:
            base_error = baseline_mtt - truth_mtt
            other_error = invariant_mass(predictions[family][name]) - truth_mtt
            paired_vs_h[family][name] = paired_rmse_difference(
                other_error,
                base_error,
                bootstrap_indices,
                reference_name="h",
            )

    output = {
        "status": "validated",
        "sealed_test_used": False,
        "probe_type": "frozen linear ridge and small early-stopped MLP readouts",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "evaluation_worktree_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "training_events_selected": len(train_selected),
        "training_events_supported": int(train_keep.sum()),
        "training_event_id_ordered_sha256": digest_ids(train_selected),
        "validation_events_selected": len(validation_selected),
        "validation_events_supported": int(validation_keep.sum()),
        "validation_event_id_ordered_sha256": digest_ids(validation_selected),
        "validation_supported_event_id_ordered_sha256": digest_ids(
            validation_selected[validation_keep]
        ),
        "validation_matches_convergence_posterior": str(
            args.validation_posterior.resolve()
        ),
        "posterior_neutrino_conditioning": {
            "assignment": "predicted MAP fixed while sampling the KNOT-J joint flow",
            "samples_per_event": args.posterior_samples,
            "heun_steps": args.posterior_steps,
            "training_generator_seed": args.seed,
            "validation_generator_seed": args.seed + 1,
        },
        "representation_definition": {
            "h": "global topology-query context concatenated with both parent-role query contexts",
            "h_A_truth": "h plus each parent root context with its truth-assigned candidate embedding",
            "h_A_map": "h plus each parent root context with its predicted MAP-assigned candidate embedding",
            "h_A_map_N_truth": "h+A_map plus the physical truth neutrino E,px,py,pz vectors",
            "h_A_map_N_pred": "h+A_map plus the posterior-mean neutrino four-vectors sampled at fixed MAP assignment",
        },
        "target": "both after-FSR parent (px,py,pz,log1p(mass)) values; mtt is computed from the probed parents",
        "metrics": probe_results,
        "KNOT_J_reference_metrics": knot_reference,
        "paired_mtt_RMSE_differences_vs_h": paired_vs_h,
        "paired_probe_RMSE_differences_vs_KNOT_J": paired_vs_knot_j,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": output["status"], "output": str(args.output), "events": int(validation_keep.sum()), "runtime_seconds": output["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
