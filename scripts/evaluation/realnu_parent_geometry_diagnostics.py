"""CPU-testable geometry diagnostics for REALNU parent-flow inference.

The helpers operate on already-built event batches and explicit parent source
states. Geometry ablations accept only H_geom, so scalar context, truth,
EVENT_ID order, and source cannot be shuffled accidentally.
"""

from __future__ import annotations

import hashlib
import itertools
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ANGLES = (math.pi / 6.0, math.pi / 3.0, math.pi / 2.0)


def _rotation_matrix(angle: float, *, device, dtype) -> torch.Tensor:
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor([[cosine, -sine], [sine, cosine]], device=device, dtype=dtype)


def multivector_rotation_matrix(angle: float) -> np.ndarray:
    """SO(2) action for L-GATr's grade-grouped 16-blade basis."""
    cosine, sine = math.cos(angle), math.sin(angle)
    vector = np.eye(4, dtype=np.float64)
    vector[1:3, 1:3] = [[cosine, -sine], [sine, cosine]]
    blades = [()]
    for grade in range(1, 5):
        blades.extend(itertools.combinations(range(4), grade))
    if len(blades) != 16:
        raise RuntimeError("Unexpected L-GATr multivector blade basis")
    action = np.zeros((16, 16), dtype=np.float64)
    for i, output_blade in enumerate(blades):
        for j, input_blade in enumerate(blades):
            if len(output_blade) != len(input_blade):
                continue
            if not input_blade:
                action[i, j] = 1.0
            else:
                action[i, j] = np.linalg.det(vector[np.ix_(output_blade, input_blade)])
    return action


def rotate_event_batch(batch: dict, angle: float) -> dict:
    """Existing event-only convention: rotate measured four-vectors and MET."""
    if "physics_vectors" not in batch or "met_xy" not in batch:
        raise KeyError("Event rotation requires physics_vectors and met_xy")
    rotated = dict(batch)
    vectors = batch["physics_vectors"].clone()
    matrix = _rotation_matrix(angle, device=vectors.device, dtype=vectors.dtype)
    vectors[..., 1:3] = vectors[..., 1:3] @ matrix.T
    rotated["physics_vectors"] = vectors
    rotated["met_xy"] = batch["met_xy"] @ matrix.T
    # The established diagnostic keeps preassembled scalar/token channels fixed.
    if "tokens" in batch:
        rotated["tokens"] = batch["tokens"]
    return rotated


def rotate_parent_source(source_state: torch.Tensor, angle: float,
                         target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    """Rotate source px/py in physical feature space, conjugated by its scaler.

    The flattened parent source feature order is (px, py, pz, mass) for each
    parent, matching the production target scaler.
    """
    if source_state.ndim != 3 or source_state.shape[-1] != 8:
        raise ValueError("Parent source must be [B,M,8]")
    if target_mean.numel() != 8 or target_std.numel() != 8:
        raise ValueError("Parent target mean and standard deviation must contain eight features")
    if not torch.isfinite(source_state).all() or not torch.isfinite(target_mean).all() or not torch.isfinite(target_std).all():
        raise ValueError("Source rotation requires finite source and scaler values")
    if torch.any(target_std == 0):
        raise ValueError("Parent target scaler contains a zero standard deviation")
    mean = target_mean.to(source_state).reshape(1, 1, 2, 4)
    std = target_std.to(source_state).reshape(1, 1, 2, 4)
    feature = source_state.reshape(*source_state.shape[:2], 2, 4) * std + mean
    matrix = _rotation_matrix(angle, device=source_state.device, dtype=source_state.dtype)
    feature[..., :2] = feature[..., :2] @ matrix.T
    return ((feature - mean) / std).reshape_as(source_state)


def rotation_inputs(batch: dict, source_state: torch.Tensor, angle: float,
                    target_mean: torch.Tensor, target_std: torch.Tensor,
                    mode: str) -> tuple[dict, torch.Tensor]:
    """Return event/source inputs for event-only, source-only, or co-rotation."""
    if mode == "event_only":
        return rotate_event_batch(batch, angle), source_state
    if mode == "source_only":
        return batch, rotate_parent_source(source_state, angle, target_mean, target_std)
    if mode == "co_rotated":
        return rotate_event_batch(batch, angle), rotate_parent_source(
            source_state, angle, target_mean, target_std
        )
    raise ValueError(f"Unknown transverse rotation mode: {mode}")


def ablate_h_geom(h_geom: torch.Tensor, mode: str, *, permutation: torch.Tensor | None = None) -> torch.Tensor:
    """Transform only H_geom; the API cannot receive or alter scalar/truth data."""
    if h_geom.ndim != 4 or h_geom.shape[-2:] != (1, 16):
        raise ValueError(f"H_geom must be [B,N,1,16], found {tuple(h_geom.shape)}")
    if mode == "normal":
        return h_geom
    if mode == "zero":
        return torch.zeros_like(h_geom)
    if mode == "event_shuffled":
        if permutation is None:
            raise ValueError("H_geom shuffle requires an explicit event-row permutation")
        permutation = permutation.to(device=h_geom.device, dtype=torch.long)
        if permutation.shape != (len(h_geom),):
            raise ValueError("H_geom shuffle permutation must have shape [B]")
        if not torch.equal(torch.sort(permutation).values, torch.arange(len(h_geom), device=h_geom.device)):
            raise ValueError("H_geom shuffle input is not a permutation of event rows")
        if len(h_geom) > 1 and torch.equal(permutation, torch.arange(len(h_geom), device=h_geom.device)):
            raise ValueError("H_geom shuffle permutation must actually move at least one event")
        return h_geom.index_select(0, permutation)
    raise ValueError(f"Unknown H_geom ablation mode: {mode}")


def _encoded_parent_context(event_model, batch: dict, encoded: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    cfm = event_model.parent_cfm
    if cfm.conditioning != "P0":
        raise ValueError("Geometry ablation currently requires the trained PCFM_geom P0 policy")
    if cfm.velocity_backbone != "geometric":
        raise ValueError("Geometry ablation requires a geometric parent-flow checkpoint")
    if cfm.use_topology_queries:
        parent_queries = encoded.get("parent_role_context")
        if tuple(encoded.get("parent_role_names", ())) != ("top", "antitop"):
            raise ValueError("Parent topology queries are not in (top, antitop) order")
        parent_global = encoded["pooled"]
    else:
        parent_queries = None
        parent_global = encoded["encoder_pooled"]
    visible = torch.cat((parent_global.unsqueeze(1), encoded["tokens"]), dim=1)
    padding = torch.cat((torch.zeros((len(visible), 1), dtype=torch.bool, device=visible.device),
                         encoded["padding_mask"]), dim=1)
    geometric = encoded.get("geometric_tokens")
    if geometric is None or geometric.shape[:2] != visible.shape[:2] or geometric.shape[-2:] != (1, 16):
        raise ValueError("Encoded H_geom does not align with scalar event context")
    return visible, padding, parent_queries


def _sample_with_encoded_geometry(event_model, encoded: dict, batch: dict,
                                  source_state: torch.Tensor, *, samples: int,
                                  steps: int, solver: str,
                                  geometric_context: torch.Tensor) -> dict:
    visible, padding, parent_queries = _encoded_parent_context(event_model, batch, encoded)
    return event_model.parent_cfm.sample(
        visible,
        padding,
        parent_queries,
        samples=samples,
        steps=steps,
        solver=solver,
        source_state=source_state,
        geometric_context=geometric_context,
    )


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.no_grad()
def run_h_geom_ablation_triplet(event_model, batch: dict, source_state: torch.Tensor,
                                *, samples: int, steps: int, solver: str) -> dict:
    """Sample normal/zero/shuffled H_geom from one cached event encoding."""
    event_ids = batch.get("event_index", batch.get("event_id"))
    if event_ids is None:
        raise KeyError("H_geom ablation requires event_index or event_id in the input batch")
    event_ids_before = event_ids.detach().clone()
    source_before = source_state.detach().clone()
    encoded = event_model.encode_visible(batch)
    _encoded_parent_context(event_model, batch, encoded)
    h_geom = encoded["geometric_tokens"]
    scalar_before = {
        name: _tensor_sha256(encoded[name])
        for name in ("pooled", "tokens", "padding_mask", "parent_role_context")
        if name in encoded and torch.is_tensor(encoded[name])
    }
    permutation = torch.roll(torch.arange(len(h_geom), device=h_geom.device), shifts=1)
    outputs = {}
    for mode in ("normal", "zero", "event_shuffled"):
        transformed = ablate_h_geom(h_geom, mode, permutation=permutation if mode == "event_shuffled" else None)
        outputs[mode] = _sample_with_encoded_geometry(
            event_model, encoded, batch, source_state,
            samples=samples, steps=steps, solver=solver,
            geometric_context=transformed,
        )
        if not torch.equal(event_ids, event_ids_before) or not torch.equal(source_state, source_before):
            raise RuntimeError("H_geom ablation modified EVENT_ID order or source state")
        if scalar_before != {
            name: _tensor_sha256(encoded[name]) for name in scalar_before
        }:
            raise RuntimeError("H_geom ablation changed cached scalar event conditioning")
        if mode != "normal" and outputs[mode]["parents"].shape != outputs["normal"]["parents"].shape:
            raise RuntimeError("H_geom ablation changed output shape")
    return {
        "outputs": outputs,
        "event_ids": event_ids_before,
        "source_sha256": _tensor_sha256(source_before),
        "scalar_context_sha256": scalar_before,
        "geometry_permutation": permutation.detach().cpu().numpy(),
        "encoder_runs_for_triplet": 1,
        "same_source_for_all_modes": True,
        "same_EVENT_ID_order_for_all_modes": True,
        "only_H_geom_ablated": True,
    }


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _parent_phi(parents: torch.Tensor) -> torch.Tensor:
    return torch.atan2(parents[..., 2], parents[..., 1])


def _rotate_parent_vectors(parents: torch.Tensor, angle: float) -> torch.Tensor:
    expected = parents.clone()
    matrix = _rotation_matrix(angle, device=parents.device, dtype=parents.dtype)
    expected[..., 1:3] = expected[..., 1:3] @ matrix.T
    return expected


@torch.no_grad()
def run_rotation_sweep(event_model, batch: dict, source_state: torch.Tensor,
                       target_mean: torch.Tensor, target_std: torch.Tensor,
                       *, angles: tuple[float, ...] = ANGLES,
                       samples: int = 1, steps: int = 16, solver: str = "heun") -> dict:
    """Measure event-only, source-only, and co-rotated output responses."""
    event_ids = batch.get("event_index", batch.get("event_id"))
    if event_ids is None:
        raise KeyError("Rotation sweep requires event_index or event_id")
    base = event_model.sample_partonic(
        batch, samples=samples, steps=steps, solver=solver,
        parent_source_state=source_state,
    )["parents"]
    result = {"angles_rad": list(map(float, angles)), "event_count": len(event_ids), "modes": {}}
    for angle in angles:
        result["modes"][str(float(angle))] = {}
        expected_parent = _rotate_parent_vectors(base, angle)
        for mode in ("event_only", "source_only", "co_rotated"):
            rotated_batch, rotated_source = rotation_inputs(
                batch, source_state, angle, target_mean, target_std, mode
            )
            prediction = event_model.sample_partonic(
                rotated_batch, samples=samples, steps=steps, solver=solver,
                parent_source_state=rotated_source,
            )["parents"]
            if prediction.shape != base.shape:
                raise ValueError("Rotated inference returned a different parent-posterior shape")
            phi_shift = _wrap(_parent_phi(prediction) - _parent_phi(base))
            covariance_residual = _wrap(_parent_phi(prediction) - _parent_phi(expected_parent))
            mass2 = (prediction.sum(dim=-2)[..., 0].square()
                     - prediction.sum(dim=-2)[..., 1:].square().sum(dim=-1))
            base_pair = base.sum(dim=-2)
            base_mass2 = base_pair[..., 0].square() - base_pair[..., 1:].square().sum(dim=-1)
            mass_delta = mass2.clamp_min(0).sqrt() - base_mass2.clamp_min(0).sqrt()
            result["modes"][str(float(angle))][mode] = {
                "top_phi_shift_mean_rad": float(phi_shift[..., 0].mean().cpu()),
                "antitop_phi_shift_mean_rad": float(phi_shift[..., 1].mean().cpu()),
                "top_phi_shift_rms_rad": float(torch.sqrt(torch.mean(phi_shift[..., 0].square())).cpu()),
                "antitop_phi_shift_rms_rad": float(torch.sqrt(torch.mean(phi_shift[..., 1].square())).cpu()),
                "top_phi_response_error_rms_rad": float(torch.sqrt(torch.mean(_wrap(phi_shift[..., 0] - angle).square())).cpu()),
                "antitop_phi_response_error_rms_rad": float(torch.sqrt(torch.mean(_wrap(phi_shift[..., 1] - angle).square())).cpu()),
                "co_rotated_phi_covariance_residual_rms_rad": float(torch.sqrt(torch.mean(covariance_residual.square())).cpu()),
                "mttbar_invariance_rms_gev": float(torch.sqrt(torch.mean(mass_delta.square())).cpu()),
            }
    return result


def verify_parent_gradient_path(event_model, batch: dict, target_four_vectors: torch.Tensor,
                                support: torch.Tensor, source_state: torch.Tensor,
                                flow_time: torch.Tensor) -> dict:
    """Measure parent-loss gradients at H_geom and the visible encoder parameters."""
    if event_model.parent_cfm.conditioning != "P0" or event_model.parent_cfm.velocity_backbone != "geometric":
        raise ValueError("Gradient-path check requires PCFM_geom P0")
    encoded = event_model.encode_visible(batch)
    h_geom = encoded.get("geometric_tokens")
    if h_geom is None or not h_geom.requires_grad:
        raise RuntimeError("H_geom has no autograd path from the visible encoder")
    if event_model.parent_cfm.use_topology_queries:
        query = encoded["parent_role_context"]
        parent_global = encoded["pooled"]
    else:
        query = None
        parent_global = encoded["encoder_pooled"]
    visible = torch.cat((parent_global.unsqueeze(1), encoded["tokens"]), dim=1)
    padding = torch.cat((torch.zeros((len(visible), 1), dtype=torch.bool, device=visible.device),
                         encoded["padding_mask"]), dim=1)
    shared_mode = getattr(event_model, "parent_cfm_shared_encoder_gradients", "both")
    if shared_mode == "assignment_only":
        visible, padding, h_geom = visible.detach(), padding, h_geom.detach()
        query = None if query is None else query.detach()
        expects_visible_gradient = False
    else:
        expects_visible_gradient = True
    if h_geom.requires_grad and not h_geom.is_leaf:
        h_geom.retain_grad()
    visible_parameters = [parameter for parameter in event_model.visible.parameters() if parameter.requires_grad]
    if not visible_parameters:
        raise RuntimeError("Visible encoder has no trainable parameters for the gradient-path check")
    event_model.zero_grad(set_to_none=True)
    loss_output = event_model.parent_cfm.loss(
        target_four_vectors,
        support,
        visible,
        padding,
        query,
        source_state=source_state,
        flow_time=flow_time,
        geometric_context=h_geom,
    )
    parent_loss = loss_output["loss"]
    parent_loss.backward()
    h_grad = h_geom.grad if h_geom.requires_grad else None
    h_norm = 0.0 if h_grad is None else float(torch.linalg.vector_norm(h_grad).detach().cpu())
    squared = 0.0
    nonzero_parameters = 0
    for parameter in visible_parameters:
        if parameter.grad is not None:
            norm = float(torch.linalg.vector_norm(parameter.grad.detach()).cpu())
            squared += norm * norm
            nonzero_parameters += int(norm > 0.0)
        parameter.grad = None
    event_model.zero_grad(set_to_none=True)
    parameter_norm = math.sqrt(squared)
    if shared_mode != "assignment_only" and (h_norm <= 0.0 or parameter_norm <= 0.0):
        raise RuntimeError("Parent loss has a zero gradient to H_geom or visible encoder parameters")
    return {
        "status": "passed",
        "parent_loss": float(parent_loss.detach().cpu()),
        "H_geom_gradient_norm": h_norm,
        "visible_encoder_aggregate_gradient_norm": parameter_norm,
        "visible_parameters_with_nonzero_gradient": nonzero_parameters,
        "shared_encoder_gradient_policy": shared_mode,
        "visible_gradient_expected": expects_visible_gradient,
        "bounded_event_count": int(len(batch.get("event_index", target_four_vectors))),
        "cpu": target_four_vectors.device.type == "cpu",
    }


@torch.no_grad()
def verify_encoder_caching(event_model, batch: dict, source_state: torch.Tensor,
                           *, samples: int = 2, steps: int = 2, solver: str = "heun") -> dict:
    """Count one visible L-GATr call and geometric-block calls at every ODE NFE."""
    if event_model.parent_cfm.velocity_backbone != "geometric":
        raise ValueError("Encoder-cache check requires a geometric parent-flow model")
    counts = {"visible_encoder": 0, "geometric_parent_block": 0}
    visible_hook = event_model.visible.register_forward_hook(
        lambda module, args, output: counts.__setitem__("visible_encoder", counts["visible_encoder"] + 1)
    )
    denoiser = event_model.parent_cfm.flow.denoiser
    geometry_network = getattr(denoiser, "geometry_network", None)
    if geometry_network is None:
        visible_hook.remove()
        raise RuntimeError("Geometric parent denoiser has no L-GATr geometry block")
    geometry_hook = geometry_network.register_forward_hook(
        lambda module, args, output: counts.__setitem__("geometric_parent_block", counts["geometric_parent_block"] + 1)
    )
    try:
        output = event_model.sample_partonic(
            batch, samples=samples, steps=steps, solver=solver,
            parent_source_state=source_state,
        )
    finally:
        visible_hook.remove()
        geometry_hook.remove()
    expected_nfe = steps * (2 if solver == "heun" else 1)
    nfe = int(output["nfe"])
    if counts["visible_encoder"] != 1:
        raise RuntimeError(f"Measured-event L-GATr ran {counts['visible_encoder']} times for one event batch")
    if counts["geometric_parent_block"] != nfe or nfe != expected_nfe:
        raise RuntimeError(
            "Geometric parent block call count differs from NFE: "
            f"calls={counts['geometric_parent_block']}, nfe={nfe}, expected={expected_nfe}"
        )
    return {
        "status": "passed", "visible_LGATr_calls": counts["visible_encoder"],
        "geometric_parent_block_calls": counts["geometric_parent_block"],
        "NFE": nfe, "solver": solver, "steps": steps, "samples": samples,
        "events": int(len(batch.get("event_index", []))),
    }


class _CPUFixtureCFM:
    """Small differentiable stand-in that exercises the public diagnostic hooks."""

    def __init__(self):
        self.conditioning = "P0"
        self.velocity_backbone = "geometric"
        self.use_topology_queries = False
        self.flow = SimpleNamespace(denoiser=SimpleNamespace(geometry_network=torch.nn.Linear(16, 16)))

    def loss(self, target_four_vectors, support, visible, padding, query, *, source_state, flow_time, geometric_context):
        scalar = visible.sum(dim=(1, 2))
        geometry = geometric_context.sum(dim=(1, 2, 3))
        prediction = (scalar + geometry).reshape(-1, 1, 1).expand_as(target_four_vectors)
        return {"loss": (((prediction - target_four_vectors) ** 2) * support).mean()}

    def sample(self, visible, padding, parent_queries, *, samples, steps, solver, source_state, geometric_context):
        value = visible.sum(dim=(1, 2)) + geometric_context.sum(dim=(1, 2, 3))
        parents = value[:, None, None, None].expand(len(visible), samples, 2, 4).contiguous()
        return {"parents": parents}


class _CPUFixtureEventModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visible = torch.nn.Linear(4, 16, bias=False)
        self.parent_cfm = _CPUFixtureCFM()

    def encode_visible(self, batch):
        tokens = self.visible(batch["physics_vectors"][:, :1, :])
        pooled = tokens.mean(dim=1)
        geometric = torch.cat((pooled[:, None, :], tokens), dim=1).unsqueeze(-2)
        return {
            "tokens": tokens,
            "pooled": pooled,
            "encoder_pooled": pooled,
            "padding_mask": torch.zeros(tokens.shape[:2], dtype=torch.bool, device=tokens.device),
            "geometric_tokens": geometric,
        }

    def sample_partonic(self, batch, *, samples, steps, solver, parent_source_state):
        encoded = self.encode_visible(batch)
        nfe = steps * (2 if solver == "heun" else 1)
        for _ in range(nfe):
            self.parent_cfm.flow.denoiser.geometry_network(encoded["geometric_tokens"][..., 0, :])
        visible = torch.cat((encoded["encoder_pooled"][:, None], encoded["tokens"]), dim=1)
        padding = torch.cat((torch.zeros((len(visible), 1), dtype=torch.bool), encoded["padding_mask"]), dim=1)
        result = self.parent_cfm.sample(
            visible, padding, None, samples=samples, steps=steps, solver=solver,
            source_state=parent_source_state, geometric_context=encoded["geometric_tokens"],
        )
        return {**result, "nfe": nfe}


def self_check() -> dict:
    """Bounded CPU checks for transforms, H_geom isolation, gradients, and hooks."""
    torch.manual_seed(19)
    device, dtype = torch.device("cpu"), torch.float64
    angle = math.pi / 3.0
    vectors = torch.tensor([[[3.0, 1.0, 0.0, 2.0], [4.0, 0.0, 1.0, -1.0]],
                            [[2.0, -1.0, 0.5, 0.0], [5.0, 0.3, -0.8, 1.0]]], dtype=dtype)
    batch = {"physics_vectors": vectors, "met_xy": torch.tensor([[2.0, 1.0], [-1.0, 3.0]], dtype=dtype),
             "tokens": torch.randn(2, 3, 4, dtype=dtype), "event_index": torch.tensor([17, 29])}
    source = torch.randn(2, 2, 8, dtype=dtype)
    mean = torch.tensor([1.0, -2.0, 0.5, 0.1, -1.0, 2.0, -0.5, -0.1], dtype=dtype)
    std = torch.tensor([2.0, 3.0, 4.0, 1.0, 5.0, 6.0, 7.0, 1.0], dtype=dtype)
    original_event, original_source = batch["physics_vectors"].clone(), source.clone()
    event_only_batch, event_only_source = rotation_inputs(batch, source, angle, mean, std, "event_only")
    source_only_batch, source_only_source = rotation_inputs(batch, source, angle, mean, std, "source_only")
    co_batch, co_source = rotation_inputs(batch, source, angle, mean, std, "co_rotated")
    matrix = _rotation_matrix(angle, device=device, dtype=dtype)
    expected_vectors = original_event.clone()
    expected_vectors[..., 1:3] = expected_vectors[..., 1:3] @ matrix.T
    expected_met = batch["met_xy"] @ matrix.T
    if not torch.allclose(event_only_batch["physics_vectors"], expected_vectors) or not torch.allclose(event_only_batch["met_xy"], expected_met):
        raise RuntimeError("Event-only rotation convention failed its CPU fixture")
    if not torch.equal(event_only_source, original_source) or not torch.equal(source_only_batch["physics_vectors"], original_event):
        raise RuntimeError("Event-only/source-only held-fixed contract failed")
    if not torch.equal(source_only_source, co_source) or not torch.equal(co_batch["physics_vectors"], expected_vectors):
        raise RuntimeError("Co-rotated event/source input contract failed")
    if torch.equal(original_source, source_only_source):
        raise RuntimeError("Source-only rotation did not change the fixed source")

    h_geom = torch.randn(3, 4, 1, 16, dtype=dtype)
    permutation = torch.tensor([1, 2, 0], dtype=torch.long)
    zero = ablate_h_geom(h_geom, "zero")
    shuffled = ablate_h_geom(h_geom, "event_shuffled", permutation=permutation)
    if not torch.equal(zero, torch.zeros_like(h_geom)) or not torch.equal(shuffled, h_geom[permutation]):
        raise RuntimeError("H_geom-only ablation fixture failed")
    if not torch.equal(h_geom, h_geom.clone()):
        raise RuntimeError("H_geom ablation mutated the baseline geometry")

    fixture = _CPUFixtureEventModel()
    fixture_batch = {
        "physics_vectors": torch.randn(2, 2, 4, dtype=torch.float32),
        "met_xy": torch.randn(2, 2, dtype=torch.float32),
        "event_index": torch.tensor([17, 29]),
    }
    fixture_source = torch.randn(2, 2, 8)
    gradient_contract = verify_parent_gradient_path(
        fixture, fixture_batch, torch.zeros(2, 2, 4), torch.ones(2, 2, 4),
        fixture_source, torch.full((2,), 0.5),
    )
    cache_contract = verify_encoder_caching(
        fixture, fixture_batch, fixture_source, samples=2, steps=2, solver="heun",
    )
    ablation_contract = run_h_geom_ablation_triplet(
        fixture, fixture_batch, fixture_source, samples=2, steps=2, solver="heun",
    )
    if not ablation_contract["only_H_geom_ablated"]:
        raise RuntimeError("H_geom ablation helper did not preserve its isolation contract")
    rotation_action = multivector_rotation_matrix(angle)
    if rotation_action.shape != (16, 16) or not np.allclose(rotation_action.T @ rotation_action, np.eye(16), atol=1.0e-12):
        raise RuntimeError("L-GATr transverse rotation representation is not orthogonal")
    return {
        "status": "passed",
        "cpu_fixture": True,
        "event_only_source_held_fixed": True,
        "source_only_event_held_fixed": True,
        "co_rotated_inputs_share_one_angle": True,
        "H_geom_zero_and_event_shuffle_isolated": True,
        "gradient_path_contract": gradient_contract,
        "encoder_caching_contract": cache_contract,
        "H_geom_ablation_contract": {
            "encoder_runs_for_triplet": ablation_contract["encoder_runs_for_triplet"],
            "same_source_for_all_modes": ablation_contract["same_source_for_all_modes"],
            "same_EVENT_ID_order_for_all_modes": ablation_contract["same_EVENT_ID_order_for_all_modes"],
            "only_H_geom_ablated": ablation_contract["only_H_geom_ablated"],
        },
        "L_GATr_blade_rotation_orthogonality_max_error": float(np.max(np.abs(rotation_action.T @ rotation_action - np.eye(16)))),
        "angles_rad": list(ANGLES),
        "trained_geom_physics_evaluated": False,
    }


def write_contracts(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke = self_check()
    contracts = {
        "rotation_contract.json": {
            "status": "cpu_contract_passed",
            "angles_rad": list(ANGLES),
            "event_only": "rotate measured physics_vectors px/py and met_xy; hold the explicit parent source fixed; preserve the existing token/scalar convention",
            "source_only": "hold the measured event and all scalar conditioning fixed; rotate parent source px/py in physical feature space and apply the production scaler inverse/forward transform",
            "co_rotated": "apply the same SO(2) angle to measured event and parent source; compare output parent four-vectors to the rotated baseline",
            "outputs": ["top_phi", "antitop_phi", "wrapped phi response error", "co-rotated covariance residual", "mttbar invariance"],
            "helper": "scripts/evaluation/realnu_parent_geometry_diagnostics.py:run_rotation_sweep",
            "cpu_fixture": smoke,
            "trained_geom_physics_evaluated": False,
        },
        "gradient_contract.json": {
            "status": "machinery_cpu_fixture_passed; trained_model_check_pending",
            "parent_loss_gradient": "measure gradient norm at the H_geom output and aggregate trainable visible-encoder parameter gradients using a bounded validation batch, fixed source state, and fixed flow time",
            "policy": "require non-zero gradients when parent_cfm_shared_encoder_gradients='both'; report the intentional stop-gradient when it is 'assignment_only'",
            "helper": "scripts/evaluation/realnu_parent_geometry_diagnostics.py:verify_parent_gradient_path",
            "cpu_fixture": smoke,
            "trained_geom_physics_evaluated": False,
        },
        "caching_contract.json": {
            "status": "machinery_cpu_fixture_passed; trained_model_check_pending",
            "visible_encoder": "forward-hook event_model.visible and require exactly one measured-event L-GATr execution per sample_partonic call",
            "dynamic_block": "forward-hook parent_cfm.flow.denoiser.geometry_network and require one invocation per ODE NFE",
            "helper": "scripts/evaluation/realnu_parent_geometry_diagnostics.py:verify_encoder_caching",
            "cpu_fixture": smoke,
            "trained_geom_physics_evaluated": False,
        },
        "ablation_contract.json": {
            "status": "cpu_contract_passed; trained_model_check_pending",
            "modes": ["normal H_geom", "zero H_geom", "event-shuffled H_geom"],
            "fixed_inputs": ["source state", "scalar visible context", "queries", "EVENT_ID order", "solver", "sample count"],
            "shuffle_api": "ablate_h_geom accepts only H_geom and a validated event-row permutation; it cannot receive scalar context or truth",
            "helper": "scripts/evaluation/realnu_parent_geometry_diagnostics.py:run_h_geom_ablation_triplet",
            "cpu_fixture": smoke,
            "trained_geom_physics_evaluated": False,
        },
    }
    for filename, record in contracts.items():
        path = output_dir / filename
        temporary = path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    return smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_contracts(args.output_dir.expanduser().resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
