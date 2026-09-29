#!/usr/bin/env python3
"""Real-data CPU gate for the opt-in PCFM_geom parent velocity path."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))

from joint_event_reconstruction.cli.run import build_data
from joint_event_reconstruction.cli.run import _load_initial_components
from joint_event_reconstruction.configuration import load_visible_config
from joint_event_reconstruction.prediction_partonic import predict_partonic_hdf5
from joint_event_reconstruction.public_config import load_public_experiment
from joint_event_reconstruction.training.joint import JointEventModule
from joint_event_reconstruction.models.partonic import (
    _package_embed_vector,
    four_vectors_from_partonic_features,
)
from diagnose_realnu_parent_rotation import (
    ANGLES as ROTATION_ANGLES,
    _multivector_rotation_matrix,
    _rotate_batch,
    _rotate_parents,
    _rotate_source,
    _summary as _rotation_summary,
)


BASE_CONFIG = ROOT / "configs/experiments/realnu_v1/production_draft/pcfm_geom.public.yaml"
PROJECT_ROOT = Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
INITIAL_CHECKPOINT = PROJECT_ROOT / (
    "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/"
    "initialisations_objective_matched_v5/PCFM_Q_initial.ckpt"
)
INITIAL_COMPONENTS = ("visible", "queries")


def _overrides(measured: Path, truth: Path, output: Path) -> list[str]:
    return [
        f"data.input={measured}",
        f"data.inference_input={measured}",
        f"model.parent_cfm.target_path={truth}",
        "data.cache=null",
        "data.events.train=4",
        "data.events.validation=2",
        "data.events.test=null",
        "data.loader.pin_memory=false",
        "data.loader.persistent_workers=false",
        "model.parent_cfm.velocity_backbone=geometric",
        "training.trainable=[visible,queries,partonic_cfm]",
        "training.batch_size=4",
        "training.workers=0",
        "training.epochs=1",
        "training.precision=32-true",
        "training.optimizer.fused_adamw=false",
        "training.runtime.compile=false",
        f"training.output={output}",
    ]


def _make_module(config, *, training_data=None, device="cpu"):
    visible = load_visible_config(config)
    visible.dataset.pin_memory = str(device) != "cpu"
    visible.dataset.persistent_workers = False
    if training_data is None:
        training_data = build_data(config, visible, device=str(device))
        training_data.setup("fit")
    module = JointEventModule(
        config,
        visible,
        invisible_scaler=training_data.invisible_scaler,
        point_scaler=training_data.point_scaler,
        partonic_scaler=training_data.partonic_scaler,
        parent_cfm_scaler=training_data.partonic_scaler,
    ).to(device)
    return visible, training_data, module


def _gradient_norm(module, prefix: str) -> float:
    values = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in module.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    ]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt())


def _move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    return value


def _parameter_counts(module) -> dict[str, int]:
    prefixes = {
        "visible": "event_model.visible.",
        "queries": "event_model.topology_queries.",
        "parent_cfm": "event_model.parent_cfm.",
        "geometric_velocity": "event_model.parent_cfm.flow.denoiser.geometry_network.",
    }
    result = {
        key: sum(
            parameter.numel()
            for name, parameter in module.named_parameters()
            if name.startswith(prefix)
        )
        for key, prefix in prefixes.items()
    }
    result["all"] = sum(parameter.numel() for parameter in module.parameters())
    result["trainable"] = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return result


def run(measured: Path, truth: Path, *, device="cpu") -> dict:
    if not measured.is_file():
        raise FileNotFoundError(f"Train+validation measured input is missing: {measured}")
    if not truth.is_file():
        raise FileNotFoundError(f"Train+validation parent sidecar is missing: {truth}")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA PCFM_geom smoke has no accessible GPU")
    torch.set_num_threads(1)
    torch.manual_seed(731950)

    with tempfile.TemporaryDirectory(prefix="realnu_pcfm_geom_") as temporary:
        temp = Path(temporary)
        overrides = _overrides(measured, truth, temp / "training")
        config = load_public_experiment(BASE_CONFIG, overrides=overrides)
        if config.model.visible.type != "lgatr":
            raise AssertionError("PCFM_geom requires the full measured-event L-GATr encoder")
        if config.model.parent_cfm.velocity_backbone != "geometric":
            raise AssertionError("Geometry validation config did not select the geometry path")
        if config.model.parent_cfm.use_topology_queries is not True:
            raise AssertionError("PCFM_geom must retain PCFM_Q scalar query context")
        if config.model.parent_cfm.conditioning != "P0":
            raise AssertionError("PCFM_geom must retain the P0 parent conditioning mode")
        if list(config.training.trainable) != ["visible", "queries", "partonic_cfm"]:
            raise AssertionError("PCFM_geom trainable components differ from PCFM_Q")
        if config.training.loss_weights.assignment != 0.0 or config.training.loss_weights.invisible != 0.0:
            raise AssertionError("PCFM_geom must retain the direct-parent-only objective")

        visible_config, data, module = _make_module(config, device=device)
        if not INITIAL_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"PCFM_Q shared initial checkpoint is missing: {INITIAL_CHECKPOINT}"
            )
        checkpoint_state = torch.load(
            INITIAL_CHECKPOINT, map_location="cpu", weights_only=False
        )["state_dict"]
        geometry_before = {
            name: value.detach().clone()
            for name, value in module.state_dict().items()
            if name.startswith("event_model.parent_cfm.flow.denoiser.geometry_network.")
        }
        loaded_tensor_count = _load_initial_components(
            module, INITIAL_CHECKPOINT, INITIAL_COMPONENTS
        )
        loaded_state = module.state_dict()
        loaded_components = (
            "event_model.visible.",
            "event_model.topology_queries.",
        )
        loaded_keys = [
            name
            for name in loaded_state
            if name.startswith(loaded_components)
        ]
        if not loaded_keys or any(
            not torch.equal(loaded_state[name].cpu(), checkpoint_state[name].cpu())
            for name in loaded_keys
        ):
            raise AssertionError("PCFM_geom visible/query tensors differ from PCFM_Q initialisation")
        if any(
            not torch.equal(module.state_dict()[name], value)
            for name, value in geometry_before.items()
        ):
            raise AssertionError("Loading shared components changed the scratch geometry velocity path")
        if data.structured_collator.partonic_store is None:
            raise AssertionError("Local geometry training requires explicit after-FSR truth")
        if module.event_model.topology_queries is None:
            raise AssertionError("PCFM_geom unexpectedly removed shared topology queries")
        if module.event_model.parent_cfm.flow.denoiser.geometry_network is None:
            raise AssertionError("PCFM_geom did not construct its conditional L-GATr block")

        batch = _move(next(iter(data.train_dataloader())), device)
        if "partonic_truth" not in batch or "partonic_support" not in batch:
            raise AssertionError("Real training batch is missing its parent target contract")
        if batch["partonic_truth"].shape != (4, 2, 4):
            raise AssertionError("Real training batch is not [B=4, top/antitop, E/px/py/pz]")

        module.train()
        torch.manual_seed(731951)
        output = module(batch, compute_objectives=True)
        loss = output.get("partonic_cfm_loss")
        if loss is None or not torch.isfinite(loss):
            raise AssertionError("PCFM_geom forward did not produce finite CFM loss")
        loss.backward()
        gradients = {
            "visible_encoder": _gradient_norm(module, "event_model.visible."),
            "topology_queries": _gradient_norm(module, "event_model.topology_queries."),
            "geometry_velocity": _gradient_norm(
                module, "event_model.parent_cfm.flow.denoiser.geometry_network."
            ),
        }
        if min(gradients.values()) <= 0.0:
            raise AssertionError(f"PCFM_geom loss missed an intended gradient path: {gradients}")

        # This is an optimization smoke, not a miniature training run.  Hold
        # the CFM source and interpolation time fixed so the loss comparison
        # measures parameter updates instead of Monte Carlo variation.
        fixed_source = torch.randn(
            (len(batch["tokens"]), 8),
            dtype=batch["tokens"].dtype,
            device=device,
            generator=torch.Generator(device=device).manual_seed(731952),
        )
        fixed_time = torch.linspace(
            0.2, 0.8, len(batch["tokens"]), dtype=batch["tokens"].dtype, device=device
        )

        def fixed_objective():
            visible = module.event_model.encode_visible(batch)
            parent_queries = visible["parent_role_context"]
            if tuple(visible["parent_role_names"]) != ("top", "antitop"):
                raise AssertionError("Fixed CFM smoke lost top/antitop query order")
            measured = torch.cat(
                (visible["pooled"].unsqueeze(1), visible["tokens"]), dim=1
            )
            measured_mask = torch.cat(
                (
                    torch.zeros(
                        (len(measured), 1), dtype=torch.bool, device=device
                    ),
                    visible["padding_mask"],
                ),
                dim=1,
            )
            return module.event_model.parent_cfm.loss(
                batch["partonic_truth"],
                batch["partonic_support"],
                measured,
                measured_mask,
                parent_queries,
                source_state=fixed_source,
                flow_time=fixed_time,
                geometric_context=visible["geometric_tokens"],
            )["loss"]

        optimizer = torch.optim.AdamW(
            [parameter for parameter in module.parameters() if parameter.requires_grad],
            lr=1.0e-4,
            weight_decay=0.0,
        )
        tiny_losses = [float(fixed_objective().detach())]
        for _ in range(16):
            optimizer.zero_grad(set_to_none=True)
            step_loss = fixed_objective()
            step_loss.backward()
            optimizer.step()
            tiny_losses.append(float(fixed_objective().detach()))
        if not tiny_losses[-1] < tiny_losses[0]:
            raise AssertionError(f"Fixed-source real-batch updates did not reduce CFM loss: {tiny_losses}")

        predict_config = load_public_experiment(BASE_CONFIG, overrides=overrides)
        predict_visible = load_visible_config(predict_config)
        predict_data = build_data(
            predict_config, predict_visible, include_truth=False, device=str(device)
        )
        predict_data.setup("predict")
        if predict_data.structured_collator.partonic_store is not None:
            raise AssertionError("Truth-free geometry inference opened the parent truth sidecar")
        truth_free_batch = _move(next(iter(predict_data.val_dataloader())), device)
        if any(
            name in truth_free_batch
            for name in ("partonic_truth", "partonic_support", "partonic_truth_scaled")
        ):
            raise AssertionError("Truth-free geometry batch contains parent truth")

        module.eval()
        source = torch.randn(
            len(truth_free_batch["tokens"]), 2, 8,
            generator=torch.Generator(device=device).manual_seed(731955),
        )
        with torch.no_grad():
            sample = module.event_model.sample_partonic(
                truth_free_batch,
                samples=2,
                steps=2,
                solver="heun",
                parent_source_state=source,
            )
        expected = (len(truth_free_batch["tokens"]), 2, 2, 4)
        if sample["parents"].shape != expected or not torch.isfinite(sample["parents"]).all():
            raise AssertionError("PCFM_geom truth-free parent sample is invalid")
        visible_reference = module.event_model.encode_visible(truth_free_batch)
        geometry_shape = list(visible_reference["geometric_tokens"].shape)
        cfm = module.event_model.parent_cfm
        geometry_denoiser = cfm.flow.denoiser
        mean = cfm.target_mean.detach()
        std = cfm.target_std.detach()
        source_features = (
            source.reshape(len(source), 2, 2, 4)
            * std.to(source).reshape(1, 1, 2, 4)
            + mean.to(source).reshape(1, 1, 2, 4)
        )
        source_parent_p4 = four_vectors_from_partonic_features(source_features)
        source_parent_tokens = _package_embed_vector(
            source_parent_p4 / geometry_denoiser.geometry_momentum_scale_gev,
            geometry_denoiser.geometry_spurions,
        )
        representation_rotation = {}
        dynamic_parent_token_rotation = {}
        for angle in ROTATION_ANGLES:
            rotated_batch = _rotate_batch(truth_free_batch, angle)
            rotated_visible = module.event_model.encode_visible(rotated_batch)
            action = torch.as_tensor(
                _multivector_rotation_matrix(angle),
                dtype=visible_reference["geometric_tokens"].dtype,
                device=device,
            )
            representation_rotation[str(float(angle))] = {
                "events": int(len(truth_free_batch["tokens"])),
                "h_geom_shape": geometry_shape,
                "h_geom_covariance_residual": _rotation_summary(
                    (
                        rotated_visible["geometric_tokens"]
                        - visible_reference["geometric_tokens"] @ action.T
                    ).detach().cpu().numpy()
                ),
                "scalar_context": {
                    name: _rotation_summary(
                        (rotated_visible[name] - visible_reference[name])
                        .detach().cpu().numpy()
                    )
                    for name in ("pooled", "tokens", "query_states", "parent_role_context")
                },
                "interpretation": "H_geom is compared with the multivector SO(2) action; scalar event/query features should be invariant.",
            }

            rotated_source = _rotate_source(source, angle, mean, std)
            rotated_features = (
                rotated_source.reshape(len(source), 2, 2, 4)
                * std.to(source).reshape(1, 1, 2, 4)
                + mean.to(source).reshape(1, 1, 2, 4)
            )
            rotated_parent_p4 = four_vectors_from_partonic_features(rotated_features)
            rotated_parent_tokens = _package_embed_vector(
                rotated_parent_p4 / geometry_denoiser.geometry_momentum_scale_gev,
                geometry_denoiser.geometry_spurions,
            )
            dynamic_parent_token_rotation[str(float(angle))] = {
                "events": int(len(source)),
                "samples_per_event": int(source.shape[1]),
                "physical_parent_p4_residual_gev": _rotation_summary(
                    (rotated_parent_p4 - _rotate_parents(source_parent_p4, angle))
                    .detach().cpu().numpy()
                ),
                "geometric_parent_token_residual": _rotation_summary(
                    (rotated_parent_tokens - source_parent_tokens @ action.T)
                    .detach().cpu().numpy()
                ),
                "source_coordinate_action": "inverse production scaler, rotate physical px/py, and reapply production scaler",
            }

        checkpoint = temp / "pcfm_geom.ckpt"
        torch.save({"state_dict": module.state_dict()}, checkpoint)
        _, _, restored = _make_module(config, training_data=data, device=device)
        restored.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"],
            strict=True,
        )
        restored.eval()
        with torch.no_grad():
            reloaded = restored.event_model.sample_partonic(
                truth_free_batch,
                samples=2,
                steps=2,
                solver="heun",
                parent_source_state=source,
            )
        if not torch.equal(sample["parents"], reloaded["parents"]):
            raise AssertionError("PCFM_geom save/reload changed fixed-source predictions")

        output_path = temp / "truth_free_geom_predictions.h5"
        prediction_summary = predict_partonic_hdf5(
            restored,
            predict_data,
            output_path,
            split="validation",
            samples=2,
            steps=2,
            solver="heun",
            seed=731956,
            device=str(device),
            checkpoint_path=checkpoint,
        )
        if not output_path.is_file() or not prediction_summary["truth_free_input"]:
            raise AssertionError("PCFM_geom truth-free HDF5 writer did not complete")
        with h5py.File(output_path, "r") as prediction:
            if "TRUTH" in prediction:
                raise AssertionError("PCFM_geom prediction unexpectedly contains truth")
            if prediction.attrs.get("parent_velocity_backbone") != "geometric":
                raise AssertionError("PCFM_geom writer omitted its velocity-backbone identity")
            roles = json.loads(prediction.attrs["parent_role_order"])
            if roles != ["top", "antitop"]:
                raise AssertionError("PCFM_geom writer changed parent role order")

        return {
            "status": "passed",
            "source": {
                "measured": str(measured),
                "truth_sidecar": str(truth),
                "scope": "four train events and two validation events from train+validation only",
                "sealed_test_used": False,
            },
            "architecture": {
                "visible_encoder": str(config.model.visible.type),
                "scalar_queries_retained": True,
                "parent_conditioning": "P0",
            "velocity_backbone": "geometric",
            "initial_checkpoint": str(INITIAL_CHECKPOINT),
            "initial_components": list(INITIAL_COMPONENTS),
            "initial_component_tensor_count": int(loaded_tensor_count),
                "geometric_tokens_shape": geometry_shape,
                "representation_rotation": representation_rotation,
                "dynamic_parent_token_rotation": dynamic_parent_token_rotation,
                "parent_sample_shape": list(sample["parents"].shape),
                "nfe": int(sample["nfe"]),
            },
            "forward_backward": {
                "loss": float(loss.detach()),
                "gradient_norms": gradients,
            },
            "tiny_overfit_losses": tiny_losses,
            "parameter_counts": _parameter_counts(module),
            "save_reload_fixed_source_equal": True,
            "truth_free_prediction": prediction_summary,
            "roles": ["top", "antitop"],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(args.measured.expanduser().resolve(), args.truth.expanduser().resolve(), device=args.device)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
