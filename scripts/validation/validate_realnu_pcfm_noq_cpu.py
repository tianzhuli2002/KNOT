"""Bounded validation of a parent CFM trained without query conditioning.

This uses only the train+validation measured input and after-FSR sidecar. It
does not construct a test loader or access any sealed-test source.
"""

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

from joint_event_reconstruction.cli.run import build_data
from joint_event_reconstruction.configuration import load_visible_config
from joint_event_reconstruction.prediction_partonic import predict_partonic_hdf5
from joint_event_reconstruction.public_config import load_public_experiment
from joint_event_reconstruction.training.joint import JointEventModule


BASE_CONFIG = ROOT / "configs/experiments/realnu_v1/production_draft/pcfm_q.public.yaml"


def _overrides(measured: Path, truth: Path, output: Path, *, use_queries: bool) -> list[str]:
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
        f"model.parent_cfm.use_topology_queries={'true' if use_queries else 'false'}",
        "training.trainable=[visible,queries,partonic_cfm]",
        "training.batch_size=4",
        "training.workers=0",
        "training.epochs=1",
        "training.precision=32-true",
        "training.optimizer.fused_adamw=false",
        "training.runtime.compile=false",
        f"training.output={output}",
    ]


def _parameter_counts(module) -> dict[str, int]:
    groups = {
        "visible": "event_model.visible.",
        "topology_queries": "event_model.topology_queries.",
        "parent_cfm": "event_model.parent_cfm.",
    }
    counts = {
        name: sum(
            parameter.numel()
            for key, parameter in module.named_parameters()
            if key.startswith(prefix)
        )
        for name, prefix in groups.items()
    }
    counts["all"] = sum(parameter.numel() for parameter in module.parameters())
    counts["trainable"] = sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
    return counts


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
    pieces = [
        parameter.grad.detach().float().square().sum()
        for name, parameter in module.named_parameters()
        if name.startswith(prefix) and parameter.grad is not None
    ]
    return 0.0 if not pieces else float(torch.stack(pieces).sum().sqrt())


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def run(measured: Path, truth: Path, *, device="cpu") -> dict:
    if not measured.is_file():
        raise FileNotFoundError(f"Train+validation measured input is missing: {measured}")
    if not truth.is_file():
        raise FileNotFoundError(f"Train+validation parent sidecar is missing: {truth}")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA PCFM_noQ smoke has no accessible GPU")
    torch.set_num_threads(1)
    torch.manual_seed(731940)

    with tempfile.TemporaryDirectory(prefix="realnu_pcfm_noq_") as temporary:
        temp = Path(temporary)
        overrides = _overrides(measured, truth, temp / "training", use_queries=False)
        config = load_public_experiment(BASE_CONFIG, overrides=overrides)
        if config.model.topology_queries.enabled is not True:
            raise AssertionError("NoQ integration fixture must keep queries available for other tasks")
        if config.model.parent_cfm.use_topology_queries is not False:
            raise AssertionError("NoQ configuration did not disable parent query conditioning")
        if list(config.training.trainable) != ["visible", "queries", "partonic_cfm"]:
            raise AssertionError("The local NoQ fixture must preserve PCFM_Q trainable components")
        if config.training.loss_weights.assignment != 0.0 or config.training.loss_weights.invisible != 0.0:
            raise AssertionError("PCFM_Q control must remain a direct-parent-only objective")

        visible, data, module = _make_module(config, device=device)
        if module.event_model.topology_queries is None:
            raise AssertionError("NoQ parent setting incorrectly removed topology queries globally")
        if module.event_model.parent_cfm.use_topology_queries:
            raise AssertionError("Parent CFM retained topology-query conditioning")
        if data.structured_collator.partonic_store is None:
            raise AssertionError("Local training requires the explicit after-FSR training sidecar")

        batch = _move_to_device(next(iter(data.train_dataloader())), device)
        support = batch.get("partonic_support")
        if support is None or not bool(support.any()):
            raise AssertionError("Local PCFM_noQ batch has no supported parent truth")

        module.train()
        torch.manual_seed(731941)
        output = module(batch, compute_objectives=True)
        loss = output.get("partonic_cfm_loss")
        if loss is None or not torch.isfinite(loss):
            raise AssertionError("PCFM_noQ forward did not produce finite parent CFM loss")
        loss.backward()
        gradients = {
            "visible": _gradient_norm(module, "event_model.visible."),
            "parent_cfm": _gradient_norm(module, "event_model.parent_cfm."),
        }
        if min(gradients.values()) <= 0.0:
            raise AssertionError(f"PCFM_noQ loss did not reach visible and parent modules: {gradients}")
        if any(
            parameter.grad is not None
            for parameter in module.event_model.topology_queries.parameters()
        ):
            raise AssertionError("NoQ parent loss leaked gradients into topology-query states")

        query_state = {
            name: value.detach().clone()
            for name, value in module.event_model.topology_queries.state_dict().items()
        }
        optimizer = torch.optim.AdamW(
            [parameter for parameter in module.parameters() if parameter.requires_grad],
            lr=1.0e-3,
            weight_decay=0.0,
        )
        losses = []
        for _ in range(6):
            torch.manual_seed(731942)
            optimizer.zero_grad(set_to_none=True)
            step_output = module(batch, compute_objectives=True)
            step_loss = step_output["partonic_cfm_loss"]
            losses.append(float(step_loss.detach()))
            step_loss.backward()
            optimizer.step()
        torch.manual_seed(731942)
        final_output = module(batch, compute_objectives=True)
        final_loss = float(final_output["partonic_cfm_loss"].detach())
        losses.append(final_loss)
        if not final_loss < losses[0]:
            raise AssertionError(
                f"Six repeated fixed-noise steps did not reduce the tiny-batch loss: {losses}"
            )
        after_query_state = module.event_model.topology_queries.state_dict()
        if any(
            not torch.equal(query_state[name], after_query_state[name])
            for name in query_state
        ):
            raise AssertionError("Direct parent-only updates changed disconnected query weights")

        # PCFM_Q and PCFM_noQ retain the same trainable component list. The
        # noQ parent route therefore proves its independence by giving query
        # parameters no gradient, rather than by deleting unrelated paths.
        q_config = load_public_experiment(
            BASE_CONFIG,
            overrides=_overrides(measured, truth, temp / "q_control", use_queries=True),
        )
        q_module = JointEventModule(
            q_config,
            visible,
            invisible_scaler=data.invisible_scaler,
            point_scaler=data.point_scaler,
            partonic_scaler=data.partonic_scaler,
            parent_cfm_scaler=data.partonic_scaler,
        )
        q_counts = _parameter_counts(q_module)
        noq_counts = _parameter_counts(module)

        predict_config = load_public_experiment(BASE_CONFIG, overrides=overrides)
        predict_visible = load_visible_config(predict_config)
        predict_data = build_data(
            predict_config, predict_visible, include_truth=False, device=str(device)
        )
        predict_data.setup("predict")
        if predict_data.structured_collator.partonic_store is not None:
            raise AssertionError("Truth-free prediction data opened a parent truth sidecar")
        truth_free_batch = _move_to_device(
            next(iter(predict_data.val_dataloader())), device
        )
        if any(
            key in truth_free_batch
            for key in ("partonic_truth", "partonic_support", "partonic_truth_scaled")
        ):
            raise AssertionError("Truth-free validation batch contains parent truth")

        module.eval()
        source = torch.randn(
            len(truth_free_batch["tokens"]), 2, 8,
            generator=torch.Generator(device=device).manual_seed(731943),
        )
        with torch.no_grad():
            no_query_sample = module.event_model.sample_partonic(
                truth_free_batch,
                samples=2,
                steps=2,
                solver="heun",
                parent_source_state=source,
            )
        if no_query_sample["parents"].shape != (len(truth_free_batch["tokens"]), 2, 2, 4):
            raise AssertionError("Explicit-source PCFM_noQ parent sample shape is invalid")
        if not torch.isfinite(no_query_sample["parents"]).all():
            raise AssertionError("Explicit-source PCFM_noQ prediction is non-finite")
        if no_query_sample["parent_role_conditioning"] != "learned_semantic_role_embeddings":
            raise AssertionError("PCFM_noQ sample did not record its role-conditioning mode")

        with torch.no_grad():
            for parameter in module.event_model.topology_queries.parameters():
                parameter.add_(0.25)
            perturbed_query_sample = module.event_model.sample_partonic(
                truth_free_batch,
                samples=2,
                steps=2,
                solver="heun",
                parent_source_state=source,
            )
            module.event_model.topology_queries.load_state_dict(query_state, strict=True)
        if not torch.equal(no_query_sample["parents"], perturbed_query_sample["parents"]):
            raise AssertionError("PCFM_noQ prediction changed when topology-query weights changed")

        checkpoint = temp / "pcfm_noq.ckpt"
        torch.save({"state_dict": module.state_dict()}, checkpoint)
        _, _, restored = _make_module(config, training_data=data, device=device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restored.load_state_dict(payload["state_dict"], strict=True)
        restored.eval()
        with torch.no_grad():
            restored_sample = restored.event_model.sample_partonic(
                truth_free_batch,
                samples=2,
                steps=2,
                solver="heun",
                parent_source_state=source,
            )
        if not torch.equal(no_query_sample["parents"], restored_sample["parents"]):
            raise AssertionError("PCFM_noQ save/reload changed explicit-source predictions")

        # The maintained HDF5 writer is exercised on a truth-free data module
        # built from the train+validation measured file alone.
        prediction_path = temp / "truth_free_predictions.h5"
        prediction_summary = predict_partonic_hdf5(
            restored,
            predict_data,
            prediction_path,
            split="validation",
            samples=2,
            steps=2,
            solver="heun",
            seed=731944,
            device=str(device),
            checkpoint_path=checkpoint,
        )
        if not prediction_path.is_file() or not prediction_summary["truth_free_input"]:
            raise AssertionError("Truth-free noQ prediction writer did not complete")
        with h5py.File(prediction_path, "r") as prediction:
            if prediction.attrs["parent_role_conditioning"] != "learned_semantic_role_embeddings":
                raise AssertionError("NoQ HDF5 metadata lost its role-conditioning contract")
            if "TRUTH" in prediction:
                raise AssertionError("NoQ prediction output unexpectedly contains truth")
            prediction_schema = str(prediction.attrs["prediction_schema"])
        prediction_summary["output"] = "temporary HDF5; removed after validation"
        prediction_summary["prediction_schema"] = prediction_schema

        return {
            "source": {
                "measured": str(measured),
                "truth_sidecar": str(truth),
            "partition_scope": "train+validation only; test loader not requested",
            "device": str(device),
            },
            "query_path": {
                "topology_queries_module_present": True,
                "parent_cfm_use_topology_queries": False,
                "parent_loss_query_gradient": 0.0,
                "query_weights_unchanged_after_parent_updates": True,
                "prediction_unchanged_after_query_weight_perturbation": True,
            },
            "parameter_counts": {"PCFM_Q": q_counts, "PCFM_noQ": noq_counts},
            "forward_backward": {
                "loss": float(loss.detach()),
                "gradient_norms": gradients,
            },
            "tiny_overfit_losses": losses,
            "explicit_source_prediction": {
                "source_shape": list(source.shape),
                "parent_shape": list(no_query_sample["parents"].shape),
                "save_reload_exact": True,
            },
            "truth_free_prediction": prediction_summary,
            "prediction_role_conditioning": "learned_semantic_role_embeddings",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run(
        args.measured.expanduser().resolve(),
        args.truth.expanduser().resolve(),
        device=args.device,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
