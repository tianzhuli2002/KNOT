"""One maintained command line for reconstruction training, prediction and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf

from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.invisible import InvisibleScaler
from joint_event_reconstruction.data.structured import StructuredDataModule
from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions
from joint_event_reconstruction.prediction import predict_hdf5
from joint_event_reconstruction.prediction_discrete import predict_discrete_flow_hdf5
from joint_event_reconstruction.prediction_joint import predict_joint_flow_hdf5
from joint_event_reconstruction.training.joint import JointEventModule


def build_data(config, visible):
    invisible = config.data.invisible
    if config.model.invisible.enabled and invisible.path is None:
        raise ValueError("Invisible inference is enabled but data.invisible.path is null")
    build_assignments = (
        str(config.model.get("mode", "factorized")) == "joint_flow"
        or str(config.model.reconstruction.formulation) == "flow_matching"
    )
    return StructuredDataModule(
        visible,
        config.topology_config,
        tensor_cache=config.data.tensor_cache,
        invisible_path=invisible.path if config.model.invisible.enabled else None,
        allow_synthetic=bool(invisible.allow_synthetic),
        max_events=OmegaConf.to_container(config.data.max_events, resolve=True),
        subset_seed=int(config.experiment.seed),
        build_assignments=build_assignments,
        max_assignment_states=(
            int(config.model.reconstruction.flow_matching.max_states)
            if build_assignments
            else None
        ),
    )


def train(config) -> None:
    visible = OmegaConf.load(config.visible_config)
    data = build_data(config, visible)
    data.setup("fit")
    module = JointEventModule(
        config,
        visible,
        invisible_scaler=data.invisible_scaler,
        synthetic_data=(
            bool(data.structured_collator.invisible_store.synthetic)
            if data.structured_collator.invisible_store is not None
            else False
        ),
    )
    callbacks = [
        ModelCheckpoint(
            dirpath=Path(config.training.output) / "checkpoints",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename="best-loss",
        )
    ]
    if config.training.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=int(config.training.early_stopping.patience),
            )
        )
    trainer = L.Trainer(
        max_epochs=int(config.training.epochs),
        precision=str(config.training.precision),
        gradient_clip_val=float(config.training.gradient_clip),
        accumulate_grad_batches=int(config.training.accumulate_grad_batches),
        callbacks=callbacks,
        default_root_dir=config.training.output,
    )
    trainer.fit(module, datamodule=data)


def load_trained(config, visible, data, checkpoint_path: str) -> JointEventModule:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_scaler = checkpoint.get("invisible_scaler")
    if checkpoint_scaler is not None:
        scaler = InvisibleScaler.from_dict(checkpoint_scaler)
        data.invisible_scaler = scaler
        data.structured_collator.invisible_scaler = scaler
    else:
        scaler = data.invisible_scaler

    module = JointEventModule(
        config,
        visible,
        invisible_scaler=scaler,
        synthetic_data=(
            bool(data.structured_collator.invisible_store.synthetic)
            if data.structured_collator.invisible_store is not None
            else False
        ),
    )
    module.validate_checkpoint_metadata(checkpoint)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    return module


def main() -> None:
    # Mixed precision already handles the main tensor-core path. This also permits
    # TF32-style high-throughput float32 GEMMs for operations that remain in fp32.
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("train", "smoke", "validate-data", "predict", "evaluate"),
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one experiment value using OmegaConf dot notation. "
            "Repeat for multiple scientific choices, e.g. "
            "--set model.invisible.formulation=flow_matching"
        ),
    )
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    parser.add_argument("--output")
    parser.add_argument("--predictions")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--recurrent-steps", type=int)
    args = parser.parse_args()

    config = load_experiment(args.experiment, args.overrides)
    if args.command == "train":
        train(config)
        return
    if args.command == "smoke":
        from scripts.smoke_all import run_experiment_smoke
        from scripts.smoke_flows import run_flow_smoke

        run_experiment_smoke(Path(args.experiment), device=args.device, events=8)
        run_flow_smoke(Path(args.experiment), device=args.device, events=8)
        return
    if args.command == "validate-data":
        from scripts.data.validate_joint_h5 import validate

        print(
            json.dumps(
                validate(Path(config.data.invisible.path)), indent=2, sort_keys=True
            )
        )
        return
    if args.command == "evaluate":
        if not args.predictions:
            raise ValueError("evaluate requires --predictions")
        print(
            json.dumps(
                evaluate_any_predictions(args.predictions), indent=2, sort_keys=True
            )
        )
        return

    if not args.checkpoint or not args.output:
        raise ValueError("predict requires --checkpoint and --output")
    visible = OmegaConf.load(config.visible_config)
    data = build_data(config, visible)
    data.setup("predict")
    module = load_trained(config, visible, data, args.checkpoint)

    mode = str(config.model.get("mode", "factorized"))
    assignment_formulation = str(config.model.reconstruction.formulation)
    if mode == "joint_flow":
        predict_joint_flow_hdf5(
            module,
            data,
            data.invisible_scaler,
            args.output,
            split=args.split,
            posterior_samples=int(config.inference.posterior_samples),
            steps=int(config.inference.joint_flow.steps),
            sample_chunk_size=int(
                config.inference.joint_flow.get("sample_chunk_size", 1)
            ),
            device=args.device,
            checkpoint_path=args.checkpoint,
        )
    elif assignment_formulation == "flow_matching":
        predict_discrete_flow_hdf5(
            module,
            data,
            args.output,
            split=args.split,
            posterior_samples=int(config.inference.posterior_samples),
            steps=int(config.inference.assignment_flow.steps),
            device=args.device,
            checkpoint_path=args.checkpoint,
        )
    else:
        predict_hdf5(
            module,
            data,
            data.invisible_scaler,
            args.output,
            split=args.split,
            top_k=int(config.inference.top_k_assignments),
            posterior_samples=int(config.inference.posterior_samples),
            recurrent_steps=(
                args.recurrent_steps
                if args.recurrent_steps is not None
                else int(config.model.reasoning.eval_steps)
                if config.model.reasoning.enabled
                else None
            ),
            device=args.device,
            checkpoint_path=args.checkpoint,
            compatibility_lambda=float(
                config.model.compatibility.get("lambda", 1.0)
            ),
        )


if __name__ == "__main__":
    main()
