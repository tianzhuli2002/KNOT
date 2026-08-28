"""One maintained CLI for training, prediction, evaluation and plotting."""

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
from joint_event_reconstruction.plotting import plot_comparison, plot_prediction
from joint_event_reconstruction.prediction import predict_hdf5
from joint_event_reconstruction.prediction_discrete import predict_discrete_flow_hdf5
from joint_event_reconstruction.prediction_joint import predict_joint_flow_hdf5
from joint_event_reconstruction.training.joint import JointEventModule


def build_data(config, visible):
    invisible = config.data.invisible
    if config.model.invisible.enabled and invisible.path is None:
        raise ValueError("Invisible inference is enabled but data.invisible.path is null")
    build_assignments = (
        str(config.model.mode) == "joint_flow"
        or str(config.model.assignment.formulation) == "flow_matching"
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
            int(config.data.max_assignment_states) if build_assignments else None
        ),
    )


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def train(config, *, device: str = "auto") -> None:
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
    accelerator = {"auto": "auto", "cpu": "cpu", "cuda": "gpu"}[device]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
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


def predict(config, args) -> None:
    device = _resolve_device(args.device)
    visible = OmegaConf.load(config.visible_config)
    data = build_data(config, visible)
    data.setup("predict")
    module = load_trained(config, visible, data, args.checkpoint)

    mode = str(config.model.mode)
    assignment_formulation = str(config.model.assignment.formulation)
    if mode == "joint_flow":
        predict_joint_flow_hdf5(
            module,
            data,
            data.invisible_scaler,
            args.output,
            split=args.split,
            posterior_samples=int(config.inference.posterior_samples),
            steps=int(config.inference.joint_flow.steps),
            sample_chunk_size=int(config.inference.joint_flow.sample_chunk_size),
            device=device,
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
            device=device,
            checkpoint_path=args.checkpoint,
        )
    else:
        reasoning = config.model.assignment.reasoning
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
                else int(reasoning.eval_steps)
                if reasoning.enabled
                else None
            ),
            device=device,
            checkpoint_path=args.checkpoint,
            compatibility_weight=float(config.inference.compatibility_weight),
        )


def _plot_predictions(
    paths: list[str],
    output: str,
    labels: list[str] | None,
    formats: tuple[str, ...],
) -> None:
    output_dir = Path(output).expanduser().resolve()
    if len(paths) == 1:
        written = plot_prediction(paths[0], output_dir, formats=formats)
    else:
        written = []
        for path in paths:
            written.extend(
                plot_prediction(
                    path,
                    output_dir / Path(path).stem,
                    formats=formats,
                )
            )
        written.extend(
            plot_comparison(
                paths,
                output_dir / "comparison",
                labels=labels,
                formats=formats,
            )
        )
    print(f"Wrote {len(written)} evaluation/plot products to {output_dir}")


def _experiment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one experiment value using OmegaConf dot notation; repeat as needed.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Topology-aware event reconstruction",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train one configured experiment")
    _experiment_options(train_parser)
    train_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )

    predict_parser = commands.add_parser(
        "predict", help="write a self-describing prediction HDF5"
    )
    _experiment_options(predict_parser)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    predict_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    predict_parser.add_argument("--recurrent-steps", type=int)

    evaluate_parser = commands.add_parser(
        "evaluate", help="evaluate one prediction HDF5; no experiment YAML required"
    )
    evaluate_parser.add_argument("--predictions", required=True)

    plot_parser = commands.add_parser(
        "plot", help="render diagnostics for one or more prediction HDF5 files"
    )
    plot_parser.add_argument("--predictions", nargs="+", required=True)
    plot_parser.add_argument("--output", required=True)
    plot_parser.add_argument("--labels", nargs="+")
    plot_parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf")
    )

    smoke_parser = commands.add_parser(
        "smoke", help="run the maintained software/mechanism smoke suite"
    )
    smoke_parser.add_argument("--experiment", required=True)
    smoke_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    smoke_parser.add_argument("--events", type=int, default=8)

    validate_parser = commands.add_parser(
        "validate-data", help="validate the configured canonical invisible HDF5"
    )
    _experiment_options(validate_parser)

    show_parser = commands.add_parser(
        "show-config", help="validate and print the fully resolved experiment YAML"
    )
    _experiment_options(show_parser)
    return parser


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = build_parser().parse_args()

    if args.command == "evaluate":
        print(
            json.dumps(
                evaluate_any_predictions(args.predictions), indent=2, sort_keys=True
            )
        )
        return
    if args.command == "plot":
        if args.labels is not None and len(args.labels) != len(args.predictions):
            raise ValueError("--labels must match the number of prediction files")
        _plot_predictions(
            args.predictions,
            args.output,
            args.labels,
            tuple(args.formats),
        )
        return

    overrides = getattr(args, "overrides", [])
    config = load_experiment(args.experiment, overrides)
    if args.command == "show-config":
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    if args.command == "train":
        train(config, device=args.device)
        return
    if args.command == "predict":
        predict(config, args)
        return
    if args.command == "validate-data":
        from scripts.data.validate_joint_h5 import validate

        if config.data.invisible.path is None:
            raise ValueError("validate-data requires data.invisible.path")
        print(
            json.dumps(
                validate(Path(config.data.invisible.path)), indent=2, sort_keys=True
            )
        )
        return
    if args.command == "smoke":
        from scripts.smoke_all import run_experiment_smoke
        from scripts.smoke_flows import run_flow_smoke

        device = _resolve_device(args.device)
        run_experiment_smoke(
            Path(args.experiment), device=device, events=args.events
        )
        run_flow_smoke(
            Path(args.experiment), device=device, events=args.events
        )
        return
    raise RuntimeError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    main()
