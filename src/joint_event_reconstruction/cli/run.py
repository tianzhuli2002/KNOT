"""One maintained CLI for training, prediction, evaluation and plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from joint_event_reconstruction.configuration import load_experiment, load_visible_config

if TYPE_CHECKING:
    from joint_event_reconstruction.training.joint import JointEventModule


_COMPONENT_PREFIXES = {
    "visible": ("event_model.visible.",),
    "assignment": (
        "event_model.reconstruction.",
        "event_model.reasoner.",
        "event_model.feedback.",
        "event_model.assignment_flow.",
    ),
    "invisible": ("event_model.invisible.",),
    "c2": ("event_model.posterior_feedback.",),
    "compatibility": ("event_model.compatibility.",),
}


def _load_repo_script(relative_path: str, module_name: str):
    """Load an executable repository script without depending on a package named scripts."""
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "scripts" / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load repository script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_data(config, visible):
    from joint_event_reconstruction.data.structured import StructuredDataModule

    invisible = config.data.invisible
    if config.model.invisible.enabled and invisible.path is None:
        raise ValueError("Invisible inference is enabled but the data config has no invisible_truth")
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
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _selected_component_key(key: str, components: tuple[str, ...]) -> bool:
    return any(
        key.startswith(prefix)
        for component in components
        for prefix in _COMPONENT_PREFIXES[component]
    )


def _load_initial_components(
    module: JointEventModule,
    checkpoint_path: str | Path,
    components: tuple[str, ...],
) -> int:
    """Load complete semantic components without relaxing checkpoint shape checks."""
    import torch

    if not components:
        raise ValueError("Initial checkpoint requires at least one component")
    unknown = sorted(set(components) - set(_COMPONENT_PREFIXES))
    if unknown:
        raise ValueError(f"Unknown initialisation components: {unknown}")

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict")
    if not isinstance(source, dict):
        raise ValueError(f"Initial checkpoint has no state_dict: {path}")

    saved = checkpoint.get("hyper_parameters", {})
    for key in ("topology_signature", "input_features", "source_type_ids"):
        if saved.get(key) != module.hparams.get(key):
            raise ValueError(
                f"Initial checkpoint has incompatible {key}: "
                f"checkpoint={saved.get(key)!r}, current={module.hparams.get(key)!r}"
            )
    if "invisible" in components and checkpoint.get("invisible_scaler") != module.scaler_metadata:
        raise ValueError("Invisible component initialisation requires identical scaler metadata")

    target = module.state_dict()
    expected = {key for key in target if _selected_component_key(key, components)}
    available = {key for key in source if _selected_component_key(key, components)}
    missing = sorted(expected - available)
    unexpected = sorted(available - expected)
    mismatched = sorted(
        key
        for key in expected & available
        if tuple(source[key].shape) != tuple(target[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Initial checkpoint component mismatch: "
            f"components={list(components)}, missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}, shape_mismatch={mismatched[:20]}"
        )
    if not expected:
        raise ValueError(
            f"Selected initialisation components have no parameters/buffers: {components}"
        )

    module.load_state_dict({key: source[key] for key in expected}, strict=False)
    module.hparams["initial_checkpoint"] = str(path)
    module.hparams["initial_components"] = list(components)
    return len(expected)


def train(
    config,
    *,
    device: str = "auto",
    initial_checkpoint: str | None = None,
    initial_components: tuple[str, ...] = (),
) -> None:
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from joint_event_reconstruction.training.joint import JointEventModule

    torch.set_float32_matmul_precision("high")
    L.seed_everything(int(config.experiment.seed), workers=True)
    visible = load_visible_config(config)
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
    if initial_checkpoint is not None:
        loaded = _load_initial_components(module, initial_checkpoint, initial_components)
        print(
            f"Initialised {loaded} state entries from {initial_checkpoint} "
            f"for components={list(initial_components)}"
        )
    elif initial_components:
        raise ValueError("--initial-components requires --initial-checkpoint")

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
    import torch

    from joint_event_reconstruction.data.invisible import InvisibleScaler
    from joint_event_reconstruction.training.joint import JointEventModule

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
    import h5py
    import lightning as L
    import torch

    from joint_event_reconstruction.prediction import predict_hdf5
    from joint_event_reconstruction.prediction_discrete import predict_discrete_flow_hdf5
    from joint_event_reconstruction.prediction_joint import predict_joint_flow_hdf5

    torch.set_float32_matmul_precision("high")
    prediction_seed = int(config.inference.seed)
    L.seed_everything(prediction_seed, workers=True)
    device = _resolve_device(args.device)
    visible = load_visible_config(config)
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

    with h5py.File(Path(args.output).expanduser().resolve(), "r+") as output:
        output.attrs["prediction_seed"] = prediction_seed


def _plot_predictions(
    paths: list[str],
    output: str,
    labels: list[str] | None,
    formats: tuple[str, ...],
) -> None:
    from joint_event_reconstruction.plotting import plot_comparison, plot_prediction

    output_dir = Path(output).expanduser().resolve()
    if len(paths) == 1:
        written = plot_prediction(paths[0], output_dir, formats=formats)
    else:
        written = []
        for path in paths:
            written.extend(
                plot_prediction(path, output_dir / Path(path).stem, formats=formats)
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
        help="Override one compact public setting using OmegaConf dot notation; repeat as needed.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topology-aware event reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train one configured experiment")
    _experiment_options(train_parser)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_parser.add_argument(
        "--initial-checkpoint",
        help="Initialise selected semantic components before training.",
    )
    train_parser.add_argument(
        "--initial-components",
        nargs="+",
        choices=tuple(_COMPONENT_PREFIXES),
        default=(),
        help="Components copied exactly from --initial-checkpoint.",
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
    predict_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
    smoke_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    smoke_parser.add_argument("--events", type=int, default=8)

    validate_parser = commands.add_parser(
        "validate-data", help="validate the configured canonical invisible HDF5"
    )
    _experiment_options(validate_parser)

    show_parser = commands.add_parser(
        "show-config", help="validate and print the fully resolved internal experiment"
    )
    _experiment_options(show_parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "evaluate":
        from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions

        print(json.dumps(evaluate_any_predictions(args.predictions), indent=2, sort_keys=True))
        return
    if args.command == "plot":
        if args.labels is not None and len(args.labels) != len(args.predictions):
            raise ValueError("--labels must match the number of prediction files")
        _plot_predictions(args.predictions, args.output, args.labels, tuple(args.formats))
        return

    overrides = getattr(args, "overrides", [])
    config = load_experiment(args.experiment, overrides)
    if args.command == "show-config":
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    if args.command == "train":
        if bool(args.initial_checkpoint) != bool(args.initial_components):
            raise ValueError(
                "--initial-checkpoint and --initial-components must be supplied together"
            )
        train(
            config,
            device=args.device,
            initial_checkpoint=args.initial_checkpoint,
            initial_components=tuple(args.initial_components),
        )
        return
    if args.command == "predict":
        predict(config, args)
        return
    if args.command == "validate-data":
        validator = _load_repo_script("data/validate_joint_h5.py", "jer_validate_joint_h5")

        if config.data.invisible.path is None:
            raise ValueError("validate-data requires invisible_truth in the JER data config")
        print(json.dumps(validator.validate(Path(config.data.invisible.path)), indent=2, sort_keys=True))
        return
    if args.command == "smoke":
        smoke_all = _load_repo_script("smoke_all.py", "jer_smoke_all")
        smoke_flows = _load_repo_script("smoke_flows.py", "jer_smoke_flows")

        device = _resolve_device(args.device)
        import torch

        torch.set_float32_matmul_precision("high")
        real_data = config.data.tensor_cache is None
        smoke_all.run_experiment_smoke(
            Path(args.experiment), device=device, events=args.events, real_data=real_data
        )
        smoke_flows.run_flow_smoke(
            Path(args.experiment), device=device, events=args.events, real_data=real_data
        )
        return
    raise RuntimeError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    main()
