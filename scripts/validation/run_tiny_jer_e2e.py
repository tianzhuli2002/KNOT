#!/usr/bin/env python3
"""Run one real-data JER train/reload/predict/evaluate validation cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import torch
from omegaconf import OmegaConf
from lightning.pytorch.callbacks import ModelCheckpoint

from joint_event_reconstruction.cli.run import build_data, load_trained
from joint_event_reconstruction.configuration import load_experiment, load_visible_config
from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions
from joint_event_reconstruction.prediction import predict_hdf5
from joint_event_reconstruction.training.joint import JointEventModule


def _configure_variant(config, head: str, parent_target: Path | None) -> None:
    if head not in {"none", "point", "parent", "both"}:
        raise ValueError(f"Unknown optional head variant {head!r}")
    if head in {"point", "both"}:
        config.model.heads.point_invisible.enabled = True
        config.training.trainable.append("point")
        config.training.loss_weights.point = 1.0
    if head in {"parent", "both"}:
        if parent_target is None:
            raise ValueError("Parent variant requires --parent-target")
        roles = ["top_e", "top_mu"]
        config.model.heads.parent_four_vector.enabled = True
        config.model.heads.parent_four_vector.roles = roles
        config.model.heads.parent_four_vector.target_path = str(parent_target)
        config.data.parent.path = str(parent_target)
        config.data.parent.roles = roles
        config.training.trainable.append("parent")
        config.training.loss_weights.parent = 1.0


def run(args) -> dict:
    config = load_experiment(args.config, args.overrides)
    _configure_variant(
        config,
        args.head,
        None if args.parent_target is None else args.parent_target.expanduser().resolve(),
    )
    if args.encoder == "transformer":
        config.model.visible.type = "transformer"
        config.model.visible.layers = 2
        config.model.visible.d_feedforward = 384
    if args.head in {"point", "both"}:
        # Exercise the direct point interface in this validation, including
        # the no-CFM-solver inference path.  The base tiny config deliberately
        # leaves the estimator unspecified/disabled.
        config.inference.invisible.point_estimator = "direct_point_head"

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite tiny E2E output: {output}")
    output.mkdir(parents=True)
    (output / "config.resolved.yaml").write_text(
        OmegaConf.to_yaml(config, resolve=True)
    )
    visible = load_visible_config(config)
    # A login node can report CUDA as installed while no device is allocatable.
    # Avoid pin-memory worker failures during the explicitly CPU-first gate.
    if args.device == "cpu":
        visible.dataset.pin_memory = False
        visible.dataset.persistent_workers = False
    data = build_data(config, visible)
    data.setup("fit")
    module = JointEventModule(
        config,
        visible,
        invisible_scaler=data.invisible_scaler,
        point_scaler=data.point_scaler,
    )
    if bool(args.initial_checkpoint) != bool(args.initial_components):
        raise ValueError(
            "--initial-checkpoint and --initial-components must be supplied together"
        )
    if args.initial_checkpoint is not None:
        from joint_event_reconstruction.cli.run import _load_initial_components

        _load_initial_components(
            module,
            args.initial_checkpoint.expanduser().resolve(),
            tuple(args.initial_components),
        )
    L.seed_everything(int(config.experiment.seed), workers=True)
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best",
    )
    trainer = L.Trainer(
        accelerator=args.device,
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        precision="32-true",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checkpoint],
    )
    trainer.fit(module, datamodule=data)
    checkpoint_path = Path(checkpoint.best_model_path).resolve()
    if not checkpoint_path.is_file():
        raise RuntimeError("Tiny JER training did not produce a best checkpoint")

    reloaded = load_trained(config, visible, data, str(checkpoint_path))
    prediction = output / "predictions.h5"
    predict_hdf5(
        reloaded,
        data,
        data.invisible_scaler,
        prediction,
        split="test",
        top_k=1,
        posterior_samples=int(config.inference.posterior_samples),
        device=args.device,
        prediction_seed=int(config.inference.seed),
        checkpoint_path=str(checkpoint_path),
        point_estimator=str(config.inference.invisible.point_estimator),
    )
    evaluation = evaluate_any_predictions(prediction)
    (output / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2, sort_keys=True) + "\n"
    )
    result = {
        "config": str(args.config.resolve()),
        "head": args.head,
        "encoder": args.encoder,
        "checkpoint": str(checkpoint_path),
        "prediction": str(prediction),
        "evaluation": str(output / "evaluation.json"),
        "initial_checkpoint": (
            None if args.initial_checkpoint is None
            else str(args.initial_checkpoint.expanduser().resolve())
        ),
        "initial_components": list(args.initial_components),
        "events": int(evaluation.get("events", 0)),
        "visible": evaluation.get("visible", {}).get("complete", {}),
    }
    (output / "e2e.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a compact public setting; repeat as needed.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--encoder", choices=("configured", "transformer"), default="configured")
    parser.add_argument("--head", choices=("none", "point", "parent", "both"), default="none")
    parser.add_argument("--parent-target", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument(
        "--initial-components",
        nargs="+",
        choices=("visible", "assignment", "invisible", "point", "parent"),
        default=(),
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
