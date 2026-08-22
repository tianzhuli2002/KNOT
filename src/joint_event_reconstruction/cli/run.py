"""One maintained command line for joint training and bounded smoke execution."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf

from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.structured import StructuredDataModule
from joint_event_reconstruction.training.joint import JointEventModule


def build_data(config, visible):
    invisible = config.data.invisible
    if config.model.invisible.enabled and invisible.path is None:
        raise ValueError("Invisible inference is enabled but data.invisible.path is null")
    return StructuredDataModule(
        visible, config.topology_config, tensor_cache=config.data.tensor_cache,
        invisible_path=invisible.path if config.model.invisible.enabled else None,
        allow_synthetic=bool(invisible.allow_synthetic),
        max_events=OmegaConf.to_container(config.data.max_events, resolve=True),
        subset_seed=int(config.experiment.seed),
    )


def train(config) -> None:
    visible = OmegaConf.load(config.visible_config)
    data = build_data(config, visible)
    data.setup("fit")
    module = JointEventModule(
        config, visible, invisible_scaler=data.invisible_scaler,
        synthetic_data=bool(data.structured_collator.invisible_store.synthetic)
        if data.structured_collator.invisible_store is not None else False,
    )
    callbacks = [ModelCheckpoint(
        dirpath=Path(config.training.output) / "checkpoints", monitor="val_loss", mode="min",
        save_top_k=1, filename="best-loss",
    )]
    if config.training.early_stopping.enabled:
        callbacks.append(EarlyStopping(
            monitor="val_loss", mode="min", patience=int(config.training.early_stopping.patience),
        ))
    trainer = L.Trainer(
        max_epochs=int(config.training.epochs), precision=str(config.training.precision),
        gradient_clip_val=float(config.training.gradient_clip),
        accumulate_grad_batches=int(config.training.accumulate_grad_batches), callbacks=callbacks,
        default_root_dir=config.training.output,
    )
    trainer.fit(module, datamodule=data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "smoke"))
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    config = load_experiment(args.experiment)
    if args.command == "train":
        train(config)
    else:
        from scripts.smoke_all import run_experiment_smoke
        run_experiment_smoke(Path(args.experiment), device="cpu", events=8)


if __name__ == "__main__":
    main()
