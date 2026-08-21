"""Train the topology-structured M1 event model."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf

from joint_event_reconstruction.data.structured import StructuredDataModule
from joint_event_reconstruction.training.structured import StructuredEventModule


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train",))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    experiment_path = Path(args.experiment).expanduser().resolve()
    if not experiment_path.is_file():
        raise FileNotFoundError(experiment_path)
    experiment = OmegaConf.load(experiment_path)
    visible = OmegaConf.load(experiment.visible_config)
    data = StructuredDataModule(visible, experiment.topology_config, tensor_cache=experiment.data.tensor_cache)
    model = StructuredEventModule(experiment, visible)
    output = Path(experiment.training.output).expanduser().resolve()
    if args.smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(experiment, output / "experiment.yaml")
    OmegaConf.save(visible, output / "visible_data.yaml")
    checkpoint = ModelCheckpoint(
        dirpath=output / "checkpoints", filename="best", monitor="val_loss",
        mode="min", save_top_k=1, save_last=False,
    )
    trainer = L.Trainer(
        accelerator="gpu", devices=1, precision=str(experiment.training.precision),
        max_epochs=1 if args.smoke else int(experiment.training.epochs),
        gradient_clip_val=float(experiment.training.gradient_clip),
        callbacks=[checkpoint, EarlyStopping("val_loss", mode="min", patience=20)],
        default_root_dir=output, log_every_n_steps=1 if args.smoke else 20,
        limit_train_batches=2 if args.smoke else 1.0,
        limit_val_batches=1 if args.smoke else 1.0,
    )
    trainer.fit(model, datamodule=data)


if __name__ == "__main__":
    main()
