"""Minimal M0 train, predict and performance interface."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from joint_event_reconstruction.m0.data.module import TransformerDataModule
from joint_event_reconstruction.m0.model import TransformerModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("train", "predict", "evaluate"))
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"M0 configuration does not exist: {config_path}")
    config = OmegaConf.load(config_path)
    data = TransformerDataModule(config)
    model = TransformerModel(config)
    if args.command == "train":
        from joint_event_reconstruction.m0.training import train

        train(data, model, config)
    elif args.command == "predict":
        from joint_event_reconstruction.m0.prediction import predict

        predict(data, model, config)
    else:
        from joint_event_reconstruction.m0.evaluation.performance import run_performance

        run_performance(None, None, config)


if __name__ == "__main__":
    main()
