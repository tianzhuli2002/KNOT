#!/usr/bin/env python3
"""Reload an existing JER checkpoint, predict a bounded split and evaluate it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
from omegaconf import OmegaConf

from joint_event_reconstruction.cli.run import build_data, load_trained, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions
from joint_event_reconstruction.prediction import predict_hdf5
from joint_event_reconstruction.prediction_schema import validate_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    config = load_experiment(args.config.expanduser().resolve(), args.overrides)
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reload validation output: {output}")
    output.mkdir(parents=True)
    (output / "config.resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    visible = load_visible_config(config)
    if args.device == "cpu":
        visible.dataset.pin_memory = False
        visible.dataset.persistent_workers = False
    data = build_data(config, visible)
    data.setup("predict")
    L.seed_everything(int(config.inference.seed), workers=True)
    module = load_trained(config, visible, data, str(checkpoint))
    prediction = output / "predictions.h5"
    predict_hdf5(
        module,
        data,
        data.invisible_scaler,
        prediction,
        split="test",
        top_k=int(config.inference.top_k_assignments),
        posterior_samples=int(config.inference.posterior_samples),
        device=args.device,
        prediction_seed=int(config.inference.seed),
        checkpoint_path=str(checkpoint),
        compatibility_weight=float(config.inference.compatibility_weight),
        point_estimator=str(config.inference.invisible.point_estimator),
    )
    validate_prediction(prediction, deep=True)
    evaluation = evaluate_any_predictions(prediction)
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    result = {
        "config": str(args.config.expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "prediction": str(prediction),
        "evaluation": str(output / "evaluation.json"),
        "events": int(evaluation.get("events", 0)),
        "schema_valid": True,
        "reloaded": True,
    }
    (output / "e2e.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
