"""Bounded KNOT CPU train/predict using JER YAML, data and Lightning checkpoints."""

import argparse
import copy
import json
import subprocess
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.cli.run import build_data, load_visible_config
from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.data.invisible import InvisibleScaler
from joint_event_reconstruction.data.partonic import PartonicScaler
from joint_event_reconstruction.knot import KnotModule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "predict"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--research", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--steps",
        type=int,
        help="Optional bounded optimizer-step cap; otherwise use training.epochs",
    )
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if (args.steps is not None and args.steps <= 0) or args.batches <= 0:
        parser.error("Positive steps/batches required")
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    if not output.is_relative_to(root / "results/experiments/knot"):
        raise ValueError("Use isolated results/experiments/knot output")
    output.mkdir(parents=True, exist_ok=False)
    config = load_experiment(args.config)
    research = OmegaConf.to_container(OmegaConf.load(args.research), resolve=True)
    # The parent CFM flag tells build_data to load and scale parent truth. KNOT
    # owns its P0/P2/J continuous generator, so its base EventModel must not
    # also construct the separate standalone parent CFM.
    knot_config = copy.deepcopy(config)
    if knot_config.model.get("parent_cfm") is not None:
        knot_config.model.parent_cfm.enabled = False
    torch.manual_seed(int(config.experiment.seed))
    visible = load_visible_config(config)
    # Development runner deliberately accepts train/validation-only inputs.
    input_path = (
        config.data.get("inference_input") if args.action == "predict" else None
    )
    input_path = input_path or config.data.input
    with h5py.File(input_path, "r") as f:
        if "SPLIT" not in f or not np.isin(f["SPLIT"][:], (0, 1)).all():
            raise ValueError(
                "Provide a development HDF5 with explicit train/validation SPLIT only"
            )
    if args.action == "train":
        data = build_data(config, visible, device="cpu", complete_assignments=True)
        data.setup("fit")
        torch.manual_seed(int(config.experiment.seed))
        module = KnotModule(
            knot_config,
            visible,
            research=research,
            invisible_scaler=data.invisible_scaler,
            partonic_scaler=data.partonic_scaler,
            synthetic_data=data.structured_collator.invisible_store.synthetic,
        )
        trainer = L.Trainer(
            accelerator=args.accelerator,
            devices=1,
            max_steps=args.steps if args.steps is not None else -1,
            max_epochs=args.steps if args.steps is not None else int(config.training.epochs),
            precision=str(config.training.precision),
            num_sanity_val_steps=0,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            default_root_dir=output,
        )
        # Reset after model initialization so M/C/J start with the same
        # shuffled training order under their common experiment seed.
        torch.manual_seed(int(config.experiment.seed))
        trainer.fit(
            module,
            train_dataloaders=data.train_dataloader(),
            val_dataloaders=data.val_dataloader(),
        )
        trainer.save_checkpoint(output / "model.ckpt")
    else:
        if not args.checkpoint:
            parser.error("--checkpoint required for prediction")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        module = KnotModule(
            knot_config,
            visible,
            research=research,
            invisible_scaler=InvisibleScaler.from_dict(saved["invisible_scaler"]),
            partonic_scaler=PartonicScaler.from_dict(saved["partonic_scaler"]),
            synthetic_data=bool(config.data.invisible.allow_synthetic),
        )
        module.validate_checkpoint_metadata(saved)
        module.load_state_dict(saved["state_dict"])
        module.eval()
        data = build_data(
            config,
            visible,
            include_truth=False,
            device="cpu",
            complete_assignments=True,
        )
        data.setup("fit")
        generator = torch.Generator().manual_seed(int(config.experiment.seed))
        for index, batch in enumerate(data.val_dataloader()):
            if index >= args.batches:
                break
            module.posterior(batch, samples=args.samples, generator=generator).write(
                output / f"posterior_batch_{index}.h5"
            )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=root, text=True
                ).strip(),
                "branch": subprocess.check_output(
                    ["git", "branch", "--show-current"], cwd=root, text=True
                ).strip(),
                "worktree": str(root),
                "config": str(Path(args.config).resolve()),
                "research": research,
                "input": input_path,
                "action": args.action,
                "bounded_steps": args.steps,
                "configured_epochs": int(config.training.epochs),
                "accelerator": args.accelerator,
                "sealed_test_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
