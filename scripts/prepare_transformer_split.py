#!/usr/bin/env python3
"""Create or validate one exact selected-subset Transformer split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import hydra

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.m0.data.module import validate_transformer_task_config
from joint_event_reconstruction.m0.data.labels import (
    read_transformer_classification_labels,
    resolve_transformer_label_path,
)
from joint_event_reconstruction.m0.data.splits import load_or_create_split, split_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or validate the deterministic Transformer split.",
        epilog="Example: python scripts/prepare_transformer_split.py transformer_ttH_joint --output data/ttH_split.npz",
    )
    parser.add_argument("config_name", help="Hydra Transformer configuration name.")
    parser.add_argument("--output", default=None, help="Split NPZ path or directory; defaults to dataset.split.indices_path")
    args = parser.parse_args()

    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name=args.config_name)
    validate_transformer_task_config(cfg)

    input_value = cfg.dataset.get("input_h5_path", None)
    if input_value in {None, ""}:
        raise ValueError("A Transformer input H5 path is required through dataset.input_h5_path.")
    source_path = Path(str(input_value)).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Transformer source H5 does not exist: {source_path}")
    with h5py.File(source_path, "r") as handle:
        label_path = resolve_transformer_label_path(handle)
        try:
            labels = read_transformer_classification_labels(handle)
        except ValueError as exc:
            raise ValueError(f"Invalid classification labels in {source_path}:{label_path}: {exc}") from exc

    split_cfg = cfg.dataset.get("split", {}) or {}
    output = args.output or split_cfg.get("indices_path", None)
    if output in {None, ""}:
        raise ValueError("A split output path is required through --output or dataset.split.indices_path.")
    arrays, metadata = load_or_create_split(
        output,
        labels,
        source_path=source_path,
        seed=int(cfg.general.seed),
        limit_events=cfg.dataset.get("limit_events", None),
        test_size=float(cfg.dataset.get("test_size", 0.2)),
        require_existing=False,
    )
    npz_path, metadata_path = split_paths(output)
    summary = {
        "config_name": args.config_name,
        "source_h5": str(source_path),
        "label_path": label_path,
        "split_indices": str(npz_path),
        "split_metadata": str(metadata_path),
        "seed": int(cfg.general.seed),
        "limit_events": cfg.dataset.get("limit_events", None),
        "test_size": float(cfg.dataset.get("test_size", 0.2)),
        "sizes": {name: int(len(values)) for name, values in arrays.items()},
        "index_hashes": metadata["index_hashes"],
    }
    summary.update({f"{name}_size": int(len(arrays[name])) for name in ("selected", "train", "validation", "test")})
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
