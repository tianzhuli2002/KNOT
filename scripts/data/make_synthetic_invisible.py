#!/usr/bin/env python3
"""Make a deliberately shuffled invisible sidecar for pipeline tests only."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.topology.compiler import load_topology


DTYPE = np.dtype([("px", "<f4"), ("py", "<f4"), ("pz", "<f4")])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-events", type=int)
    args = parser.parse_args()
    experiment_path = Path(args.experiment).resolve()
    config = OmegaConf.load(experiment_path)
    visible_path = (experiment_path.parent.parent.parent / config.visible_config).resolve()
    visible = OmegaConf.load(visible_path)
    topology_path = (experiment_path.parent.parent.parent / config.topology_config).resolve()
    spec = load_topology(topology_path)
    cache_path = Path(config.data.tensor_cache).expanduser().resolve()
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    selected_cache_ids = np.asarray(payload["indices"], dtype=np.int64)
    split_file = np.load(Path(visible.dataset.split.indices_path).expanduser().resolve())
    chosen = []
    for offset, split in enumerate(("train", "validation", "test")):
        split_ids = np.asarray(split_file[split], dtype=np.int64)
        configured = config.data.max_events["val" if split == "validation" else split]
        count = args.max_events if args.max_events is not None else configured
        if count is not None and int(count) < len(split_ids):
            rng_split = np.random.default_rng(int(config.experiment.seed) + offset)
            positions = np.sort(rng_split.choice(len(split_ids), size=int(count), replace=False))
            split_ids = split_ids[positions]
        chosen.append(split_ids)
    event_ids = np.unique(np.concatenate(chosen))
    positions = np.searchsorted(selected_cache_ids, event_ids)
    if (positions >= len(selected_cache_ids)).any() or not np.array_equal(selected_cache_ids[positions], event_ids):
        raise ValueError("Stable split contains EVENT_ID values absent from the configured tensor cache")
    source_h5 = Path(visible.dataset.input_h5_path).expanduser().resolve()
    with h5py.File(source_h5, "r") as source:
        met = source["INPUTS/MET"][event_ids, 0]
        lepton = source["INPUTS/LEPTON"][event_ids, 0]
    if "phi" in met.dtype.names:
        met_cos, met_sin = np.cos(met["phi"]), np.sin(met["phi"])
    elif {"sin_phi", "cos_phi"}.issubset(met.dtype.names):
        met_cos, met_sin = met["cos_phi"], met["sin_phi"]
    else:
        raise TypeError(f"{source_h5}: INPUTS/MET has no canonical phi representation")
    rng = np.random.default_rng(args.seed)
    role_count = len(spec.invisible_roles)
    dense = np.empty((len(event_ids), role_count, 3), dtype=np.float32)
    for role_index in range(role_count):
        offset = np.float32(0.05 * (role_index + 1))
        dense[:, role_index, 0] = met["pt"] * met_cos + offset * lepton["pt"]
        dense[:, role_index, 1] = met["pt"] * met_sin - offset * lepton["pt"]
        dense[:, role_index, 2] = (
            0.35 * lepton["pt"] * np.sinh(lepton["eta"])
            + rng.normal(0.0, 2.0 + role_index, len(event_ids))
        )
    structured = np.empty((len(event_ids), role_count), dtype=DTYPE)
    for index, name in enumerate(("px", "py", "pz")):
        structured[name] = dense[:, :, index]
    order = rng.permutation(len(event_ids))
    if len(order) > 1 and np.array_equal(order, np.arange(len(order))):
        order = np.roll(order, 1)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with h5py.File(temporary, "w") as target:
        target.create_dataset("EVENT_ID", data=event_ids[order])
        labels = target.create_group("LABELS")
        labels.create_dataset("NEUTRINO", data=structured[order])
        target.attrs["synthetic"] = True
        target.attrs["purpose"] = "PIPELINE_TEST_ONLY_NOT_PHYSICS"
        target.attrs["seed"] = args.seed
        target.attrs["source_visible_h5"] = str(source_h5)
        target.attrs["identity_semantics"] = "source_hdf5_row_index"
        target.attrs["rows_deliberately_shuffled"] = True
    temporary.replace(output)
    print(f"Wrote {len(event_ids)} shuffled synthetic events with {role_count} invisible role(s) to {output}")


if __name__ == "__main__":
    main()
