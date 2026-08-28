#!/usr/bin/env python3
"""File-free CPU/CUDA smoke of every maintained reconstruction formulation.

This is a software/mechanism test only.  It constructs topology-consistent random
visible tensors and random invisible truth entirely in memory, then reuses the
same per-mode assertions as ``smoke_all.py`` and ``smoke_flows.py``.  No metric
from this script is a physics-performance result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.data.invisible import InvisibleScaler
from joint_event_reconstruction.data.structured import StructuredCollator
from joint_event_reconstruction.m0.data.module import build_token_schema
from joint_event_reconstruction.topology.compiler import load_topology
from scripts.smoke_all import MODES, run_mode as run_legacy_mode
from scripts.smoke_flows import FLOW_MODES, run_mode as run_flow_mode


def _synthetic_items(experiment, visible, *, events: int, seed: int) -> tuple[list[dict], object]:
    if events < 2:
        raise ValueError("Synthetic smoke needs at least two events for finite scaling")
    schema = build_token_schema(visible.dataset)
    spec = load_topology(experiment.topology_config)
    generator = torch.Generator().manual_seed(seed)

    source_positions: dict[str, torch.Tensor] = {}
    type_ids = []
    cursor = 0
    for source, plan in schema.sources.items():
        width = 1 if source == "GLOBAL" else int(plan.num_particles or 0)
        if width <= 0:
            continue
        positions = torch.arange(cursor, cursor + width, dtype=torch.long)
        source_positions[source] = positions
        type_ids.extend([int(schema.source_type_ids[source])] * width)
        cursor += width
    if cursor == 0:
        raise RuntimeError("Visible schema produced no synthetic token slots")

    # Assign each distinct truth label to one deterministic slot of its source.
    label_slots: dict[tuple[str, int], int] = {}
    by_source: dict[str, set[int]] = {}
    for role in spec.visible_leaf_roles:
        by_source.setdefault(role.source, set()).update(int(x) for x in role.truth_indices)
    for source, labels in by_source.items():
        if source not in source_positions:
            raise RuntimeError(f"Topology source {source} is absent from visible token schema")
        positions = source_positions[source]
        if len(labels) > len(positions):
            raise RuntimeError(
                f"Synthetic schema has {len(positions)} {source} slots but topology needs "
                f"{len(labels)} distinct truth objects"
            )
        for local, label in enumerate(sorted(labels)):
            label_slots[(source, label)] = int(positions[local])

    common_type_ids = torch.tensor(type_ids, dtype=torch.long)
    items = []
    for event in range(events):
        tokens = torch.randn((cursor, schema.dim), generator=generator)
        # Small event-dependent shifts ensure the batch is not made of clones.
        tokens += event / max(events, 1) * 0.05
        padding = torch.zeros(cursor, dtype=torch.bool)
        raw_labels = torch.full((cursor,), -100.0, dtype=torch.float32)
        for (source, label), position in label_slots.items():
            raw_labels[position] = float(label)
        items.append({
            "tokens": tokens,
            "type_ids": common_type_ids.clone(),
            "padding_mask": padding,
            "labels": torch.tensor(0.0, dtype=torch.float32),
            "event_index": torch.tensor(event, dtype=torch.long),
            "reco_raw_labels": raw_labels,
            "reco_source_ids": common_type_ids.to(torch.int8).clone(),
        })
    return items, schema


def build_synthetic_batch(experiment_path: Path, *, events: int, device: str):
    base = OmegaConf.load(experiment_path)
    visible = OmegaConf.load(base.visible_config)
    items, schema = _synthetic_items(
        base, visible, events=events, seed=int(base.experiment.seed)
    )
    collator = StructuredCollator(base.topology_config, dict(schema.source_type_ids))
    batch = collator(items)
    spec = collator.spec

    generator = torch.Generator().manual_seed(int(base.experiment.seed) + 17)
    invisible_truth = torch.randn(
        (events, len(spec.invisible_roles), 3), generator=generator
    )
    role_offsets = torch.arange(
        len(spec.invisible_roles), dtype=invisible_truth.dtype
    ).reshape(1, -1, 1)
    invisible_truth = invisible_truth + 0.25 * role_offsets
    scaler = InvisibleScaler.fit(
        invisible_truth, tuple(role.key for role in spec.invisible_roles)
    )
    batch["invisible_truth"] = invisible_truth
    batch["invisible_truth_scaled"] = scaler.transform(invisible_truth)

    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    return base, visible, move(batch), scaler


def run_synthetic_smoke(
    experiment_path: Path,
    *,
    device: str = "cpu",
    events: int = 4,
    legacy: bool = True,
    flows: bool = True,
):
    torch.set_num_threads(1)
    base, visible, batch, scaler = build_synthetic_batch(
        experiment_path, events=events, device=device
    )
    results = {"legacy": {}, "flows": {}}

    if legacy:
        for mode in MODES:
            results["legacy"][mode] = {
                "status": "PASS",
                **run_legacy_mode(base, visible, batch, scaler, mode, device),
            }
    if flows:
        for mode in FLOW_MODES:
            results["flows"][mode] = {
                "status": "PASS",
                **run_flow_mode(base, visible, batch, scaler, mode, device),
            }

    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--topology", required=True, choices=("ttbar1L", "ttbar2L", "ttH1L")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--events", type=int, default=4)
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--flows-only", action="store_true")
    args = parser.parse_args()
    if args.legacy_only and args.flows_only:
        raise ValueError("Choose at most one of --legacy-only and --flows-only")
    run_synthetic_smoke(
        Path(f"configs/experiments/{args.topology}.yaml"),
        device=args.device,
        events=args.events,
        legacy=not args.flows_only,
        flows=not args.legacy_only,
    )


if __name__ == "__main__":
    main()
