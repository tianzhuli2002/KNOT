#!/usr/bin/env python3
"""Compile a topology and validate semantic truth/candidates on real HDF5 events."""

from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
import torch

from joint_event_reconstruction.topology.assignments import build_assignment_template
from joint_event_reconstruction.topology.candidates import build_event_candidates
from joint_event_reconstruction.topology.compiler import compile_topology, load_topology
from joint_event_reconstruction.topology.decoder import ScoredCandidate, decode_topk
from joint_event_reconstruction.topology.truth import compile_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--events", type=int, default=128)
    parser.add_argument("--max-jets", type=int, default=8)
    args = parser.parse_args()
    if args.events <= 0 or args.max_jets <= 0:
        parser.error("--events and --max-jets must be positive")

    spec = load_topology(args.topology)
    split = np.load(args.split, allow_pickle=False)
    if "validation" not in split:
        raise KeyError(f"Split {args.split} has no 'validation' indices")
    event_indices = np.sort(
        np.asarray(split["validation"][:args.events], dtype=np.int64)
    )
    with h5py.File(args.h5, "r") as source:
        labels = {}
        masks = {}
        valid_indices = []
        for name in ("JET", "LEPTON"):
            label_path = f"LABELS/{name}"
            input_path = f"INPUTS/{name}"
            if label_path not in source or input_path not in source:
                raise KeyError(
                    f"Required topology data {label_path} or {input_path} is absent from {args.h5}"
                )
            labels[name] = torch.from_numpy(
                np.asarray(source[label_path][event_indices], dtype=np.float32)
            )
            objects = source[input_path][event_indices]
            if objects.dtype.names is None or "pt" not in objects.dtype.names:
                raise TypeError(
                    f"{args.h5}:{input_path} must be structured and contain pt"
                )
            mask = np.isfinite(objects["pt"]) & (objects["pt"] > 0.0)
            masks[name] = torch.from_numpy(mask)
        for event in range(len(event_indices)):
            valid_indices.append({
                "JET": tuple(
                    np.flatnonzero(
                        masks["JET"][event, :args.max_jets].numpy()
                    ).tolist()
                ),
                "LEPTON": tuple(
                    np.flatnonzero(masks["LEPTON"][event, :1].numpy()).tolist()
                ),
            })

    truth = compile_truth(
        spec,
        labels,
        masks,
        selected_widths={"JET": args.max_jets, "LEPTON": 1},
    )
    visible_full = torch.stack(
        [truth.nodes[root].visible_reconstructable for root in spec.roots], dim=1
    ).all(dim=1)
    candidate_counts = {node: [] for node in spec.dag_order}
    assignment_state_counts = []
    decoded_truth = 0
    for event, available in enumerate(valid_indices):
        pools = build_event_candidates(spec, available, truth, event=event)
        for node, pool in pools.items():
            candidate_counts[node].append(len(pool))
        assignment_state_counts.append(
            build_assignment_template(spec, pools).state_count
        )
        if visible_full[event]:
            scored = {
                root: tuple(
                    ScoredCandidate(candidate, 0.9 if candidate.truth else 0.1)
                    for candidate in pools[root]
                )
                for root in spec.roots
            }
            result = decode_topk(spec, scored, k=3)
            if not result or not result[0].truth:
                raise RuntimeError(
                    "Exact decoder failed to select synthetic truth optimum; "
                    f"event={int(event_indices[event])}"
                )
            decoded_truth += 1

    try:
        compile_topology({
            "topology": {"a": {"child": "b"}, "b": {"child": "a"}}
        })
    except ValueError as error:
        if "cycle" not in str(error):
            raise
    else:
        raise RuntimeError("Topology compiler accepted a cycle")

    summary = {
        "signature": spec.signature,
        "dag_order": spec.dag_order,
        "roots": spec.roots,
        "leaf_roles": {
            key: {
                "source": role.source,
                "truth_indices": role.truth_indices,
                "exchangeable": role.exchangeable,
            }
            for key, role in spec.leaf_roles.items()
        },
        "visible_support": dict(spec.visible_support),
        "invisible_completion": dict(spec.invisible_completion),
        "events": len(event_indices),
        "root_visible_reconstructable": int(visible_full.sum()),
        "synthetic_truth_decoded": decoded_truth,
        "candidate_count_range": {
            node: [min(counts), max(counts)]
            for node, counts in candidate_counts.items()
        },
        "complete_assignment_state_count": {
            "min": int(min(assignment_state_counts)),
            "max": int(max(assignment_state_counts)),
            "mean": float(np.mean(assignment_state_counts)),
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
