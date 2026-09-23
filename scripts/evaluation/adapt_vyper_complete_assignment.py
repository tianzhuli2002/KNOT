#!/usr/bin/env python3
"""Convert VyPER's edge scores into the common complete b_e,b_mu metric.

Pinned upstream VyPER predicts a binary b--lepton edge score.  Its native
writer does not emit EVENT_ID and does not impose the one-to-one two-branch
assignment.  This adapter therefore joins rows to the immutable common view,
maximises the summed log positive-edge score over distinct jet pairs, and
compares that pair with canonical JET1/JET2 truth.  It never uses truth to
select a prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def _edge_lookup(edge_index: np.ndarray, edge_scores: np.ndarray, n_nodes: int) -> dict[tuple[int, int], float]:
    if edge_index.shape != (2, edge_scores.shape[1]):
        raise ValueError(
            "VyPER edge index/probability shapes disagree: "
            f"index={edge_index.shape}, scores={edge_scores.shape}"
        )
    result = {}
    for column in range(edge_index.shape[1]):
        first, second = (int(edge_index[0, column]), int(edge_index[1, column]))
        if not 0 <= first < n_nodes or not 0 <= second < n_nodes or first == second:
            raise ValueError(f"VyPER emitted an invalid local edge ({first},{second})")
        key = tuple(sorted((first, second)))
        if key in result:
            raise ValueError(f"VyPER emitted duplicate reduced edge {key}")
        result[key] = float(edge_scores[0, column])
    return result


def evaluate(prediction: Path, view: Path) -> dict:
    with h5py.File(prediction, "r") as pred, h5py.File(view, "r") as source:
        if "VyPER/EdgeIndex" not in pred or "VyPER/EdgeSoftP" not in pred:
            raise KeyError(f"{prediction} must contain VyPER/EdgeIndex and VyPER/EdgeSoftP")
        if "EVENT_ID" not in source or "INPUTS/JET" not in source or "LABELS/JET" not in source:
            raise KeyError(f"{view} is missing the common event/jet contract")
        edge_indices = pred["VyPER/EdgeIndex"]
        edge_scores = pred["VyPER/EdgeSoftP"]
        event_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        jets = source["INPUTS/JET"]
        labels = np.asarray(source["LABELS/JET"][:], dtype=np.float32)
        if len(edge_indices) != len(event_ids) or len(edge_scores) != len(event_ids):
            raise ValueError(
                f"VyPER row count does not match the common view: prediction="
                f"({len(edge_indices)},{len(edge_scores)}), view={len(event_ids)}"
            )

        topk_hits = {k: 0 for k in (1, 2, 4, 8)}
        supported = 0
        top1 = 0
        predicted = []
        neutrino_shape = None
        for row, event_id in enumerate(event_ids):
            njet = int(np.isfinite(jets["pt"][row]).sum())
            if njet < 2:
                continue
            truth_e = np.flatnonzero(labels[row, :njet] == 1)
            truth_mu = np.flatnonzero(labels[row, :njet] == 2)
            if len(truth_e) != 1 or len(truth_mu) != 1 or int(truth_e[0]) == int(truth_mu[0]):
                continue
            # HDF5 variable-length rows are object arrays whose channels are
            # separate one-dimensional arrays; stack explicitly before shape
            # validation.
            scores = np.stack([np.asarray(channel, dtype=np.float64) for channel in edge_scores[row]])
            indices = np.stack([np.asarray(channel, dtype=np.int64) for channel in edge_indices[row]])
            if scores.ndim != 2 or scores.shape[0] != 2:
                raise ValueError(
                    "This adapter requires upstream's binary b--lepton edge head "
                    f"(channels=2); event={int(event_id)}, shape={scores.shape}"
                )
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError(f"Non-finite or non-probability VyPER edge scores at event={int(event_id)}")
            if not np.allclose(scores.sum(axis=0), 1.0, atol=2.0e-5, rtol=2.0e-5):
                raise ValueError(f"VyPER edge probabilities do not close at event={int(event_id)}")
            lookup = _edge_lookup(indices, scores, njet + 2)
            electron_node, muon_node = njet, njet + 1
            candidates = []
            for electron_jet in range(njet):
                e_key = tuple(sorted((electron_jet, electron_node)))
                if e_key not in lookup:
                    raise KeyError(f"Missing VyPER jet-electron edge {e_key} at event={int(event_id)}")
                for muon_jet in range(njet):
                    if electron_jet == muon_jet:
                        continue
                    mu_key = tuple(sorted((muon_jet, muon_node)))
                    if mu_key not in lookup:
                        raise KeyError(f"Missing VyPER jet-muon edge {mu_key} at event={int(event_id)}")
                    p_e = lookup[e_key]
                    p_mu = lookup[mu_key]
                    if p_e <= 0.0 or p_mu <= 0.0:
                        raise ValueError(f"Zero positive-edge probability at event={int(event_id)}")
                    candidates.append((float(np.log(p_e) + np.log(p_mu)), electron_jet, muon_jet))
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            truth_pair = (int(truth_e[0]), int(truth_mu[0]))
            rank = next(
                (position for position, (_, ejet, mujet) in enumerate(candidates) if (ejet, mujet) == truth_pair),
                None,
            )
            if rank is None:
                raise RuntimeError(f"Truth assignment was absent from VyPER candidate list at event={int(event_id)}")
            supported += 1
            top1 += int(rank == 0)
            for k in topk_hits:
                topk_hits[k] += int(rank < k)
            predicted.append((int(event_id), int(candidates[0][1]), int(candidates[0][2]), int(rank)))

        if "VyPER/Neutrino" in pred:
            first = pred["VyPER/Neutrino"][0]
            neutrino_shape = [list(np.asarray(item).shape) for item in first]
            for row in pred["VyPER/Neutrino"]:
                values = np.stack([np.asarray(item, dtype=np.float32) for item in row])
                if values.shape != (2, 3) or not np.isfinite(values).all():
                    raise ValueError("VyPER neutrino output is not finite [electron,muon,3]")

    result = {
        "prediction": str(prediction.resolve()),
        "common_view": str(view.resolve()),
        "edge_contract": "binary_b_lepton_scores_to_distinct_jet_pair_v1",
        "neutrino_contract": "VyPER.Neutrino rows are [electron_side, muon_side] Cartesian GeV",
        "events": int(len(event_ids)),
        "supported_events": int(supported),
        "complete_assignment_top1": None if supported == 0 else float(top1 / supported),
        "complete_assignment_topk": {
            str(k): None if supported == 0 else float(value / supported)
            for k, value in topk_hits.items()
        },
        "neutrino_first_event_shape": neutrino_shape,
        "truth_independent_selection": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.prediction.expanduser().resolve(), args.view.expanduser().resolve())
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
