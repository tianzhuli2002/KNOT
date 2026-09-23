#!/usr/bin/env python3
"""Build a small truth-labelled, detector-like ttbar2L development sample.

The partonic sidecar stores parent four-vectors made from the generated truth
decay vectors before visible smearing.  It is intentionally not derived from
the measured objects and therefore is not the production 30_4 hybrid target.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def p4(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px * px + py * py + pz * pz + mass * mass)
    return np.stack((energy, px, py, pz), axis=-1)


def add(*vectors):
    result = np.zeros_like(vectors[0])
    for vector in vectors:
        result = result + vector
    return result


def _rows(dtype, values):
    result = np.zeros(len(values), dtype=dtype)
    for name in result.dtype.names:
        result[name] = np.asarray([row[name] for row in values], dtype=result.dtype[name])
    return result


def build(output: Path, partonic_output: Path, events: int, seed: int) -> dict:
    if events < 16:
        raise ValueError("The synthetic development sample needs at least 16 events")
    if output.exists() or partonic_output.exists():
        raise FileExistsError("Refusing to overwrite a synthetic development product")
    output.parent.mkdir(parents=True, exist_ok=True)
    partonic_output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    jet_dtype = np.dtype([
        ("e", "f4"), ("eta", "f4"), ("phi", "f4"), ("pt", "f4"),
        ("btag", "f4"), ("charge", "f4"),
    ])
    lepton_dtype = np.dtype([
        ("e", "f4"), ("eta", "f4"), ("phi", "f4"), ("pt", "f4"),
        ("charge", "f4"),
    ])
    met_dtype = np.dtype([("phi", "f4"), ("pt", "f4")])
    global_dtype = np.dtype([
        ("njet", "f4"), ("nelectron", "f4"), ("nmuon", "f4"),
        ("nBJets_90", "f4"), ("nBJets_85", "f4"), ("nBJets_77", "f4"),
        ("nBJets_70", "f4"), ("nBJets_65", "f4"),
    ])
    neutrino_dtype = np.dtype([("px", "f4"), ("py", "f4"), ("pz", "f4")])

    jets = np.zeros((events, 8), dtype=jet_dtype)
    electrons = np.zeros((events, 1), dtype=lepton_dtype)
    muons = np.zeros((events, 1), dtype=lepton_dtype)
    met = np.zeros(events, dtype=met_dtype)
    global_values = np.zeros(events, dtype=global_dtype)
    jet_labels = np.zeros((events, 8), dtype=np.float32)
    electron_labels = np.ones((events, 1), dtype=np.float32)
    muon_labels = np.ones((events, 1), dtype=np.float32)
    neutrinos = np.zeros((events, 2), dtype=neutrino_dtype)
    parents = np.zeros((events, 3, 4), dtype=np.float32)
    support = np.ones((events, 3), dtype=np.bool_)
    event_ids = (8_000_000 + np.arange(events, dtype=np.int64))

    for event in range(events):
        # The generated decay vectors vary in boost and invariant mass.  The
        # visible records below are independently smeared copies.
        b_e = p4(rng.uniform(25, 160), rng.uniform(-2.1, 2.1), rng.uniform(-np.pi, np.pi), 4.8)
        b_mu = p4(rng.uniform(25, 160), rng.uniform(-2.1, 2.1), rng.uniform(-np.pi, np.pi), 4.8)
        lepton_e = p4(rng.uniform(20, 180), rng.uniform(-2.4, 2.4), rng.uniform(-np.pi, np.pi), 0.0005)
        lepton_mu = p4(rng.uniform(20, 180), rng.uniform(-2.4, 2.4), rng.uniform(-np.pi, np.pi), 0.105)
        nu_e = p4(rng.uniform(10, 150), rng.uniform(-3.0, 3.0), rng.uniform(-np.pi, np.pi), 0.0)
        nu_mu = p4(rng.uniform(10, 150), rng.uniform(-3.0, 3.0), rng.uniform(-np.pi, np.pi), 0.0)
        top_e = add(b_e, lepton_e, nu_e)
        top_mu = add(b_mu, lepton_mu, nu_mu)
        parents[event] = np.stack((top_e, top_mu, add(top_e, top_mu)))
        neutrinos[event, 0] = (nu_e[1], nu_e[2], nu_e[3])
        neutrinos[event, 1] = (nu_mu[1], nu_mu[2], nu_mu[3])

        truth_jets = (b_e, b_mu)
        for jet in range(8):
            if jet < 2:
                truth = truth_jets[jet]
                scale = rng.normal(1.0, 0.10)
                pt = max(8.0, np.hypot(truth[1], truth[2]) * scale)
                eta = np.arcsinh(truth[3] / max(np.hypot(truth[1], truth[2]), 1.0)) + rng.normal(0, 0.08)
                phi = np.arctan2(truth[2], truth[1]) + rng.normal(0, 0.05)
                btag = np.clip(rng.normal(0.82, 0.18), 0, 1)
                jet_labels[event, jet] = float(jet + 1)
            else:
                pt = rng.uniform(18, 180)
                eta = rng.uniform(-2.7, 2.7)
                phi = rng.uniform(-np.pi, np.pi)
                # A few additional jets look b-like, making the assignment
                # problem nontrivial without using truth in the inputs.
                btag = np.clip(rng.normal(0.45 if jet % 2 else 0.72, 0.25), 0, 1)
            energy = pt * np.cosh(eta)
            jets[event, jet] = (energy, eta, phi, pt, btag, rng.uniform(-0.2, 0.2))

        for target, source in ((electrons, lepton_e), (muons, lepton_mu)):
            pt = max(5.0, np.hypot(source[1], source[2]) * rng.normal(1.0, 0.03))
            eta = np.arcsinh(source[3] / max(np.hypot(source[1], source[2]), 1.0)) + rng.normal(0, 0.01)
            phi = np.arctan2(source[2], source[1]) + rng.normal(0, 0.01)
            target[event, 0] = (pt * np.cosh(eta), eta, phi, pt, 1.0 if target is electrons else -1.0)
        met_px = nu_e[1] + nu_mu[1] + rng.normal(0, 8.0)
        met_py = nu_e[2] + nu_mu[2] + rng.normal(0, 8.0)
        met[event] = (np.arctan2(met_py, met_px), np.hypot(met_px, met_py))
        global_values[event] = (8, 1, 1, 2, 2, 2, 2, 2)
        # Explicit unsupported targets exercise masking in the independent
        # validation population; placeholders remain finite and are never
        # used by the partonic loss.
        if event % 17 == 0:
            support[event, 2] = False
            parents[event, 2] = 0.0

    split = np.full(events, 2, dtype=np.uint8)
    train_end = int(round(0.60 * events))
    val_end = int(round(0.80 * events))
    split[:train_end] = 0
    split[train_end:val_end] = 1

    with h5py.File(output, "w") as target:
        target.create_dataset("EVENT_ID", data=event_ids)
        target.create_dataset("SPLIT", data=split)
        inputs = target.create_group("INPUTS")
        inputs.create_dataset("JET", data=jets)
        inputs.create_dataset("ELECTRON", data=electrons)
        inputs.create_dataset("MUON", data=muons)
        inputs.create_dataset("MET", data=met)
        inputs.create_dataset("GLOBAL", data=global_values)
        labels = target.create_group("LABELS")
        labels.create_dataset("GLOBAL", data=(np.arange(events) % 2).astype(np.int8))
        labels.create_dataset("JET", data=jet_labels)
        labels.create_dataset("ELECTRON", data=electron_labels)
        labels.create_dataset("MUON", data=muon_labels)
        labels.create_dataset("NEUTRINO", data=neutrinos)
        target.attrs["schema_version"] = "topology_query_synthetic_measured_v1"
        target.attrs["synthetic"] = True
        target.attrs["invisible_roles"] = json.dumps(
            ["top_e.nu", "top_mu.nu"]
        )
        target.attrs["physics_truth_note"] = "partonic parents are stored in a separate explicit truth sidecar"

    bindings = {
        "top_e": "synthetic_hard_process_top_e",
        "top_mu": "synthetic_hard_process_top_mu",
        "ttbar": "synthetic_hard_process_ttbar",
    }
    with h5py.File(partonic_output, "w") as target:
        target.create_dataset("EVENT_ID", data=event_ids)
        group = target.create_group("PARTONIC")
        for index, role in enumerate(("top_e", "top_mu", "ttbar")):
            group.create_dataset(role, data=parents[:, index].astype(np.float32))
        group.create_dataset("SUPPORT", data=support)
        target.attrs["schema_version"] = "topology_query_partonic_truth_v1"
        target.attrs["synthetic"] = True
        target.attrs["partonic_roles"] = json.dumps(["top_e", "top_mu", "ttbar"])
        target.attrs["partonic_truth_bindings"] = json.dumps(bindings, sort_keys=True)
        target.attrs["truth_definition"] = "generated hard-process decay-vector sums before detector-like visible smearing"

    return {
        "events": events,
        "train": int((split == 0).sum()),
        "validation": int((split == 1).sum()),
        "test": int((split == 2).sum()),
        "event_id_first": int(event_ids[0]),
        "event_id_last": int(event_ids[-1]),
        "unsupported_partonic_targets": int((~support).sum()),
        "source": str(output.resolve()),
        "partonic_truth": str(partonic_output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--partonic-output", required=True)
    parser.add_argument("--events", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260923)
    args = parser.parse_args()
    summary = build(
        Path(args.output).expanduser().resolve(),
        Path(args.partonic_output).expanduser().resolve(),
        int(args.events),
        int(args.seed),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
