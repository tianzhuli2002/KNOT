#!/usr/bin/env python3
"""Build the explicitly named parent target supported by the canonical 30_4 data.

The source files do not contain generator-level top four-vectors.  This product
therefore constructs a hybrid target from the selected reconstructed visible
objects and the stored truth neutrino momenta.  It must never be described as
generator-level parent truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


ROLES = ("top_e", "top_mu")


def _p4(values: np.ndarray) -> np.ndarray:
    energy = np.asarray(values["e"], dtype=np.float32)
    pt = np.asarray(values["pt"], dtype=np.float32)
    phi = np.asarray(values["phi"], dtype=np.float32)
    eta = np.asarray(values["eta"], dtype=np.float32)
    return np.stack(
        (energy, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)),
        axis=-1,
    ).astype(np.float32, copy=False)


def build(source_path: Path, target_path: Path, chunk_size: int) -> None:
    source_path = source_path.expanduser().resolve()
    target_path = target_path.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if target_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing parent target: {target_path}"
        )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = target_path.with_suffix(target_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    with h5py.File(source_path, "r") as source:
        required = (
            "EVENT_ID",
            "INPUTS/JET",
            "INPUTS/ELECTRON",
            "INPUTS/MUON",
            "LABELS/JET",
            "LABELS/ELECTRON",
            "LABELS/MUON",
            "LABELS/NEUTRINO",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"{source_path} is missing required parent-target fields: {missing}")
        count = len(source["EVENT_ID"])
        with h5py.File(temporary, "w") as target:
            target.attrs.update(
                {
                    "schema_version": "joint_event_reconstruction_parent_truth_v1",
                    "parent_roles": json.dumps(list(ROLES)),
                    "target_semantics": "hybrid_visible_reconstructed_plus_truth_invisible_v1",
                    "generator_parent_truth_available": False,
                    "visible_target": "selected INPUTS/ELECTRON,MUON and LABELS/JET=1,2",
                    "invisible_target": "LABELS/NEUTRINO slots 0=e,1=mu",
                    "four_vector_representation": "E_px_py_pz_GeV_v1",
                    "source_file": str(source_path),
                    "source_schema": str(source.attrs.get("schema_version", "unknown")),
                }
            )
            target.create_dataset("EVENT_ID", shape=(count,), dtype="<i8")
            parents = target.create_group("PARENTS")
            for role in ROLES:
                parents.create_dataset(
                    role,
                    shape=(count, 4),
                    dtype="<f4",
                    chunks=(min(chunk_size, max(count, 1)), 4),
                )
            for start in range(0, count, chunk_size):
                stop = min(count, start + chunk_size)
                event_ids = np.asarray(source["EVENT_ID"][start:stop], dtype=np.int64)
                jets = _p4(source["INPUTS/JET"][start:stop])
                electron = _p4(source["INPUTS/ELECTRON"][start:stop, :1])[:, 0]
                muon = _p4(source["INPUTS/MUON"][start:stop, :1])[:, 0]
                jet_labels = np.asarray(source["LABELS/JET"][start:stop])
                electron_labels = np.asarray(source["LABELS/ELECTRON"][start:stop])
                muon_labels = np.asarray(source["LABELS/MUON"][start:stop])
                electron_index = (jet_labels == 1).nonzero()[1]
                muon_index = (jet_labels == 2).nonzero()[1]
                electron_counts = (jet_labels == 1).sum(axis=1)
                muon_counts = (jet_labels == 2).sum(axis=1)
                if (
                    len(electron_index) != len(event_ids)
                    or len(muon_index) != len(event_ids)
                    or not np.all(electron_counts == 1)
                    or not np.all(muon_counts == 1)
                ):
                    raise ValueError(
                        f"{source_path}: chunk {start}:{stop} does not have one JET label 1 and 2 per event"
                    )
                if not np.all(electron_labels == 1) or not np.all(muon_labels == 1):
                    raise ValueError(
                        f"{source_path}: selected lepton labels are not exactly one per event"
                    )
                jet_e = jets[np.arange(len(event_ids)), electron_index]
                jet_mu = jets[np.arange(len(event_ids)), muon_index]
                neutrino = np.asarray(source["LABELS/NEUTRINO"][start:stop])
                neutrino_energy = np.sqrt(
                    neutrino["px"] ** 2 + neutrino["py"] ** 2 + neutrino["pz"] ** 2
                ).astype(np.float32)
                neutrino_p4 = np.stack(
                    (
                        neutrino_energy,
                        neutrino["px"],
                        neutrino["py"],
                        neutrino["pz"],
                    ),
                    axis=-1,
                ).astype(np.float32)
                top_e = jet_e + electron + neutrino_p4[:, 0]
                top_mu = jet_mu + muon + neutrino_p4[:, 1]
                if not np.isfinite(top_e).all() or not np.isfinite(top_mu).all():
                    raise ValueError(f"{source_path}: non-finite parent target in {start}:{stop}")
                if (top_e[:, 0] < 0).any() or (top_mu[:, 0] < 0).any():
                    raise ValueError(f"{source_path}: negative parent energy in {start}:{stop}")
                target["EVENT_ID"][start:stop] = event_ids
                parents["top_e"][start:stop] = top_e
                parents["top_mu"][start:stop] = top_mu
    os.replace(temporary, target_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=131072)
    args = parser.parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    build(args.source, args.output, args.chunk_size)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
