"""Create a clearly synthetic, bounded P2 plumbing fixture from measured MET."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def create(measured_path: Path, output_path: Path, seed: int) -> dict:
    with h5py.File(measured_path, "r") as source:
        event_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        met = source["INPUTS/MET"][:]
    px = met["pt"][:, 0] * np.cos(met["phi"][:, 0])
    py = met["pt"][:, 0] * np.sin(met["phi"][:, 0])
    rng = np.random.default_rng(seed)
    split_x = 0.5 * px + rng.normal(0.0, 12.0, len(px))
    split_y = 0.5 * py + rng.normal(0.0, 12.0, len(py))
    pz_top = rng.normal(0.0, 60.0, len(px))
    pz_antitop = rng.normal(0.0, 60.0, len(px))
    neutrino_dtype = np.dtype([("px", "<f4"), ("py", "<f4"), ("pz", "<f4")])
    values = np.empty((len(event_ids), 2), dtype=neutrino_dtype)
    values["px"][:, 0] = split_x
    values["py"][:, 0] = split_y
    values["pz"][:, 0] = pz_top
    values["px"][:, 1] = px - split_x
    values["py"][:, 1] = py - split_y
    values["pz"][:, 1] = pz_antitop
    if not np.isfinite(np.stack([values[name] for name in ("px", "py", "pz")], axis=-1)).all():
        raise ValueError("Synthetic neutrino fixture contains non-finite values")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with h5py.File(temporary, "w") as output:
            output.create_dataset("EVENT_ID", data=event_ids, dtype="<i8")
            output.create_dataset("LABELS/NEUTRINO", data=values, dtype=neutrino_dtype)
            support = output.create_dataset(
                "SUPPORT/INVISIBLE_ROLE",
                data=np.ones((len(event_ids), 2), dtype=np.bool_),
                dtype=np.bool_,
            )
            support.attrs["role_order"] = json.dumps(["top.nu", "antitop.nu"])
            support.attrs["definition"] = "all synthetic fixture role targets are supported"
            output.attrs["synthetic"] = True
            output.attrs["invisible_roles"] = json.dumps(["top.nu", "antitop.nu"])
            output.attrs["units"] = "GeV"
            output.attrs["components"] = "px,py,pz"
            output.attrs["truth_definition"] = (
                "synthetic P2 wiring fixture: MET transverse vector randomly partitioned "
                "between charge-side roles and independent Gaussian pz values"
            )
            output.attrs["seed"] = int(seed)
            output.attrs["source_measured_file"] = str(measured_path.resolve())
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "events": len(event_ids),
        "output": str(output_path.resolve()),
        "seed": int(seed),
        "truth_definition": "synthetic wiring fixture only; not a physical neutrino mapping",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=40821)
    args = parser.parse_args()
    print(json.dumps(create(args.measured, args.output, args.seed), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
