#!/usr/bin/env python3

import argparse
from pathlib import Path

import h5py
import numpy as np


def make_nan_structured(shape, dtype):
    arr = np.empty(shape, dtype=dtype)
    for field in dtype.names:
        arr[field].fill(np.nan)
    return arr


def get_field(arr, *names, required=True):
    if arr.dtype.names is None:
        raise ValueError(f"Expected structured array, got dtype={arr.dtype}")

    for name in names:
        if name in arr.dtype.names:
            return np.asarray(arr[name])

    if required:
        raise KeyError(f"Could not find any of fields {names}. Available: {arr.dtype.names}")

    return None


def finite_positive(x):
    return np.isfinite(x) & (x > 0.0)


def main():
    parser = argparse.ArgumentParser(
        description="Convert old ttH/ttbb HDF5 schema to Transformer/HyPER-like node schema."
    )
    parser.add_argument("input_h5", help="Input HDF5 file")
    parser.add_argument("output_h5", help="Output converted HDF5 file")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_h5)
    output_path = Path(args.output_h5)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    global_dtype = np.dtype(
        [
            ("njet", np.float32),
            ("nBjets", np.float32),
        ]
    )

    object_dtype = np.dtype(
        [
            ("e", np.float32),
            ("eta", np.float32),
            ("sin_phi", np.float32),
            ("cos_phi", np.float32),
            ("pt", np.float32),
            ("btag", np.float32),
            ("charge", np.float32),
            ("id", np.float32),
        ]
    )

    target_widths = {
        "JET": 20,
        "LEPTON": 3,
        "MET": 1,
    }

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        required_paths = [
            "INPUTS/GLOBAL",
            "INPUTS/JET",
            "INPUTS/LEPTON",
            "INPUTS/MET",
            "LABELS/CLASS",
        ]

        missing = [p for p in required_paths if p not in src]
        if missing:
            raise KeyError(f"Input file is missing required paths: {missing}")

        src_global = src["INPUTS/GLOBAL"]
        src_jet = src["INPUTS/JET"]
        src_lepton = src["INPUTS/LEPTON"]
        src_met = src["INPUTS/MET"]
        src_class = src["LABELS/CLASS"]

        n_events = src_global.shape[0]

        for name, ds in [
            ("INPUTS/JET", src_jet),
            ("INPUTS/LEPTON", src_lepton),
            ("INPUTS/MET", src_met),
            ("LABELS/CLASS", src_class),
        ]:
            if ds.shape[0] != n_events:
                raise ValueError(f"{name} has {ds.shape[0]} events, expected {n_events}")

        if src_jet.shape[1] > target_widths["JET"]:
            raise ValueError(f"Input has {src_jet.shape[1]} jets, target width is {target_widths['JET']}")

        if src_lepton.shape[1] > target_widths["LEPTON"]:
            raise ValueError(f"Input has {src_lepton.shape[1]} leptons, target width is {target_widths['LEPTON']}")

        if src_met.shape[1] > target_widths["MET"]:
            raise ValueError(f"Input has {src_met.shape[1]} MET objects, target width is {target_widths['MET']}")

        inputs = dst.create_group("INPUTS")
        labels = dst.create_group("LABELS")

        chunk_events = min(args.chunk_size, n_events)

        out_global = inputs.create_dataset(
            "GLOBAL",
            shape=(n_events, 1),
            dtype=global_dtype,
            chunks=(chunk_events, 1),
            compression="gzip",
            compression_opts=4,
        )

        out_jet = inputs.create_dataset(
            "JET",
            shape=(n_events, target_widths["JET"]),
            dtype=object_dtype,
            chunks=(chunk_events, target_widths["JET"]),
            compression="gzip",
            compression_opts=4,
        )

        out_lepton = inputs.create_dataset(
            "LEPTON",
            shape=(n_events, target_widths["LEPTON"]),
            dtype=object_dtype,
            chunks=(chunk_events, target_widths["LEPTON"]),
            compression="gzip",
            compression_opts=4,
        )

        out_met = inputs.create_dataset(
            "MET",
            shape=(n_events, target_widths["MET"]),
            dtype=object_dtype,
            chunks=(chunk_events, target_widths["MET"]),
            compression="gzip",
            compression_opts=4,
        )

        out_label_global = labels.create_dataset(
            "GLOBAL",
            shape=(n_events, 1),
            dtype=np.float64,
            chunks=(chunk_events, 1),
            compression="gzip",
            compression_opts=4,
        )

        # JET labels:
        #   NaN = no jet exists / padded slot
        #   0   = real jet exists but original label was -1 / unmatched
        #   1+  = original assigned labels
        out_label_jet = labels.create_dataset(
            "JET",
            shape=(n_events, target_widths["JET"]),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=(chunk_events, target_widths["JET"]),
            compression="gzip",
            compression_opts=4,
        )

        # LEPTON labels:
        #   1   = real lepton exists
        #   NaN = padded slot
        out_label_lepton = labels.create_dataset(
            "LEPTON",
            shape=(n_events, target_widths["LEPTON"]),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=(chunk_events, target_widths["LEPTON"]),
            compression="gzip",
            compression_opts=4,
        )

        # MET labels:
        #   1   = real MET exists
        #   NaN = missing MET
        out_label_met = labels.create_dataset(
            "MET",
            shape=(n_events, target_widths["MET"]),
            dtype=np.float64,
            fillvalue=np.nan,
            chunks=(chunk_events, target_widths["MET"]),
            compression="gzip",
            compression_opts=4,
        )

        print(f"Converting {n_events} events")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")

        for start in range(0, n_events, args.chunk_size):
            stop = min(start + args.chunk_size, n_events)
            n = stop - start

            g = src_global[start:stop]
            j = src_jet[start:stop]
            l = src_lepton[start:stop]
            m = src_met[start:stop]
            y = src_class[start:stop]

            # ====================================================
            # GLOBAL
            # ====================================================

            converted_global = np.zeros((n, 1), dtype=global_dtype)

            converted_global["njet"][:, 0] = (
                get_field(g, "njet", "Njets")
                .reshape(n, -1)[:, 0]
                .astype(np.float32)
            )

            converted_global["nBjets"][:, 0] = (
                get_field(g, "nBjets", "Nbjets", "nbjets")
                .reshape(n, -1)[:, 0]
                .astype(np.float32)
            )

            out_global[start:stop] = converted_global

            # ====================================================
            # JET
            #
            # Existing jet:
            #   e/eta/pt/btag/sin_phi/cos_phi copied/computed
            #   charge = 0 because unavailable
            #   id = 1
            #
            # Missing/padded jet:
            #   all fields remain NaN
            # ====================================================

            converted_jets = make_nan_structured((n, target_widths["JET"]), object_dtype)
            jw = j.shape[1]

            jet_e = get_field(j, "e").astype(np.float32)
            jet_eta = get_field(j, "eta").astype(np.float32)
            jet_pt = get_field(j, "pt").astype(np.float32)
            jet_btag = get_field(j, "btag").astype(np.float32)

            jet_sin_phi = get_field(j, "sin_phi", required=False)
            jet_cos_phi = get_field(j, "cos_phi", required=False)

            if jet_sin_phi is None or jet_cos_phi is None:
                jet_phi = get_field(j, "phi").astype(np.float32)
                jet_sin_phi = np.sin(jet_phi)
                jet_cos_phi = np.cos(jet_phi)

            jet_sin_phi = jet_sin_phi.astype(np.float32)
            jet_cos_phi = jet_cos_phi.astype(np.float32)

            jet_valid = finite_positive(jet_pt)

            converted_jets["e"][:, :jw] = np.where(jet_valid, jet_e, np.nan)
            converted_jets["eta"][:, :jw] = np.where(jet_valid, jet_eta, np.nan)
            converted_jets["sin_phi"][:, :jw] = np.where(jet_valid, jet_sin_phi, np.nan)
            converted_jets["cos_phi"][:, :jw] = np.where(jet_valid, jet_cos_phi, np.nan)
            converted_jets["pt"][:, :jw] = np.where(jet_valid, jet_pt, np.nan)
            converted_jets["btag"][:, :jw] = np.where(jet_valid, jet_btag, np.nan)
            converted_jets["charge"][:, :jw] = np.where(jet_valid, 0.0, np.nan)
            converted_jets["id"][:, :jw] = np.where(jet_valid, 1.0, np.nan)

            out_jet[start:stop] = converted_jets

            # ====================================================
            # LEPTON
            #
            # Existing lepton:
            #   e/eta/pt/sin_phi/cos_phi copied/computed
            #   btag = 0 because unavailable/not applicable
            #   charge = 0 because unavailable in input
            #   id = 2 electron, 3 muon
            #
            # Missing/padded lepton:
            #   all fields remain NaN
            # ====================================================

            converted_leptons = make_nan_structured((n, target_widths["LEPTON"]), object_dtype)
            lw = l.shape[1]

            lep_e = get_field(l, "e").astype(np.float32)
            lep_eta = get_field(l, "eta").astype(np.float32)
            lep_pt = get_field(l, "pt").astype(np.float32)

            lep_sin_phi = get_field(l, "sin_phi", required=False)
            lep_cos_phi = get_field(l, "cos_phi", required=False)

            if lep_sin_phi is None or lep_cos_phi is None:
                lep_phi = get_field(l, "phi").astype(np.float32)
                lep_sin_phi = np.sin(lep_phi)
                lep_cos_phi = np.cos(lep_phi)

            lep_sin_phi = lep_sin_phi.astype(np.float32)
            lep_cos_phi = lep_cos_phi.astype(np.float32)

            is_el = get_field(l, "is_el", required=False)
            is_mu = get_field(l, "is_mu", required=False)

            if is_el is None:
                is_el = np.zeros((n, lw), dtype=np.float32)
            if is_mu is None:
                is_mu = np.zeros((n, lw), dtype=np.float32)

            lep_valid = finite_positive(lep_pt)

            lep_id = np.where(
                np.asarray(is_el) > 0,
                2.0,
                np.where(np.asarray(is_mu) > 0, 3.0, np.nan),
            ).astype(np.float32)

            converted_leptons["e"][:, :lw] = np.where(lep_valid, lep_e, np.nan)
            converted_leptons["eta"][:, :lw] = np.where(lep_valid, lep_eta, np.nan)
            converted_leptons["sin_phi"][:, :lw] = np.where(lep_valid, lep_sin_phi, np.nan)
            converted_leptons["cos_phi"][:, :lw] = np.where(lep_valid, lep_cos_phi, np.nan)
            converted_leptons["pt"][:, :lw] = np.where(lep_valid, lep_pt, np.nan)
            converted_leptons["btag"][:, :lw] = np.where(lep_valid, 0.0, np.nan)
            converted_leptons["charge"][:, :lw] = np.where(lep_valid, 0.0, np.nan)
            converted_leptons["id"][:, :lw] = np.where(lep_valid, lep_id, np.nan)

            out_lepton[start:stop] = converted_leptons

            # ====================================================
            # MET
            #
            # Existing MET:
            #   e = met
            #   pt = met
            #   sin_phi/cos_phi copied/computed
            #   eta = 0 because unavailable/not applicable
            #   btag = 0 because unavailable/not applicable
            #   charge = 0 because unavailable/not applicable
            #   id = 4
            #
            # Missing MET:
            #   all fields remain NaN
            # ====================================================

            converted_met = make_nan_structured((n, target_widths["MET"]), object_dtype)
            mw = m.shape[1]

            met_pt = get_field(m, "pt", "met").astype(np.float32)

            met_sin_phi = get_field(m, "sin_phi", required=False)
            met_cos_phi = get_field(m, "cos_phi", required=False)

            if met_sin_phi is None or met_cos_phi is None:
                met_phi = get_field(m, "phi").astype(np.float32)
                met_sin_phi = np.sin(met_phi)
                met_cos_phi = np.cos(met_phi)

            met_sin_phi = met_sin_phi.astype(np.float32)
            met_cos_phi = met_cos_phi.astype(np.float32)

            met_valid = np.isfinite(met_pt)

            converted_met["e"][:, :mw] = np.where(met_valid, met_pt, np.nan)
            converted_met["eta"][:, :mw] = np.where(met_valid, 0.0, np.nan)
            converted_met["sin_phi"][:, :mw] = np.where(met_valid, met_sin_phi, np.nan)
            converted_met["cos_phi"][:, :mw] = np.where(met_valid, met_cos_phi, np.nan)
            converted_met["pt"][:, :mw] = np.where(met_valid, met_pt, np.nan)
            converted_met["btag"][:, :mw] = np.where(met_valid, 0.0, np.nan)
            converted_met["charge"][:, :mw] = np.where(met_valid, 0.0, np.nan)
            converted_met["id"][:, :mw] = np.where(met_valid, 4.0, np.nan)

            out_met[start:stop] = converted_met

            # ====================================================
            # LABELS
            # ====================================================

            # Event-level S/B label.
            labels_global = (
                get_field(y, "signal")
                .reshape(n, -1)[:, :1]
                .astype(np.float64)
            )
            out_label_global[start:stop] = labels_global

            # Jet labels:
            #   NaN only where the jet does not exist.
            #   Real jets keep original assigned labels, except original -1 -> 0.
            #   Padded target slots also remain NaN.
            converted_jet_labels = np.full((n, target_widths["JET"]), np.nan, dtype=np.float64)

            if "LABELS/JET" in src:
                old_jet_labels = np.asarray(src["LABELS/JET"][start:stop], dtype=np.float64)
                old_width = old_jet_labels.shape[1]

                copied_labels = old_jet_labels.copy()

                # For real jets, convert original -1 labels to 0.
                copied_labels = np.where(
                    (jet_valid[:, :old_width]) & (copied_labels == -1.0),
                    0.0,
                    copied_labels,
                )

                # For missing/padded jets, force NaN.
                copied_labels = np.where(
                    jet_valid[:, :old_width],
                    copied_labels,
                    np.nan,
                )

                converted_jet_labels[:, :old_width] = copied_labels

            out_label_jet[start:stop] = converted_jet_labels

            # Lepton labels:
            #   1 for existing leptons.
            #   NaN only for padded/missing lepton slots.
            converted_lepton_labels = np.full((n, target_widths["LEPTON"]), np.nan, dtype=np.float64)
            converted_lepton_labels[:, :lw] = np.where(lep_valid, 1.0, np.nan)
            out_label_lepton[start:stop] = converted_lepton_labels

            # MET labels:
            #   1 for existing MET.
            #   NaN only if MET is missing/non-finite.
            converted_met_labels = np.full((n, target_widths["MET"]), np.nan, dtype=np.float64)
            converted_met_labels[:, :mw] = np.where(met_valid, 1.0, np.nan)
            out_label_met[start:stop] = converted_met_labels

            print(f"Converted events: {stop}/{n_events}", end="\r" if stop < n_events else "\n")

        dst.attrs["source_file"] = str(input_path)
        dst.attrs["schema"] = "transformer_hyper_like_sincos_phi_nan_padding_zero_unavailable"
        dst.attrs["jet_width"] = target_widths["JET"]
        dst.attrs["lepton_width"] = target_widths["LEPTON"]
        dst.attrs["met_width"] = target_widths["MET"]
        dst.attrs["id_convention"] = "NaN=missing/padded, 1=jet, 2=electron, 3=muon, 4=MET"
        dst.attrs["padding_convention"] = "Non-existent object slots are all NaN"
        dst.attrs["unavailable_feature_convention"] = "Existing object with unavailable/non-applicable feature uses 0"
        dst.attrs["jet_label_convention"] = "NaN only for missing/padded jets; original -1 labels are converted to 0 for real jets"
        dst.attrs["lepton_met_label_convention"] = "1 for existing object; NaN for missing/padded object"

    print("Done.")


if __name__ == "__main__":
    main()