"""Configured reconstructed kinematics for the Transformer performance table."""

from __future__ import annotations

import json
import gzip
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from joint_event_reconstruction.m0.evaluation.plot_style import TRUTH_COLOUR, decorate_axis
from joint_event_reconstruction.m0.evaluation.topology import topology_spec


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _atomic_frame(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    kwargs = {"index": False}
    if path.suffix == ".gz":
        kwargs["compression"] = {"method": "gzip", "compresslevel": 1}
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def _save(fig, directory: Path, stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        final = directory / f"{stem}.{extension}"
        temporary = directory / f".{stem}.{extension}.tmp"
        kwargs = {"format": extension, "bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 250
        fig.savefig(temporary, **kwargs)
        temporary.replace(final)
    plt.close(fig)


def _phi(row) -> float:
    if row.dtype.names and "phi" in row.dtype.names:
        return float(row["phi"])
    return float(np.arctan2(row["sin_phi"], row["cos_phi"]))


def _p4(row, source: str) -> np.ndarray:
    pt = float(row["pt"])
    phi = _phi(row)
    eta = float(row["eta"]) if source != "MET" else 0.0
    energy = float(row["e"]) if source != "MET" else pt
    return np.asarray([energy, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)], dtype=np.float64)


def _mass(vector: np.ndarray) -> float:
    return float(np.sqrt(max(vector[0] ** 2 - np.dot(vector[1:], vector[1:]), 0.0)))


def _assignment(value):
    if value is None or (isinstance(value, float) and np.isnan(value)) or not str(value).strip():
        return {}
    decoded = json.loads(value) if isinstance(value, str) else value
    if not isinstance(decoded, dict):
        raise ValueError("Reconstruction assignment must be a JSON object.")
    return decoded


def _members(assignment, role):
    candidate = assignment.get("selected", {}).get(role, {})
    return [
        (str(source), int(index))
        for source, index in zip(candidate.get("member_sources", ()), candidate.get("source_indices", ()))
    ]


def _sum_members(rows, members, event_index):
    if not members:
        raise ValueError(f"Event {event_index} has no members for a configured kinematic object.")
    values = []
    for source, index in members:
        if source not in rows or index < 0 or index >= len(rows[source]):
            raise IndexError(f"Kinematic assignment index is out of range: event_index={event_index}, source={source}, index={index}")
        values.append(_p4(rows[source][index], source))
    return np.sum(values, axis=0)


def _truth_members(row, topology, component):
    truth = json.loads(row.truth_roles_json)
    return [(str(source), int(index)) for source, index in truth[component]]


def _event_records(row, source_rows, topology):
    assignment = _assignment(row.reco_assignment_json)
    objects = {
        "mWhad": ("Whad", "whad"),
        "mthad": ("thad", "thad"),
        "visible_mtlep": ("tlep", "tlep"),
    }
    if topology == "ttH":
        objects.update({"mH": ("H", "higgs"), "visible_mttH": ("tlep+thad+H", None)})
    else:
        objects["visible_mttbar"] = ("tlep+thad", None)
    predicted_vectors = {}
    truth_vectors = {}
    for observable, (assignment_role, truth_component) in objects.items():
        if assignment_role == "tlep+thad+H":
            predicted_members = _members(assignment, "tlep") + _members(assignment, "thad") + _members(assignment, "H")
            truth_members = _truth_members(row, topology, "tlep") + _truth_members(row, topology, "thad") + _truth_members(row, topology, "higgs")
        elif assignment_role == "tlep+thad":
            predicted_members = _members(assignment, "tlep") + _members(assignment, "thad")
            truth_members = _truth_members(row, topology, "tlep") + _truth_members(row, topology, "thad")
        else:
            predicted_members = _members(assignment, assignment_role)
            truth_members = _truth_members(row, topology, truth_component)
        predicted_vectors[observable] = _sum_members(source_rows, predicted_members, int(row.event_index))
        truth_vectors[observable] = _sum_members(source_rows, truth_members, int(row.event_index))
    records = []
    for observable in objects:
        predicted = _mass(predicted_vectors[observable])
        truth = _mass(truth_vectors[observable])
        records.append({
            "path": str(row.path), "sample": str(row.sample), "event_index": int(row.event_index),
            "topology": topology, "observable": observable, "predicted": predicted, "truth": truth,
            "resolution": predicted - truth, "response": predicted / truth if truth > 0 else np.nan,
        })
    return records


def _plot_distributions(values_by_observable: dict[str, dict[str, list[float]]], directory: Path) -> None:
    for observable, values_by_kind in values_by_observable.items():
        values = np.concatenate([
            np.asarray(values_by_kind["predicted"], dtype=float),
            np.asarray(values_by_kind["truth"], dtype=float),
        ])
        finite = values[np.isfinite(values)]
        if not finite.size:
            continue
        low, high = np.quantile(finite, [0.005, 0.995])
        if high <= low:
            high = low + 1.0
        bins = np.linspace(low, high, 41)
        fig, ax = plt.subplots(figsize=(7.4, 5.5))
        ax.hist(values_by_kind["truth"], bins=bins, density=True, histtype="step", linewidth=2, color=TRUTH_COLOUR, label="Truth")
        ax.hist(values_by_kind["predicted"], bins=bins, density=True, histtype="step", linewidth=1.8, color="#3B82F6", label="Transformer")
        ax.set_xlabel(f"{observable} [GeV]")
        ax.set_ylabel("Density")
        decorate_axis(ax)
        ax.legend()
        _save(fig, directory, f"{observable}_distribution")


def write_kinematic_products(events: pd.DataFrame | Path, output_dir: Path, topology: str) -> pd.DataFrame:
    """Read selected source-event kinematics in chunks and write compact products."""
    topology_spec(topology)
    if isinstance(events, (str, Path)):
        event_chunks = pd.read_csv(
            events,
            usecols=[
                "path", "sample", "event_index", "fully_matched", "reco_valid",
                "reco_assignment_json", "truth_roles_json",
            ],
            chunksize=25000,
        )
    else:
        event_chunks = (events,)
    columns = ["path", "sample", "event_index", "topology", "observable", "predicted", "truth", "resolution", "response"]
    values_by_observable = {}
    temporary = output_dir / ".kinematics.csv.gz.tmp"
    output_path = output_dir / "kinematics.csv.gz"
    output_dir.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    h5_files = {}
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
            for event_chunk in event_chunks:
                selected = event_chunk[
                    event_chunk["fully_matched"].map(_as_bool)
                    & event_chunk["reco_valid"].map(_as_bool)
                ]
                if selected.empty:
                    continue
                records = []
                for source_path, group in selected.groupby("path", sort=False):
                    source_path = Path(source_path)
                    if not source_path.is_file():
                        raise FileNotFoundError(f"Kinematic source H5 does not exist: {source_path}")
                    h5_file = h5_files.get(str(source_path))
                    if h5_file is None:
                        h5_file = h5py.File(source_path, "r")
                        h5_files[str(source_path)] = h5_file
                    ordered = group.sort_values("event_index")
                    indices = ordered["event_index"].to_numpy(dtype=np.int64)
                    for start in range(0, len(ordered), 4096):
                        stop = min(len(ordered), start + 4096)
                        chunk = ordered.iloc[start:stop]
                        chunk_indices = indices[start:stop]
                        unique, inverse = np.unique(chunk_indices, return_inverse=True)
                        rows = {
                            "JET": np.asarray(h5_file["INPUTS/JET"][unique]),
                            "LEPTON": np.asarray(h5_file["INPUTS/LEPTON"][unique]),
                            "MET": np.asarray(h5_file["INPUTS/MET"][unique]),
                        }
                        for local, row in enumerate(chunk.itertuples(index=False)):
                            source_rows = {name: values[inverse[local]] for name, values in rows.items()}
                            records.extend(_event_records(row, source_rows, topology))
                if not records:
                    continue
                record_frame = pd.DataFrame(records, columns=columns)
                record_frame.to_csv(stream, index=False, header=not wrote_header)
                wrote_header = True
                for observable, group in record_frame.groupby("observable", sort=False):
                    values = values_by_observable.setdefault(observable, {"predicted": [], "truth": [], "resolution": []})
                    values["predicted"].extend(group["predicted"].tolist())
                    values["truth"].extend(group["truth"].tolist())
                    values["resolution"].extend(group["resolution"].tolist())
        temporary.replace(output_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    finally:
        for h5_file in h5_files.values():
            h5_file.close()
    if values_by_observable:
        _plot_distributions(values_by_observable, output_dir)
        summary = pd.DataFrame([
            {
                "observable": observable,
                "n_events": len(values["resolution"]),
                "median_resolution": float(np.median(values["resolution"])),
                "q16": float(np.quantile(values["resolution"], 0.16)),
                "q84": float(np.quantile(values["resolution"], 0.84)),
            }
            for observable, values in values_by_observable.items()
        ])
    else:
        summary = pd.DataFrame(columns=["observable", "n_events", "median_resolution", "q16", "q84"])
    _atomic_frame(summary, output_dir / "resolution_summary.csv")
    return summary
