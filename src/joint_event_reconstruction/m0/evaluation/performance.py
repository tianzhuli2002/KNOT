"""Authoritative Transformer prediction and performance evaluation."""

from __future__ import annotations

import json
import logging
import math
import gzip
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from joint_event_reconstruction.m0.evaluation.plot_style import (
    BACKGROUND_COLOUR,
    CORRECT_COLOUR,
    RANDOM_COLOUR,
    SIGNAL_COLOUR,
    decorate_axis,
    efficiency_axis,
)
from joint_event_reconstruction.m0.evaluation.provenance import normalise_path, resolve_prediction_manifest
from joint_event_reconstruction.m0.evaluation.topology import (
    component_names,
    fully_matched_from_labels,
    component_column,
    njet_bin,
    required_truth_members,
    topology_spec,
)

logger = logging.getLogger(__name__)

EVENT_KEYS = ["path", "sample", "event_index"]
BASE_COLUMNS = EVENT_KEYS + ["label", "classification_score", "n_input_jets"]
SOURCE_COLUMNS = ["n_jets", "jets_truncated", "fully_matched", "truth_roles_json"]
RECO_COLUMNS = ["reco_valid", "reco_score", "reco_confidence", "reco_assignment_json"]


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _atomic_frame(frame: pd.DataFrame, path: Path, *, gzip: bool = False, columns=None) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    kwargs = {"index": False}
    if gzip:
        kwargs["compression"] = {"method": "gzip", "compresslevel": 1}
    if columns is not None:
        kwargs["columns"] = columns
    frame.to_csv(temporary, **kwargs)
    temporary.replace(path)


def _remove_csv_column(path: Path, column: str) -> None:
    """Rewrite one streamed CSV product without an internal-only column."""
    temporary = path.with_name(f".{path.name}.tmp")
    wrote_header = False
    found = False
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
            for frame in pd.read_csv(path, chunksize=50000):
                if column not in frame:
                    frame.to_csv(stream, index=False, header=not wrote_header)
                    wrote_header = True
                    continue
                found = True
                frame.drop(columns=[column]).to_csv(stream, index=False, header=not wrote_header)
                wrote_header = True
        if found:
            temporary.replace(path)
        else:
            temporary.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_json(value: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        final = output_dir / f"{stem}.{extension}"
        temporary = output_dir / f".{stem}.{extension}.tmp"
        kwargs = {"format": extension, "bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 250
        fig.savefig(temporary, **kwargs)
        temporary.replace(final)
    plt.close(fig)


def _read_csv(path: Path, *, usecols=None, nrows=None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required prediction product does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Required prediction product is empty: {path}")
    try:
        return pd.read_csv(path, usecols=usecols, nrows=nrows)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read prediction product {path}: {exc}") from exc


def _prepare_modern_events(resolved: dict) -> pd.DataFrame:
    path = resolved["paths"]["prediction_events"]
    frame = _read_csv(path)
    required = set(EVENT_KEYS + ["label", "n_input_jets"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Compact prediction table {path} is missing required columns: {missing}")
    if "classification_score" not in frame:
        frame["classification_score"] = np.nan
    return frame


def _normalise_identity(frame: pd.DataFrame, resolved: dict) -> None:
    for column in EVENT_KEYS:
        if column not in frame:
            raise ValueError(f"Prediction table is missing provenance column {column!r}.")
    if frame["path"].isna().any() or frame["sample"].isna().any():
        raise ValueError("Prediction provenance contains missing path or sample values.")
    raw_paths = frame["path"].astype(str)
    normalised_paths = {value: normalise_path(value) for value in raw_paths.unique()}
    frame["path"] = raw_paths.map(normalised_paths)
    frame["sample"] = frame["sample"].astype(str)
    frame["event_index"] = pd.to_numeric(frame["event_index"], errors="raise").astype(np.int64)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int8)
    if not frame["label"].isin([0, 1]).all():
        bad = frame.loc[~frame["label"].isin([0, 1]), EVENT_KEYS + ["label"]].head(5)
        raise ValueError("Prediction labels must be binary; offending rows:\n" + bad.to_string(index=False))
    duplicate = frame.duplicated(EVENT_KEYS, keep=False)
    if duplicate.any():
        raise ValueError("Prediction output contains duplicate event identities:\n" + frame.loc[duplicate, EVENT_KEYS].head(5).to_string(index=False))
    source_paths = set(resolved["source_h5_paths"])
    found_paths = set(frame["path"])
    if found_paths != source_paths:
        raise ValueError(
            "Prediction provenance paths disagree with the manifest source H5 paths: "
            f"manifest={sorted(map(str, source_paths))}, predictions={sorted(map(str, found_paths))}."
        )
    if "classification_score" in frame:
        frame["classification_score"] = pd.to_numeric(frame["classification_score"], errors="coerce")
        finite = frame["classification_score"].notna()
        if (~np.isfinite(frame.loc[finite, "classification_score"])).any() or not frame.loc[finite, "classification_score"].between(0, 1).all():
            raise ValueError("Classification scores must be finite values in [0, 1].")
    if "n_input_jets" in frame:
        frame["n_input_jets"] = pd.to_numeric(frame["n_input_jets"], errors="raise").astype(np.int16)
        if (frame["n_input_jets"] <= 0).any() or (frame["n_input_jets"] > int(resolved["model_max_jets"])).any():
            raise ValueError(
                f"Prediction n_input_jets must be positive and no greater than model cap {resolved['model_max_jets']}."
            )


def _decode_json(value, *, column: str, key: str) -> dict:
    if value is None or (isinstance(value, float) and np.isnan(value)) or not str(value).strip():
        return {}
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {column}: {value!r}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"Expected an object in {column}, found {type(decoded).__name__}.")
    return decoded


def _adapt_legacy_assignment(value) -> str:
    """Convert the completed-run assignment JSON to the current public names."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or not str(value).strip():
        return "{}"
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid legacy topology assignment JSON: {value!r}") from exc

    def convert(item):
        if isinstance(item, dict):
            converted = {}
            for key, nested in item.items():
                if key == "global_indices":
                    key = "token_indices"
                elif key == "global_topology_score":
                    key = "topology_score"
                elif key == "strategy":
                    continue
                converted[key] = convert(nested)
            return converted
        if isinstance(item, list):
            return [convert(nested) for nested in item]
        return item

    converted = convert(decoded)
    if not isinstance(converted, dict):
        raise ValueError("Legacy topology assignment JSON must contain an object.")
    return json.dumps(converted, sort_keys=True, separators=(",", ":"))


def _candidate_members(candidate: dict, source: str | None = None) -> tuple[tuple[str, int], ...]:
    if not isinstance(candidate, dict):
        return ()
    sources = candidate.get("member_sources", ())
    indices = candidate.get("source_indices", ())
    if source is None:
        return tuple((str(member_source), int(index)) for member_source, index in zip(sources, indices))
    return tuple((str(member_source), int(index)) for member_source, index in zip(sources, indices) if member_source == source)


def _assignment_members(assignment: dict, role: str, source: str | None = None):
    return _candidate_members(assignment.get("selected", {}).get(role, {}), source)


def _component_flags(row, topology: str) -> dict[str, bool]:
    names = component_names(topology)
    result = {component_column(name): False for name in names}
    result["event_correct"] = False
    if not _as_bool(row.reco_valid) or not bool(row.fully_matched):
        return result
    assignment = _decode_json(row.reco_assignment_json, column="reco_assignment_json", key="selected")
    truth = _decode_json(row.truth_roles_json, column="truth_roles_json", key="roles")
    def exact(role):
        truth_key = {"Whad": "whad", "H": "higgs"}.get(role, role)
        truth_members = tuple((str(source), int(index)) for source, index in truth.get(truth_key, []))
        assigned_members = tuple(_assignment_members(assignment, role))
        return bool(truth_members) and sorted(assigned_members) == sorted(truth_members)

    # The decoder stores b roles at the topology level and the complete tlep/thad/H
    # candidates in selected.  Compare the direct roles to the same source-index truth.
    result["blep_correct"] = assignment.get("blep") is not None and [("JET", int(assignment["blep"]))] == [tuple(value) for value in truth.get("blep", [])]
    result["bhad_correct"] = assignment.get("bhad") is not None and [("JET", int(assignment["bhad"]))] == [tuple(value) for value in truth.get("bhad", [])]
    result["whad_correct"] = exact("Whad")
    result["tlep_correct"] = exact("tlep")
    result["thad_correct"] = exact("thad")
    if topology == "ttH":
        result["higgs_correct"] = exact("H")
    result["event_correct"] = all(result[component_column(name)] for name in names)
    return result


def _source_dataset_values(dataset, indices):
    values = np.asarray(dataset[indices])
    if values.dtype.names:
        field = "signal" if "signal" in values.dtype.names else values.dtype.names[0]
        values = values[field]
    return np.asarray(values).reshape(len(indices), -1)


def _read_source_truth(events: pd.DataFrame, resolved: dict, topology: str) -> pd.DataFrame:
    events.reset_index(drop=True, inplace=True)
    n_rows = len(events)
    source_n_jets = np.full(n_rows, -1, dtype=np.int16)
    fully_matched = np.zeros(n_rows, dtype=bool)
    truth_roles = [None] * n_rows
    labels_disagree = []
    event_labels = events["label"].to_numpy(dtype=np.int8)
    for source_path, positions in events.groupby("path", sort=False).groups.items():
        source_path = Path(source_path)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source H5 from prediction provenance does not exist: {source_path}")
        row_positions = np.asarray(list(positions), dtype=np.int64)
        event_indices = events.iloc[row_positions]["event_index"].to_numpy(dtype=np.int64)
        order = np.argsort(event_indices, kind="stable")
        sorted_positions = row_positions[order]
        sorted_event_indices = event_indices[order]
        with h5py.File(source_path, "r") as h5_file:
            required = ("INPUTS/GLOBAL", "LABELS/GLOBAL", "LABELS/JET", "LABELS/LEPTON", "LABELS/MET")
            missing = [name for name in required if name not in h5_file]
            if missing:
                raise KeyError(f"Source H5 {source_path} is missing required datasets: {missing}")
            n_events = h5_file["LABELS/GLOBAL"].shape[0]
            if (sorted_event_indices < 0).any() or (sorted_event_indices >= n_events).any():
                bad = sorted_event_indices[(sorted_event_indices < 0) | (sorted_event_indices >= n_events)][:5]
                raise IndexError(f"Prediction event indices cannot be matched to {source_path}; offending indices={bad.tolist()}")
            for start in range(0, len(sorted_positions), 8192):
                stop = min(len(sorted_positions), start + 8192)
                positions_chunk = sorted_positions[start:stop]
                indices_chunk = sorted_event_indices[start:stop]
                unique_indices, inverse = np.unique(indices_chunk, return_inverse=True)
                global_labels = _source_dataset_values(h5_file["LABELS/GLOBAL"], unique_indices).reshape(-1)
                jet_labels = _source_dataset_values(h5_file["LABELS/JET"], unique_indices)
                lepton_labels = _source_dataset_values(h5_file["LABELS/LEPTON"], unique_indices)
                met_labels = _source_dataset_values(h5_file["LABELS/MET"], unique_indices)
                global_inputs = np.asarray(h5_file["INPUTS/GLOBAL"][unique_indices])
                if global_inputs.dtype.names and "njet" in global_inputs.dtype.names:
                    njet_values = np.asarray(global_inputs["njet"]).reshape(-1)
                else:
                    finite_jets = np.asarray(h5_file["INPUTS/JET"][unique_indices])
                    field = "pt" if finite_jets.dtype.names and "pt" in finite_jets.dtype.names else finite_jets.dtype.names[0]
                    njet_values = np.isfinite(finite_jets[field]).sum(axis=1)
                for local, (row_position, source_index) in enumerate(zip(positions_chunk, inverse)):
                    event_label = int(event_labels[row_position])
                    source_label = global_labels[source_index]
                    if not np.isfinite(source_label) or int(source_label) != source_label:
                        raise ValueError(f"Source label is not finite binary at file={source_path}, event_index={int(indices_chunk[local])}.")
                    if int(source_label) not in (0, 1):
                        raise ValueError(f"Source label is not binary at file={source_path}, event_index={int(indices_chunk[local])}: {source_label}")
                    if event_label != int(source_label):
                        labels_disagree.append((str(source_path), int(indices_chunk[local]), event_label, int(source_label)))
                    njet = njet_values[source_index]
                    if not np.isfinite(njet) or int(njet) != njet or int(njet) < 0:
                        raise ValueError(f"Source n_jets is invalid at file={source_path}, event_index={int(indices_chunk[local])}: {njet}")
                    njet = int(njet)
                    source_n_jets[row_position] = njet
                    role_map = {}
                    for role in topology_spec(topology)["required_jet_roles"]:
                        values = [int(value) for value in np.flatnonzero(np.isfinite(jet_labels[source_index]) & (jet_labels[source_index] == role))]
                        role_map[str(role)] = values
                        role_map[f"JET:{role}"] = values
                    role_map["LEPTON:1"] = [int(value) for value in np.flatnonzero(np.isfinite(lepton_labels[source_index]) & (lepton_labels[source_index] == 1))]
                    role_map["MET:1"] = [int(value) for value in np.flatnonzero(np.isfinite(met_labels[source_index]) & (met_labels[source_index] == 1))]
                    truth_by_component = {}
                    for component, members in required_truth_members(topology).items():
                        truth_by_component[component] = [
                            [source, int(role_map[f"{source}:{role}"][0])]
                            for source, role in members
                            if len(role_map[f"{source}:{role}"]) == 1
                        ]
                    truth_roles[row_position] = json.dumps(
                        truth_by_component, sort_keys=True, separators=(",", ":")
                    )
                    fully_matched[row_position] = fully_matched_from_labels(
                        source_label, jet_labels[source_index], lepton_labels[source_index], met_labels[source_index], topology
                    )
    if labels_disagree:
        examples = "\n".join(str(item) for item in labels_disagree[:5])
        raise ValueError("Prediction and source H5 labels disagree; examples (file, event_index, prediction, source):\n" + examples)
    if (source_n_jets < 0).any() or any(value is None for value in truth_roles):
        raise ValueError("Source H5 matching did not produce a truth record for every prediction event.")
    if "n_input_jets" not in events:
        events["n_input_jets"] = np.minimum(source_n_jets, int(resolved["model_max_jets"])).astype(np.int16)
    events["n_jets"] = source_n_jets
    events["jets_truncated"] = events["n_jets"] > events["n_input_jets"]
    events["fully_matched"] = fully_matched
    events["truth_roles_json"] = truth_roles
    return events


def _prepare_events(resolved: dict) -> pd.DataFrame:
    if resolved["legacy"]:
        raise ValueError("Legacy prediction products must be evaluated through the bounded compatibility reader.")
    frame = _prepare_modern_events(resolved)
    _normalise_identity(frame, resolved)
    if "reco_valid" in frame:
        frame["reco_valid"] = frame["reco_valid"].map(_as_bool)
        for column in ("reco_score", "reco_confidence"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["reco_assignment_json"] = frame["reco_assignment_json"].fillna("")
    frame = _read_source_truth(frame, resolved, resolved["topology"])
    if "reco_valid" in frame:
        flag_columns = [*(component_column(name) for name in component_names(resolved["topology"])), "event_correct"]
        flag_values = {column: np.zeros(len(frame), dtype=bool) for column in flag_columns}
        for index, row in enumerate(frame.itertuples(index=False)):
            values = _component_flags(row, resolved["topology"])
            for column in flag_columns:
                flag_values[column][index] = bool(values[column])
        for column, values in flag_values.items():
            frame[column] = values
    return frame


def _legacy_chunks(resolved: dict, chunk_size: int = 50000):
    """Read the completed legacy products in bounded chunks."""
    paths = resolved["paths"]
    classification_reader = (
        pd.read_csv(paths["classification_predictions"], usecols=["path", "sample", "event_index", "label", "score"], chunksize=chunk_size)
        if paths["classification_predictions"] else None
    )
    metadata_reader = (
        pd.read_csv(
            paths["reconstruction_event_metadata"],
            usecols=["path", "sample", "event_index", "label", "n_valid_jets", "global_reco_valid", "global_topology_confidence", "global_topology_json"],
            chunksize=chunk_size,
        )
        if paths["reconstruction_event_metadata"] else None
    )
    if classification_reader is None and metadata_reader is None:
        raise ValueError("Legacy prediction manifest has no supported event products.")
    seen = set()
    while True:
        class_chunk = next(classification_reader, None) if classification_reader is not None else None
        metadata_chunk = next(metadata_reader, None) if metadata_reader is not None else None
        if class_chunk is None and metadata_chunk is None:
            break
        if class_chunk is not None:
            class_chunk = class_chunk.rename(columns={"score": "classification_score"})
            _normalise_identity(class_chunk, resolved)
        if metadata_chunk is not None:
            metadata_chunk = metadata_chunk.rename(
                columns={
                    "n_valid_jets": "n_input_jets", "global_reco_valid": "reco_valid",
                    "global_topology_confidence": "reco_score", "global_topology_json": "reco_assignment_json",
                }
            )
            metadata_chunk["reco_confidence"] = metadata_chunk["reco_score"]
            metadata_chunk["reco_assignment_json"] = metadata_chunk["reco_assignment_json"].map(_adapt_legacy_assignment)
            _normalise_identity(metadata_chunk, resolved)
        if class_chunk is not None and metadata_chunk is not None:
            if len(class_chunk) != len(metadata_chunk):
                raise ValueError("Legacy classification and reconstruction products have different row counts.")
            for column in EVENT_KEYS:
                if not class_chunk[column].equals(metadata_chunk[column]):
                    examples = pd.concat([class_chunk[EVENT_KEYS], metadata_chunk[EVENT_KEYS]], axis=1).head(5)
                    raise ValueError("Legacy classification and reconstruction identities disagree:\n" + examples.to_string(index=False))
            if not class_chunk["label"].equals(metadata_chunk["label"]):
                raise ValueError("Legacy classification and reconstruction labels disagree within a chunk.")
            frame = metadata_chunk
            frame["classification_score"] = class_chunk["classification_score"].to_numpy()
        else:
            frame = class_chunk if class_chunk is not None else metadata_chunk
            if "classification_score" not in frame:
                frame["classification_score"] = np.nan
        keys = list(zip(frame["path"].astype(str), frame["sample"], frame["event_index"].astype(int)))
        duplicate = [key for key in keys if key in seen]
        if duplicate:
            raise ValueError(f"Legacy prediction products contain duplicate event identities, examples={duplicate[:5]}")
        seen.update(keys)
        frame = _read_source_truth(frame, resolved, resolved["topology"])
        if "reco_valid" in frame:
            frame["reco_valid"] = frame["reco_valid"].map(_as_bool)
            frame["reco_score"] = pd.to_numeric(frame["reco_score"], errors="coerce")
            frame["reco_confidence"] = pd.to_numeric(frame["reco_confidence"], errors="coerce")
            frame["reco_assignment_json"] = frame["reco_assignment_json"].fillna("")
            flag_columns = [*(component_column(name) for name in component_names(resolved["topology"])), "event_correct"]
            flag_values = {column: np.zeros(len(frame), dtype=bool) for column in flag_columns}
            for index, row in enumerate(frame.itertuples(index=False)):
                values = _component_flags(row, resolved["topology"])
                for column in flag_columns:
                    flag_values[column][index] = bool(values[column])
            for column, values in flag_values.items():
                frame[column] = values
        yield frame


def _legacy_metrics_update(frame: pd.DataFrame, topology: str, totals: dict) -> None:
    totals["n_events"] += len(frame)
    totals["n_signal"] += int((frame["label"] == 1).sum())
    totals["n_background"] += int((frame["label"] == 0).sum())
    totals["truncated"] += int(frame["jets_truncated"].sum())
    totals["fully_matched"] += int(frame["fully_matched"].sum())
    for category in topology_spec(topology)["jet_bins"]:
        mask = frame["n_jets"].map(lambda value: njet_bin(topology, value) == category).to_numpy(dtype=bool)
        totals["bins"][category]["n_events"] += int(mask.sum())
        totals["bins"][category]["n_signal"] += int((frame.loc[mask, "label"] == 1).sum())
        totals["bins"][category]["n_background"] += int((frame.loc[mask, "label"] == 0).sum())
        totals["bins"][category]["n_fully_matched_signal"] += int(frame.loc[mask, "fully_matched"].sum())
    if "event_correct" not in frame:
        return
    selected_all = frame["fully_matched"].to_numpy(dtype=bool)
    totals["reco"]["overall"]["denominator"] += int(selected_all.sum())
    for component in (*[component_column(name) for name in component_names(topology)], "event_correct"):
        totals["reco"]["overall"][component] += int(frame.loc[selected_all, component].sum())
    for category in topology_spec(topology)["jet_bins"]:
        mask = frame["n_jets"].map(lambda value: njet_bin(topology, value) == category).to_numpy(dtype=bool)
        selected = mask & frame["fully_matched"].to_numpy(dtype=bool)
        totals["reco"][category]["denominator"] += int(selected.sum())
        for component in (*[component_column(name) for name in component_names(topology)], "event_correct"):
            totals["reco"][category][component] += int(frame.loc[selected, component].sum())


def _legacy_metrics_frame(totals: dict, topology: str) -> pd.DataFrame:
    rows = []
    for category in ("overall", *topology_spec(topology)["jet_bins"]):
        if category == "overall":
            denominator = totals["fully_matched"]
            event_count = totals["n_events"]
            values = totals["reco"]["overall"]
        else:
            denominator = totals["reco"][category]["denominator"]
            event_count = totals["bins"][category]["n_events"]
            values = totals["reco"][category]
        if denominator == 0:
            logger.warning("Reconstruction bin %s has no fully matched signal events.", category)
        for component in (*[component_column(name) for name in component_names(topology)], "event_correct"):
            numerator = values[component]
            efficiency, uncertainty = _efficiency(numerator, denominator)
            rows.append({"n_jets_bin": category, "component": component, "n_events": event_count, "denominator": denominator, "numerator": numerator, "efficiency": efficiency, "binomial_uncertainty": uncertainty})
    return pd.DataFrame(rows)


def _run_legacy_performance(resolved: dict, output_dir: Path, performance_cfg: dict, classification_available: bool, reconstruction_available: bool):
    topology = resolved["topology"]
    bins = topology_spec(topology)["jet_bins"]
    totals = {
        "n_events": 0, "n_signal": 0, "n_background": 0, "truncated": 0, "fully_matched": 0,
        "bins": {category: {"n_events": 0, "n_signal": 0, "n_background": 0, "n_fully_matched_signal": 0} for category in bins},
        "reco": {category: {"denominator": 0, **{component_column(name): 0 for name in component_names(topology)}, "event_correct": 0} for category in ("overall", *bins)},
    }
    score_values, label_values, njet_values = [], [], []
    event_path = output_dir / "events.csv.gz"
    temporary = output_dir / ".events.csv.gz.tmp"
    wrote_header = False
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as stream:
            for frame in _legacy_chunks(resolved):
                _legacy_metrics_update(frame, topology, totals)
                if classification_available:
                    finite = frame["classification_score"].notna().to_numpy(dtype=bool)
                    score_values.extend(frame.loc[finite, "classification_score"].to_numpy(dtype=float).tolist())
                    label_values.extend(frame.loc[finite, "label"].to_numpy(dtype=np.int8).tolist())
                    njet_values.extend(frame.loc[finite, "n_jets"].to_numpy(dtype=np.int16).tolist())
                output_columns = (
                    list(frame.columns)
                    if reconstruction_available and bool(performance_cfg.get("kinematics", True))
                    else [column for column in frame.columns if column != "truth_roles_json"]
                )
                frame.to_csv(stream, index=False, header=not wrote_header, columns=output_columns)
                wrote_header = True
        temporary.replace(event_path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    if reconstruction_available and bool(performance_cfg.get("kinematics", True)):
        from joint_event_reconstruction.m0.evaluation.kinematics import write_kinematic_products
        write_kinematic_products(event_path, output_dir / "reconstruction" / "kinematics", topology)
        _remove_csv_column(event_path, "truth_roles_json")
    plot_frame = pd.DataFrame({"label": label_values, "classification_score": score_values, "n_jets": njet_values})
    auc = _write_classification(plot_frame, output_dir, topology) if classification_available else None
    reco_efficiency = None
    if reconstruction_available:
        metrics = _legacy_metrics_frame(totals, topology)
        reco_efficiency = _plot_reconstruction(metrics, output_dir, topology)
    return totals, auc, reco_efficiency


def _validate_binary_labels(frame: pd.DataFrame) -> None:
    labels = frame["label"].to_numpy(dtype=np.int8)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"Classification performance requires both labels 0 and 1; found {sorted(set(labels))}.")


def _auc_subset(frame: pd.DataFrame) -> float | None:
    labels = frame["label"].to_numpy(dtype=np.int8)
    if len(np.unique(labels)) < 2:
        return None
    value = float(roc_auc_score(labels, frame["classification_score"].to_numpy(dtype=float)))
    return value if np.isfinite(value) else None


def _write_classification(frame: pd.DataFrame, output_dir: Path, topology: str) -> float:
    if frame["classification_score"].isna().all():
        raise ValueError("Prediction manifest contains no classification scores; classification plots are unavailable.")
    _validate_binary_labels(frame)
    classification_dir = output_dir / "classification"
    classification_dir.mkdir(parents=True, exist_ok=True)
    score = frame["classification_score"].to_numpy(dtype=float)
    labels = frame["label"].to_numpy(dtype=np.int8)
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    bins = np.linspace(0, 1, 51)
    for label, name, colour in ((0, "Background", BACKGROUND_COLOUR), (1, "Signal", SIGNAL_COLOUR)):
        values = score[labels == label]
        ax.hist(values, bins=bins, density=True, alpha=0.55, color=colour, label=f"{name} (n={len(values):,})")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Classification score")
    ax.set_ylabel("Density")
    decorate_axis(ax)
    ax.legend()
    _save_figure(fig, classification_dir, "score_distribution")

    fpr, tpr, _ = roc_curve(labels, score)
    auc = float(roc_auc_score(labels, score))
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    ax.plot(fpr, tpr, color=SIGNAL_COLOUR, linewidth=2, label=f"Transformer (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color=RANDOM_COLOUR, label="Random classifier")
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False positive rate", ylabel="True positive rate")
    decorate_axis(ax)
    ax.legend(loc="lower right")
    _save_figure(fig, classification_dir, "roc")

    auc_rows = []
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    for category in topology_spec(topology)["jet_bins"]:
        mask = frame["n_jets"].map(lambda value: njet_bin(topology, value) == category).to_numpy(dtype=bool)
        subset = frame.loc[mask, ["label", "classification_score"]]
        category_auc = _auc_subset(subset)
        auc_rows.append({
            "n_jets_bin": category, "n_events": int(len(subset)),
            "n_signal": int((subset["label"] == 1).sum()), "n_background": int((subset["label"] == 0).sum()),
            "auc": category_auc,
        })
        if category_auc is None:
            logger.warning("Classification ROC bin %s has fewer than two labels; AUC is undefined.", category)
            continue
        fpr_bin, tpr_bin, _ = roc_curve(subset["label"], subset["classification_score"])
        ax.plot(fpr_bin, tpr_bin, linewidth=1.6, label=f"{category} (AUC = {category_auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color=RANDOM_COLOUR, label="Random classifier")
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title(f"Source jet multiplicity: {topology}")
    decorate_axis(ax)
    ax.legend(fontsize=9, loc="lower right")
    _save_figure(fig, classification_dir, "roc_by_njet")
    _atomic_frame(pd.DataFrame(auc_rows), classification_dir / "auc_by_njet.csv")
    _atomic_frame(pd.DataFrame([{
        "auc": auc, "n_events": int(len(frame)), "n_signal": int((labels == 1).sum()), "n_background": int((labels == 0).sum()),
    }]), classification_dir / "metrics.csv")
    return auc


def _efficiency(numerator: int, denominator: int):
    if denominator == 0:
        return None, None
    value = numerator / denominator
    return float(value), float(math.sqrt(value * (1 - value) / denominator))


def _reconstruction_metrics(frame: pd.DataFrame, topology: str) -> pd.DataFrame:
    bins = ("overall", *topology_spec(topology)["jet_bins"])
    rows = []
    for category in bins:
        category_mask = np.ones(len(frame), dtype=bool) if category == "overall" else frame["n_jets"].map(lambda value: njet_bin(topology, value) == category).to_numpy(dtype=bool)
        selected_mask = frame["fully_matched"].to_numpy(dtype=bool) & category_mask
        denominator = int(selected_mask.sum())
        for component in (*(component_column(name) for name in component_names(topology)), "event_correct"):
            numerator = int(frame.loc[selected_mask, component].sum()) if component in frame else 0
            efficiency, uncertainty = _efficiency(numerator, denominator)
            rows.append({
                "n_jets_bin": category, "component": component, "n_events": int(category_mask.sum()),
                "denominator": denominator, "numerator": numerator, "efficiency": efficiency,
                "binomial_uncertainty": uncertainty,
            })
        if denominator == 0:
            logger.warning("Reconstruction bin %s has no fully matched signal events.", category)
    return pd.DataFrame(rows)


def _plot_reconstruction(metrics: pd.DataFrame, output_dir: Path, topology: str) -> float | None:
    reconstruction_dir = output_dir / "reconstruction"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    overall = metrics[metrics["n_jets_bin"] == "overall"].set_index("component")
    event_efficiency = overall.loc["event_correct", "efficiency"] if "event_correct" in overall.index else None
    fig, ax = plt.subplots(figsize=(7.4, 5.5))
    if event_efficiency is not None:
        ax.bar(["Transformer"], [event_efficiency], color=CORRECT_COLOUR, width=0.55)
    ax.set_ylabel("Complete-event efficiency")
    ax.set_title(topology)
    efficiency_axis(ax)
    _save_figure(fig, reconstruction_dir, "efficiency")

    bins = topology_spec(topology)["jet_bins"]
    complete = metrics[(metrics["component"] == "event_correct") & metrics["n_jets_bin"].isin(bins)].set_index("n_jets_bin").reindex(bins)
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    values = complete["efficiency"].to_numpy(dtype=float)
    errors = complete["binomial_uncertainty"].to_numpy(dtype=float)
    ax.errorbar(np.arange(len(bins)), values, yerr=errors, marker="o", color=CORRECT_COLOUR, linewidth=1.8)
    ax.set_xticks(np.arange(len(bins)), bins)
    ax.set_xlabel(r"Source jet multiplicity $n_{\mathrm{jets}}$")
    ax.set_ylabel("Complete-event efficiency")
    efficiency_axis(ax)
    _save_figure(fig, reconstruction_dir, "efficiency_by_njet")

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for component in component_names(topology):
        column = component_column(component)
        rows = metrics[(metrics["component"] == column) & metrics["n_jets_bin"].isin(bins)].set_index("n_jets_bin").reindex(bins)
        ax.errorbar(
            np.arange(len(bins)), rows["efficiency"], yerr=rows["binomial_uncertainty"], marker="o", linewidth=1.7,
            label=topology_spec(topology)["component_labels"][component],
        )
    ax.set_xticks(np.arange(len(bins)), bins)
    ax.set_xlabel(r"Source jet multiplicity $n_{\mathrm{jets}}$")
    ax.set_ylabel("Component efficiency")
    efficiency_axis(ax)
    ax.legend(ncol=2, fontsize=9)
    _save_figure(fig, reconstruction_dir, "components_by_njet")
    _atomic_frame(metrics, reconstruction_dir / "metrics.csv")
    return event_efficiency


def _write_diagnostics(frame: pd.DataFrame, resolved: dict, output_dir: Path, topology: str) -> None:
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    signal = frame[frame["fully_matched"]]
    rows = []
    for n_input, group in signal.groupby("n_input_jets", sort=True):
        numerator = int(group["event_correct"].sum()) if "event_correct" in group else 0
        efficiency, uncertainty = _efficiency(numerator, len(group))
        rows.append({"n_input_jets": int(n_input), "denominator": int(len(group)), "numerator": numerator, "efficiency": efficiency, "binomial_uncertainty": uncertainty})
    input_table = pd.DataFrame(rows)
    _atomic_frame(input_table, diagnostics_dir / "efficiency_by_input_njet.csv")
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    if not input_table.empty:
        ax.errorbar(input_table["n_input_jets"], input_table["efficiency"], yerr=input_table["binomial_uncertainty"], marker="o", color=SIGNAL_COLOUR)
    ax.set_xlabel(r"Input jet multiplicity $n_{\mathrm{input\ jets}}$")
    ax.set_ylabel("Complete-event efficiency")
    ax.set_title("Transformer reconstruction by input jet multiplicity")
    efficiency_axis(ax)
    _save_figure(fig, diagnostics_dir, "efficiency_by_input_njet")
    candidate_path = resolved["paths"].get("candidate_table")
    if candidate_path and candidate_path.is_file():
        candidates = _read_csv(candidate_path)
        _atomic_frame(candidates.groupby("target", as_index=False).size().rename(columns={"size": "candidate_rows"}), diagnostics_dir / "candidate_counts.csv")


def _output_is_complete(output_dir: Path, manifest_path: Path, *, classification: bool, reconstruction: bool, kinematics: bool) -> bool:
    summary = output_dir / "summary.json"
    events = output_dir / "events.csv.gz"
    if not summary.is_file() or not events.is_file():
        return False
    try:
        saved = json.loads(summary.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if saved.get("prediction_manifest") != str(manifest_path):
        return False
    required = []
    if classification:
        required.extend(output_dir / "classification" / name for name in ("score_distribution.pdf", "score_distribution.png", "roc.pdf", "roc.png", "roc_by_njet.pdf", "roc_by_njet.png", "metrics.csv", "auc_by_njet.csv"))
    if reconstruction:
        required.extend(output_dir / "reconstruction" / name for name in ("efficiency.pdf", "efficiency.png", "efficiency_by_njet.pdf", "efficiency_by_njet.png", "components_by_njet.pdf", "components_by_njet.png", "metrics.csv"))
        if kinematics:
            required.extend(
                output_dir / "reconstruction" / "kinematics" / name
                for name in ("kinematics.csv.gz", "resolution_summary.csv")
            )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _resolve_output_dir(resolved: dict, cfg) -> Path:
    configured = (cfg.get("performance", {}) or {}).get("output_dir")
    if configured not in (None, ""):
        return normalise_path(configured)
    return Path(resolved["run_output_dir"]).resolve() / "performance"


def _print_summary(summary: dict) -> None:
    print("Transformer performance summary")
    print(f"  topology: {summary['topology']}")
    print(f"  config name: {summary.get('config_name', 'unknown')}")
    print(f"  model jet cap: {summary['model_max_jets']}")
    print(f"  source H5: {', '.join(summary['source_h5'])}")
    print(f"  evaluated events: {summary['n_events']:,}")
    print(f"  signal/background: {summary['n_signal']:,}/{summary['n_background']:,}")
    print(f"  fully matched signal: {summary['fully_matched_signal']:,}")
    print(
        f"  truncated events: {summary['truncated_events']:,} "
        f"({summary['truncated_fraction']:.6f})"
    )
    print(f"  overall AUC: {summary.get('overall_auc')}")
    print(f"  overall complete reconstruction efficiency: {summary.get('overall_reconstruction_efficiency')}")
    for category, counts in summary.get("n_jets_bins", {}).items():
        print(
            f"  source n_jets {category}: events={counts['n_events']:,}, "
            f"signal={counts['n_signal']:,}, fully_matched={counts['n_fully_matched_signal']:,}"
        )
    print(f"  output directory: {summary['output_directory']}")


def run_performance(datamodule, model, cfg) -> None:
    """Evaluate an existing prediction manifest; checkpoint rescoring is not supported."""
    performance_cfg = cfg.get("performance", {}) or {}
    manifest_value = performance_cfg.get("prediction_manifest")
    if not manifest_value:
        raise ValueError("Performance requires performance.prediction_manifest; run predict first.")
    resolved = resolve_prediction_manifest(manifest_value)
    output_dir = _resolve_output_dir(resolved, cfg)
    overwrite = bool(performance_cfg.get("overwrite", False))
    if resolved["legacy"]:
        classification_available = resolved["paths"]["classification_predictions"] is not None
    else:
        manifest = resolved["manifest"]
        if "classification_enabled" in manifest:
            classification_available = bool(manifest["classification_enabled"])
        else:
            header = pd.read_csv(resolved["paths"]["prediction_events"], nrows=0)
            classification_available = "classification_score" in header.columns
    reconstruction_available = resolved["paths"]["prediction_events"] is not None and False
    if resolved["legacy"]:
        reconstruction_available = resolved["paths"]["reconstruction_event_metadata"] is not None
    else:
        # The compact table has an explicit reco_valid column when reconstruction was predicted.
        reconstruction_available = False
        header = pd.read_csv(resolved["paths"]["prediction_events"], nrows=0)
        reconstruction_available = "reco_valid" in header.columns
    if not classification_available and not reconstruction_available:
        raise ValueError(f"Prediction products in {resolved['manifest_path']} contain no performance task.")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        if _output_is_complete(
            output_dir, resolved["manifest_path"], classification=classification_available,
            reconstruction=reconstruction_available, kinematics=bool(performance_cfg.get("kinematics", True)),
        ):
            logger.info("Performance result already exists for %s: %s", resolved["manifest_path"], output_dir)
            _print_summary(json.loads((output_dir / "summary.json").read_text()))
            return
        raise FileExistsError(
            f"Performance output directory is non-empty and incompatible or incomplete: {output_dir}. "
            "Use performance.overwrite=true to replace it explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Resolved topology=%s, model cap=%d, source H5=%s", resolved["topology"], resolved["model_max_jets"], ", ".join(map(str, resolved["source_h5_paths"])))
    if resolved["legacy"]:
        totals, auc, reco_efficiency = _run_legacy_performance(
            resolved, output_dir, performance_cfg, classification_available, reconstruction_available
        )
        bin_counts = totals["bins"]
        summary = {
            "prediction_manifest": str(resolved["manifest_path"]), "topology": resolved["topology"],
            "config_name": resolved["config_name"], "model_max_jets": int(resolved["model_max_jets"]),
            "source_h5": [str(path) for path in resolved["source_h5_paths"]], "n_events": int(totals["n_events"]),
            "n_signal": int(totals["n_signal"]), "n_background": int(totals["n_background"]),
            "fully_matched_signal": int(totals["fully_matched"]), "truncated_events": int(totals["truncated"]),
            "truncated_fraction": float(totals["truncated"] / totals["n_events"]), "overall_auc": auc,
            "overall_reconstruction_efficiency": reco_efficiency, "n_jets_bins": bin_counts,
            "output_directory": str(output_dir.resolve()),
        }
        _atomic_json(summary, output_dir / "summary.json")
        logger.info("Performance summary: %s", summary)
        _print_summary(summary)
        return
    events = _prepare_events(resolved)
    if events.empty:
        raise ValueError("Performance received no prediction events.")
    if reconstruction_available and bool(performance_cfg.get("kinematics", True)):
        from joint_event_reconstruction.m0.evaluation.kinematics import write_kinematic_products
        write_kinematic_products(events, output_dir / "reconstruction" / "kinematics", resolved["topology"])
    # Persist the compact event table before plotting.  Large legacy assignment
    # strings are needed for kinematics, but not for any efficiency calculation.
    output_columns = [column for column in events.columns if column != "truth_roles_json"]
    _atomic_frame(events, output_dir / "events.csv.gz", gzip=True, columns=output_columns)
    events.drop(columns=["truth_roles_json", "reco_assignment_json"], errors="ignore", inplace=True)
    auc = _write_classification(events, output_dir, resolved["topology"]) if classification_available else None
    reco_efficiency = None
    if reconstruction_available:
        metrics = _reconstruction_metrics(events, resolved["topology"])
        reco_efficiency = _plot_reconstruction(metrics, output_dir, resolved["topology"])
        if bool(performance_cfg.get("diagnostics", False)):
            _write_diagnostics(events, resolved, output_dir, resolved["topology"])
    signal = events[events["label"] == 1]
    fully = events[events["fully_matched"]]
    bin_counts = {}
    for category in topology_spec(resolved["topology"])["jet_bins"]:
        subset = events[events["n_jets"].map(lambda value: njet_bin(resolved["topology"], value) == category)]
        bin_counts[category] = {
            "n_events": int(len(subset)), "n_signal": int((subset["label"] == 1).sum()),
            "n_background": int((subset["label"] == 0).sum()), "n_fully_matched_signal": int(subset["fully_matched"].sum()),
        }
    summary = {
        "prediction_manifest": str(resolved["manifest_path"]),
        "topology": resolved["topology"], "config_name": resolved["config_name"], "model_max_jets": int(resolved["model_max_jets"]),
        "source_h5": [str(path) for path in resolved["source_h5_paths"]], "n_events": int(len(events)),
        "n_signal": int((events["label"] == 1).sum()), "n_background": int((events["label"] == 0).sum()),
        "fully_matched_signal": int(len(fully)), "truncated_events": int(events["jets_truncated"].sum()),
        "truncated_fraction": float(events["jets_truncated"].mean()), "overall_auc": auc,
        "overall_reconstruction_efficiency": reco_efficiency, "n_jets_bins": bin_counts,
        "output_directory": str(output_dir.resolve()),
    }
    _atomic_json(summary, output_dir / "summary.json")
    logger.info(
        "Performance summary: topology=%s config=%s model_cap=%d events=%d signal=%d background=%d fully_matched=%d truncated=%d (%.3f) AUC=%s reco_efficiency=%s output=%s",
        summary["topology"], summary["config_name"], summary["model_max_jets"], summary["n_events"], summary["n_signal"], summary["n_background"],
        summary["fully_matched_signal"], summary["truncated_events"], summary["truncated_fraction"], summary["overall_auc"], summary["overall_reconstruction_efficiency"], output_dir,
    )
    _print_summary(summary)
