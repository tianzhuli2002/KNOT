import hashlib
import fcntl
import json
import logging
import os
import re
import socket
import tempfile
import time
from collections import Counter
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from joint_event_reconstruction.m0.data.helpers import build_observable_specs
from joint_event_reconstruction.m0.data.scaling import get_transformer_by_name
from joint_event_reconstruction.m0.data.labels import (
    LABEL_PATH_CANDIDATES,
    read_transformer_classification_labels,
    resolve_transformer_label_path,
)
from joint_event_reconstruction.m0.data.splits import index_hash, load_or_create_split

logger = logging.getLogger(__name__)

LABEL_PATH = "/LABELS/GLOBAL"
LOG_OFFSET = float(get_transformer_by_name("Log").offset)
CACHE_VERSION = 9

RECO_IGNORE_INDEX = -100.0
PARTICLE_IDS = OrderedDict(
    [
        ("JET", 1),
        ("ELECTRON", 2),
        ("MUON", 3),
        ("MET", 4),
    ]
)
OBJECT_SOURCES = ("JET", "LEPTON", "MET")

TOKEN_FEATURES = (
    "log_e",
    "e",
    "eta",
    "phi",
    "sin_phi",
    "cos_phi",
    "log_pt",
    "pt",
    "btag",
    "charge",
    "id",
    "is_valid",
)
GLOBAL_FEATURES = (
    "njet",
    "nBjets",
    "nBJets_90",
    "nBJets_85",
    "nBJets_77",
    "nBJets_70",
    "nBJets_65",
    "nelectron",
    "nmuon",
)
STATEFUL_SCALERS = {"standard", "minmax"}


@dataclass(frozen=True)
class FeaturePlan:
    name: str
    variable: str
    scaler: str
    indices: np.ndarray
    expression_offset: float | None = None
    expression_divisor: float | None = None
    transformer: object | None = None


@dataclass(frozen=True)
class SourcePlan:
    name: str
    variables: tuple[str, ...]
    scalers: tuple[str, ...]
    expanded_names: tuple[str, ...]
    features: tuple[FeaturePlan, ...]
    kinematics: dict | None = None
    num_particles: int | None = None


@dataclass(frozen=True)
class TokenSchema:
    names: tuple[str, ...]
    dim: int
    sources: OrderedDict[str, SourcePlan]
    source_type_ids: OrderedDict[str, int]
    has_observables: bool
    has_global_token: bool


def _safe_cache_name(text):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))


def _plain_config_value(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _as_list(value):
    if value is None:
        return []
    if OmegaConf.is_config(value):
        return list(OmegaConf.to_container(value, resolve=True))
    return list(value)


def _normalise_inputs_name(value):
    return str(value).strip().upper()


def _normalise_scaler_name(scaler_name):
    return str(scaler_name).strip().lower()


def _normalise_observables(cfg):
    return build_observable_specs(_as_list(cfg.get("observables", None)))


def _observable_lookup(observables):
    return OrderedDict(
        (_normalise_inputs_name(spec.get("INPUTS", "")), spec)
        for spec in observables
    )


def _has_observables(cfg):
    return len(_normalise_observables(cfg)) > 0


def _stable_json_hash(value, length=12):
    encoded = json.dumps(_plain_config_value(value), sort_keys=True, default=str).encode()
    return hashlib.sha1(encoded).hexdigest()[:length]


def _external_link_identity(h5_path):
    """Fingerprint explicitly linked sidecars in an HDF5 training view."""
    result = {}
    with h5py.File(str(h5_path), "r") as source:
        for name in ("INPUTS", "EVENT_ID", "SPLIT", "META", "LABELS"):
            link = source.get(name, getlink=True)
            if not isinstance(link, h5py.ExternalLink):
                continue
            target = Path(link.filename).expanduser()
            if not target.is_absolute():
                target = Path(h5_path).expanduser().resolve().parent / target
            target = target.resolve()
            if not target.is_file():
                raise FileNotFoundError(
                    f"External HDF5 link {name} points to missing file {target}"
                )
            stat = target.stat()
            result[name] = {
                "path": str(target),
                "object": str(link.path),
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
    return result


def _indices_hash(indices):
    indices = np.asarray(indices, dtype=np.int64)
    return hashlib.sha1(indices.tobytes()).hexdigest()[:12]


def _safe_log(value):
    value = float(value) if np.isfinite(value) else 0.0
    return float(np.log(max(value, 0.0) + LOG_OFFSET))


FIELD_ALIASES = {
    "pt": ("pt", "met"),
    "njet": ("njet", "Njets"),
    "nBjets": ("nBjets", "Nbjets"),
}


def _field(row, name, default=np.nan):
    if row.dtype.names is None:
        return default
    for candidate in FIELD_ALIASES.get(name, (name,)):
        if candidate in row.dtype.names:
            return float(row[candidate])
    return default


def _fields(rows, name, default=0.0):
    if rows.dtype.names is None:
        return np.full(len(rows), default, dtype=np.float32)
    for candidate in FIELD_ALIASES.get(name, (name,)):
        if candidate in rows.dtype.names:
            return rows[candidate]
    return np.full(len(rows), default, dtype=np.float32)


def _is_valid_object(row):
    pt = _field(row, "pt")
    return np.isfinite(pt) and pt > 0.0


def _valid_object_mask(rows):
    if len(rows) == 0 or rows.dtype.names is None:
        return np.zeros(len(rows), dtype=bool)
    if not any(name in rows.dtype.names for name in FIELD_ALIASES.get("pt", ("pt",))):
        return np.zeros(len(rows), dtype=bool)
    pt = _fields(rows, "pt", default=np.nan)
    return np.isfinite(pt) & (pt > 0.0)


def _raw_object_id(row, default=0):
    if row.dtype.names is not None:
        if "is_el" in row.dtype.names and int(row["is_el"]) == 1:
            return 2
        if "is_mu" in row.dtype.names and int(row["is_mu"]) == 1:
            return 3
    return int(round(_field(row, "id", default)))


def _pid_one_hot(raw_id):
    values = np.zeros(len(PARTICLE_IDS), dtype=np.float32)
    raw_to_pid_index = {raw_id: idx for idx, raw_id in enumerate(PARTICLE_IDS.values())}
    type_id = raw_to_pid_index.get(int(round(raw_id)))
    if type_id is not None:
        values[type_id] = 1.0
    return values


def _object_value(row, name, *, default_id=0.0, is_valid=True):
    if name == "log_e":
        return _safe_log(_field(row, "e", 0.0))
    if name == "e":
        return _field(row, "e", 0.0)
    if name == "eta":
        return _field(row, "eta", 0.0)
    if name == "phi":
        return _field(row, "phi", 0.0)
    if name == "sin_phi":
        value = _field(row, "sin_phi", np.nan)
        if np.isfinite(value):
            return value
        return float(np.sin(_field(row, "phi", 0.0)))
    if name == "cos_phi":
        value = _field(row, "cos_phi", np.nan)
        if np.isfinite(value):
            return value
        return float(np.cos(_field(row, "phi", 0.0)))
    if name == "log_pt":
        return _safe_log(_field(row, "pt", 0.0))
    if name == "pt":
        return _field(row, "pt", 0.0)
    if name == "btag":
        return _field(row, "btag", 0.0)
    if name == "charge":
        return _field(row, "charge", 0.0)
    if name == "id":
        return float(_raw_object_id(row, default=default_id))
    if name == "is_valid":
        return 1.0 if is_valid else 0.0
    return _field(row, name, 0.0)


def _object_values(rows, name, *, default_id=0.0):
    if len(rows) == 0:
        return np.empty(0, dtype=np.float32)
    if name == "log_e":
        values = rows["e"] if rows.dtype.names and "e" in rows.dtype.names else np.zeros(len(rows), dtype=np.float32)
        return np.log(np.clip(np.nan_to_num(values, nan=0.0), 0.0, None) + LOG_OFFSET)
    if name == "log_pt":
        values = _fields(rows, "pt", default=0.0)
        return np.log(np.clip(np.nan_to_num(values, nan=0.0), 0.0, None) + LOG_OFFSET)
    if name == "sin_phi":
        if rows.dtype.names and "sin_phi" in rows.dtype.names:
            return np.nan_to_num(rows["sin_phi"], nan=0.0)
        values = rows["phi"] if rows.dtype.names and "phi" in rows.dtype.names else np.zeros(len(rows), dtype=np.float32)
        return np.sin(np.nan_to_num(values, nan=0.0))
    if name == "cos_phi":
        if rows.dtype.names and "cos_phi" in rows.dtype.names:
            return np.nan_to_num(rows["cos_phi"], nan=0.0)
        values = rows["phi"] if rows.dtype.names and "phi" in rows.dtype.names else np.zeros(len(rows), dtype=np.float32)
        return np.cos(np.nan_to_num(values, nan=0.0))
    if name == "id":
        return np.asarray([_raw_object_id(row, default=default_id) for row in rows], dtype=np.float32)
    if name == "is_valid":
        return _valid_object_mask(rows).astype(np.float32)
    if rows.dtype.names:
        return np.nan_to_num(_fields(rows, name, default=0.0), nan=0.0)
    return np.zeros(len(rows), dtype=np.float32)


def _expanded_names(variable, scaler):
    scaler = _normalise_scaler_name(scaler)
    if scaler == "pid":
        return [f"{variable}:pid:{kind}" for kind in ("jet", "electron", "muon", "met")]
    if scaler == "sincos":
        return [f"{variable}:sin", f"{variable}:cos"]
    if _parse_affine_expression(scaler, warn=False) is not None:
        return [f"{variable}:affine"]
    return [f"{variable}:{scaler}"]


def _h5_schema_summary(h5_path, paths):
    summary = {}
    with h5py.File(str(h5_path), "r") as h5_file:
        for path in paths:
            if path not in h5_file:
                summary[path] = None
                continue
            ds = h5_file[path]
            summary[path] = {
                "shape": tuple(int(v) for v in ds.shape),
                "dtype": str(ds.dtype),
                "dtype_names": list(ds.dtype.names or ()),
            }
    return summary


def _parse_variables_and_scalers(spec, *, inputs_name):
    if spec is None:
        raise ValueError(f"Missing observable spec for INPUTS:{inputs_name}.")
    variables = spec.get("variables", spec.get("variable", []))
    variables = [variables] if isinstance(variables, str) else list(variables)
    scalers = list(spec.get("scalers", []))
    if len(variables) != len(scalers):
        raise ValueError(
            f"Transformer observable INPUTS:{inputs_name} has {len(variables)} variables "
            f"but {len(scalers)} scalers."
        )

    supported = GLOBAL_FEATURES if inputs_name == "GLOBAL" else TOKEN_FEATURES
    unknown = [variable for variable in variables if variable not in supported]
    if inputs_name == "GLOBAL" and unknown:
        raise ValueError(
            f"Unsupported transformer observable variable(s) for INPUTS:{inputs_name}: {unknown}. "
            f"Supported variables are: {list(supported)}"
        )
    feature_names = list(spec.get("feature_names", variables))
    if len(feature_names) != len(variables) or len(set(feature_names)) != len(feature_names):
        raise ValueError(
            f"Transformer observable INPUTS:{inputs_name} requires one unique "
            "feature_names entry per source variable."
        )
    return variables, scalers, feature_names


_FLOAT_PATTERN = r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:e[+-]?\d+)?"


def _safe_divisor(divisor, expression, *, warn=True):
    if np.isfinite(divisor) and divisor != 0.0:
        return float(divisor)
    if warn:
        logger.warning(
            "Frozen affine scaler %s has invalid divisor/range %s; using divisor=1.0.",
            expression,
            divisor,
        )
    return 1.0


def _parse_affine_expression(scaler, *, warn=True):
    normalized = _normalise_scaler_name(scaler)

    match = re.fullmatch(rf"x\s*/\s*({_FLOAT_PATTERN})", normalized)
    if match:
        return 0.0, _safe_divisor(float(match.group(1)), normalized, warn=warn)

    match = re.fullmatch(rf"\(\s*x\s*-\s*({_FLOAT_PATTERN})\s*\)\s*/\s*({_FLOAT_PATTERN})", normalized)
    if match:
        return float(match.group(1)), _safe_divisor(float(match.group(2)), normalized, warn=warn)

    match = re.fullmatch(rf"affine:min=({_FLOAT_PATTERN}),scale=({_FLOAT_PATTERN})", normalized)
    if match:
        return float(match.group(1)), _safe_divisor(float(match.group(2)), normalized, warn=warn)

    return None


def _parse_fixed_minmax(scaler):
    normalized = _normalise_scaler_name(scaler)
    match = re.fullmatch(
        rf"minmax:min=({_FLOAT_PATTERN}),max=({_FLOAT_PATTERN})",
        normalized,
    )
    if match is None:
        return None
    minimum, maximum = float(match.group(1)), float(match.group(2))
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(
            f"Fixed MinMax transform {scaler!r} requires finite max > min."
        )
    return minimum, maximum - minimum


def _compile_expression(scaler):
    normalized = _normalise_scaler_name(scaler)
    fixed_minmax = _parse_fixed_minmax(normalized)
    if fixed_minmax is not None:
        return fixed_minmax[0], fixed_minmax[1], None
    affine = _parse_affine_expression(normalized)
    if affine is not None:
        offset, divisor = affine
        return offset, divisor, None
    raise ValueError(
        f"Unsupported transformer transform {scaler!r}. Use none, log, sincos, pid, standard, "
        "minmax, fixed 'minmax:min=<value>,max=<value>', or simple x/<scale>. "
        "Arbitrary Python expressions are not evaluated in the transformer data path."
    )


def _make_feature_plan(variable, scaler, schema_index, *, name=None):
    scaler_name = _normalise_scaler_name(scaler)
    semantic_name = str(name if name is not None else variable)
    names = _expanded_names(semantic_name, scaler_name)
    indices = np.asarray([schema_index[name] for name in names], dtype=np.int64)
    expression_divisor = None
    expression_offset = None
    transformer = None
    if scaler_name not in {"none", "passthrough", "log", "sincos", "pid", "standard", "minmax"}:
        expression_offset, expression_divisor, transformer = _compile_expression(scaler_name)
    return FeaturePlan(
        name=semantic_name,
        variable=variable,
        scaler=scaler_name,
        indices=indices,
        expression_offset=expression_offset,
        expression_divisor=expression_divisor,
        transformer=transformer,
    )


def _source_num_particles(spec, cfg, source_name):
    if source_name == "JET":
        fallback = int(cfg["max_jets"])
    elif source_name == "LEPTON":
        fallback = int(cfg["max_leptons"])
    elif source_name == "MET":
        fallback = 1
    else:
        fallback = 1
    return int(spec.get("num_particles", fallback)) if spec is not None else fallback


def _build_observable_schema(cfg):
    observables = _normalise_observables(cfg)
    lookup = _observable_lookup(observables)
    include_met = bool(cfg.get("include_met", True))
    requested_sources = [
        source_name for source_name in lookup
        if source_name != "MET" or include_met
    ]

    raw_sources = OrderedDict()
    schema_names = []

    for source_name in requested_sources:
        spec = lookup.get(source_name)
        if spec is None:
            raise ValueError(f"Missing observable spec for INPUTS:{source_name}.")
        variables, scalers, feature_names = _parse_variables_and_scalers(spec, inputs_name=source_name)
        expanded = []
        for feature_name, scaler in zip(feature_names, scalers):
            expanded.extend(_expanded_names(feature_name, scaler))
        raw_sources[source_name] = (
            spec,
            tuple(variables),
            tuple(_normalise_scaler_name(s) for s in scalers),
            tuple(feature_names),
            tuple(expanded),
        )
        for name in expanded:
            if name not in schema_names:
                schema_names.append(name)

    schema_index = {name: idx for idx, name in enumerate(schema_names)}
    sources = OrderedDict()

    for source_name, (spec, variables, scalers, feature_names, expanded) in raw_sources.items():
        features = tuple(
            _make_feature_plan(variable, scaler, schema_index, name=feature_name)
            for variable, scaler, feature_name in zip(variables, scalers, feature_names)
        )
        sources[source_name] = SourcePlan(
            name=source_name,
            variables=variables,
            scalers=scalers,
            expanded_names=expanded,
            features=features,
            kinematics=dict(spec.get("kinematics", {})) or None,
            num_particles=_source_num_particles(spec, cfg, source_name),
        )

    return TokenSchema(
        names=tuple(schema_names),
        dim=len(schema_names),
        sources=sources,
        source_type_ids=OrderedDict(
            (source_name, type_id) for type_id, source_name in enumerate(sources)
        ),
        has_observables=True,
        has_global_token="GLOBAL" in sources,
    )


def _validate_token_features(token_features):
    unknown = [name for name in token_features if name not in TOKEN_FEATURES]
    if unknown:
        raise ValueError(
            f"Unsupported transformer token feature(s): {unknown}. "
            f"Supported token features are: {list(TOKEN_FEATURES)}"
        )


def _build_legacy_schema(cfg):
    token_features = tuple(cfg["token_features"])
    _validate_token_features(token_features)
    sources = OrderedDict()
    for source_name in OBJECT_SOURCES:
        features = tuple(
            FeaturePlan(
                name=variable,
                variable=variable,
                scaler="none",
                indices=np.asarray([idx], dtype=np.int64),
            )
            for idx, variable in enumerate(token_features)
        )
        sources[source_name] = SourcePlan(
            name=source_name,
            variables=token_features,
            scalers=tuple("none" for _ in token_features),
            expanded_names=token_features,
            features=features,
            kinematics=None,
            num_particles=_source_num_particles(None, cfg, source_name),
        )
    return TokenSchema(
        names=token_features,
        dim=len(token_features),
        sources=sources,
        source_type_ids=OrderedDict(
            (source_name, type_id) for type_id, source_name in enumerate(sources)
        ),
        has_observables=False,
        has_global_token=False,
    )


def build_token_schema(cfg):
    if _has_observables(cfg):
        return _build_observable_schema(cfg)
    return _build_legacy_schema(cfg)


def token_feature_dim_from_dataset_cfg(cfg):
    return build_token_schema(cfg).dim


def token_feature_names_from_dataset_cfg(cfg):
    return list(build_token_schema(cfg).names)


def _stratify_or_none(labels):
    _, counts = np.unique(labels, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        return None
    return labels


def _label_counts(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def _configured_widths_from_h5(h5_file, schema, cfg):
    widths = {}
    for source_name in _token_source_names(schema):
        if source_name not in schema.sources:
            widths[source_name] = 0
            continue
        if source_name == "MET" and not bool(cfg.get("include_met", True)):
            widths[source_name] = 0
            continue
        h5_path = f"/INPUTS/{source_name}"
        if h5_path not in h5_file:
            widths[source_name] = 0
            continue
        dataset_shape = h5_file[h5_path].shape
        dataset_width = dataset_shape[1] if len(dataset_shape) > 1 else 1
        configured = schema.sources[source_name].num_particles
        widths[source_name] = min(int(configured), dataset_width)
    return widths


def _feature_h5_requirements(source_name, variable, available_fields, cfg):
    """Return alternative upstream field sets for one explicitly derived feature."""
    fields = set(available_fields)
    if source_name == "GLOBAL":
        if variable == "njet":
            return () if variable in fields else (("/INPUTS/JET", "pt"),)
        if variable.startswith("nBJets_"):
            if variable in fields:
                return ()
            thresholds = _plain_config_value(cfg.get("btag_working_point_thresholds", {}) or {})
            if variable in thresholds:
                return (("/INPUTS/JET", "pt"), ("/INPUTS/JET", "btag"))
            return None
        if variable == "nBjets":
            return () if any(name in fields for name in FIELD_ALIASES[variable]) else None
        if variable in {"nelectron", "nmuon"}:
            if variable in fields:
                return ()
            return (("/INPUTS/LEPTON", "pt"), ("/INPUTS/LEPTON", "id|is_el+is_mu"))
        return () if variable in fields else None

    if variable in fields:
        return ()
    if variable == "log_e":
        return () if "e" in fields else None
    if variable in {"log_pt", "is_valid"}:
        return () if any(name in fields for name in FIELD_ALIASES.get("pt", ("pt",))) else None
    if variable in {"sin_phi", "cos_phi"}:
        return () if "phi" in fields else None
    if variable == "id" and source_name in {"JET", "MET"}:
        return ()
    if variable == "id" and source_name == "LEPTON" and {"is_el", "is_mu"}.issubset(fields):
        return ()
    return None


def validate_transformer_h5_inputs(
    h5_path,
    schema,
    dataset_cfg,
    *,
    reconstruction_cfg=None,
    classification_enabled=True,
    label_override=None,
    config_name="transformer",
):
    """Validate all configured scientific inputs together before reading events."""
    h5_path = str(h5_path)
    if not Path(h5_path).is_file():
        raise FileNotFoundError(f"Transformer input H5 does not exist: {h5_path} (config={config_name}).")

    problems = []
    with h5py.File(h5_path, "r") as h5_file:
        available_inputs = sorted(h5_file.get("/INPUTS", {}).keys())
        reference_events = None
        for source_name, plan in schema.sources.items():
            path = f"/INPUTS/{source_name}"
            if path not in h5_file:
                problems.append(
                    f"missing source={source_name} expected={path} available /INPUTS keys={available_inputs}"
                )
                continue
            dataset = h5_file[path]
            if not isinstance(dataset, h5py.Dataset):
                problems.append(f"source={source_name} expected dataset at {path}, found {type(dataset).__name__}")
                continue
            if dataset.ndim < 1:
                problems.append(f"source={source_name} expected event dimension at {path}, shape={dataset.shape}")
                continue
            if reference_events is None:
                reference_events = int(dataset.shape[0])
            elif int(dataset.shape[0]) != reference_events:
                problems.append(
                    f"source={source_name} event dimension at {path} is {dataset.shape[0]}, expected {reference_events}"
                )
            required_width = 1 if source_name == "GLOBAL" else int(plan.num_particles or 1)
            actual_width = int(dataset.shape[1]) if dataset.ndim > 1 else 1
            if actual_width < required_width:
                problems.append(
                    f"source={source_name} wrong width at {path}: found {actual_width}, expected at least {required_width}"
                )
            if dataset.dtype.names is None:
                problems.append(
                    f"source={source_name} at {path} must have named numeric feature fields; dtype={dataset.dtype}"
                )
                continue
            fields = tuple(dataset.dtype.names)
            for variable in plan.variables:
                requirements = _feature_h5_requirements(source_name, variable, fields, dataset_cfg)
                if requirements is None:
                    expected = f"{path}/{variable}"
                    if source_name == "GLOBAL" and variable.startswith("nBJets_"):
                        expected += f" or dataset.btag_working_point_thresholds.{variable} with /INPUTS/JET/btag"
                    problems.append(
                        f"missing feature={variable} source={source_name} expected={expected} "
                        f"available fields at {path}={list(fields)}"
                    )
                    continue
                for required_path, required_field in requirements:
                    if required_path not in h5_file:
                        problems.append(
                            f"missing derived-input source for feature={variable}: expected={required_path} "
                            f"available /INPUTS keys={available_inputs}"
                        )
                        continue
                    required_fields = set(h5_file[required_path].dtype.names or ())
                    if required_field == "id|is_el+is_mu":
                        found = "id" in required_fields or {"is_el", "is_mu"}.issubset(required_fields)
                    else:
                        found = required_field in required_fields
                    if not found:
                        problems.append(
                            f"missing derived-input feature for config feature={variable}: "
                            f"expected={required_path}/{required_field} available fields={sorted(required_fields)}"
                        )
            for field in fields:
                if not np.issubdtype(dataset.dtype[field], np.number):
                    problems.append(
                        f"non-numeric configured source field at {path}/{field}: dtype={dataset.dtype[field]}"
                    )

        if classification_enabled and label_override is None:
            try:
                label_path = resolve_transformer_label_path(h5_file)
            except KeyError:
                problems.append(
                    f"missing classification label expected one of={list(LABEL_PATH_CANDIDATES)} "
                    f"available root keys={sorted(h5_file.keys())}"
                )
            else:
                label_dataset = h5_file[label_path]
                if reference_events is not None and int(label_dataset.shape[0]) != reference_events:
                    problems.append(
                        f"classification label event dimension at {label_path} is {label_dataset.shape[0]}, "
                        f"expected {reference_events}"
                    )

        reco_cfg = reconstruction_cfg or {}
        if bool(reco_cfg.get("enabled", False)):
            label_group = _normalise_h5_group(reco_cfg.get("label_source", "LABELS"))
            targets = parse_reconstruction_targets(reco_cfg, _token_source_names(schema))
            requested_sources = sorted({source for target in targets for source, _ in target.members})
            for source_name in requested_sources:
                label_path = f"{label_group}/{source_name}"
                if label_path not in h5_file:
                    nearby = sorted(h5_file.get(label_group, {}).keys()) if label_group in h5_file else []
                    problems.append(
                        f"missing reconstruction label source={source_name} expected={label_path} "
                        f"available keys at {label_group}={nearby}"
                    )
                    continue
                label_dataset = h5_file[label_path]
                required_width = int(schema.sources[source_name].num_particles or 1)
                if label_dataset.ndim < 2 or int(label_dataset.shape[1]) < required_width:
                    problems.append(
                        f"reconstruction labels wrong width at {label_path}: shape={label_dataset.shape}, "
                        f"expected [event, >= {required_width}]"
                    )
                if reference_events is not None and int(label_dataset.shape[0]) != reference_events:
                    problems.append(
                        f"reconstruction label event dimension at {label_path} is {label_dataset.shape[0]}, "
                        f"expected {reference_events}"
                    )

    if problems:
        formatted = "\n  - ".join(problems)
        raise ValueError(
            f"Configured transformer H5 inputs are invalid.\n"
            f"input H5 path: {h5_path}\nconfig name: {config_name}\n  - {formatted}"
        )


def _token_source_names(schema):
    return tuple(source_name for source_name in schema.sources if source_name != "GLOBAL")


def _source_token_ranges(schema, widths):
    ranges = OrderedDict()
    cursor = 0
    for source_name in _token_source_names(schema):
        width = int(widths.get(source_name, 0))
        if width > 0:
            ranges[source_name] = tuple(range(cursor, cursor + width))
        cursor += width
    return ranges


def _reconstruction_cfg(full_config):
    tasks_cfg = full_config.get("tasks", {}) if full_config is not None else {}
    return tasks_cfg.get("reconstruction", {}) or {}


def _reconstruction_enabled(full_config):
    cfg = _reconstruction_cfg(full_config)
    return bool(cfg.get("enabled", False))


@dataclass(frozen=True)
class ReconstructionTarget:
    name: str
    members: tuple

    @property
    def size(self):
        return len(self.members)

    @property
    def member_counter(self):
        return Counter(self.members)

    @property
    def source_counter(self):
        return Counter(source for source, _ in self.members)


@dataclass(frozen=True)
class CandidatePool:
    source_signature: tuple
    targets: tuple

    @property
    def name(self):
        return "__".join(f"{source}x{count}" for source, count in self.source_signature)

    @property
    def candidate_size(self):
        return sum(count for _, count in self.source_signature)

    @property
    def member_sources(self):
        return tuple(source for source, count in self.source_signature for _ in range(count))

    @property
    def class_names(self):
        return ("background",) + tuple(target.name for target in self.targets)

    @property
    def class_to_index(self):
        return {name: index for index, name in enumerate(self.class_names)}


def parse_reconstruction_targets(reconstruction_cfg, configured_sources):
    """Parse physical reconstruction targets in configured order."""
    cfg = reconstruction_cfg or {}
    obsolete = sorted(set(cfg) & {"target", "target_weights", "encoding", "edge", "hyperedge", "kind"})
    if obsolete:
        raise ValueError(
            f"Obsolete reconstruction setting(s) {obsolete}; use only tasks.reconstruction.targets. "
            "Old binary typed-target checkpoints and configs are incompatible."
        )
    target_cfg = cfg.get("targets", None)
    if target_cfg is None:
        raise ValueError("Enabled transformer reconstruction requires tasks.reconstruction.targets.")
    if OmegaConf.is_config(target_cfg):
        target_cfg = OmegaConf.to_container(target_cfg, resolve=True)
    if not isinstance(target_cfg, dict) or not target_cfg:
        raise ValueError("tasks.reconstruction.targets must be a non-empty mapping.")
    sources = {str(source).upper() for source in configured_sources}
    parsed = []
    for raw_name, raw_members in target_cfg.items():
        name = "" if raw_name is None else str(raw_name).strip()
        if not name:
            raise ValueError("Every reconstruction target requires a non-empty name.")
        members = []
        for member in _as_list(raw_members):
            values = _as_list(member)
            if len(values) != 2:
                raise ValueError(f"Reconstruction target {name!r} member must be [SOURCE, LABEL], found {values!r}.")
            source = str(values[0]).upper()
            if source not in sources:
                raise ValueError(f"Reconstruction target {name!r} uses unknown source {source!r}; configured={sorted(sources)}.")
            label = values[1]
            if isinstance(label, bool):
                raise ValueError(f"Reconstruction target {name!r} label must be a finite integer, found {label!r}.")
            try:
                numeric, integer = float(label), int(label)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Reconstruction target {name!r} label must be a finite integer, found {label!r}.") from exc
            if not np.isfinite(numeric) or numeric != float(integer):
                raise ValueError(f"Reconstruction target {name!r} label must be a finite integer, found {label!r}.")
            if integer == int(cfg.get("ignore_index", RECO_IGNORE_INDEX)):
                raise ValueError(f"Reconstruction target {name!r} cannot require ignore_index label {integer}.")
            members.append((source, integer))
        if not members:
            raise ValueError(f"Reconstruction target {name!r} requires at least one member.")
        parsed.append(ReconstructionTarget(name=name, members=tuple(members)))
    return tuple(parsed)


def infer_candidate_pools(targets, configured_sources):
    """Group targets by deterministic source multiplicity signature."""
    source_order = {str(source).upper(): index for index, source in enumerate(configured_sources)}
    grouped = OrderedDict()
    for target in targets:
        counts = target.source_counter
        signature = tuple(
            (source, counts[source]) for source in sorted(counts, key=source_order.__getitem__)
        )
        grouped.setdefault(signature, []).append(target)
    return tuple(CandidatePool(signature, tuple(pool_targets)) for signature, pool_targets in grouped.items())


def _normalise_h5_group(value):
    group = str(value if value is not None else "LABELS").strip().strip("/")
    if not group:
        raise ValueError("tasks.reconstruction.label_source must name a non-empty H5 group.")
    return f"/{group}"


def validate_transformer_task_config(config):
    """Fail early on task combinations that cannot produce meaningful outputs."""
    tasks = config.get("tasks", {}) or {}
    classification = tasks.get("classification", {}) or {}
    reconstruction = tasks.get("reconstruction", {}) or {}
    classification_enabled = bool(classification.get("enabled", True))
    reconstruction_enabled = bool(reconstruction.get("enabled", False))

    if not classification_enabled and not reconstruction_enabled:
        mode = str((config.get("general", {}) or {}).get("mode", "train"))
        if mode not in {"predict", "sample"}:
            raise ValueError("Transformer classification and reconstruction cannot both be disabled during training.")
    for task_name, task_cfg in (("classification", classification), ("reconstruction", reconstruction)):
        weight = float(task_cfg.get("weight", 1.0 if task_name == "classification" else 0.0))
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(f"tasks.{task_name}.weight must be finite and non-negative, found {weight!r}.")

    if bool(config.get("train", {}).get("compute_val_auc", False)) and not classification_enabled:
        raise ValueError("train.compute_val_auc=true requires classification to be enabled.")

    predict_cfg = config.get("predict", {}) or {}
    if bool(predict_cfg.get("write_classification", classification_enabled)) and not classification_enabled:
        raise ValueError("predict.write_classification=true requires classification to be enabled.")
    if bool(predict_cfg.get("write_reconstruction", reconstruction_enabled)) and not reconstruction_enabled:
        raise ValueError("predict.write_reconstruction=true requires reconstruction to be enabled.")
    candidate_limit = predict_cfg.get("candidate_table_max_events", None)
    if bool(predict_cfg.get("write_candidate_table", reconstruction_enabled)) and candidate_limit is not None:
        if int(candidate_limit) <= 0:
            raise ValueError("predict.candidate_table_max_events must be positive or null.")

    if reconstruction_enabled:
        _normalise_h5_group(reconstruction.get("label_source", "LABELS"))
        schema = build_token_schema(config.dataset)
        source_names = set(_token_source_names(schema))
        targets = parse_reconstruction_targets(reconstruction, source_names)
        infer_candidate_pools(targets, _token_source_names(schema))

    train_cfg = config.get("train", {}) or {}
    checkpoint_cfg = train_cfg.get("checkpoint", None)
    if checkpoint_cfg is None:
        raise ValueError(
            "Maintained transformer configurations require train.checkpoint.monitor, mode, save_top_k, and save_last."
        )
    monitor = str(checkpoint_cfg.get("monitor", ""))
    logged_metrics = {"val_loss"}
    if classification_enabled:
        logged_metrics.add("val_classification_loss")
        if bool(train_cfg.get("compute_val_auc", False)):
            logged_metrics.add("val_auc")
    if reconstruction_enabled:
        logged_metrics.add("val_reconstruction_loss")
        parsed = parse_reconstruction_targets(reconstruction, _token_source_names(build_token_schema(config.dataset)))
        pools = infer_candidate_pools(parsed, _token_source_names(build_token_schema(config.dataset)))
        for pool in pools:
            name = pool.name
            for suffix in (
                "loss", "candidate_count", "active_event_count", "active_fraction",
            ):
                logged_metrics.add(f"val_reco_{name}_{suffix}")
            for role in pool.class_names[1:]:
                for suffix in ("class_count", "top1_accuracy", "top3_accuracy", "mean_reciprocal_rank", "random_top1_expectation"):
                    logged_metrics.add(f"val_reco_{role}_{suffix}")
    if monitor not in logged_metrics:
        raise ValueError(
            f"train.checkpoint.monitor={monitor!r} is not logged for the enabled transformer tasks; "
            f"available validation metrics are {sorted(logged_metrics)}."
        )
    monitor_mode = str(checkpoint_cfg.get("mode", ""))
    if monitor_mode not in {"min", "max"}:
        raise ValueError("train.checkpoint.mode must be 'min' or 'max'.")
    if monitor == "val_auc" and monitor_mode != "max":
        raise ValueError("val_auc must be monitored with train.checkpoint.mode=max.")
    if monitor.endswith(("_top1_accuracy", "_positive_recall_05")) and monitor_mode != "max":
        raise ValueError(f"Accuracy metric {monitor!r} must be monitored with train.monitor_mode=max.")
    if (monitor.endswith("loss") or monitor == "val_loss") and monitor_mode != "min":
        raise ValueError(f"Loss metric {monitor!r} must be monitored with train.monitor_mode=min.")
    optimise_stage = str((config.get("optimise", {}) or {}).get("stage", ""))
    if optimise_stage in {"architecture", "task_weight"} and (monitor, monitor_mode) != ("val_loss", "min"):
        raise ValueError(
            f"Transformer {optimise_stage} tuning requires train.checkpoint=(val_loss, min); "
            f"found {(monitor, monitor_mode)}."
        )
    if int(checkpoint_cfg.get("save_top_k", 1)) != 1:
        raise ValueError("Transformer train.checkpoint.save_top_k must be exactly 1.")
    if bool(checkpoint_cfg.get("save_last", False)):
        raise ValueError("Transformer train.checkpoint.save_last must be false for the one-best-checkpoint policy.")

    additional_checkpoints = train_cfg.get("additional_checkpoints", {}) or {}
    for checkpoint_name, additional in additional_checkpoints.items():
        additional = additional or {}
        additional_monitor = str(additional.get("monitor", ""))
        additional_mode = str(additional.get("mode", ""))
        if additional_monitor not in logged_metrics:
            raise ValueError(
                f"train.additional_checkpoints.{checkpoint_name}.monitor={additional_monitor!r} "
                f"is not logged for the enabled tasks; available={sorted(logged_metrics)}."
            )
        if additional_mode not in {"min", "max"}:
            raise ValueError(
                f"train.additional_checkpoints.{checkpoint_name}.mode must be 'min' or 'max'."
            )
        if additional_monitor == "val_auc" and additional_mode != "max":
            raise ValueError("Additional val_auc checkpoints must use mode=max.")
        if (additional_monitor.endswith("loss") or additional_monitor == "val_loss") and additional_mode != "min":
            raise ValueError(
                f"Additional loss checkpoint {additional_monitor!r} must use mode=min."
            )
        if int(additional.get("save_top_k", 1)) != 1:
            raise ValueError("Transformer additional checkpoints must use save_top_k=1.")
        if bool(additional.get("save_last", False)):
            raise ValueError("Transformer additional checkpoints must use save_last=false.")

    scheduler_cfg = train_cfg.get("lr_scheduler", {}) or {}
    if bool(scheduler_cfg.get("enabled", False)):
        if "monitor" not in scheduler_cfg:
            raise ValueError("Enabled transformer train.lr_scheduler requires an explicit monitor.")
        scheduler_monitor = str(scheduler_cfg.get("monitor"))
        if scheduler_monitor not in logged_metrics:
            raise ValueError(
                f"train.lr_scheduler.monitor={scheduler_monitor!r} is not logged for the enabled tasks."
            )
        scheduler_mode = str(scheduler_cfg.get("mode", ""))
        if scheduler_mode not in {"min", "max"}:
            raise ValueError("train.lr_scheduler.mode must be 'min' or 'max'.")
        if scheduler_monitor == "val_auc" and scheduler_mode != "max":
            raise ValueError("A val_auc scheduler monitor requires train.lr_scheduler.mode=max.")
        if scheduler_monitor.endswith("loss") and scheduler_mode != "min":
            raise ValueError(
                f"Loss scheduler monitor {scheduler_monitor!r} requires train.lr_scheduler.mode=min."
            )

    early_cfg = train_cfg.get("early_stopping", {}) or {}
    if bool(early_cfg.get("enabled", True)):
        early_monitor = str(early_cfg.get("monitor", monitor))
        early_mode = str(early_cfg.get("mode", monitor_mode))
        if early_monitor != monitor or early_mode != monitor_mode:
            raise ValueError(
                "Transformer early stopping must use the checkpoint monitor and mode; "
                f"checkpoint=({monitor}, {monitor_mode}), early_stopping=({early_monitor}, {early_mode})."
            )

    trainer_cfg = config.get("trainer", {}) or {}
    cadence = int(trainer_cfg.get("check_val_every_n_epoch", 1))
    n_epochs = int(train_cfg.get("n_epochs", 0))
    if cadence < 1:
        raise ValueError("trainer.check_val_every_n_epoch must be at least 1.")
    if trainer_cfg.get("limit_val_batches", None) == 0:
        raise ValueError("Validation-dependent transformer training requires trainer.limit_val_batches > 0.")
    if n_epochs < cadence:
        raise ValueError(
            f"No scheduled validation pass: train.n_epochs={n_epochs}, "
            f"trainer.check_val_every_n_epoch={cadence}. Checkpointing, scheduling, and early stopping "
            "require at least one validation check."
        )


class ReconstructionCandidateBuilder:
    """Build each source-constrained candidate pool once and assign semantic truth."""

    def __init__(self, reconstruction_cfg, schema, widths):
        self.config = reconstruction_cfg or {}
        self.schema = schema
        self.ignore_index = float(self.config.get("ignore_index", RECO_IGNORE_INDEX))
        self.label_group = _normalise_h5_group(self.config.get("label_source", "LABELS"))
        self.source_ranges = _source_token_ranges(schema, widths)
        self.targets = parse_reconstruction_targets(self.config, self.source_ranges)
        self.pools = infer_candidate_pools(self.targets, self.source_ranges)
        self._candidate_templates = {}

    def build_event(self, raw_labels, padding_mask, event_index=None):
        raw_labels = np.asarray(raw_labels, dtype=np.float32)
        padding_mask = np.asarray(padding_mask, dtype=bool)
        valid_tokens = ~padding_mask[: raw_labels.shape[0]]
        if not np.isfinite(raw_labels[valid_tokens]).all():
            bad = float(raw_labels[valid_tokens][~np.isfinite(raw_labels[valid_tokens])][0])
            raise ValueError(f"Non-finite reconstruction label {bad!r}; event={event_index}.")
        result = OrderedDict()
        for pool in self.pools:
            candidates_arr, local_arr, source_slices = self._candidate_template(pool, padding_mask)
            truth_class = np.zeros(candidates_arr.shape[0], dtype=np.int64)
            matched_roles = [[] for _ in range(candidates_arr.shape[0])]
            for class_index, target in enumerate(pool.targets, start=1):
                matches = np.ones(candidates_arr.shape[0], dtype=bool)
                for source, source_slice in source_slices.items():
                    values = raw_labels[candidates_arr[:, source_slice]]
                    invalid_fractional = (
                        np.isfinite(values) & (values != self.ignore_index) & (values != np.rint(values))
                    )
                    if invalid_fractional.any():
                        bad = float(values[invalid_fractional][0])
                        raise ValueError(
                            f"Non-integer reconstruction label {bad!r}; event={event_index}, "
                            f"candidate pool={pool.source_signature}, source={source}."
                        )
                    valid = np.isfinite(values) & (values != self.ignore_index) & (values == np.rint(values))
                    expected = np.sort(np.asarray(
                        [label for member_source, label in target.members if member_source == source],
                        dtype=np.float32,
                    ))
                    matches &= valid.all(axis=1)
                    matches &= (np.sort(values, axis=1) == expected).all(axis=1)
                for candidate_index in np.flatnonzero(matches):
                    matched_roles[candidate_index].append(target.name)
                    truth_class[candidate_index] = class_index
            ambiguous = [index for index, roles in enumerate(matched_roles) if len(roles) > 1]
            if ambiguous:
                index = ambiguous[0]
                raise ValueError(
                    f"Ambiguous semantic reconstruction truth: event={event_index}, "
                    f"candidate pool={pool.source_signature}, candidate_indices={candidates_arr[index].tolist()}, "
                    f"matched_roles={matched_roles[index]}."
                )
            result[pool.name] = {
                "candidate_indices": candidates_arr,
                "candidate_source_indices": local_arr,
                "truth_class": truth_class,
                "event_active": bool(np.any(truth_class != 0)),
                "source_signature": pool.source_signature,
                "class_names": pool.class_names,
                "class_to_index": pool.class_to_index,
                "candidate_size": pool.candidate_size,
                "member_sources": pool.member_sources,
            }
        return result

    def _candidate_template(self, pool, padding_mask):
        valid_locals = OrderedDict()
        for source, _ in pool.source_signature:
            indices = self.source_ranges[source]
            valid_locals[source] = tuple(
                local for local, global_idx in enumerate(indices)
                if global_idx < len(padding_mask) and not bool(padding_mask[global_idx])
            )
        key = (pool.source_signature, tuple((source, locals_) for source, locals_ in valid_locals.items()))
        cached = self._candidate_templates.get(key)
        if cached is not None:
            return cached

        per_source_choices = []
        for source, multiplicity in pool.source_signature:
            choices = tuple(combinations(valid_locals[source], multiplicity))
            if not choices:
                empty = (
                    np.empty((0, pool.candidate_size), dtype=np.int64),
                    np.empty((0, pool.candidate_size), dtype=np.int64),
                    OrderedDict(),
                )
                self._candidate_templates[key] = empty
                return empty
            per_source_choices.append((source, choices))

        global_rows = []
        local_rows = []
        source_slices = OrderedDict()
        cursor = 0
        for source, multiplicity in pool.source_signature:
            source_slices[source] = slice(cursor, cursor + multiplicity)
            cursor += multiplicity
        for choice_product in product(*(choices for _, choices in per_source_choices)):
            local_row = tuple(local for choice in choice_product for local in choice)
            global_row = tuple(
                self.source_ranges[source][local]
                for (source, _), choice in zip(per_source_choices, choice_product)
                for local in choice
            )
            local_rows.append(local_row)
            global_rows.append(global_row)
        template = (
            np.asarray(global_rows, dtype=np.int64),
            np.asarray(local_rows, dtype=np.int64),
            source_slices,
        )
        self._candidate_templates[key] = template
        return template

    def build_batch(self, per_event_targets):
        batch_size = len(per_event_targets)
        batch_pools = {}
        for pool in self.pools:
            name = pool.name
            max_candidates = max((event[name]["candidate_indices"].shape[0] for event in per_event_targets), default=0)
            candidate_indices = torch.zeros((batch_size, max_candidates, pool.candidate_size), dtype=torch.long)
            candidate_source_indices = torch.zeros_like(candidate_indices)
            candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool)
            truth_class = torch.zeros((batch_size, max_candidates), dtype=torch.long)
            event_active = torch.zeros(batch_size, dtype=torch.bool)

            for event_idx, event in enumerate(per_event_targets):
                entry = event[name]
                n_candidates = int(entry["candidate_indices"].shape[0])
                if n_candidates > 0:
                    candidate_indices[event_idx, :n_candidates].copy_(torch.from_numpy(entry["candidate_indices"]))
                    candidate_source_indices[event_idx, :n_candidates].copy_(torch.from_numpy(entry["candidate_source_indices"]))
                    truth_class[event_idx, :n_candidates].copy_(torch.from_numpy(entry["truth_class"]))
                    candidate_mask[event_idx, :n_candidates] = True
                event_active[event_idx] = bool(entry["event_active"])

            batch_pools[name] = {
                "candidate_indices": candidate_indices,
                "candidate_source_indices": candidate_source_indices,
                "candidate_mask": candidate_mask,
                "truth_class": truth_class,
                "event_active": event_active,
                "source_signature": pool.source_signature,
                "class_names": pool.class_names,
                "class_to_index": pool.class_to_index,
                "candidate_size": pool.candidate_size,
                "member_sources": pool.member_sources,
            }
        return {"pools": batch_pools}


class TransformerH5Dataset(Dataset):
    def __init__(
        self,
        h5_path,
        cfg,
        indices,
        *,
        label_override=None,
        scaler_state=None,
        schema=None,
        configured_widths=None,
        reconstruction_cfg=None,
        classification_enabled=True,
    ):
        self.h5_path = str(h5_path)
        self.config = cfg
        self.indices = np.asarray(indices, dtype=np.int64)
        self.label_override = None if label_override is None else float(label_override)
        self.schema = schema or build_token_schema(cfg)
        self.scaler_state = scaler_state or {}
        self.btag_working_point_thresholds = {
            str(name): float(threshold)
            for name, threshold in _plain_config_value(
                cfg.get("btag_working_point_thresholds", {}) or {}
            ).items()
        }
        self.include_met = bool(cfg.get("include_met", True))
        self._files = {}
        self.reconstruction_cfg = reconstruction_cfg or {}
        self.reconstruction_enabled = bool(self.reconstruction_cfg.get("enabled", False))
        self.classification_enabled = bool(classification_enabled)

        with h5py.File(self.h5_path, "r") as h5_file:
            self.widths = dict(configured_widths or _configured_widths_from_h5(h5_file, self.schema, cfg))
            self.label_path = (
                resolve_transformer_label_path(h5_file)
                if self.label_override is None and self.classification_enabled
                else None
            )

        self.total_tokens = (
            sum(self.widths.get(source_name, 0) for source_name in _token_source_names(self.schema))
            + int(self.schema.has_global_token)
        )
        self.reco_builder = (
            ReconstructionCandidateBuilder(self.reconstruction_cfg, self.schema, self.widths)
            if self.reconstruction_enabled
            else None
        )
        self.reco_source_ids = self._reco_source_ids() if self.reco_builder is not None else None

    def _reco_source_ids(self):
        """Stable source identity for prediction metadata; never used by the model."""
        source_ids = np.zeros(self.total_tokens, dtype=np.int8)
        for source, indices in self.reco_builder.source_ranges.items():
            if source in self.schema.source_type_ids:
                source_ids[list(indices)] = self.schema.source_type_ids[source]
        return source_ids

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_files"] = {}
        return state

    def _h5_file(self):
        if self.h5_path not in self._files:
            self._files[self.h5_path] = h5py.File(self.h5_path, "r")
        return self._files[self.h5_path]

    def close(self):
        for h5_file in self._files.values():
            h5_file.close()
        self._files = {}

    def __getitem__(self, item):
        event_idx = int(self.indices[item])
        tokens, type_ids, padding_mask, label = self._build_event_arrays(event_idx)
        result = {
            "tokens": torch.from_numpy(tokens),
            "type_ids": torch.from_numpy(type_ids),
            "padding_mask": torch.from_numpy(padding_mask),
            "labels": torch.tensor(label, dtype=torch.float32),
            "event_index": torch.tensor(
                int(self._h5_file()["/EVENT_ID"][event_idx]) if "/EVENT_ID" in self._h5_file() else event_idx,
                dtype=torch.long,
            ),
        }
        result["physics_vectors"] = torch.from_numpy(self._build_physics_vectors(event_idx))
        if self.reconstruction_enabled:
            h5_file = self._h5_file()
            raw_labels = self._build_reco_raw_labels(h5_file, event_idx, padding_mask)
            result["reco_raw_labels"] = torch.from_numpy(raw_labels)
            result["reco_source_ids"] = torch.from_numpy(self.reco_source_ids.copy())
            result["reco_targets"] = self.reco_builder.build_event(
                raw_labels, padding_mask, event_index=event_idx
            )
        return result

    def _build_physics_vectors(self, event_idx):
        """Raw `(E,px,py,pz)` vectors aligned with model token positions."""
        h5_file = self._h5_file()
        source_rows = {
            source: self._read_source_rows(h5_file, source, event_idx)
            for source in _token_source_names(self.schema)
            if source != "GLOBAL"
        }
        return self._build_physics_vectors_from_rows(source_rows)

    def _build_physics_vectors_from_rows(self, source_rows):
        vectors = np.zeros((self.total_tokens, 4), dtype=np.float32)
        cursor = 0
        for source_name in _token_source_names(self.schema):
            width = self.schema.sources[source_name].num_particles
            if source_name == "GLOBAL":
                cursor += 1
                continue
            rows = source_rows.get(source_name, [])
            kinematics = self.schema.sources[source_name].kinematics
            if not kinematics:
                raise ValueError(
                    f"INPUTS:{source_name} has no explicit physical kinematics contract"
                )
            for local, row in enumerate(rows[:width]):
                fields = row.dtype.names or ()
                if source_name != "MET" and not _is_valid_object(row):
                    continue
                required = {
                    str(value)
                    for key, value in kinematics.items()
                    if key != "representation"
                }
                missing = sorted(required - set(fields))
                if missing:
                    raise KeyError(
                        f"{self.h5_path}:/INPUTS/{source_name} lacks configured "
                        f"kinematic fields {missing}"
                    )
                pt = float(row[str(kinematics["pt"])])
                if "phi" in kinematics:
                    phi = float(row[str(kinematics["phi"])])
                else:
                    phi = float(np.arctan2(
                        row[str(kinematics["sin_phi"])],
                        row[str(kinematics["cos_phi"])],
                    ))
                eta = float(row[str(kinematics["eta"])]) if "eta" in kinematics else 0.0
                energy = float(row[str(kinematics["energy"])]) if "energy" in kinematics else pt * np.cosh(eta)
                if np.isfinite((energy, pt, eta, phi)).all():
                    vectors[cursor + local] = (energy, pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta))
            cursor += width
        return vectors

    def _build_event_arrays(self, event_idx):
        h5_file = self._h5_file()
        source_rows = {
            source_name: self._read_source_rows(h5_file, source_name, event_idx)
            for source_name in _token_source_names(self.schema)
            if source_name != "GLOBAL"
        }
        label = self._read_label(h5_file, event_idx)
        return self._build_event_arrays_from_loaded(
            h5_file, event_idx, source_rows.get("JET", []), source_rows.get("LEPTON", []),
            source_rows.get("MET", []), label, source_rows=source_rows,
        )

    def _build_event_arrays_from_loaded(self, h5_file, event_idx, jets, leptons, met, label, source_rows=None):
        source_rows = dict(source_rows or {"JET": jets, "LEPTON": leptons, "MET": met})
        for source_name in _token_source_names(self.schema):
            if source_name not in source_rows:
                source_rows[source_name] = self._read_source_rows(h5_file, source_name, event_idx)
        global_row = None
        if self.schema.has_global_token:
            global_ds = h5_file["/INPUTS/GLOBAL"]
            global_row = global_ds[event_idx]
            if getattr(global_row, "shape", ()):
                global_row = global_row.reshape(-1)[0]
        return self._build_event_arrays_from_rows(source_rows, global_row, label)

    def _build_event_arrays_from_rows(self, source_rows, global_row, label):
        tokens = np.zeros((self.total_tokens, self.schema.dim), dtype=np.float32)
        type_ids = np.zeros(self.total_tokens, dtype=np.int64)
        padding_mask = np.ones(self.total_tokens, dtype=bool)
        cursor = 0

        jets = source_rows.get("JET", [])
        leptons = source_rows.get("LEPTON", [])
        for source_name in _token_source_names(self.schema):
            if source_name == "GLOBAL":
                continue
            rows = source_rows[source_name]
            cursor = self._fill_object_tokens(
                source_name, rows, cursor, tokens, type_ids, padding_mask,
                default_id=1.0 if source_name == "JET" else (4.0 if source_name == "MET" else 0.0),
                accept_all=source_name == "MET",
            )

        if self.schema.has_global_token:
            self._fill_global_token_from_row(global_row, jets, leptons, tokens[cursor])
            type_ids[cursor] = self._source_type_id("GLOBAL")
            padding_mask[cursor] = False

        return tokens, type_ids, padding_mask, label

    def _build_reco_raw_labels(self, h5_file, event_idx, padding_mask):
        source_rows = {
            source_name: self._read_source_rows(h5_file, source_name, event_idx)
            for source_name in self.reco_builder.source_ranges
        }
        return self._build_reco_raw_labels_from_loaded(
            h5_file, event_idx, source_rows.get("JET", []), source_rows.get("LEPTON", []),
            source_rows.get("MET", []), padding_mask, source_rows=source_rows,
        )

    def _build_reco_raw_labels_from_loaded(self, h5_file, event_idx, jets, leptons, met, padding_mask, source_rows=None):
        source_rows = dict(source_rows or {"JET": jets, "LEPTON": leptons, "MET": met})
        for source_name in self.reco_builder.source_ranges:
            if source_name not in source_rows:
                source_rows[source_name] = self._read_source_rows(h5_file, source_name, event_idx)
        label_rows = {}
        for source_name in self.reco_builder.source_ranges:
            label_path = f"{self.reco_builder.label_group}/{source_name}"
            if label_path in h5_file:
                width = len(self.reco_builder.source_ranges[source_name])
                label_rows[source_name] = h5_file[label_path][event_idx, :width]
        return self._build_reco_raw_labels_from_rows(source_rows, label_rows, padding_mask, event_idx)

    def _build_reco_raw_labels_from_rows(self, source_rows, label_rows, padding_mask, event_idx):
        raw_labels = np.full(self.total_tokens, self.reco_builder.ignore_index, dtype=np.float32)
        target_sources = {
            source
            for target in self.reco_builder.targets
            for source, _ in target.members
        }

        for source_name, token_indices in self.reco_builder.source_ranges.items():
            rows = source_rows[source_name]
            width = len(token_indices)
            if width == 0:
                continue

            label_path = f"{self.reco_builder.label_group}/{source_name}"
            if source_name not in label_rows:
                if source_name in target_sources:
                    raise KeyError(f"Reconstruction target needs {label_path}, but it is absent from {self.h5_path}.")
                continue

            labels = np.asarray(label_rows[source_name], dtype=np.float32).reshape(-1)
            labels = labels[:width]
            valid_objects = _valid_object_mask(rows)[:width] if len(rows) else np.zeros(width, dtype=bool)

            for offset, token_idx in enumerate(token_indices):
                if offset >= labels.shape[0]:
                    continue
                if bool(padding_mask[token_idx]) or not bool(valid_objects[offset]):
                    continue
                value = float(labels[offset])
                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite reconstruction label in file {self.h5_path}, dataset {label_path}, "
                        f"event {event_idx}, object {offset}, token {token_idx}: {value!r}."
                    )
                raw_labels[token_idx] = value

        return raw_labels

    def _read_source_rows(self, h5_file, source_name, event_idx):
        width = self.widths.get(source_name, 0)
        if width == 0:
            return []
        path = f"/INPUTS/{source_name}"
        if path not in h5_file:
            return []
        ds = h5_file[path]
        if len(ds.shape) > 1:
            return ds[event_idx, :width]
        return np.asarray([ds[event_idx]], dtype=ds.dtype)

    def _fill_object_tokens(self, source_name, rows, cursor, tokens, type_ids, padding_mask, *, default_id=0.0, accept_all=False):
        width = self.widths.get(source_name, 0)
        if width == 0:
            return cursor
        plan = self.schema.sources.get(source_name)
        if plan is None:
            raise KeyError(
                f"Unknown transformer token source {source_name!r}; configured sources are "
                f"{list(self.schema.sources)}."
            )
        source_type_id = self._source_type_id(source_name)

        for row_offset, row in enumerate(rows):
            token_idx = cursor + row_offset
            if accept_all or _is_valid_object(row):
                self._encode_object_row(plan, row, tokens[token_idx], default_id=default_id)
                type_ids[token_idx] = source_type_id
                padding_mask[token_idx] = False

        return cursor + width

    def _source_type_id(self, source_name):
        try:
            return int(self.schema.source_type_ids[source_name])
        except KeyError as exc:
            raise KeyError(
                f"Unknown transformer token source {source_name!r}; configured source type IDs are "
                f"{dict(self.schema.source_type_ids)}."
            ) from exc

    def _encode_object_row(self, plan, row, token, *, default_id=0.0):
        for feature in plan.features:
            value = _object_value(row, feature.variable, default_id=default_id, is_valid=True)
            self._write_scaled_value(token, feature, value, group=plan.name)

    def _fill_global_token(self, h5_file, event_idx, jets, leptons, token):
        global_ds = h5_file.get("/INPUTS/GLOBAL")
        row = global_ds[event_idx] if global_ds is not None else None
        if row is not None and getattr(row, "shape", ()):
            row = row.reshape(-1)[0]
        self._fill_global_token_from_row(row, jets, leptons, token)

    def _fill_global_token_from_row(self, global_row, jets, leptons, token):
        plan = self.schema.sources["GLOBAL"]
        for feature in plan.features:
            value = self._global_feature_value_from_row(global_row, feature.variable, jets, leptons)
            self._write_scaled_value(token, feature, value, group="GLOBAL")

    def _write_scaled_value(self, token, feature, value, *, group):
        scaler = feature.scaler
        value = float(value) if np.isfinite(value) else 0.0

        if scaler in {"none", "passthrough"}:
            token[feature.indices[0]] = value
        elif scaler == "log":
            token[feature.indices[0]] = _safe_log(value)
        elif scaler == "sincos":
            token[feature.indices] = (np.sin(value), np.cos(value))
        elif scaler == "pid":
            token[feature.indices] = _pid_one_hot(value)
        elif scaler == "standard":
            state = self.scaler_state.get(group, {}).get(feature.variable, {}).get("standard")
            token[feature.indices[0]] = value if state is None else (value - state["mean"]) / state["scale"]
        elif scaler == "minmax":
            state = self.scaler_state.get(group, {}).get(feature.variable, {}).get("minmax")
            token[feature.indices[0]] = value if state is None else (value - state["min"]) / state["range"]
        elif feature.expression_divisor is not None:
            token[feature.indices[0]] = (value - float(feature.expression_offset or 0.0)) / feature.expression_divisor
        else:
            token[feature.indices[0]] = np.asarray(
                feature.transformer.transform(np.asarray([[value]], dtype=np.float32))
            ).reshape(-1)[0]

    def _global_feature_value(self, h5_file, name, event_idx, jets, leptons):
        global_ds = h5_file.get("/INPUTS/GLOBAL")
        row = global_ds[event_idx] if global_ds is not None else None
        if row is not None and getattr(row, "shape", ()):
            row = row.reshape(-1)[0]
        return self._global_feature_value_from_row(row, name, jets, leptons)

    def _global_feature_value_from_row(self, row, name, jets, leptons):
        if row is not None and row.dtype.names is not None:
            value = _field(row, name, default=np.nan)
            if np.isfinite(value):
                return value

        jet_valid = _valid_object_mask(jets)
        lepton_valid = _valid_object_mask(leptons)

        if name == "njet":
            return float(jet_valid.sum())

        if name.startswith("nBJets_"):
            threshold = self.btag_working_point_thresholds.get(name)
            if threshold is None:
                raise KeyError(
                    f"Configured global feature {name!r} is absent from /INPUTS/GLOBAL in "
                    f"{self.h5_path}, and dataset.btag_working_point_thresholds.{name} is not set."
                )
            btag = jets["btag"]
            return float((jet_valid & (np.asarray(btag) > threshold)).sum())

        if name == "nBjets":
            raise KeyError(
                f"Configured generic global feature 'nBjets' is absent from /INPUTS/GLOBAL in {self.h5_path}; "
                "generic multiplicity is never derived from a working-point threshold."
            )

        if name == "nelectron":
            ids = leptons["id"] if leptons.dtype.names and "id" in leptons.dtype.names else np.zeros(len(leptons), dtype=np.float32)
            return float((lepton_valid & (np.rint(np.nan_to_num(ids, nan=0.0)).astype(np.int32) == 2)).sum())

        if name == "nmuon":
            ids = leptons["id"] if leptons.dtype.names and "id" in leptons.dtype.names else np.zeros(len(leptons), dtype=np.float32)
            return float((lepton_valid & (np.rint(np.nan_to_num(ids, nan=0.0)).astype(np.int32) == 3)).sum())

        raise KeyError(
            f"Configured global feature {name!r} has no direct or explicit derived implementation "
            f"for {self.h5_path}."
        )

    def _read_label(self, h5_file, event_idx):
        if self.label_override is not None:
            return self.label_override
        if not self.classification_enabled:
            return 0.0
        value = float(read_transformer_classification_labels(h5_file, event_idx)[0])
        if not np.isfinite(value):
            raise ValueError(
                f"Non-finite classification label in {self.h5_path}:{self.label_path}, event {event_idx}: {value!r}."
            )
        return value


class CachedTransformerDataset(Dataset):
    def __init__(self, cache_payload, positions, reco_builder=None):
        self.cache_payload = cache_payload
        self.positions = np.asarray(positions, dtype=np.int64)
        self.reco_builder = reco_builder
        self.reco_source_ids = None
        if reco_builder is not None:
            total_tokens = int(cache_payload["padding_mask"].shape[1])
            source_ids = np.zeros(total_tokens, dtype=np.int8)
            for source, indices in reco_builder.source_ranges.items():
                if source in reco_builder.schema.source_type_ids:
                    source_ids[list(indices)] = reco_builder.schema.source_type_ids[source]
            self.reco_source_ids = torch.from_numpy(source_ids)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, item):
        pos = int(self.positions[item])
        result = {
            "tokens": self.cache_payload["tokens"][pos],
            "type_ids": self.cache_payload["type_ids"][pos],
            "padding_mask": self.cache_payload["padding_mask"][pos],
            "labels": self.cache_payload["labels"][pos],
            "event_index": self.cache_payload.get("event_id", self.cache_payload["indices"])[pos],
            "physics_vectors": self.cache_payload["physics_vectors"][pos],
        }
        if self.reco_builder is not None:
            raw_labels = self.cache_payload["reco_raw_labels"][pos]
            padding_mask = self.cache_payload["padding_mask"][pos]
            result["reco_raw_labels"] = raw_labels
            result["reco_source_ids"] = self.reco_source_ids.clone()
            result["reco_targets"] = self.reco_builder.build_event(
                raw_labels.detach().cpu().numpy(),
                padding_mask.detach().cpu().numpy(),
                event_index=int(result["event_index"]),
            )
        return result

    def __getitems__(self, items):
        """PyTorch batched fetch hook; all expensive preprocessing is already cached."""
        return [self[item] for item in items]


def _process_loaded_cache_chunk(dataset, output_positions, event_indices, source_arrays, label_values, reco_label_arrays):
    """Pure in-memory preprocessing; worker threads never touch a live H5 handle."""
    n_chunk = len(event_indices)
    tokens = np.zeros((n_chunk, dataset.total_tokens, dataset.schema.dim), dtype=np.float32)
    type_ids = np.zeros((n_chunk, dataset.total_tokens), dtype=np.int8)
    padding_masks = np.ones((n_chunk, dataset.total_tokens), dtype=np.bool_)
    labels = np.asarray(label_values, dtype=np.float32).reshape(-1).copy()
    physics_vectors = np.zeros((n_chunk, dataset.total_tokens, 4), dtype=np.float32)
    reco_raw_labels = (
        np.full((n_chunk, dataset.total_tokens), dataset.reco_builder.ignore_index, dtype=np.float32)
        if dataset.reconstruction_enabled else None
    )

    cursor = 0
    for source_name in _token_source_names(dataset.schema):
        plan = dataset.schema.sources[source_name]
        if source_name == "GLOBAL":
            continue

        width = int(dataset.widths[source_name])
        rows = np.asarray(source_arrays[source_name])[:, :width]
        valid = np.ones((n_chunk, width), dtype=bool) if source_name == "MET" else (
            np.isfinite(_fields(rows, "pt", default=np.nan))
            & (_fields(rows, "pt", default=np.nan) > 0)
        )
        slots = slice(cursor, cursor + width)
        type_ids[:, slots] = np.where(valid, dataset._source_type_id(source_name), 0)
        padding_masks[:, slots] = ~valid
        default_id = 1.0 if source_name == "JET" else 4.0 if source_name == "MET" else 0.0
        for feature in plan.features:
            if feature.variable == "id" and rows.dtype.names and "id" not in rows.dtype.names:
                values = np.full((n_chunk, width), default_id, dtype=np.float32)
                if "is_el" in rows.dtype.names:
                    values[np.asarray(rows["is_el"]) == 1] = 2
                if "is_mu" in rows.dtype.names:
                    values[np.asarray(rows["is_mu"]) == 1] = 3
            else:
                values = np.asarray(_fields(rows, feature.variable), dtype=np.float32)
            values = np.where(valid, values, 0.0)
            _write_vectorized_feature(tokens[:, slots], feature, values, dataset, source_name)
        tokens[:, slots] *= valid[..., None]

        kinematics = plan.kinematics
        required = [str(value) for key, value in kinematics.items() if key != "representation"]
        missing = sorted(set(required) - set(rows.dtype.names or ()))
        if missing:
            raise KeyError(f"{dataset.h5_path}:/INPUTS/{source_name} lacks kinematic fields {missing}")
        # Match the scalar reference path: HDF5 float32 values are promoted to
        # Python/NumPy float64 for the trigonometric arithmetic, then stored in
        # the final float32 cache tensor.  Keeping these intermediates in
        # float64 avoids a representation-dependent p4 rounding difference
        # between raw and vectorised loading.
        pt = np.asarray(rows[str(kinematics["pt"])], dtype=np.float64)
        if "phi" in kinematics:
            phi = np.asarray(rows[str(kinematics["phi"])], dtype=np.float64)
        else:
            phi = np.arctan2(
                np.asarray(rows[str(kinematics["sin_phi"])], dtype=np.float64),
                np.asarray(rows[str(kinematics["cos_phi"])], dtype=np.float64),
            )
        eta = (
            np.asarray(rows[str(kinematics["eta"])], dtype=np.float64)
            if "eta" in kinematics else np.zeros_like(pt)
        )
        energy = (
            np.asarray(rows[str(kinematics["energy"])], dtype=np.float64)
            if "energy" in kinematics else pt * np.cosh(eta)
        )
        finite = valid & np.isfinite(energy) & np.isfinite(pt) & np.isfinite(eta) & np.isfinite(phi)
        physics_vectors[:, slots, 0] = np.where(finite, energy, 0)
        physics_vectors[:, slots, 1] = np.where(finite, pt * np.cos(phi), 0)
        physics_vectors[:, slots, 2] = np.where(finite, pt * np.sin(phi), 0)
        physics_vectors[:, slots, 3] = np.where(finite, pt * np.sinh(eta), 0)

        if dataset.reconstruction_enabled and source_name in reco_label_arrays:
            raw = np.asarray(reco_label_arrays[source_name], dtype=np.float32)[:, :width]
            if not np.isfinite(raw[valid]).all():
                bad = np.argwhere(valid & ~np.isfinite(raw))[0]
                raise ValueError(
                    f"Non-finite reconstruction label in {dataset.h5_path}, source {source_name}, "
                    f"event {int(event_indices[bad[0]])}, object {int(bad[1])}"
                )
            reco_raw_labels[:, slots] = np.where(valid, raw, dataset.reco_builder.ignore_index)
        cursor += width
    if dataset.schema.has_global_token:
        plan = dataset.schema.sources["GLOBAL"]
        rows = np.asarray(source_arrays["GLOBAL"]).reshape(n_chunk)
        for feature in plan.features:
            values = np.asarray(_fields(rows, feature.variable), dtype=np.float32)
            _write_vectorized_feature(tokens[:, cursor], feature, values, dataset, "GLOBAL")
        type_ids[:, cursor] = dataset._source_type_id("GLOBAL")
        padding_masks[:, cursor] = False
    return (
        output_positions, tokens, type_ids, padding_masks, labels,
        physics_vectors, reco_raw_labels,
    )


def _write_vectorized_feature(tokens, feature, values, dataset, group):
    """Write one feature for a whole HDF5 chunk without per-event Python calls."""
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = feature.scaler
    if scaler in {"none", "passthrough"}:
        tokens[..., feature.indices[0]] = values
    elif scaler == "log":
        tokens[..., feature.indices[0]] = np.log(np.clip(values, 0.0, None) + LOG_OFFSET)
    elif scaler == "sincos":
        tokens[..., feature.indices[0]] = np.sin(values)
        tokens[..., feature.indices[1]] = np.cos(values)
    elif scaler == "pid":
        for index, raw_id in enumerate(PARTICLE_IDS.values()):
            tokens[..., feature.indices[index]] = np.rint(values) == raw_id
    elif scaler in {"standard", "minmax"}:
        state = dataset.scaler_state.get(group, {}).get(feature.variable, {}).get(scaler)
        if state is None:
            tokens[..., feature.indices[0]] = values
        elif scaler == "standard":
            tokens[..., feature.indices[0]] = (values - state["mean"]) / state["scale"]
        else:
            tokens[..., feature.indices[0]] = (values - state["min"]) / state["range"]
    elif feature.expression_divisor is not None:
        tokens[..., feature.indices[0]] = (
            values - float(feature.expression_offset or 0.0)
        ) / feature.expression_divisor
    else:
        raise ValueError(f"Unsupported vectorized cache transform {scaler!r}")


def _collate_transformer(batch):
    collated = {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        # Cache type IDs use compact integer storage; embedding layers require
        # a supported index dtype at the model boundary.
        "type_ids": torch.stack([item["type_ids"] for item in batch]).long(),
        "padding_mask": torch.stack([item["padding_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "event_index": torch.stack([item["event_index"] for item in batch]),
    }
    if "reco_raw_labels" in batch[0]:
        collated["reco_raw_labels"] = torch.stack([item["reco_raw_labels"] for item in batch])
        collated["reco_source_ids"] = torch.stack([item["reco_source_ids"] for item in batch])
    if "physics_vectors" in batch[0]:
        collated["physics_vectors"] = torch.stack([item["physics_vectors"] for item in batch])
    if "reco_targets" in batch[0]:
        pool_names = list(batch[0]["reco_targets"].keys())
        pools = {}
        for name in pool_names:
            candidate_size = int(batch[0]["reco_targets"][name]["candidate_size"])
            max_candidates = max(item["reco_targets"][name]["candidate_indices"].shape[0] for item in batch)
            candidate_indices = torch.zeros((len(batch), max_candidates, candidate_size), dtype=torch.long)
            candidate_source_indices = torch.zeros_like(candidate_indices)
            candidate_mask = torch.zeros((len(batch), max_candidates), dtype=torch.bool)
            truth_class = torch.zeros((len(batch), max_candidates), dtype=torch.long)
            event_active = torch.zeros(len(batch), dtype=torch.bool)

            for batch_idx, item in enumerate(batch):
                entry = item["reco_targets"][name]
                n_candidates = int(entry["candidate_indices"].shape[0])
                if n_candidates > 0:
                    candidate_indices[batch_idx, :n_candidates].copy_(torch.from_numpy(entry["candidate_indices"]))
                    candidate_source_indices[batch_idx, :n_candidates].copy_(torch.from_numpy(entry["candidate_source_indices"]))
                    candidate_mask[batch_idx, :n_candidates] = True
                    truth_class[batch_idx, :n_candidates].copy_(torch.from_numpy(entry["truth_class"]))
                event_active[batch_idx] = bool(entry["event_active"])

            entry0 = batch[0]["reco_targets"][name]
            pools[name] = {
                "candidate_indices": candidate_indices,
                "candidate_source_indices": candidate_source_indices,
                "candidate_mask": candidate_mask,
                "truth_class": truth_class,
                "event_active": event_active,
                "source_signature": entry0["source_signature"],
                "class_names": entry0["class_names"],
                "class_to_index": entry0["class_to_index"],
                "candidate_size": candidate_size,
                "member_sources": entry0["member_sources"],
            }
        collated["reco"] = {"pools": pools}
    return collated


class TransformerDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        logger.info("Initializing TransformerDataModule.")

        validate_transformer_task_config(config)

        self.full_config = config
        self.config = config.dataset
        self.classification_enabled = bool(
            (config.get("tasks", {}).get("classification", {}) or {}).get("enabled", True)
        )
        self.h5_path = self.config.get("input_h5_path", None)
        self.random_seed = config.general.seed
        self.schema = build_token_schema(self.config)
        self.configured_widths = None
        self.reconstruction_cfg = _reconstruction_cfg(config)
        self.reconstruction_enabled = _reconstruction_enabled(config)
        self.reco_builder = None
        self.is_setup = False
        self.cache_payload = None
        self.scaler_state = None
        self.split_metadata = None
        self.split_indices = None

        # Fraction of data to use (1.0 = all). Can be overridden externally.
        self.data_percentage = 1.0

    def setup(self, stage=None):
        if self.is_setup:
            return

        self.is_setup = True
        setup_start = time.perf_counter()
        logger.info("Setting up TransformerDataModule.")

        classification_enabled = self.classification_enabled
        validate_transformer_h5_inputs(
            self.h5_path,
            self.schema,
            self.config,
            reconstruction_cfg=self.reconstruction_cfg if self.reconstruction_enabled else None,
            classification_enabled=classification_enabled,
            config_name=str(self.full_config.get("outputs", {}).get("config_name", "transformer")),
        )
        self._validate_observable_widths()
        self._log_btag_feature_sources()

        with h5py.File(self.h5_path, "r") as h5_file:
            n_events = h5_file["/INPUTS/JET"].shape[0]
            self.configured_widths = _configured_widths_from_h5(h5_file, self.schema, self.config)
            if self.reconstruction_enabled:
                label_group = _normalise_h5_group(self.reconstruction_cfg.get("label_source", "LABELS"))
                if label_group not in h5_file:
                    raise KeyError(
                        f"Reconstruction label group {label_group} is absent from {self.h5_path}."
                    )
                requested_sources = {
                    source for target in parse_reconstruction_targets(self.reconstruction_cfg, _token_source_names(self.schema))
                    for source, _ in target.members
                }
                for source in sorted(requested_sources):
                    label_path = f"{label_group}/{source}"
                    if label_path not in h5_file:
                        raise KeyError(
                            f"Reconstruction target source {source} requires {label_path}, "
                            f"but it is absent from {self.h5_path}."
                        )
                    dataset = h5_file[label_path]
                    required_width = int(self.configured_widths.get(source, 0))
                    if dataset.ndim < 2:
                        raise ValueError(
                            f"Reconstruction label dataset {label_path} in {self.h5_path} must be at least "
                            f"two-dimensional [event, object], found shape {dataset.shape}."
                        )
                    if dataset.shape[0] != n_events or dataset.shape[1] < required_width:
                        raise ValueError(
                            f"Reconstruction label dataset {label_path} in {self.h5_path} has shape "
                            f"{dataset.shape}; expected {n_events} events and at least {required_width} objects."
                        )
                    audit_events = min(n_events, int(self.config.get("label_audit_events", 10000)))
                    audit = np.asarray(dataset[:audit_events, :required_width], dtype=np.float64)
                    input_rows = np.asarray(h5_file[f"/INPUTS/{source}"][:audit_events, :required_width])
                    valid_objects = np.asarray(
                        [_valid_object_mask(rows) for rows in input_rows], dtype=bool
                    )
                    invalid_valid_label = valid_objects & ~np.isfinite(audit)
                    if invalid_valid_label.any():
                        bad_event, bad_object = np.argwhere(invalid_valid_label)[0]
                        raise ValueError(
                            f"Non-finite reconstruction label in {self.h5_path}:{label_path}, "
                            f"event {int(bad_event)}, object {int(bad_object)}: {audit[bad_event, bad_object]!r}."
                        )
                    finite = audit[
                        valid_objects & np.isfinite(audit)
                        & (audit != float(self.reconstruction_cfg.get("ignore_index", RECO_IGNORE_INDEX)))
                    ]
                    if finite.size and not np.equal(finite, np.rint(finite)).all():
                        bad = float(finite[np.not_equal(finite, np.rint(finite))][0])
                        raise ValueError(f"Non-integer raw label {bad!r} in {self.h5_path}:{label_path}.")
                    observed = sorted({int(value) for value in finite.tolist()})
                    required = sorted({label for target in parse_reconstruction_targets(self.reconstruction_cfg, _token_source_names(self.schema))
                                       for member_source, label in target.members if member_source == source})
                    logger.info("Reconstruction-label audit %s: events=%d observed=%s required=%s.", label_path, audit_events, observed, required)
                    missing = sorted(set(required) - set(observed))
                    if missing:
                        raise ValueError(f"Reconstruction targets require raw labels {missing} for source {source}, but {self.h5_path}:{label_path} observed {observed} in the first {audit_events} events.")
        if self.reconstruction_enabled:
            self.reco_builder = ReconstructionCandidateBuilder(self.reconstruction_cfg, self.schema, self.configured_widths)

        if classification_enabled:
            all_labels = self._read_all_labels(self.h5_path, n_events)
        else:
            # Reconstruction-only source files need not synthesize an unrelated
            # event-classification truth dataset.  Canonical /SPLIT or the
            # configured stable split controls population membership below.
            all_labels = np.zeros(n_events, dtype=np.int8)
        split_cfg = self.config.get("split", {}) or {}
        arrays, self.split_metadata = load_or_create_split(
            split_cfg.get("indices_path", None), all_labels,
            source_path=self.h5_path, seed=self.random_seed,
            limit_events=self.config.get("limit_events", None),
            test_size=float(self.config.get("test_size", 0.2)),
            require_existing=bool(split_cfg.get("require_existing", False)),
        )
        with h5py.File(self.h5_path, "r") as identity_file:
            if "EVENT_ID" in identity_file:
                event_dataset = identity_file["EVENT_ID"]
                if event_dataset.ndim != 1 or not np.issubdtype(
                    event_dataset.dtype, np.integer
                ) or len(event_dataset) != n_events:
                    raise TypeError(
                        f"Stable EVENT_ID must be integer [{n_events}]: "
                        f"{self.h5_path}:EVENT_ID shape={event_dataset.shape}, "
                        f"dtype={event_dataset.dtype}"
                    )
                all_event_ids = np.asarray(event_dataset[:], dtype=np.int64)
                if len(np.unique(all_event_ids)) != n_events:
                    raise ValueError(
                        f"Stable EVENT_ID contains duplicates: {self.h5_path}:EVENT_ID"
                    )
                event_id_hashes = {}
                event_id_counts = {}
                for split_name in ("train", "validation", "test"):
                    rows = np.sort(np.asarray(arrays[split_name], dtype=np.int64))
                    if len(rows):
                        ids = all_event_ids[rows]
                        event_id_hashes[split_name] = index_hash(np.sort(ids))
                    else:
                        event_id_hashes[split_name] = index_hash(np.empty(0, dtype=np.int64))
                    event_id_counts[split_name] = len(rows)
                self.split_metadata["event_id_hashes"] = event_id_hashes
                self.split_metadata["event_id_counts"] = event_id_counts
                owner = event_dataset.file
                source_contract = {}
                for key in (
                    "source_manifest_sha256",
                    "split_membership_sha256",
                    "split_version",
                    "adapter_version",
                    "source_path",
                    "max_jets",
                ):
                    value = owner.attrs.get(key)
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    if value is not None:
                        source_contract[key] = str(value)
                if source_contract:
                    self.split_metadata["source_contract"] = source_contract
        self.split_indices = arrays
        selected_indices = arrays["selected"]
        train_idx = arrays["train"]
        val_idx = arrays["validation"]
        test_idx = arrays["test"]
        selected_labels = all_labels[selected_indices]
        selected_position = {int(event): position for position, event in enumerate(selected_indices)}
        train_pos = np.asarray([selected_position[int(event)] for event in train_idx], dtype=np.int64)
        val_pos = np.asarray([selected_position[int(event)] for event in val_idx], dtype=np.int64)
        test_pos = np.asarray([selected_position[int(event)] for event in test_idx], dtype=np.int64)

        if self.schema.has_observables and self._needs_scaler_state():
            self.scaler_state = self._fit_scaler_state(train_idx, h5_path=self.h5_path)
        else:
            self.scaler_state = {}
            logger.info("No stateful transformer scalers requested; skipping scaler fit.")
        self._log_frozen_affine_transforms()

        logger.info(
            "Transformer H5 source: %s; selected %d/%d events with label counts %s.",
            self.h5_path,
            len(selected_indices),
            n_events,
            _label_counts(selected_labels),
        )
        logger.info(
            "Transformer split sizes: train=%d %s, val=%d %s, test=%d %s.",
            len(train_idx),
            _label_counts(all_labels[train_idx]),
            len(val_idx),
            _label_counts(all_labels[val_idx]),
            len(test_idx),
            _label_counts(all_labels[test_idx]),
        )
        logger.info(
            "Transformer batch tensors: tokens=(batch, %d, %d); GLOBAL token=%s.",
            self._configured_total_tokens(),
            self.schema.dim,
            self.schema.has_global_token,
        )
        logger.info("Transformer token schema=%s.", list(self.schema.names))

        if self._cache_enabled():
            self.cache_payload = self._load_or_build_cache(
                self.h5_path,
                selected_indices,
                cache_context={
                    "limit_events": self.config.get("limit_events", None),
                    "label_override": None,
                    "selected_indices_hash": _indices_hash(selected_indices),
                    "data_percentage": self.data_percentage,
                },
            )
            self.train_dataset = CachedTransformerDataset(self.cache_payload, train_pos, reco_builder=self.reco_builder)
            self.val_dataset = CachedTransformerDataset(self.cache_payload, val_pos, reco_builder=self.reco_builder)
            self.test_dataset = CachedTransformerDataset(self.cache_payload, test_pos, reco_builder=self.reco_builder)
        else:
            self.train_dataset = self._make_h5_dataset(train_idx)
            self.val_dataset = self._make_h5_dataset(val_idx)
            self.test_dataset = self._make_h5_dataset(test_idx)

        logger.info("TransformerDataModule setup completed in %.2f s.", time.perf_counter() - setup_start)

    def _make_h5_dataset(self, indices, *, h5_path=None, label_override=None):
        return TransformerH5Dataset(
            h5_path or self.h5_path,
            self.config,
            indices,
            label_override=label_override,
            scaler_state=self.scaler_state,
            schema=self.schema,
            configured_widths=self.configured_widths,
            reconstruction_cfg=self.reconstruction_cfg if self.reconstruction_enabled else None,
            classification_enabled=self.classification_enabled,
        )

    def _needs_scaler_state(self) -> bool:
        """Return True only when the active schema uses fitted stateful scalers.

        Fixed transforms such as Log, none, PID, SinCos, and x/N do not require
        scanning the training set. This avoids an expensive full-H5 pass before
        cache building.
        """
        return any(
            feature.scaler in STATEFUL_SCALERS
            for plan in self.schema.sources.values()
            for feature in plan.features
        )

    def _log_frozen_affine_transforms(self):
        affine = [
            (
                source_name,
                feature.variable,
                float(feature.expression_offset or 0.0),
                float(feature.expression_divisor),
            )
            for source_name, plan in self.schema.sources.items()
            for feature in plan.features
            if feature.expression_divisor is not None and float(feature.expression_offset or 0.0) != 0.0
        ]
        if not affine:
            return
        logger.info(
            "Frozen affine transformer scalers: %s.",
            [
                f"{source}:{variable}=(x-{offset:.12g})/{divisor:.12g}"
                for source, variable, offset, divisor in affine
            ],
        )


    def _validate_observable_widths(self):
        jet_plan = self.schema.sources.get("JET")
        if jet_plan is not None and jet_plan.num_particles != int(self.config.max_jets):
            raise ValueError(
                "Transformer dataset.max_jets and INPUTS:JET num_particles must match. "
                f"Got max_jets={self.config.max_jets}, JET num_particles={jet_plan.num_particles}."
            )
        logger.info("Validated all configured transformer H5 sources, features, dimensions, and labels.")

    def _log_btag_feature_sources(self):
        global_plan = self.schema.sources.get("GLOBAL")
        if global_plan is None:
            return
        thresholds = _plain_config_value(
            self.config.get("btag_working_point_thresholds", {}) or {}
        )
        with h5py.File(self.h5_path, "r") as h5_file:
            fields = set(h5_file["/INPUTS/GLOBAL"].dtype.names or ())
        for name in global_plan.variables:
            if name.startswith("nBJets_"):
                if name in fields:
                    logger.info("%s: read directly from /INPUTS/GLOBAL/%s", name, name)
                else:
                    logger.info("%s: derived from /INPUTS/JET/btag > %.12g", name, float(thresholds[name]))

    def _configured_total_tokens(self):
        if self.configured_widths is None:
            with h5py.File(self.h5_path, "r") as h5_file:
                self.configured_widths = _configured_widths_from_h5(h5_file, self.schema, self.config)
        return (
            sum(self.configured_widths.get(source_name, 0) for source_name in _token_source_names(self.schema))
            + int(self.schema.has_global_token)
        )

    def _read_all_labels(self, h5_path, n_events):
        with h5py.File(str(h5_path), "r") as h5_file:
            label_path = resolve_transformer_label_path(h5_file)
            try:
                labels = read_transformer_classification_labels(h5_file, slice(0, n_events))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid classification labels in {h5_path}:{label_path}: {exc}"
                ) from exc
        if not np.isfinite(labels).all():
            bad_index = int(np.flatnonzero(~np.isfinite(labels))[0])
            raise ValueError(
                f"Non-finite classification label in {h5_path}:{label_path}, event {bad_index}: {labels[bad_index]!r}."
            )
        if not np.equal(labels, np.rint(labels)).all():
            bad_index = int(np.flatnonzero(labels != np.rint(labels))[0])
            raise ValueError(
                f"Non-integer classification label in {h5_path}:{label_path}, event {bad_index}: {labels[bad_index]!r}."
            )
        return labels.astype(int, copy=False)

    def _fit_scaler_state(self, train_indices, h5_path=None):
        start = time.perf_counter()
        values_by_group = {
            source_name: {
                feature.variable: []
                for feature in plan.features
                if feature.scaler in STATEFUL_SCALERS
            }
            for source_name, plan in self.schema.sources.items()
        }

        source_h5_path = str(h5_path if h5_path is not None else self.h5_path)

        with h5py.File(source_h5_path, "r") as h5_file:
            widths = self.configured_widths or _configured_widths_from_h5(h5_file, self.schema, self.config)
            reader = TransformerH5Dataset(
                source_h5_path,
                self.config,
                [],
                scaler_state={},
                schema=self.schema,
                configured_widths=widths,
                classification_enabled=self.classification_enabled,
            )
            sorted_indices = np.sort(np.asarray(train_indices, dtype=np.int64))

            for i, event_idx in enumerate(sorted_indices):
                if i > 0 and i % 50000 == 0:
                    logger.info("Transformer scaler fit progress: %d/%d training events.", i, len(sorted_indices))

                source_rows = {
                    source_name: reader._read_source_rows(h5_file, source_name, event_idx)
                    for source_name in _token_source_names(self.schema) if source_name != "GLOBAL"
                }
                jets = source_rows.get("JET", [])
                leptons = source_rows.get("LEPTON", [])
                for source_name, rows in source_rows.items():
                    self._append_object_state_values(
                        values_by_group, source_name, rows,
                        default_id=1.0 if source_name == "JET" else (4.0 if source_name == "MET" else 0.0),
                        accept_all=source_name == "MET",
                    )

                if "GLOBAL" in self.schema.sources:
                    for feature in self.schema.sources["GLOBAL"].features:
                        if feature.scaler in STATEFUL_SCALERS:
                            values_by_group["GLOBAL"][feature.variable].append(
                                reader._global_feature_value(h5_file, feature.variable, event_idx, jets, leptons)
                            )
            reader.close()

        state = {
            source_name: self._stats_from_values(values_by_group[source_name], plan)
            for source_name, plan in self.schema.sources.items()
        }
        logger.info("Fitted transformer observable scaler state in %.2f s.", time.perf_counter() - start)
        return state

    def _append_object_state_values(self, values_by_group, source_name, rows, *, default_id=0.0, accept_all=False):
        plan = self.schema.sources.get(source_name)
        if plan is None or source_name not in values_by_group or len(rows) == 0:
            return
        stateful = [feature for feature in plan.features if feature.scaler in STATEFUL_SCALERS]
        if not stateful:
            return
        mask = np.ones(len(rows), dtype=bool) if accept_all else _valid_object_mask(rows)
        if not mask.any():
            return
        for feature in stateful:
            values = _object_values(rows, feature.variable, default_id=default_id)[mask]
            finite = values[np.isfinite(values)]
            if finite.size:
                values_by_group[source_name][feature.variable].extend(finite.astype(np.float64).tolist())

    @staticmethod
    def _stats_from_values(values_by_variable, plan):
        state = {}
        scaler_by_variable = {
            feature.variable: feature.scaler
            for feature in plan.features
            if feature.scaler in STATEFUL_SCALERS
        }
        for variable, values in values_by_variable.items():
            scaler = scaler_by_variable[variable]
            arr = np.asarray(values, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                logger.warning("No finite values found while fitting %s scaler for %s; using identity stats.", scaler, variable)
                arr = np.asarray([0.0], dtype=np.float64)

            if scaler == "standard":
                scale = float(arr.std())
                state.setdefault(variable, {})[scaler] = {
                    "mean": float(arr.mean()),
                    "scale": scale if scale > 0 else 1.0,
                }
            elif scaler == "minmax":
                min_value = float(arr.min())
                max_value = float(arr.max())
                value_range = max_value - min_value
                state.setdefault(variable, {})[scaler] = {
                    "min": min_value,
                    "max": max_value,
                    "range": value_range if value_range > 0 else 1.0,
                }
        return state

    def _cache_enabled(self):
        enabled = bool(self.config.get("cache", {}).get("enabled", False))
        logger.info("Transformer tensor caching %s.", "enabled" if enabled else "disabled")
        return enabled

    def _cache_cfg(self):
        return self.config.get("cache", {}) or {}

    def _cache_chunk_size(self):
        return max(1, int(self._cache_cfg().get("build_chunk_size", 4096)))

    def _cache_num_workers(self):
        return max(1, int(self._cache_cfg().get("build_workers", 1)))

    def _cache_dataset_id(self, h5_path=None):
        dataset_id = self._cache_cfg().get("dataset_id", None)
        if dataset_id is None or str(dataset_id).strip() == "":
            dataset_id = Path(str(h5_path if h5_path is not None else self.h5_path)).stem
        if getattr(self, "data_percentage", 1.0) < 1.0:
            pct_tag = int(round(self.data_percentage * 100))
            dataset_id = f"{dataset_id}_optim_{pct_tag}"
        return _safe_cache_name(dataset_id)

    def _cache_source_h5_path(self):
        source_path = self._cache_cfg().get("source_h5_path", None)
        return None if source_path in {None, ""} else str(source_path)

    def _cache_source_file_name(self, h5_path=None):
        source_path = self._cache_source_h5_path()
        return Path(source_path if source_path else str(h5_path if h5_path is not None else self.h5_path)).name

    def _cache_dir(self, h5_path=None):
        cache_dir = self._cache_cfg().get("cache_dir", None)
        if cache_dir is None or str(cache_dir).strip() == "":
            cache_dir = Path(str(h5_path if h5_path is not None else self.h5_path)).parent / ".transformer_cache" / "transformer"
        cache_dir = Path(str(cache_dir)).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _h5_size_bytes(self, h5_path=None):
        try:
            return int(Path(str(h5_path if h5_path is not None else self.h5_path)).stat().st_size)
        except OSError:
            logger.warning("Could not stat H5 path for cache fingerprint: %s", h5_path if h5_path is not None else self.h5_path)
            return None

    def _dataset_event_identity(self, h5_path=None):
        path = str(h5_path if h5_path is not None else self.h5_path)
        with h5py.File(path, "r") as source:
            if "EVENT_ID" not in source:
                return None
            event_ids = source["EVENT_ID"]
            count = len(event_ids)
            sample = np.concatenate(
                [np.asarray(event_ids[: min(count, 4096)], dtype=np.int64),
                 np.asarray(event_ids[max(0, count - 4096):], dtype=np.int64)]
            )
        return hashlib.sha256(sample.tobytes()).hexdigest()

    def _n_events_for_h5(self, h5_path):
        with h5py.File(str(h5_path), "r") as h5_file:
            return int(h5_file["/INPUTS/JET"].shape[0])

    def _cache_subset_tag(self, metadata):
        if metadata.get("max_events") is not None:
            return f"max{metadata['max_events']}"
        if metadata.get("limit_events") is not None:
            return f"limit{metadata['limit_events']}"
        return "all"

    def _cache_label_tag(self, metadata):
        label_override = metadata.get("label_override", None)
        return "filelabels" if label_override is None else f"label{int(float(label_override))}"

    def _cache_payload_kind(self):
        return "reconstruction" if self.reconstruction_enabled else "classification"

    def _cache_reconstruction_metadata(self):
        if self.reconstruction_enabled:
            return _plain_config_value(self.reconstruction_cfg)
        return None

    def _cache_key_metadata(self, h5_path=None, cache_context=None):
        cache_context = cache_context or {}
        h5_path = h5_path if h5_path is not None else self.h5_path
        fingerprint = str(self._cache_cfg().get("fingerprint", "schema_size"))
        cache_reconstruction = self._cache_reconstruction_metadata()

        metadata = {
            "cache_version": CACHE_VERSION,
            "payload_kind": self._cache_payload_kind(),
            "dataset_id": self._cache_dataset_id(h5_path),
            "source_file_name": self._cache_source_file_name(h5_path),
            "source_identity": {
                "path": str(Path(str(h5_path)).expanduser().resolve()),
                "size_bytes": self._h5_size_bytes(h5_path),
                "n_events": int(self._n_events_for_h5(h5_path)),
                "event_id_fingerprint": self._dataset_event_identity(h5_path),
                "external_links": _external_link_identity(h5_path),
            },
            "fingerprint": fingerprint,
            "input_schema": _h5_schema_summary(
                h5_path,
                [
                    "/INPUTS/JET", "/INPUTS/LEPTON", "/INPUTS/MET",
                    "/INPUTS/GLOBAL", LABEL_PATH, "/LABELS/CLASS",
                ],
            ),
            "n_events": int(self._n_events_for_h5(h5_path)),
            "widths": dict(self.configured_widths or {}),
            "observables": _plain_config_value(self.config.get("observables", [])),
            "token_features": list(self.schema.names),
            "token_feature_dim": int(self.schema.dim),
            "source_type_ids": dict(self.schema.source_type_ids),
            "has_global_token": bool(self.schema.has_global_token),
            "btag_working_point_thresholds": _plain_config_value(
                self.config.get("btag_working_point_thresholds", {}) or {}
            ),
            "scaler_state": self.scaler_state,
            "limit_events": _plain_config_value(cache_context.get("limit_events", self.config.get("limit_events", None))),
            "max_events": _plain_config_value(cache_context.get("max_events", None)),
            "label_override": _plain_config_value(cache_context.get("label_override", None)),
            "selected_indices_hash": _plain_config_value(cache_context.get("selected_indices_hash", None)),
        }
        if cache_reconstruction is not None:
            metadata["reconstruction"] = cache_reconstruction
            metadata["input_schema"].update(
                _h5_schema_summary(
                    h5_path,
                    [
                        "/LABELS/JET",
                        "/LABELS/LEPTON",
                        "/LABELS/MET",
                    ],
                )
            )

        if fingerprint in {"schema_size", "size", "file_size"}:
            metadata["input_h5_size_bytes"] = self._h5_size_bytes(h5_path)

        return metadata

    def _cache_path(self, h5_path=None, cache_context=None):
        mode = str(self._cache_cfg().get("mode", "disk")).lower()
        if mode != "disk":
            raise ValueError(f"Unsupported transformer cache mode: {mode}")

        metadata = self._cache_key_metadata(h5_path=h5_path, cache_context=cache_context)
        cache_key = _stable_json_hash(metadata, length=12)
        explicit = self._cache_cfg().get("path")
        if explicit not in {None, ""}:
            cache_path = Path(str(explicit)).expanduser().resolve()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            return cache_path, metadata, cache_key
        dataset_id = _safe_cache_name(metadata["dataset_id"])
        widths = metadata["widths"]

        filename = (
            f"{dataset_id}"
            f"__jets{widths.get('JET', 0)}_lep{widths.get('LEPTON', 0)}_met{widths.get('MET', 0)}"
            f"__tok{len(metadata['token_features'])}"
            f"__{self._cache_subset_tag(metadata)}"
            f"__{self._cache_label_tag(metadata)}"
            f"__{metadata['payload_kind']}"
            f"__{cache_key}.pt"
        )
        cache_path = self._cache_dir(h5_path) / filename
        return cache_path, metadata, cache_key

    def _selected_event_count_from_context(self, metadata):
        for key in ("max_events", "limit_events"):
            value = metadata.get(key)
            if value is not None:
                return int(value)
        return metadata.get("n_events")

    def _cache_manifest_metadata(self, h5_path, cache_path, key_metadata, cache_key, payload):
        has_reco_raw_labels = "reco_raw_labels" in payload
        padding_shape = list(payload["padding_mask"].shape)
        reco_shape = list(payload["reco_raw_labels"].shape) if has_reco_raw_labels else None
        payload_kind = payload.get("metadata", {}).get("payload_kind")
        if payload_kind not in {"classification", "reconstruction"}:
            raise ValueError(f"Cache payload has invalid payload_kind={payload_kind!r}.")
        if has_reco_raw_labels != (payload_kind == "reconstruction"):
            raise ValueError(
                "Cache payload kind and reco_raw_labels presence disagree: "
                f"payload_kind={payload_kind!r}, has_reco_raw_labels={has_reco_raw_labels}."
            )
        if has_reco_raw_labels and reco_shape != padding_shape:
            raise ValueError(
                "Reconstruction cache payload has an invalid reco_raw_labels shape: "
                f"{reco_shape} != padding_mask {padding_shape}."
            )
        manifest = {
            "cache_version": CACHE_VERSION,
            "payload_kind": payload_kind,
            "cache_key": cache_key,
            "cache_path": str(cache_path),
            "payload_file_size_bytes": int(cache_path.stat().st_size),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "dataset_id": key_metadata.get("dataset_id"),
            "runtime_input_h5_path": str(h5_path),
            "source_h5_path": self._cache_source_h5_path(),
            "source_file_name": key_metadata.get("source_file_name"),
            "source_identity": key_metadata.get("source_identity"),
            "n_events": key_metadata.get("n_events"),
            "selected_events": int(payload["indices"].shape[0]),
            "indices_shape": list(payload["indices"].shape),
            "tokens_shape": list(payload["tokens"].shape),
            "type_ids_shape": list(payload["type_ids"].shape),
            "padding_mask_shape": padding_shape,
            "physics_vectors_shape": list(payload["physics_vectors"].shape),
            "physics_vector_representation": "E_px_py_pz_GeV_v1",
            "metric_convention": "+---",
            "event_id_shape": list(payload["event_id"].shape),
            "event_id_fingerprint": key_metadata.get("source_identity", {}).get("event_id_fingerprint"),
            "split_shape": list(payload["split"].shape),
            "invisible_truth_shape": (
                list(payload["invisible_truth"].shape)
                if "invisible_truth" in payload else None
            ),
            "labels_shape": list(payload["labels"].shape),
            "reco_raw_labels_shape": reco_shape,
            "has_reco_raw_labels": bool(has_reco_raw_labels),
            "widths": key_metadata.get("widths"),
            "token_features": key_metadata.get("token_features"),
            "token_feature_dim": key_metadata.get("token_feature_dim"),
            "source_type_ids": key_metadata.get("source_type_ids"),
            "has_global_token": key_metadata.get("has_global_token"),
            "observables": key_metadata.get("observables"),
            "limit_events": key_metadata.get("limit_events"),
            "max_events": key_metadata.get("max_events"),
            "label_override": key_metadata.get("label_override"),
            "selected_indices_hash": key_metadata.get("selected_indices_hash"),
            "cache_build_chunk_size": self._cache_chunk_size(),
            "fingerprint": key_metadata.get("fingerprint"),
            "input_h5_size_bytes": key_metadata.get("input_h5_size_bytes"),
            "input_schema": key_metadata.get("input_schema"),
            "btag_working_point_thresholds": key_metadata.get("btag_working_point_thresholds"),
        }
        if "reconstruction" in key_metadata:
            manifest["reconstruction"] = key_metadata.get("reconstruction")
        return manifest

    def _write_cache_manifest(self, h5_path, cache_path, metadata, cache_key, payload):
        manifest_path = cache_path.with_suffix(".json")
        manifest = self._cache_manifest_metadata(h5_path, cache_path, metadata, cache_key, payload)
        fd, temporary = tempfile.mkstemp(prefix=f".{manifest_path.name}.", dir=manifest_path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, manifest_path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        logger.info("Wrote transformer cache manifest to %s.", manifest_path)
        return manifest_path

    @staticmethod
    def _cache_manifest_matches_config(manifest_path, cache_path, metadata, cache_key):
        if not manifest_path.is_file():
            return False, "cache completion manifest is missing"
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"cache completion manifest is unreadable: {exc}"
        checks = {
            "cache_version": (manifest.get("cache_version"), CACHE_VERSION),
            "cache_key": (manifest.get("cache_key"), cache_key),
            "cache_path": (manifest.get("cache_path"), str(cache_path)),
            "payload_kind": (manifest.get("payload_kind"), metadata.get("payload_kind")),
            "dataset_id": (manifest.get("dataset_id"), metadata.get("dataset_id")),
            "n_events": (manifest.get("n_events"), metadata.get("n_events")),
            "selected_indices_hash": (
                manifest.get("selected_indices_hash"), metadata.get("selected_indices_hash")
            ),
            "input_h5_size_bytes": (
                manifest.get("input_h5_size_bytes"), metadata.get("input_h5_size_bytes")
            ),
            "payload_file_size_bytes": (
                manifest.get("payload_file_size_bytes"),
                cache_path.stat().st_size if cache_path.is_file() else None,
            ),
        }
        for name, (actual, expected) in checks.items():
            if actual != expected:
                return False, f"cache completion manifest {name} mismatch: {actual!r} != {expected!r}"
        expected_reco = metadata.get("payload_kind") == "reconstruction"
        if manifest.get("has_reco_raw_labels") is not expected_reco:
            return False, "cache completion manifest has_reco_raw_labels disagrees with payload_kind"
        if expected_reco:
            if manifest.get("reconstruction") != metadata["reconstruction"]:
                return False, "cache completion manifest reconstruction metadata mismatch"
            if manifest.get("reco_raw_labels_shape") != manifest.get("padding_mask_shape"):
                return False, "cache completion manifest reconstruction label shape is incompatible"
        elif manifest.get("reco_raw_labels_shape") is not None:
            return False, "classification cache manifest claims a reconstruction label shape"
        return True, "ok"

    @staticmethod
    def _atomic_torch_save(payload, cache_path):
        fd, temporary = tempfile.mkstemp(prefix=f".{cache_path.name}.", dir=cache_path.parent)
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, cache_path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _log_cache_dir_summary(cache_dir):
        try:
            files = [path for path in Path(cache_dir).glob("**/*") if path.is_file()]
            total_bytes = sum(path.stat().st_size for path in files)
            logger.info("Transformer cache dir summary: %s files=%d size=%.3f GB", cache_dir, len(files), total_bytes / 1.0e9)
        except OSError as exc:
            logger.warning("Could not summarize transformer cache directory %s: %s", cache_dir, exc)

    def _save_validated_cache_payload(self, payload, *, h5_path, selected_indices, metadata, cache_path, cache_key):
        valid, reason = self._cache_payload_matches_config(payload, selected_indices, metadata)
        if not valid:
            raise ValueError(
                f"Transformer cache builder produced an invalid {metadata['payload_kind']} payload: {reason}."
            )
        self._atomic_torch_save(payload, cache_path)
        self._write_cache_manifest(h5_path, cache_path, metadata, cache_key, payload)

    def _load_or_build_cache(self, h5_path, selected_indices, label_override=None, cache_context=None):
        cache_context = {
            **(cache_context or {}),
            "selected_indices_hash": _indices_hash(selected_indices),
        }
        if label_override is not None:
            cache_context = {**cache_context, "label_override": label_override}

        cache_path, metadata, cache_key = self._cache_path(h5_path=h5_path, cache_context=cache_context)
        manifest_path = cache_path.with_suffix(".json")
        rebuild = bool(self._cache_cfg().get("rebuild", False))
        require_existing = bool(self._cache_cfg().get("require_existing", False))
        start = time.perf_counter()

        logger.info("Transformer cache runtime H5 path: %s", h5_path)
        logger.info("Transformer cache source H5 path: %s", self._cache_source_h5_path() or "<unset>")
        logger.info("Transformer cache dataset_id: %s", metadata["dataset_id"])
        logger.info("Transformer cache directory: %s", cache_path.parent)
        logger.info("Transformer cache payload path: %s", cache_path)
        logger.info("Transformer cache manifest path: %s", manifest_path)
        self._log_cache_dir_summary(cache_path.parent)

        manifest_ok, manifest_reason = self._cache_manifest_matches_config(
            manifest_path, cache_path, metadata, cache_key
        )
        if cache_path.exists() and not rebuild and manifest_ok:
            logger.info("Loading transformer tensor cache from %s.", cache_path)
            payload = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
            ok, reason = self._cache_payload_matches_config(payload, selected_indices, metadata)
            if not ok:
                if require_existing:
                    raise RuntimeError(
                        f"Transformer tensor cache at {cache_path} is stale or incompatible ({reason}) "
                        "and dataset.cache.require_existing=true."
                    )
                raise ValueError(
                    f"cache exists but is incompatible with the current input configuration ({reason}); "
                    "use --force-rebuild-cache"
                )
            else:
                logger.info("Loaded compatible transformer tensor cache from %s.", cache_path)

            logger.info("Loaded transformer tensor cache in %.2f s.", time.perf_counter() - start)
            return payload

        if require_existing:
            if cache_path.exists() and not manifest_ok:
                raise RuntimeError(
                    f"Transformer tensor cache at {cache_path} is not complete ({manifest_reason}) "
                    "and dataset.cache.require_existing=true."
                )
            raise FileNotFoundError(
                f"No compatible transformer tensor cache found at {cache_path} "
                "with dataset.cache.require_existing=true."
            )

        if rebuild:
            logger.info("dataset.cache.rebuild=true; rebuilding transformer tensor cache at %s.", cache_path)
        else:
            logger.info("No transformer tensor cache found at %s; building it.", cache_path)

        if cache_path.exists() and not manifest_ok and not rebuild:
            raise ValueError(
                f"cache exists but is incompatible with the current input configuration ({manifest_reason}); "
                "use --force-rebuild-cache"
            )
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            manifest_ok, _ = self._cache_manifest_matches_config(
                manifest_path, cache_path, metadata, cache_key
            )
            if manifest_ok and not rebuild:
                payload = torch.load(
                    cache_path, map_location="cpu", weights_only=True, mmap=True
                )
            else:
                payload = self._build_cache_payload_chunked(
                    h5_path, selected_indices, metadata, label_override=label_override
                )
                self._save_validated_cache_payload(
                    payload, h5_path=h5_path, selected_indices=selected_indices,
                    metadata=metadata, cache_path=cache_path, cache_key=cache_key,
                )

        logger.info(
            "Built transformer tensor cache with %d events in %.2f s at %s (%.3f GB).",
            len(selected_indices),
            time.perf_counter() - start,
            cache_path,
            cache_path.stat().st_size / 1.0e9,
        )
        return payload

    def _cache_payload_matches_config(self, payload, selected_indices, metadata):
        try:
            tokens = payload["tokens"]
            type_ids = payload["type_ids"]
            padding_mask = payload["padding_mask"]
            labels = payload["labels"]
            physics_vectors = payload["physics_vectors"]
            indices = payload["indices"]
            event_ids = payload["event_id"]
            split = payload["split"]
            payload_metadata = payload.get("metadata", {})
        except (TypeError, KeyError) as exc:
            return False, f"missing payload key: {exc}"

        expected_kind = metadata.get("payload_kind")
        actual_kind = payload_metadata.get("payload_kind")
        if actual_kind != expected_kind:
            return False, f"payload_kind {actual_kind!r} != expected {expected_kind!r}"

        expected_shape = (len(selected_indices), self._configured_total_tokens(), self.schema.dim)

        if tuple(tokens.shape) != expected_shape:
            return False, f"tokens shape {tuple(tokens.shape)} != {expected_shape}"
        if tuple(type_ids.shape) != expected_shape[:2]:
            return False, f"type_ids shape {tuple(type_ids.shape)} != {expected_shape[:2]}"
        if tuple(padding_mask.shape) != expected_shape[:2]:
            return False, f"padding_mask shape {tuple(padding_mask.shape)} != {expected_shape[:2]}"
        if tuple(physics_vectors.shape) != expected_shape[:2] + (4,):
            return False, (
                f"physics_vectors shape {tuple(physics_vectors.shape)} != "
                f"{expected_shape[:2] + (4,)}"
            )
        if type_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            return False, f"type_ids must have an integer dtype, found {type_ids.dtype}"
        if padding_mask.dtype != torch.bool:
            return False, f"padding_mask must have boolean dtype, found {padding_mask.dtype}"
        if len(labels) != len(selected_indices):
            return False, f"labels length {len(labels)} != {len(selected_indices)}"
        if len(indices) != len(selected_indices):
            return False, f"indices length {len(indices)} != {len(selected_indices)}"
        if not torch.equal(indices.cpu(), torch.as_tensor(selected_indices, dtype=torch.long)):
            return False, "selected event indices differ"
        if tuple(event_ids.shape) != (len(selected_indices),):
            return False, "event_id shape mismatch"
        if event_ids.dtype != torch.int64 or len(torch.unique(event_ids)) != len(event_ids):
            return False, "EVENT_ID must be unique int64"
        if tuple(split.shape) != (len(selected_indices),) or split.dtype != torch.uint8:
            return False, "split identity must be uint8 [events]"
        if not set(torch.unique(split).tolist()) <= {0, 1, 2}:
            return False, "split identity contains unknown values"
        if "invisible_truth" in payload:
            invisible = payload["invisible_truth"]
            if invisible.shape[0] != len(selected_indices) or invisible.shape[-1] != 3:
                return False, "invisible_truth shape mismatch"
            if invisible.dtype != torch.float32 or not torch.isfinite(invisible).all():
                return False, "invisible_truth must be finite float32"
        if not torch.isfinite(tokens).all():
            return False, "tokens contain non-finite values"
        if not torch.isfinite(physics_vectors).all():
            return False, "physics_vectors contain non-finite values"
        if not torch.equal(
            physics_vectors[padding_mask], torch.zeros_like(physics_vectors[padding_mask])
        ):
            return False, "padded physics vectors are not exactly zero"
        if not torch.isfinite(labels).all():
            return False, "labels contain non-finite values"
        has_reco = "reco_raw_labels" in payload
        if has_reco != (expected_kind == "reconstruction"):
            return False, "payload reco_raw_labels presence disagrees with payload_kind"
        if expected_kind == "reconstruction":
            if "reco_raw_labels" not in payload:
                return False, "reconstruction is enabled but payload has no reco_raw_labels"
            if tuple(payload["reco_raw_labels"].shape) != expected_shape[:2]:
                return False, f"reco_raw_labels shape {tuple(payload['reco_raw_labels'].shape)} != {expected_shape[:2]}"
            if not torch.isfinite(payload["reco_raw_labels"]).all():
                return False, "reco_raw_labels contain non-finite values"

        actual = _plain_config_value(payload_metadata)
        expected = _plain_config_value(metadata)
        if actual != expected:
            actual_keys = set(actual or {})
            expected_keys = set(expected or {})
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            differing = sorted(key for key in actual_keys & expected_keys if actual.get(key) != expected.get(key))
            logger.info(
                "Transformer cache metadata mismatch: missing=%s extra=%s differing=%s",
                missing,
                extra,
                differing,
            )
            return False, "metadata mismatch"

        return True, "ok"

    def _build_cache_payload(self, h5_path, selected_indices, metadata, label_override=None):
        return self._build_cache_payload_chunked(h5_path, selected_indices, metadata, label_override=label_override)

    def _build_cache_payload_chunked(self, h5_path, selected_indices, metadata, label_override=None):
        num_workers = self._cache_num_workers()
        if num_workers == 1:
            return self._build_cache_payload_chunked_serial(
                h5_path,
                selected_indices,
                metadata,
                label_override=label_override,
            )
        return self._build_cache_payload_chunked_parallel(
            h5_path,
            selected_indices,
            metadata,
            label_override=label_override,
            num_workers=num_workers,
        )

    def _read_loaded_cache_chunk(self, h5_file, dataset, event_indices, label_override):
        event_indices = np.asarray(event_indices, dtype=np.int64)
        order = np.argsort(event_indices, kind="stable")
        sorted_indices = event_indices[order]
        restore = np.argsort(order, kind="stable")
        source_arrays = {}
        for source_name in dataset.schema.sources:
            path = f"/INPUTS/{source_name}"
            ds = h5_file[path]
            width = 1 if source_name == "GLOBAL" else int(dataset.widths[source_name])
            values = ds[sorted_indices, :width] if ds.ndim > 1 else ds[sorted_indices]
            source_arrays[source_name] = values[restore]
        if label_override is not None:
            label_values = np.full(len(event_indices), float(label_override), dtype=np.float32)
        elif dataset.classification_enabled:
            label_values = read_transformer_classification_labels(h5_file, sorted_indices)[restore]
        else:
            # Measured-only caches for structured objectives do not invent an
            # unrelated event-classification target.
            label_values = np.zeros(len(event_indices), dtype=np.float32)
        reco_label_arrays = {}
        if dataset.reconstruction_enabled:
            for source_name, token_indices in dataset.reco_builder.source_ranges.items():
                path = f"{dataset.reco_builder.label_group}/{source_name}"
                width = len(token_indices)
                values = h5_file[path][sorted_indices, :width]
                reco_label_arrays[source_name] = values[restore]
        return source_arrays, label_values, reco_label_arrays

    @staticmethod
    def _copy_cache_chunk(
        result, tokens, type_ids, padding_masks, labels, physics_vectors,
        reco_raw_labels,
    ):
        (
            output_positions, chunk_tokens, chunk_type_ids, chunk_masks,
            chunk_labels, chunk_physics, chunk_reco,
        ) = result
        output_positions = torch.as_tensor(output_positions, dtype=torch.long)
        tokens[output_positions] = torch.from_numpy(chunk_tokens)
        type_ids[output_positions] = torch.from_numpy(chunk_type_ids).to(type_ids.dtype)
        padding_masks[output_positions] = torch.from_numpy(chunk_masks)
        labels[output_positions] = torch.from_numpy(chunk_labels)
        physics_vectors[output_positions] = torch.from_numpy(chunk_physics)
        if reco_raw_labels is not None:
            reco_raw_labels[output_positions] = torch.from_numpy(chunk_reco)
        return len(output_positions)

    def _build_cache_payload_chunked_serial(self, h5_path, selected_indices, metadata, label_override=None):
        if self.scaler_state is None:
            if self.schema.has_observables and self._needs_scaler_state():
                self.scaler_state = self._fit_scaler_state(selected_indices, h5_path=h5_path)
            else:
                self.scaler_state = {}

        dataset = TransformerH5Dataset(
            h5_path,
            self.config,
            selected_indices,
            label_override=label_override,
            scaler_state=self.scaler_state,
            schema=self.schema,
            configured_widths=self.configured_widths,
            reconstruction_cfg=self.reconstruction_cfg if self.reconstruction_enabled else None,
            classification_enabled=self.classification_enabled,
        )

        n_events = len(dataset)
        tokens = torch.empty((n_events, dataset.total_tokens, self.schema.dim), dtype=torch.float32)
        type_ids = torch.empty((n_events, dataset.total_tokens), dtype=torch.int8)
        padding_masks = torch.empty((n_events, dataset.total_tokens), dtype=torch.bool)
        labels = torch.empty(n_events, dtype=torch.float32)
        physics_vectors = torch.empty(
            (n_events, dataset.total_tokens, 4), dtype=torch.float32
        )
        reco_raw_labels = (
            torch.empty((n_events, dataset.total_tokens), dtype=torch.float32)
            if self.reconstruction_enabled
            else None
        )
        chunk_size = self._cache_chunk_size()

        logger.info(
            "Transformer serial chunked cache build: events=%d chunk_size=%d chunks=%d host=%s pid=%d cpu_count=%s "
            "nproc=%s OMP=%s MKL=%s OPENBLAS=%s.",
            n_events,
            chunk_size,
            len(range(0, n_events, chunk_size)),
            socket.gethostname(),
            os.getpid(),
            os.cpu_count(),
            os.popen("nproc 2>/dev/null").read().strip() or "unknown",
            os.getenv("OMP_NUM_THREADS", "unset"),
            os.getenv("MKL_NUM_THREADS", "unset"),
            os.getenv("OPENBLAS_NUM_THREADS", "unset"),
        )

        try:
            with h5py.File(str(h5_path), "r") as h5_file:
                io_order = np.argsort(np.asarray(selected_indices, dtype=np.int64), kind="stable")
                for start_pos in range(0, n_events, chunk_size):
                    stop_pos = min(start_pos + chunk_size, n_events)
                    output_positions = io_order[start_pos:stop_pos]
                    event_indices = np.asarray(selected_indices[output_positions], dtype=np.int64)
                    source_arrays, label_values, reco_label_arrays = self._read_loaded_cache_chunk(
                        h5_file, dataset, event_indices, label_override
                    )
                    result = _process_loaded_cache_chunk(
                        dataset, output_positions, event_indices, source_arrays, label_values, reco_label_arrays
                    )
                    self._copy_cache_chunk(
                        result, tokens, type_ids, padding_masks, labels,
                        physics_vectors, reco_raw_labels
                    )
                    logger.info(
                        "Transformer cache progress: events=%d/%d workers=1.", stop_pos, n_events
                    )
        finally:
            dataset.close()

        payload = {
            "indices": torch.as_tensor(selected_indices, dtype=torch.long),
            "tokens": tokens,
            "type_ids": type_ids,
            "padding_mask": padding_masks,
            "labels": labels,
            "physics_vectors": physics_vectors,
            "metadata": metadata,
        }
        if self.reconstruction_enabled:
            payload["reco_raw_labels"] = reco_raw_labels
        self._add_canonical_payload(payload, h5_path, selected_indices)
        return payload

    def _build_cache_payload_chunked_parallel(
        self,
        h5_path,
        selected_indices,
        metadata,
        *,
        label_override=None,
        num_workers,
    ):
        if self.scaler_state is None:
            self.scaler_state = (
                self._fit_scaler_state(selected_indices, h5_path=h5_path)
                if self.schema.has_observables and self._needs_scaler_state() else {}
            )
        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        dataset = TransformerH5Dataset(
            h5_path,
            self.config,
            selected_indices[:0],
            label_override=label_override,
            scaler_state=self.scaler_state,
            schema=self.schema,
            configured_widths=self.configured_widths,
            reconstruction_cfg=self.reconstruction_cfg if self.reconstruction_enabled else None,
            classification_enabled=(
                label_override is None and self.classification_enabled
            ),
        )
        n_events = len(selected_indices)
        tokens = torch.empty((n_events, dataset.total_tokens, self.schema.dim), dtype=torch.float32)
        type_ids = torch.empty((n_events, dataset.total_tokens), dtype=torch.int8)
        padding_masks = torch.empty((n_events, dataset.total_tokens), dtype=torch.bool)
        labels = torch.empty(n_events, dtype=torch.float32)
        physics_vectors = torch.empty(
            (n_events, dataset.total_tokens, 4), dtype=torch.float32
        )
        reco_raw_labels = (
            torch.empty((n_events, dataset.total_tokens), dtype=torch.float32)
            if self.reconstruction_enabled else None
        )
        chunk_size = self._cache_chunk_size()
        chunk_specs = [
            (start, min(start + chunk_size, n_events))
            for start in range(0, n_events, chunk_size)
        ]
        io_order = np.argsort(selected_indices, kind="stable")
        max_in_flight = max(1, 2 * num_workers)
        logger.info(
            "Transformer threaded cache build: events=%d chunk_size=%d chunks=%d workers=%d host=%s.",
            n_events, chunk_size, len(chunk_specs), num_workers, socket.gethostname(),
        )
        start_time = time.perf_counter()
        completed_events = 0
        pending = set()

        def collect(done):
            nonlocal completed_events
            for future in done:
                try:
                    completed_events += self._copy_cache_chunk(
                        future.result(), tokens, type_ids, padding_masks, labels,
                        physics_vectors, reco_raw_labels
                    )
                except Exception as exc:
                    chunk_range = getattr(future, "chunk_range", "unknown")
                    raise RuntimeError(
                        f"Transformer cache preprocessing failed for event-position chunk {chunk_range}."
                    ) from exc
                elapsed = max(time.perf_counter() - start_time, 1.0e-9)
                logger.info(
                    "Transformer cache progress: events=%d/%d workers=%d elapsed=%.2f s rate=%.1f events/s.",
                    completed_events, n_events, num_workers, elapsed, completed_events / elapsed,
                )

        try:
            with h5py.File(str(h5_path), "r") as h5_file, ThreadPoolExecutor(max_workers=num_workers) as executor:
                for start_pos, stop_pos in chunk_specs:
                    while len(pending) >= max_in_flight:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        collect(done)
                    output_positions = io_order[start_pos:stop_pos].copy()
                    event_indices = selected_indices[output_positions].copy()
                    source_arrays, label_values, reco_label_arrays = self._read_loaded_cache_chunk(
                        h5_file, dataset, event_indices, label_override
                    )
                    future = executor.submit(
                        _process_loaded_cache_chunk,
                        dataset,
                        output_positions,
                        event_indices,
                        source_arrays,
                        label_values,
                        reco_label_arrays,
                    )
                    future.chunk_range = (int(event_indices[0]), int(event_indices[-1]))
                    pending.add(future)
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    collect(done)
        finally:
            dataset.close()

        elapsed = max(time.perf_counter() - start_time, 1.0e-9)
        logger.info(
            "Transformer threaded cache build completed: events=%d workers=%d elapsed=%.2f s rate=%.1f events/s.",
            n_events, num_workers, elapsed, n_events / elapsed,
        )
        payload = {
            "indices": torch.as_tensor(selected_indices, dtype=torch.long),
            "tokens": tokens,
            "type_ids": type_ids,
            "padding_mask": padding_masks,
            "labels": labels,
            "physics_vectors": physics_vectors,
            "metadata": metadata,
        }
        if self.reconstruction_enabled:
            payload["reco_raw_labels"] = reco_raw_labels
        self._add_canonical_payload(payload, h5_path, selected_indices)
        return payload

    @staticmethod
    def _add_canonical_payload(payload, h5_path, selected_indices):
        """Attach model-agnostic event identity, split, and invisible truth."""
        indices = np.asarray(selected_indices, dtype=np.int64)
        with h5py.File(str(h5_path), "r") as source:
            if "EVENT_ID" not in source:
                raise KeyError(f"Canonical dataset {h5_path} is missing EVENT_ID")
            if "SPLIT" not in source:
                raise KeyError(f"Canonical dataset {h5_path} is missing SPLIT")
            order = np.argsort(indices, kind="stable")
            restore = np.argsort(order, kind="stable")
            sorted_indices = indices[order]
            payload["event_id"] = torch.from_numpy(
                np.asarray(source["EVENT_ID"][sorted_indices], dtype=np.int64)[restore].copy()
            )
            payload["split"] = torch.from_numpy(
                np.asarray(source["SPLIT"][sorted_indices], dtype=np.uint8)[restore].copy()
            )
            if "LABELS/NEUTRINO" in source:
                raw = source["LABELS/NEUTRINO"][sorted_indices][restore]
                if raw.dtype.names != ("px", "py", "pz"):
                    raise TypeError(
                        f"{h5_path}:LABELS/NEUTRINO fields must be exactly px,py,pz"
                    )
                invisible = np.stack([raw[name] for name in raw.dtype.names], axis=-1)
                if not np.isfinite(invisible).all():
                    raise ValueError(f"{h5_path}:LABELS/NEUTRINO contains non-finite values")
                payload["invisible_truth"] = torch.from_numpy(
                    np.asarray(invisible, dtype=np.float32).copy()
                )

    def sample_dataloader(self, sample_cfg, batch_size=None):
        h5_path = sample_cfg.get("path")
        if h5_path is None:
            raise ValueError("Sample config requires a path.")

        sample_name = sample_cfg.get("name", h5_path)
        label_override = sample_cfg.get("label", None)
        max_events = sample_cfg.get("max_events", None)

        validate_transformer_h5_inputs(
            h5_path,
            self.schema,
            self.config,
            reconstruction_cfg=self.reconstruction_cfg if self.reconstruction_enabled else None,
            classification_enabled=(
                label_override is None
                and bool((self.full_config.get("tasks", {}).get("classification", {}) or {}).get("enabled", True))
            ),
            label_override=label_override,
            config_name=str(self.full_config.get("outputs", {}).get("config_name", "transformer")),
        )

        with h5py.File(str(h5_path), "r") as h5_file:
            n_events = h5_file["/INPUTS/JET"].shape[0]
            sample_widths = _configured_widths_from_h5(h5_file, self.schema, self.config)

        selected_indices = np.arange(n_events, dtype=np.int64)
        if max_events is not None and int(max_events) < n_events:
            selected_indices = selected_indices[: int(max_events)]

        if self.scaler_state is None:
            if self.schema.has_observables and self._needs_scaler_state():
                logger.info(
                    "Fitting transformer observable scalers before sample scoring. "
                    "For production evaluation, prefer calling setup() on the training datamodule first."
                )
                old_widths = self.configured_widths
                self.configured_widths = sample_widths
                self.scaler_state = self._fit_scaler_state(selected_indices, h5_path=h5_path)
                self.configured_widths = old_widths or sample_widths
            else:
                self.scaler_state = {}

        logger.info(
            "Preparing sample dataloader for %s: %s (%d/%d events, label_override=%s).",
            sample_name,
            h5_path,
            len(selected_indices),
            n_events,
            label_override,
        )

        old_widths = self.configured_widths
        self.configured_widths = sample_widths
        sample_reco_builder = (
            ReconstructionCandidateBuilder(self.reconstruction_cfg, self.schema, sample_widths)
            if self.reconstruction_enabled
            else None
        )

        if self._cache_enabled():
            payload = self._load_or_build_cache(
                h5_path,
                selected_indices,
                label_override=label_override,
                cache_context={
                    "max_events": max_events,
                    "label_override": label_override,
                    "selected_indices_hash": _indices_hash(selected_indices),
                },
            )
            dataset = CachedTransformerDataset(
                payload,
                np.arange(len(selected_indices), dtype=np.int64),
                reco_builder=sample_reco_builder,
            )
        else:
            dataset = TransformerH5Dataset(
                h5_path,
                self.config,
                selected_indices,
                label_override=label_override,
                scaler_state=self.scaler_state,
                schema=self.schema,
                configured_widths=sample_widths,
                reconstruction_cfg=self.reconstruction_cfg if self.reconstruction_enabled else None,
                classification_enabled=(
                    label_override is None and self.classification_enabled
                ),
            )

        self.configured_widths = old_widths or sample_widths

        return DataLoader(
            dataset,
            batch_size=int(batch_size or self.config.get("test_batch_size", 1024)),
            shuffle=False,
            collate_fn=_collate_transformer,
            **self._dataloader_kwargs("test"),
        )

    def _dataloader_kwargs(self, split):
        num_workers = int(self.config.get(f"{split}_num_workers", 0))
        pin_memory = bool(self.config.get("pin_memory", False)) and torch.cuda.is_available()
        if bool(self.config.get("pin_memory", False)) and not pin_memory:
            logger.info("Disabling pin_memory because CUDA is not available in this process.")

        kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "persistent_workers": bool(self.config.get("persistent_workers", False)) if num_workers > 0 else False,
        }

        prefetch_factor = self.config.get("prefetch_factor", None)
        if num_workers > 0 and prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)

        return kwargs

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            collate_fn=_collate_transformer,
            **self._dataloader_kwargs("train"),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            collate_fn=_collate_transformer,
            **self._dataloader_kwargs("val"),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            collate_fn=_collate_transformer,
            **self._dataloader_kwargs("test"),
        )
