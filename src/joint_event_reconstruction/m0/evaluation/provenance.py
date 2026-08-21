"""Prediction-manifest and saved-run provenance resolution."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from joint_event_reconstruction.m0.evaluation.topology import topology_spec


def normalise_path(value, *, base_dir: Path | None = None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        if base_dir is None:
            path = Path.cwd() / path
        else:
            path = base_dir / path
    return path.resolve()


def _path_list(value, *, base_dir: Path) -> tuple[Path, ...]:
    if value in (None, ""):
        return ()
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(normalise_path(item, base_dir=base_dir) for item in values)


def _find_saved_predict_config(manifest_path: Path) -> Path | None:
    for parent in (manifest_path.parent, *manifest_path.parent.parents):
        candidate = parent / "hydra" / "predict" / ".hydra" / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def _load_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read saved prediction configuration {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Saved prediction configuration must be a mapping: {path}")
    return value


def _saved_value(config: dict, *keys, default=None):
    value = config
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _same_paths(left: tuple[Path, ...], right: tuple[Path, ...]) -> bool:
    return set(left) == set(right)


def _require_consistent(name: str, manifest_value, saved_value, *, normalise=None):
    if manifest_value in (None, ""):
        return saved_value
    if saved_value in (None, ""):
        return manifest_value
    left = normalise(manifest_value) if normalise else manifest_value
    right = normalise(saved_value) if normalise else saved_value
    if left != right:
        raise ValueError(
            f"Prediction manifest and saved run configuration disagree for {name}: "
            f"manifest={manifest_value!r}, saved={saved_value!r}."
        )
    return manifest_value


def resolve_prediction_manifest(manifest_path: str | Path) -> dict:
    """Resolve modern manifests and the narrow completed-run legacy form."""
    path = normalise_path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction manifest does not exist: {path}")
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read prediction manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"Prediction manifest must contain a JSON object: {path}")

    saved_path = _find_saved_predict_config(path)
    saved = _load_yaml(saved_path) if saved_path else {}
    saved_topology = _saved_value(saved, "general", "topology")
    saved_cap = _saved_value(saved, "dataset", "max_jets")
    saved_h5 = _saved_value(saved, "dataset", "input_h5_path")
    saved_config_name = _saved_value(saved, "outputs", "config_name")

    topology = _require_consistent("topology", manifest.get("topology"), saved_topology)
    model_cap = _require_consistent("model_max_jets", manifest.get("model_max_jets"), saved_cap)
    config_name = _require_consistent("config_name", manifest.get("config_name"), saved_config_name)

    manifest_h5 = manifest.get("source_h5_path", manifest.get("input_h5_path"))
    source_paths = _path_list(manifest_h5, base_dir=path.parent)
    saved_paths = _path_list(saved_h5, base_dir=saved_path.parent if saved_path else path.parent)
    if source_paths and saved_paths and not _same_paths(source_paths, saved_paths):
        raise ValueError(
            "Prediction manifest and saved run configuration disagree for source H5 paths: "
            f"manifest={source_paths}, saved={saved_paths}."
        )
    source_paths = source_paths or saved_paths
    if not source_paths:
        raise ValueError(
            f"Source H5 path is absent from prediction manifest {path} and saved configuration "
            f"{saved_path or '<not found>'}."
        )

    if topology in (None, ""):
        raise ValueError(
            f"Topology is absent from prediction manifest {path} and saved configuration "
            f"{saved_path or '<not found>'}; it cannot be inferred from filenames."
        )
    topology = str(topology)
    topology_spec(topology)
    if model_cap in (None, ""):
        raise ValueError(
            f"Training jet cap is absent from prediction manifest {path} and saved configuration "
            f"{saved_path or '<not found>'}."
        )
    try:
        model_cap = int(model_cap)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"model_max_jets must be an integer in {path}: {model_cap!r}") from exc
    if model_cap <= 0:
        raise ValueError(f"model_max_jets must be positive in {path}: {model_cap}")

    output_directory_value = manifest.get("output_directory", manifest.get("run_output_dir"))
    output_directory = normalise_path(output_directory_value, base_dir=path.parent) if output_directory_value else path.parent
    prediction_events_value = manifest.get("prediction_events")
    prediction_events = (
        normalise_path(prediction_events_value, base_dir=path.parent)
        if prediction_events_value
        else None
    )
    legacy_classification = manifest.get("classification_predictions")
    legacy_reco_events = manifest.get("reconstruction_event_predictions")
    legacy_metadata = manifest.get("reconstruction_event_metadata")
    paths = {
        "prediction_events": prediction_events,
        "classification_predictions": normalise_path(legacy_classification, base_dir=path.parent) if legacy_classification else None,
        "reconstruction_event_predictions": normalise_path(legacy_reco_events, base_dir=path.parent) if legacy_reco_events else None,
        "reconstruction_event_metadata": normalise_path(legacy_metadata, base_dir=path.parent) if legacy_metadata else None,
        "candidate_table": normalise_path(manifest["candidate_table"], base_dir=path.parent) if manifest.get("candidate_table") else None,
    }
    if paths["prediction_events"] is None and not paths["classification_predictions"] and not paths["reconstruction_event_metadata"]:
        raise ValueError(
            f"Prediction manifest {path} contains no supported prediction product. "
            "Run prediction first and use its prediction_manifest.json."
        )

    run_output_value = manifest.get("run_output_dir")
    run_output = normalise_path(run_output_value, base_dir=path.parent) if run_output_value else None
    if run_output is None or (run_output == output_directory and prediction_events is None):
        if saved_path:
            run_output = saved_path.parents[3]
        elif output_directory.name == "predictions":
            run_output = output_directory.parent
        else:
            run_output = output_directory.parent.parent

    return {
        "manifest_path": path,
        "manifest": manifest,
        "saved_config_path": saved_path,
        "topology": topology,
        "model_max_jets": model_cap,
        "source_h5_paths": source_paths,
        "config_name": str(config_name or manifest.get("config_name") or "unknown"),
        "checkpoint_path": normalise_path(manifest["checkpoint_path"], base_dir=path.parent) if manifest.get("checkpoint_path") else None,
        "output_directory": output_directory,
        "run_output_dir": run_output,
        "paths": paths,
        "legacy": paths["prediction_events"] is None,
    }
