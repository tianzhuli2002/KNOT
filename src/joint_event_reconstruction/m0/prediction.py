"""Transformer prediction with compact event-level output."""

from __future__ import annotations

import json
import gzip
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from joint_event_reconstruction.m0.evaluation.decoder import decode_tth, decode_ttbar
from joint_event_reconstruction.m0.evaluation.topology import topology_spec
from joint_event_reconstruction.m0.runtime import (
    get_hydra_config_name,
    get_latest_checkpoint_path,
    load_checkpoint_into_model,
    resolve_prediction_output_dir,
    set_execution_device,
)

logger = logging.getLogger(__name__)

PREDICTION_COLUMNS = ["path", "sample", "event_index", "label", "classification_score"]
COMMON_EVENT_COLUMNS = PREDICTION_COLUMNS + ["n_input_jets"]
RECO_COLUMNS = [
    "reco_valid",
    "reco_score",
    "reco_confidence",
    "reco_assignment_json",
]
TTBAR_RECO_COLUMNS = [
    "blep_jet_index",
    "bhad_jet_index",
    "whad_jet_indices",
    "tlep_jet_index",
    "tlep_lepton_index",
    "tlep_met_index",
]
TTH_RECO_COLUMNS = TTBAR_RECO_COLUMNS + ["higgs_jet_indices"]
CANDIDATE_COLUMNS = [
    "path", "sample", "event_index", "label", "target", "rank",
    "token_indices", "member_sources", "source_indices", "candidate_score", "is_positive",
]


def prediction_event_columns(topology: str, *, reconstruction: bool) -> list[str]:
    columns = list(COMMON_EVENT_COLUMNS)
    if reconstruction:
        columns.extend(RECO_COLUMNS)
        columns.extend(TTH_RECO_COLUMNS if topology == "ttH" else TTBAR_RECO_COLUMNS)
    return columns


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _has_samples(cfg_section) -> bool:
    samples = _cfg_get(cfg_section, "samples", None)
    return samples is not None and len(samples) > 0


def _batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: _batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(_batch_to_device(value, device) for value in batch)
    return batch


def _extract_logits(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("classification_logits", "logits", "logit", "pred", "prediction", "score"):
            if key in output:
                return output[key]
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Could not extract classification logits from model output type {type(output)}")


def _extract_reconstruction_logits(output) -> dict:
    if not isinstance(output, dict):
        return {}
    return output.get("reconstruction_logits") or {}


def _forward_model(model, batch):
    if isinstance(batch, dict):
        output = model(
            batch["tokens"], batch["type_ids"], batch["padding_mask"], reco=batch.get("reco")
        )
        return output, _extract_logits(output), batch["labels"]
    inputs, labels = batch
    output = model(inputs)
    return output, _extract_logits(output), labels


def _event_indices(batch, size: int, fallback: int) -> np.ndarray:
    if isinstance(batch, dict) and "event_index" in batch:
        return batch["event_index"].detach().cpu().numpy().reshape(-1).astype(np.int64)
    return np.arange(fallback, fallback + size, dtype=np.int64)


def _input_jet_counts(batch, jet_type_id: int) -> np.ndarray:
    """Count valid input JET tokens using the authoritative token schema."""
    if not isinstance(batch, dict) or "type_ids" not in batch or "padding_mask" not in batch:
        raise ValueError("Transformer prediction requires type_ids and padding_mask to count input jets.")
    type_ids = batch["type_ids"]
    padding_mask = batch["padding_mask"]
    if type_ids.shape != padding_mask.shape:
        raise ValueError(
            "Transformer prediction type_ids and padding_mask shapes disagree: "
            f"{tuple(type_ids.shape)} != {tuple(padding_mask.shape)}."
        )
    valid_jets = (type_ids == int(jet_type_id)) & ~padding_mask.bool()
    return valid_jets.sum(dim=1).detach().cpu().numpy().astype(np.int64)


def _candidate_string(values) -> str:
    return json.dumps([int(value) for value in values])


def _jsonable_candidate(candidate) -> dict:
    return {
        "token_indices": [int(value) for value in candidate["token_indices"]],
        "source_indices": [int(value) for value in candidate["source_indices"]],
        "member_sources": [str(value) for value in candidate["member_sources"]],
        "probability": float(candidate["probability"]),
        "truth_correct": bool(candidate.get("truth_correct", False)),
    }


def _jsonable_topology(result) -> dict:
    return {
        "reco_valid": bool(result["reco_valid"]),
        "topology_score": float(result["topology_score"]) if np.isfinite(result["topology_score"]) else None,
        "topology_confidence": float(result["topology_confidence"]) if np.isfinite(result["topology_confidence"]) else None,
        "selected": {
            role: _jsonable_candidate(candidate)
            for role, candidate in result.get("selected", {}).items()
        },
        "blep": result.get("blep"),
        "bhad": result.get("bhad"),
        "Wlep": result.get("Wlep", {}),
        "Whad_daughters": [int(value) for value in result.get("Whad_daughters", ())],
        "H_daughters": [int(value) for value in result.get("H_daughters", ())],
    }


def _candidate_rows(output, batch, labels, event_indices, sample_name, sample_path, export_events):
    logits_by_pool = _extract_reconstruction_logits(output)
    pools = batch.get("reco", {}).get("pools", {}) if isinstance(batch, dict) else {}
    rows = []
    if not logits_by_pool:
        return rows
    export_events = set(int(value) for value in export_events)
    for pool_name, logits in logits_by_pool.items():
        if pool_name not in pools:
            raise ValueError(f"Reconstruction logits contain pool {pool_name!r} absent from the input batch.")
        pool = pools[pool_name]
        probabilities = logits.detach().float().softmax(dim=-1).cpu().numpy()
        mask = pool["candidate_mask"].detach().cpu().numpy().astype(bool)
        truth = pool["truth_class"].detach().cpu().numpy()
        token_indices = pool["candidate_indices"].detach().cpu().numpy()
        source_indices = pool["candidate_source_indices"].detach().cpu().numpy()
        sources = tuple(pool["member_sources"])
        for class_index, target_name in enumerate(pool["class_names"][1:], start=1):
            for batch_index, event_index in enumerate(event_indices):
                if int(event_index) not in export_events:
                    continue
                valid = np.flatnonzero(mask[batch_index])
                order = valid[np.argsort(-probabilities[batch_index, valid, class_index], kind="stable")]
                for rank, candidate_index in enumerate(order, start=1):
                    rows.append({
                        "path": str(sample_path), "sample": str(sample_name), "event_index": int(event_index),
                        "label": int(labels[batch_index]), "target": str(target_name), "rank": int(rank),
                        "token_indices": _candidate_string(token_indices[batch_index, candidate_index]),
                        "member_sources": json.dumps(list(sources)),
                        "source_indices": _candidate_string(source_indices[batch_index, candidate_index]),
                        "candidate_score": float(probabilities[batch_index, candidate_index, class_index]),
                        "is_positive": bool(int(truth[batch_index, candidate_index]) == class_index),
                    })
    return rows


def _decode_batch_topologies(output, batch, topology: str) -> list[dict]:
    logits_by_pool = _extract_reconstruction_logits(output)
    pools = batch.get("reco", {}).get("pools", {})
    if not pools:
        return [{} for _ in range(batch["labels"].shape[0])]
    batch_size = next(iter(pools.values()))["candidate_mask"].shape[0]
    roles_by_event = [dict() for _ in range(batch_size)]
    for pool_name, pool in pools.items():
        if pool_name not in logits_by_pool:
            raise ValueError(f"Reconstruction input pool {pool_name!r} has no model output.")
        probabilities = logits_by_pool[pool_name].detach().float().softmax(dim=-1).cpu()
        mask = pool["candidate_mask"].detach().cpu()
        truth = pool["truth_class"].detach().cpu()
        token_indices = pool["candidate_indices"].detach().cpu()
        source_indices = pool["candidate_source_indices"].detach().cpu()
        member_sources = tuple(pool["member_sources"])
        for class_index, role in enumerate(pool["class_names"][1:], start=1):
            for event in range(batch_size):
                candidates = []
                for candidate_index in torch.nonzero(mask[event], as_tuple=False).reshape(-1).tolist():
                    candidates.append({
                        "token_indices": [int(value) for value in token_indices[event, candidate_index].tolist()],
                        "source_indices": [int(value) for value in source_indices[event, candidate_index].tolist()],
                        "member_sources": member_sources,
                        "probability": float(probabilities[event, candidate_index, class_index]),
                        "truth_correct": bool(truth[event, candidate_index] == class_index),
                    })
                roles_by_event[event][role] = candidates
    decoder = decode_tth if topology == "ttH" else decode_ttbar
    return [decoder(roles) for roles in roles_by_event]


def _assignment_columns(result: dict, topology: str) -> dict:
    selected = result.get("selected", {})

    def members(role, source):
        candidate = selected.get(role, {})
        return [
            int(index)
            for member_source, index in zip(candidate.get("member_sources", ()), candidate.get("source_indices", ()))
            if member_source == source
        ]

    tlep_jets = members("tlep", "JET")
    tlep_leptons = members("tlep", "LEPTON")
    tlep_met = members("tlep", "MET")
    values = {
        "blep_jet_index": result.get("blep"),
        "bhad_jet_index": result.get("bhad"),
        "whad_jet_indices": _candidate_string(members("Whad", "JET")),
        "tlep_jet_index": tlep_jets[0] if len(tlep_jets) == 1 else None,
        "tlep_lepton_index": tlep_leptons[0] if len(tlep_leptons) == 1 else None,
        "tlep_met_index": tlep_met[0] if len(tlep_met) == 1 else None,
    }
    if topology == "ttH":
        values["higgs_jet_indices"] = _candidate_string(members("H", "JET"))
    return values


def _iter_scored_batches(
    dataloader, model, device, *, topology: str, sample_name: str, sample_path: str,
    jet_type_id: int,
    label_override=None, write_classification=True, write_reconstruction=False,
    diagnostics=False, candidate_table_max_events=None,
):
    fallback_event_index = 0
    exported_event_indices = set()
    with torch.inference_mode():
        for batch in dataloader:
            batch = _batch_to_device(batch, device)
            output, logits, labels = _forward_model(model, batch)
            raw_labels = labels.detach().cpu().numpy().reshape(-1)
            batch_labels = (
                raw_labels if label_override is None else np.full(raw_labels.shape, int(label_override), dtype=np.int64)
            )
            event_indices = _event_indices(batch, len(batch_labels), fallback_event_index)
            fallback_event_index += len(batch_labels)
            input_counts = _input_jet_counts(batch, jet_type_id)
            decoded = _decode_batch_topologies(output, batch, topology) if write_reconstruction else [{}] * len(batch_labels)
            scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1) if write_classification else None
            if diagnostics:
                remaining = None if candidate_table_max_events is None else max(0, int(candidate_table_max_events) - len(exported_event_indices))
                if remaining is None:
                    exported_event_indices.update(int(value) for value in event_indices)
                else:
                    exported_event_indices.update(int(value) for value in event_indices[:remaining])
            event_rows = [
                {
                    "path": str(sample_path), "sample": str(sample_name), "event_index": int(event_index),
                    "label": int(label), "classification_score": float(scores[index]) if scores is not None else np.nan,
                    "n_input_jets": int(n_input_jets),
                    **(
                        {
                            "reco_valid": bool(result.get("reco_valid", False)),
                            "reco_score": result.get("topology_score"),
                            "reco_confidence": result.get("topology_confidence"),
                            "reco_assignment_json": json.dumps(_jsonable_topology(result), sort_keys=True, separators=(",", ":")),
                            **_assignment_columns(result, topology),
                        } if write_reconstruction else {}
                    ),
                }
                for index, (event_index, label, n_input_jets, result) in enumerate(
                    zip(event_indices, batch_labels, input_counts, decoded)
                )
            ]
            candidates = (
                _candidate_rows(output, batch, batch_labels, event_indices, sample_name, sample_path, exported_event_indices)
                if diagnostics else []
            )
            yield pd.DataFrame(event_rows), pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS)


def _score_dataloader_outputs(
    dataloader, model, device, *, topology: str, sample_name: str, sample_path: str,
    jet_type_id: int,
    label_override=None, write_classification=True, write_reconstruction=False,
    diagnostics=False, candidate_table_max_events=None,
):
    event_frames = []
    candidate_frames = []
    for event_frame, candidate_frame in _iter_scored_batches(
        dataloader, model, device, topology=topology, sample_name=sample_name, sample_path=sample_path,
        jet_type_id=jet_type_id,
        label_override=label_override, write_classification=write_classification,
        write_reconstruction=write_reconstruction, diagnostics=diagnostics,
        candidate_table_max_events=candidate_table_max_events,
    ):
        event_frames.append(event_frame)
        if diagnostics:
            candidate_frames.append(candidate_frame)
    if not event_frames:
        raise ValueError(f"Prediction for sample {sample_name!r} produced no events.")
    return pd.concat(event_frames, ignore_index=True), pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame(columns=CANDIDATE_COLUMNS)


def _checkpoint_summary(checkpoint_path, cfg):
    monitor = ((cfg.get("train", {}) or {}).get("checkpoint", {}) or {}).get("monitor")
    best_value = None
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for state in (payload.get("callbacks", {}) or {}).values():
            if isinstance(state, dict) and (monitor is None or state.get("monitor", monitor) == monitor):
                if state.get("best_model_score") is not None:
                    best_value = float(state["best_model_score"])
                    break
    except (OSError, RuntimeError, TypeError, ValueError):
        logger.warning("Could not read checkpoint monitor provenance from %s.", checkpoint_path)
    return monitor, best_value


def _write_prediction_manifest(
    prediction_dir, checkpoint_path, cfg, *, candidate_path, topology, source_paths,
    n_events, label_counts, classification_enabled, reconstruction_enabled, source_type_ids,
):
    prediction_path = prediction_dir / "prediction_events.csv.gz"
    configured_name = (cfg.get("outputs", {}) or {}).get("config_name") or get_hydra_config_name("config")
    monitor, best_value = _checkpoint_summary(checkpoint_path, cfg)
    manifest = {
        "topology": str(topology),
        "model_max_jets": int(cfg.dataset.max_jets),
        "source_h5_path": str(source_paths[0]) if len(source_paths) == 1 else [str(path) for path in source_paths],
        "config_name": str(configured_name),
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "prediction_events": str(prediction_path.resolve()),
        "candidate_table": str(candidate_path.resolve()) if candidate_path else None,
        "output_directory": str(prediction_dir.resolve()),
        "run_output_dir": str(prediction_dir.parent.parent.resolve()),
        "classification_enabled": bool(classification_enabled),
        "reconstruction_enabled": bool(reconstruction_enabled),
        "source_type_ids": {str(key): int(value) for key, value in source_type_ids.items()},
        "n_events": int(n_events),
        "label_counts": {str(key): int(value) for key, value in sorted(label_counts.items())},
        "checkpoint_monitor": monitor,
        "checkpoint_best_value": best_value,
    }
    temporary = prediction_dir / ".prediction_manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(prediction_dir / "prediction_manifest.json")


def predict(datamodule, model, cfg) -> None:
    """Score the configured prediction split and write one compact event table."""
    predict_cfg = cfg.get("predict", {}) or {}
    tasks_cfg = cfg.get("tasks", {}) or {}
    topology = str(cfg.general.topology)
    topology_spec(topology)
    write_classification = bool(predict_cfg.get("write_classification", (tasks_cfg.get("classification", {}) or {}).get("enabled", True)))
    write_reconstruction = bool(predict_cfg.get("write_reconstruction", (tasks_cfg.get("reconstruction", {}) or {}).get("enabled", False)))
    diagnostics = bool(predict_cfg.get("diagnostics", False))
    candidate_limit = predict_cfg.get("candidate_table_max_events")
    if diagnostics and candidate_limit is not None and int(candidate_limit) <= 0:
        raise ValueError("predict.candidate_table_max_events must be positive or null.")
    if not write_classification and not write_reconstruction:
        raise ValueError("Prediction must enable classification or reconstruction output.")
    checkpoint_value = predict_cfg.get("checkpoint_path") or (cfg.get("performance", {}) or {}).get("checkpoint_path")
    if not checkpoint_value:
        raise ValueError("Prediction requires predict.checkpoint_path.")
    checkpoint_path = get_latest_checkpoint_path(str(checkpoint_value))
    prediction_dir = resolve_prediction_output_dir(cfg)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    output_path = prediction_dir / "prediction_events.csv.gz"
    manifest_path = prediction_dir / "prediction_manifest.json"
    if (output_path.exists() or manifest_path.exists()) and not bool(predict_cfg.get("overwrite", False)):
        raise FileExistsError(
            f"Prediction output already exists in {prediction_dir}. Set predict.overwrite=true to replace it explicitly."
        )

    device = torch.device(set_execution_device(cfg.general.device))
    model = load_checkpoint_into_model(model, checkpoint_path).to(device)
    model.eval()
    if not _has_samples(predict_cfg):
        datamodule.setup()
        dataloaders = [("test_split", str(Path(str(cfg.dataset.input_h5_path)).expanduser().resolve()), None, datamodule.test_dataloader())]
    else:
        dataloaders = []
        for sample_cfg in predict_cfg.samples:
            sample_name = _cfg_get(sample_cfg, "name")
            sample_path = str(Path(str(_cfg_get(sample_cfg, "path"))).expanduser().resolve())
            dataloaders.append((str(sample_name), sample_path, _cfg_get(sample_cfg, "label"), datamodule.sample_dataloader(sample_cfg, batch_size=predict_cfg.get("batch_size"))))

    source_paths = []
    try:
        jet_type_id = int(datamodule.schema.source_type_ids["JET"])
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError("Transformer datamodule schema does not define a JET source type ID.") from exc
    columns = prediction_event_columns(topology, reconstruction=write_reconstruction)
    event_temporary = output_path.with_name(f".{output_path.name}.tmp")
    candidate_path = None
    candidate_temporary = None
    if diagnostics:
        candidate_path = prediction_dir / "candidates.csv.gz"
        candidate_temporary = candidate_path.with_name(f".{candidate_path.name}.tmp")
    n_events = 0
    label_counts = {}
    event_header = False
    candidate_header = False
    try:
        with gzip.open(event_temporary, "wt", encoding="utf-8", newline="") as event_stream:
            candidate_stream = gzip.open(candidate_temporary, "wt", encoding="utf-8", newline="") if diagnostics else None
            try:
                for sample_name, sample_path, label_override, dataloader in dataloaders:
                    source_paths.append(Path(sample_path))
                    sample_events = 0
                    for event_frame, candidate_frame in _iter_scored_batches(
                        dataloader, model, device, topology=topology, sample_name=sample_name, sample_path=sample_path,
                        jet_type_id=jet_type_id,
                        label_override=label_override, write_classification=write_classification,
                        write_reconstruction=write_reconstruction, diagnostics=diagnostics,
                        candidate_table_max_events=candidate_limit,
                    ):
                        event_frame = event_frame[columns]
                        if event_frame.duplicated(["path", "sample", "event_index"]).any():
                            raise ValueError("Prediction produced duplicate path/sample/event_index identities within a batch.")
                        event_frame.to_csv(event_stream, index=False, header=not event_header)
                        event_header = True
                        sample_events += len(event_frame)
                        n_events += len(event_frame)
                        for label, count in event_frame["label"].value_counts().items():
                            label_counts[int(label)] = label_counts.get(int(label), 0) + int(count)
                        if diagnostics:
                            candidate_frame.to_csv(candidate_stream, index=False, header=not candidate_header)
                            candidate_header = True
                    if sample_events == 0:
                        raise ValueError(f"Prediction for sample {sample_name!r} produced no events.")
            finally:
                if candidate_stream is not None:
                    candidate_stream.close()
        if n_events == 0:
            raise ValueError("Prediction produced no events.")
        event_temporary.replace(output_path)
        if diagnostics:
            candidate_temporary.replace(candidate_path)
    except Exception:
        for temporary in (event_temporary, candidate_temporary):
            if temporary is not None and temporary.exists():
                temporary.unlink()
        raise
    _write_prediction_manifest(
        prediction_dir, checkpoint_path, cfg, candidate_path=candidate_path,
        topology=topology, source_paths=tuple(dict.fromkeys(source_paths)), n_events=n_events,
        label_counts=label_counts, classification_enabled=write_classification,
        reconstruction_enabled=write_reconstruction, source_type_ids=datamodule.schema.source_type_ids,
    )
    logger.info("Wrote %d compact prediction events to %s", n_events, output_path)
    if candidate_path:
        logger.info("Wrote explicit candidate diagnostics to %s", candidate_path)


def predict_sample(sample_cfg, base_cfg, model, checkpoint_path, device, datamodule=None, batch_size=None) -> pd.DataFrame:
    """Score one configured sample for callers that need an in-memory frame."""
    if datamodule is None:
        from joint_event_reconstruction.m0.data.module import TransformerDataModule
        datamodule = TransformerDataModule(base_cfg)
    sample_name = _cfg_get(sample_cfg, "name")
    sample_path = str(Path(str(_cfg_get(sample_cfg, "path"))).expanduser().resolve())
    frame, _ = _score_dataloader_outputs(
        datamodule.sample_dataloader(sample_cfg, batch_size=batch_size), model, device,
        topology=str(base_cfg.general.topology), sample_name=str(sample_name), sample_path=sample_path,
        jet_type_id=int(datamodule.schema.source_type_ids["JET"]),
        label_override=_cfg_get(sample_cfg, "label"), write_classification=True,
    )
    return frame


def predict_samples(samples, base_cfg, model, checkpoint_path, device, datamodule=None, batch_size=None) -> pd.DataFrame:
    frames = [predict_sample(sample, base_cfg, model, checkpoint_path, device, datamodule, batch_size) for sample in samples]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PREDICTION_COLUMNS)
