"""Exact transformer event-subset and split persistence."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def index_hash(values):
    values = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def source_identity(path, total_events):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Transformer split source dataset does not exist: {path}")
    stat = path.stat()
    description = {
        "path": str(path), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "total_events": int(total_events),
    }
    encoded = json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    description["fingerprint"] = hashlib.sha256(encoded).hexdigest()
    return description


def split_paths(indices_path):
    path = Path(indices_path).expanduser().resolve()
    if path.suffix == ".npz":
        return path, path.with_name("split_metadata.json")
    return path / "split_indices.npz", path / "split_metadata.json"


def generate_indices(labels, *, seed, limit_events, test_size):
    labels = np.asarray(labels)
    total = len(labels)
    selected = np.arange(total, dtype=np.int64)
    selected_labels = labels
    if limit_events is not None:
        limit = int(limit_events)
        if limit <= 0:
            raise ValueError(f"dataset.limit_events must be positive, found {limit}.")
        if limit < total:
            positions, _ = train_test_split(
                np.arange(total, dtype=np.int64), train_size=limit,
                random_state=int(seed), shuffle=True,
                stratify=labels if len(np.unique(labels)) > 1 else None,
            )
            selected = selected[positions]
            selected_labels = selected_labels[positions]
    positions = np.arange(len(selected), dtype=np.int64)
    def stratify_or_none(values):
        _, counts = np.unique(values, return_counts=True)
        return values if len(counts) >= 2 and counts.min() >= 2 else None

    stratify = stratify_or_none(selected_labels)
    train_pos, temporary_pos = train_test_split(
        positions, test_size=float(test_size), random_state=int(seed), shuffle=True, stratify=stratify,
    )
    temporary_labels = selected_labels[temporary_pos]
    temporary_stratify = stratify_or_none(temporary_labels)
    validation_pos, test_pos = train_test_split(
        temporary_pos, test_size=0.5, random_state=int(seed), shuffle=True,
        stratify=temporary_stratify,
    )
    return {
        "selected": selected,
        "train": selected[train_pos],
        "validation": selected[validation_pos],
        "test": selected[test_pos],
    }


def validate_indices(arrays, *, total_events):
    required = {"selected", "train", "validation", "test"}
    if set(arrays) != required:
        raise ValueError(f"Split arrays must be exactly {sorted(required)}, found {sorted(arrays)}.")
    clean = {}
    for name, values in arrays.items():
        raw = np.asarray(values)
        if raw.ndim != 1:
            raise ValueError(f"Transformer {name} split indices must be one-dimensional.")
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f"Transformer {name} split indices must have an integer dtype, found {raw.dtype}.")
        clean[name] = np.asarray(raw, dtype=np.int64)
    for name, values in clean.items():
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Transformer {name} split contains duplicate event indices.")
        if len(values) and (values.min() < 0 or values.max() >= int(total_events)):
            raise ValueError(f"Transformer {name} split contains an out-of-range event index for {total_events} events.")
    split_names = ("train", "validation", "test")
    for index, first in enumerate(split_names):
        for second in split_names[index + 1:]:
            overlap = np.intersect1d(clean[first], clean[second])
            if len(overlap):
                raise ValueError(f"Transformer {first}/{second} splits overlap at event {int(overlap[0])}.")
    combined = np.concatenate([clean[name] for name in split_names])
    selected = clean["selected"]
    if len(combined) != len(selected):
        raise ValueError(
            "Transformer train/validation/test sizes do not close to the selected event subset."
        )
    if not np.array_equal(np.sort(combined), np.sort(selected)):
        raise ValueError(
            "Transformer train/validation/test indices do not close to the selected event subset."
        )
    return clean


def _atomic_json(path, payload):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def save_split(indices_path, arrays, metadata):
    npz_path, metadata_path = split_paths(indices_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = validate_indices(arrays, total_events=metadata["source"]["total_events"])
    hashes = {name: index_hash(values) for name, values in arrays.items()}
    metadata = {**metadata, "index_hashes": hashes}
    fd, temporary = tempfile.mkstemp(prefix=f".{npz_path.name}.", suffix=".npz", dir=npz_path.parent)
    os.close(fd)
    try:
        np.savez(temporary, **arrays)
        os.replace(temporary, npz_path)
    except Exception:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise
    _atomic_json(metadata_path, metadata)
    return metadata


def load_split(indices_path, *, source, seed, limit_events, test_size):
    npz_path, metadata_path = split_paths(indices_path)
    if not npz_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Existing transformer split requires both {npz_path} and {metadata_path}.")
    metadata = json.loads(metadata_path.read_text())
    checks = {
        "source fingerprint": (metadata.get("source", {}).get("fingerprint"), source["fingerprint"]),
        "total event count": (metadata.get("source", {}).get("total_events"), source["total_events"]),
        "seed": (metadata.get("seed"), int(seed)),
        "requested event limit": (metadata.get("limit_events"), None if limit_events is None else int(limit_events)),
        "test size": (metadata.get("test_size"), float(test_size)),
    }
    for label, (stored, expected) in checks.items():
        if stored != expected:
            raise ValueError(f"Transformer split {label} mismatch: stored={stored!r}, current={expected!r}.")
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {name: payload[name].astype(np.int64, copy=False) for name in payload.files}
    arrays = validate_indices(arrays, total_events=source["total_events"])
    actual_hashes = {name: index_hash(values) for name, values in arrays.items()}
    if actual_hashes != metadata.get("index_hashes"):
        raise ValueError(f"Transformer split index hash mismatch in {npz_path}.")
    return arrays, metadata


def load_or_create_split(indices_path, labels, *, source_path, seed, limit_events, test_size, require_existing=False):
    source = source_identity(source_path, len(labels))
    if indices_path is None:
        if require_existing:
            raise ValueError("dataset.split.require_existing=true requires dataset.split.indices_path.")
        arrays = generate_indices(labels, seed=seed, limit_events=limit_events, test_size=test_size)
        return arrays, {"source": source, "seed": int(seed), "limit_events": limit_events, "test_size": float(test_size), "index_hashes": {name: index_hash(values) for name, values in arrays.items()}}
    npz_path, metadata_path = split_paths(indices_path)
    if npz_path.exists() or metadata_path.exists():
        return load_split(indices_path, source=source, seed=seed, limit_events=limit_events, test_size=test_size)
    if require_existing:
        raise FileNotFoundError(f"Required transformer split does not exist at {npz_path}.")
    arrays = generate_indices(labels, seed=seed, limit_events=limit_events, test_size=test_size)
    metadata = save_split(indices_path, arrays, {
        "source": source, "seed": int(seed),
        "limit_events": None if limit_events is None else int(limit_events), "test_size": float(test_size),
    })
    return arrays, metadata
