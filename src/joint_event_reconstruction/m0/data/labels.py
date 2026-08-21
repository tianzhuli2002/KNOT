"""Public classification-label access for Transformer event splits."""

from pathlib import Path

import h5py
import numpy as np


LABEL_PATH_CANDIDATES = ("/LABELS/GLOBAL", "/LABELS/CLASS")


def resolve_transformer_label_path(h5_file_or_path) -> str:
    """Return the first maintained binary-classification label dataset path."""
    if isinstance(h5_file_or_path, (str, Path)):
        with h5py.File(str(h5_file_or_path), "r") as handle:
            return resolve_transformer_label_path(handle)
    for path in LABEL_PATH_CANDIDATES:
        if path in h5_file_or_path:
            return path
    raise KeyError(
        "H5 file is missing all supported Transformer classification label paths: "
        f"{list(LABEL_PATH_CANDIDATES)}"
    )


def validate_transformer_classification_labels(labels, *, require_binary=True) -> np.ndarray:
    """Validate and return one-dimensional integer classification labels."""
    values = np.asarray(labels)
    if values.dtype.names:
        field = "signal" if "signal" in values.dtype.names else values.dtype.names[0]
        values = values[field]
    values = np.asarray(values).reshape(-1)
    try:
        numeric = values.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Transformer classification labels must be numeric.") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("Transformer classification labels must be finite.")
    if not np.equal(numeric, np.rint(numeric)).all():
        raise ValueError("Transformer classification labels must be integral.")
    integer = numeric.astype(np.int64)
    if require_binary and not np.isin(integer, [0, 1]).all():
        raise ValueError(
            "Transformer classification labels must be binary {0, 1}; "
            f"found {np.unique(integer).tolist()}.")
    return integer


def read_transformer_classification_labels(h5_file_or_path, selection=slice(None)) -> np.ndarray:
    """Read only the maintained classification label dataset in event order."""
    if isinstance(h5_file_or_path, (str, Path)):
        with h5py.File(str(h5_file_or_path), "r") as handle:
            return read_transformer_classification_labels(handle, selection=selection)
    path = resolve_transformer_label_path(h5_file_or_path)
    return validate_transformer_classification_labels(
        h5_file_or_path[path][selection], require_binary=True
    )
