"""Strict event-identity join and train-only scaling for invisible truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


REQUIRED_COMPONENTS = ("px", "py", "pz")


class InvisibleTruthStore:
    def __init__(self, path: str | Path, role_count: int, *, allow_synthetic: bool):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Invisible truth sidecar does not exist: {self.path}")
        with h5py.File(self.path, "r") as source:
            for key in ("EVENT_ID", "LABELS/NEUTRINO"):
                if key not in source:
                    raise KeyError(f"Invisible truth sidecar {self.path} is missing {key}")
            event_ids = np.asarray(source["EVENT_ID"][:])
            values = source["LABELS/NEUTRINO"][:]
            self.synthetic = bool(source.attrs.get("synthetic", False))
            self.metadata = {key: source.attrs[key] for key in source.attrs}
        if self.synthetic and not allow_synthetic:
            raise ValueError(
                f"Synthetic invisible truth is forbidden unless allow_synthetic=true: {self.path}"
            )
        if event_ids.ndim != 1 or not np.issubdtype(event_ids.dtype, np.integer):
            raise TypeError(f"{self.path}: EVENT_ID must be a one-dimensional integer dataset")
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError(f"{self.path}: EVENT_ID contains duplicates")
        if values.shape != (len(event_ids), role_count):
            raise ValueError(
                f"{self.path}: LABELS/NEUTRINO must have shape {(len(event_ids), role_count)}, "
                f"found {values.shape}"
            )
        if values.dtype.names != REQUIRED_COMPONENTS:
            raise TypeError(
                f"{self.path}: LABELS/NEUTRINO fields must be exactly {REQUIRED_COMPONENTS}, "
                f"found {values.dtype.names}"
            )
        if any(values.dtype.fields[name][0] != np.dtype("float32") for name in REQUIRED_COMPONENTS):
            raise TypeError(f"{self.path}: px, py and pz must all be float32")
        dense = np.stack([values[name] for name in REQUIRED_COMPONENTS], axis=-1)
        if not np.isfinite(dense).all():
            raise ValueError(f"{self.path}: LABELS/NEUTRINO contains non-finite values")
        order = np.argsort(event_ids)
        self.event_ids = event_ids[order].astype(np.int64, copy=False)
        self.values = dense[order].astype(np.float32, copy=False)

    def lookup(self, event_ids: torch.Tensor | np.ndarray) -> torch.Tensor:
        requested = np.asarray(event_ids.cpu() if torch.is_tensor(event_ids) else event_ids, dtype=np.int64)
        positions = np.searchsorted(self.event_ids, requested)
        valid = positions < len(self.event_ids)
        valid &= self.event_ids[np.minimum(positions, len(self.event_ids) - 1)] == requested
        if not valid.all():
            missing = requested[~valid][:10].tolist()
            raise KeyError(f"Invisible truth sidecar {self.path} is missing EVENT_ID values {missing}")
        result = torch.from_numpy(self.values[positions].copy())
        if not torch.isfinite(result).all():
            raise RuntimeError(f"Non-finite invisible truth selected from {self.path}")
        return result


@dataclass(frozen=True)
class InvisibleScaler:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    representation: str = "cartesian_px_py_pz_v1"

    @classmethod
    def fit(cls, values: torch.Tensor) -> "InvisibleScaler":
        if values.ndim != 3 or values.shape[-1] != 3 or not torch.isfinite(values).all():
            raise ValueError(f"Scaler input must be finite [N,Rnu,3], found {tuple(values.shape)}")
        flattened = values.reshape(-1, 3).to(torch.float64)
        mean = flattened.mean(dim=0)
        std = flattened.std(dim=0, unbiased=False)
        if not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError(f"Invisible training standard deviations must be finite and positive: {std}")
        return cls(tuple(mean.tolist()), tuple(std.tolist()))

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.new_tensor(self.mean)
        std = values.new_tensor(self.std)
        return (values - mean) / std

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.new_tensor(self.mean)
        std = values.new_tensor(self.std)
        return values * std + mean

    def as_dict(self) -> dict:
        return {"representation": self.representation, "mean": list(self.mean), "std": list(self.std)}
