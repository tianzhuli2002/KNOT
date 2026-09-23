"""Strict event-ID joins and train-only scaling for explicit partonic truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from joint_event_reconstruction.models.partonic import (
    four_vectors_from_partonic_features,
    partonic_features,
)


class PartonicTruthStore:
    """Read an explicit ``PARTONIC/<role>`` four-vector sidecar.

    The sidecar is intentionally independent of detector-level composites. It
    contains EVENT_ID, float32 physical E-px-py-pz vectors, a boolean SUPPORT
    matrix, role ordering, and semantic truth-binding metadata.
    """

    def __init__(
        self,
        path: str | Path,
        role_names: tuple[str, ...],
        truth_bindings: dict[str, str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.role_names = tuple(str(role) for role in role_names)
        if not self.role_names or len(set(self.role_names)) != len(self.role_names):
            raise ValueError("Partonic truth requires unique non-empty role names")
        if not self.path.is_file():
            raise FileNotFoundError(f"Partonic truth sidecar does not exist: {self.path}")
        with h5py.File(self.path, "r") as source:
            if "EVENT_ID" not in source or "PARTONIC" not in source:
                raise KeyError(f"{self.path} must contain EVENT_ID and PARTONIC/<role>")
            raw_roles = source.attrs.get("partonic_roles")
            if isinstance(raw_roles, bytes):
                raw_roles = raw_roles.decode()
            try:
                observed_roles = tuple(str(role) for role in json.loads(str(raw_roles)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{self.path}: partonic_roles must be a JSON list") from error
            if observed_roles != self.role_names:
                raise ValueError(
                    f"{self.path}: partonic role order mismatch; expected={self.role_names}, observed={observed_roles}"
                )
            raw_bindings = source.attrs.get("partonic_truth_bindings", "{}")
            if isinstance(raw_bindings, bytes):
                raw_bindings = raw_bindings.decode()
            try:
                observed_bindings = {str(key): str(value) for key, value in json.loads(str(raw_bindings)).items()}
            except (TypeError, ValueError, json.JSONDecodeError, AttributeError) as error:
                raise ValueError(f"{self.path}: partonic_truth_bindings must be a JSON object") from error
            if truth_bindings is not None and observed_bindings != {
                str(key): str(value) for key, value in truth_bindings.items()
            }:
                raise ValueError(
                    f"{self.path}: partonic truth-binding mismatch; expected={truth_bindings}, observed={observed_bindings}"
                )
            event_ids = np.asarray(source["EVENT_ID"][:])
            if event_ids.ndim != 1 or not np.issubdtype(event_ids.dtype, np.integer):
                raise TypeError(f"{self.path}: EVENT_ID must be a one-dimensional integer array")
            values = []
            for role in self.role_names:
                name = f"PARTONIC/{role}"
                if name not in source:
                    raise KeyError(f"{self.path}: missing {name}")
                array = np.asarray(source[name][:])
                if array.shape != (len(event_ids), 4) or array.dtype != np.float32:
                    raise TypeError(f"{self.path}:{name} must be float32 [{len(event_ids)},4]")
                values.append(array)
            if "SUPPORT" not in source["PARTONIC"]:
                raise KeyError(f"{self.path}: missing PARTONIC/SUPPORT")
            support = np.asarray(source["PARTONIC/SUPPORT"][:])
            if support.shape != (len(event_ids), len(self.role_names)) or support.dtype != np.bool_:
                raise TypeError(f"{self.path}: PARTONIC/SUPPORT must be boolean [{len(event_ids)},{len(self.role_names)}]")
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError(f"{self.path}: EVENT_ID contains duplicates")
        dense = np.stack(values, axis=1)
        if not np.isfinite(dense).all():
            raise ValueError(f"{self.path}: partonic truth contains non-finite values")
        spatial_norm = np.sqrt(np.square(dense[..., 1:]).sum(axis=-1))
        if (dense[..., 0] < 0).any() or (dense[..., 0] + 1.0e-4 < spatial_norm).any():
            raise ValueError(f"{self.path}: partonic truth contains invalid four-vectors")
        if not np.isfinite(dense[~support]).all():
            raise ValueError(f"{self.path}: unsupported partonic targets must still be finite placeholders")
        order = np.argsort(event_ids)
        self.event_ids = event_ids[order].astype(np.int64, copy=False)
        self.values = dense[order].astype(np.float32, copy=False)
        self.support = support[order].astype(np.bool_, copy=False)
        self.truth_bindings = observed_bindings
        self.synthetic = bool(False)
        with h5py.File(self.path, "r") as source:
            self.synthetic = bool(source.attrs.get("synthetic", False))

    def lookup(self, event_ids: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        requested = np.asarray(event_ids.cpu() if torch.is_tensor(event_ids) else event_ids, dtype=np.int64)
        positions = np.searchsorted(self.event_ids, requested)
        valid = positions < len(self.event_ids)
        valid &= self.event_ids[np.minimum(positions, len(self.event_ids) - 1)] == requested
        if not valid.all():
            raise KeyError(f"Partonic truth sidecar {self.path} is missing EVENT_ID values {requested[~valid][:10].tolist()}")
        return (
            torch.from_numpy(self.values[positions].copy()),
            torch.from_numpy(self.support[positions].copy()),
        )


@dataclass(frozen=True)
class PartonicScaler:
    mean: tuple[tuple[float, ...], ...]
    std: tuple[tuple[float, ...], ...]
    role_names: tuple[str, ...]
    representation: str = "partonic_px_py_pz_log1p_mass_v1"

    @classmethod
    def fit(
        cls,
        values: torch.Tensor,
        support: torch.Tensor,
        role_names: tuple[str, ...],
    ) -> "PartonicScaler":
        if values.ndim != 3 or values.shape[-1] != 4 or support.shape != values.shape[:2]:
            raise ValueError("Partonic scaler expects values [N,P,4] and support [N,P]")
        if not support.any():
            raise ValueError("Partonic scaler has no supported training targets")
        features = partonic_features(values).to(torch.float64)
        means = []
        stds = []
        for role in range(values.shape[1]):
            selected = features[:, role][support[:, role]]
            if not len(selected):
                raise ValueError(f"Partonic target role {role_names[role]!r} has no supported training entries")
            mean = selected.mean(dim=0)
            std = selected.std(dim=0, unbiased=False)
            if not torch.isfinite(std).all() or (std <= 0).any():
                raise ValueError(f"Partonic target role {role_names[role]!r} has non-positive training scale: {std}")
            means.append(mean)
            stds.append(std)
        return cls(
            tuple(tuple(float(value) for value in row) for row in torch.stack(means).tolist()),
            tuple(tuple(float(value) for value in row) for row in torch.stack(stds).tolist()),
            tuple(str(role) for role in role_names),
        )

    def as_dict(self) -> dict:
        return {
            "mean": [list(row) for row in self.mean],
            "std": [list(row) for row in self.std],
            "role_names": list(self.role_names),
            "representation": self.representation,
        }

    @classmethod
    def from_dict(cls, values: dict) -> "PartonicScaler":
        if values is None or values.get("representation") != "partonic_px_py_pz_log1p_mass_v1":
            raise ValueError("Unsupported or missing partonic scaler representation")
        return cls(
            tuple(tuple(float(value) for value in row) for row in values["mean"]),
            tuple(tuple(float(value) for value in row) for row in values["std"]),
            tuple(str(value) for value in values["role_names"]),
            str(values["representation"]),
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        features = partonic_features(values)
        mean = features.new_tensor(self.mean)
        std = features.new_tensor(self.std)
        return (features - mean) / std

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.new_tensor(self.mean)
        std = values.new_tensor(self.std)
        return four_vectors_from_partonic_features(values * std + mean)


__all__ = ["PartonicScaler", "PartonicTruthStore"]
