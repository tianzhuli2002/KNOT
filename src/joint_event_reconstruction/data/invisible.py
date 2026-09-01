"""Strict event-identity join and train-only scaling for invisible truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import h5py
import numpy as np
import torch


REQUIRED_COMPONENTS = ("px", "py", "pz")


class InvisibleTruthStore:
    def __init__(
        self,
        path: str | Path,
        role_names: tuple[str, ...],
        *,
        allow_synthetic: bool,
    ):
        self.path = Path(path).expanduser().resolve()
        self.role_names = tuple(str(name) for name in role_names)
        if not self.role_names:
            raise ValueError("Invisible truth store requires at least one topology role")
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Invisible truth sidecar does not exist: {self.path}"
            )
        with h5py.File(self.path, "r") as source:
            for key in ("EVENT_ID", "LABELS/NEUTRINO"):
                if key not in source:
                    raise KeyError(
                        f"Invisible truth sidecar {self.path} is missing {key}"
                    )
            event_ids = np.asarray(source["EVENT_ID"][:])
            values = source["LABELS/NEUTRINO"][:]
            self.synthetic = bool(source.attrs.get("synthetic", False))
            self.metadata = {key: source.attrs[key] for key in source.attrs}
            raw_roles = source.attrs.get("invisible_roles")
        if self.synthetic and not allow_synthetic:
            raise ValueError(
                "Synthetic invisible truth is forbidden unless allow_synthetic=true: "
                f"{self.path}"
            )
        if raw_roles is None:
            raise ValueError(
                f"{self.path}: missing required invisible_roles metadata; "
                "regenerate the canonical/synthetic sidecar with the current converter"
            )
        if isinstance(raw_roles, bytes):
            raw_roles = raw_roles.decode()
        try:
            observed_roles = tuple(str(name) for name in json.loads(str(raw_roles)))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{self.path}: invisible_roles must be a JSON list of topology role names"
            ) from exc
        if observed_roles != self.role_names:
            raise ValueError(
                f"{self.path}: invisible role order does not match topology; "
                f"expected={self.role_names}, observed={observed_roles}"
            )
        if event_ids.ndim != 1 or not np.issubdtype(event_ids.dtype, np.integer):
            raise TypeError(
                f"{self.path}: EVENT_ID must be a one-dimensional integer dataset"
            )
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError(f"{self.path}: EVENT_ID contains duplicates")
        role_count = len(self.role_names)
        if values.shape != (len(event_ids), role_count):
            raise ValueError(
                f"{self.path}: LABELS/NEUTRINO must have shape "
                f"{(len(event_ids), role_count)}, found {values.shape}"
            )
        if values.dtype.names != REQUIRED_COMPONENTS:
            raise TypeError(
                f"{self.path}: LABELS/NEUTRINO fields must be exactly "
                f"{REQUIRED_COMPONENTS}, found {values.dtype.names}"
            )
        if any(
            values.dtype.fields[name][0] != np.dtype("float32")
            for name in REQUIRED_COMPONENTS
        ):
            raise TypeError(f"{self.path}: px, py and pz must all be float32")
        dense = np.stack([values[name] for name in REQUIRED_COMPONENTS], axis=-1)
        if not np.isfinite(dense).all():
            raise ValueError(
                f"{self.path}: LABELS/NEUTRINO contains non-finite values"
            )
        order = np.argsort(event_ids)
        self.event_ids = event_ids[order].astype(np.int64, copy=False)
        self.values = dense[order].astype(np.float32, copy=False)

    def lookup(self, event_ids: torch.Tensor | np.ndarray) -> torch.Tensor:
        requested = np.asarray(
            event_ids.cpu() if torch.is_tensor(event_ids) else event_ids,
            dtype=np.int64,
        )
        positions = np.searchsorted(self.event_ids, requested)
        valid = positions < len(self.event_ids)
        valid &= (
            self.event_ids[np.minimum(positions, len(self.event_ids) - 1)]
            == requested
        )
        if not valid.all():
            missing = requested[~valid][:10].tolist()
            raise KeyError(
                f"Invisible truth sidecar {self.path} is missing EVENT_ID values "
                f"{missing}"
            )
        result = torch.from_numpy(self.values[positions].copy())
        if not torch.isfinite(result).all():
            raise RuntimeError(
                f"Non-finite invisible truth selected from {self.path}"
            )
        return result


@dataclass(frozen=True)
class InvisibleScaler:
    mean: tuple[tuple[float, float, float], ...]
    std: tuple[tuple[float, float, float], ...]
    role_names: tuple[str, ...]
    representation: str = "cartesian_px_py_pz_v1"
    active_mask: tuple[tuple[bool, bool, bool], ...] | None = None

    @classmethod
    def fit(
        cls,
        values: torch.Tensor,
        role_names: tuple[str, ...] | None = None,
        *,
        representation: str = "cartesian_px_py_pz_v1",
        active_mask: torch.Tensor | None = None,
    ) -> "InvisibleScaler":
        if (
            values.ndim != 3
            or values.shape[-1] != 3
            or not torch.isfinite(values).all()
        ):
            raise ValueError(
                f"Scaler input must be finite [N,Rnu,3], found {tuple(values.shape)}"
            )
        values = values.to(torch.float64)
        mean = values.mean(dim=0)
        if active_mask is None:
            active_mask = torch.ones(values.shape[1:], dtype=torch.bool)
        active_mask = torch.as_tensor(active_mask, dtype=torch.bool)
        if tuple(active_mask.shape) != tuple(values.shape[1:]):
            raise ValueError("Invisible scaler active_mask must have shape [T,3]")
        active_mask = active_mask.to(values.device)
        if not torch.equal(
            values[..., ~active_mask], torch.zeros_like(values[..., ~active_mask])
        ):
            raise ValueError("Inactive invisible-coordinate target components must be zero")
        std = values.std(dim=0, unbiased=False)
        mean = torch.where(active_mask, mean, torch.zeros_like(mean))
        std = torch.where(active_mask, std, torch.ones_like(std))
        if not torch.isfinite(std).all() or (std[active_mask] <= 0).any():
            raise ValueError(
                "Invisible training standard deviations must be finite and positive: "
                f"{std}"
            )
        names = tuple(
            role_names
            or (f"invisible[{index}]" for index in range(values.shape[1]))
        )
        if len(names) != values.shape[1]:
            raise ValueError(
                f"Invisible role-name count {len(names)} does not match "
                f"Rnu={values.shape[1]}"
            )
        return cls(
            tuple(tuple(row) for row in mean.tolist()),
            tuple(tuple(row) for row in std.tolist()),
            names,
            str(representation),
            tuple(tuple(bool(component) for component in row) for row in active_mask.tolist()),
        )

    @classmethod
    def from_dict(cls, values: dict) -> "InvisibleScaler":
        if values is None:
            raise ValueError("Invisible scaler metadata is null")
        representation = str(values.get("representation", ""))
        if representation != "cartesian_px_py_pz_v1" and not representation.startswith(
            "invisible_coordinate_state_"
        ):
            raise ValueError(
                f"Unsupported invisible scaler representation {representation!r}"
            )
        role_names = tuple(str(name) for name in values["role_names"])
        mean = tuple(
            tuple(float(component) for component in row) for row in values["mean"]
        )
        std = tuple(
            tuple(float(component) for component in row) for row in values["std"]
        )
        raw_active = values.get("active_mask")
        active_mask = (
            tuple(tuple(bool(component) for component in row) for row in raw_active)
            if raw_active is not None
            else tuple((True, True, True) for _ in role_names)
        )
        if len(mean) != len(role_names) or len(std) != len(role_names):
            raise ValueError("Invisible scaler role and statistic counts differ")
        if any(len(row) != 3 for row in (*mean, *std)):
            raise ValueError(
                "Invisible scaler statistics must contain px, py and pz for every role"
            )
        if any(component <= 0 for row in std for component in row):
            raise ValueError(
                "Invisible scaler standard deviations must be positive"
            )
        if len(active_mask) != len(role_names) or any(
            len(row) != 3 for row in active_mask
        ):
            raise ValueError("Invisible scaler active_mask must be [T,3]")
        if any(
            not np.isfinite(component)
            for row in (*mean, *std)
            for component in row
        ):
            raise ValueError("Invisible scaler statistics must be finite")
        for token, active in enumerate(active_mask):
            for component, is_active in enumerate(active):
                if not is_active and (
                    mean[token][component] != 0 or std[token][component] != 1
                ):
                    raise ValueError(
                        "Inactive invisible scaler components require mean=0 and std=1"
                    )
        return cls(
            mean=mean,
            std=std,
            role_names=role_names,
            representation=representation,
            active_mask=active_mask,
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.new_tensor(self.mean)
        std = values.new_tensor(self.std)
        transformed = (values - mean) / std
        if self.active_mask is not None:
            transformed = transformed * values.new_tensor(self.active_mask)
        return transformed

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.new_tensor(self.mean)
        std = values.new_tensor(self.std)
        result = values * std + mean
        if self.active_mask is not None:
            result = result * values.new_tensor(self.active_mask)
        return result

    def as_dict(self) -> dict:
        return {
            "representation": self.representation,
            "role_names": list(self.role_names),
            "mean": [list(row) for row in self.mean],
            "std": [list(row) for row in self.std],
            "active_mask": (
                None if self.active_mask is None else [list(row) for row in self.active_mask]
            ),
        }
