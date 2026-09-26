"""Strict event-identity join and train-only scaling for invisible truth."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import h5py
import numpy as np
import torch


REQUIRED_COMPONENTS = ("px", "py", "pz")


class ParentTruthStore:
    """Strict event-ID join for an explicit generator-parent truth sidecar.

    The sidecar contract is intentionally separate from the canonical event
    file: ``EVENT_ID`` plus ``PARENTS/<role>`` float32 datasets of shape
    ``[N,4]`` in ``E_px_py_pz_GeV`` order, with a JSON ``parent_roles`` attr.
    No reconstructed or model-derived composite is accepted as this target.
    """

    def __init__(self, path: str | Path, role_names: tuple[str, ...]):
        self.path = Path(path).expanduser().resolve()
        self.role_names = tuple(str(name) for name in role_names)
        if not self.role_names:
            raise ValueError("Parent truth store requires at least one role")
        if not self.path.is_file():
            raise FileNotFoundError(f"Parent truth sidecar does not exist: {self.path}")
        with h5py.File(self.path, "r") as source:
            if "EVENT_ID" not in source or "PARENTS" not in source:
                raise KeyError(
                    f"{self.path} must contain EVENT_ID and PARENTS/<role> datasets"
                )
            raw_roles = source.attrs.get("parent_roles")
            if isinstance(raw_roles, bytes):
                raw_roles = raw_roles.decode()
            try:
                observed_roles = tuple(str(role) for role in json.loads(str(raw_roles)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{self.path}: parent_roles must be a JSON list") from error
            if observed_roles != self.role_names:
                raise ValueError(
                    f"{self.path}: parent role order mismatch; expected={self.role_names}, observed={observed_roles}"
                )
            event_ids = np.asarray(source["EVENT_ID"][:])
            if event_ids.ndim != 1 or not np.issubdtype(event_ids.dtype, np.integer):
                raise TypeError(f"{self.path}: EVENT_ID must be one-dimensional integer")
            values = []
            for role in self.role_names:
                name = f"PARENTS/{role}"
                if name not in source:
                    raise KeyError(f"{self.path}: missing {name}")
                array = np.asarray(source[name][:])
                if array.shape != (len(event_ids), 4) or array.dtype != np.float32:
                    raise TypeError(f"{self.path}:{name} must be float32 [{len(event_ids)},4]")
                values.append(array)
        if len(np.unique(event_ids)) != len(event_ids):
            raise ValueError(f"{self.path}: EVENT_ID contains duplicates")
        dense = np.stack(values, axis=1)
        spatial_norm = np.sqrt(np.square(dense[..., 1:]).sum(axis=-1))
        if (
            not np.isfinite(dense).all()
            or (dense[..., 0] < 0).any()
            or (dense[..., 0] + 1.0e-5 < spatial_norm).any()
        ):
            raise ValueError(f"{self.path}: parent truth contains invalid four-vectors")
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
        valid &= self.event_ids[np.minimum(positions, len(self.event_ids) - 1)] == requested
        if not valid.all():
            raise KeyError(
                f"Parent truth sidecar {self.path} is missing EVENT_ID values "
                f"{requested[~valid][:10].tolist()}"
            )
        return torch.from_numpy(self.values[positions].copy())


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
            neutrino_dataset = source["LABELS/NEUTRINO"]
            values = neutrino_dataset[:]
            neutrino_metadata = {
                key: neutrino_dataset.attrs[key] for key in neutrino_dataset.attrs
            }
            support_path = "SUPPORT/INVISIBLE_ROLE"
            if support_path in source:
                support = np.asarray(source[support_path][:])
                support_policy = "explicit_role_support"
            else:
                # Existing canonical invisible sidecars were written only
                # after every role target had passed finite-value validation.
                # Keep that explicit legacy contract while new adapters write
                # SUPPORT/INVISIBLE_ROLE for populations with unsupported rows.
                support = np.ones(values.shape, dtype=np.bool_)
                support_policy = "legacy_all_roles_required"
            self.synthetic = bool(source.attrs.get("synthetic", False))
            self.metadata = {key: source.attrs[key] for key in source.attrs}
            raw_roles = source.attrs.get("invisible_roles")
            raw_role_order = source.attrs.get("invisible_role_order")
            role_convention = source.attrs.get("invisible_role_convention")
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
        if self.role_names == ("top.nu", "antitop.nu") and not self.synthetic:
            if isinstance(raw_role_order, bytes):
                raw_role_order = raw_role_order.decode()
            try:
                observed_role_order = tuple(str(name) for name in json.loads(str(raw_role_order)))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{self.path}: invisible_role_order must be a JSON list") from error
            if (
                role_convention not in ("top_antitop", b"top_antitop")
                or observed_role_order != self.role_names
                or self.metadata.get("units") not in ("GeV", b"GeV")
                or self.metadata.get("neutrino_truth_definition")
                not in (
                    "generator top-side neutrino and generator antitop-side neutrino",
                    b"generator top-side neutrino and generator antitop-side neutrino",
                )
                or self.metadata.get("synthetic") not in (False, np.bool_(False))
                or neutrino_metadata.get("invisible_role_convention")
                not in ("top_antitop", b"top_antitop")
                or neutrino_metadata.get("units") not in ("GeV", b"GeV")
                or neutrino_metadata.get("truth_definition")
                not in (
                    "generator top-side neutrino and generator antitop-side neutrino",
                    b"generator top-side neutrino and generator antitop-side neutrino",
                )
                or neutrino_metadata.get("component_order") not in ("px,py,pz", b"px,py,pz")
            ):
                raise ValueError(
                    f"{self.path}: real top/antitop neutrinos require explicit top_antitop role metadata"
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
        if support.shape != (len(event_ids), role_count) or support.dtype != np.bool_:
            raise TypeError(
                f"{self.path}:SUPPORT/INVISIBLE_ROLE must be boolean "
                f"[{len(event_ids)},{role_count}], found {support.shape} {support.dtype}"
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
        if not np.isfinite(dense[support]).all():
            raise ValueError(
                f"{self.path}: supported LABELS/NEUTRINO targets contain non-finite values"
            )
        if (
            self.role_names == ("top.nu", "antitop.nu")
            and not self.synthetic
            and not np.array_equal(support, np.isfinite(dense).all(axis=-1))
        ):
            raise ValueError(
                f"{self.path}: top/antitop support must equal finite producer-declared role truth"
            )
        order = np.argsort(event_ids)
        self.event_ids = event_ids[order].astype(np.int64, copy=False)
        self.values = dense[order].astype(np.float32, copy=False)
        self.support = support[order].astype(np.bool_, copy=False)
        self.support_policy = support_policy
        self.all_supported = bool(self.support.all())
        self.metadata["support_policy"] = support_policy

    def lookup(self, event_ids: torch.Tensor | np.ndarray) -> torch.Tensor:
        values, support = self.lookup_with_support(event_ids)
        if not bool(support.all()):
            raise ValueError(
                f"Invisible truth sidecar {self.path} has unsupported role targets; "
                "use lookup_with_support() so they cannot enter a physics loss unmasked"
            )
        return values

    def lookup_with_support(
        self, event_ids: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        values = torch.from_numpy(self.values[positions].copy())
        support = torch.from_numpy(self.support[positions].copy())
        # Unsupported values are deliberately replaced only after their mask
        # has been retained. This keeps coordinate transforms finite while
        # preventing placeholders from reaching the supported-event loss.
        values = torch.where(support.unsqueeze(-1), values, torch.zeros_like(values))
        if not torch.isfinite(values).all():
            raise RuntimeError(f"Supported invisible truth selected from {self.path} is non-finite")
        return values, support


@dataclass(frozen=True)
class InvisibleScaler:
    mean: tuple[tuple[float, ...], ...]
    std: tuple[tuple[float, ...], ...]
    role_names: tuple[str, ...]
    representation: str = "cartesian_px_py_pz_v1"
    active_mask: tuple[tuple[bool, ...], ...] | None = None

    @classmethod
    def fit(
        cls,
        values: torch.Tensor,
        role_names: tuple[str, ...] | None = None,
        *,
        representation: str = "cartesian_px_py_pz_v1",
        active_mask: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
    ) -> "InvisibleScaler":
        if (
            values.ndim != 3
            or values.shape[-1] <= 0
            or not torch.isfinite(values).all()
        ):
            raise ValueError(f"Scaler input must be finite [N,T,D], found {tuple(values.shape)}")
        values = values.to(torch.float64)
        if support_mask is None:
            support_mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
        else:
            support_mask = torch.as_tensor(support_mask, dtype=torch.bool, device=values.device)
            if tuple(support_mask.shape) != tuple(values.shape[:2]):
                raise ValueError("Invisible scaler support_mask must have shape [N,R]")
        mean_rows = []
        std_rows = []
        for role in range(values.shape[1]):
            selected = values[:, role][support_mask[:, role]]
            if not len(selected):
                raise ValueError(f"Invisible training role {role} has no supported targets")
            mean_rows.append(selected.mean(dim=0))
            std_rows.append(selected.std(dim=0, unbiased=False))
        mean = torch.stack(mean_rows)
        if active_mask is None:
            active_mask = torch.ones(values.shape[1:], dtype=torch.bool)
        active_mask = torch.as_tensor(active_mask, dtype=torch.bool)
        if tuple(active_mask.shape) != tuple(values.shape[1:]):
            raise ValueError("Invisible scaler active_mask must have shape [T,D]")
        active_mask = active_mask.to(values.device)
        if not torch.equal(
            values[..., ~active_mask], torch.zeros_like(values[..., ~active_mask])
        ):
            raise ValueError("Inactive invisible-coordinate target components must be zero")
        std = torch.stack(std_rows)
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
        if representation not in {"cartesian_px_py_pz_v1", "direct_point_cartesian_v1"} and not representation.startswith(
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
        if len(mean) != len(role_names) or len(std) != len(role_names):
            raise ValueError("Invisible scaler role and statistic counts differ")
        dimensions = {len(row) for row in (*mean, *std)}
        if len(dimensions) != 1 or not dimensions or next(iter(dimensions)) <= 0:
            raise ValueError("Invisible scaler statistics must have a consistent positive state width")
        width = next(iter(dimensions))
        active_mask = (
            tuple(tuple(bool(component) for component in row) for row in raw_active)
            if raw_active is not None
            else tuple(tuple(True for _ in range(width)) for _ in role_names)
        )
        if any(component <= 0 for row in std for component in row):
            raise ValueError(
                "Invisible scaler standard deviations must be positive"
            )
        if len(active_mask) != len(role_names) or any(
            len(row) != next(iter(dimensions)) for row in active_mask
        ):
            raise ValueError("Invisible scaler active_mask must match [T,D] statistics")
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
