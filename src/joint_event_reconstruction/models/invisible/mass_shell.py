"""Configurable physical four-vector construction for invisible particles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


def _value(config, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass(frozen=True)
class InvisibleMassShell:
    """Map native generated coordinates to physical ``(E,px,py,pz)``.

    ``fixed`` and ``generated`` modes use spatial momentum coordinates and impose
    an exact positive mass shell.  ``unconstrained`` mode uses a full physical
    four-vector and deliberately performs no projection.
    """

    mode: str = "fixed"
    value: float | None = 0.0
    parameterization: str | None = None
    enforcement: str = "exact"

    def __post_init__(self) -> None:
        mode = str(self.mode)
        enforcement = str(self.enforcement)
        if mode not in {"fixed", "generated", "unconstrained"}:
            raise ValueError(f"Unknown invisible mass mode {mode!r}")
        if mode == "unconstrained":
            if enforcement not in {"unconstrained", "learned"}:
                raise ValueError(
                    "Unconstrained invisible four-vectors require enforcement="
                    "unconstrained or learned"
                )
            if self.value is not None or self.parameterization is not None:
                raise ValueError(
                    "Unconstrained invisible four-vectors must not define mass value "
                    "or parameterization"
                )
            return
        if enforcement != "exact":
            raise ValueError(
                f"Invisible mass mode {mode!r} requires enforcement=exact"
            )
        if mode == "fixed":
            if self.value is None or not math.isfinite(float(self.value)):
                raise ValueError("Fixed invisible mass must be finite")
            if float(self.value) < 0:
                raise ValueError("Fixed invisible mass must be non-negative")
            if self.parameterization is not None:
                raise ValueError("Fixed invisible mass must not define parameterization")
        else:
            if self.value is not None:
                raise ValueError("Generated invisible mass must not define a fixed value")
            if self.parameterization not in {"log_mass", "softplus"}:
                raise ValueError(
                    "Generated invisible mass parameterization must be log_mass or softplus"
                )

    @property
    def latent_dim(self) -> int:
        return 3 if self.mode == "fixed" else 4

    @property
    def exact(self) -> bool:
        return self.enforcement == "exact"

    def raw_to_mass(self, raw_mass: torch.Tensor) -> torch.Tensor:
        """Convert a generated mass coordinate to a strictly positive mass."""
        if self.mode != "generated":
            raise ValueError("raw_to_mass is only defined for generated mass")
        floor = torch.finfo(raw_mass.dtype).tiny
        if self.parameterization == "log_mass":
            return raw_mass.exp().clamp_min(floor)
        return F.softplus(raw_mass) + floor

    def mass_to_raw(self, mass: torch.Tensor) -> torch.Tensor:
        """Return the native positive-mass coordinate for a physical mass."""
        if self.mode != "generated":
            raise ValueError("mass_to_raw is only defined for generated mass")
        if not torch.isfinite(mass).all() or (mass <= 0).any():
            raise ValueError("Generated invisible masses must be finite and positive")
        if self.parameterization == "log_mass":
            return mass.log()
        # log(expm1(m)) is the inverse softplus.  This branch avoids overflow for
        # the large-mass end of the physically useful range.
        return mass + torch.log(-torch.expm1(-mass))

    def to_four_vector(
        self,
        latent: torch.Tensor,
        *,
        generated_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert native coordinates to a physical ``(E,px,py,pz)`` tensor."""
        if not torch.is_tensor(latent) or not torch.isfinite(latent).all():
            raise ValueError("Invisible mass-shell coordinates must be finite tensors")
        if self.mode == "unconstrained":
            if latent.shape[-1] != 4:
                raise ValueError("Unconstrained invisible coordinates must end in four-vector [E,px,py,pz]")
            return latent
        if self.mode == "fixed":
            if latent.shape[-1] != 3:
                raise ValueError("Fixed-mass invisible coordinates must end in [px,py,pz]")
            mass = latent.new_tensor(float(self.value))
            spatial = latent
        else:
            if latent.shape[-1] == 4:
                spatial = latent[..., :3]
                mass = self.raw_to_mass(latent[..., 3])
            elif latent.shape[-1] == 3 and generated_mass is not None:
                spatial = latent
                if not torch.isfinite(generated_mass).all() or (generated_mass <= 0).any():
                    raise ValueError("Generated invisible masses must be finite and positive")
                mass = generated_mass
            else:
                raise ValueError(
                    "Generated-mass coordinates must end in [px,py,pz,raw_mass]"
                )
        energy = (spatial.square().sum(dim=-1) + mass.square()).clamp_min(0).sqrt()
        return torch.cat((energy.unsqueeze(-1), spatial), dim=-1)

    def from_four_vector(self, four_vector: torch.Tensor) -> torch.Tensor:
        """Encode a physical four-vector in this shell's native coordinates."""
        if four_vector.shape[-1] != 4 or not torch.isfinite(four_vector).all():
            raise ValueError("Invisible four-vectors must end in [E,px,py,pz]")
        spatial = four_vector[..., 1:]
        if self.mode == "unconstrained":
            return four_vector
        if (four_vector[..., 0] < 0).any():
            raise ValueError("Exact invisible mass-shell vectors require non-negative energy")
        mass2 = four_vector[..., 0].square() - spatial.square().sum(dim=-1)
        if self.mode == "fixed":
            expected = four_vector.new_tensor(float(self.value)).square()
            if not torch.allclose(mass2, expected, rtol=2.0e-4, atol=2.0e-4):
                raise ValueError("Four-vector does not lie on the configured fixed mass shell")
            return spatial
        mass = mass2.clamp_min(0).sqrt()
        if (mass <= 0).any():
            raise ValueError("Generated invisible mass must be strictly positive")
        return torch.cat((spatial, self.mass_to_raw(mass).unsqueeze(-1)), dim=-1)

    def metadata(self) -> dict:
        return {
            "mode": self.mode,
            "value": self.value,
            "parameterization": self.parameterization,
            "enforcement": self.enforcement,
            "latent_dim": self.latent_dim,
            "physical_representation": "E_px_py_pz",
        }


def _normalise_shell(config) -> InvisibleMassShell:
    if config is None:
        return InvisibleMassShell()
    enforcement = _value(config, "enforcement", None)
    mass_config = _value(config, "mass", None)
    if mass_config is None and enforcement in {"learned", "unconstrained"}:
        mode = "unconstrained"
        mass_config = {}
    else:
        mass_config = config if mass_config is None else mass_config
        mode = _value(mass_config, "mode", "fixed")
    if enforcement is None:
        enforcement = "unconstrained" if str(mode) == "unconstrained" else "exact"
    value = _value(mass_config, "value", None)
    if str(mode) == "fixed" and value is None:
        value = 0.0
    parameterization = _value(mass_config, "parameterization", None)
    return InvisibleMassShell(
        mode=str(mode),
        value=None if value is None else float(value),
        parameterization=None if parameterization is None else str(parameterization),
        enforcement=str(enforcement),
    )


def build_mass_shells(config, role_names: tuple[str, ...]) -> dict[str, InvisibleMassShell]:
    """Build one explicit mass-shell policy for every topology invisible role."""
    names = tuple(str(name) for name in role_names)
    if not names:
        raise ValueError("Invisible mass-shell configuration requires at least one role")
    if config is None:
        shell = InvisibleMassShell()
        return {name: shell for name in names}
    roles = _value(config, "roles", {}) or {}
    if not isinstance(roles, Mapping):
        raise ValueError("model.invisible.mass_shell.roles must be a mapping")
    unknown = sorted(set(roles) - set(names))
    if unknown:
        raise ValueError(f"Mass-shell configuration names unknown invisible roles: {unknown}")
    return {
        name: _normalise_shell(roles[name] if name in roles else config)
        for name in names
    }


def mass_shell_config(config=None) -> dict:
    """Return the canonical public representation used by resolved configs."""
    if config is None:
        return {
            "enforcement": "exact",
            "mass": {"mode": "fixed", "value": 0.0, "parameterization": None},
            "roles": {},
        }
    shell = _normalise_shell(config)
    roles = _value(config, "roles", {}) or {}
    return {
        "enforcement": shell.enforcement,
        "mass": {
            "mode": shell.mode,
            "value": shell.value,
            "parameterization": shell.parameterization,
        },
        "roles": {
            str(name): mass_shell_config(value)
            for name, value in roles.items()
        },
    }
