"""Maintained invisible coordinates and their physical mass-shell adapter."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from joint_event_reconstruction.models.invisible.mass_shell import (
    InvisibleMassShell,
    build_mass_shells,
)


def helmert_contrast_basis(
    role_count: int, *, device=None, dtype=torch.float32
) -> torch.Tensor:
    if role_count <= 0:
        raise ValueError("Invisible coordinates require at least one role")
    basis = torch.zeros(role_count, max(role_count - 1, 0), device=device, dtype=dtype)
    for column in range(role_count - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


class InvisibleCoordinates:
    coordinate_type = ""
    version = "v1"

    def __init__(
        self,
        role_names: tuple[str, ...],
        mass_shells: dict[str, InvisibleMassShell] | None = None,
    ):
        self.role_names = tuple(role_names)
        if not self.role_names:
            raise ValueError("Invisible coordinates require at least one role")
        if mass_shells and all(
            isinstance(value, InvisibleMassShell) for value in mass_shells.values()
        ):
            self.mass_shells = dict(mass_shells)
            if set(self.mass_shells) != set(self.role_names):
                raise ValueError(
                    "Invisible mass-shell objects must define exactly the topology roles"
                )
        else:
            self.mass_shells = build_mass_shells(mass_shells, self.role_names)
        modes = {shell.mode for shell in self.mass_shells.values()}
        if "unconstrained" in modes and modes != {"unconstrained"}:
            raise ValueError(
                "Unconstrained invisible four-vectors cannot be mixed with exact mass shells"
            )
        if "unconstrained" in modes and self.coordinate_type not in {"", "cartesian"}:
            raise ValueError(
                "Unconstrained invisible four-vectors require Cartesian coordinates"
            )

    @property
    def role_count(self) -> int:
        return len(self.role_names)

    @property
    def state_dim(self) -> int:
        return 4 if all(shell.mode == "unconstrained" for shell in self.mass_shells.values()) else 3

    @property
    def generated_mass_roles(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.role_names if self.mass_shells[name].mode == "generated"
        )

    @property
    def base_token_count(self) -> int:
        raise NotImplementedError

    @property
    def base_token_names(self) -> tuple[str, ...]:
        raise NotImplementedError

    @property
    def base_active_mask(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    def token_count(self) -> int:
        return self.base_token_count + len(self.generated_mass_roles)

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.base_token_names + tuple(
            f"mass:{name}" for name in self.generated_mass_roles
        )

    @property
    def active_mask(self) -> torch.Tensor:
        base = self.base_active_mask
        if not self.generated_mass_roles:
            return base
        mass = torch.zeros(
            len(self.generated_mass_roles), self.state_dim, dtype=torch.bool
        )
        mass[:, 0] = True
        return torch.cat((base, mass), dim=0)

    def _validate_target(self, momenta: torch.Tensor, met_xy: torch.Tensor) -> None:
        if momenta.shape[-2:] != (self.role_count, self.state_dim):
            raise ValueError(
                f"Invisible targets must end in [{self.role_count},{self.state_dim}], "
                f"found {tuple(momenta.shape)}"
            )
        if met_xy.shape != momenta.shape[:-2] + (2,):
            raise ValueError("met_xy must align with the invisible-momentum batch axes")
        if not torch.isfinite(momenta).all() or not torch.isfinite(met_xy).all():
            raise ValueError("Invisible targets and met_xy must be finite")

    def _validate_state(self, state: torch.Tensor, met_xy: torch.Tensor) -> None:
        if state.shape[-2:] != (self.token_count, self.state_dim):
            raise ValueError(
                f"Coordinate state must end in [{self.token_count},{self.state_dim}], "
                f"found {tuple(state.shape)}"
            )
        if met_xy.shape != state.shape[:-2] + (2,):
            raise ValueError("met_xy must align with coordinate-state batch axes")
        if not torch.isfinite(state).all() or not torch.isfinite(met_xy).all():
            raise ValueError("Coordinate state and met_xy must be finite")
        active = self.active_mask.to(device=state.device)
        if not torch.equal(state[..., ~active], torch.zeros_like(state[..., ~active])):
            raise ValueError("Inactive invisible-coordinate components must be zero")

    def _encode_base(self, momenta: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _decode_base(self, state: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def encode(
        self,
        momenta: torch.Tensor,
        met_xy: torch.Tensor,
        masses: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode target values, optionally including generated physical masses."""
        self._validate_target(momenta, met_xy)
        state = self._encode_base(momenta, met_xy)
        if state.shape[-2:] != (self.base_token_count, self.state_dim):
            raise RuntimeError("Coordinate implementation returned an invalid base state")
        if not self.generated_mass_roles:
            return state
        expected_shape = momenta.shape[:-2] + (self.role_count,)
        if masses is None or masses.shape != expected_shape:
            raise ValueError(
                "Generated-mass coordinates require positive masses with shape "
                f"{tuple(expected_shape)}"
            )
        for role_name in self.generated_mass_roles:
            role_index = self.role_names.index(role_name)
            role_masses = masses[..., role_index]
            if not torch.isfinite(role_masses).all() or (role_masses <= 0).any():
                raise ValueError("Generated invisible masses must be finite and positive")
        mass_state = state.new_zeros(
            state.shape[:-2] + (len(self.generated_mass_roles), self.state_dim)
        )
        for index, role_name in enumerate(self.generated_mass_roles):
            role_index = self.role_names.index(role_name)
            mass_state[..., index, 0] = self.mass_shells[role_name].mass_to_raw(
                masses[..., role_index]
            )
        return torch.cat((state, mass_state), dim=-2)

    def decode(self, state: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        """Decode native coordinates to spatial momenta (or p4 spatial components)."""
        self._validate_state(state, met_xy)
        base = self._decode_base(state[..., : self.base_token_count, :], met_xy)
        return base[..., 1:] if self.state_dim == 4 else base

    def decode_physical(self, state: torch.Tensor, met_xy: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode native coordinates and expose a consistent physical p4."""
        self._validate_state(state, met_xy)
        base = self._decode_base(state[..., : self.base_token_count, :], met_xy)
        if self.state_dim == 4:
            four_vector = base
            masses = (
                four_vector[..., 0].square() - four_vector[..., 1:].square().sum(dim=-1)
            ).clamp_min(0).sqrt()
            return {
                "four_vector": four_vector,
                "momenta": four_vector[..., 1:],
                "mass": masses,
            }

        masses = state.new_empty(state.shape[:-2] + (self.role_count,))
        four_vectors = []
        for role_index, role_name in enumerate(self.role_names):
            shell = self.mass_shells[role_name]
            if shell.mode == "fixed":
                mass = state.new_full(base.shape[:-2] + (self.role_count,), float(shell.value))[..., role_index]
            elif shell.mode == "generated":
                mass_token = self.base_token_count + self.generated_mass_roles.index(role_name)
                mass = shell.raw_to_mass(state[..., mass_token, 0])
            else:
                raise RuntimeError("Unconstrained shell reached the spatial-coordinate path")
            masses[..., role_index] = mass
            four_vectors.append(
                torch.cat(
                    (
                        (base[..., role_index, :].square().sum(dim=-1) + mass.square())
                        .clamp_min(0)
                        .sqrt()
                        .unsqueeze(-1),
                        base[..., role_index, :],
                    ),
                    dim=-1,
                )
            )
        return {
            "four_vector": torch.stack(four_vectors, dim=-2),
            "momenta": base,
            "mass": masses,
        }

    def projection_diagnostics(
        self, momenta: torch.Tensor, met_xy: torch.Tensor
    ) -> dict[str, torch.Tensor] | None:
        del momenta, met_xy
        return None

    def metadata(self) -> dict:
        return {
            "type": self.coordinate_type,
            "version": self.version,
            "role_ordering": list(self.role_names),
            "state_token_ordering": list(self.token_names),
            "active_mask": self.active_mask.tolist(),
            "state_dim": self.state_dim,
            "mass_shells": {
                name: self.mass_shells[name].metadata() for name in self.role_names
            },
        }


class CartesianCoordinates(InvisibleCoordinates):
    coordinate_type = "cartesian"

    @property
    def base_token_count(self) -> int:
        return self.role_count

    @property
    def base_token_names(self) -> tuple[str, ...]:
        return self.role_names

    @property
    def base_active_mask(self) -> torch.Tensor:
        return torch.ones(self.base_token_count, self.state_dim, dtype=torch.bool)

    def _encode_base(self, momenta: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        del met_xy
        return momenta

    def _decode_base(self, state: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        del met_xy
        return state

    def parent_context(self, role_context: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        del pooled
        return role_context


class _ContrastCoordinates(InvisibleCoordinates):
    include_residual = False

    @property
    def base_token_count(self) -> int:
        return 2 * self.role_count - (0 if self.include_residual else 1)

    @property
    def base_token_names(self) -> tuple[str, ...]:
        names = [f"pz:{name}" for name in self.role_names]
        names.extend(f"transverse_contrast:{index}" for index in range(self.role_count - 1))
        if self.include_residual:
            names.append("met_residual")
        return tuple(names)

    @property
    def base_active_mask(self) -> torch.Tensor:
        if self.state_dim != 3:
            raise ValueError("MET contrast coordinates require three spatial components")
        mask = torch.zeros(self.base_token_count, self.state_dim, dtype=torch.bool)
        mask[: self.role_count, 0] = True
        mask[self.role_count :, :2] = True
        return mask

    def _basis(self, reference: torch.Tensor) -> torch.Tensor:
        return helmert_contrast_basis(
            self.role_count, device=reference.device, dtype=reference.dtype
        )

    def _encode_base(self, momenta: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        state = momenta.new_zeros(momenta.shape[:-2] + (self.base_token_count, self.state_dim))
        state[..., : self.role_count, 0] = momenta[..., :, 2]
        basis = self._basis(momenta)
        contrasts = torch.matmul(momenta[..., :, :2].transpose(-2, -1), basis)
        state[..., self.role_count : self.role_count + self.role_count - 1, :2] = (
            contrasts.transpose(-2, -1)
        )
        if self.include_residual:
            state[..., -1, :2] = met_xy - momenta[..., :, :2].sum(dim=-2)
        return state

    def _decode_base(self, state: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        transverse_sum = met_xy
        if self.include_residual:
            transverse_sum = transverse_sum - state[..., -1, :2]
        mean = transverse_sum / self.role_count
        contrasts = state[
            ..., self.role_count : self.role_count + self.role_count - 1, :2
        ]
        basis = self._basis(state)
        transverse = mean.unsqueeze(-2) + torch.matmul(basis, contrasts)
        momenta = state.new_empty(state.shape[:-2] + (self.role_count, 3))
        momenta[..., :, :2] = transverse
        momenta[..., :, 2] = state[..., : self.role_count, 0]
        return momenta

    def parent_context(self, role_context: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        if role_context.shape[-2] != self.role_count:
            raise ValueError("Parent context role axis does not match coordinates")
        pieces = [role_context]
        if self.role_count > 1:
            mean = role_context.mean(dim=-2, keepdim=True)
            pieces.append(mean.expand(*mean.shape[:-2], self.role_count - 1, mean.shape[-1]))
        if self.include_residual:
            pieces.append(pooled.unsqueeze(-2))
        return torch.cat(pieces, dim=-2)

    def metadata(self) -> dict:
        result = super().metadata()
        result.update(
            {
                "contrast_basis": helmert_contrast_basis(self.role_count).tolist(),
                "contrast_basis_convention": "helmert_orthonormal_v1",
            }
        )
        return result


class MetResidualCoordinates(_ContrastCoordinates):
    coordinate_type = "met_residual"
    include_residual = True


class MetExactCoordinates(_ContrastCoordinates):
    coordinate_type = "met_exact"

    def projection_diagnostics(
        self, momenta: torch.Tensor, met_xy: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self._validate_target(momenta, met_xy)
        shift = (met_xy - momenta[..., :, :2].sum(dim=-2)) / self.role_count
        return {
            "truth_projected_px_shift": shift[..., 0],
            "truth_projected_py_shift": shift[..., 1],
            # The projection applies the same transverse shift to every
            # invisible role.  Keep the role axis explicit in the diagnostic
            # so consumers can validate one value per generated invisible
            # object rather than accidentally treating an event-level value
            # as a role vector.
            "per_role_transverse_projection_magnitude": torch.linalg.vector_norm(
                shift, dim=-1, keepdim=True
            ).expand(*shift.shape[:-1], self.role_count),
        }


def build_invisible_coordinates(
    config,
    role_names: tuple[str, ...],
    mass_shell=None,
) -> InvisibleCoordinates:
    coordinate_type = str(
        config["type"] if isinstance(config, Mapping) else config.type
    )
    if mass_shell is None:
        mass_shell = (
            config.get("mass_shell")
            if isinstance(config, Mapping)
            else getattr(config, "mass_shell", None)
        )
    shells = build_mass_shells(mass_shell, tuple(role_names))
    if coordinate_type == "cartesian":
        return CartesianCoordinates(role_names, shells)
    if coordinate_type == "met_residual":
        return MetResidualCoordinates(role_names, shells)
    if coordinate_type == "met_exact":
        return MetExactCoordinates(role_names, shells)
    raise ValueError(f"Unknown invisible coordinate type {coordinate_type!r}")


def met_xy_from_batch(batch: dict, schema) -> torch.Tensor:
    met_type = schema.source_type_ids.get("MET")
    if met_type is None:
        raise KeyError("Invisible coordinates require a MET source in the token schema")
    matches = (batch["type_ids"] == int(met_type)) & ~batch["padding_mask"]
    counts = matches.sum(dim=1)
    if not torch.equal(counts, torch.ones_like(counts)):
        raise ValueError("Every event must contain exactly one unpadded MET token")
    rows, columns = matches.nonzero(as_tuple=True)
    if not torch.equal(rows, torch.arange(len(matches), device=rows.device)):
        raise RuntimeError("MET token selection did not preserve event ordering")
    met_xy = batch["physics_vectors"][rows, columns, 1:3]
    if not torch.isfinite(met_xy).all():
        raise ValueError("Physical MET px/py contains non-finite values")
    return met_xy
