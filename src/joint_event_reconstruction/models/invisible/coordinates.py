"""The three maintained invisible-momentum coordinate systems."""

from __future__ import annotations

import math

import torch


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

    def __init__(self, role_names: tuple[str, ...]):
        self.role_names = tuple(role_names)
        if not self.role_names:
            raise ValueError("Invisible coordinates require at least one role")

    @property
    def role_count(self) -> int:
        return len(self.role_names)

    def _validate(self, momenta: torch.Tensor, met_xy: torch.Tensor) -> None:
        if momenta.shape[-2:] != (self.role_count, 3):
            raise ValueError(
                f"Invisible momenta must end in [{self.role_count},3], "
                f"found {tuple(momenta.shape)}"
            )
        if met_xy.shape != momenta.shape[:-2] + (2,):
            raise ValueError("met_xy must align with the invisible-momentum batch axes")
        if not torch.isfinite(momenta).all() or not torch.isfinite(met_xy).all():
            raise ValueError("Invisible momenta and met_xy must be finite")

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
        }


class CartesianCoordinates(InvisibleCoordinates):
    coordinate_type = "cartesian"

    @property
    def token_count(self) -> int:
        return self.role_count

    @property
    def token_names(self) -> tuple[str, ...]:
        return self.role_names

    @property
    def active_mask(self) -> torch.Tensor:
        return torch.ones(self.token_count, 3, dtype=torch.bool)

    def encode(self, momenta: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        self._validate(momenta, met_xy)
        return momenta

    def decode(self, state: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        self._validate(state, met_xy)
        return state

    def parent_context(self, role_context: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        del pooled
        return role_context


class _ContrastCoordinates(InvisibleCoordinates):
    include_residual = False

    @property
    def token_count(self) -> int:
        return 2 * self.role_count - (0 if self.include_residual else 1)

    @property
    def token_names(self) -> tuple[str, ...]:
        names = [f"pz:{name}" for name in self.role_names]
        names.extend(f"transverse_contrast:{index}" for index in range(self.role_count - 1))
        if self.include_residual:
            names.append("met_residual")
        return tuple(names)

    @property
    def active_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.token_count, 3, dtype=torch.bool)
        mask[: self.role_count, 0] = True
        mask[self.role_count :, :2] = True
        return mask

    def _basis(self, reference: torch.Tensor) -> torch.Tensor:
        return helmert_contrast_basis(
            self.role_count, device=reference.device, dtype=reference.dtype
        )

    def encode(self, momenta: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        self._validate(momenta, met_xy)
        state = momenta.new_zeros(momenta.shape[:-2] + (self.token_count, 3))
        state[..., : self.role_count, 0] = momenta[..., :, 2]
        basis = self._basis(momenta)
        contrasts = torch.matmul(momenta[..., :, :2].transpose(-2, -1), basis)
        state[..., self.role_count : self.role_count + self.role_count - 1, :2] = (
            contrasts.transpose(-2, -1)
        )
        if self.include_residual:
            state[..., -1, :2] = met_xy - momenta[..., :, :2].sum(dim=-2)
        return state

    def decode(self, state: torch.Tensor, met_xy: torch.Tensor) -> torch.Tensor:
        if state.shape[-2:] != (self.token_count, 3):
            raise ValueError(
                f"Coordinate state must end in [{self.token_count},3], "
                f"found {tuple(state.shape)}"
            )
        if met_xy.shape != state.shape[:-2] + (2,):
            raise ValueError("met_xy must align with coordinate-state batch axes")
        active = self.active_mask.to(device=state.device)
        if not torch.equal(
            state[..., ~active], torch.zeros_like(state[..., ~active])
        ):
            raise ValueError("Inactive invisible-coordinate components must be zero")
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
        self._validate(momenta, met_xy)
        shift = (met_xy - momenta[..., :, :2].sum(dim=-2)) / self.role_count
        return {
            "truth_projected_px_shift": shift[..., 0],
            "truth_projected_py_shift": shift[..., 1],
            "per_role_transverse_projection_magnitude": torch.linalg.vector_norm(
                shift, dim=-1
            ),
        }


def build_invisible_coordinates(config, role_names: tuple[str, ...]) -> InvisibleCoordinates:
    coordinate_type = str(config.type)
    if coordinate_type == "cartesian":
        return CartesianCoordinates(role_names)
    if coordinate_type == "met_residual":
        return MetResidualCoordinates(role_names)
    if coordinate_type == "met_exact":
        return MetExactCoordinates(role_names)
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
