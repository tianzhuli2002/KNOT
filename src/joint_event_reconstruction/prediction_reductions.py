"""Truth-independent reductions of a joint invisible posterior."""

from __future__ import annotations

import torch


POINT_ESTIMATORS = {
    "cfm_single_draw",
    "cfm_cartesian_mean",
    "cfm_joint_medoid",
}


def reduce_joint_posterior(
    samples: torch.Tensor,
    strategy: str,
    *,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reduce ``[..., M, R, 3]`` while preserving the joint role sample.

    The medoid is selected once per complete ``(R,3)`` configuration.  No
    truth-dependent quantity is accepted by this function and roles are never
    reduced independently.
    """
    strategy = str(strategy)
    if strategy not in POINT_ESTIMATORS:
        raise ValueError(f"Unknown posterior point-estimator strategy {strategy!r}")
    if samples.ndim < 4 or samples.shape[-1] != 3 or samples.shape[-2] <= 0:
        raise ValueError(f"Posterior samples must end in [M,R,3], found {tuple(samples.shape)}")
    if not torch.isfinite(samples).all():
        raise ValueError("Posterior samples must be finite before point reduction")
    if strategy == "cfm_single_draw":
        return samples[..., 0, :, :]
    if strategy == "cfm_cartesian_mean":
        return samples.mean(dim=-3)

    roles = samples.shape[-2]
    if scale is None:
        scale = samples.new_ones((roles, 3))
    scale = torch.as_tensor(scale, device=samples.device, dtype=samples.dtype)
    if tuple(scale.shape) != (roles, 3) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError(f"Medoid normalisation scale must be finite positive [{roles},3]")
    normalised = samples / scale
    pairwise = normalised.unsqueeze(-2) - normalised.unsqueeze(-3)
    distance = pairwise.square().sum(dim=(-1, -2)).sqrt().mean(dim=-1)
    selected = distance.argmin(dim=-1)
    gather_index = selected[..., None, None, None].expand(*selected.shape, 1, roles, 3)
    return samples.gather(-3, gather_index).squeeze(-3)


def physical_four_vectors_from_momenta(coordinates, momenta: torch.Tensor) -> torch.Tensor:
    """Construct physical four-vectors from spatial momentum and declared masses."""
    if momenta.shape[-1] != 3 or momenta.shape[-2] != coordinates.role_count:
        raise ValueError(
            "Physical point output must have shape [..., invisible_roles, 3], "
            f"found {tuple(momenta.shape)}"
        )
    vectors = []
    for index, role in enumerate(coordinates.role_names):
        shell = coordinates.mass_shells[role]
        if shell.mode == "unconstrained":
            raise ValueError(
                f"Point-invisible output cannot construct E from unconstrained mass semantics for role {role!r}"
            )
        if shell.mode == "generated":
            raise ValueError(
                f"Point-invisible output requires a predicted mass for generated-mass role {role!r}"
            )
        spatial = momenta[..., index, :]
        mass = spatial.new_full(spatial.shape[:-1], float(shell.value))
        energy = (spatial.square().sum(dim=-1) + mass.square()).clamp_min(0).sqrt()
        vectors.append(torch.cat((energy.unsqueeze(-1), spatial), dim=-1))
    return torch.stack(vectors, dim=-2)
