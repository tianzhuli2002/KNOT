"""Conditional Flow Matching for invisible momenta."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from joint_event_reconstruction.flows.continuous import linear_path
from joint_event_reconstruction.flows.solvers import integrate


def _minibatch_ot_source(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pair independent source noise to batch targets by exact linear assignment."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as error:
        raise RuntimeError(
            "minibatch_ot coupling requires scipy; install the project dependencies"
        ) from error
    source_flat = source.detach().flatten(1)
    target_flat = target.detach().flatten(1)
    cost = torch.cdist(source_flat, target_flat).square().cpu().numpy()
    source_rows, target_columns = linear_sum_assignment(cost)
    if len(source_rows) != len(source) or sorted(target_columns.tolist()) != list(
        range(len(target))
    ):
        raise RuntimeError(
            "Minibatch OT did not produce a complete source-target permutation"
        )
    source_for_target = torch.empty(len(target), dtype=torch.long)
    source_for_target[
        torch.as_tensor(target_columns, dtype=torch.long)
    ] = torch.as_tensor(source_rows, dtype=torch.long)
    return source[source_for_target.to(source.device)]


class ContinuousFlowMatching(nn.Module):
    def __init__(self, denoiser: nn.Module, *, coupling: str = "independent"):
        super().__init__()
        if coupling not in {"independent", "minibatch_ot"}:
            raise ValueError(
                f"Flow coupling must be independent or minibatch_ot, found {coupling!r}"
            )
        self.denoiser = denoiser
        self.coupling = coupling

    def source(
        self,
        target: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        source = torch.randn(
            target.shape,
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        )
        if self.coupling == "minibatch_ot" and len(target) > 1:
            source = _minibatch_ot_source(source, target)
        return source

    def velocity(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        visible: torch.Tensor,
        mask: torch.Tensor,
        parent: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(state, time, visible, mask, parent)

    def loss(
        self,
        target: torch.Tensor,
        visible: torch.Tensor,
        mask: torch.Tensor,
        parent: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> dict:
        source = self.source(target, generator=generator)
        time = torch.rand(
            len(target), device=target.device, dtype=target.dtype, generator=generator
        )
        state, target_velocity = linear_path(source, target, time)
        prediction = self.velocity(state, time, visible, mask, parent)
        return {
            "loss": F.mse_loss(prediction, target_velocity),
            "time": time,
            "state": state,
            "source": source,
            "target_velocity": target_velocity,
            "prediction": prediction,
        }

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, int, int],
        visible: torch.Tensor,
        mask: torch.Tensor,
        parent: torch.Tensor,
        *,
        steps: int,
        solver: str,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, int]:
        initial = torch.randn(
            shape,
            device=visible.device,
            dtype=visible.dtype,
            generator=generator,
        )

        def velocity_fn(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return self.velocity(state, time, visible, mask, parent)

        return integrate(initial, velocity_fn, steps=steps, solver=solver)
