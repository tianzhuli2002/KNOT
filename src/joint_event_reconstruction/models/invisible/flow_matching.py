"""Conditional Flow Matching for invisible momenta."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.flows.continuous import linear_path
from joint_event_reconstruction.flows.solvers import integrate

class ContinuousFlowMatching(nn.Module):
    def __init__(self, denoiser: nn.Module, *, coupling, active_mask: torch.Tensor):
        super().__init__()
        self.denoiser = denoiser
        self.coupling = coupling
        self.register_buffer("active_mask", torch.as_tensor(active_mask, dtype=torch.bool))

    def source(
        self,
        target: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = torch.randn(
            target.shape,
            device=target.device,
            dtype=target.dtype,
            generator=generator,
        ) * self.active_mask.to(dtype=target.dtype)
        return self.coupling.pair(source, target, condition)

    def velocity(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        visible: torch.Tensor,
        mask: torch.Tensor,
        parent: torch.Tensor,
        *,
        visible_fourmomenta: torch.Tensor | None = None,
        met_xy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.denoiser(
            state,
            time,
            visible,
            mask,
            parent,
            visible_fourmomenta=visible_fourmomenta,
            met_xy=met_xy,
        ) * self.active_mask.to(dtype=state.dtype)

    def loss(
        self,
        target: torch.Tensor,
        visible: torch.Tensor,
        mask: torch.Tensor,
        parent: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        condition: torch.Tensor | None = None,
        visible_fourmomenta: torch.Tensor | None = None,
        met_xy: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
    ) -> dict:
        source = self.source(target, generator=generator, condition=condition)
        time = torch.rand(
            len(target), device=target.device, dtype=target.dtype, generator=generator
        )
        state, target_velocity = linear_path(source, target, time)
        prediction = self.velocity(
            state,
            time,
            visible,
            mask,
            parent,
            visible_fourmomenta=visible_fourmomenta,
            met_xy=met_xy,
        )
        squared_error = (prediction - target_velocity).square()
        active = self.active_mask.to(device=target.device, dtype=target.dtype)
        active_dimensions = active.sum().clamp_min(1.0)
        per_event = (squared_error * active).reshape(len(target), -1).sum(dim=1)
        per_event = per_event / active_dimensions
        if support_mask is None:
            loss = per_event.mean()
        else:
            if support_mask.dtype != torch.bool or support_mask.shape != (len(target),):
                raise ValueError(
                    "CFM support_mask must be boolean [B], found "
                    f"dtype={support_mask.dtype}, shape={tuple(support_mask.shape)}"
                )
            weights = support_mask.to(dtype=per_event.dtype)
            loss = (per_event * weights).sum() / weights.sum().clamp_min(1.0)
        return {
            "loss": loss,
            "per_event_loss": per_event,
            "time": time,
            "state": state,
            "source": source,
            "target_velocity": target_velocity,
            "prediction": prediction,
        }

    @torch.no_grad()
    def sample(
        self,
        shape: tuple[int, ...],
        visible: torch.Tensor,
        mask: torch.Tensor,
        parent: torch.Tensor,
        *,
        steps: int,
        solver: str,
        generator: torch.Generator | None = None,
        visible_fourmomenta: torch.Tensor | None = None,
        met_xy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        active = self.active_mask.to(device=visible.device, dtype=visible.dtype)
        initial = torch.randn(
            shape,
            device=visible.device,
            dtype=visible.dtype,
            generator=generator,
        ) * active

        def velocity_fn(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return self.velocity(
                state,
                time,
                visible,
                mask,
                parent,
                visible_fourmomenta=visible_fourmomenta,
                met_xy=met_xy,
            )

        values, nfe = integrate(initial, velocity_fn, steps=steps, solver=solver)
        return values * active, nfe
