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
        prepared_context=None,
    ) -> torch.Tensor:
        arguments = {
            "visible_fourmomenta": visible_fourmomenta,
            "met_xy": met_xy,
        }
        if prepared_context is not None:
            arguments["prepared_context"] = prepared_context
        return self.denoiser(
            state,
            time,
            visible,
            mask,
            parent,
            **arguments,
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
        source_state: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        visible_fourmomenta: torch.Tensor | None = None,
        met_xy: torch.Tensor | None = None,
        prepared_context=None,
        support_mask: torch.Tensor | None = None,
    ) -> dict:
        if source_state is None:
            source = self.source(target, generator=generator, condition=condition)
        else:
            if source_state.shape != target.shape or source_state.device != target.device:
                raise ValueError(
                    "Explicit CFM source_state must match target shape and device: "
                    f"source={tuple(source_state.shape)}@{source_state.device}, "
                    f"target={tuple(target.shape)}@{target.device}"
                )
            if source_state.dtype != target.dtype or not torch.isfinite(source_state).all():
                raise ValueError("Explicit CFM source_state must be finite and match target dtype")
            source = self.coupling.pair(
                source_state * self.active_mask.to(dtype=target.dtype),
                target,
                condition,
            )
        if flow_time is None:
            time = torch.rand(
                len(target), device=target.device, dtype=target.dtype, generator=generator
            )
        else:
            if flow_time.shape != (len(target),) or flow_time.device != target.device:
                raise ValueError("Explicit CFM flow_time must be [B] on the target device")
            if (
                flow_time.dtype != target.dtype
                or not torch.isfinite(flow_time).all()
                or (flow_time < 0).any()
                or (flow_time > 1).any()
            ):
                raise ValueError("Explicit CFM flow_time must be finite, target-dtype, and in [0,1]")
            time = flow_time
        state, target_velocity = linear_path(source, target, time)
        prediction = self.velocity(
            state,
            time,
            visible,
            mask,
            parent,
            visible_fourmomenta=visible_fourmomenta,
            met_xy=met_xy,
            prepared_context=prepared_context,
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
        initial_state: torch.Tensor | None = None,
        visible_fourmomenta: torch.Tensor | None = None,
        met_xy: torch.Tensor | None = None,
        prepared_context=None,
    ) -> tuple[torch.Tensor, int]:
        active = self.active_mask.to(device=visible.device, dtype=visible.dtype)
        if initial_state is None:
            initial = torch.randn(
                shape,
                device=visible.device,
                dtype=visible.dtype,
                generator=generator,
            ) * active
        else:
            if tuple(initial_state.shape) != tuple(shape) or initial_state.device != visible.device:
                raise ValueError(
                    "Explicit CFM initial_state must match requested shape and device: "
                    f"source={tuple(initial_state.shape)}@{initial_state.device}, "
                    f"requested={tuple(shape)}@{visible.device}"
                )
            if initial_state.dtype != visible.dtype or not torch.isfinite(initial_state).all():
                raise ValueError("Explicit CFM initial_state must be finite and match visible dtype")
            initial = initial_state * active
        if prepared_context is None:
            prepare_conditioning = getattr(self.denoiser, "prepare_conditioning", None)
            prepared_context = (
                prepare_conditioning(visible, mask, parent)
                if callable(prepare_conditioning)
                else None
            )

        def velocity_fn(state: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return self.velocity(
                state,
                time,
                visible,
                mask,
                parent,
                visible_fourmomenta=visible_fourmomenta,
                met_xy=met_xy,
                prepared_context=prepared_context,
            )

        values, nfe = integrate(initial, velocity_fn, steps=steps, solver=solver)
        return values * active, nfe
