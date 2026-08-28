"""Small explicit ODE integration helpers used by continuous flow models."""

from __future__ import annotations

from collections.abc import Callable

import torch


Velocity = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def euler_step(
    state: torch.Tensor,
    velocity: torch.Tensor,
    dt: float | torch.Tensor,
) -> torch.Tensor:
    return state + dt * velocity


def integrate(
    initial: torch.Tensor,
    velocity_fn: Velocity,
    *,
    steps: int,
    solver: str,
) -> tuple[torch.Tensor, int]:
    """Integrate t=0 -> 1 with fixed Euler or Heun steps."""
    if steps <= 0:
        raise ValueError("Flow integration steps must be positive")
    if solver not in {"euler", "heun"}:
        raise ValueError(f"Flow solver must be euler or heun, found {solver!r}")
    state = initial
    schedule = torch.linspace(
        0.0, 1.0, steps + 1, device=initial.device, dtype=initial.dtype
    )
    evaluations = 0
    batch = len(initial)
    for current, following in zip(schedule[:-1], schedule[1:]):
        time = current.expand(batch)
        velocity = velocity_fn(state, time)
        evaluations += 1
        dt = following - current
        proposed = euler_step(state, velocity, dt)
        if solver == "heun":
            next_velocity = velocity_fn(proposed, following.expand(batch))
            evaluations += 1
            state = state + dt * (velocity + next_velocity) / 2
        else:
            state = proposed
    return state, evaluations
