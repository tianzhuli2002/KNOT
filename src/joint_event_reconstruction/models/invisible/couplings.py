"""Training-only Gaussian-to-target couplings for continuous Flow Matching."""

from __future__ import annotations

import torch


def _linear_assignment(cost: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("OT flow coupling requires scipy") from exc
    source_rows, target_columns = linear_sum_assignment(cost.detach().cpu().numpy())
    if len(source_rows) != len(source) or sorted(target_columns.tolist()) != list(
        range(len(source))
    ):
        raise RuntimeError("OT flow coupling did not produce a complete permutation")
    source_for_target = torch.empty(len(source), dtype=torch.long)
    source_for_target[torch.as_tensor(target_columns)] = torch.as_tensor(source_rows)
    return source[source_for_target.to(source.device)]


class IndependentCoupling:
    method = "independent"
    condition_weight = 0.0

    def pair(self, source, target, condition=None):
        del target, condition
        return source


class MinibatchOTCoupling:
    method = "minibatch_ot"
    condition_weight = 0.0

    def __init__(self, active_mask: torch.Tensor):
        self.active_mask = torch.as_tensor(active_mask, dtype=torch.bool)

    def state_cost(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        active = self.active_mask.to(source.device)
        dimensions = int(active.sum())
        if dimensions <= 0:
            raise ValueError("Flow coupling needs at least one active state dimension")
        source_flat = source[..., active]
        target_flat = target[..., active]
        return torch.cdist(source_flat, target_flat).square() / dimensions

    def pair(self, source, target, condition=None):
        del condition
        if len(source) <= 1:
            return source
        return _linear_assignment(self.state_cost(source, target), source)


class ConditionalOTCoupling(MinibatchOTCoupling):
    method = "conditional_ot"

    def __init__(self, active_mask: torch.Tensor, condition_weight: float):
        super().__init__(active_mask)
        self.condition_weight = float(condition_weight)
        if not torch.isfinite(torch.tensor(self.condition_weight)) or self.condition_weight < 0:
            raise ValueError("Conditional OT weight must be finite and nonnegative")

    def pair(self, source, target, condition=None):
        if condition is None:
            raise ValueError("conditional_ot requires one condition vector per event")
        if condition.ndim != 2 or len(condition) != len(source):
            raise ValueError("Flow coupling condition must be [B,D]")
        if len(source) <= 1:
            return source
        condition = condition.detach()
        condition = condition / torch.linalg.vector_norm(
            condition, dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        condition_cost = torch.cdist(condition, condition).square() / condition.shape[1]
        cost = self.state_cost(source, target) + self.condition_weight * condition_cost
        return _linear_assignment(cost, source)


def build_flow_coupling(config, active_mask: torch.Tensor):
    method = str(config.method)
    weight = float(config.condition_weight)
    if method == "independent":
        if weight != 0:
            raise ValueError("Independent coupling requires condition_weight=0")
        return IndependentCoupling()
    if method == "minibatch_ot":
        if weight != 0:
            raise ValueError("Minibatch OT coupling requires condition_weight=0")
        return MinibatchOTCoupling(active_mask)
    if method == "conditional_ot":
        return ConditionalOTCoupling(active_mask, weight)
    raise ValueError(f"Unknown flow coupling method {method!r}")
