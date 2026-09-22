"""Small, explicit contracts for the controlled Study-10 scorer analysis.

The expensive model prediction is deliberately outside this module.  These
helpers operate on one stored feature artifact so scorer training, ablations,
transitions and K/M scans never regenerate a posterior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from joint_event_reconstruction.topology.physics import LearnedHypothesisScorer, PosteriorReducer


K_PREFIXES = (1, 2, 3, 5, 10)
M_PREFIXES = (4, 8, 16, 32, 64)
FEATURE_ABLATIONS = (
    "W_mass_only",
    "top_masses_only",
    "top_mass_balance",
    "MET_residual_only",
    "W_plus_top",
    "all",
)


def reduced_scorer_input(
    reduction,
    visible_log_score: torch.Tensor,
    visible_probability: torch.Tensor,
) -> torch.Tensor:
    """Build the deployable scorer input from reduced physics and visible data."""
    reduced = reduction.as_tensor()
    if visible_log_score.shape != reduced.shape[:2] or visible_probability.shape != reduced.shape[:2]:
        raise ValueError("Study-10 visible scores disagree with reduced physics shape")
    return torch.cat(
        (reduced, visible_log_score.unsqueeze(-1), visible_probability.unsqueeze(-1)),
        dim=-1,
    )


def visible_only_scorer_input(
    visible_log_score: torch.Tensor,
    visible_probability: torch.Tensor,
    assignment_entropy: torch.Tensor,
    top1_top2_margin: torch.Tensor,
    *,
    target_features: int,
) -> torch.Tensor:
    """Build a padded, explicitly physics-blind control input.

    The first four columns are the only allowed visible-only inputs.  Padding
    is deterministic zero padding solely to match the physics scorer input
    width; it does not carry any invisible or physics-derived value.
    """
    if visible_log_score.ndim != 2:
        raise ValueError("Visible-only Study-10 log scores must have shape [B,K]")
    shape = visible_log_score.shape
    if visible_probability.shape != shape:
        raise ValueError("Visible-only Study-10 probabilities must have shape [B,K]")

    def expand_event_feature(value: torch.Tensor, name: str) -> torch.Tensor:
        if value.shape == shape:
            return value
        if value.ndim == 1 and value.shape[0] == shape[0]:
            return value[:, None].expand(-1, shape[1])
        raise ValueError(
            f"Visible-only Study-10 {name} must have shape [B] or [B,K]; "
            f"received {tuple(value.shape)} for [B,K]={tuple(shape)}"
        )

    assignment_entropy = expand_event_feature(assignment_entropy, "assignment entropy")
    top1_top2_margin = expand_event_feature(top1_top2_margin, "top1-top2 margin")
    if target_features < 4:
        raise ValueError("Matched visible-only control needs at least four input features")
    base = torch.stack(
        (visible_log_score, visible_probability, assignment_entropy, top1_top2_margin),
        dim=-1,
    )
    return torch.nn.functional.pad(base, (0, target_features - base.shape[-1]))


def nested_prefix_reduction(
    scaled_features: torch.Tensor,
    valid_hypotheses: torch.Tensor,
    *,
    k_prefixes: Sequence[int] = K_PREFIXES,
    m_prefixes: Sequence[int] = M_PREFIXES,
) -> torch.Tensor:
    """Return deterministic analytic scores for nested K/M posterior prefixes.

    The input is generated once at Kmax/Mmax.  Prefixes are always the first
    hypotheses/samples in stored order, making the scan reproducible and free
    of a second stochastic inference call.
    """
    if scaled_features.ndim != 4:
        raise ValueError("Study-10 posterior features must have shape [B,K,M,F]")
    if valid_hypotheses.shape != scaled_features.shape[:2]:
        raise ValueError("Study-10 valid hypothesis mask has the wrong shape")
    if any(k <= 0 or k > scaled_features.shape[1] for k in k_prefixes):
        raise ValueError("Study-10 K prefix exceeds stored Kmax")
    if any(m <= 0 or m > scaled_features.shape[2] for m in m_prefixes):
        raise ValueError("Study-10 M prefix exceeds stored Mmax")
    reducer = PosteriorReducer()
    result = torch.empty(
        len(k_prefixes), len(m_prefixes), scaled_features.shape[0], scaled_features.shape[1],
        dtype=scaled_features.dtype, device=scaled_features.device,
    )
    for k_index, k in enumerate(k_prefixes):
        for m_index, m in enumerate(m_prefixes):
            reduction = reducer.reduce(
                scaled_features[:, :k, :m],
                valid_samples=valid_hypotheses[:, :k, None].expand(-1, k, m),
                valid_hypotheses=valid_hypotheses[:, :k],
            )
            result[k_index, m_index, :, :k] = reduction.analytic_log_evidence
            if k < scaled_features.shape[1]:
                result[k_index, m_index, :, k:] = -torch.inf
    return result


def top_k_coverage(truth: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    if truth.shape != valid.shape or truth.ndim != 2 or not 0 < k <= truth.shape[1]:
        raise ValueError("Study-10 truth and valid arrays have incompatible shape")
    return (truth[:, :k] & valid[:, :k]).any(dim=1)


def rank_correctness(score: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if score.shape != truth.shape or score.shape != valid.shape:
        raise ValueError("Study-10 score, truth and valid arrays must have the same shape")
    masked = score.masked_fill(~valid, -torch.inf)
    ranking = torch.argsort(masked, dim=1, descending=True, stable=True)
    return truth.gather(1, ranking[:, :1]).squeeze(1)


def transition_counts(baseline: torch.Tensor, selected: torch.Tensor) -> dict[str, int]:
    if baseline.shape != selected.shape:
        raise ValueError("Study-10 transition arrays must have the same shape")
    return {
        "baseline_correct_to_remains_correct": int((baseline & selected).sum()),
        "baseline_correct_to_broken": int((baseline & ~selected).sum()),
        "baseline_wrong_to_rescued": int((~baseline & selected).sum()),
        "baseline_wrong_to_remains_wrong": int((~baseline & ~selected).sum()),
    }


def coupling_efficiency(
    visible_accuracy: float,
    coupled_accuracy: float,
    truth_neutrino_accuracy: float,
) -> dict[str, float | None]:
    delta_reco = float(coupled_accuracy - visible_accuracy)
    delta_oracle = float(truth_neutrino_accuracy - visible_accuracy)
    return {
        "delta_reco": delta_reco,
        "delta_oracle": delta_oracle,
        "Rcoupling": None if delta_oracle <= 0 else delta_reco / delta_oracle,
    }


def select_alpha(
    visible_log_score: torch.Tensor,
    analytic_score: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    alphas: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Select alpha on TRAIN only; ties resolve to the smallest alpha."""
    results: dict[str, float] = {}
    for alpha in alphas:
        score = (
            visible_log_score
            if float(alpha) == 0.0
            else visible_log_score + float(alpha) * analytic_score
        )
        correct = rank_correctness(score, truth, valid)
        results[str(float(alpha))] = float(correct.float().mean())
    selected = min(alphas, key=lambda alpha: (-results[str(float(alpha))], float(alpha)))
    return float(selected), results


@dataclass(frozen=True)
class TrainedScorers:
    physics: LearnedHypothesisScorer
    visible_only: LearnedHypothesisScorer
    physics_input_features: int
    hidden_features: int


def _truth_loss(score: torch.Tensor, truth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if score.ndim != 2 or truth.shape != score.shape or valid.shape != score.shape:
        raise ValueError("Study-10 truth loss expects score, truth and valid arrays with shape [B,K]")
    target = truth & valid
    if not target.any(dim=1).all():
        raise ValueError("Study-10 training feature artifact has an event without a valid truth hypothesis")
    masked = score.masked_fill(~valid, -torch.inf)
    return (torch.logsumexp(masked, dim=1) - torch.logsumexp(masked.masked_fill(~target, -torch.inf), dim=1)).mean()


def train_scorers(
    visible_log_score: torch.Tensor,
    visible_probability: torch.Tensor,
    physics_features: torch.Tensor,
    visible_only_features: torch.Tensor,
    truth: torch.Tensor,
    valid: torch.Tensor,
    *,
    hidden: int = 32,
    epochs: int = 32,
    learning_rate: float = 1.0e-3,
) -> TrainedScorers:
    tensors = {
        "visible_log_score": visible_log_score,
        "visible_probability": visible_probability,
        "physics_features": physics_features,
        "visible_only_features": visible_only_features,
        "truth": truth,
        "valid": valid,
    }
    if truth.dtype is not torch.bool or valid.dtype is not torch.bool:
        raise ValueError("Study-10 truth and valid masks must have dtype torch.bool")
    if truth.ndim != 2 or valid.shape != truth.shape:
        raise ValueError("Study-10 truth and valid masks must have shape [B,K]")
    if visible_log_score.ndim != 2 or visible_probability.shape != visible_log_score.shape:
        raise ValueError("Study-10 visible scores and probabilities must have shape [B,K]")
    if physics_features.ndim != 3 or visible_only_features.shape != physics_features.shape:
        raise ValueError("Matched Study-10 scorer inputs must have identical shape [B,K,F]")
    batch, candidates = truth.shape
    if batch <= 0:
        raise ValueError("Study-10 scorer training cannot run with zero events")
    if candidates <= 0:
        raise ValueError("Study-10 scorer training cannot run with zero candidates")
    if physics_features.shape[:2] != (batch, candidates):
        raise ValueError("Study-10 scorer features disagree with truth/valid shape")
    for name, tensor in tensors.items():
        if tensor.shape[0] != batch:
            raise ValueError(
                f"Study-10 {name} has event dimension {tensor.shape[0]}, expected {batch}"
            )
    if (truth & ~valid).any():
        raise ValueError("Study-10 truth is marked on an invalid candidate")
    if (~valid).all(dim=1).any():
        raise ValueError("Study-10 training feature artifact has an event without a valid candidate")
    support = (truth & valid).any(dim=1)
    if not support.any():
        raise ValueError("Study-10 scorer training has zero truth-supported events")

    valid_supported = valid & support[:, None]
    valid_features = valid_supported.unsqueeze(-1)
    for name, tensor in (
        ("visible_log_score", visible_log_score),
        ("visible_probability", visible_probability),
        ("physics_features", physics_features),
        ("visible_only_features", visible_only_features),
    ):
        finite_mask = valid_supported if tensor.ndim == 2 else valid_features
        if not torch.isfinite(tensor.masked_select(finite_mask)).all():
            raise FloatingPointError(
                f"Study-10 {name} has non-finite values for a valid candidate"
            )

    # The two scorers are fitted on precisely the same supported events.  Any
    # invalid-candidate sentinels are replaced before a neural-network layer
    # sees them; they remain masked by _truth_loss for the candidate sum.
    visible_log_score = visible_log_score[support]
    visible_probability = visible_probability[support]
    physics_features = physics_features[support]
    visible_only_features = visible_only_features[support]
    truth = truth[support]
    valid = valid[support]
    valid_features = valid.unsqueeze(-1)
    visible_log_score = torch.where(valid, visible_log_score, torch.zeros_like(visible_log_score))
    visible_probability = torch.where(valid, visible_probability, torch.zeros_like(visible_probability))
    physics_features = torch.where(valid_features, physics_features, torch.zeros_like(physics_features))
    visible_only_features = torch.where(
        valid_features, visible_only_features, torch.zeros_like(visible_only_features)
    )
    if epochs <= 0 or hidden <= 0:
        raise ValueError("Study-10 scorer epochs and hidden width must be positive")
    physics = LearnedHypothesisScorer(physics_features.shape[-1], hidden=hidden)
    visible = LearnedHypothesisScorer(visible_only_features.shape[-1], hidden=hidden)
    optim = torch.optim.AdamW(
        list(physics.parameters()) + list(visible.parameters()), lr=learning_rate
    )
    for _ in range(epochs):
        optim.zero_grad(set_to_none=True)
        physics_score = visible_log_score + physics(physics_features)
        visible_score = visible_log_score + visible(visible_only_features)
        loss = _truth_loss(physics_score, truth, valid) + _truth_loss(visible_score, truth, valid)
        if not torch.isfinite(loss):
            raise FloatingPointError("Study-10 scorer training produced a non-finite loss")
        loss.backward()
        gradients = [parameter.grad for parameter in (*physics.parameters(), *visible.parameters())]
        if not all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Study-10 scorer training produced non-finite gradients")
        optim.step()
    physics.eval()
    visible.eval()
    return TrainedScorers(physics, visible, physics_features.shape[-1], hidden)


__all__ = [
    "FEATURE_ABLATIONS",
    "K_PREFIXES",
    "M_PREFIXES",
    "TrainedScorers",
    "coupling_efficiency",
    "nested_prefix_reduction",
    "rank_correctness",
    "reduced_scorer_input",
    "select_alpha",
    "top_k_coverage",
    "train_scorers",
    "transition_counts",
    "visible_only_scorer_input",
]
