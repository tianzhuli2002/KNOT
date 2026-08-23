"""Small energy-style scorer for complete visible/invisible hypotheses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CompatibilityHead(nn.Module):
    """Score mutual consistency; the output is not a normalised likelihood."""

    def __init__(self, d_model: int, invisible_roles: int, *, include_invisible: bool = True):
        super().__init__()
        self.include_invisible = include_invisible
        invisible_width = invisible_roles * 3 if include_invisible else 0
        self.network = nn.Sequential(
            nn.Linear(2 * d_model + invisible_width, d_model), nn.GELU(),
            nn.Linear(d_model, max(d_model // 2, 16)), nn.GELU(), nn.Linear(max(d_model // 2, 16), 1),
        )

    def forward(
        self, event: torch.Tensor, assignment: torch.Tensor, invisible: torch.Tensor,
    ) -> torch.Tensor:
        if event.shape[:-1] != assignment.shape[:-1]:
            raise ValueError("Event and assignment compatibility contexts have different batch shapes")
        pieces = [event, assignment]
        if self.include_invisible:
            if invisible.shape[:-2] != event.shape[:-1] or invisible.shape[-1] != 3:
                raise ValueError("Invisible compatibility input must end in [Rnu,3]")
            pieces.append(invisible.flatten(-2))
        return self.network(torch.cat(pieces, dim=-1)).squeeze(-1)

    def contrastive_loss(
        self, event: torch.Tensor, truth_assignment: torch.Tensor,
        wrong_assignment: torch.Tensor, truth_invisible: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        positive = self(event, truth_assignment, truth_invisible)
        wrong_visible = self(event, wrong_assignment, truth_invisible)
        if len(event) > 1:
            mismatched_invisible = truth_invisible.roll(1, dims=0)
        else:
            mismatched_invisible = -truth_invisible
        wrong_invisible = self(event, truth_assignment, mismatched_invisible)
        logits = torch.cat((positive, wrong_visible, wrong_invisible))
        labels = torch.cat((torch.ones_like(positive), torch.zeros_like(wrong_visible), torch.zeros_like(wrong_invisible)))
        return {
            "loss": F.binary_cross_entropy_with_logits(logits, labels),
            "positive_logit": positive.detach(), "wrong_assignment_logit": wrong_visible.detach(),
            "mismatched_invisible_logit": wrong_invisible.detach(),
        }
