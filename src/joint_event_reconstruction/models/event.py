"""Composable visible-event reconstruction model."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.models.feedback import SoftReconstructionFeedback
from joint_event_reconstruction.models.recurrent import RecurrentReasoner
from joint_event_reconstruction.models.reconstruction.structured import StructuredReconstruction
from joint_event_reconstruction.models.visible.encoder import VisibleEventEncoder
from joint_event_reconstruction.topology.spec import TopologySpec


class JointEventModel(nn.Module):
    def __init__(
        self, spec: TopologySpec, *, n_features: int, n_object_types: int,
        d_model: int, n_heads: int, n_layers: int, d_feedforward: int, dropout: float,
        reasoning_enabled: bool = False, reasoning_block_layers: int = 2,
        train_steps: tuple[int, ...] = (1, 2, 3, 4), eval_steps: int = 4,
        fixed_input_reinjection: bool = True, feedback_enabled: bool = False,
        feedback_start_step: int = 1, feedback_gate_initial_bias: float = -2.0,
    ):
        super().__init__()
        if feedback_enabled and not reasoning_enabled:
            raise ValueError("Soft reconstruction feedback requires recurrent reasoning")
        if not train_steps or any(step <= 0 for step in train_steps):
            raise ValueError(f"Recurrent train_steps must contain positive integers, found {train_steps}")
        if eval_steps <= 0:
            raise ValueError("Recurrent eval_steps must be positive")
        self.visible = VisibleEventEncoder(
            n_features, d_model, n_heads, n_layers, d_feedforward, dropout, n_object_types,
        )
        self.reconstruction = StructuredReconstruction(spec, d_model, dropout)
        self.reasoning_enabled = reasoning_enabled
        self.feedback_enabled = feedback_enabled
        self.train_steps = tuple(train_steps)
        self.eval_steps = eval_steps
        self.feedback_start_step = feedback_start_step
        self.reasoner = None
        self.feedback = None
        if reasoning_enabled:
            self.reasoner = RecurrentReasoner(
                spec, d_model, n_heads, reasoning_block_layers, d_feedforward, dropout,
                fixed_input_reinjection=fixed_input_reinjection,
            )
        if feedback_enabled:
            self.feedback = SoftReconstructionFeedback(spec, d_model, feedback_gate_initial_bias)

    def _steps(self, requested: int | None) -> int:
        if requested is not None:
            if requested <= 0:
                raise ValueError("Requested recurrent steps must be positive")
            return requested
        if self.training:
            index = int(torch.randint(len(self.train_steps), (), device=self.visible.cls_token.device))
            return self.train_steps[index]
        return self.eval_steps

    def forward(self, batch: dict, *, recurrent_steps: int | None = None) -> dict:
        visible = self.visible(batch["tokens"], batch["type_ids"], batch["padding_mask"])
        if not self.reasoning_enabled:
            reconstruction = self.reconstruction(visible["tokens"], visible["pooled"], batch["structured"])
            return {
                "visible": visible, **reconstruction, "loss": reconstruction["reconstruction_loss"],
                "recurrent_steps": 0, "recurrent_predictions": (),
            }

        steps = self._steps(recurrent_steps)
        fixed = torch.cat((visible["pooled"].unsqueeze(1), visible["tokens"]), dim=1)
        state = self.reasoner.initialise(visible["pooled"], visible["tokens"])
        mask = self.reasoner.mask(visible["padding_mask"])
        predictions = []
        feedback_diagnostics = []
        reconstruction = None
        for step in range(steps):
            state = self.reasoner.step(state, fixed, mask)
            pooled, encoded_tokens, node_tokens = self.reasoner.split(state, visible["tokens"].shape[1])
            reconstruction = self.reconstruction(
                encoded_tokens, pooled, batch["structured"], node_context=node_tokens,
                calculate_loss=step == steps - 1,
            )
            predictions.append({
                "assignment_probabilities": reconstruction["assignment_probabilities"],
                "assignment_logits": reconstruction["assignment_logits"],
                "token_norm": state.norm(dim=-1).mean(dim=1),
            })
            if self.feedback_enabled and step + 1 >= self.feedback_start_step and step < steps - 1:
                updated, diagnostics = self.feedback(
                    node_tokens, reconstruction["candidate_embeddings"],
                    reconstruction["assignment_probabilities"],
                )
                state = self.reasoner.replace_topology(state, visible["tokens"].shape[1], updated)
                feedback_diagnostics.append(diagnostics)
        return {
            "visible": visible, **reconstruction, "loss": reconstruction["reconstruction_loss"],
            "recurrent_steps": steps, "recurrent_predictions": tuple(predictions),
            "feedback_diagnostics": tuple(feedback_diagnostics), "recurrent_state": state,
        }


# Kept as a source-level name for the existing M1 Lightning wrapper.  It is the
# same maintained implementation, not a separate model branch.
StructuredEventModel = JointEventModel
