"""Composable visible-event reconstruction model."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.models.feedback import SoftReconstructionFeedback
from joint_event_reconstruction.models.invisible.feedback import InvisiblePosteriorFeedback
from joint_event_reconstruction.models.invisible.generator import InvisibleGenerator
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
        invisible_config=None, coupling_config=None, invisible_loss_weight: float = 1.0,
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
        self.invisible_loss_weight = invisible_loss_weight
        self.reasoner = None
        self.feedback = None
        if reasoning_enabled:
            self.reasoner = RecurrentReasoner(
                spec, d_model, n_heads, reasoning_block_layers, d_feedforward, dropout,
                fixed_input_reinjection=fixed_input_reinjection,
            )
        if feedback_enabled:
            self.feedback = SoftReconstructionFeedback(spec, d_model, feedback_gate_initial_bias)
        self.invisible = None
        self.posterior_feedback = None
        self.coupling_config = coupling_config
        if invisible_config is not None and bool(invisible_config.enabled):
            self.invisible = InvisibleGenerator(spec, d_model, invisible_config)
        if coupling_config is not None and bool(coupling_config.invisible_to_reconstruction.enabled):
            if not reasoning_enabled or self.invisible is None:
                raise ValueError("Invisible posterior feedback requires reasoning and an invisible generator")
            self.posterior_feedback = InvisiblePosteriorFeedback(
                spec, d_model, float(coupling_config.invisible_to_reconstruction.gate_initial_bias),
            )

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
            result = {
                "visible": visible, **reconstruction, "loss": reconstruction["reconstruction_loss"],
                "recurrent_steps": 0, "recurrent_predictions": (),
            }
            return self._invisible_result(result, batch, visible, None)

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
        result = {
            "visible": visible, **reconstruction, "loss": reconstruction["reconstruction_loss"],
            "recurrent_steps": steps, "recurrent_predictions": tuple(predictions),
            "feedback_diagnostics": tuple(feedback_diagnostics), "recurrent_state": state,
            "node_context": node_tokens,
        }
        if self.posterior_feedback is not None and int(self.coupling_config.rounds) == 2:
            parent = self.invisible.parent_context(
                reconstruction, batch["structured"], pooled, node_tokens,
            )
            feedback_config = self.coupling_config.invisible_to_reconstruction
            samples, nfe = self.invisible.sample(
                visible, parent, samples=int(feedback_config.posterior_samples),
            )
            updated, gates = self.posterior_feedback(
                node_tokens, samples, detach=bool(feedback_config.detach),
            )
            state = self.reasoner.replace_topology(state, visible["tokens"].shape[1], updated)
            state = self.reasoner.step(state, fixed, mask)
            pooled, encoded_tokens, node_tokens = self.reasoner.split(state, visible["tokens"].shape[1])
            reconstruction = self.reconstruction(
                encoded_tokens, pooled, batch["structured"], node_context=node_tokens,
            )
            result.update(reconstruction)
            result["loss"] = reconstruction["reconstruction_loss"]
            result["recurrent_state"] = state
            result["node_context"] = node_tokens
            result["posterior_feedback"] = {
                "samples": samples, "nfe": nfe, "gates": gates,
                "detached": bool(feedback_config.detach),
            }
        return self._invisible_result(result, batch, visible, result.get("node_context"))

    def _invisible_result(
        self, result: dict, batch: dict, visible: dict, node_context: dict[str, torch.Tensor] | None,
    ) -> dict:
        if self.invisible is None:
            return result
        parent = self.invisible.parent_context(
            result, batch["structured"], result["visible"]["pooled"], node_context,
        )
        result["invisible_parent_context"] = parent
        if "invisible_truth_scaled" in batch:
            invisible = self.invisible.loss(batch["invisible_truth_scaled"], visible, parent)
            result["invisible_loss"] = invisible["loss"]
            result["invisible_training"] = invisible
            result["loss"] = result["loss"] + self.invisible_loss_weight * invisible["loss"]
        return result


# Kept as a source-level name for the existing M1 Lightning wrapper.  It is the
# same maintained implementation, not a separate model branch.
StructuredEventModel = JointEventModel
