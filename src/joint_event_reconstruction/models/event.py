"""Composable topology-aware event reconstruction model."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.models.compatibility import CompatibilityHead
from joint_event_reconstruction.models.feedback import SoftReconstructionFeedback
from joint_event_reconstruction.models.invisible.feedback import InvisiblePosteriorFeedback
from joint_event_reconstruction.models.invisible.generator import InvisibleGenerator
from joint_event_reconstruction.models.joint_flow import JointFlow
from joint_event_reconstruction.models.recurrent import RecurrentReasoner
from joint_event_reconstruction.models.reconstruction.discrete_flow import DiscreteAssignmentFlow
from joint_event_reconstruction.models.reconstruction.structured import StructuredReconstruction
from joint_event_reconstruction.models.visible.encoder import VisibleEventEncoder
from joint_event_reconstruction.topology.spec import TopologySpec


class JointEventModel(nn.Module):
    def __init__(
        self, spec: TopologySpec, *, n_features: int, n_object_types: int,
        d_model: int, n_heads: int, n_layers: int, d_feedforward: int, dropout: float,
        model_mode: str = "factorized", reconstruction_formulation: str = "discriminative",
        reasoning_enabled: bool = False, reasoning_block_layers: int = 2,
        train_steps: tuple[int, ...] = (1, 2, 3, 4), eval_steps: int = 4,
        fixed_input_reinjection: bool = True, feedback_enabled: bool = False,
        feedback_start_step: int = 1, feedback_gate_initial_bias: float = -2.0,
        invisible_config=None, coupling_config=None, invisible_loss_weight: float = 1.0,
        compatibility_config=None, compatibility_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 1.0, joint_flow_config=None,
    ):
        super().__init__()
        if model_mode not in {"factorized", "joint_flow"}:
            raise ValueError(f"Unknown model mode {model_mode!r}")
        if reconstruction_formulation not in {"discriminative", "flow_matching"}:
            raise ValueError(f"Unknown reconstruction formulation {reconstruction_formulation!r}")
        if feedback_enabled and not reasoning_enabled:
            raise ValueError("Soft reconstruction feedback requires recurrent reasoning")
        if not train_steps or any(step <= 0 for step in train_steps):
            raise ValueError(f"Recurrent train_steps must contain positive integers, found {train_steps}")
        if eval_steps <= 0:
            raise ValueError("Recurrent eval_steps must be positive")

        self.spec = spec
        self.model_mode = model_mode
        self.reconstruction_formulation = reconstruction_formulation
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
        self.compatibility_loss_weight = compatibility_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight

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
        if invisible_config is not None and bool(invisible_config.enabled):
            self.invisible = InvisibleGenerator(spec, d_model, invisible_config)

        self.assignment_flow = None
        if model_mode == "factorized" and reconstruction_formulation == "flow_matching":
            self.assignment_flow = DiscreteAssignmentFlow(
                spec, self.reconstruction, d_model=d_model, n_heads=n_heads,
                block_layers=reasoning_block_layers, d_feedforward=d_feedforward,
                dropout=dropout,
            )

        self.joint_flow = None
        if model_mode == "joint_flow":
            if self.invisible is None or joint_flow_config is None:
                raise ValueError("joint_flow requires invisible generation and joint_flow configuration")
            self.joint_flow = JointFlow(
                spec, self.reconstruction, self.invisible,
                d_model=d_model, n_heads=n_heads,
                block_layers=int(joint_flow_config.block_layers),
                d_feedforward=d_feedforward, dropout=dropout,
                recurrent_state=bool(joint_flow_config.recurrent_state),
                train_steps=tuple(int(step) for step in joint_flow_config.train_steps),
                assignment_loss_weight=reconstruction_loss_weight,
                invisible_loss_weight=invisible_loss_weight,
            )

        self.posterior_feedback = None
        self.coupling_config = coupling_config
        if model_mode == "factorized" and coupling_config is not None and bool(coupling_config.invisible_to_reconstruction.enabled):
            if not reasoning_enabled or self.invisible is None:
                raise ValueError("Invisible posterior feedback requires reasoning and an invisible generator")
            self.posterior_feedback = InvisiblePosteriorFeedback(
                spec, d_model, float(coupling_config.invisible_to_reconstruction.gate_initial_bias),
            )

        self.compatibility = None
        if model_mode == "factorized" and compatibility_config is not None and bool(compatibility_config.enabled):
            if self.invisible is None:
                raise ValueError("Compatibility scoring requires invisible generation")
            self.compatibility = CompatibilityHead(
                d_model, len(spec.invisible_roles),
                include_invisible=bool(compatibility_config.get("include_invisible", True)),
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
        if self.model_mode == "joint_flow":
            if "invisible_truth_scaled" not in batch:
                raise ValueError("joint_flow training requires scaled invisible truth")
            result = self.joint_flow.loss(visible, batch["structured"], batch["invisible_truth_scaled"])
            result["visible"] = visible
            return result

        if self.assignment_flow is not None:
            result = self.assignment_flow.loss(visible, batch["structured"])
            result["complete_assignment_logits"] = result.pop("assignment_logits")
            result["complete_assignment_probabilities"] = result.pop("assignment_probabilities")
            result["assignment_logits"] = result.pop("local_assignment_logits")
            result["assignment_probabilities"] = result.pop("local_assignment_probabilities")
            result["visible"] = visible
            result["loss"] = self.reconstruction_loss_weight * result["reconstruction_loss"]
            result["recurrent_steps"] = 0
            result["recurrent_predictions"] = ()
            return self._invisible_result(result, batch, visible, result.get("node_context"))

        return self._factorized_discriminative(batch, visible, recurrent_steps=recurrent_steps)

    def _factorized_discriminative(self, batch: dict, visible: dict, *, recurrent_steps: int | None) -> dict:
        if not self.reasoning_enabled:
            reconstruction = self.reconstruction(visible["tokens"], visible["pooled"], batch["structured"])
            result = {
                "visible": visible, **reconstruction,
                "loss": self.reconstruction_loss_weight * reconstruction["reconstruction_loss"],
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
        node_tokens = None
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
                    node_tokens, reconstruction["candidate_embeddings"], reconstruction["assignment_probabilities"],
                )
                state = self.reasoner.replace_topology(state, visible["tokens"].shape[1], updated)
                feedback_diagnostics.append(diagnostics)

        result = {
            "visible": visible, **reconstruction,
            "loss": self.reconstruction_loss_weight * reconstruction["reconstruction_loss"],
            "recurrent_steps": steps, "recurrent_predictions": tuple(predictions),
            "feedback_diagnostics": tuple(feedback_diagnostics), "recurrent_state": state,
            "node_context": node_tokens,
        }

        if self.posterior_feedback is not None and int(self.coupling_config.rounds) == 2:
            parent = self.invisible.parent_context(reconstruction, batch["structured"], pooled, node_tokens)
            feedback_config = self.coupling_config.invisible_to_reconstruction
            samples, nfe = self.invisible.sample(visible, parent, samples=int(feedback_config.posterior_samples))
            updated, gates = self.posterior_feedback(node_tokens, samples, detach=bool(feedback_config.detach))
            state = self.reasoner.replace_topology(state, visible["tokens"].shape[1], updated)
            state = self.reasoner.step(state, fixed, mask)
            pooled, encoded_tokens, node_tokens = self.reasoner.split(state, visible["tokens"].shape[1])
            reconstruction = self.reconstruction(encoded_tokens, pooled, batch["structured"], node_context=node_tokens)
            result.update(reconstruction)
            result["loss"] = self.reconstruction_loss_weight * reconstruction["reconstruction_loss"]
            result["recurrent_state"] = state
            result["node_context"] = node_tokens
            result["posterior_feedback"] = {"samples": samples, "nfe": nfe, "gates": gates, "detached": bool(feedback_config.detach)}
        return self._invisible_result(result, batch, visible, result.get("node_context"))

    def _invisible_result(self, result: dict, batch: dict, visible: dict, node_context: dict[str, torch.Tensor] | None) -> dict:
        if self.invisible is None:
            return result
        parent = self.invisible.parent_context(result, batch["structured"], visible["pooled"], node_context)
        result["invisible_parent_context"] = parent
        if "invisible_truth_scaled" in batch:
            invisible = self.invisible.loss(batch["invisible_truth_scaled"], visible, parent)
            result["invisible_loss"] = invisible["loss"]
            result["invisible_training"] = invisible
            result["loss"] = result["loss"] + self.invisible_loss_weight * invisible["loss"]
            if self.compatibility is not None:
                truth_summaries = []
                wrong_summaries = []
                for root in self.spec.roots:
                    candidates = result["candidate_embeddings"][root]
                    truth = batch["structured"]["pools"][root]["truth_index"]
                    if (truth < 0).any():
                        raise RuntimeError(f"Compatibility training requires reconstructable truth root {root}")
                    rows = torch.arange(len(candidates), device=candidates.device)
                    truth_summaries.append(candidates[rows, truth])
                    candidate_probabilities = result["assignment_probabilities"][root][:, 1:].clone()
                    candidate_probabilities[rows, truth] = -1
                    wrong = candidate_probabilities.argmax(dim=1)
                    wrong_summaries.append(candidates[rows, wrong])
                truth_assignment = torch.stack(truth_summaries, dim=1).mean(dim=1)
                wrong_assignment = torch.stack(wrong_summaries, dim=1).mean(dim=1)
                compatibility = self.compatibility.contrastive_loss(visible["pooled"], truth_assignment, wrong_assignment, batch["invisible_truth_scaled"])
                result["compatibility_training"] = compatibility
                result["compatibility_loss"] = compatibility["loss"]
                result["loss"] = result["loss"] + self.compatibility_loss_weight * compatibility["loss"]
        return result


StructuredEventModel = JointEventModel
