"""Composable topology-aware visible and invisible event reconstruction."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.models.compatibility import CompatibilityHead
from joint_event_reconstruction.models.feedback import SoftReconstructionFeedback
from joint_event_reconstruction.models.invisible.feedback import InvisiblePosteriorFeedback
from joint_event_reconstruction.models.invisible.generator import InvisibleGenerator
from joint_event_reconstruction.models.joint_flow import JointFlow
from joint_event_reconstruction.models.recurrent import RecurrentReasoner
from joint_event_reconstruction.models.reconstruction.discrete_flow import (
    DiscreteAssignmentFlow,
)
from joint_event_reconstruction.models.reconstruction.structured import (
    StructuredReconstruction,
)
from joint_event_reconstruction.models.visible.encoder import build_visible_encoder
from joint_event_reconstruction.topology.assignments import AssignmentSpace
from joint_event_reconstruction.topology.spec import TopologySpec


class JointEventModel(nn.Module):
    def __init__(
        self,
        spec: TopologySpec,
        *,
        n_features: int,
        n_object_types: int,
        token_schema,
        latent_config,
        visible_config,
        model_mode: str = "factorized",
        assignment_formulation: str = "discriminative",
        reasoning_enabled: bool = False,
        reasoning_block_layers: int = 2,
        train_steps: tuple[int, ...] = (1, 2, 3, 4),
        eval_steps: int = 4,
        fixed_input_reinjection: bool = True,
        feedback_enabled: bool = False,
        feedback_start_step: int = 1,
        feedback_gate_initial_bias: float = -2.0,
        invisible_config=None,
        invisible_inference_config=None,
        invisible_scaler=None,
        coupling_config=None,
        invisible_loss_weight: float = 1.0,
        compatibility_config=None,
        compatibility_loss_weight: float = 1.0,
        assignment_loss_weight: float = 1.0,
        joint_flow_config=None,
    ):
        super().__init__()
        if model_mode not in {"factorized", "joint_flow"}:
            raise ValueError(f"Unknown model mode {model_mode!r}")
        if assignment_formulation not in {"discriminative", "flow_matching"}:
            raise ValueError(
                f"Unknown assignment formulation {assignment_formulation!r}"
            )
        if feedback_enabled and not reasoning_enabled:
            raise ValueError("Soft assignment feedback requires recurrent reasoning")
        if not train_steps or any(step <= 0 for step in train_steps):
            raise ValueError("Recurrent train_steps must contain positive integers")
        if eval_steps <= 0:
            raise ValueError("Recurrent eval_steps must be positive")

        self.spec = spec
        self.model_mode = model_mode
        self.assignment_formulation = assignment_formulation
        d_model = int(latent_config.d_model)
        n_heads = int(latent_config.n_heads)
        d_feedforward = int(latent_config.d_feedforward)
        dropout = float(latent_config.dropout)
        self.visible = build_visible_encoder(
            visible_config,
            schema=token_schema,
            n_features=n_features,
            n_object_types=n_object_types,
        )
        # The implementation still reconstructs hierarchical topology candidates;
        # the public scientific quantity it scores is their assignment.
        self.reconstruction = StructuredReconstruction(spec, d_model, dropout)
        self.reasoning_enabled = reasoning_enabled
        self.feedback_enabled = feedback_enabled
        self.train_steps = tuple(train_steps)
        self.eval_steps = eval_steps
        self.feedback_start_step = feedback_start_step
        self.assignment_loss_weight = assignment_loss_weight
        self.invisible_loss_weight = invisible_loss_weight
        self.compatibility_loss_weight = compatibility_loss_weight

        self.reasoner = None
        self.feedback = None
        if reasoning_enabled:
            self.reasoner = RecurrentReasoner(
                spec,
                d_model,
                n_heads,
                reasoning_block_layers,
                d_feedforward,
                dropout,
                fixed_input_reinjection=fixed_input_reinjection,
            )
        if feedback_enabled:
            self.feedback = SoftReconstructionFeedback(
                spec, d_model, feedback_gate_initial_bias
            )

        self.invisible = None
        if invisible_config is not None and bool(invisible_config.enabled):
            if invisible_inference_config is None:
                raise ValueError("Invisible inference configuration is required")
            self.invisible = InvisibleGenerator(
                spec,
                d_model,
                invisible_config,
                invisible_inference_config,
                scaler=invisible_scaler,
            )

        self.assignment_flow = None
        if model_mode == "factorized" and assignment_formulation == "flow_matching":
            self.assignment_flow = DiscreteAssignmentFlow(
                spec,
                d_model=d_model,
                n_heads=n_heads,
                block_layers=reasoning_block_layers,
                d_feedforward=d_feedforward,
                dropout=dropout,
            )

        self.joint_flow = None
        if model_mode == "joint_flow":
            if self.invisible is None or joint_flow_config is None:
                raise ValueError(
                    "joint_flow requires invisible generation and joint_flow configuration"
                )
            self.joint_flow = JointFlow(
                spec,
                d_model=d_model,
                n_heads=n_heads,
                block_layers=int(joint_flow_config.block_layers),
                d_feedforward=d_feedforward,
                dropout=dropout,
                recurrent_state=bool(joint_flow_config.recurrent_state),
                train_steps=tuple(int(step) for step in joint_flow_config.train_steps),
                assignment_loss_weight=assignment_loss_weight,
                invisible_loss_weight=invisible_loss_weight,
            )

        self.posterior_feedback = None
        self.coupling_config = coupling_config
        if (
            model_mode == "factorized"
            and coupling_config is not None
            and bool(coupling_config.invisible_to_assignment.enabled)
        ):
            if not reasoning_enabled or self.invisible is None:
                raise ValueError(
                    "Invisible posterior feedback requires assignment reasoning and "
                    "an invisible generator"
                )
            self.posterior_feedback = InvisiblePosteriorFeedback(
                spec,
                d_model,
                float(coupling_config.invisible_to_assignment.gate_initial_bias),
            )

        self.compatibility = None
        if (
            model_mode == "factorized"
            and compatibility_config is not None
            and bool(compatibility_config.enabled)
        ):
            if self.invisible is None:
                raise ValueError("Compatibility scoring requires invisible generation")
            self.compatibility = CompatibilityHead(
                d_model,
                self.invisible.coordinates.token_count,
                include_invisible=bool(compatibility_config.include_invisible),
            )

    def _steps(self, requested: int | None) -> int:
        if requested is not None:
            if requested <= 0:
                raise ValueError("Requested recurrent steps must be positive")
            return requested
        if self.training:
            index = int(
                torch.randint(
                    len(self.train_steps),
                    (),
                    device=next(self.visible.parameters()).device,
                )
            )
            return self.train_steps[index]
        return self.eval_steps

    def forward(
        self, batch: dict, *, recurrent_steps: int | None = None
    ) -> dict:
        visible = self.encode_visible(batch)
        if self.model_mode == "joint_flow":
            if "invisible_truth_scaled" not in batch:
                raise ValueError("joint_flow training requires scaled invisible truth")
            result = self.joint_flow.loss(
                visible,
                batch["structured"],
                batch["invisible_truth_scaled"],
                batch["met_xy"],
                self.reconstruction,
                self.invisible,
            )
            result["visible"] = visible
            return result

        if self.assignment_flow is not None:
            result = self.assignment_flow.loss(
                visible, batch["structured"], self.reconstruction
            )
            result["complete_assignment_logits"] = result.pop("assignment_logits")
            result["complete_assignment_probabilities"] = result.pop(
                "assignment_probabilities"
            )
            result["assignment_logits"] = result.pop("local_assignment_logits")
            result["assignment_probabilities"] = result.pop(
                "local_assignment_probabilities"
            )
            result["visible"] = visible
            result["loss"] = self.assignment_loss_weight * result["assignment_loss"]
            result["recurrent_steps"] = 0
            result["recurrent_predictions"] = ()
            return self._invisible_result(
                result, batch, visible, result.get("node_context")
            )

        return self._factorized_discriminative(
            batch, visible, recurrent_steps=recurrent_steps
        )

    def _factorized_discriminative(
        self,
        batch: dict,
        visible: dict,
        *,
        recurrent_steps: int | None,
    ) -> dict:
        if not self.reasoning_enabled:
            assignment = self.reconstruction(
                visible["tokens"],
                visible["pooled"],
                batch["structured"],
            )
            result = {
                "visible": visible,
                **assignment,
                "loss": self.assignment_loss_weight * assignment["assignment_loss"],
                "recurrent_steps": 0,
                "recurrent_predictions": (),
            }
            return self._invisible_result(result, batch, visible, None)

        steps = self._steps(recurrent_steps)
        fixed = torch.cat(
            (visible["pooled"].unsqueeze(1), visible["tokens"]), dim=1
        )
        state = self.reasoner.initialise(visible["pooled"], visible["tokens"])
        mask = self.reasoner.mask(visible["padding_mask"])
        predictions = []
        feedback_diagnostics = []
        assignment = None
        node_tokens = None
        pooled = visible["pooled"]
        for step in range(steps):
            state = self.reasoner.step(state, fixed, mask)
            pooled, encoded_tokens, node_tokens = self.reasoner.split(
                state, visible["tokens"].shape[1]
            )
            assignment = self.reconstruction(
                encoded_tokens,
                pooled,
                batch["structured"],
                node_context=node_tokens,
                calculate_loss=step == steps - 1,
            )
            predictions.append(
                {
                    "assignment_probabilities": assignment["assignment_probabilities"],
                    "assignment_logits": assignment["assignment_logits"],
                    "token_norm": state.norm(dim=-1).mean(dim=1),
                }
            )
            if (
                self.feedback_enabled
                and step + 1 >= self.feedback_start_step
                and step < steps - 1
            ):
                updated, diagnostics = self.feedback(
                    node_tokens,
                    assignment["candidate_embeddings"],
                    assignment["assignment_probabilities"],
                )
                state = self.reasoner.replace_topology(
                    state,
                    visible["tokens"].shape[1],
                    updated,
                )
                feedback_diagnostics.append(diagnostics)

        result = {
            "visible": visible,
            **assignment,
            "loss": self.assignment_loss_weight * assignment["assignment_loss"],
            "recurrent_steps": steps,
            "recurrent_predictions": tuple(predictions),
            "feedback_diagnostics": tuple(feedback_diagnostics),
            "recurrent_state": state,
            "node_context": node_tokens,
        }

        if self.posterior_feedback is not None:
            parent = self.invisible.parent_context(
                assignment,
                batch["structured"],
                pooled,
                node_tokens,
            )
            feedback_config = self.coupling_config.invisible_to_assignment
            samples, nfe = self.invisible.sample(
                visible,
                parent,
                samples=int(feedback_config.posterior_samples),
                budget=int(feedback_config.training_sampler.budget),
                sampler=feedback_config.training_sampler.solver,
            )
            if self.invisible.scaler is None:
                raise RuntimeError("C2 sampling requires invisible scaler metadata")
            physical_samples = self.invisible.decode_state(
                samples, batch["met_xy"], self.invisible.scaler
            )
            updated, gates = self.posterior_feedback(
                node_tokens,
                physical_samples,
                detach=True,
            )
            state = self.reasoner.replace_topology(
                state,
                visible["tokens"].shape[1],
                updated,
            )
            state = self.reasoner.step(state, fixed, mask)
            pooled, encoded_tokens, node_tokens = self.reasoner.split(
                state, visible["tokens"].shape[1]
            )
            assignment = self.reconstruction(
                encoded_tokens,
                pooled,
                batch["structured"],
                node_context=node_tokens,
            )
            result.update(assignment)
            result["loss"] = self.assignment_loss_weight * assignment["assignment_loss"]
            result["recurrent_state"] = state
            result["node_context"] = node_tokens
            result["posterior_feedback"] = {
                "samples": physical_samples,
                "training_sampler": {
                    "formulation": str(feedback_config.training_sampler.formulation),
                    "budget": int(feedback_config.training_sampler.budget),
                    "solver": feedback_config.training_sampler.solver,
                },
                "nfe": nfe,
                "gates": gates,
                "detached": True,
            }
        return self._invisible_result(
            result,
            batch,
            visible,
            result.get("node_context"),
        )

    def encode_visible(self, batch: dict) -> dict:
        return self.visible(batch)

    def _hypothesis_training_mask(self, batch: dict) -> torch.Tensor:
        valid = torch.ones(
            len(batch["tokens"]),
            dtype=torch.bool,
            device=batch["tokens"].device,
        )
        for role in self.spec.invisible_roles:
            valid &= batch["structured"]["pools"][role.node]["reconstructable"]
        return valid

    def _compatibility_assignment_summaries(
        self, batch: dict, result: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return truth and globally feasible wrong complete assignments."""
        space = AssignmentSpace.from_structured(batch["structured"])
        rows = torch.arange(
            len(batch["tokens"]), device=batch["tokens"].device
        )
        valid_states = space.valid_mask & space.non_null_mask.unsqueeze(0)
        truth_state = space.truth_state
        truth_valid = valid_states[rows, truth_state]

        state_log_score = torch.zeros(
            valid_states.shape,
            dtype=result["assignment_probabilities"][space.roots[0]].dtype,
            device=rows.device,
        )
        for root_index, root in enumerate(space.roots):
            selected = space.root_candidate_indices[:, root_index]
            probabilities = result["assignment_probabilities"][root]
            state_log_score = state_log_score + probabilities[:, selected + 1].clamp_min(
                1.0e-12
            ).log()

        wrong_states = valid_states.clone()
        wrong_states[rows, truth_state] = False
        wrong_valid = wrong_states.any(dim=1)
        wrong_state = state_log_score.masked_fill(~wrong_states, -torch.inf).argmax(
            dim=1
        )
        valid = truth_valid & wrong_valid

        def summary(states: torch.Tensor) -> torch.Tensor:
            root_indices = space.root_candidate_indices[states].clamp_min(0)
            return torch.stack(
                [
                    result["candidate_embeddings"][root][rows, root_indices[:, index]]
                    for index, root in enumerate(space.roots)
                ],
                dim=1,
            ).mean(dim=1)

        return valid, summary(truth_state), summary(wrong_state)

    def _invisible_result(
        self,
        result: dict,
        batch: dict,
        visible: dict,
        node_context: dict[str, torch.Tensor] | None,
    ) -> dict:
        if self.invisible is None:
            return result
        role_parent = self.invisible.role_parent_context(
            result,
            batch["structured"],
            visible["pooled"],
            node_context,
        )
        parent = self.invisible.coordinates.parent_context(
            role_parent, visible["pooled"]
        )
        result["invisible_parent_context"] = parent
        event_mask = None
        if self.invisible.assignment_mode in {"hypothesis", "truth"}:
            event_mask = self._hypothesis_training_mask(batch)
            result["invisible_training_mask"] = event_mask
            result["invisible_training_valid_fraction"] = event_mask.to(
                visible["tokens"].dtype
            ).mean()
        if "invisible_truth_scaled" not in batch:
            return result

        invisible = self.invisible.loss(
            batch["invisible_truth_scaled"],
            visible,
            parent,
            role_context=role_parent,
            event_mask=event_mask,
        )
        result["invisible_loss"] = invisible["loss"]
        result["invisible_training"] = invisible
        result["loss"] = result["loss"] + self.invisible_loss_weight * invisible["loss"]
        if self.compatibility is None:
            return result

        valid, truth_assignment, wrong_assignment = (
            self._compatibility_assignment_summaries(batch, result)
        )

        result["compatibility_training_valid_fraction"] = valid.to(
            visible["pooled"].dtype
        ).mean()
        if valid.any():
            compatibility = self.compatibility.contrastive_loss(
                visible["pooled"][valid],
                truth_assignment[valid],
                wrong_assignment[valid],
                batch["invisible_truth_scaled"][valid],
            )
        else:
            zero = visible["pooled"].sum() * 0.0
            for parameter in self.compatibility.parameters():
                zero = zero + parameter.sum() * 0.0
            empty = visible["pooled"].new_empty((0,))
            compatibility = {
                "loss": zero,
                "positive_logit": empty,
                "wrong_assignment_logit": empty,
                "mismatched_invisible_logit": empty,
            }

        result["compatibility_training"] = compatibility
        result["compatibility_loss"] = compatibility["loss"]
        result["loss"] = (
            result["loss"]
            + self.compatibility_loss_weight * compatibility["loss"]
        )
        return result


StructuredEventModel = JointEventModel
