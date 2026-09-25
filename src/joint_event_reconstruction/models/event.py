"""Composable topology-aware visible and invisible event reconstruction."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.models.compatibility import CompatibilityHead
from joint_event_reconstruction.models.feedback import SoftReconstructionFeedback
from joint_event_reconstruction.models.invisible.feedback import InvisiblePosteriorFeedback
from joint_event_reconstruction.models.invisible.generator import InvisibleGenerator
from joint_event_reconstruction.models.invisible.heads import (
    DirectInvisiblePointHead,
    DirectParentFourVectorHead,
)
from joint_event_reconstruction.models.partonic import PartonicFourVectorHead
from joint_event_reconstruction.models.partonic import PartonicCFM
from joint_event_reconstruction.models.topology_queries import TopologyQueryBlock
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
        heads_config=None,
        point_scaler=None,
        point_loss_weight: float = 0.0,
        parent_loss_weight: float = 0.0,
        shared_encoder_gradients: str = "both",
        topology_queries_config=None,
        partonic_config=None,
        partonic_scaler=None,
        partonic_loss_weight: float = 0.0,
        partonic_shared_encoder_gradients: str = "both",
        parent_cfm_config=None,
        parent_cfm_scaler=None,
        parent_cfm_loss_weight: float = 0.0,
        parent_cfm_shared_encoder_gradients: str = "both",
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
        self.topology_queries = None
        self.query_names = tuple()
        if topology_queries_config is not None and bool(topology_queries_config.get("enabled", False)):
            raw_roles = topology_queries_config.get("roles", ())
            query_roles = []
            for role in raw_roles:
                item = {str(key): value for key, value in dict(role).items()}
                node = item.get("node")
                if node is not None and str(node) not in spec.nodes:
                    raise ValueError(
                        f"Topology query {item.get('name')!r} refers to unknown compiled node {node!r}"
                    )
                query_roles.append(item)
            invisible_roles = tuple(spec.invisible_roles)
            self.topology_queries = TopologyQueryBlock(
                d_model,
                int(topology_queries_config.get("n_heads", n_heads)),
                query_roles=query_roles,
                interaction=str(topology_queries_config.get("interaction", "full")),
                layers=int(topology_queries_config.get("layers", 1)),
                dropout=float(topology_queries_config.get("dropout", dropout)),
                relation_edges=tuple(
                    (int(edge[0]), int(edge[1]))
                    for edge in topology_queries_config.get("relation_edges", ())
                ),
                required_node_names=spec.dag_order,
                invisible_role_names=tuple(role.key for role in invisible_roles),
                invisible_role_nodes=tuple(role.node for role in invisible_roles),
                topology_identity=spec.signature,
            )
            self.query_names = self.topology_queries.query_names
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
        self.point_loss_weight = float(point_loss_weight)
        self.parent_loss_weight = float(parent_loss_weight)
        if shared_encoder_gradients not in {"both", "assignment_only"}:
            raise ValueError("shared_encoder_gradients must be both or assignment_only")
        self.shared_encoder_gradients = str(shared_encoder_gradients)
        self.point_scaler = point_scaler

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

        heads_config = heads_config or {}
        point_config = heads_config.get("point_invisible") if isinstance(heads_config, dict) else getattr(heads_config, "point_invisible", None)
        parent_config = heads_config.get("parent_four_vector") if isinstance(heads_config, dict) else getattr(heads_config, "parent_four_vector", None)
        point_enabled = bool(point_config.get("enabled", False) if isinstance(point_config, dict) else getattr(point_config, "enabled", False))
        parent_enabled = bool(parent_config.get("enabled", False) if isinstance(parent_config, dict) else getattr(parent_config, "enabled", False))
        self.point_head = None
        if point_enabled:
            if self.invisible is None:
                raise ValueError("The direct invisible point head requires an active invisible model")
            if point_scaler is None:
                raise ValueError("The direct invisible point head requires a train-fitted Cartesian scaler")
            unsupported_masses = {
                role: shell.mode
                for role, shell in self.invisible.coordinates.mass_shells.items()
                if shell.mode != "fixed"
            }
            if unsupported_masses:
                raise ValueError(
                    "The direct point head currently requires fixed declared masses "
                    f"for every invisible role; unsupported={unsupported_masses}"
                )
            self.point_head = DirectInvisiblePointHead(d_model, len(spec.invisible_roles))
        self.parent_head = None
        self.parent_roles = tuple()
        if parent_enabled:
            raw_roles = parent_config.get("roles", ()) if isinstance(parent_config, dict) else getattr(parent_config, "roles", ())
            self.parent_roles = tuple(str(role) for role in raw_roles)
            if not self.parent_roles:
                raise ValueError("The parent four-vector head requires non-empty configured roles")
            self.parent_head = DirectParentFourVectorHead(d_model, len(self.parent_roles))

        self.partonic_head = None
        self.partonic_roles = tuple()
        self.partonic_scaler = partonic_scaler
        self.partonic_loss_weight = float(partonic_loss_weight)
        if partonic_config is not None and bool(partonic_config.get("enabled", False)):
            if self.topology_queries is None:
                raise ValueError("The partonic head requires enabled topology queries")
            targets = tuple(
                str(target["role"])
                for target in partonic_config.get("targets", ())
            )
            if not targets:
                raise ValueError("Enabled partonic head requires non-empty targets")
            if partonic_scaler is None:
                raise ValueError("Enabled partonic head requires a train-fitted partonic scaler")
            self.partonic_roles = targets
            self.partonic_head = PartonicFourVectorHead(d_model, targets)
        if partonic_shared_encoder_gradients not in {"both", "assignment_only"}:
            raise ValueError(
                "partonic_shared_encoder_gradients must be both or assignment_only"
            )
        self.partonic_shared_encoder_gradients = str(partonic_shared_encoder_gradients)

        self.parent_cfm = None
        self.parent_cfm_loss_weight = float(parent_cfm_loss_weight)
        self.parent_cfm_config = parent_cfm_config
        if parent_cfm_config is not None and bool(parent_cfm_config.get("enabled", False)):
            if self.topology_queries is None:
                raise ValueError("Partonic CFM requires the shared topology-query module")
            if self.topology_queries.parent_role_names != ("top", "antitop"):
                raise ValueError(
                    "Partonic CFM queries must be declared in (top, antitop) order "
                    "with kind=parent"
                )
            conditioning = str(parent_cfm_config.get("conditioning", "P0"))
            if conditioning in {"P1", "P2"} and set(spec.roots) != {"top", "antitop"}:
                raise ValueError(
                    "P1/P2 require exactly the compiled root nodes top and antitop "
                    "to derive complete predicted assignment context"
                )
            if conditioning == "P2":
                if self.invisible is None:
                    raise ValueError("P2 requires the existing invisible CFM generator")
                if self.invisible.formulation_name != "flow_matching":
                    raise ValueError("P2 currently samples the existing invisible CFM only")
                if self.invisible.assignment_mode in {"truth", "hypothesis"}:
                    raise ValueError("P2 deployable conditioning cannot use truth/hypothesis assignments")
                role_nodes = tuple(role.node for role in spec.invisible_roles)
                if any(role_nodes.count(name) != 1 for name in ("top", "antitop")):
                    raise ValueError(
                        "P2 requires exactly one compiled invisible role on each parent side"
                    )
            if parent_cfm_scaler is None:
                raise ValueError("Partonic CFM requires the train-fitted parent scaler")
            self.parent_cfm = PartonicCFM(
                d_model=d_model,
                n_heads=int(parent_cfm_config.get("n_heads", n_heads)),
                layers=int(parent_cfm_config.get("layers", 2)),
                d_feedforward=int(parent_cfm_config.get("d_feedforward", d_feedforward)),
                dropout=float(parent_cfm_config.get("dropout", dropout)),
                scaler=parent_cfm_scaler,
                conditioning=conditioning,
            )
        if parent_cfm_shared_encoder_gradients not in {"both", "assignment_only"}:
            raise ValueError(
                "parent_cfm_shared_encoder_gradients must be both or assignment_only"
            )
        self.parent_cfm_shared_encoder_gradients = str(
            parent_cfm_shared_encoder_gradients
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
                invisible_state_dim=self.invisible.coordinates.state_dim,
            )

        self.posterior_feedback = None
        self.coupling_config = coupling_config
        if (
            model_mode == "factorized"
            and coupling_config is not None
            and bool(coupling_config.invisible_to_assignment.enabled)
        ):
            if self.invisible is None:
                raise ValueError(
                    "Invisible posterior feedback requires an invisible generator"
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
                invisible_state_dim=self.invisible.coordinates.state_dim,
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
        self,
        batch: dict,
        *,
        recurrent_steps: int | None = None,
        compute_objectives: bool = False,
    ) -> dict:
        visible = self.encode_visible(batch)
        if (
            compute_objectives
            and self.model_mode == "factorized"
            and self.assignment_flow is None
            and not self.reasoning_enabled
            and self.posterior_feedback is None
            and self.compatibility is None
        ):
            return self._objective_forward(batch, visible)
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

    def _objective_forward(self, batch: dict, visible: dict) -> dict:
        """Execute only the dependencies of the configured training losses."""
        parent_conditioning = (
            None if self.parent_cfm is None else self.parent_cfm.conditioning
        )
        assignment_for_invisible = (
            self.invisible is not None
            and self.invisible_loss_weight > 0
            and self.invisible.assignment_mode != "none"
        )
        assignment_for_parent = parent_conditioning in {"P1", "P2"}
        need_assignment = (
            self.assignment_loss_weight > 0
            or assignment_for_invisible
            or assignment_for_parent
        )
        zero = visible["pooled"].sum() * 0.0
        result = {
            "visible": visible,
            "loss": zero,
            "recurrent_steps": 0,
            "recurrent_predictions": (),
        }
        if need_assignment:
            assignment = self.reconstruction(
                visible["tokens"],
                visible["pooled"],
                batch["structured"],
                node_context=visible.get("node_context"),
                calculate_loss=self.assignment_loss_weight > 0,
            )
            result.update(assignment)
            if "assignment_loss" in assignment:
                result["loss"] = (
                    self.assignment_loss_weight * assignment["assignment_loss"]
                )
        if self.invisible is not None and self.invisible_loss_weight > 0:
            return self._invisible_result(
                result, batch, visible, result.get("node_context")
            )
        return self._optional_head_result(result, batch, visible)

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
                node_context=visible.get("node_context"),
            )
            result = {
                "visible": visible,
                **assignment,
                "loss": self.assignment_loss_weight * assignment["assignment_loss"],
                "recurrent_steps": 0,
                "recurrent_predictions": (),
            }
            if self.posterior_feedback is not None:
                result = self._one_pass_posterior_correction(
                    result, batch, visible, assignment
                )
            return self._invisible_result(
                result, batch, visible, result.get("node_context")
            )

        steps = self._steps(recurrent_steps)
        fixed = torch.cat(
            (visible["pooled"].unsqueeze(1), visible["tokens"]), dim=1
        )
        state = self.reasoner.initialise(
            visible["pooled"],
            visible["tokens"],
            node_context=visible.get("node_context"),
        )
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
                query_role_context=visible.get("invisible_role_context"),
                query_role_names=tuple(visible.get("invisible_role_names", ())),
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

    def _one_pass_posterior_correction(
        self, result: dict, batch: dict, visible: dict, assignment: dict
    ) -> dict:
        """Apply one detached posterior-informed assignment update.

        This narrow C2 diagnostic starts from the selected one-pass
        assignment, samples one invisible posterior, and updates the
        assignment scorer once. It deliberately does not create a
        visible-only recurrent loop.
        """
        feedback_config = self.coupling_config.invisible_to_assignment
        parent = self.invisible.parent_context(
            assignment,
            batch["structured"],
            visible["pooled"],
            None,
            query_role_context=visible.get("invisible_role_context"),
            query_role_names=tuple(visible.get("invisible_role_names", ())),
        )
        with torch.no_grad():
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

        # With no persistent reasoner, the probability-weighted candidate
        # embeddings are the explicit initial assignment state.
        initial_node_context = {
            node: (
                assignment["candidate_embeddings"][node]
                * assignment["assignment_probabilities"][node][:, 1:].unsqueeze(-1)
            ).sum(dim=1)
            for node in self.spec.dag_order
        }
        updated, gates = self.posterior_feedback(
            initial_node_context,
            physical_samples,
            detach=True,
        )
        logits, probabilities = self.reconstruction.score_candidates(
            assignment["candidate_embeddings"],
            visible["pooled"],
            batch["structured"],
            node_context=updated,
        )
        assignment_update = self.reconstruction.assignment_loss(
            logits, batch["structured"]
        )
        result.update(
            assignment_update,
            assignment_logits=logits,
            assignment_probabilities=probabilities,
            node_context=updated,
        )
        result["loss"] = self.assignment_loss_weight * assignment_update["assignment_loss"]
        result["posterior_feedback"] = {
            "samples": physical_samples,
            "nfe": nfe,
            "gates": gates,
            "detached": True,
            "correction_steps": 1,
            "base_recurrent_steps": 0,
        }
        return result

    def encode_visible(self, batch: dict) -> dict:
        encoded = self.visible(batch)
        if self.topology_queries is not None:
            query = self.topology_queries(
                encoded["pooled"], encoded["tokens"], encoded["padding_mask"]
            )
            if query.topology_identity != self.spec.signature:
                raise RuntimeError(
                    "Topology-query state was compiled for a different topology"
                )
            encoded["query_states"] = query.query_states
            encoded["query_summary"] = query.query_summary
            encoded["query_context"] = query.global_context
            encoded["node_context"] = query.node_contexts
            encoded["invisible_role_names"] = query.invisible_role_names
            encoded["invisible_role_context"] = query.invisible_contexts
            encoded["parent_role_names"] = query.parent_role_names
            encoded["parent_role_context"] = query.parent_contexts
            # Candidate scoring and invisible flow conditioning use the
            # individual compiled role states. The pooled query summary is
            # retained only as the global event context.
            encoded["pooled"] = query.global_context
        # Keep physical conditioning alongside the learned visible embedding.
        # Conventional invisible backbones ignore these optional fields; the
        # Conditional L-GATr backbone consumes the p4 and MET channels.
        if "physics_vectors" in batch:
            encoded["physics_vectors"] = batch["physics_vectors"]
        if "met_xy" in batch:
            encoded["met_xy"] = batch["met_xy"]
        return encoded

    def _parent_assignment_context(
        self, result: dict, structured: dict
    ) -> torch.Tensor:
        """Marginalise predicted root assignments over feasible complete states."""
        space = AssignmentSpace.from_structured(structured)
        probabilities = space.state_probabilities(result["assignment_logits"])
        rows = torch.arange(len(probabilities), device=probabilities.device)
        contexts = []
        for role in PartonicCFM.ROLES:
            if role not in space.roots:
                raise ValueError(
                    f"P1 assignment context needs compiled root role {role!r}; "
                    f"roots={space.roots}"
                )
            root_index = space.roots.index(role)
            selected = space.root_candidate_indices[:, root_index]
            embeddings = result["candidate_embeddings"][role]
            if embeddings.shape[1] == 0:
                raise ValueError(f"P1 has no candidate embeddings for parent role {role!r}")
            weights = probabilities.new_zeros(
                (len(probabilities), embeddings.shape[1] + 1)
            )
            weights.scatter_add_(
                1,
                (selected + 1).unsqueeze(0).expand(len(probabilities), -1),
                probabilities,
            )
            candidate_summary = torch.sum(
                weights[:, 1:, None] * embeddings, dim=1
            )
            null = self.reconstruction.null_embeddings[role].unsqueeze(0)
            contexts.append(candidate_summary + weights[:, :1] * null)
        return torch.stack(contexts, dim=1)

    def _predicted_neutrino_sample(
        self,
        result: dict,
        batch: dict,
        visible: dict,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.invisible is None:
            raise RuntimeError("P2 requires an active invisible generator")
        if "met_xy" not in batch:
            raise KeyError("P2 requires measured MET conditioning")
        role_parent = self.invisible.role_parent_context(
            result,
            batch["structured"],
            visible["pooled"],
            result.get("node_context", visible.get("node_context")),
            mode=self.invisible.assignment_mode,
            query_role_context=visible.get("invisible_role_context"),
            query_role_names=tuple(visible.get("invisible_role_names", ())),
        )
        parent = self.invisible.coordinates.parent_context(role_parent, visible["pooled"])
        settings = self.parent_cfm_config
        steps = int(settings.get("p2_neutrino_steps", self.invisible.inference_budget))
        solver = str(settings.get("p2_neutrino_solver", self.invisible.sampler))
        samples, nfe = self.invisible.sample(
            visible,
            parent,
            samples=1,
            budget=steps,
            sampler=solver,
            generator=generator,
        )
        physical = self.invisible.decode_state(samples, batch["met_xy"], self.invisible.scaler)
        node_to_index = {
            role.node: index for index, role in enumerate(self.spec.invisible_roles)
        }
        if set(("top", "antitop")) - set(node_to_index):
            raise ValueError("P2 cannot map invisible roles to top/antitop parent order")
        ordered = torch.stack(
            (physical[:, 0, node_to_index["top"]], physical[:, 0, node_to_index["antitop"]]),
            dim=1,
        )
        return ordered, samples, nfe

    def sample_partonic(
        self,
        batch: dict,
        *,
        samples: int,
        steps: int,
        solver: str,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Truth-free P0/P1/P2 parent sampling with explicit upstream indices."""
        if self.parent_cfm is None:
            raise RuntimeError("Partonic CFM is not enabled")
        visible = self.encode_visible(batch)
        conditioning = self.parent_cfm.conditioning
        assignment = None
        assignment_context = None
        predicted_neutrino = None
        neutrino_sample_id = torch.full(
            (len(visible["tokens"]),), -1, dtype=torch.long,
            device=visible["tokens"].device,
        )
        if conditioning in {"P1", "P2"}:
            if "structured" not in batch:
                raise KeyError("P1/P2 prediction requires structured candidate tensors")
            assignment = self.reconstruction(
                visible["tokens"],
                visible["pooled"],
                batch["structured"],
                node_context=visible.get("node_context"),
                calculate_loss=False,
            )
            assignment_context = self._parent_assignment_context(
                assignment, batch["structured"]
            )
        if conditioning == "P2":
            if assignment is None:
                raise RuntimeError("P2 prediction requires its predicted assignment context")
            predicted_neutrino, _, _ = self._predicted_neutrino_sample(
                assignment, batch, visible, generator=generator
            )
            neutrino_sample_id.zero_()
        parent_queries = visible.get("parent_role_context")
        if tuple(visible.get("parent_role_names", ())) != PartonicCFM.ROLES:
            raise RuntimeError("Prediction parent queries are not ordered as (top, antitop)")
        if self.parent_cfm_shared_encoder_gradients == "assignment_only":
            visible_context = visible["tokens"].detach()
            visible_mask = visible["padding_mask"]
            parent_queries = parent_queries.detach()
            if assignment_context is not None:
                assignment_context = assignment_context.detach()
            if predicted_neutrino is not None:
                predicted_neutrino = predicted_neutrino.detach()
        else:
            visible_context = visible["tokens"]
            visible_mask = visible["padding_mask"]
        # Retain the global measured-event query context as a token.
        visible_context = torch.cat((visible["pooled"].unsqueeze(1), visible_context), dim=1)
        visible_mask = torch.cat(
            (torch.zeros((len(visible_mask), 1), dtype=torch.bool, device=visible_mask.device), visible_mask),
            dim=1,
        )
        output = self.parent_cfm.sample(
            visible_context,
            visible_mask,
            parent_queries,
            samples=samples,
            steps=steps,
            solver=solver,
            generator=generator,
            assignment_context=assignment_context,
            predicted_neutrino_momentum=predicted_neutrino,
        )
        batch_size = len(visible_context)
        sample_indices = torch.arange(samples, device=visible_context.device).expand(batch_size, -1)
        output["event_id"] = batch.get("event_index", batch.get("event_id"))
        output["assignment_candidate_indices"] = (
            None if assignment is None else {
                name: probabilities.argmax(dim=1) - 1
                for name, probabilities in assignment["assignment_probabilities"].items()
                if name in PartonicCFM.ROLES
            }
        )
        output["assignment_hypothesis_index"] = torch.full_like(sample_indices, -1)
        output["invisible_sample_index"] = neutrino_sample_id.unsqueeze(1).expand_as(sample_indices)
        output["parent_sample_index"] = sample_indices
        return output

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
            return self._optional_head_result(result, batch, visible)
        invisible_visible = visible
        invisible_result = result
        if self.shared_encoder_gradients == "assignment_only":
            invisible_visible = {
                key: value.detach() if torch.is_tensor(value) else value
                for key, value in visible.items()
            }
            invisible_result = dict(result)
            for key in ("candidate_embeddings", "assignment_probabilities", "assignment_logits"):
                value = result.get(key)
                if isinstance(value, dict):
                    invisible_result[key] = {
                        name: tensor.detach() if torch.is_tensor(tensor) else tensor
                        for name, tensor in value.items()
                    }
            if isinstance(result.get("node_context"), dict):
                invisible_result["node_context"] = {
                    name: tensor.detach() if torch.is_tensor(tensor) else tensor
                    for name, tensor in result["node_context"].items()
                }
            if torch.is_tensor(invisible_visible.get("invisible_role_context")):
                invisible_visible["invisible_role_context"] = invisible_visible[
                    "invisible_role_context"
                ].detach()
        role_parent = self.invisible.role_parent_context(
            invisible_result,
            batch["structured"],
            invisible_visible["pooled"],
            invisible_result.get("node_context", node_context),
            query_role_context=invisible_visible.get("invisible_role_context"),
            query_role_names=tuple(invisible_visible.get("invisible_role_names", ())),
        )
        parent = self.invisible.coordinates.parent_context(
            role_parent, invisible_visible["pooled"]
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
            return self._optional_head_result(result, batch, visible)

        invisible = self.invisible.loss(
            batch["invisible_truth_scaled"],
            invisible_visible,
            parent,
            role_context=role_parent,
            event_mask=event_mask,
        )
        result["invisible_loss"] = invisible["loss"]
        result["invisible_training"] = invisible
        result["loss"] = result["loss"] + self.invisible_loss_weight * invisible["loss"]
        if self.compatibility is None:
            return self._optional_head_result(result, batch, visible)

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
        return self._optional_head_result(result, batch, visible)

    def _optional_head_result(self, result: dict, batch: dict, visible: dict) -> dict:
        """Attach optional deployable point outputs and supervised losses."""
        if self.point_head is not None:
            scaled = self.point_head(visible["pooled"])
            result["point_momenta_scaled"] = scaled
            target = batch.get("invisible_truth_point_scaled")
            if target is None:
                if self.training:
                    raise ValueError(
                        "Direct point-head training requires invisible_truth_point_scaled"
                    )
            else:
                if target.shape != scaled.shape or not torch.isfinite(target).all():
                    raise ValueError(
                        "Direct point-head target must be finite and match [B,R,3]: "
                        f"prediction={tuple(scaled.shape)}, target={tuple(target.shape)}"
                    )
                support = batch.get("invisible_point_support")
                if support is None:
                    raise ValueError(
                        "Direct point-head training requires invisible_point_support"
                    )
                if support.shape != scaled.shape[:2] or support.dtype != torch.bool:
                    raise ValueError(
                        "invisible_point_support must be boolean [B,R], found "
                        f"{tuple(support.shape)} {support.dtype}"
                    )
                supported = support.unsqueeze(-1).expand_as(scaled)
                if not support.any():
                    raise RuntimeError(
                        "Direct point-head training batch has no supported invisible targets"
                    )
                loss = (scaled - target).square().masked_select(supported).mean()
                result["point_loss"] = loss
                result["point_training_support_fraction"] = support.to(
                    scaled.dtype
                ).mean()
                result["loss"] = result["loss"] + self.point_loss_weight * loss

        if self.parent_head is not None:
            prediction = self.parent_head(visible["pooled"])
            result["parent_four_vectors"] = prediction
            target = batch.get("parent_truth")
            if target is None:
                if self.training:
                    raise ValueError(
                        "Direct parent-head training requires the configured parent_truth target"
                    )
            else:
                if target.shape != prediction.shape or not torch.isfinite(target).all():
                    raise ValueError(
                        "Parent target must be finite and match [B,P,4]: "
                        f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}"
                    )
                loss = (prediction - target).square().mean()
                result["parent_loss"] = loss
                result["loss"] = result["loss"] + self.parent_loss_weight * loss

        if self.partonic_head is not None and (
            not self.training or self.partonic_loss_weight > 0.0
        ):
            if "query_states" not in visible:
                raise RuntimeError("Partonic head has no topology-query states")
            query_states = visible["query_states"]
            if self.partonic_shared_encoder_gradients == "assignment_only":
                query_states = query_states.detach()
            scaled_prediction = self.partonic_head(query_states, self.query_names)
            if self.partonic_scaler is None:
                raise RuntimeError("Partonic head has no target scaler")
            prediction = self.partonic_scaler.inverse(scaled_prediction)
            result["partonic_features_scaled"] = scaled_prediction
            result["partonic_four_vectors"] = prediction
            target = batch.get("partonic_truth_scaled")
            if target is None:
                if self.training:
                    raise ValueError(
                        "Partonic-head training requires partonic_truth_scaled"
                    )
            else:
                if target.shape != scaled_prediction.shape or not torch.isfinite(target).all():
                    raise ValueError(
                        "Partonic target must be finite and match [B,P,4]: "
                        f"prediction={tuple(scaled_prediction.shape)}, target={tuple(target.shape)}"
                    )
                support = batch.get("partonic_support")
                if support is None or support.shape != target.shape[:2] or support.dtype != torch.bool:
                    raise ValueError("partonic_support must be boolean [B,P]")
                # A minibatch may contain only unsupported targets.  This
                # weighted reduction gives a gradient-safe exact zero without
                # synchronising a device scalar to Python.
                weights = support.unsqueeze(-1).to(scaled_prediction.dtype)
                loss = (
                    (scaled_prediction - target).square() * weights
                ).sum() / (weights.sum() * target.shape[-1]).clamp_min(1.0)
                result["partonic_loss"] = loss
                result["partonic_training_support_fraction"] = support.to(
                    scaled_prediction.dtype
                ).mean()
                result["loss"] = result["loss"] + self.partonic_loss_weight * loss

        if self.parent_cfm is not None and (
            not self.training or self.parent_cfm_loss_weight > 0.0
        ):
            target = batch.get("partonic_truth")
            support = batch.get("partonic_support")
            if target is None or support is None:
                if self.training:
                    raise ValueError(
                        "Partonic CFM training requires explicit after-FSR truth and support"
                    )
                return result
            parent_queries = visible.get("parent_role_context")
            parent_names = tuple(visible.get("parent_role_names", ()))
            if parent_names != PartonicCFM.ROLES or parent_queries is None:
                raise RuntimeError(
                    "Partonic CFM requires ordered shared (top, antitop) query contexts"
                )
            assignment_context = None
            if self.parent_cfm.conditioning in {"P1", "P2"}:
                assignment_context = self._parent_assignment_context(
                    result, batch["structured"]
                )
            predicted_neutrino = None
            if self.parent_cfm.conditioning == "P2":
                predicted_neutrino, _, nfe = self._predicted_neutrino_sample(
                    result, batch, visible
                )
                result["partonic_cfm_neutrino_nfe"] = nfe
            if self.parent_cfm_shared_encoder_gradients == "assignment_only":
                measured = visible["tokens"].detach()
                measured_global = visible["pooled"].detach()
                parent_queries = parent_queries.detach()
                if assignment_context is not None:
                    assignment_context = assignment_context.detach()
                if predicted_neutrino is not None:
                    predicted_neutrino = predicted_neutrino.detach()
            else:
                measured = visible["tokens"]
                measured_global = visible["pooled"]
            measured = torch.cat((measured_global.unsqueeze(1), measured), dim=1)
            measured_mask = torch.cat(
                (
                    torch.zeros(
                        (len(measured), 1), dtype=torch.bool, device=measured.device
                    ),
                    visible["padding_mask"],
                ),
                dim=1,
            )
            cfm = self.parent_cfm.loss(
                target,
                support,
                measured,
                measured_mask,
                parent_queries,
                assignment_context=assignment_context,
                predicted_neutrino_momentum=predicted_neutrino,
            )
            result["partonic_cfm_training"] = cfm
            result["partonic_cfm_loss"] = cfm["loss"]
            result["partonic_cfm_support_fraction"] = cfm["support_fraction"]
            result["loss"] = result["loss"] + self.parent_cfm_loss_weight * cfm["loss"]
        return result


StructuredEventModel = JointEventModel
