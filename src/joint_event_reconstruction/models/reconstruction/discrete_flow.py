"""Discrete Flow Matching for topology-constrained complete assignments."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.flows.discrete import MaskedDiscreteFlow
from joint_event_reconstruction.models.joint_reasoner import TopologyFlowReasoner
from joint_event_reconstruction.topology.assignments import AssignmentSpace, MASK_NODE
from joint_event_reconstruction.topology.spec import TopologySpec


class AssignmentStateEncoder(nn.Module):
    """Represent categorical states through the physical candidates they select."""

    def __init__(self, spec: TopologySpec, d_model: int):
        super().__init__()
        self.spec = spec
        self.null_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(d_model))
                for name in spec.recurrent_tokens
            }
        )
        self.mask_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(d_model))
                for name in spec.recurrent_tokens
            }
        )
        for parameter in tuple(self.null_embeddings.values()) + tuple(
            self.mask_embeddings.values()
        ):
            nn.init.normal_(parameter, std=0.02)

    def encode(
        self,
        state: torch.Tensor,
        space: AssignmentSpace,
        candidate_embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        indices = space.node_indices(state)
        if indices.ndim != 2:
            raise ValueError(
                "Assignment state encoder expects one state per event"
            )
        rows = torch.arange(len(state), device=state.device)
        result = {}
        for node_index, name in enumerate(space.nodes):
            selected = indices[:, node_index]
            candidates = candidate_embeddings[name]
            if candidates.shape[1] == 0:
                gathered = candidates.new_zeros(
                    (len(state), candidates.shape[-1])
                )
            else:
                gathered = candidates[rows, selected.clamp_min(0)]
            null = self.null_embeddings[name].unsqueeze(0).expand_as(gathered)
            mask = self.mask_embeddings[name].unsqueeze(0).expand_as(gathered)
            gathered = torch.where(
                (selected == -1).unsqueeze(1), null, gathered
            )
            gathered = torch.where(
                (selected == MASK_NODE).unsqueeze(1), mask, gathered
            )
            result[name] = gathered
        return result

    def expectation(
        self,
        probabilities: torch.Tensor,
        space: AssignmentSpace,
        candidate_embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if probabilities.shape != space.valid_mask.shape:
            raise ValueError(
                "Complete-state probabilities do not match AssignmentSpace"
            )
        batch = len(probabilities)
        result = {}
        for node_index, name in enumerate(space.nodes):
            selected = space.node_candidate_indices[:, node_index]
            candidates = candidate_embeddings[name]
            candidate_weights = probabilities.new_zeros(
                (batch, candidates.shape[1])
            )
            non_null = selected >= 0
            if non_null.any():
                index = selected[non_null].unsqueeze(0).expand(batch, -1)
                candidate_weights.scatter_add_(
                    1, index, probabilities[:, non_null]
                )
            summary = torch.sum(
                candidate_weights.unsqueeze(-1) * candidates, dim=1
            )
            null_weight = probabilities[:, selected < 0].sum(
                dim=1, keepdim=True
            )
            result[name] = (
                summary + null_weight * self.null_embeddings[name]
            )
        return result


class DiscreteAssignmentFlow(nn.Module):
    """Flow-specific assignment state and reasoner; reconstruction stays external."""

    def __init__(
        self,
        spec: TopologySpec,
        *,
        d_model: int,
        n_heads: int,
        block_layers: int,
        d_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.spec = spec
        self.state_encoder = AssignmentStateEncoder(spec, d_model)
        self.reasoner = TopologyFlowReasoner(
            spec,
            d_model,
            n_heads,
            block_layers,
            d_feedforward,
            dropout,
        )
        self.flow = MaskedDiscreteFlow()

    def _predict(
        self,
        visible: dict,
        structured: dict,
        reconstruction: nn.Module,
        space: AssignmentSpace,
        candidate_embeddings: dict[str, torch.Tensor],
        state: torch.Tensor,
        time: torch.Tensor,
    ) -> dict:
        assignment_context = self.state_encoder.encode(
            state, space, candidate_embeddings
        )
        topology = self.reasoner.step(
            visible, assignment_context, None, time, state=None
        )
        node_context = self.reasoner.as_dict(topology)
        local_logits, local_probabilities = reconstruction.score_candidates(
            candidate_embeddings,
            visible["pooled"],
            structured,
            node_context=node_context,
        )
        complete_logits = space.state_logits(local_logits)
        return {
            "assignment_logits": complete_logits,
            "assignment_probabilities": complete_logits.softmax(dim=1),
            "local_assignment_logits": local_logits,
            "local_assignment_probabilities": local_probabilities,
            "node_context": node_context,
            "topology_state": topology,
        }

    def loss(
        self, visible: dict, structured: dict, reconstruction: nn.Module
    ) -> dict:
        space = AssignmentSpace.from_structured(structured)
        candidates = reconstruction.encode_candidates(
            visible["tokens"], structured
        )
        time = self.flow.sample_time(
            len(visible["tokens"]),
            device=visible["tokens"].device,
            dtype=visible["tokens"].dtype,
        )
        state = self.flow.training_state(
            space.truth_state, time, space.mask_state
        )
        prediction = self._predict(
            visible,
            structured,
            reconstruction,
            space,
            candidates,
            state,
            time,
        )
        loss = self.flow.loss(
            prediction["assignment_logits"],
            space.truth_state,
            state,
            time,
            mask_state=space.mask_state,
        )
        return {
            **prediction,
            "candidate_embeddings": candidates,
            "reconstruction_loss": loss,
            "loss": loss,
            "flow_time": time,
            "flow_state": state,
            "truth_state": space.truth_state,
        }

    def _sample_with_candidates(
        self,
        visible: dict,
        structured: dict,
        reconstruction: nn.Module,
        space: AssignmentSpace,
        candidates: dict[str, torch.Tensor],
        *,
        steps: int,
        generator: torch.Generator | None,
    ) -> dict:
        state = torch.full(
            (len(visible["tokens"]),),
            space.mask_state,
            dtype=torch.long,
            device=visible["tokens"].device,
        )
        schedule = torch.linspace(
            0,
            1,
            steps + 1,
            device=visible["tokens"].device,
            dtype=visible["tokens"].dtype,
        )
        prediction = None
        for current, following in zip(schedule[:-1], schedule[1:]):
            time = current.expand(len(state))
            prediction = self._predict(
                visible,
                structured,
                reconstruction,
                space,
                candidates,
                state,
                time,
            )
            state = self.flow.step(
                state,
                prediction["assignment_logits"],
                time,
                following.expand(len(state)),
                mask_state=space.mask_state,
                generator=generator,
            )
        if (state == space.mask_state).any():
            raise RuntimeError(
                "Discrete Flow Matching ended with unresolved MASK states"
            )
        return {
            **prediction,
            "sampled_assignment_state": state,
            "nfe": steps,
        }

    @torch.no_grad()
    def sample(
        self,
        visible: dict,
        structured: dict,
        reconstruction: nn.Module,
        *,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> dict:
        if steps <= 0:
            raise ValueError("Discrete flow inference steps must be positive")
        space = AssignmentSpace.from_structured(structured)
        candidates = reconstruction.encode_candidates(
            visible["tokens"], structured
        )
        sampled = self._sample_with_candidates(
            visible,
            structured,
            reconstruction,
            space,
            candidates,
            steps=steps,
            generator=generator,
        )
        return {**sampled, "candidate_embeddings": candidates}

    @torch.no_grad()
    def sample_many(
        self,
        visible: dict,
        structured: dict,
        reconstruction: nn.Module,
        *,
        samples: int,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> dict:
        """Sample several DFM trajectories while encoding candidates only once."""
        if samples <= 0 or steps <= 0:
            raise ValueError(
                "Discrete posterior samples and flow steps must be positive"
            )
        space = AssignmentSpace.from_structured(structured)
        candidates = reconstruction.encode_candidates(
            visible["tokens"], structured
        )
        states = []
        last = None
        for _ in range(samples):
            last = self._sample_with_candidates(
                visible,
                structured,
                reconstruction,
                space,
                candidates,
                steps=steps,
                generator=generator,
            )
            states.append(last["sampled_assignment_state"])
        return {
            **last,
            "candidate_embeddings": candidates,
            "sampled_assignment_state": torch.stack(states, dim=1),
            "posterior_samples": samples,
            "nfe": steps,
        }
