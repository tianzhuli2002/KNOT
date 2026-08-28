"""Joint discrete-continuous Flow Matching for complete event reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from joint_event_reconstruction.flows.continuous import linear_path
from joint_event_reconstruction.flows.discrete import MaskedDiscreteFlow
from joint_event_reconstruction.models.joint_reasoner import TopologyFlowReasoner
from joint_event_reconstruction.models.reconstruction.discrete_flow import AssignmentStateEncoder
from joint_event_reconstruction.topology.assignments import AssignmentSpace
from joint_event_reconstruction.topology.spec import TopologySpec


@dataclass
class JointFlowState:
    assignment: torch.Tensor
    invisible: torch.Tensor
    topology: torch.Tensor | None = None


@dataclass
class JointStepOutput:
    complete_assignment_logits: torch.Tensor
    complete_assignment_probabilities: torch.Tensor
    invisible_velocity: torch.Tensor
    topology: torch.Tensor
    node_context: dict[str, torch.Tensor]
    local_assignment_logits: dict[str, torch.Tensor]
    local_assignment_probabilities: dict[str, torch.Tensor]


class JointFlow(nn.Module):
    """Shared topology reasoning with discrete assignment and continuous invisible heads."""

    def __init__(self, spec: TopologySpec, reconstruction: nn.Module, invisible: nn.Module, *, d_model: int, n_heads: int, block_layers: int, d_feedforward: int, dropout: float, recurrent_state: bool, train_steps: tuple[int, ...], assignment_loss_weight: float = 1.0, invisible_loss_weight: float = 1.0):
        super().__init__()
        if invisible.formulation_name != "flow_matching":
            raise ValueError("JointFlow requires a continuous Flow Matching invisible model")
        if not train_steps or any(step <= 0 for step in train_steps):
            raise ValueError("JointFlow train_steps must contain positive integers")
        self.spec = spec
        self.reconstruction = reconstruction
        self.invisible = invisible
        self.state_encoder = AssignmentStateEncoder(spec, d_model)
        self.reasoner = TopologyFlowReasoner(spec, d_model, n_heads, block_layers, d_feedforward, dropout)
        self.discrete_flow = MaskedDiscreteFlow()
        self.recurrent_state = recurrent_state
        self.train_steps = tuple(train_steps)
        self.assignment_loss_weight = assignment_loss_weight
        self.invisible_loss_weight = invisible_loss_weight

    def _parent_context(self, node_context: dict[str, torch.Tensor], assignment_expectation: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([node_context[role.node] + assignment_expectation[role.node] for role in self.spec.invisible_roles], dim=1)

    def step(self, visible: dict, structured: dict, space: AssignmentSpace, candidate_embeddings: dict[str, torch.Tensor], state: JointFlowState, time: torch.Tensor) -> JointStepOutput:
        assignment_context = self.state_encoder.encode(state.assignment, space, candidate_embeddings)
        topology = self.reasoner.step(visible, assignment_context, state.invisible, time, state.topology if self.recurrent_state else None)
        node_context = self.reasoner.as_dict(topology)
        local_logits, local_probabilities = self.reconstruction.score_candidates(candidate_embeddings, visible["pooled"], structured, node_context=node_context)
        complete_logits = space.state_logits(local_logits)
        complete_probabilities = complete_logits.softmax(dim=1)
        expected_assignment = self.state_encoder.expectation(complete_probabilities, space, candidate_embeddings)
        parent_context = self._parent_context(node_context, expected_assignment)
        velocity = self.invisible.velocity(state.invisible, time, visible, parent_context)
        return JointStepOutput(complete_assignment_logits=complete_logits, complete_assignment_probabilities=complete_probabilities, invisible_velocity=velocity, topology=topology, node_context=node_context, local_assignment_logits=local_logits, local_assignment_probabilities=local_probabilities)

    def _loss_from_step(self, output: JointStepOutput, *, space: AssignmentSpace, state: JointFlowState, time: torch.Tensor, target_velocity: torch.Tensor) -> dict:
        assignment_loss = self.discrete_flow.loss(output.complete_assignment_logits, space.truth_state, state.assignment, time, mask_state=space.mask_state)
        invisible_loss = F.mse_loss(output.invisible_velocity, target_velocity)
        total = self.assignment_loss_weight * assignment_loss + self.invisible_loss_weight * invisible_loss
        return {"loss": total, "reconstruction_loss": assignment_loss, "invisible_loss": invisible_loss}

    def _stateless_loss(self, visible: dict, structured: dict, space: AssignmentSpace, candidate_embeddings: dict[str, torch.Tensor], target: torch.Tensor) -> dict:
        time = self.discrete_flow.sample_time(len(target), device=target.device, dtype=target.dtype)
        source = torch.randn_like(target)
        invisible_state, target_velocity = linear_path(source, target, time)
        assignment_state = self.discrete_flow.training_state(space.truth_state, time, space.mask_state)
        state = JointFlowState(assignment_state, invisible_state, None)
        output = self.step(visible, structured, space, candidate_embeddings, state, time)
        losses = self._loss_from_step(output, space=space, state=state, time=time, target_velocity=target_velocity)
        return {**losses, "candidate_embeddings": candidate_embeddings, "assignment_logits": output.local_assignment_logits, "assignment_probabilities": output.local_assignment_probabilities, "complete_assignment_logits": output.complete_assignment_logits, "complete_assignment_probabilities": output.complete_assignment_probabilities, "node_context": output.node_context, "topology_state": output.topology, "flow_time": time, "flow_state": state, "truth_state": space.truth_state, "recurrent_steps": 0}

    def _recurrent_loss(self, visible: dict, structured: dict, space: AssignmentSpace, candidate_embeddings: dict[str, torch.Tensor], target: torch.Tensor) -> dict:
        step_index = int(torch.randint(len(self.train_steps), (), device=target.device))
        steps = self.train_steps[step_index]
        final_time = self.discrete_flow.sample_time(len(target), device=target.device, dtype=target.dtype)
        source = torch.randn_like(target)
        reveal = torch.rand(len(target), device=target.device, dtype=target.dtype)
        topology = None
        last_state = None
        output = None
        target_velocity = target - source
        for index in range(1, steps + 1):
            time = final_time * (index / steps)
            invisible_state, _ = linear_path(source, target, time)
            assignment_state = torch.where(reveal < time, space.truth_state, torch.full_like(space.truth_state, space.mask_state))
            last_state = JointFlowState(assignment_state, invisible_state, topology)
            output = self.step(visible, structured, space, candidate_embeddings, last_state, time)
            topology = output.topology
        losses = self._loss_from_step(output, space=space, state=last_state, time=final_time, target_velocity=target_velocity)
        return {**losses, "candidate_embeddings": candidate_embeddings, "assignment_logits": output.local_assignment_logits, "assignment_probabilities": output.local_assignment_probabilities, "complete_assignment_logits": output.complete_assignment_logits, "complete_assignment_probabilities": output.complete_assignment_probabilities, "node_context": output.node_context, "topology_state": output.topology, "flow_time": final_time, "flow_state": last_state, "truth_state": space.truth_state, "recurrent_steps": steps}

    def loss(self, visible: dict, structured: dict, target: torch.Tensor) -> dict:
        if target.ndim != 3 or target.shape[1:] != (len(self.spec.invisible_roles), 3):
            raise ValueError(f"Joint invisible target must be [B,{len(self.spec.invisible_roles)},3]")
        space = AssignmentSpace.from_structured(structured)
        candidates = self.reconstruction.encode_candidates(visible["tokens"], structured)
        if self.recurrent_state:
            return self._recurrent_loss(visible, structured, space, candidates, target)
        return self._stateless_loss(visible, structured, space, candidates, target)

    @torch.no_grad()
    def sample(self, visible: dict, structured: dict, *, samples: int, steps: int, generator: torch.Generator | None = None) -> dict:
        if samples <= 0 or steps <= 0:
            raise ValueError("Joint posterior samples and flow steps must be positive")
        space = AssignmentSpace.from_structured(structured)
        candidates = self.reconstruction.encode_candidates(visible["tokens"], structured)
        assignments = []
        invisibles = []
        final_outputs = []
        batch = len(visible["tokens"])
        schedule = torch.linspace(0, 1, steps + 1, device=visible["tokens"].device, dtype=visible["tokens"].dtype)
        for _ in range(samples):
            state = JointFlowState(assignment=torch.full((batch,), space.mask_state, dtype=torch.long, device=visible["tokens"].device), invisible=torch.randn((batch, len(self.spec.invisible_roles), 3), device=visible["tokens"].device, dtype=visible["tokens"].dtype, generator=generator), topology=None)
            output = None
            for current, following in zip(schedule[:-1], schedule[1:]):
                time = current.expand(batch)
                output = self.step(visible, structured, space, candidates, state, time)
                dt = following - current
                state.invisible = state.invisible + dt * output.invisible_velocity
                state.assignment = self.discrete_flow.step(state.assignment, output.complete_assignment_logits, time, following.expand(batch), mask_state=space.mask_state, generator=generator)
                state.topology = output.topology if self.recurrent_state else None
            if (state.assignment == space.mask_state).any():
                raise RuntimeError("Joint Flow ended with unresolved assignment MASK")
            assignments.append(state.assignment)
            invisibles.append(state.invisible)
            final_outputs.append(output)
        return {"assignment_state": torch.stack(assignments, dim=1), "invisible": torch.stack(invisibles, dim=1), "candidate_embeddings": candidates, "nfe": steps, "recurrent_state": self.recurrent_state, "last_step": final_outputs[-1]}
