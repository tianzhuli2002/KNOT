"""Joint discrete-continuous Flow Matching for complete event reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from joint_event_reconstruction.flows.continuous import linear_path
from joint_event_reconstruction.flows.discrete import MaskedDiscreteFlow
from joint_event_reconstruction.models.joint_reasoner import TopologyFlowReasoner
from joint_event_reconstruction.models.reconstruction.discrete_flow import (
    AssignmentStateEncoder,
)
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


def _repeat_visible(visible: dict, repeats: int) -> dict:
    return {
        "pooled": visible["pooled"].repeat_interleave(repeats, dim=0),
        "tokens": visible["tokens"].repeat_interleave(repeats, dim=0),
        "padding_mask": visible["padding_mask"].repeat_interleave(repeats, dim=0),
    }


def _repeat_structured(structured: dict, repeats: int) -> dict:
    """Repeat event tensors while sharing topology-static candidate templates."""
    pools = {}
    for name, pool in structured["pools"].items():
        pools[name] = dict(pool)
        for field in ("candidate_mask", "truth_index", "reconstructable"):
            if field in pool:
                pools[name][field] = pool[field].repeat_interleave(repeats, dim=0)
    result = {"pools": pools}
    if "assignments" in structured:
        assignments = dict(structured["assignments"])
        assignments["valid_mask"] = structured["assignments"][
            "valid_mask"
        ].repeat_interleave(repeats, dim=0)
        assignments["truth_state"] = structured["assignments"][
            "truth_state"
        ].repeat_interleave(repeats, dim=0)
        result["assignments"] = assignments
    return result


class JointFlow(nn.Module):
    """Shared dynamics coupling complete assignments and invisible momenta."""

    def __init__(
        self,
        spec: TopologySpec,
        *,
        d_model: int,
        n_heads: int,
        block_layers: int,
        d_feedforward: int,
        dropout: float,
        recurrent_state: bool,
        train_steps: tuple[int, ...],
        assignment_loss_weight: float = 1.0,
        invisible_loss_weight: float = 1.0,
    ):
        super().__init__()
        if not train_steps or any(step <= 0 for step in train_steps):
            raise ValueError("JointFlow train_steps must contain positive integers")
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
        self.discrete_flow = MaskedDiscreteFlow()
        self.recurrent_state = recurrent_state
        self.train_steps = tuple(train_steps)
        self.assignment_loss_weight = assignment_loss_weight
        self.invisible_loss_weight = invisible_loss_weight

    def _parent_context(
        self,
        node_context: dict[str, torch.Tensor],
        assignment_expectation: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.stack(
            [
                node_context[role.node] + assignment_expectation[role.node]
                for role in self.spec.invisible_roles
            ],
            dim=1,
        )

    def step(
        self,
        visible: dict,
        structured: dict,
        reconstruction: nn.Module,
        invisible: nn.Module,
        space: AssignmentSpace,
        candidate_embeddings: dict[str, torch.Tensor],
        state: JointFlowState,
        time: torch.Tensor,
        met_xy: torch.Tensor,
    ) -> JointStepOutput:
        if invisible.formulation_name != "flow_matching":
            raise ValueError("JointFlow requires continuous Flow Matching")
        assignment_context = self.state_encoder.encode(
            state.assignment, space, candidate_embeddings
        )
        if invisible.scaler is None:
            raise RuntimeError("Joint Flow requires invisible scaler metadata")
        physical_invisible = invisible.decode_state(
            state.invisible, met_xy, invisible.scaler
        )
        topology = self.reasoner.step(
            visible,
            assignment_context,
            physical_invisible,
            time,
            state.topology if self.recurrent_state else None,
        )
        node_context = self.reasoner.as_dict(topology)
        local_logits, local_probabilities = reconstruction.score_candidates(
            candidate_embeddings,
            visible["pooled"],
            structured,
            node_context=node_context,
        )
        complete_logits = space.state_logits(local_logits)
        complete_probabilities = complete_logits.softmax(dim=1)
        expected_assignment = self.state_encoder.expectation(
            complete_probabilities, space, candidate_embeddings
        )
        role_parent_context = self._parent_context(
            node_context, expected_assignment
        )
        parent_context = invisible.coordinates.parent_context(
            role_parent_context, visible["pooled"]
        )
        velocity = invisible.velocity(
            state.invisible, time, visible, parent_context
        )
        return JointStepOutput(
            complete_assignment_logits=complete_logits,
            complete_assignment_probabilities=complete_probabilities,
            invisible_velocity=velocity,
            topology=topology,
            node_context=node_context,
            local_assignment_logits=local_logits,
            local_assignment_probabilities=local_probabilities,
        )

    def _loss_from_step(
        self,
        output: JointStepOutput,
        *,
        space: AssignmentSpace,
        state: JointFlowState,
        time: torch.Tensor,
        target_velocity: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> dict:
        assignment_loss = self.discrete_flow.loss(
            output.complete_assignment_logits,
            space.truth_state,
            state.assignment,
            time,
            mask_state=space.mask_state,
        )
        invisible_loss = (
            (output.invisible_velocity - target_velocity).square().sum()
            / (len(target_velocity) * active_mask.sum())
        )
        return {
            "loss": (
                self.assignment_loss_weight * assignment_loss
                + self.invisible_loss_weight * invisible_loss
            ),
            "assignment_loss": assignment_loss,
            "invisible_loss": invisible_loss,
        }

    def _stateless_loss(
        self,
        visible: dict,
        structured: dict,
        reconstruction: nn.Module,
        invisible: nn.Module,
        space: AssignmentSpace,
        candidate_embeddings: dict[str, torch.Tensor],
        target: torch.Tensor,
        met_xy: torch.Tensor,
    ) -> dict:
        time = self.discrete_flow.sample_time(
            len(target), device=target.device, dtype=target.dtype
        )
        role_base = visible["pooled"].unsqueeze(1).expand(
            -1, len(self.spec.invisible_roles), -1
        )
        coordinate_base = invisible.coordinates.parent_context(
            role_base, visible["pooled"]
        )
        condition = invisible.coupling_condition(
            visible,
            coordinate_base,
            role_context=role_base,
        )
        source = invisible.flow_source(target, condition=condition)
        invisible_state, target_velocity = linear_path(source, target, time)
        assignment_state = self.discrete_flow.training_state(
            space.truth_state, time, space.mask_state
        )
        state = JointFlowState(assignment_state, invisible_state, None)
        output = self.step(
            visible,
            structured,
            reconstruction,
            invisible,
            space,
            candidate_embeddings,
            state,
            time,
            met_xy,
        )
        losses = self._loss_from_step(
            output,
            space=space,
            state=state,
            time=time,
            target_velocity=target_velocity,
            active_mask=invisible.formulation.active_mask,
        )
        return self._training_output(
            losses, candidate_embeddings, output, space, state, time, steps=0
        )

    def _recurrent_loss(
        self,
        visible: dict,
        structured: dict,
        reconstruction: nn.Module,
        invisible: nn.Module,
        space: AssignmentSpace,
        candidate_embeddings: dict[str, torch.Tensor],
        target: torch.Tensor,
        met_xy: torch.Tensor,
    ) -> dict:
        steps = self.train_steps[
            int(torch.randint(len(self.train_steps), (), device=target.device))
        ]
        final_time = self.discrete_flow.sample_time(
            len(target), device=target.device, dtype=target.dtype
        )
        role_base = visible["pooled"].unsqueeze(1).expand(
            -1, len(self.spec.invisible_roles), -1
        )
        coordinate_base = invisible.coordinates.parent_context(
            role_base, visible["pooled"]
        )
        condition = invisible.coupling_condition(
            visible,
            coordinate_base,
            role_context=role_base,
        )
        source = invisible.flow_source(target, condition=condition)
        reveal = torch.rand(len(target), device=target.device, dtype=target.dtype)
        target_velocity = target - source
        topology = None
        state = None
        output = None
        for index in range(1, steps + 1):
            time = final_time * (index / steps)
            invisible_state, _ = linear_path(source, target, time)
            assignment_state = torch.where(
                reveal < time,
                space.truth_state,
                torch.full_like(space.truth_state, space.mask_state),
            )
            state = JointFlowState(assignment_state, invisible_state, topology)
            output = self.step(
                visible,
                structured,
                reconstruction,
                invisible,
                space,
                candidate_embeddings,
                state,
                time,
                met_xy,
            )
            topology = output.topology
        losses = self._loss_from_step(
            output,
            space=space,
            state=state,
            time=final_time,
            target_velocity=target_velocity,
            active_mask=invisible.formulation.active_mask,
        )
        return self._training_output(
            losses,
            candidate_embeddings,
            output,
            space,
            state,
            final_time,
            steps=steps,
        )

    @staticmethod
    def _training_output(
        losses: dict,
        candidate_embeddings: dict[str, torch.Tensor],
        output: JointStepOutput,
        space: AssignmentSpace,
        state: JointFlowState,
        time: torch.Tensor,
        *,
        steps: int,
    ) -> dict:
        return {
            **losses,
            "candidate_embeddings": candidate_embeddings,
            "assignment_logits": output.local_assignment_logits,
            "assignment_probabilities": output.local_assignment_probabilities,
            "complete_assignment_logits": output.complete_assignment_logits,
            "complete_assignment_probabilities": output.complete_assignment_probabilities,
            "node_context": output.node_context,
            "topology_state": output.topology,
            "flow_time": time,
            "flow_state": state,
            "truth_state": space.truth_state,
            "recurrent_steps": steps,
        }

    def loss(
        self,
        visible: dict,
        structured: dict,
        target: torch.Tensor,
        met_xy: torch.Tensor,
        reconstruction: nn.Module,
        invisible: nn.Module,
    ) -> dict:
        expected = (invisible.coordinates.token_count, 3)
        if target.ndim != 3 or target.shape[1:] != expected:
            raise ValueError(
                f"Joint invisible target must be [B,{expected[0]},3], "
                f"found {tuple(target.shape)}"
            )
        space = AssignmentSpace.from_structured(structured)
        candidates = reconstruction.encode_candidates(visible["tokens"], structured)
        if self.recurrent_state:
            return self._recurrent_loss(
                visible,
                structured,
                reconstruction,
                invisible,
                space,
                candidates,
                target,
                met_xy,
            )
        return self._stateless_loss(
            visible,
            structured,
            reconstruction,
            invisible,
            space,
            candidates,
            target,
            met_xy,
        )

    @torch.no_grad()
    def sample(
        self,
        visible: dict,
        structured: dict,
        reconstruction: nn.Module,
        invisible: nn.Module,
        *,
        samples: int,
        steps: int,
        sample_chunk_size: int = 1,
        generator: torch.Generator | None = None,
        met_xy: torch.Tensor,
    ) -> dict:
        if samples <= 0 or steps <= 0 or sample_chunk_size <= 0:
            raise ValueError(
                "Joint posterior samples, flow steps and sample_chunk_size must be positive"
            )
        if invisible.formulation_name != "flow_matching":
            raise ValueError("JointFlow requires continuous Flow Matching")

        base_space = AssignmentSpace.from_structured(structured)
        base_candidates = reconstruction.encode_candidates(
            visible["tokens"], structured
        )
        batch = len(visible["tokens"])
        schedule = torch.linspace(
            0,
            1,
            steps + 1,
            device=visible["tokens"].device,
            dtype=visible["tokens"].dtype,
        )
        assignment_chunks = []
        invisible_chunks = []
        last_output = None
        chunk_size = min(samples, sample_chunk_size)

        for start in range(0, samples, chunk_size):
            width = min(chunk_size, samples - start)
            repeated_visible = _repeat_visible(visible, width)
            repeated_structured = _repeat_structured(structured, width)
            repeated_met = met_xy.repeat_interleave(width, dim=0)
            repeated_space = AssignmentSpace.from_structured(repeated_structured)
            repeated_candidates = {
                name: values.repeat_interleave(width, dim=0)
                for name, values in base_candidates.items()
            }
            expanded_batch = batch * width
            state = JointFlowState(
                assignment=torch.full(
                    (expanded_batch,),
                    base_space.mask_state,
                    dtype=torch.long,
                    device=visible["tokens"].device,
                ),
                invisible=torch.randn(
                    (expanded_batch, invisible.coordinates.token_count, 3),
                    device=visible["tokens"].device,
                    dtype=visible["tokens"].dtype,
                    generator=generator,
                ) * invisible.formulation.active_mask.to(
                    device=visible["tokens"].device,
                    dtype=visible["tokens"].dtype,
                ),
                topology=None,
            )
            output = None
            for current, following in zip(schedule[:-1], schedule[1:]):
                time = current.expand(expanded_batch)
                output = self.step(
                    repeated_visible,
                    repeated_structured,
                    reconstruction,
                    invisible,
                    repeated_space,
                    repeated_candidates,
                    state,
                    time,
                    repeated_met,
                )
                dt = following - current
                state.invisible = (
                    state.invisible + dt * output.invisible_velocity
                ) * invisible.formulation.active_mask.to(
                    device=state.invisible.device, dtype=state.invisible.dtype
                )
                state.assignment = self.discrete_flow.step(
                    state.assignment,
                    output.complete_assignment_logits,
                    time,
                    following.expand(expanded_batch),
                    mask_state=repeated_space.mask_state,
                    generator=generator,
                )
                state.topology = output.topology if self.recurrent_state else None
            if (state.assignment == repeated_space.mask_state).any():
                raise RuntimeError("Joint Flow ended with unresolved assignment MASK")
            assignment_chunks.append(state.assignment.reshape(batch, width))
            invisible_chunks.append(
                state.invisible.reshape(
                    batch, width, invisible.coordinates.token_count, 3
                )
            )
            last_output = output

        return {
            "assignment_state": torch.cat(assignment_chunks, dim=1),
            "invisible": torch.cat(invisible_chunks, dim=1),
            "candidate_embeddings": base_candidates,
            "nfe": steps,
            "posterior_samples": samples,
            "sample_chunk_size": chunk_size,
            "recurrent_state": self.recurrent_state,
            "last_step": last_output,
        }
