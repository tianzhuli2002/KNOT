"""Coherent event posterior research surface, using the maintained JER encoder/CFM.

M indices pair independent marginals for export only. C/J indices are ancestral.
Training conditioning is explicit; ordinary prediction accepts measured tensors
and feasible candidates only, and never reads supervision fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import torch
from omegaconf import OmegaConf
from torch import nn

from joint_event_reconstruction.models.invisible.couplings import IndependentCoupling
from joint_event_reconstruction.models.invisible.denoiser import InvisibleDenoiser
from joint_event_reconstruction.models.invisible.flow_matching import (
    ContinuousFlowMatching,
)
from joint_event_reconstruction.models.partonic import (
    PartonicCFM,
    four_vectors_from_partonic_features,
    partonic_features,
)
from joint_event_reconstruction.topology.assignments import AssignmentSpace
from joint_event_reconstruction.training.joint import JointEventModule


@dataclass
class EventPosterior:
    event_id: torch.Tensor
    sample_index: torch.Tensor
    assignment_state: torch.Tensor
    assignment_candidates: torch.Tensor
    invisible: torch.Tensor
    parents: torch.Tensor
    assignment_nodes: tuple[str, ...]
    invisible_roles: tuple[str, ...]
    parent_roles: tuple[str, ...]
    topology_identity: str
    assignment_support: torch.Tensor
    invisible_support: torch.Tensor
    parent_support: torch.Tensor
    factorisation: str
    assignment_log_probability: torch.Tensor | None = None

    def validate(self):
        b, m = self.sample_index.shape
        if self.event_id.shape != (b,) or self.event_id.unique().numel() != b:
            raise ValueError("Posterior EVENT_ID must be unique [event]")
        if not torch.equal(
            self.sample_index,
            torch.arange(m, device=self.sample_index.device).expand(b, m),
        ):
            raise ValueError(
                "Posterior sample indices must be explicit event-local indices"
            )
        for value, roles, components in (
            (self.invisible, self.invisible_roles, 4),
            (self.parents, self.parent_roles, 4),
        ):
            if (
                value.shape != (b, m, len(roles), components)
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"Invalid posterior continuous state for roles={roles}"
                )
        if self.assignment_state.shape != (
            b,
            m,
        ) or self.assignment_candidates.shape != (b, m, len(self.assignment_nodes)):
            raise ValueError("Assignment axes differ from continuous posterior axes")
        for mask, roles in (
            (self.assignment_support, self.assignment_nodes),
            (self.invisible_support, self.invisible_roles),
            (self.parent_support, self.parent_roles),
        ):
            if mask.dtype != torch.bool or mask.shape != (b, m, len(roles)):
                raise ValueError(f"Invalid posterior support for {roles}")
        if self.assignment_log_probability is not None and (
            self.assignment_log_probability.shape != (b, m)
            or not torch.isfinite(self.assignment_log_probability).all()
            or (self.assignment_log_probability > 1e-6).any()
        ):
            raise ValueError(
                "Sampled assignment log probabilities must be finite [event,sample] and <=0"
            )
        return self

    def write(self, path):
        self.validate()
        with h5py.File(path, "x") as f:
            for name in (
                "event_id",
                "sample_index",
                "assignment_state",
                "assignment_candidates",
                "invisible",
                "parents",
                "assignment_support",
                "invisible_support",
                "parent_support",
            ):
                f.create_dataset(
                    "EVENT_ID" if name == "event_id" else name,
                    data=getattr(self, name).detach().cpu().numpy(),
                )
            import json

            for name in ("assignment_nodes", "invisible_roles", "parent_roles"):
                f.attrs[name] = json.dumps(getattr(self, name))
            f.attrs["topology_identity"] = self.topology_identity
            f.attrs["factorisation"] = self.factorisation
            f.attrs["posterior_semantics"] = {
                "M": "marginal",
                "C": "ancestral",
                "J": "joint",
            }[self.factorisation]
            if self.assignment_log_probability is not None:
                f.create_dataset(
                    "assignment_log_probability",
                    data=self.assignment_log_probability.cpu().numpy(),
                )
            f.attrs["sample_semantics"] = (
                "independent_marginals"
                if self.factorisation == "M"
                else "coherent_event_hypothesis"
            )
            # Pair observables are derived from physical draws, never latent targets.
            pair = self.parents.sum(dim=2)
            f.create_dataset(
                "derived/m_parent_pair",
                data=(pair[..., 0].square() - pair[..., 1:].square().sum(-1))
                .clamp_min(0)
                .sqrt()
                .cpu(),
            )
            f.create_dataset(
                "derived/pT_parent_pair", data=pair[..., 1:3].norm(dim=-1).cpu()
            )


class CompositeVelocity(nn.Module):
    """Pack only active coordinates; all tokens interact in one denoiser."""

    def __init__(self, widths, config):
        super().__init__()
        self.widths = tuple(widths)
        self.network = InvisibleDenoiser(
            len(widths),
            state_dim=max(widths),
            visible_dim=config.d_model,
            parent_dim=config.d_model,
            d_model=config.d_model,
            n_heads=config.n_heads,
            layers=1,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
            zero_init_output=False,
        )
        active = torch.zeros(len(widths), max(widths), dtype=torch.bool)
        for i, w in enumerate(widths):
            active[i, :w] = True
        self.register_buffer("packing_mask", active)

    def forward(self, state, time, visible, mask, parent, **kwargs):
        padded = state.new_zeros((len(state), *self.packing_mask.shape))
        padded[:, self.packing_mask] = state
        velocity = self.network(padded, time, visible, mask, parent)
        return velocity[:, self.packing_mask]


class KnotModule(JointEventModule):
    def __init__(self, experiment, visible_config, *, research, **kwargs):
        super().__init__(experiment, visible_config, **kwargs)
        self.research = dict(research)
        self.variant = str(research["variant"])
        if self.variant not in {"M", "C", "J"}:
            raise ValueError("KNOT variant must be M/C/J")
        self.conditioning = str(research["training_conditioning"])
        if self.conditioning not in {"truth", "predicted"}:
            raise ValueError("Explicit truth/predicted conditioning required")
        base = self.event_model
        if (
            base.reasoning_enabled
            or base.feedback_enabled
            or base.model_mode != "factorized"
            or base.assignment_formulation != "discriminative"
        ):
            raise ValueError(
                "KNOT currently supports one-pass structured discriminative JER only"
            )
        if base.topology_queries is None:
            raise ValueError("KNOT requires the maintained shared topology query block")
        if base.invisible is None or base.invisible.formulation_name != "flow_matching":
            raise ValueError("KNOT requires maintained invisible CFM")
        if base.invisible.assignment_mode != "none":
            raise ValueError(
                "Baseline invisible conditioning must be none; KNOT supplies hard contexts explicitly"
            )
        if (
            base.parent_cfm is not None
            or base.point_head is not None
            or base.parent_head is not None
        ):
            raise ValueError("Disable unused parent/direct objectives in KNOT")
        scaler = kwargs["partonic_scaler"]
        # KNOT owns the parent flow even though it disables the standalone
        # parent-CFM head inherited from JointEventModule.  Persist the scaler
        # explicitly so checkpoints can be reconstructed and validated.
        if scaler is None:
            raise ValueError("KNOT requires a train-fitted partonic scaler")
        self.partonic_scaler_metadata = scaler.as_dict()
        self.parent_roles = tuple(scaler.role_names)
        # Compiled roots are sorted names, not a tensor-axis convention.
        # Parent query order is the maintained tensor order throughout KNOT.
        query_roles = tuple(base.topology_queries.parent_role_names)
        if self.parent_roles != query_roles or sorted(self.parent_roles) != sorted(
            base.spec.roots
        ):
            raise ValueError(
                "Configure precisely the compiled parent roots, avoiding "
                f"redundant pair targets: topology={base.spec.roots}, "
                f"scaler={self.parent_roles}"
            )
        self.invisible_roles = tuple(r.key for r in base.spec.invisible_roles)
        if self.variant == "C":
            if any(
                shell.mode != "fixed"
                for shell in base.invisible.coordinates.mass_shells.values()
            ):
                raise ValueError(
                    "KNOT-C maintained P2 conditioner currently requires fixed invisible masses"
                )
            ordered = []
            for parent in self.parent_roles:
                indices = [
                    i
                    for i, r in enumerate(base.spec.invisible_roles)
                    if r.node == parent
                ]
                if len(indices) != 1:
                    raise ValueError(
                        f"KNOT-C maintained P2 requires one invisible role per parent: parent={parent}, topology={base.spec.signature}"
                    )
                ordered.append(indices[0])
            if len(ordered) != len(self.invisible_roles):
                raise ValueError(
                    "KNOT-C has invisible roles outside configured parents"
                )
            self.register_buffer(
                "parent_invisible_indices", torch.tensor(ordered, dtype=torch.long)
            )
        if (
            base.shared_encoder_gradients != "both"
            or base.partonic_shared_encoder_gradients != "both"
        ):
            raise ValueError(
                "KNOT currently requires both shared encoder/query gradient policies"
            )
        coordinates = base.invisible.coordinates
        self.register_buffer("invisible_active", coordinates.active_mask)
        self.register_buffer(
            "parent_mean", torch.tensor(scaler.mean, dtype=torch.float32)
        )
        self.register_buffer(
            "parent_std", torch.tensor(scaler.std, dtype=torch.float32)
        )
        if not torch.isfinite(self.parent_std).all() or (self.parent_std <= 0).any():
            raise ValueError("Invalid parent scaler")
        self.n_invisible = int(self.invisible_active.sum())
        # No extra direct head is executed or optimised.
        base.partonic_head = None
        widths = (
            [int(row.sum()) for row in self.invisible_active]
            if self.variant == "J"
            else []
        )
        if any(w <= 0 for w in widths):
            raise ValueError("Empty invisible state tokens are unsupported")
        widths += [4] * len(self.parent_roles)
        if self.variant == "J":
            self.joint_continuous = ContinuousFlowMatching(
                CompositeVelocity(widths, experiment.model.latent),
                coupling=IndependentCoupling(),
                active_mask=torch.ones(sum(widths), dtype=torch.bool),
            )
            self.parent_generator = None
            for parameter in base.invisible.parameters():
                parameter.requires_grad_(False)
        else:
            latent = experiment.model.latent
            self.parent_generator = PartonicCFM(
                latent.d_model,
                latent.n_heads,
                1,
                latent.d_feedforward,
                latent.dropout,
                scaler=scaler,
                roles=self.parent_roles,
                conditioning="P0" if self.variant == "M" else "P2",
            )
            self.joint_continuous = None
        self.weights = {k: float(v) for k, v in research["loss_weights"].items()}
        if set(self.weights) != {"assignment", "invisible", "parent", "joint"}:
            raise ValueError("Explicit KNOT loss weights required")
        if (
            self.variant == "J"
            and (self.weights["invisible"] or self.weights["parent"])
        ) or (self.variant != "J" and self.weights["joint"]):
            raise ValueError("Loss weights activate an incompatible KNOT objective")
        if any(v < 0 for v in self.weights.values()):
            raise ValueError("Loss weights must be nonnegative")
        self.hparams["knot"] = OmegaConf.to_container(
            OmegaConf.create(research), resolve=True
        )

    @property
    def continuous(self):
        return (
            self.joint_continuous if self.variant == "J" else self.parent_generator.flow
        )

    def validate_checkpoint_metadata(self, checkpoint):
        super().validate_checkpoint_metadata(checkpoint)
        if checkpoint["hyper_parameters"].get("knot") != self.hparams["knot"]:
            raise ValueError("KNOT checkpoint formulation/conditioning mismatch")

    def assignment(self, batch, visible, loss=False):
        result = self.event_model.reconstruction(
            visible["tokens"],
            visible["pooled"],
            batch["structured"],
            node_context=visible.get("node_context"),
            calculate_loss=loss,
        )
        return result, AssignmentSpace.from_structured(batch["structured"])

    def hard_context(self, result, space, states, visible, names, role_bases=None):
        indices = space.node_indices(states)
        b, _m = states.shape
        contexts = []
        for role_index, name in enumerate(names):
            selected = indices[..., space.nodes.index(name)]
            candidates = result["candidate_embeddings"][name]
            null = self.event_model.reconstruction.null_embeddings[name].expand(
                b, 1, -1
            )
            choices = torch.cat((null, candidates), dim=1)
            chosen = choices.gather(
                1, (selected + 1)[..., None].expand(-1, -1, choices.shape[-1])
            )
            base = (
                role_bases[:, role_index]
                if role_bases is not None
                else (
                    visible["node_context"][name]
                    if "node_context" in visible
                    else visible["pooled"]
                )
            )
            contexts.append(chosen + base[:, None])
        return torch.stack(contexts, dim=2)

    def invisible_role_context(self, result, space, states, visible):
        invisible = self.event_model.invisible
        if self.variant == "M":
            role = invisible.role_parent_context(
                result,
                {},
                visible["pooled"],
                visible.get("node_context"),
                mode="none",
                query_role_context=visible.get("invisible_role_context"),
                query_role_names=tuple(visible.get("invisible_role_names", ())),
            )
            return role[:, None].expand(-1, states.shape[1], -1, -1)
        if tuple(visible.get("invisible_role_names", ())) != self.invisible_roles:
            raise ValueError("KNOT invisible query role order mismatch")
        role = self.hard_context(
            result,
            space,
            states,
            visible,
            tuple(r.node for r in self.event_model.spec.invisible_roles),
            role_bases=visible["invisible_role_context"],
        )
        return role

    def invisible_context(self, result, space, states, visible):
        role = self.invisible_role_context(result, space, states, visible)
        return self.event_model.invisible.coordinates.parent_context(
            role, visible["pooled"][:, None].expand(-1, states.shape[1], -1)
        )

    @staticmethod
    def repeat_visible(visible, m):
        return {
            key: (value.repeat_interleave(m, 0) if torch.is_tensor(value) else value)
            for key, value in visible.items()
        }

    def parent_context(self, result, space, states, visible, neutrino=None):
        b, m = states.shape
        if tuple(visible["parent_role_names"]) != self.parent_roles:
            raise ValueError("Parent query order mismatch")
        queries = visible["parent_role_context"][:, None].expand(-1, m, -1, -1)
        if self.variant == "J":
            return self.hard_context(result, space, states, visible, self.parent_roles)
        assignment = None
        if self.variant == "C":
            indices = space.node_indices(states)
            selected = []
            for role in self.parent_roles:
                candidates = result["candidate_embeddings"][role]
                choices = torch.cat(
                    (
                        self.event_model.reconstruction.null_embeddings[role].expand(
                            b, 1, -1
                        ),
                        candidates,
                    ),
                    dim=1,
                )
                index = indices[..., space.nodes.index(role)] + 1
                selected.append(
                    choices.gather(
                        1, index[..., None].expand(-1, -1, choices.shape[-1])
                    )
                )
            assignment = torch.stack(selected, dim=2).flatten(0, 1)
        tokens = self.parent_generator._conditioning_tokens(
            visible["tokens"].repeat_interleave(m, 0),
            queries.flatten(0, 1),
            assignment,
            neutrino.index_select(-2, self.parent_invisible_indices).flatten(0, 1)
            if self.variant == "C"
            else None,
        )
        return tokens.reshape(b, m, *tokens.shape[1:])

    def unpack_invisible(self, packed):
        shape = (*packed.shape[:-1], *self.invisible_active.shape)
        state = packed.new_zeros(shape)
        state[..., self.invisible_active] = packed
        return state

    def forward(self, batch, compute_objectives=True, **kwargs):
        base = self.event_model
        visible = base.encode_visible(batch)
        if not compute_objectives:
            return {"visible": visible}
        assignment_weight = self.weights.get("assignment", 0)
        if not any(self.weights.values()):
            return {"loss": visible["pooled"].sum() * 0}
        if self.variant == "M" and not assignment_weight:
            result, space = {}, None
            states = torch.zeros(
                (len(visible["pooled"]), 1),
                dtype=torch.long,
                device=visible["pooled"].device,
            )
        elif self.conditioning == "truth" and not assignment_weight:
            result = {
                "candidate_embeddings": base.reconstruction.encode_candidates(
                    visible["tokens"], batch["structured"]
                )
            }
            space = AssignmentSpace.from_structured(batch["structured"])
            states = space.truth_state[:, None]
        else:
            result, space = self.assignment(batch, visible, loss=assignment_weight > 0)
            states = (
                space.truth_state[:, None]
                if self.conditioning == "truth"
                else torch.multinomial(
                    space.state_probabilities(result["assignment_logits"]), 1
                )
            )
        output = {}
        total = visible["pooled"].sum() * 0
        if assignment_weight:
            output["assignment_loss"] = result["assignment_loss"]
            total = total + assignment_weight * output["assignment_loss"]
        nweight = self.weights.get("invisible", 0)
        pweight = self.weights.get("parent", 0)
        jweight = self.weights.get("joint", 0)
        if self.variant != "J" and nweight:
            context = self.invisible_context(result, space, states, visible)[:, 0]
            invisible_support = batch["invisible_support"].all(-1)
            if self.conditioning == "truth" and self.variant != "M":
                # A truth assignment outside the model's feasible state set
                # cannot provide a valid conditional context for C or J.
                invisible_support = (
                    invisible_support & batch["visible_assignment_support"]
                )
            output["invisible_loss"] = base.invisible.loss(
                batch["invisible_truth_scaled"],
                visible,
                context,
                event_mask=invisible_support,
                role_context=self.invisible_role_context(
                    result, space, states, visible
                )[:, 0],
            )["loss"]
            total = total + nweight * output["invisible_loss"]
        if (self.variant == "J" and jweight) or (self.variant != "J" and pweight):
            target = batch["partonic_truth"]
            support = batch["partonic_support"].all(-1)
            if self.conditioning == "truth" and self.variant != "M":
                support = support & batch["visible_assignment_support"]
            p = (
                (partonic_features(target) - self.parent_mean) / self.parent_std
            ).flatten(1)
            neutrino = None
            if self.variant == "C":
                if self.conditioning == "truth":
                    neutrino = base.invisible.decode_state(
                        batch["invisible_truth_scaled"][:, None],
                        batch["met_xy"],
                        base.invisible.scaler,
                    )
                else:
                    context = self.invisible_context(result, space, states, visible)[
                        :, 0
                    ]
                    n, _ = base.invisible.sample(
                        visible, context, samples=1, budget=int(self.research["steps"])
                    )
                    neutrino = base.invisible.decode_state(
                        n, batch["met_xy"], base.invisible.scaler
                    )
                support = support & batch["invisible_support"].all(-1)
            context = self.parent_context(result, space, states, visible, neutrino)[
                :, 0
            ]
            if self.variant == "J":
                n = batch["invisible_truth_scaled"][:, self.invisible_active]
                p = torch.cat((n, p), dim=-1)
                ncontext = self.invisible_context(result, space, states, visible)[:, 0]
                context = torch.cat((ncontext, context), dim=1)
                support = support & batch["invisible_support"].all(-1)
            memory, mask = base.invisible.visible_context(visible)
            key = "joint_loss" if self.variant == "J" else "partonic_cfm_loss"
            output[key] = self.continuous.loss(
                p, memory, mask, context, support_mask=support
            )["loss"]
            total = total + (jweight if self.variant == "J" else pweight) * output[key]
        output["loss"] = total
        return output

    def common_validation_selection(self, output):
        """Dimension-weighted validation score shared by the frozen M/C/J pilot."""
        assignment = output["assignment_loss"]
        if self.variant == "J":
            return assignment + output["joint_loss"]
        return (
            assignment
            + (6.0 / 14.0) * output["invisible_loss"]
            + (8.0 / 14.0) * output["partonic_cfm_loss"]
        )

    @torch.no_grad()
    def posterior(self, batch, *, samples, generator=None, oracle=None):
        if samples <= 0:
            raise ValueError("Positive posterior count required")
        # One encoder call, before expansion; no encoder calls inside either ODE.
        visible = self.event_model.encode_visible(batch)
        result, space = self.assignment(batch, visible)
        states = torch.multinomial(
            space.state_probabilities(result["assignment_logits"]),
            samples,
            replacement=True,
            generator=generator,
        )
        if oracle is not None:
            if self.variant == "C" and oracle in {
                "truth_assignment_truth_neutrino",
                "predicted_assignment_truth_neutrino",
            }:
                if oracle == "truth_assignment_truth_neutrino":
                    states = space.truth_state[:, None].expand(-1, samples)
            elif self.variant == "J" and oracle == "truth_assignment":
                states = space.truth_state[:, None].expand(-1, samples)
            else:
                raise ValueError(
                    "C parent oracles are limited to KNOT-C; the J assignment oracle is "
                    "oracle='truth_assignment'"
                )
        b, m = states.shape
        repeated = self.repeat_visible(visible, m)
        invisible = self.event_model.invisible
        ncontext = self.invisible_context(result, space, states, visible).flatten(0, 1)
        memory, mask = invisible.visible_context(repeated)
        if self.variant != "J":
            if oracle is None:
                n, _ = invisible.sample(
                    repeated,
                    ncontext,
                    samples=1,
                    budget=int(self.research["steps"]),
                    sampler=self.research["solver"],
                    generator=generator,
                )
                n = n[:, 0].reshape(b, m, *n.shape[2:])
            else:
                # Consume the same Gaussian source draw as deployed C, keeping
                # parent noise matched without integrating an unused N flow.
                torch.randn(
                    (b * m, ncontext.shape[1], invisible.coordinates.state_dim),
                    dtype=memory.dtype,
                    device=memory.device,
                    generator=generator,
                )
                n = batch["invisible_truth_scaled"][:, None].expand(
                    -1, samples, *batch["invisible_truth_scaled"].shape[1:]
                )
            neutrino = invisible.decode_state(n, batch["met_xy"], invisible.scaler)
        else:
            neutrino = None
        context = self.parent_context(result, space, states, visible, neutrino).flatten(
            0, 1
        )
        if self.variant == "J":
            context = torch.cat((ncontext, context), dim=1)
        dimensions = int(self.continuous.active_mask.sum())
        draw, _ = self.continuous.sample(
            (b * m, dimensions),
            memory,
            mask,
            context,
            steps=int(self.research["steps"]),
            solver=self.research["solver"],
            generator=generator,
        )
        draw = draw.reshape(b, m, -1)
        if self.variant == "J":
            n = self.unpack_invisible(draw[..., : self.n_invisible])
            draw = draw[..., self.n_invisible :]
        physical = invisible.decode_physical_state(n, batch["met_xy"], invisible.scaler)
        parents = four_vectors_from_partonic_features(
            draw.reshape(b, m, len(self.parent_roles), 4) * self.parent_std
            + self.parent_mean
        )
        candidates = space.node_indices(states)
        # Masks mean latent roles exist; NULL visible support is recorded separately.
        return EventPosterior(
            batch["event_index"],
            torch.arange(m, device=states.device).expand(b, m),
            states,
            candidates,
            physical["four_vector"],
            parents,
            space.nodes,
            self.invisible_roles,
            self.parent_roles,
            self.spec.signature,
            candidates >= 0,
            torch.ones(
                (b, m, len(self.invisible_roles)),
                dtype=torch.bool,
                device=states.device,
            ),
            torch.ones(
                (b, m, len(self.parent_roles)), dtype=torch.bool, device=states.device
            ),
            self.variant,
            space.state_probabilities(result["assignment_logits"])
            .gather(1, states)
            .log(),
        ).validate()
