"""Topology-conditioned generative inference for invisible momenta."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.models.invisible.coordinates import (
    build_invisible_coordinates,
)
from joint_event_reconstruction.models.invisible.couplings import build_flow_coupling
from joint_event_reconstruction.models.invisible.denoiser import InvisibleDenoiser
from joint_event_reconstruction.models.invisible.edm import EDMDiffusion
from joint_event_reconstruction.models.invisible.flow_matching import ContinuousFlowMatching
from joint_event_reconstruction.models.invisible.standard import StandardDiffusion
from joint_event_reconstruction.topology.spec import TopologySpec


class InvisibleGenerator(nn.Module):
    def __init__(
        self, spec: TopologySpec, visible_dim: int, config, inference_config,
        scaler=None,
    ):
        super().__init__()
        self.spec = spec
        self.assignment_mode = str(config.conditioning.assignment)
        self.scaler = scaler
        role_names = tuple(role.key for role in spec.invisible_roles)
        self.coordinates = build_invisible_coordinates(
            config.coordinates, role_names, getattr(config, "mass_shell", None)
        )
        active_mask = self.coordinates.active_mask
        denoiser_config = config.denoiser
        backbone = getattr(config, "backbone", None)
        if backbone is not None and str(backbone.type) == "lgatr":
            from joint_event_reconstruction.models.invisible.lgatr import (
                ConditionalLGATrInvisibleDenoiser,
            )

            denoiser = ConditionalLGATrInvisibleDenoiser(
                self.coordinates.token_count,
                state_dim=self.coordinates.state_dim,
                visible_dim=visible_dim,
                parent_dim=visible_dim,
                d_model=int(denoiser_config.d_model),
                backbone_config=backbone,
                zero_init_output=bool(denoiser_config.zero_init_output),
            )
        else:
            denoiser = InvisibleDenoiser(
                self.coordinates.token_count,
                state_dim=self.coordinates.state_dim,
                visible_dim=visible_dim,
                parent_dim=visible_dim,
                d_model=int(denoiser_config.d_model),
                n_heads=int(denoiser_config.n_heads),
                layers=int(denoiser_config.layers),
                d_feedforward=int(denoiser_config.d_feedforward),
                dropout=float(denoiser_config.dropout),
                zero_init_output=bool(denoiser_config.zero_init_output),
            )

        self.formulation_name = str(config.formulation)
        if self.formulation_name == "standard":
            self.formulation = StandardDiffusion(
                denoiser,
                int(config.standard.training_steps),
                active_mask=active_mask,
            )
            self.inference_budget = int(inference_config.standard.steps)
            self.sampler = "ddim"
        elif self.formulation_name == "edm":
            training = config.edm
            sampling = inference_config.edm
            self.formulation = EDMDiffusion(
                denoiser,
                sigma_data=float(training.sigma_data),
                p_mean=float(training.p_mean),
                p_std=float(training.p_std),
                sigma_min=float(sampling.sigma_min),
                sigma_max=float(sampling.sigma_max),
                rho=float(sampling.rho),
                active_mask=active_mask,
            )
            self.inference_budget = int(sampling.nfe)
            self.sampler = "heun"
        elif self.formulation_name == "flow_matching":
            coupling = build_flow_coupling(
                config.flow_matching.coupling, active_mask
            )
            self.formulation = ContinuousFlowMatching(
                denoiser, coupling=coupling, active_mask=active_mask
            )
            self.inference_budget = int(inference_config.flow_matching.steps)
            self.sampler = str(inference_config.flow_matching.solver)
        else:
            raise ValueError(
                f"Unknown invisible formulation {self.formulation_name!r}"
            )

    @property
    def denoiser(self) -> nn.Module:
        return self.formulation.denoiser

    def visible_context(
        self, visible: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = torch.cat(
            (visible["pooled"].unsqueeze(1), visible["tokens"]), dim=1
        )
        cls_mask = torch.zeros(
            (len(context), 1), dtype=torch.bool, device=context.device
        )
        return context, torch.cat((cls_mask, visible["padding_mask"]), dim=1)

    @staticmethod
    def visible_fourmomenta_context(
        visible: dict,
    ) -> torch.Tensor | None:
        """Align token-level physical vectors with the prepended pooled token."""
        vectors = visible.get("physics_vectors")
        if vectors is None:
            return None
        expected = visible["tokens"].shape[:2] + (4,)
        if vectors.shape != expected:
            raise ValueError(
                "Visible physical vectors must align with visible tokens; "
                f"expected={expected}, found={tuple(vectors.shape)}"
            )
        pooled = vectors.new_zeros((len(vectors), 1, 4))
        return torch.cat((pooled, vectors), dim=1)

    def role_parent_context(
        self,
        reconstruction: dict,
        structured: dict,
        pooled: torch.Tensor,
        node_context: dict[str, torch.Tensor] | None = None,
        *,
        mode: str | None = None,
    ) -> torch.Tensor:
        """Build parent contexts for non-hypothesis sampling and training.

        ``hypothesis`` and ``truth`` use the truth assignment only when it is
        reconstructable. Prediction for ``hypothesis`` uses explicit decoded
        assignments through :meth:`assignment_parent_context`; it is not an
        oracle at inference time.
        """
        mode = self.assignment_mode if mode is None else mode
        contexts = []
        for role in self.spec.invisible_roles:
            node = role.node
            base = pooled if node_context is None else node_context[node]
            if mode == "none":
                summary = torch.zeros_like(base)
            else:
                candidates = reconstruction["candidate_embeddings"][node]
                probabilities = reconstruction["assignment_probabilities"][node]
                if mode == "soft":
                    summary = torch.sum(
                        probabilities[:, 1:].unsqueeze(-1) * candidates, dim=1
                    )
                elif mode == "hard":
                    selected = probabilities.argmax(dim=1) - 1
                    safe = selected.clamp_min(0)
                    summary = candidates[
                        torch.arange(len(candidates), device=candidates.device), safe
                    ]
                    summary = summary * (selected >= 0).unsqueeze(1)
                elif mode in {"hypothesis", "truth"}:
                    pool = structured["pools"][node]
                    selected = pool["truth_index"]
                    safe = selected.clamp_min(0)
                    summary = candidates[
                        torch.arange(len(candidates), device=candidates.device), safe
                    ]
                    summary = summary * pool["reconstructable"].unsqueeze(1)
                else:
                    raise ValueError(
                        f"Unknown assignment conditioning mode {mode!r}"
                    )
            contexts.append(base + summary)
        return torch.stack(contexts, dim=1)

    def parent_context(
        self,
        reconstruction: dict,
        structured: dict,
        pooled: torch.Tensor,
        node_context: dict[str, torch.Tensor] | None = None,
        *,
        mode: str | None = None,
    ) -> torch.Tensor:
        """Build coordinate-token contexts from one context per invisible role."""
        role_context = self.role_parent_context(
            reconstruction,
            structured,
            pooled,
            node_context,
            mode=mode,
        )
        return self.coordinates.parent_context(role_context, pooled)

    @staticmethod
    def _expanded_met(met_xy: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        leading = state.shape[:-2]
        if met_xy.shape == leading + (2,):
            return met_xy
        if met_xy.ndim != 2 or met_xy.shape != (leading[0], 2):
            raise ValueError("met_xy cannot be aligned with invisible state")
        shape = (leading[0],) + (1,) * (len(leading) - 1) + (2,)
        return met_xy.reshape(shape).expand(leading + (2,))

    def encode_target(self, momenta, met_xy, scaler, masses=None):
        state = self.coordinates.encode(momenta, met_xy, masses=masses)
        return scaler.transform(state)

    def decode_state(self, scaled_state, met_xy, scaler):
        state = scaler.inverse(scaled_state)
        return self.coordinates.decode(state, self._expanded_met(met_xy, state))

    def decode_physical_state(self, scaled_state, met_xy, scaler):
        state = scaler.inverse(scaled_state)
        return self.coordinates.decode_physical(
            state, self._expanded_met(met_xy, state)
        )

    def coupling_condition(
        self,
        visible: dict,
        parent_context: torch.Tensor,
        *,
        role_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the conditional-OT feature from role-level parent contexts.

        Coordinate systems such as ``met_residual`` and ``met_exact`` expand
        each role context into additional state-token contexts.  The coupling
        condition is intentionally defined before that expansion as
        ``[z_pooled, mean_r z_parent,r]``; using the expanded token mean would
        make the condition depend on the coordinate parameterisation.
        """
        if parent_context.ndim != 3:
            raise ValueError(
                "Invisible parent context must be [B,T,D] for flow coupling"
            )
        if role_context is None:
            role_context = parent_context
        if role_context.ndim != 3 or role_context.shape[1] != len(self.spec.invisible_roles):
            raise ValueError(
                "Flow coupling requires one role parent context per invisible role"
            )
        if role_context.shape[0] != parent_context.shape[0] or role_context.shape[2] != parent_context.shape[2]:
            raise ValueError("Role and coordinate parent contexts must share batch and width")
        return torch.cat(
            (visible["pooled"], role_context.mean(dim=1)), dim=1
        )

    def loss(
        self,
        target: torch.Tensor,
        visible: dict,
        parent_context: torch.Tensor,
        *,
        role_context: torch.Tensor | None = None,
        event_mask: torch.Tensor | None = None,
    ) -> dict:
        context, mask = self.visible_context(visible)
        if event_mask is not None:
            if event_mask.dtype != torch.bool or event_mask.shape != (len(target),):
                raise ValueError(
                    "Invisible event_mask must be boolean [B], found "
                    f"dtype={event_mask.dtype}, shape={tuple(event_mask.shape)}"
                )
            if not event_mask.any():
                return {
                    "loss": target.sum() * 0.0,
                    "valid_events": 0,
                    "skipped_all_events": True,
                }
            target = target[event_mask]
            context = context[event_mask]
            mask = mask[event_mask]
            parent_context = parent_context[event_mask]
            if role_context is not None:
                role_context = role_context[event_mask]
        # ``target``, ``context`` and ``parent_context`` have already been
        # restricted to supported events above.  Build the flow-matching
        # condition from the same event subset; using the unfiltered pooled
        # tensor here breaks truth-oracle training as soon as one event is
        # unreconstructable.
        condition_visible = visible
        if event_mask is not None:
            condition_visible = {"pooled": visible["pooled"][event_mask]}
        condition = self.coupling_condition(
            condition_visible,
            parent_context,
            role_context=role_context,
        )
        visible_fourmomenta = self.visible_fourmomenta_context(visible)
        met_xy = visible.get("met_xy")
        if event_mask is not None:
            if visible_fourmomenta is not None:
                visible_fourmomenta = visible_fourmomenta[event_mask]
            if met_xy is not None:
                met_xy = met_xy[event_mask]
        if self.formulation_name == "flow_matching":
            result = self.formulation.loss(
                target,
                context,
                mask,
                parent_context,
                condition=condition,
                visible_fourmomenta=visible_fourmomenta,
                met_xy=met_xy,
            )
        else:
            result = self.formulation.loss(
                target,
                context,
                mask,
                parent_context,
                visible_fourmomenta=visible_fourmomenta,
                met_xy=met_xy,
            )
        result["valid_events"] = len(target)
        return result

    def flow_source(
        self,
        target: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.formulation_name != "flow_matching":
            raise RuntimeError("Flow source is only defined for Flow Matching")
        return self.formulation.source(
            target, generator=generator, condition=condition
        )

    def velocity(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        visible: dict,
        parent_context: torch.Tensor,
    ) -> torch.Tensor:
        if self.formulation_name != "flow_matching":
            raise RuntimeError("Invisible velocity is only defined for Flow Matching")
        context, mask = self.visible_context(visible)
        visible_fourmomenta = self.visible_fourmomenta_context(visible)
        met_xy = visible.get("met_xy")
        return self.formulation.velocity(
            state,
            time,
            context,
            mask,
            parent_context,
            visible_fourmomenta=visible_fourmomenta,
            met_xy=met_xy,
        )

    def sample(
        self,
        visible: dict,
        parent_context: torch.Tensor,
        *,
        samples: int,
        budget: int | None = None,
        sampler: str | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, int]:
        if samples <= 0:
            raise ValueError("Posterior sample count must be positive")
        context, mask = self.visible_context(visible)
        visible_fourmomenta = self.visible_fourmomenta_context(visible)
        met_xy = visible.get("met_xy")
        if visible_fourmomenta is not None:
            visible_fourmomenta = visible_fourmomenta.repeat_interleave(
                samples, dim=0
            )
        if met_xy is not None:
            met_xy = met_xy.repeat_interleave(samples, dim=0)
        batch, state_tokens, _ = parent_context.shape
        context = context.repeat_interleave(samples, dim=0)
        mask = mask.repeat_interleave(samples, dim=0)
        parent = parent_context.repeat_interleave(samples, dim=0)
        state_dim = self.coordinates.state_dim
        shape = (batch * samples, state_tokens, state_dim)
        selected_budget = self.inference_budget if budget is None else int(budget)
        if selected_budget <= 0:
            raise ValueError("Invisible inference budget must be positive")

        if self.formulation_name == "standard":
            values, nfe = self.formulation.sample(
                shape,
                context,
                mask,
                parent,
                inference_steps=selected_budget,
                generator=generator,
                visible_fourmomenta=visible_fourmomenta,
                met_xy=met_xy,
            )
        elif self.formulation_name == "edm":
            values, nfe = self.formulation.sample(
                shape,
                context,
                mask,
                parent,
                nfe=selected_budget,
                generator=generator,
                visible_fourmomenta=visible_fourmomenta,
                met_xy=met_xy,
            )
        else:
            selected_sampler = self.sampler if sampler is None else str(sampler)
            if selected_sampler not in {"euler", "heun"}:
                raise ValueError("Flow Matching sampler must be euler or heun")
            values, nfe = self.formulation.sample(
                shape,
                context,
                mask,
                parent,
                steps=selected_budget,
                solver=selected_sampler,
                generator=generator,
                visible_fourmomenta=visible_fourmomenta,
                met_xy=met_xy,
            )
        return values.reshape(batch, samples, state_tokens, state_dim), nfe

    def assignment_parent_context(
        self,
        reconstruction: dict,
        structured: dict,
        pooled: torch.Tensor,
        assignments: list[tuple],
        node_context: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Build exact hard contexts for complete predicted assignment hypotheses."""
        if not assignments or any(
            len(items) != len(assignments[0]) for items in assignments
        ):
            raise ValueError(
                "Every event must provide the same non-zero number of assignments"
            )
        per_event = []
        for event, hypotheses in enumerate(assignments):
            per_hypothesis = []
            for hypothesis in hypotheses:
                selected = dict(hypothesis.roots)
                for node in reversed(self.spec.dag_order):
                    if node not in selected:
                        continue
                    candidate_index = selected[node]
                    for role in self.spec.nodes[node].composite_roles:
                        child_index = int(
                            structured["pools"][node]["child_indices"][role.key][
                                candidate_index
                            ]
                        )
                        previous = selected.setdefault(role.child, child_index)
                        if previous != child_index:
                            raise RuntimeError(
                                "Decoded hypothesis has inconsistent child "
                                f"{role.child}"
                            )
                contexts = []
                for role in self.spec.invisible_roles:
                    if role.node not in selected:
                        raise RuntimeError(
                            "Complete hypothesis does not determine invisible "
                            f"parent {role.node}"
                        )
                    base = (
                        pooled[event]
                        if node_context is None
                        else node_context[role.node][event]
                    )
                    candidate = reconstruction["candidate_embeddings"][role.node][
                        event, selected[role.node]
                    ]
                    contexts.append(base + candidate)
                per_hypothesis.append(torch.stack(contexts, dim=0))
            per_event.append(torch.stack(per_hypothesis, dim=0))
        role_context = torch.stack(per_event, dim=0)
        pooled_hypotheses = pooled[:, None, :].expand(
            role_context.shape[0], role_context.shape[1], pooled.shape[-1]
        )
        return self.coordinates.parent_context(role_context, pooled_hypotheses)

    def sample_hypotheses(
        self,
        visible: dict,
        parent_context: torch.Tensor,
        *,
        samples: int,
        budget: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, int]:
        if parent_context.ndim != 4:
            raise ValueError("Hypothesis parent context must be [B,K,Rnu,D]")
        batch, hypotheses, state_tokens, width = parent_context.shape
        repeated_visible = {
            "pooled": visible["pooled"].repeat_interleave(hypotheses, dim=0),
            "tokens": visible["tokens"].repeat_interleave(hypotheses, dim=0),
            "padding_mask": visible["padding_mask"].repeat_interleave(
                hypotheses, dim=0
            ),
        }
        for key in ("physics_vectors", "met_xy"):
            if key in visible:
                repeated_visible[key] = visible[key].repeat_interleave(
                    hypotheses, dim=0
                )
        generated, nfe = self.sample(
            repeated_visible,
            parent_context.reshape(batch * hypotheses, state_tokens, width),
            samples=samples,
            budget=budget,
            generator=generator,
        )
        return generated.reshape(
            batch, hypotheses, samples, state_tokens, self.coordinates.state_dim
        ), nfe
