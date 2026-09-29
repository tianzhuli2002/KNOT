"""Topology-generic direct truth-level four-vector inference head."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.models.invisible.couplings import IndependentCoupling
from joint_event_reconstruction.models.invisible.denoiser import noise_embedding
from joint_event_reconstruction.models.invisible.flow_matching import (
    ContinuousFlowMatching,
)


class PartonicFourVectorHead(nn.Module):
    """Predict scaled ``px, py, pz, log1p(invariant_mass)`` features.

    The training scaler supplies the physical units.  Its inverse uses
    ``expm1`` for a positive event-dependent mass before constructing energy;
    no nominal mass or reconstructed constituent is used.
    """

    def __init__(self, d_model: int, roles: tuple[str, ...]) -> None:
        super().__init__()
        if not roles or len(set(roles)) != len(roles):
            raise ValueError("Partonic head requires unique configured target roles")
        self.roles = tuple(str(role) for role in roles)
        self.output = nn.Sequential(
            nn.LayerNorm(int(d_model)),
            nn.Linear(int(d_model), int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), 4),
        )

    def forward(self, query_states: torch.Tensor, query_names: tuple[str, ...]) -> torch.Tensor:
        if query_states.ndim != 3:
            raise ValueError(f"Partonic query states must be [B,Q,D], found {tuple(query_states.shape)}")
        locations = []
        for role in self.roles:
            if role not in query_names:
                raise ValueError(f"Partonic target role {role!r} is absent from topology queries")
            locations.append(query_names.index(role))
        selected = query_states[:, locations]
        return self.output(selected)


def partonic_features(
    four_vectors: torch.Tensor, *, validate_physics: bool = True
) -> torch.Tensor:
    """Convert physical ``(E,px,py,pz)`` to ``(px,py,pz,m)`` features."""
    if four_vectors.ndim < 3 or four_vectors.shape[-1] != 4:
        raise ValueError("Partonic truth must end in [P,4] in E_px_py_pz order")
    energy = four_vectors[..., 0]
    spatial = four_vectors[..., 1:]
    mass_squared = energy.square() - spatial.square().sum(dim=-1)
    if validate_physics and (
        not torch.isfinite(four_vectors).all()
        or (energy < 0).any()
        or (mass_squared < -1.0e-3).any()
    ):
        raise ValueError("Partonic truth contains an invalid physical four-vector")
    mass = mass_squared.clamp_min(0.0).sqrt()
    return torch.cat((spatial, torch.log1p(mass).unsqueeze(-1)), dim=-1)


def four_vectors_from_partonic_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 3 or features.shape[-1] != 4:
        raise ValueError("Partonic features must end in [P,4]")
    spatial = features[..., :3]
    mass = torch.expm1(features[..., 3]).clamp_min(0.0)
    energy = torch.sqrt(spatial.square().sum(dim=-1) + mass.square())
    return torch.cat((energy.unsqueeze(-1), spatial), dim=-1)


class PartonicCFMDenoiser(nn.Module):
    """Velocity model for the single joint eight-dimensional parent state."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layers: int,
        d_feedforward: int,
        dropout: float,
        *,
        velocity_backbone: str = "conventional",
        geometry_encoder_config=None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads:
            raise ValueError("Partonic CFM width must be positive and divisible by n_heads")
        if layers <= 0 or d_feedforward <= 0:
            raise ValueError("Partonic CFM layers and feedforward width must be positive")
        if velocity_backbone not in {"conventional", "geometric"}:
            raise ValueError(
                "Partonic CFM velocity_backbone must be conventional or geometric"
            )
        self.d_model = int(d_model)
        self.velocity_backbone = str(velocity_backbone)
        self.geometry_network = None
        if self.velocity_backbone == "conventional":
            self.input_projection = nn.Linear(8, d_model)
            self.time_projection = nn.Sequential(
                nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
            )
            self.state_embedding = nn.Parameter(torch.empty(1, 1, d_model))
            nn.init.normal_(self.state_embedding, mean=0.0, std=0.02)
            self.blocks = nn.ModuleList(
                _ParentCFMBlock(d_model, n_heads, d_feedforward, dropout)
                for _ in range(layers)
            )
            self.output_norm = nn.LayerNorm(d_model)
            self.output_projection = nn.Linear(d_model, 8)
        else:
            if geometry_encoder_config is None:
                raise ValueError(
                    "Geometric parent CFM requires the full L-GATr visible encoder config"
                )
            try:
                import lgatr
            except ImportError as exc:
                raise ImportError(
                    "Geometric parent CFM requires the optional L-GATr dependency"
                ) from exc
            symmetry = geometry_encoder_config.symmetry_breaking
            beam = str(symmetry.beam)
            add_time = bool(symmetry.time)
            self.geometry_momentum_scale_gev = float(
                geometry_encoder_config.momentum_scale_gev
            )
            if self.geometry_momentum_scale_gev <= 0:
                raise ValueError("Geometric parent momentum scale must be positive")
            n_spurions = int(lgatr.get_num_spurions(beam, add_time))
            self.register_buffer(
                "geometry_spurions",
                lgatr.get_spurions(beam, add_time),
                persistent=False,
            )
            hidden = geometry_encoder_config.hidden
            geometry_heads = int(geometry_encoder_config.n_heads)
            self.geometry_network = lgatr.ConditionalLGATr(
                num_blocks=int(layers),
                in_mv_channels=1 + n_spurions,
                mv_channels_cond=1,
                out_mv_channels=1,
                hidden_mv_channels=int(hidden.multivector_channels),
                in_s_channels=2 * d_model,
                s_channels_cond=d_model,
                out_s_channels=1,
                hidden_s_channels=int(hidden.scalar_channels),
                attention={"num_heads": geometry_heads},
                crossattention={"num_heads": geometry_heads},
                mlp={},
                dropout_prob=float(dropout),
                checkpoint_blocks=False,
                naive_amp=False,
                compile=False,
            )

    def prepare_conditioning(
        self,
        visible: torch.Tensor,
        visible_padding_mask: torch.Tensor,
        parent_context: torch.Tensor,
        *,
        geometric_context: torch.Tensor | None = None,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ):
        if self.velocity_backbone == "geometric":
            if geometric_context is None or target_mean is None or target_std is None:
                raise ValueError(
                    "Geometric parent CFM conditioning requires H_geom and its production scaler"
                )
            if (
                geometric_context.ndim != 4
                or geometric_context.shape[:2] != visible.shape[:2]
                or geometric_context.shape[-2:] != (1, 16)
            ):
                raise ValueError(
                    "Parent H_geom must align with scalar event tokens as [B,N,1,16]"
                )
            if visible_padding_mask.shape != visible.shape[:2]:
                raise ValueError("Geometric parent mask does not align with H_geom")
            if parent_context.shape[:1] != visible.shape[:1] or parent_context.shape[1] < 2:
                raise ValueError("Geometric parent conditioning needs ordered top/antitop context")
            return (
                visible,
                visible_padding_mask,
                parent_context,
                geometric_context,
                target_mean,
                target_std,
            )
        memory = torch.cat((visible, parent_context), dim=1)
        parent_padding = torch.zeros(
            (len(visible), parent_context.shape[1]),
            dtype=torch.bool,
            device=visible.device,
        )
        return memory, torch.cat((visible_padding_mask, parent_padding), dim=1)

    def forward(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        visible: torch.Tensor,
        visible_padding_mask: torch.Tensor,
        parent_context: torch.Tensor,
        *,
        prepared_context=None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if self.velocity_backbone == "geometric":
            return self._geometric_velocity(
                state,
                time,
                visible,
                visible_padding_mask,
                parent_context,
                prepared_context,
            )
        if state.ndim != 2 or state.shape[1] != 8:
            raise ValueError(f"Joint parent state must be [B,8], found {tuple(state.shape)}")
        if time.shape != (len(state),):
            raise ValueError(f"Parent CFM time must be [B], found {tuple(time.shape)}")
        if visible.ndim != 3 or visible.shape[0] != len(state) or visible.shape[-1] != self.d_model:
            raise ValueError("Partonic CFM measured context must be [B,N,D]")
        if visible_padding_mask.shape != visible.shape[:2] or visible_padding_mask.dtype != torch.bool:
            raise ValueError("Partonic CFM visible padding mask must be boolean [B,N]")
        if parent_context.ndim != 3 or parent_context.shape[0] != len(state) or parent_context.shape[-1] != self.d_model:
            raise ValueError("Partonic CFM conditioning tokens must be [B,C,D]")
        if parent_context.shape[1] < 2:
            raise ValueError("Partonic CFM requires ordered top and antitop query contexts")
        if prepared_context is None:
            prepared_context = self.prepare_conditioning(
                visible, visible_padding_mask, parent_context
            )
        memory, memory_padding = prepared_context
        hidden = (
            self.input_projection(state).unsqueeze(1)
            + self.state_embedding
            + self.time_projection(noise_embedding(time, self.d_model)).unsqueeze(1)
        )
        for block in self.blocks:
            hidden = block(hidden, memory, memory_padding)
        return self.output_projection(self.output_norm(hidden[:, 0]))

    def _geometric_velocity(
        self,
        state: torch.Tensor,
        time: torch.Tensor,
        visible: torch.Tensor,
        visible_padding_mask: torch.Tensor,
        parent_context: torch.Tensor,
        prepared_context,
    ) -> torch.Tensor:
        if state.ndim != 2 or state.shape[-1] != 8 or time.shape != (len(state),):
            raise ValueError("Geometric parent state/time must be [B*M,8] and [B*M]")
        if prepared_context is None:
            raise ValueError("Geometric parent velocity requires prepared static event context")
        (
            visible,
            visible_padding_mask,
            parent_context,
            geometric_context,
            target_mean,
            target_std,
        ) = prepared_context
        if visible.ndim != 3 or visible.shape[-1] != self.d_model:
            raise ValueError("Geometric parent scalar context must be [B,N,D]")
        if visible_padding_mask.shape != visible.shape[:2] or visible_padding_mask.dtype != torch.bool:
            raise ValueError("Geometric parent padding mask must be boolean [B,N]")
        if geometric_context.shape[:2] != visible.shape[:2] or geometric_context.shape[-2:] != (1, 16):
            raise ValueError("Geometric parent H_geom must be [B,N,1,16]")
        batch = len(visible)
        if batch == 0 or len(state) % batch:
            raise ValueError("Geometric parent state batch is not an integer event-sample multiple")
        sample_count = len(state) // batch
        if parent_context.shape[0] != batch or parent_context.shape[1] < 2:
            raise ValueError("Geometric parent context must start with top/antitop role tokens")

        # Leading B,M axes let L-GATr broadcast each cached event across its
        # posterior samples without copying the static geometric representation.
        scaled = state.reshape(batch, sample_count, 2, 4)
        physical_features = (
            scaled * target_std.to(dtype=scaled.dtype)
            + target_mean.to(dtype=scaled.dtype)
        )
        physical_four_vectors = four_vectors_from_partonic_features(physical_features)
        momenta = physical_four_vectors / self.geometry_momentum_scale_gev
        geometric_state = _package_embed_vector(momenta, self.geometry_spurions)

        time_features = noise_embedding(
            time.reshape(batch, sample_count).reshape(-1), self.d_model
        ).reshape(batch, sample_count, self.d_model)
        role_context = parent_context[:, :2].unsqueeze(1).expand(
            -1, sample_count, -1, -1
        )
        time_context = time_features.unsqueeze(2).expand(-1, -1, 2, -1)
        state_scalars = torch.cat((role_context, time_context), dim=-1)

        extra_context = parent_context[:, 2:]
        condition_geometry = geometric_context
        condition_scalars = visible
        condition_padding = visible_padding_mask
        if extra_context.shape[1]:
            condition_geometry = torch.cat(
                (
                    condition_geometry,
                    geometric_context.new_zeros(
                        (batch, extra_context.shape[1], 1, 16)
                    ),
                ),
                dim=1,
            )
            condition_scalars = torch.cat((condition_scalars, extra_context), dim=1)
            condition_padding = torch.cat(
                (
                    condition_padding,
                    torch.zeros(
                        batch,
                        extra_context.shape[1],
                        dtype=torch.bool,
                        device=condition_padding.device,
                    ),
                ),
                dim=1,
            )
        condition_geometry = condition_geometry.unsqueeze(1).expand(
            -1, sample_count, -1, -1, -1
        )
        condition_scalars = condition_scalars.unsqueeze(1).expand(
            -1, sample_count, -1, -1
        )
        attention_mask = (~condition_padding)[:, None, None, None, :]
        output_mv, output_s = self.geometry_network(
            geometric_state,
            condition_geometry,
            scalars=state_scalars,
            scalars_cond=condition_scalars,
            crossattn_kwargs={"attn_mask": attention_mask},
        )
        if output_s is None or output_s.shape[-1] != 1:
            raise RuntimeError("Geometric parent velocity requires one scalar mass output per role")
        try:
            import lgatr
        except ImportError as exc:
            raise ImportError("Geometric parent CFM requires the optional L-GATr dependency") from exc
        vector_velocity = lgatr.extract_vector(output_mv.squeeze(-2))[..., 1:]
        scale = target_std.to(dtype=vector_velocity.dtype)[..., :3].view(1, 1, 2, 3)
        momentum_velocity = vector_velocity * self.geometry_momentum_scale_gev / scale
        mass_velocity = output_s.squeeze(-1)
        return torch.cat((momentum_velocity, mass_velocity.unsqueeze(-1)), dim=-1).reshape(
            len(state), 8
        )


def _package_embed_vector(
    vectors: torch.Tensor, spurions: torch.Tensor
) -> torch.Tensor:
    try:
        import lgatr
    except ImportError as exc:
        raise ImportError("Geometric parent CFM requires the optional L-GATr dependency") from exc
    embedded = lgatr.embed_vector(vectors).unsqueeze(-2)
    if spurions.shape[0] == 0:
        return embedded
    spurion_shape = (1,) * (vectors.ndim - 1) + tuple(spurions.shape)
    expanded = spurions.to(device=vectors.device, dtype=embedded.dtype).reshape(
        spurion_shape
    ).expand(*vectors.shape[:-1], *spurions.shape)
    return torch.cat((embedded, expanded), dim=-2)


class _ParentCFMBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_feedforward: int, dropout: float):
        super().__init__()
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_feedforward, d_model),
        )

    def forward(
        self,
        state: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.cross_norm(state)
        state = state + self.cross_attention(
            query,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )[0]
        return state + self.feedforward(self.feedforward_norm(state))


class PartonicCFM(nn.Module):
    """One conditional flow over both after-FSR parents in role order.

    P0 consumes measured tokens and the shared parent-query states. P1 adds
    predicted soft complete-assignment context. P2 adds an explicitly
    declared sample from the predicted invisible posterior in (top,
    antitop) order. No truth conditioning enters this module.
    """

    ROLES = ("top", "antitop")
    MODE_CONDITION_COUNTS = {"P0": 2, "P1": 4, "P2": 6}

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        layers: int,
        d_feedforward: int,
        dropout: float,
        *,
        scaler,
        conditioning: str = "P0",
        use_topology_queries: bool = True,
        velocity_backbone: str = "conventional",
        geometry_encoder_config=None,
    ) -> None:
        super().__init__()
        if conditioning not in self.MODE_CONDITION_COUNTS:
            raise ValueError(f"Partonic CFM conditioning must be P0, P1 or P2, found {conditioning!r}")
        if scaler is None or tuple(scaler.role_names) != self.ROLES:
            raise ValueError("Partonic CFM requires a train-fitted scaler ordered as (top, antitop)")
        self.scaler = scaler
        self.conditioning = str(conditioning)
        self.use_topology_queries = bool(use_topology_queries)
        self.velocity_backbone = str(velocity_backbone)
        self.d_model = int(d_model)
        self.register_buffer(
            "target_mean", torch.tensor(scaler.mean, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "target_std", torch.tensor(scaler.std, dtype=torch.float32), persistent=True
        )
        self.neutrino_projection = nn.Sequential(
            nn.LayerNorm(3), nn.Linear(3, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.condition_kind_embedding = nn.Parameter(torch.empty(3, d_model))
        self.parent_role_embedding = nn.Parameter(torch.empty(2, d_model))
        nn.init.normal_(self.condition_kind_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.parent_role_embedding, mean=0.0, std=0.02)
        denoiser = PartonicCFMDenoiser(
            d_model,
            n_heads,
            layers,
            d_feedforward,
            dropout,
            velocity_backbone=self.velocity_backbone,
            geometry_encoder_config=geometry_encoder_config,
        )
        self.flow = ContinuousFlowMatching(
            denoiser,
            coupling=IndependentCoupling(),
            active_mask=torch.ones(8, dtype=torch.bool),
        )

    def _conditioning_tokens(
        self,
        visible: torch.Tensor,
        parent_queries: torch.Tensor | None,
        assignment_context: torch.Tensor | None,
        predicted_neutrino_momentum: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.use_topology_queries:
            if (
                parent_queries is None
                or parent_queries.ndim != 3
                or parent_queries.shape[1:] != (2, self.d_model)
            ):
                observed = None if parent_queries is None else tuple(parent_queries.shape)
                raise ValueError(
                    "Parent query contexts must be [B,2,D] in (top, antitop) order; "
                    f"found {observed}"
                )
            if parent_queries.shape[0] != len(visible):
                raise ValueError("Parent query context batch differs from measured event batch")
        else:
            if parent_queries is not None:
                raise ValueError("No-query parent CFM cannot receive topology-query states")
            if visible.ndim != 3 or visible.shape[-1] != self.d_model:
                raise ValueError("No-query parent CFM measured context must be [B,N,D]")
            # Learned parent_role_embedding supplies semantic role identity.
            # The measured event remains available through the visible memory.
            parent_queries = visible.new_zeros((len(visible), 2, self.d_model))
        expected_assignment = self.conditioning in {"P1", "P2"}
        expected_neutrino = self.conditioning == "P2"
        if expected_assignment != (assignment_context is not None):
            raise ValueError(f"{self.conditioning} has an incompatible assignment-context input")
        if expected_neutrino != (predicted_neutrino_momentum is not None):
            raise ValueError(
                f"{self.conditioning} has an incompatible predicted-neutrino input"
            )
        role_embedding = self.parent_role_embedding.unsqueeze(0)
        tokens = [parent_queries + role_embedding + self.condition_kind_embedding[0]]
        if assignment_context is not None:
            if assignment_context.shape != parent_queries.shape:
                raise ValueError("Predicted assignment context must be [B,2,D] in parent role order")
            tokens.append(assignment_context + role_embedding + self.condition_kind_embedding[1])
        if predicted_neutrino_momentum is not None:
            if predicted_neutrino_momentum.shape != (len(parent_queries), 2, 3):
                raise ValueError(
                    "P2 requires one role-aligned predicted neutrino momentum per parent "
                    "with shape [B,2,3] in GeV"
                )
            neutrino = self.neutrino_projection(predicted_neutrino_momentum)
            tokens.append(neutrino + role_embedding + self.condition_kind_embedding[2])
        result = torch.cat(tokens, dim=1)
        if result.shape[1] != self.MODE_CONDITION_COUNTS[self.conditioning]:
            raise RuntimeError("Partonic CFM conditioning token count is inconsistent")
        return result

    def loss(
        self,
        target_four_vectors: torch.Tensor,
        support: torch.Tensor,
        visible: torch.Tensor,
        visible_padding_mask: torch.Tensor,
        parent_queries: torch.Tensor | None,
        *,
        assignment_context: torch.Tensor | None = None,
        predicted_neutrino_momentum: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        source_state: torch.Tensor | None = None,
        flow_time: torch.Tensor | None = None,
        geometric_context: torch.Tensor | None = None,
    ) -> dict:
        if target_four_vectors.shape != (len(visible), 2, 4):
            raise ValueError("Partonic CFM targets must be [B,2,4] in (top, antitop), (E,px,py,pz) order")
        if support.shape == (len(visible), 2):
            support = support.all(dim=1)
        if support.dtype != torch.bool or support.shape != (len(visible),):
            raise ValueError("Partonic CFM support must be boolean [B] or [B,2]")
        # The explicit target sidecar is fully physics-validated when opened;
        # avoid a CUDA scalar-to-Python synchronisation in every training step.
        target_features = partonic_features(
            target_four_vectors, validate_physics=False
        )
        scaled = ((target_features - self.target_mean) / self.target_std).reshape(
            len(visible), 8
        )
        parent_context = self._conditioning_tokens(
            visible,
            parent_queries,
            assignment_context,
            predicted_neutrino_momentum,
        )
        prepared_context = None
        if self.velocity_backbone == "geometric":
            prepared_context = self.flow.denoiser.prepare_conditioning(
                visible,
                visible_padding_mask,
                parent_context,
                geometric_context=geometric_context,
                target_mean=self.target_mean,
                target_std=self.target_std,
            )
        result = self.flow.loss(
            scaled,
            visible,
            visible_padding_mask,
            parent_context,
            generator=generator,
            source_state=source_state,
            flow_time=flow_time,
            support_mask=support,
            prepared_context=prepared_context,
        )
        result["target_scaled"] = scaled
        result["support_mask"] = support
        result["support_fraction"] = support.to(dtype=scaled.dtype).mean()
        return result

    @torch.no_grad()
    def sample(
        self,
        visible: torch.Tensor,
        visible_padding_mask: torch.Tensor,
        parent_queries: torch.Tensor | None,
        *,
        samples: int,
        steps: int,
        solver: str,
        generator: torch.Generator | None = None,
        assignment_context: torch.Tensor | None = None,
        predicted_neutrino_momentum: torch.Tensor | None = None,
        source_state: torch.Tensor | None = None,
        geometric_context: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int | str]:
        if samples <= 0:
            raise ValueError("Partonic posterior sample count must be positive")
        if steps <= 0:
            raise ValueError("Partonic CFM integration steps must be positive")
        if solver not in {"euler", "heun"}:
            raise ValueError("Partonic CFM solver must be euler or heun")
        parent_context = self._conditioning_tokens(
            visible,
            parent_queries,
            assignment_context,
            predicted_neutrino_momentum,
        )
        batch = len(visible)
        prepared_context = None
        if self.velocity_backbone == "geometric":
            prepared_context = self.flow.denoiser.prepare_conditioning(
                visible,
                visible_padding_mask,
                parent_context,
                geometric_context=geometric_context,
                target_mean=self.target_mean,
                target_std=self.target_std,
            )
            flow_visible = visible
            flow_mask = visible_padding_mask
            flow_parent = parent_context
        else:
            flow_visible = visible.repeat_interleave(samples, dim=0)
            flow_mask = visible_padding_mask.repeat_interleave(samples, dim=0)
            flow_parent = parent_context.repeat_interleave(samples, dim=0)
        initial_state = None
        if source_state is not None:
            expected_shape = (batch, samples, 8)
            if tuple(source_state.shape) != expected_shape:
                raise ValueError(
                    "Explicit parent source_state must be [B,samples,8] in joint role order; "
                    f"expected {expected_shape}, found {tuple(source_state.shape)}"
                )
            if source_state.device != visible.device or source_state.dtype != visible.dtype:
                raise ValueError("Explicit parent source_state must match measured context device and dtype")
            if not torch.isfinite(source_state).all():
                raise ValueError("Explicit parent source_state contains non-finite values")
            initial_state = source_state.reshape(batch * samples, 8)
        scaled, nfe = self.flow.sample(
            (batch * samples, 8),
            flow_visible,
            flow_mask,
            flow_parent,
            steps=steps,
            solver=solver,
            generator=generator,
            initial_state=initial_state,
            prepared_context=prepared_context,
        )
        scaled = scaled.reshape(batch, samples, 2, 4)
        features = scaled * self.target_std + self.target_mean
        physical = four_vectors_from_partonic_features(features)
        pair = parent_pair_observables(physical)
        return {
            "scaled": scaled,
            "parents": physical,
            "pair": pair,
            "nfe": nfe,
            "conditioning": self.conditioning,
            "parent_role_conditioning": (
                "topology_query_states"
                if self.use_topology_queries
                else "learned_semantic_role_embeddings"
            ),
        }


def parent_pair_observables(four_vectors: torch.Tensor) -> dict[str, torch.Tensor]:
    """Derive pair and angular observables from generated parent four-vectors."""
    if four_vectors.ndim < 3 or four_vectors.shape[-2:] != (2, 4):
        raise ValueError("Parent four-vectors must end in [2,4] in (top, antitop), (E,px,py,pz) order")
    if not torch.isfinite(four_vectors).all():
        raise ValueError("Generated parent four-vectors contain non-finite values")
    top = four_vectors[..., 0, :]
    antitop = four_vectors[..., 1, :]
    pair = top + antitop
    energy = pair[..., 0]
    momentum = pair[..., 1:]
    mass_squared = energy.square() - momentum.square().sum(dim=-1)
    if (mass_squared < -1.0e-3).any() or (energy <= 0).any():
        raise ValueError("Generated parent pair lies outside the physical future-timelike domain")
    mass = mass_squared.clamp_min(0).sqrt()
    pt = torch.linalg.vector_norm(momentum[..., :2], dim=-1)
    rapidity = 0.5 * (
        (energy + momentum[..., 2]).clamp_min(1.0e-12).log()
        - (energy - momentum[..., 2]).clamp_min(1.0e-12).log()
    )
    phi = torch.atan2(momentum[..., 1], momentum[..., 0])
    delta_phi = torch.atan2(
        torch.sin(torch.atan2(top[..., 2], top[..., 1]) - torch.atan2(antitop[..., 2], antitop[..., 1])),
        torch.cos(torch.atan2(top[..., 2], top[..., 1]) - torch.atan2(antitop[..., 2], antitop[..., 1])),
    )
    beta = momentum / energy.unsqueeze(-1)
    beta_squared = beta.square().sum(dim=-1)
    beta_squared_safe = beta_squared.clamp_min(1.0e-12)
    gamma = torch.rsqrt((1.0 - beta_squared).clamp_min(1.0e-12))
    top_momentum = top[..., 1:]
    beta_dot_top = (beta * top_momentum).sum(dim=-1)
    coefficient = (gamma - 1.0) * beta_dot_top / beta_squared_safe - gamma * top[..., 0]
    top_rest = top_momentum + coefficient.unsqueeze(-1) * beta
    axis_norm = torch.linalg.vector_norm(momentum, dim=-1)
    top_norm = torch.linalg.vector_norm(top_rest, dim=-1)
    fallback_axis = torch.zeros_like(momentum)
    fallback_axis[..., 2] = 1.0
    axis = torch.where((axis_norm > 1.0e-8).unsqueeze(-1), momentum, fallback_axis)
    axis = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1.0e-12)
    cos_theta_star = (top_rest * axis).sum(dim=-1) / top_norm.clamp_min(1.0e-12)
    return {
        "ttbar": pair,
        "mttbar": mass,
        "ptttbar": pt,
        "yttbar": rapidity,
        "phittbar": phi,
        "delta_phi_top_antitop": delta_phi,
        "delta_rapidity_top_antitop": (
            0.5 * ((top[..., 0] + top[..., 3]).clamp_min(1.0e-12).log() - (top[..., 0] - top[..., 3]).clamp_min(1.0e-12).log())
            - 0.5 * ((antitop[..., 0] + antitop[..., 3]).clamp_min(1.0e-12).log() - (antitop[..., 0] - antitop[..., 3]).clamp_min(1.0e-12).log())
        ),
        "cos_theta_star": cos_theta_star.clamp(-1.0, 1.0),
    }


__all__ = [
    "PartonicFourVectorHead",
    "PartonicCFM",
    "PartonicCFMDenoiser",
    "four_vectors_from_partonic_features",
    "parent_pair_observables",
    "partonic_features",
]
