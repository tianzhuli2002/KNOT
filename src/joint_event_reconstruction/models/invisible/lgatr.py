"""Conditional L-GATr backbone for generic invisible-particle generation."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.models.invisible.denoiser import noise_embedding


def _package():
    try:
        import lgatr
    except ImportError as exc:
        raise ImportError(
            "The Conditional L-GATr invisible backbone requires the optional "
            "dependency; install joint-event-reconstruction[lgatr]."
        ) from exc
    return lgatr


class ConditionalLGATrInvisibleDenoiser(nn.Module):
    """An objective-agnostic conditional geometric denoiser.

    The diffusion/flow formulation owns the native state and time semantics.
    This module only maps the noisy state and event conditions to a vector-field
    or noise prediction, so the same backbone can be used by CFM, OT-CFM and
    condition-aware OT-CFM.
    """

    def __init__(
        self,
        state_token_count: int,
        *,
        state_dim: int,
        visible_dim: int,
        parent_dim: int,
        d_model: int,
        backbone_config,
        zero_init_output: bool = True,
    ):
        super().__init__()
        if state_token_count <= 0:
            raise ValueError("Conditional L-GATr requires at least one state token")
        if state_dim not in {3, 4}:
            raise ValueError("Conditional L-GATr state_dim must be 3 or 4")
        lgatr = _package()
        self.state_token_count = int(state_token_count)
        self.state_dim = int(state_dim)
        self.visible_dim = int(visible_dim)
        self.parent_dim = int(parent_dim)
        self.d_model = int(d_model)

        beam = str(backbone_config.symmetry_breaking.beam)
        add_time = bool(backbone_config.symmetry_breaking.time)
        n_spurions = lgatr.get_num_spurions(beam, add_time)
        self.register_buffer(
            "spurions",
            lgatr.get_spurions(beam, add_time),
            persistent=False,
        )
        self.n_spurions = int(n_spurions)
        self.momentum_scale_gev = float(backbone_config.momentum_scale_gev)
        if self.momentum_scale_gev <= 0:
            raise ValueError("Conditional L-GATr momentum_scale_gev must be positive")
        hidden = backbone_config.hidden
        self.network = lgatr.ConditionalLGATr(
            num_blocks=int(backbone_config.layers),
            in_mv_channels=1 + self.n_spurions,
            mv_channels_cond=1 + self.n_spurions,
            out_mv_channels=1,
            hidden_mv_channels=int(hidden.multivector_channels),
            in_s_channels=self.parent_dim + self.d_model,
            s_channels_cond=self.visible_dim,
            out_s_channels=0,
            hidden_s_channels=int(hidden.scalar_channels),
            attention={"num_heads": int(backbone_config.n_heads)},
            crossattention={"num_heads": int(backbone_config.n_heads)},
            mlp={},
            dropout_prob=float(backbone_config.dropout),
            checkpoint_blocks=False,
            naive_amp=False,
            compile=False,
        )
        output_width = 4 if self.state_dim == 4 else 3
        self.output_projection = nn.Linear(4, output_width)
        self.condition_projection = nn.Linear(self.visible_dim + 2, self.visible_dim)
        if zero_init_output:
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

    def _state_four_vectors(self, noisy: torch.Tensor) -> torch.Tensor:
        if self.state_dim == 4:
            return noisy
        energy = noisy.square().sum(dim=-1, keepdim=True).clamp_min(0).sqrt()
        return torch.cat((energy, noisy), dim=-1)

    def _with_spurions(self, vectors: torch.Tensor) -> torch.Tensor:
        # ``vectors`` is [B,T,4] and the package expects [B,T,C,16].
        embedded = _package().embed_vector(vectors).unsqueeze(-2)
        if not self.n_spurions:
            return embedded
        spurions = self.spurions.to(device=vectors.device, dtype=embedded.dtype)
        spurions = spurions.view(1, 1, self.n_spurions, 16).expand(
            len(vectors), vectors.shape[1], -1, -1
        )
        return torch.cat((embedded, spurions), dim=-2)

    def _condition_vectors(
        self,
        visible: torch.Tensor,
        visible_fourmomenta: torch.Tensor | None,
    ) -> torch.Tensor:
        if visible_fourmomenta is None:
            vectors = visible.new_zeros(*visible.shape[:-1], 4)
        else:
            if visible_fourmomenta.shape != visible.shape[:-1] + (4,):
                raise ValueError(
                    "Visible four-momenta must align with the invisible condition "
                    f"tokens, found {tuple(visible_fourmomenta.shape)}"
                )
            vectors = visible_fourmomenta / self.momentum_scale_gev
        return self._with_spurions(vectors)

    def forward(
        self,
        noisy: torch.Tensor,
        noise_level: torch.Tensor,
        visible: torch.Tensor,
        visible_padding_mask: torch.Tensor,
        parent_context: torch.Tensor,
        *,
        visible_fourmomenta: torch.Tensor | None = None,
        met_xy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        expected = (self.state_token_count, self.state_dim)
        if noisy.ndim != 3 or noisy.shape[1:] != expected:
            raise ValueError(
                f"Noisy invisible state must be [B,{expected[0]},{expected[1]}], "
                f"found {tuple(noisy.shape)}"
            )
        if visible.ndim != 3 or visible.shape[-1] != self.visible_dim:
            raise ValueError("Conditional L-GATr visible context has an invalid shape")
        if visible_padding_mask.shape != visible.shape[:2]:
            raise ValueError("Visible padding mask does not align with L-GATr context")
        if parent_context.shape != (
            len(noisy), self.state_token_count, self.parent_dim
        ):
            raise ValueError("Conditional L-GATr parent context has an invalid shape")
        if noise_level.shape != (len(noisy),):
            raise ValueError("Conditional L-GATr noise level must have shape [B]")
        if met_xy is not None and met_xy.shape != (len(noisy), 2):
            raise ValueError("Conditional L-GATr MET must have shape [B,2]")

        condition_scalars = visible
        if met_xy is None:
            met_xy = visible.new_zeros((len(visible), 2))
        else:
            met_xy = met_xy / self.momentum_scale_gev
        condition_scalars = self.condition_projection(
            torch.cat((condition_scalars, met_xy.unsqueeze(1).expand(-1, len(visible[0]), -1)), dim=-1)
        )
        state_scalars = torch.cat(
            (
                parent_context,
                noise_embedding(noise_level, self.d_model)
                .unsqueeze(1)
                .expand(-1, self.state_token_count, -1),
            ),
            dim=-1,
        )
        output, _ = self.network(
            self._with_spurions(self._state_four_vectors(noisy)),
            self._condition_vectors(visible, visible_fourmomenta),
            scalars=state_scalars,
            scalars_cond=condition_scalars,
            crossattn_kwargs={
                "attn_mask": (~visible_padding_mask)[:, None, None, :]
            },
        )
        vector = _package().extract_vector(output[:, :, :1]).squeeze(-2)
        return self.output_projection(vector)
