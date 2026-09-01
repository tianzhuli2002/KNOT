"""Adapters for the optional standalone L-GATr 2.x package."""

from __future__ import annotations

import torch
from torch import nn

from joint_event_reconstruction.models.visible.physical_features import (
    physical_encoder_input,
)


def _package():
    try:
        import lgatr
    except ImportError as exc:
        raise ImportError(
            "The selected L-GATr visible encoder requires the optional dependency; "
            "install joint-event-reconstruction[lgatr]."
        ) from exc
    return lgatr


def _scalar_input(physical, event_state, n_object_types: int) -> torch.Tensor:
    type_one_hot = torch.nn.functional.one_hot(
        physical.type_ids, num_classes=n_object_types
    ).to(dtype=physical.scalars.dtype)
    object_state = event_state.new_zeros(
        len(physical.scalars), physical.scalars.shape[1], event_state.shape[-1]
    )
    objects = torch.cat((physical.scalars, type_one_hot, object_state), dim=-1)
    event_prefix = physical.scalars.new_zeros(
        len(physical.scalars), 1, physical.scalars.shape[-1] + n_object_types
    )
    event = torch.cat(
        (event_prefix, event_state.expand(len(physical.scalars), -1, -1)), dim=-1
    )
    return torch.cat((event, objects), dim=1)


def _attention_mask(padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    event_padding = torch.zeros(
        len(padding), 1, dtype=torch.bool, device=padding.device
    )
    full_padding = torch.cat((event_padding, padding), dim=1)
    # torch scaled_dot_product_attention uses True for participating elements.
    mask = (~full_padding)[:, None, None, :]
    return mask, full_padding


class LGATrVisibleEncoder(nn.Module):
    def __init__(self, config, schema, n_object_types: int):
        super().__init__()
        lgatr = _package()
        self.config = config
        self.schema = schema
        self.n_object_types = n_object_types
        self.event_state = nn.Parameter(torch.empty(1, 1, int(config.d_model)))
        nn.init.normal_(self.event_state, mean=0.0, std=0.02)
        beam = str(config.spurions.beam)
        add_time = bool(config.spurions.time)
        n_spurions = lgatr.get_num_spurions(beam, add_time)
        self.register_buffer(
            "spurions",
            lgatr.get_spurions(beam, add_time),
            persistent=False,
        )
        scalar_count = len(config.scalar_features) + n_object_types + int(config.d_model)
        self.network = lgatr.LGATr(
            num_blocks=int(config.blocks),
            in_mv_channels=1 + n_spurions,
            out_mv_channels=1,
            hidden_mv_channels=int(config.hidden_mv_channels),
            in_s_channels=scalar_count,
            out_s_channels=int(config.d_model),
            hidden_s_channels=int(config.hidden_s_channels),
            attention={"num_heads": int(config.heads)},
            mlp={},
            reinsert_mv_channels=tuple(range(1, 1 + n_spurions)),
            dropout_prob=float(config.dropout),
            checkpoint_blocks=False,
            naive_amp=False,
            compile=False,
        )

    def forward(self, batch: dict) -> dict:
        lgatr = _package()
        physical = physical_encoder_input(batch, self.schema, self.config)
        vectors = lgatr.embed_vector(physical.fourmomenta).unsqueeze(-2)
        spurions = self.spurions.to(dtype=vectors.dtype).view(1, 1, -1, 16)
        spurions = spurions.expand(len(vectors), vectors.shape[1], -1, -1)
        object_mv = torch.cat((vectors, spurions), dim=-2)
        event_mv = object_mv.new_zeros(len(vectors), 1, object_mv.shape[-2], 16)
        multivectors = torch.cat((event_mv, object_mv), dim=1)
        scalars = _scalar_input(physical, self.event_state, self.n_object_types)
        attention_mask, full_padding = _attention_mask(physical.padding_mask)
        _, encoded = self.network(multivectors, scalars, attn_mask=attention_mask)
        encoded = encoded.masked_fill(full_padding.unsqueeze(-1), 0.0)
        return {
            "pooled": encoded[:, 0],
            "tokens": encoded[:, 1:],
            "padding_mask": physical.padding_mask,
        }


class LGATrSlimVisibleEncoder(nn.Module):
    def __init__(self, config, schema, n_object_types: int):
        super().__init__()
        lgatr = _package()
        self.config = config
        self.schema = schema
        self.n_object_types = n_object_types
        self.event_state = nn.Parameter(torch.empty(1, 1, int(config.d_model)))
        nn.init.normal_(self.event_state, mean=0.0, std=0.02)
        self.register_buffer(
            "vector_spurions",
            self._vector_spurions(config),
            persistent=False,
        )
        scalar_count = len(config.scalar_features) + n_object_types + int(config.d_model)
        self.network = lgatr.LGATrSlim(
            num_blocks=int(config.blocks),
            in_v_channels=1 + len(self.vector_spurions),
            out_v_channels=1,
            hidden_v_channels=int(config.hidden_vector_channels),
            in_s_channels=scalar_count,
            out_s_channels=int(config.d_model),
            hidden_s_channels=int(config.hidden_scalar_channels),
            num_heads=int(config.heads),
            dropout_prob=float(config.dropout),
            checkpoint_blocks=False,
            naive_amp=False,
            compile=False,
        )

    @staticmethod
    def _vector_spurions(config) -> torch.Tensor:
        vectors = []
        beam = str(config.spurions.beam)
        if beam == "xyplane":
            # Slim has no bivector stream. Its vector-only representation uses the
            # beam-normal axis; together with time this preserves the detector O(2)
            # subgroup targeted by the xy-plane spurion in full L-GATr.
            vectors.append((0.0, 0.0, 0.0, 1.0))
        elif beam != "none":
            raise ValueError("lgatr_slim spurions.beam must be xyplane or none")
        if bool(config.spurions.time):
            vectors.append((1.0, 0.0, 0.0, 0.0))
        return torch.tensor(vectors, dtype=torch.float32).reshape(-1, 4)

    def forward(self, batch: dict) -> dict:
        physical = physical_encoder_input(batch, self.schema, self.config)
        vectors = physical.fourmomenta.unsqueeze(-2)
        spurions = self.vector_spurions.to(dtype=vectors.dtype).view(1, 1, -1, 4)
        spurions = spurions.expand(len(vectors), vectors.shape[1], -1, -1)
        object_vectors = torch.cat((vectors, spurions), dim=-2)
        event_vectors = object_vectors.new_zeros(
            len(vectors), 1, object_vectors.shape[-2], 4
        )
        vectors = torch.cat((event_vectors, object_vectors), dim=1)
        scalars = _scalar_input(physical, self.event_state, self.n_object_types)
        attention_mask, full_padding = _attention_mask(physical.padding_mask)
        _, encoded = self.network(vectors, scalars, attn_mask=attention_mask)
        encoded = encoded.masked_fill(full_padding.unsqueeze(-1), 0.0)
        return {
            "pooled": encoded[:, 0],
            "tokens": encoded[:, 1:],
            "padding_mask": physical.padding_mask,
        }
