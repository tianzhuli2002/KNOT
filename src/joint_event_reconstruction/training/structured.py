"""Lightning integration shared by M1 and later structured reconstruction stages."""

from __future__ import annotations

import lightning as L
import torch

from joint_event_reconstruction.m0.data.module import build_token_schema
from joint_event_reconstruction.models.event import StructuredEventModel
from joint_event_reconstruction.topology.compiler import load_topology


class StructuredEventModule(L.LightningModule):
    def __init__(self, experiment, visible_config):
        super().__init__()
        self.spec = load_topology(experiment.topology_config)
        schema = build_token_schema(visible_config.dataset)
        model = experiment.model
        self.event_model = StructuredEventModel(
            self.spec, n_features=schema.dim, n_object_types=len(schema.source_type_ids),
            d_model=int(model.d_model), n_heads=int(model.n_heads), n_layers=int(model.n_layers),
            d_feedforward=int(model.d_feedforward), dropout=float(model.dropout),
        )
        metadata = {
            "stage": "M1",
            "topology_signature": self.spec.signature,
            "topology_config": str(experiment.topology_config),
            "input_features": list(schema.names),
            "source_type_ids": dict(schema.source_type_ids),
            "structured_candidate_version": "hierarchical_symmetric_v1",
            "hierarchical_representation_version": "child_embedding_v1",
            "model": {key: model[key] for key in model},
        }
        self.save_hyperparameters(metadata)
        self.learning_rate = float(experiment.training.learning_rate)
        self.weight_decay = float(experiment.training.weight_decay)

    def forward(self, batch: dict) -> dict:
        return self.event_model(batch)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        output = self(batch)
        self.log(f"{stage}_loss", output["loss"], on_step=False, on_epoch=True,
                 batch_size=len(batch["tokens"]), prog_bar=stage != "train")
        for name, loss in output["node_losses"].items():
            self.log(f"{stage}_{name}_loss", loss, on_step=False, on_epoch=True,
                     batch_size=len(batch["tokens"]))
        return output["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        def move(value):
            if torch.is_tensor(value):
                return value.to(device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value
        return move(batch)

    def validate_checkpoint_metadata(self, checkpoint: dict) -> None:
        saved = checkpoint.get("hyper_parameters", {})
        for key in (
            "stage", "topology_signature", "input_features", "source_type_ids",
            "structured_candidate_version", "hierarchical_representation_version", "model",
        ):
            if saved.get(key) != self.hparams.get(key):
                raise ValueError(
                    f"Incompatible checkpoint metadata {key}: checkpoint={saved.get(key)!r}, "
                    f"current={self.hparams.get(key)!r}"
                )
