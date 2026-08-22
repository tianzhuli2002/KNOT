"""Single Lightning module for configured joint reconstruction experiments."""

from __future__ import annotations

import hashlib
import json

import lightning as L
import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.m0.data.module import build_token_schema
from joint_event_reconstruction.models.event import JointEventModel
from joint_event_reconstruction.topology.compiler import load_topology


class JointEventModule(L.LightningModule):
    def __init__(self, experiment, visible_config, *, invisible_scaler=None, synthetic_data: bool = False):
        super().__init__()
        self.spec = load_topology(experiment.topology_config)
        schema = build_token_schema(visible_config.dataset)
        model = experiment.model
        visible = model.visible
        reasoning = model.reasoning
        feedback = model.reconstruction.feedback
        invisible_config = model.invisible if bool(model.invisible.enabled) else None
        coupling_config = model.coupling if invisible_config is not None else None
        self.event_model = JointEventModel(
            self.spec, n_features=schema.dim, n_object_types=len(schema.source_type_ids),
            d_model=int(visible.d_model), n_heads=int(visible.n_heads), n_layers=int(visible.layers),
            d_feedforward=int(visible.d_feedforward), dropout=float(visible.dropout),
            reasoning_enabled=bool(reasoning.enabled), reasoning_block_layers=int(reasoning.block_layers),
            train_steps=tuple(int(x) for x in reasoning.train_steps), eval_steps=int(reasoning.eval_steps),
            fixed_input_reinjection=bool(reasoning.fixed_input_reinjection),
            feedback_enabled=bool(feedback.enabled), feedback_start_step=int(feedback.start_step),
            feedback_gate_initial_bias=float(feedback.gate_initial_bias), invisible_config=invisible_config,
            coupling_config=coupling_config,
            invisible_loss_weight=float(experiment.training.loss_weights.invisible),
        )
        architecture = OmegaConf.to_container(model, resolve=True)
        signature = hashlib.sha256(json.dumps(architecture, sort_keys=True).encode()).hexdigest()
        self.scaler_metadata = None if invisible_scaler is None else invisible_scaler.as_dict()
        self.save_hyperparameters({
            "topology_signature": self.spec.signature, "architecture_signature": signature,
            "input_features": list(schema.names), "source_type_ids": dict(schema.source_type_ids),
            "candidate_representation": "hierarchical_symmetric_v1",
            "recurrent_representation": "shared_visible_topology_tokens_v1" if reasoning.enabled else None,
            "invisible_formulation": str(model.invisible.formulation) if model.invisible.enabled else None,
            "invisible_target": "cartesian_px_py_pz_v1" if model.invisible.enabled else None,
            "synthetic_data": bool(synthetic_data),
        })
        self.learning_rate = float(experiment.training.learning_rate)
        self.weight_decay = float(experiment.training.weight_decay)
        self.scheduler_config = experiment.training.scheduler

    def forward(self, batch: dict, **kwargs) -> dict:
        return self.event_model(batch, **kwargs)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        output = self(batch)
        self.log(f"{stage}_loss", output["loss"], on_step=False, on_epoch=True,
                 batch_size=len(batch["tokens"]), prog_bar=stage != "train")
        self.log(f"{stage}_reconstruction_loss", output["reconstruction_loss"], on_step=False,
                 on_epoch=True, batch_size=len(batch["tokens"]))
        if "invisible_loss" in output:
            self.log(f"{stage}_invisible_loss", output["invisible_loss"], on_step=False,
                     on_epoch=True, batch_size=len(batch["tokens"]))
        return output["loss"]

    def training_step(self, batch, batch_idx): return self._step(batch, "train")
    def validation_step(self, batch, batch_idx): return self._step(batch, "val")
    def test_step(self, batch, batch_idx): return self._step(batch, "test")

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if not self.scheduler_config.enabled:
            return optimiser
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, factor=float(self.scheduler_config.factor), patience=int(self.scheduler_config.patience),
            min_lr=float(self.scheduler_config.min_lr),
        )
        return {"optimizer": optimiser, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["invisible_scaler"] = self.scaler_metadata

    def validate_checkpoint_metadata(self, checkpoint: dict) -> None:
        saved = checkpoint.get("hyper_parameters", {})
        for key in ("topology_signature", "architecture_signature", "input_features", "source_type_ids"):
            if saved.get(key) != self.hparams.get(key):
                raise ValueError(
                    f"Incompatible checkpoint metadata {key}: checkpoint={saved.get(key)!r}, "
                    f"current={self.hparams.get(key)!r}"
                )
        if checkpoint.get("invisible_scaler") != self.scaler_metadata:
            raise ValueError("Incompatible invisible target scaling metadata")
