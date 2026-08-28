"""Single Lightning module for configured reconstruction experiments."""

from __future__ import annotations

import hashlib
import json

import lightning as L
import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.m0.data.module import build_token_schema
from joint_event_reconstruction.models.event import JointEventModel
from joint_event_reconstruction.topology.compiler import load_topology


ARCHITECTURE_SIGNATURE_VERSION = 2


def _resolved(value):
    return OmegaConf.to_container(value, resolve=True)


def _selected(value, *keys: str) -> dict:
    resolved = _resolved(value)
    return {key: resolved[key] for key in keys}


def active_architecture(model) -> dict:
    """Return model/training semantics that must match a checkpoint.

    Sampling budgets and numerical solvers deliberately do not belong here: the
    same trained checkpoint must be usable for NFE/solver studies. Settings that
    define parameter shapes or the learned formulation remain part of the
    signature.
    """
    mode = str(model.get("mode", "factorized"))
    reconstruction_formulation = str(
        model.reconstruction.get("formulation", "discriminative")
    )
    payload = {
        "mode": mode,
        "visible": _resolved(model.visible),
        "reconstruction": {"formulation": reconstruction_formulation},
    }

    if mode == "factorized":
        payload["reasoning"] = _resolved(model.reasoning)
        if reconstruction_formulation == "discriminative":
            payload["reconstruction"]["feedback"] = _resolved(
                model.reconstruction.feedback
            )
        else:
            payload["reconstruction"]["flow_matching"] = _resolved(
                model.reconstruction.flow_matching
            )

    if bool(model.invisible.enabled):
        formulation = str(model.invisible.formulation)
        invisible = {
            "enabled": True,
            "formulation": formulation,
            "target": _resolved(model.invisible.target),
            "denoiser": _resolved(model.invisible.denoiser),
        }
        if mode == "factorized":
            invisible["conditioning"] = _resolved(model.invisible.conditioning)

        if formulation == "standard":
            invisible["standard"] = _selected(
                model.invisible.standard, "training_steps"
            )
        elif formulation == "edm":
            invisible["edm"] = _selected(
                model.invisible.edm, "sigma_data", "p_mean", "p_std"
            )
        elif formulation == "flow_matching":
            invisible["flow_matching"] = _selected(
                model.invisible.flow_matching, "path", "coupling"
            )
        else:
            raise ValueError(f"Unknown invisible formulation {formulation!r}")
        payload["invisible"] = invisible

    if mode == "joint_flow":
        payload["joint_flow"] = _resolved(model.joint_flow)
    else:
        if bool(model.coupling.invisible_to_reconstruction.enabled):
            payload["coupling"] = _resolved(model.coupling)
        if bool(model.compatibility.enabled):
            payload["compatibility"] = _resolved(model.compatibility)
    return payload


class JointEventModule(L.LightningModule):
    def __init__(
        self,
        experiment,
        visible_config,
        *,
        invisible_scaler=None,
        synthetic_data: bool = False,
    ):
        super().__init__()
        self.spec = load_topology(experiment.topology_config)
        schema = build_token_schema(visible_config.dataset)
        model = experiment.model
        visible = model.visible
        reasoning = model.reasoning
        feedback = model.reconstruction.feedback
        invisible_config = model.invisible if bool(model.invisible.enabled) else None
        coupling_config = (
            model.coupling
            if invisible_config is not None
            and str(model.get("mode", "factorized")) == "factorized"
            else None
        )
        self.event_model = JointEventModel(
            self.spec,
            n_features=schema.dim,
            n_object_types=len(schema.source_type_ids),
            d_model=int(visible.d_model),
            n_heads=int(visible.n_heads),
            n_layers=int(visible.layers),
            d_feedforward=int(visible.d_feedforward),
            dropout=float(visible.dropout),
            model_mode=str(model.get("mode", "factorized")),
            reconstruction_formulation=str(
                model.reconstruction.get("formulation", "discriminative")
            ),
            reasoning_enabled=bool(reasoning.enabled),
            reasoning_block_layers=int(reasoning.block_layers),
            train_steps=tuple(int(x) for x in reasoning.train_steps),
            eval_steps=int(reasoning.eval_steps),
            fixed_input_reinjection=bool(reasoning.fixed_input_reinjection),
            feedback_enabled=bool(feedback.enabled),
            feedback_start_step=int(feedback.start_step),
            feedback_gate_initial_bias=float(feedback.gate_initial_bias),
            invisible_config=invisible_config,
            coupling_config=coupling_config,
            invisible_loss_weight=float(experiment.training.loss_weights.invisible),
            compatibility_config=model.compatibility,
            compatibility_loss_weight=float(
                experiment.training.loss_weights.compatibility
            ),
            reconstruction_loss_weight=float(
                experiment.training.loss_weights.reconstruction
            ),
            joint_flow_config=model.get("joint_flow"),
        )
        freeze = experiment.training.freeze
        if bool(freeze.visible):
            self.event_model.visible.requires_grad_(False)
        if bool(freeze.reconstruction):
            self.event_model.reconstruction.requires_grad_(False)
            if self.event_model.assignment_flow is not None:
                self.event_model.assignment_flow.requires_grad_(False)
        if bool(freeze.invisible) and self.event_model.invisible is not None:
            self.event_model.invisible.requires_grad_(False)

        architecture = active_architecture(model)
        signature = hashlib.sha256(
            json.dumps(architecture, sort_keys=True).encode()
        ).hexdigest()
        self.scaler_metadata = (
            None if invisible_scaler is None else invisible_scaler.as_dict()
        )
        mode = str(model.get("mode", "factorized"))
        recurrent_representation = None
        if mode == "joint_flow" and bool(model.joint_flow.recurrent_state):
            recurrent_representation = "persistent_topology_flow_state_v1"
        elif bool(reasoning.enabled):
            recurrent_representation = "shared_visible_topology_tokens_v1"
        self.save_hyperparameters({
            "topology_signature": self.spec.signature,
            "architecture_signature_version": ARCHITECTURE_SIGNATURE_VERSION,
            "architecture_signature": signature,
            "model_mode": mode,
            "assignment_formulation": str(
                model.reconstruction.get("formulation", "discriminative")
            ),
            "input_features": list(schema.names),
            "source_type_ids": dict(schema.source_type_ids),
            "candidate_representation": "hierarchical_symmetric_v1",
            "assignment_state_representation": "complete_feasible_states_v1",
            "recurrent_representation": recurrent_representation,
            "invisible_formulation": (
                str(model.invisible.formulation) if model.invisible.enabled else None
            ),
            "invisible_target": (
                "cartesian_px_py_pz_v1" if model.invisible.enabled else None
            ),
            "invisible_roles": [role.key for role in self.spec.invisible_roles],
            "invisible_role_count": len(self.spec.invisible_roles),
            "canonical_data_schema": (
                "joint_event_reconstruction_canonical_v1"
                if model.invisible.enabled and not synthetic_data
                else None
            ),
            "synthetic_data": bool(synthetic_data),
        })
        self.learning_rate = float(experiment.training.learning_rate)
        self.weight_decay = float(experiment.training.weight_decay)
        self.scheduler_config = experiment.training.scheduler

    def forward(self, batch: dict, **kwargs) -> dict:
        return self.event_model(batch, **kwargs)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        output = self(batch)
        batch_size = len(batch["tokens"])
        self.log(
            f"{stage}_loss",
            output["loss"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
            prog_bar=stage != "train",
        )
        self.log(
            f"{stage}_reconstruction_loss",
            output["reconstruction_loss"],
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        if "invisible_loss" in output:
            self.log(
                f"{stage}_invisible_loss",
                output["invisible_loss"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        if "compatibility_loss" in output:
            self.log(
                f"{stage}_compatibility_loss",
                output["compatibility_loss"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
        return output["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimiser = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        if not self.scheduler_config.enabled:
            return optimiser
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            factor=float(self.scheduler_config.factor),
            patience=int(self.scheduler_config.patience),
            min_lr=float(self.scheduler_config.min_lr),
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["invisible_scaler"] = self.scaler_metadata

    def validate_checkpoint_metadata(self, checkpoint: dict) -> None:
        saved = checkpoint.get("hyper_parameters", {})
        for key in ("topology_signature", "input_features", "source_type_ids"):
            if saved.get(key) != self.hparams.get(key):
                raise ValueError(
                    f"Incompatible checkpoint metadata {key}: "
                    f"checkpoint={saved.get(key)!r}, current={self.hparams.get(key)!r}"
                )
        if saved.get("architecture_signature_version") == ARCHITECTURE_SIGNATURE_VERSION:
            if saved.get("architecture_signature") != self.hparams.get("architecture_signature"):
                raise ValueError(
                    "Incompatible active architecture signature: "
                    f"checkpoint={saved.get('architecture_signature')!r}, "
                    f"current={self.hparams.get('architecture_signature')!r}"
                )
        if checkpoint.get("invisible_scaler") != self.scaler_metadata:
            raise ValueError("Incompatible invisible target scaling metadata")
