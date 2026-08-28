"""Single Lightning module for maintained event-reconstruction experiments."""

from __future__ import annotations

import hashlib
import json

import lightning as L
import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.m0.data.module import build_token_schema
from joint_event_reconstruction.models.event import JointEventModel
from joint_event_reconstruction.topology.compiler import load_topology


ARCHITECTURE_SIGNATURE_VERSION = 4


def _resolved(value):
    return OmegaConf.to_container(value, resolve=True)


def _selected(value, *keys: str) -> dict:
    resolved = _resolved(value)
    return {key: resolved[key] for key in keys}


def active_architecture(model) -> dict:
    """Return learned/training semantics that define checkpoint identity.

    Inference depth, numerical solvers, reranking strength, oracle permission and
    assignment-state safety limits are deliberately absent. They may be scanned with the
    same trained checkpoint.
    """
    mode = str(model.mode)
    assignment = model.assignment
    formulation = str(assignment.formulation)
    payload = {
        "mode": mode,
        "visible": _resolved(model.visible),
        "assignment": {"formulation": formulation},
    }
    if mode == "factorized":
        payload["assignment"]["reasoning"] = _selected(
            assignment.reasoning,
            "enabled",
            "block_layers",
            "train_steps",
            "fixed_input_reinjection",
        )
        if formulation == "discriminative":
            payload["assignment"]["feedback"] = _resolved(assignment.feedback)

    if bool(model.invisible.enabled):
        invisible_formulation = str(model.invisible.formulation)
        invisible = {
            "enabled": True,
            "formulation": invisible_formulation,
            "denoiser": _resolved(model.invisible.denoiser),
        }
        if mode == "factorized":
            invisible["conditioning"] = {
                "assignment": str(model.invisible.conditioning.assignment)
            }
        if invisible_formulation == "standard":
            invisible["standard"] = _resolved(model.invisible.standard)
        elif invisible_formulation == "edm":
            invisible["edm"] = _resolved(model.invisible.edm)
        elif invisible_formulation == "flow_matching":
            invisible["flow_matching"] = _resolved(model.invisible.flow_matching)
        else:
            raise ValueError(
                f"Unknown invisible formulation {invisible_formulation!r}"
            )
        payload["invisible"] = invisible

    if mode == "joint_flow":
        payload["joint_flow"] = _resolved(model.joint_flow)
    else:
        if bool(model.coupling.invisible_to_assignment.enabled):
            payload["coupling"] = _resolved(model.coupling)
        if bool(model.compatibility.enabled):
            payload["compatibility"] = _resolved(model.compatibility)
    return payload


def _freeze(module) -> None:
    if module is not None:
        module.requires_grad_(False)


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
        assignment = model.assignment
        reasoning = assignment.reasoning
        feedback = assignment.feedback
        invisible_config = model.invisible if bool(model.invisible.enabled) else None
        coupling_config = (
            model.coupling
            if invisible_config is not None and str(model.mode) == "factorized"
            else None
        )
        self.event_model = JointEventModel(
            self.spec,
            n_features=schema.dim,
            n_object_types=len(schema.source_type_ids),
            d_model=int(model.visible.d_model),
            n_heads=int(model.visible.n_heads),
            n_layers=int(model.visible.layers),
            d_feedforward=int(model.visible.d_feedforward),
            dropout=float(model.visible.dropout),
            model_mode=str(model.mode),
            assignment_formulation=str(assignment.formulation),
            reasoning_enabled=bool(reasoning.enabled),
            reasoning_block_layers=int(reasoning.block_layers),
            train_steps=tuple(int(step) for step in reasoning.train_steps),
            eval_steps=int(reasoning.eval_steps),
            fixed_input_reinjection=bool(reasoning.fixed_input_reinjection),
            feedback_enabled=bool(feedback.enabled),
            feedback_start_step=int(feedback.start_step),
            feedback_gate_initial_bias=float(feedback.gate_initial_bias),
            invisible_config=invisible_config,
            invisible_inference_config=(
                experiment.inference.invisible if invisible_config is not None else None
            ),
            coupling_config=coupling_config,
            assignment_loss_weight=float(experiment.training.loss_weights.assignment),
            invisible_loss_weight=float(experiment.training.loss_weights.invisible),
            compatibility_config=model.compatibility,
            compatibility_loss_weight=float(
                experiment.training.loss_weights.compatibility
            ),
            joint_flow_config=model.joint_flow,
        )

        freeze = experiment.training.freeze
        if bool(freeze.visible):
            _freeze(self.event_model.visible)
        if bool(freeze.assignment):
            for component in (
                self.event_model.reconstruction,
                self.event_model.reasoner,
                self.event_model.feedback,
                self.event_model.assignment_flow,
                self.event_model.posterior_feedback,
            ):
                _freeze(component)
        if bool(freeze.invisible):
            _freeze(self.event_model.invisible)
        if bool(freeze.compatibility):
            _freeze(self.event_model.compatibility)

        architecture = active_architecture(model)
        signature = hashlib.sha256(
            json.dumps(architecture, sort_keys=True).encode()
        ).hexdigest()
        self.scaler_metadata = (
            None if invisible_scaler is None else invisible_scaler.as_dict()
        )
        mode = str(model.mode)
        assignment_formulation = str(assignment.formulation)
        recurrent_representation = None
        if mode == "joint_flow" and bool(model.joint_flow.recurrent_state):
            recurrent_representation = "persistent_topology_flow_state_v1"
        elif bool(reasoning.enabled):
            recurrent_representation = "shared_visible_topology_tokens_v1"
        self.save_hyperparameters(
            {
                "topology_signature": self.spec.signature,
                "architecture_signature_version": ARCHITECTURE_SIGNATURE_VERSION,
                "architecture_signature": signature,
                "model_mode": mode,
                "assignment_formulation": assignment_formulation,
                "input_features": list(schema.names),
                "source_type_ids": dict(schema.source_type_ids),
                "candidate_representation": "hierarchical_symmetric_v1",
                "assignment_state_representation": (
                    "complete_feasible_states_v1"
                    if assignment_formulation == "flow_matching"
                    else None
                ),
                "recurrent_representation": recurrent_representation,
                "invisible_formulation": (
                    str(model.invisible.formulation)
                    if model.invisible.enabled
                    else None
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
            }
        )
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
            f"{stage}_assignment_loss",
            output["assignment_loss"],
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
        trainable = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError("Experiment freezes every trainable parameter")
        optimiser = torch.optim.AdamW(
            trainable,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=bool(trainable[0].is_cuda),
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
            if saved.get("architecture_signature") != self.hparams.get(
                "architecture_signature"
            ):
                raise ValueError(
                    "Incompatible active architecture signature: "
                    f"checkpoint={saved.get('architecture_signature')!r}, "
                    f"current={self.hparams.get('architecture_signature')!r}"
                )
        if checkpoint.get("invisible_scaler") != self.scaler_metadata:
            raise ValueError("Incompatible invisible target scaling metadata")
