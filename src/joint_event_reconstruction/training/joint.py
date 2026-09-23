"""Single Lightning module for maintained event-reconstruction experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json

import lightning as L
import torch
from omegaconf import OmegaConf

from joint_event_reconstruction.m0.data.module import build_token_schema
from joint_event_reconstruction.models.event import JointEventModel
from joint_event_reconstruction.topology.compiler import load_topology


ARCHITECTURE_SIGNATURE_VERSION = 5


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
        "latent": _resolved(model.latent),
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
            "coordinates": _resolved(model.invisible.coordinates),
            "mass_shell": _resolved(model.invisible.mass_shell),
            "backbone": _resolved(model.invisible.backbone),
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
            raise ValueError(f"Unknown invisible formulation {invisible_formulation!r}")
        payload["invisible"] = invisible

    if mode == "joint_flow":
        payload["joint_flow"] = _resolved(model.joint_flow)
    else:
        if bool(model.coupling.invisible_to_assignment.enabled):
            payload["coupling"] = _resolved(model.coupling)
        if bool(model.compatibility.enabled):
            payload["compatibility"] = _resolved(model.compatibility)
    topology_queries = getattr(model, "topology_queries", None)
    if topology_queries is not None and bool(topology_queries.enabled):
        payload["topology_queries"] = _resolved(topology_queries)
    heads = getattr(model, "heads", None)
    if heads is not None and any(
        bool(getattr(heads, name).enabled)
        for name in ("point_invisible", "parent_four_vector", "partonic")
    ):
        head_payload = _resolved(heads)
        # ``partonic`` was added after the existing point/parent checkpoint
        # family.  Keep a disabled new head out of the historical active
        # identity so loading those checkpoints remains byte-for-byte
        # compatible; enabled partonic heads remain fully identified.
        if not bool(getattr(heads.partonic, "enabled", False)):
            head_payload.pop("partonic", None)
        else:
            # The sidecar is a training/evaluation input, not learned model
            # structure.  Keep role and truth-binding semantics in the
            # checkpoint identity but do not make reload depend on a path.
            head_payload["partonic"].pop("target_path", None)
        payload["heads"] = head_payload
    return payload


def architecture_component_signatures(model) -> dict[str, str]:
    """Hash the semantic architecture owned by each partial-load component."""
    architecture = active_architecture(model)
    payloads = {
        "visible": {
            "latent_width": architecture["latent"]["d_model"],
            "visible": architecture["visible"],
        },
        "assignment": {
            "latent": architecture["latent"],
            "assignment": architecture["assignment"],
            "joint_flow": architecture.get("joint_flow"),
        },
        "invisible": {
            "latent_width": architecture["latent"]["d_model"],
            "invisible": architecture.get("invisible"),
            "joint_flow": architecture.get("joint_flow"),
        },
        "c2": {
            "latent_width": architecture["latent"]["d_model"],
            "coupling": architecture.get("coupling"),
        },
        "compatibility": {
            "latent_width": architecture["latent"]["d_model"],
            "compatibility": architecture.get("compatibility"),
        },
    }
    heads = architecture.get("heads", {})
    if heads.get("point_invisible", {}).get("enabled", False):
        payloads["point"] = {
            "latent_width": architecture["latent"]["d_model"],
            "head": heads["point_invisible"],
        }
    if heads.get("parent_four_vector", {}).get("enabled", False):
        payloads["parent"] = {
            "latent_width": architecture["latent"]["d_model"],
            "head": heads["parent_four_vector"],
        }
    if heads.get("partonic", {}).get("enabled", False):
        payloads["partonic"] = {
            "latent_width": architecture["latent"]["d_model"],
            "head": heads["partonic"],
            "topology_queries": architecture.get("topology_queries"),
        }
    if architecture.get("topology_queries") is not None:
        payloads["queries"] = {
            "latent_width": architecture["latent"]["d_model"],
            "topology_queries": architecture["topology_queries"],
        }
    return {
        component: hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for component, payload in payloads.items()
    }


def _freeze(module) -> None:
    if module is not None:
        module.requires_grad_(False)


def _apply_trainable_components(event_model, mode: str, components: set[str]) -> None:
    """Freeze the complement of the positive, user-facing training component list."""
    if "visible" not in components:
        _freeze(event_model.visible)
    if "queries" not in components:
        _freeze(event_model.topology_queries)

    if mode == "joint_flow":
        # Joint-flow topology state is shared by the discrete and continuous dynamics.
        # Configuration validation therefore keeps assignment/invisible trainability tied.
        if "assignment" not in components:
            _freeze(event_model.reconstruction)
            _freeze(event_model.invisible)
            _freeze(event_model.joint_flow)
        return

    if "assignment" not in components:
        for component in (
            event_model.reconstruction,
            event_model.reasoner,
            event_model.feedback,
            event_model.assignment_flow,
        ):
            _freeze(component)
    if "invisible" not in components:
        _freeze(event_model.invisible)
    if "c2" not in components:
        _freeze(event_model.posterior_feedback)
    if "compatibility" not in components:
        _freeze(event_model.compatibility)
    if "point" not in components:
        _freeze(event_model.point_head)
    if "parent" not in components:
        _freeze(event_model.parent_head)
    if "partonic" not in components:
        _freeze(event_model.partonic_head)


class JointEventModule(L.LightningModule):
    def __init__(
        self,
        experiment,
        visible_config,
        *,
        invisible_scaler=None,
        point_scaler=None,
        partonic_scaler=None,
        synthetic_data: bool = False,
    ):
        super().__init__()
        self.spec = load_topology(
            experiment.topology_config,
            OmegaConf.to_container(experiment.data.truth_bindings, resolve=True),
        )
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
            token_schema=schema,
            latent_config=model.latent,
            visible_config=model.visible,
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
            invisible_scaler=invisible_scaler,
            coupling_config=coupling_config,
            assignment_loss_weight=float(experiment.training.loss_weights.assignment),
            invisible_loss_weight=float(experiment.training.loss_weights.invisible),
            compatibility_config=model.compatibility,
            compatibility_loss_weight=float(experiment.training.loss_weights.compatibility),
            joint_flow_config=model.joint_flow,
            heads_config=model.heads,
            point_scaler=point_scaler,
            point_loss_weight=float(experiment.training.loss_weights.point),
            parent_loss_weight=float(experiment.training.loss_weights.parent),
            shared_encoder_gradients=str(experiment.training.shared_encoder_gradients),
            topology_queries_config=(
                model.topology_queries if bool(model.topology_queries.enabled) else None
            ),
            partonic_config=(
                model.heads.partonic if bool(model.heads.partonic.enabled) else None
            ),
            partonic_scaler=partonic_scaler,
            partonic_loss_weight=float(experiment.training.loss_weights.partonic),
            partonic_shared_encoder_gradients=str(
                experiment.training.partonic_shared_encoder_gradients
            ),
        )

        trainable_components = {str(value) for value in experiment.training.trainable}
        _apply_trainable_components(
            self.event_model, str(model.mode), trainable_components
        )

        architecture = active_architecture(model)
        signature = hashlib.sha256(
            json.dumps(architecture, sort_keys=True).encode()
        ).hexdigest()
        component_signatures = architecture_component_signatures(model)
        self.scaler_metadata = (
            None if invisible_scaler is None else invisible_scaler.as_dict()
        )
        self.point_scaler_metadata = (
            None
            if self.event_model.point_head is None or point_scaler is None
            else point_scaler.as_dict()
        )
        self.partonic_scaler_metadata = (
            None
            if self.event_model.partonic_head is None or partonic_scaler is None
            else partonic_scaler.as_dict()
        )
        if invisible_config is not None:
            if invisible_scaler is None:
                raise ValueError("Invisible models require a train-fitted coordinate scaler")
            coordinates = self.event_model.invisible.coordinates
            expected_representation = (
                f"invisible_coordinate_state_{coordinates.coordinate_type}_v1"
            )
            if invisible_scaler.representation != expected_representation:
                raise ValueError(
                    "Invisible scaler coordinate representation differs from the model: "
                    f"expected={expected_representation}, "
                    f"observed={invisible_scaler.representation}"
                )
            if tuple(invisible_scaler.role_names) != tuple(coordinates.token_names):
                raise ValueError("Invisible scaler state-token ordering differs from the model")
            if tuple(invisible_scaler.active_mask or ()) != tuple(
                tuple(bool(value) for value in row)
                for row in coordinates.active_mask.tolist()
            ):
                raise ValueError("Invisible scaler active mask differs from the model")
        mode = str(model.mode)
        assignment_formulation = str(assignment.formulation)
        recurrent_representation = None
        if mode == "joint_flow" and bool(model.joint_flow.recurrent_state):
            recurrent_representation = "persistent_topology_flow_state_v1"
        elif bool(reasoning.enabled):
            recurrent_representation = "shared_visible_topology_tokens_v1"
        lgatr_package_version = None
        if str(model.visible.type) in {"lgatr", "lgatr_slim"}:
            lgatr_package_version = importlib.metadata.version("lgatr")
        self.save_hyperparameters(
            {
                "topology_signature": self.spec.signature,
                "architecture_signature_version": ARCHITECTURE_SIGNATURE_VERSION,
                "architecture_signature": signature,
                "architecture_component_signatures": component_signatures,
                "visible_encoder": str(model.visible.type),
                "visible_encoder_config": _resolved(model.visible),
                "encoder_input_contract": {
                    "kinematics": str(model.visible.inputs.kinematics),
                    "object_features": list(model.visible.inputs.object_features),
                    "event_features": list(model.visible.inputs.event_features),
                    "physical_vector_representation": "E_px_py_pz_GeV_v1",
                },
                "lgatr_package_version": lgatr_package_version,
                "model_mode": mode,
                "assignment_formulation": assignment_formulation,
                "input_features": list(schema.names),
                "source_type_ids": dict(schema.source_type_ids),
                "physics_vector_representation": "E_px_py_pz_GeV_v1",
                "candidate_representation": "hierarchical_symmetric_v1",
                "assignment_state_representation": (
                    "complete_feasible_states_v1"
                    if assignment_formulation == "flow_matching"
                    else None
                ),
                "recurrent_representation": recurrent_representation,
                "invisible_formulation": (
                    str(model.invisible.formulation) if model.invisible.enabled else None
                ),
                "invisible_target": (
                    f"coordinate_state_{model.invisible.coordinates.type}_v1"
                    if model.invisible.enabled else None
                ),
                "invisible_coordinates": (
                    str(model.invisible.coordinates.type)
                    if model.invisible.enabled else None
                ),
                "invisible_mass_shell": (
                    _resolved(model.invisible.mass_shell)
                    if model.invisible.enabled else None
                ),
                "invisible_generator_backbone": (
                    _resolved(model.invisible.backbone)
                    if model.invisible.enabled else None
                ),
                "invisible_coordinate_metadata": (
                    self.event_model.invisible.coordinates.metadata()
                    if model.invisible.enabled else None
                ),
                "invisible_roles": [role.key for role in self.spec.invisible_roles],
                "invisible_role_count": len(self.spec.invisible_roles),
                "optional_heads": (
                    _resolved(model.heads)
                    if any(
                        bool(getattr(model.heads, name).enabled)
                        for name in ("point_invisible", "parent_four_vector", "partonic")
                    )
                    else None
                ),
                "flow_coupling": (
                    _resolved(model.invisible.flow_matching.coupling)
                    if model.invisible.enabled
                    and str(model.invisible.formulation) == "flow_matching"
                    else None
                ),
                "canonical_data_schema": (
                    "joint_event_reconstruction_canonical_v1"
                    if model.invisible.enabled and not synthetic_data
                    else None
                ),
                "synthetic_data": bool(synthetic_data),
                "trainable_components": sorted(trainable_components),
                "shared_encoder_gradients": str(experiment.training.shared_encoder_gradients),
                "partonic_shared_encoder_gradients": str(
                    experiment.training.partonic_shared_encoder_gradients
                ),
                "partonic_roles": list(self.event_model.partonic_roles),
                "partonic_target_representation": (
                    partonic_scaler.representation if partonic_scaler is not None else None
                ),
                "experiment_seed": int(experiment.experiment.seed),
                "data_subset_seed": int(experiment.data.subset_seed),
                "precision": str(experiment.training.precision),
                "external_package_versions": {
                    "torch": torch.__version__,
                    "lightning": importlib.metadata.version("lightning"),
                    "lgatr": lgatr_package_version,
                },
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
        if "point_loss" in output:
            self.log(
                f"{stage}_point_loss", output["point_loss"], on_step=False,
                on_epoch=True, batch_size=batch_size,
            )
        if "parent_loss" in output:
            self.log(
                f"{stage}_parent_loss", output["parent_loss"], on_step=False,
                on_epoch=True, batch_size=batch_size,
            )
        if "partonic_loss" in output:
            self.log(
                f"{stage}_partonic_loss", output["partonic_loss"], on_step=False,
                on_epoch=True, batch_size=batch_size,
            )
        return output["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("Experiment freezes every trainable parameter")
        optimiser = torch.optim.AdamW(
            trainable,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            fused=False,
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
        checkpoint["point_scaler"] = self.point_scaler_metadata
        checkpoint["partonic_scaler"] = self.partonic_scaler_metadata

    def validate_checkpoint_metadata(self, checkpoint: dict) -> None:
        saved = checkpoint.get("hyper_parameters", {})
        for key in (
            "topology_signature",
            "input_features",
            "source_type_ids",
            "physics_vector_representation",
        ):
            if saved.get(key) != self.hparams.get(key):
                raise ValueError(
                    f"Incompatible checkpoint metadata {key}: "
                    f"checkpoint={saved.get(key)!r}, current={self.hparams.get(key)!r}"
                )
        saved_policy = saved.get("shared_encoder_gradients")
        if saved_policy is None and self.event_model.point_head is None and self.event_model.parent_head is None:
            # Checkpoints written before the configurable joint-gradient policy
            # predate this metadata field.  Their maintained loss path updated
            # the shared encoder, which is the current ``both`` semantics.
            saved_policy = "both"
        if saved_policy != self.hparams.get("shared_encoder_gradients"):
            raise ValueError(
                "Incompatible checkpoint metadata shared_encoder_gradients: "
                f"checkpoint={saved_policy!r}, current={self.hparams.get('shared_encoder_gradients')!r}"
            )
        if saved.get("architecture_signature_version") != ARCHITECTURE_SIGNATURE_VERSION:
            raise ValueError(
                "Incompatible architecture signature version: "
                f"checkpoint={saved.get('architecture_signature_version')!r}, "
                f"current={ARCHITECTURE_SIGNATURE_VERSION!r}"
            )
        if saved.get("architecture_signature") != self.hparams.get("architecture_signature"):
            raise ValueError(
                "Incompatible active architecture signature: "
                f"checkpoint={saved.get('architecture_signature')!r}, "
                f"current={self.hparams.get('architecture_signature')!r}"
            )
        if saved.get("architecture_component_signatures") != self.hparams.get(
            "architecture_component_signatures"
        ):
            raise ValueError("Incompatible architecture component signatures")
        if checkpoint.get("invisible_scaler") != self.scaler_metadata:
            raise ValueError("Incompatible invisible target scaling metadata")
        if checkpoint.get("point_scaler") != self.point_scaler_metadata:
            raise ValueError("Incompatible direct point target scaling metadata")
        if checkpoint.get("partonic_scaler") != self.partonic_scaler_metadata:
            raise ValueError("Incompatible partonic target scaling metadata")
