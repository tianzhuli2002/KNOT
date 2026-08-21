# src/models/TransformerModel.py

import logging
import time
from collections import OrderedDict

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from joint_event_reconstruction.m0.data.module import (
    build_token_schema,
    infer_candidate_pools,
    parse_reconstruction_targets,
    token_feature_dim_from_dataset_cfg,
    validate_transformer_task_config,
)

logger = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _task_cfg(config, name):
    return _cfg_get(_cfg_get(config, "tasks", {}), name, {}) or {}


def _task_enabled(config, name, default=False):
    cfg = _task_cfg(config, name)
    return bool(_cfg_get(cfg, "enabled", default))


def _task_weight(config, name, default=1.0):
    cfg = _task_cfg(config, name)
    return float(_cfg_get(cfg, "weight", default))


class CandidateEncoder(nn.Module):
    """Source-aware candidate representation followed by one compact projection."""

    def __init__(self, d_model, source_signature, candidate_dim, dropout):
        super().__init__()
        self.source_signature = tuple(source_signature)
        self.same_source = len(self.source_signature) == 1
        raw_dim = 2 * d_model if self.same_source else sum(n for _, n in self.source_signature) * d_model
        self.projection = nn.Sequential(
            nn.Linear(raw_dim, candidate_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(candidate_dim, candidate_dim), nn.LayerNorm(candidate_dim),
        )

    def forward(self, encoded_tokens, candidate_indices):
        batch, candidates, size = candidate_indices.shape
        if candidates == 0:
            return encoded_tokens.new_empty((batch, 0, self.projection[-1].normalized_shape[0]))
        hidden = encoded_tokens.shape[-1]
        gathered = encoded_tokens.gather(
            1, candidate_indices.reshape(batch, candidates * size).unsqueeze(-1).expand(-1, -1, hidden)
        ).reshape(batch, candidates, size, hidden)
        if self.same_source:
            raw = torch.cat((gathered.mean(dim=2), gathered.max(dim=2).values), dim=-1)
        else:
            raw = gathered.reshape(batch, candidates, size * hidden)
        return self.projection(raw)


class DynamicCandidateAttention(nn.Module):
    """HyPER-style feature-wise competition over real candidates in each event."""

    def __init__(self, candidate_dim, dropout):
        super().__init__()
        self.weight = nn.Linear(candidate_dim, candidate_dim, bias=False)
        self.context = nn.Sequential(
            nn.Linear(2 * candidate_dim, candidate_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(candidate_dim, candidate_dim), nn.LayerNorm(candidate_dim),
        )

    @staticmethod
    def masked_feature_softmax(candidates, candidate_mask):
        if candidates.shape[1] == 0:
            return torch.zeros_like(candidates)
        valid = candidate_mask.unsqueeze(-1)
        masked = candidates.masked_fill(~valid, -torch.inf)
        maximum = masked.amax(dim=1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        numerator = torch.exp(masked - maximum) * valid.to(candidates.dtype)
        return numerator / numerator.sum(dim=1, keepdim=True).clamp_min(torch.finfo(candidates.dtype).tiny)

    def forward(self, candidates, candidate_mask):
        attention = self.masked_feature_softmax(candidates, candidate_mask)
        gated = attention * F.relu(self.weight(candidates))
        contextual = self.context(torch.cat((candidates, gated), dim=-1))
        contextual = contextual * candidate_mask.unsqueeze(-1).to(contextual.dtype)
        return contextual, attention


class TransformerClassifier(nn.Module):
    """Shared detector encoder with semantic reconstruction and event heads."""

    def __init__(
        self,
        n_features: int = 7,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooling: str = "cls",
        use_type_embedding: bool = True,
        classifier_hidden_dim: int = 128,
        n_object_types: int = 4,
        reconstruction_cfg=None,
        source_names=(),
        classification_enabled=True,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead}).")
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'.")

        self.pooling = pooling
        self.d_model = d_model
        self.input_projection = nn.Linear(n_features, d_model)
        self.type_embedding = nn.Embedding(n_object_types, d_model) if use_type_embedding else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model)) if pooling == "cls" else None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        reconstruction_cfg = reconstruction_cfg or {}
        self.reconstruction_enabled = bool(reconstruction_cfg.get("enabled", False))
        self.classification_enabled = bool(classification_enabled)
        self.candidate_dim = int(reconstruction_cfg.get("candidate_embedding_dim", d_model))
        self.candidate_pools = ()
        self.classifier_role_order = ()
        self.candidate_encoders = nn.ModuleDict()
        self.dynamic_candidate_attention = nn.ModuleDict()
        self.reconstruction_heads = nn.ModuleDict()
        if self.reconstruction_enabled:
            targets = parse_reconstruction_targets(reconstruction_cfg, source_names)
            self.candidate_pools = infer_candidate_pools(targets, source_names)
            self.classifier_role_order = tuple(target.name for target in targets)
            for pool in self.candidate_pools:
                self.candidate_encoders[pool.name] = CandidateEncoder(
                    d_model, pool.source_signature, self.candidate_dim, dropout
                )
                self.dynamic_candidate_attention[pool.name] = DynamicCandidateAttention(self.candidate_dim, dropout)
                self.reconstruction_heads[pool.name] = nn.Linear(self.candidate_dim, len(pool.class_names))
        classifier_input_dim = d_model
        if self.reconstruction_enabled:
            classifier_input_dim += len(self.classifier_role_order) * self.candidate_dim
        self.classifier = None
        if self.classification_enabled:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_dim, classifier_hidden_dim), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(classifier_hidden_dim, 1),
            )

        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def encode(self, tokens, type_ids, padding_mask):
        x = self.input_projection(tokens)

        if self.type_embedding is not None:
            x = x + self.type_embedding(type_ids)

        encoder_padding_mask = padding_mask
        if self.cls_token is not None:
            batch_size = x.size(0)
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=padding_mask.device)
            encoder_padding_mask = torch.cat([cls_mask, padding_mask], dim=1)

        encoded = self.encoder(x, src_key_padding_mask=encoder_padding_mask)
        encoded = self.output_norm(encoded)

        if self.pooling == "cls":
            pooled = encoded[:, 0]
            encoded_tokens = encoded[:, 1:]
        else:
            pooled = self._masked_mean(encoded, padding_mask)
            encoded_tokens = encoded

        return {
            "encoded_tokens": encoded_tokens,
            "padding_mask": padding_mask,
            "pooled": pooled,
        }

    def forward(self, tokens, type_ids, padding_mask, reco=None):
        """Return raw logits. Sigmoid belongs in metrics/inference."""

        encoded = self.encode(tokens, type_ids, padding_mask)
        classification_logits = encoded["pooled"].new_zeros(encoded["pooled"].shape[0])
        reconstruction_logits = {}
        candidate_embeddings = {}
        candidate_attention = {}
        role_summaries = OrderedDict()

        if self.reconstruction_enabled:
            if reco is None:
                raise ValueError("Reconstruction-enabled model requires candidate pools in every forward pass.")
            for pool in self.candidate_pools:
                pool_data = reco.get("pools", {}).get(pool.name)
                if pool_data is None:
                    continue
                base = self.candidate_encoders[pool.name](encoded["encoded_tokens"], pool_data["candidate_indices"])
                contextual, attention = self.dynamic_candidate_attention[pool.name](base, pool_data["candidate_mask"])
                logits = self.reconstruction_heads[pool.name](contextual)
                reconstruction_logits[pool.name] = logits
                candidate_embeddings[pool.name] = contextual
                candidate_attention[pool.name] = attention
                mask = pool_data["candidate_mask"]
                for class_index, role in enumerate(pool.class_names[1:], start=1):
                    if logits.shape[1] == 0:
                        summary = contextual.new_zeros((contextual.shape[0], self.candidate_dim))
                    else:
                        role_logits = logits[:, :, class_index].masked_fill(~mask, -torch.inf)
                        maximum = role_logits.amax(dim=1, keepdim=True)
                        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
                        weights = torch.exp(role_logits - maximum) * mask.to(logits.dtype)
                        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)
                        summary = (weights.unsqueeze(-1) * contextual).sum(dim=1)
                    role_summaries[role] = summary

        if self.classifier is not None:
            classifier_parts = [encoded["pooled"]]
            classifier_parts.extend(role_summaries[role] for role in self.classifier_role_order)
            classification_logits = self.classifier(torch.cat(classifier_parts, dim=-1)).squeeze(-1)

        return {
            "classification_logits": classification_logits,
            "reconstruction_logits": reconstruction_logits,
            "candidate_embeddings": candidate_embeddings,
            "candidate_attention": candidate_attention,
            "role_summaries": role_summaries,
            "encoded_tokens": encoded["encoded_tokens"],
            "padding_mask": encoded["padding_mask"],
            "pooled": encoded["pooled"],
        }

    @staticmethod
    def _masked_mean(encoded, padding_mask):
        valid = (~padding_mask).unsqueeze(-1).to(encoded.dtype)
        summed = (encoded * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp(min=1.0)
        return summed / counts


class TransformerModel(L.LightningModule):
    """Lightning wrapper matching the existing MLPModel training interface."""

    def __init__(self, config):
        logger.info("Initializing TransformerModel.")
        super().__init__()
        validate_transformer_task_config(config)
        token_schema = build_token_schema(config.dataset)
        reconstruction_cfg = _task_cfg(config, "reconstruction")
        reconstruction_enabled = _task_enabled(config, "reconstruction", default=False)
        classification_enabled = _task_enabled(config, "classification", default=True)
        saved_config = OmegaConf.to_container(config, resolve=True)
        saved_config["source_type_ids"] = dict(token_schema.source_type_ids)
        if reconstruction_enabled:
            targets = parse_reconstruction_targets(reconstruction_cfg, token_schema.sources)
            pools = infer_candidate_pools(targets, token_schema.sources)
            saved_config["reconstruction_architecture"] = {
                "targets": [(target.name, target.members) for target in targets],
                "candidate_pools": [(pool.source_signature, pool.class_names) for pool in pools],
                "candidate_embedding_dim": int(reconstruction_cfg.get("candidate_embedding_dim", config.model.d_model)),
                "dynamic_attention": "feature_softmax_relu_residual_v1",
                "classifier_role_order": [target.name for target in targets],
            }
        self.save_hyperparameters(saved_config)

        self.model = TransformerClassifier(
            n_features=token_schema.dim,
            d_model=config.model.d_model,
            nhead=config.model.nhead,
            num_layers=config.model.num_layers,
            dim_feedforward=config.model.dim_feedforward,
            dropout=config.model.dropout,
            pooling=config.model.pooling,
            use_type_embedding=config.model.use_type_embedding,
            classifier_hidden_dim=config.model.get("classifier_hidden_dim", 128),
            n_object_types=len(token_schema.source_type_ids),
            reconstruction_cfg=reconstruction_cfg,
            source_names=tuple(name for name in token_schema.sources if name != "GLOBAL"),
            classification_enabled=classification_enabled,
        )

        logger.info(
            "Transformer input dims: token_features=%d, source_type_ids=%s, use_type_embedding=%s.",
            token_schema.dim,
            dict(token_schema.source_type_ids),
            bool(config.model.use_type_embedding),
        )

        self.classification_criterion = nn.BCEWithLogitsLoss()
        self.classification_enabled = classification_enabled
        self.reconstruction_enabled = reconstruction_enabled
        self.classification_weight = _task_weight(config, "classification", default=1.0)
        self.reconstruction_weight = _task_weight(config, "reconstruction", default=1.0 if self.reconstruction_enabled else 0.0)
        self.reconstruction_cfg = reconstruction_cfg
        self.config = config.train
        self.log_every_n_steps = int(config.get("trainer", {}).get("log_every_n_steps", 20))
        self.compute_val_auc = bool(config.train.get("compute_val_auc", False))
        self.monitor_metric = str(config.train.checkpoint.monitor)
        self._logged_first_val_batch = False
        self._train_step_times = []
        self._validation_scores = []
        self._validation_logits = []
        self._validation_labels = []

    @staticmethod
    def _task_signature(config):
        tasks = _cfg_get(config, "tasks", {}) or {}
        classification = _cfg_get(tasks, "classification", {}) or {}
        reconstruction = _cfg_get(tasks, "reconstruction", {}) or {}
        if OmegaConf.is_config(reconstruction):
            reconstruction = OmegaConf.to_container(reconstruction, resolve=True)
        else:
            reconstruction = dict(reconstruction)
        reconstruction_enabled = bool(reconstruction.get("enabled", False))
        return {
            "classification_enabled": bool(_cfg_get(classification, "enabled", True)),
            "reconstruction": reconstruction if reconstruction_enabled else {"enabled": False},
        }

    def validate_checkpoint_configuration(self, checkpoint):
        hyper_parameters = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
        checkpoint_tasks = _cfg_get(hyper_parameters, "tasks", None)
        checkpoint_type_ids = _cfg_get(hyper_parameters, "source_type_ids", None)
        current_type_ids = dict(build_token_schema(self.hparams.dataset).source_type_ids)
        if checkpoint_type_ids is None:
            raise ValueError(
                "Transformer checkpoint has no saved source_type_ids and predates unique source embeddings; "
                "retrain with the current token schema."
            )
        if dict(checkpoint_type_ids) != current_type_ids:
            raise ValueError(
                "Transformer checkpoint source type IDs do not match the current token schema: "
                f"checkpoint={dict(checkpoint_type_ids)}, current={current_type_ids}."
            )
        if checkpoint_tasks is None:
            if self.reconstruction_enabled:
                raise ValueError(
                    "Reconstruction checkpoint has no saved task definitions; cannot verify target semantics."
                )
            return
        if self.reconstruction_enabled and _cfg_get(hyper_parameters, "reconstruction_architecture", None) is None:
            raise ValueError(
                "Checkpoint has no semantic candidate-pool reconstruction architecture metadata and is an "
                "incompatible old binary BCE reconstruction checkpoint."
            )
        checkpoint_signature = self._task_signature(hyper_parameters)
        current_signature = self._task_signature(self.hparams)
        if checkpoint_signature != current_signature:
            raise ValueError(
                "Transformer checkpoint task definitions do not match the current configuration. "
                f"checkpoint={checkpoint_signature}, current={current_signature}."
            )
        if self.reconstruction_enabled:
            checkpoint_architecture = _cfg_get(hyper_parameters, "reconstruction_architecture", None)
            current_architecture = _cfg_get(self.hparams, "reconstruction_architecture", None)
            if checkpoint_architecture != current_architecture:
                raise ValueError(
                    "Transformer checkpoint candidate-pool/class ordering does not match the current architecture: "
                    f"checkpoint={checkpoint_architecture}, current={current_architecture}."
                )

    def forward(self, tokens, type_ids, padding_mask, labels=None, reco=None, compute_reconstruction_ranking=True):
        output = self.model(tokens, type_ids, padding_mask, reco=reco)
        if labels is None:
            return output
        return self._attach_losses(
            output, labels.float(), reco,
            compute_reconstruction_ranking=compute_reconstruction_ranking,
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        def move(value):
            if torch.is_tensor(value):
                return value.to(device=device, non_blocking=True)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, tuple):
                return tuple(move(item) for item in value)
            return value
        return move(batch)

    @staticmethod
    def _zero_like(logits):
        return logits.sum() * 0.0

    def _attach_losses(self, output, labels, reco, *, compute_reconstruction_ranking=True):
        logits = output["classification_logits"]
        zero = self._zero_like(logits)

        if self.classification_enabled:
            classification_loss = self.classification_criterion(logits, labels)
        else:
            classification_loss = zero

        reconstruction_loss, reconstruction_losses, reconstruction_metrics = self._reconstruction_loss(
            output.get("reconstruction_logits", {}),
            reco,
            zero,
            compute_ranking=compute_reconstruction_ranking,
        )
        total_loss = (
            self.classification_weight * classification_loss
            + self.reconstruction_weight * reconstruction_loss
        )
        output.update(
            {
                "loss": total_loss,
                "classification_loss": classification_loss,
                "reconstruction_loss": reconstruction_loss,
                "reconstruction_losses": reconstruction_losses,
                "reconstruction_metrics": reconstruction_metrics,
            }
        )
        return output

    @staticmethod
    def semantic_pool_cross_entropy(logits, truth_class, candidate_mask, event_active, class_weights=None):
        """HyPER-like event/class balanced semantic CE for one candidate pool."""
        if logits.ndim != 3 or truth_class.shape != logits.shape[:2] or candidate_mask.shape != logits.shape[:2]:
            raise ValueError("Semantic reconstruction tensors have incompatible shapes.")
        n_classes = logits.shape[-1]
        if class_weights is None:
            weights = logits.new_ones(n_classes)
        else:
            weights = torch.as_tensor(class_weights, device=logits.device, dtype=logits.dtype)
            if weights.shape != (n_classes,) or not torch.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Semantic class weights must be finite, positive, and match the pool classes.")
        valid_truth = truth_class[candidate_mask]
        if valid_truth.numel() and (valid_truth.min() < 0 or valid_truth.max() >= n_classes):
            raise ValueError("Semantic truth class is outside the pool class range.")
        raw = F.cross_entropy(
            logits.reshape(-1, n_classes), truth_class.reshape(-1), reduction="none"
        ).reshape_as(truth_class)
        membership = F.one_hot(truth_class.clamp(min=0), num_classes=n_classes).to(logits.dtype)
        membership = membership * candidate_mask.unsqueeze(-1).to(logits.dtype)
        counts = membership.sum(dim=1)
        class_means = (raw.unsqueeze(-1) * membership).sum(dim=1) / counts.clamp_min(1.0)
        present_weights = (counts > 0).to(logits.dtype) * weights.unsqueeze(0)
        per_event = (class_means * present_weights).sum(dim=1) / present_weights.sum(dim=1).clamp_min(1.0)
        active_weights = event_active.to(logits.dtype)
        return (per_event * active_weights).sum() / active_weights.sum().clamp_min(1.0), raw

    @staticmethod
    def combine_candidate_size_losses(losses_by_size, zero):
        active_groups = [torch.stack(losses).mean() for _, losses in sorted(losses_by_size.items()) if losses]
        return torch.stack(active_groups).mean() if active_groups else zero

    def _reconstruction_loss(self, reconstruction_logits, reco, zero, *, compute_ranking=True):
        losses = {}
        metrics = {}
        if not self.reconstruction_enabled:
            return zero, losses, metrics
        if not reco:
            raise ValueError("Reconstruction is enabled, but the batch has no reconstruction targets.")

        expected_pools = list(self.model.reconstruction_heads.keys())
        batch_pools = reco.get("pools", {})
        missing_batch_pools = [name for name in expected_pools if name not in batch_pools]
        missing_logits = [name for name in expected_pools if name not in reconstruction_logits]
        if missing_batch_pools or missing_logits:
            raise ValueError(
                "Reconstruction candidate-pool mismatch: "
                f"missing batch pools={missing_batch_pools}, missing logits={missing_logits}."
            )
        reconstruction_zero = sum((value.sum() * 0.0 for value in reconstruction_logits.values()), zero)

        losses_by_size = {2: [], 3: []}
        for name, logits in reconstruction_logits.items():
            pool = batch_pools.get(name)
            if pool is None:
                continue
            truth = pool["truth_class"].to(device=logits.device, dtype=torch.long)
            mask = pool["candidate_mask"].to(device=logits.device, dtype=torch.bool)
            active = pool["event_active"].to(device=logits.device, dtype=torch.bool)
            pool_loss, _ = self.semantic_pool_cross_entropy(logits, truth, mask, active)
            losses[name] = pool_loss
            if active.any():
                losses_by_size.setdefault(int(pool["candidate_size"]), []).append(pool_loss)
            with torch.no_grad():
                probabilities = logits.softmax(dim=-1)
                metrics[name] = {
                    "candidate_count": mask.sum().detach(),
                    "active_event_count": active.sum().detach(),
                    "active_fraction": active.float().mean().detach(),
                }
                for class_index, role in enumerate(pool["class_names"][1:], start=1):
                    role_truth = (truth == class_index) & mask
                    role_active = active & role_truth.any(dim=1)
                    metrics[name][f"{role}_class_count"] = role_truth.sum().detach()
                    if compute_ranking and role_active.any():
                        scores = probabilities[:, :, class_index].masked_fill(~mask, -1.0)
                        order = scores.argsort(dim=1, descending=True, stable=True)
                        ranked_truth = role_truth.gather(1, order)
                        first_rank = ranked_truth.float().argmax(dim=1) + 1
                        metrics[name].update({
                            f"{role}_top1_accuracy": ranked_truth[:, 0][role_active].float().mean().detach(),
                            f"{role}_top3_accuracy": ranked_truth[:, :min(3, ranked_truth.shape[1])].any(dim=1)[role_active].float().mean().detach(),
                            f"{role}_mean_reciprocal_rank": (1.0 / first_rank[role_active].float()).mean().detach(),
                            f"{role}_random_top1_expectation": (role_truth.sum(dim=1)[role_active].float() / mask.sum(dim=1)[role_active].float()).mean().detach(),
                        })
        reconstruction_loss = self.combine_candidate_size_losses(losses_by_size, reconstruction_zero)
        for size, size_losses in losses_by_size.items():
            if size_losses:
                losses[f"size_{size}"] = torch.stack(size_losses).mean()
        return reconstruction_loss, losses, metrics

    def _shared_step(self, batch, stage):
        output = self(
            batch["tokens"],
            batch["type_ids"],
            batch["padding_mask"],
            labels=batch["labels"],
            reco=batch.get("reco", None),
            compute_reconstruction_ranking=(stage != "train"),
        )
        logits = output["classification_logits"]
        labels = batch["labels"].float()
        loss = output["loss"]

        if stage == "val" and not self._logged_first_val_batch:
            with torch.no_grad():
                scores = torch.sigmoid(logits)
                valid_tokens = (~batch["padding_mask"]).sum(dim=1).float()
                finite_fraction = torch.isfinite(batch["tokens"]).float().mean()
                unique_labels, label_counts = torch.unique(labels.detach().cpu().int(), return_counts=True)
                label_summary = {int(k): int(v) for k, v in zip(unique_labels, label_counts)}
            logger.info(
                "First %s batch: tokens shape=%s device=%s; labels shape=%s device=%s counts=%s; "
                "logits mean/std/min/max=%.6g/%.6g/%.6g/%.6g; "
                "scores mean/std/min/max=%.6g/%.6g/%.6g/%.6g; loss=%.6g; "
                "token finite fraction=%.6g; valid tokens mean/min/max=%.3f/%.0f/%.0f; model device=%s.",
                stage,
                tuple(batch["tokens"].shape),
                batch["tokens"].device,
                tuple(labels.shape),
                labels.device,
                label_summary,
                float(logits.mean().detach().cpu()),
                float(logits.std(unbiased=False).detach().cpu()),
                float(logits.min().detach().cpu()),
                float(logits.max().detach().cpu()),
                float(scores.mean().detach().cpu()),
                float(scores.std(unbiased=False).detach().cpu()),
                float(scores.min().detach().cpu()),
                float(scores.max().detach().cpu()),
                float(loss.detach().cpu()),
                float(finite_fraction.detach().cpu()),
                float(valid_tokens.mean().detach().cpu()),
                float(valid_tokens.min().detach().cpu()),
                float(valid_tokens.max().detach().cpu()),
                next(self.parameters()).device,
            )
            self._logged_first_val_batch = True

        self.log(
            f"{stage}_loss",
            loss,
            prog_bar=(stage != "train"),
            on_step=False,
            on_epoch=True,
            batch_size=labels.shape[0],
            sync_dist=False,
        )
        if self.classification_enabled:
            self.log(
                f"{stage}_classification_loss",
                output["classification_loss"],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=labels.shape[0],
                sync_dist=False,
            )
        if self.reconstruction_enabled:
            self.log(
                f"{stage}_reconstruction_loss",
                output["reconstruction_loss"],
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=labels.shape[0],
                sync_dist=False,
            )
        for name, target_loss in output.get("reconstruction_losses", {}).items():
            self.log(
                f"{stage}_reco_{name}_loss",
                target_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=labels.shape[0],
                sync_dist=False,
            )
            if name in {"size_2", "size_3"}:
                self.log(
                    f"{stage}_size{name[-1]}_loss",
                    target_loss,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=labels.shape[0],
                    sync_dist=False,
                )
        for name, target_metrics in output.get("reconstruction_metrics", {}).items():
            for metric_name, metric_value in target_metrics.items():
                if metric_name in {"candidate_count", "active_event_count"} or metric_name.endswith("_class_count"):
                    metric_batch_size = labels.shape[0]
                    reduce_fx = "sum"
                elif metric_name.endswith(("_top1_accuracy", "_top3_accuracy", "_mean_reciprocal_rank", "_random_top1_expectation")):
                    active_count = int(target_metrics["active_event_count"].detach().cpu())
                    if active_count == 0:
                        continue
                    metric_batch_size = active_count
                    reduce_fx = "mean"
                else:
                    metric_batch_size = labels.shape[0]
                    reduce_fx = "mean"
                self.log(
                    f"{stage}_reco_{name}_{metric_name}",
                    metric_value.float(),
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=metric_batch_size,
                    reduce_fx=reduce_fx,
                    sync_dist=False,
                )
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        start = time.perf_counter()
        loss, _, _ = self._shared_step(batch, "train")

        log_every = self.log_every_n_steps
        elapsed = time.perf_counter() - start
        self._train_step_times.append(elapsed)

        if log_every > 0 and (batch_idx + 1) % log_every == 0:
            recent_times = self._train_step_times[-log_every:]
            avg_time = sum(recent_times) / len(recent_times)
            logger.info("Average train step time over last %d steps: %.4f s.", len(recent_times), avg_time)
            self.log("train_step_time_s", avg_time, prog_bar=False, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        _, logits, labels = self._shared_step(batch, "val")
        if not self.classification_enabled or not self.compute_val_auc:
            return
        self._validation_scores.append(torch.sigmoid(logits).detach().cpu())
        self._validation_logits.append(logits.detach().cpu())
        self._validation_labels.append(labels.detach().cpu())

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_validation_epoch_start(self):
        if not self.classification_enabled or not self.compute_val_auc:
            return
        self._validation_scores = []
        self._validation_logits = []
        self._validation_labels = []

    def on_validation_epoch_end(self):
        role_top1 = [
            value for name, value in self.trainer.callback_metrics.items()
            if name.startswith("val_reco_") and name.endswith("_top1_accuracy")
        ]
        if role_top1:
            self.log(
                "val_reco_mean_role_top1",
                torch.stack([value.float() for value in role_top1]).mean(),
                prog_bar=False,
                on_epoch=True,
                sync_dist=False,
            )
        if not self.classification_enabled or not self.compute_val_auc:
            return
        if not self._validation_scores or not self._validation_labels:
            return

        scores = torch.cat(self._validation_scores).float().numpy().reshape(-1)
        logits = torch.cat(self._validation_logits).float().numpy().reshape(-1)
        labels = torch.cat(self._validation_labels).float().numpy().reshape(-1).astype(int)
        unique_labels = set(labels.tolist())

        if len(unique_labels) < 2:
            if self.monitor_metric == "val_auc":
                raise ValueError(
                    "Cannot monitor val_auc because the validation split contains only one class: "
                    f"{sorted(unique_labels)}."
                )
            logger.warning(
                "Skipping val_auc because validation labels contain only one class: %s",
                sorted(unique_labels),
            )
            return

        from sklearn.metrics import roc_auc_score

        val_auc = roc_auc_score(labels, scores)
        self.log("val_auc", float(val_auc), prog_bar=True, on_epoch=True, sync_dist=False)
        accuracy = float(((scores >= 0.5).astype(int) == labels).mean())
        self.log("val_accuracy_05", accuracy, prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("val_logits_mean", float(logits.mean()), prog_bar=False, on_epoch=True, sync_dist=False)
        self.log("val_logits_std", float(logits.std()), prog_bar=False, on_epoch=True, sync_dist=False)

        for label in sorted(unique_labels):
            mask = labels == label
            label_scores = scores[mask]
            self.log(f"val_score_mean_label{label}", float(label_scores.mean()), prog_bar=False, on_epoch=True, sync_dist=False)
            self.log(f"val_score_std_label{label}", float(label_scores.std()), prog_bar=False, on_epoch=True, sync_dist=False)
            self.log(f"val_label_count_{label}", int(mask.sum()), prog_bar=False, on_epoch=True, sync_dist=False)

        logger.info(
            "Validation AUC: %.6f; accuracy@0.5=%.6f; logits mean/std=%.6g/%.6g; "
            "score mean/std by label=%s.",
            val_auc,
            accuracy,
            float(logits.mean()),
            float(logits.std()),
            {
                int(label): {
                    "mean": float(scores[labels == label].mean()),
                    "std": float(scores[labels == label].std()),
                    "n": int((labels == label).sum()),
                }
                for label in sorted(unique_labels)
            },
        )

    def configure_optimizers(self):
        optimizer_name = str(self.config.get("optimizer", "adamw")).lower()
        lr = float(self.config.get("learning_rate", self.config.get("lr", 1.0e-4)))
        weight_decay = float(self.config.get("weight_decay", 0.0))

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported transformer optimizer: {optimizer_name}")

        scheduler_cfg = self.config.get("lr_scheduler", None)
        if scheduler_cfg is None or not bool(scheduler_cfg.get("enabled", False)):
            return optimizer

        scheduler_method = str(scheduler_cfg.get("method", "reduce_on_plateau")).lower()
        scheduler_method = scheduler_method.replace("torch.optim.lr_scheduler.", "")
        if scheduler_method not in {"reduce_on_plateau", "reducelronplateau"}:
            raise ValueError(f"Unsupported transformer LR scheduler: {scheduler_method}")

        monitor = str(scheduler_cfg.get("monitor"))
        mode = str(scheduler_cfg.get("mode"))
        validation_cadence = int(self.hparams.trainer.get("check_val_every_n_epoch", 1))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=float(scheduler_cfg.get("factor", 0.8)),
            patience=int(scheduler_cfg.get("patience", 8)),
            min_lr=float(scheduler_cfg.get("min_lr", 1.0e-6)),
            threshold=float(scheduler_cfg.get("threshold", 1.0e-4)),
        )
        logger.info(
            "Configured ReduceLROnPlateau: monitor=%s mode=%s factor=%s patience=%s min_lr=%s threshold=%s.",
            monitor,
            mode,
            float(scheduler_cfg.get("factor", 0.8)),
            int(scheduler_cfg.get("patience", 8)),
            float(scheduler_cfg.get("min_lr", 1.0e-6)),
            float(scheduler_cfg.get("threshold", 1.0e-4)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": monitor,
                "interval": "epoch",
                # Lightning invokes the plateau scheduler only on validation
                # epochs, so a cached metric can never be reused.
                "frequency": validation_cadence,
                "strict": True,
            },
        }
