import logging
import json
import math
from pathlib import Path

import lightning as L
import torch
from omegaconf import DictConfig
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only

from joint_event_reconstruction.m0.runtime import set_execution_device, set_seed

logger = logging.getLogger(__name__)


class PersistentEpochSummary(Callback):
    """Print one summary after each completed Transformer validation epoch."""

    _ORDERED_METRICS = (
        "train_loss",
        "val_loss",
        "val_reconstruction_loss",
        "val_classification_loss",
        "val_auc",
        "val_size2_loss",
        "val_size3_loss",
        "val_reco_mean_role_top1",
    )

    def __init__(self):
        super().__init__()
        self._last_epoch = None

    @staticmethod
    def _finite_float(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() != 1:
                return None
            value = value.detach().cpu().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @rank_zero_only
    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        epoch = int(trainer.current_epoch)
        if self._last_epoch == epoch:
            return
        self._last_epoch = epoch
        fields = [f"Epoch {epoch:03d}"]
        metrics = trainer.callback_metrics
        for name in self._ORDERED_METRICS:
            value = self._finite_float(metrics.get(name))
            if value is not None:
                fields.append(f"{name}={value:.6f}")
        lr = None
        if trainer.optimizers:
            rates = [group.get("lr") for optimizer in trainer.optimizers for group in optimizer.param_groups]
            finite_rates = [self._finite_float(rate) for rate in rates]
            finite_rates = [rate for rate in finite_rates if rate is not None]
            if finite_rates:
                lr = finite_rates[0]
        if lr is not None:
            fields.append(f"lr={lr:.6e}")
        print(" | ".join(fields), flush=True)


class BestCheckpointLogger(Callback):
    """Log every maintained Transformer checkpoint after validation."""

    def on_validation_end(self, trainer, pl_module):
        for checkpoint_callback in trainer.checkpoint_callbacks:
            best_score = checkpoint_callback.best_model_score
            best_path = checkpoint_callback.best_model_path
            monitor = checkpoint_callback.monitor
            if best_score is not None and best_path:
                logger.info(
                    "Best checkpoint so far: epoch=%d, %s=%.6f, path=%s",
                    trainer.current_epoch,
                    monitor,
                    float(best_score.detach().cpu()),
                    best_path,
                )


class MetricsSummaryWriter(Callback):
    """Write a compact Transformer per-run metrics summary."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.best_val_loss = None
        self.best_val_auc = None
        self.validated_epochs = []
        self.best_monitor_value = None
        self.best_monitor_epoch = None

    @staticmethod
    def _to_float(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return float(value.detach().cpu())
        return float(value)

    def _update_best_metrics(self, trainer):
        val_loss = self._to_float(trainer.callback_metrics.get("val_loss"))
        val_auc = self._to_float(trainer.callback_metrics.get("val_auc"))
        if val_loss is not None and (self.best_val_loss is None or val_loss < self.best_val_loss):
            self.best_val_loss = val_loss
        if val_auc is not None and (self.best_val_auc is None or val_auc > self.best_val_auc):
            self.best_val_auc = val_auc

    def on_validation_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            epoch = int(trainer.current_epoch) + 1
            self.validated_epochs.append(epoch)
            self._update_best_metrics(trainer)
            checkpoint_cfg = self.cfg.train.get("checkpoint", {}) or {}
            monitor = str(checkpoint_cfg.get("monitor", self.cfg.train.get("monitor_metric", "val_loss")))
            mode = str(checkpoint_cfg.get("mode", self.cfg.train.get("monitor_mode", "min")))
            current = self._to_float(trainer.callback_metrics.get(monitor))
            if current is not None:
                better = self.best_monitor_value is None or (
                    current < self.best_monitor_value if mode == "min" else current > self.best_monitor_value
                )
                if better:
                    self.best_monitor_value = current
                    self.best_monitor_epoch = epoch

    def on_fit_end(self, trainer, pl_module):
        try:
            logger_obj = trainer.logger
            log_dir_value = getattr(logger_obj, "log_dir", None) if logger_obj is not None else None
            if not log_dir_value:
                return
            log_dir = Path(log_dir_value)
            log_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_callback = trainer.checkpoint_callback
            checkpoint_summaries = {}
            for index, callback in enumerate(trainer.checkpoint_callbacks):
                key = "primary" if callback is checkpoint_callback else str(callback.monitor or f"checkpoint_{index}")
                checkpoint_summaries[key] = {
                    "monitor": callback.monitor,
                    "mode": callback.mode,
                    "best_model_score": self._to_float(callback.best_model_score),
                    "best_model_path": callback.best_model_path,
                    "last_model_path": callback.last_model_path,
                }
            if checkpoint_callback is not None and checkpoint_callback.best_model_score is not None:
                best_score = self._to_float(checkpoint_callback.best_model_score)
                if checkpoint_callback.monitor == "val_auc":
                    self.best_val_auc = best_score
                elif checkpoint_callback.monitor == "val_loss":
                    self.best_val_loss = best_score

            model_cfg = self.cfg.get("model", {})

            learning_rate = self._to_float(self.cfg.train.get("learning_rate", self.cfg.train.get("lr", None)))
            weight_decay = self._to_float(self.cfg.train.get("weight_decay", None))
            checkpoint_cfg = self.cfg.train.get("checkpoint", {}) or {}
            monitor_metric = str(checkpoint_cfg.get("monitor", self.cfg.train.get("monitor_metric", "val_loss")))
            monitor_mode = str(checkpoint_cfg.get("mode", self.cfg.train.get("monitor_mode", "min")))
            scheduler_cfg = self.cfg.train.get("lr_scheduler", {}) or {}

            common_summary = {
                "run_name": self.cfg.logging.train.experiment_name,
                "model": "transformer",
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "optimizer": str(self.cfg.train.get("optimizer", "")),
                "monitor_metric": monitor_metric,
                "monitor_mode": monitor_mode,
                "scheduler_monitor": str(scheduler_cfg.get("monitor", "")),
                "best_monitor_value": self.best_monitor_value,
                "best_monitor_epoch": self.best_monitor_epoch,
                "validated_epochs": self.validated_epochs,
                "best_val_loss": self.best_val_loss,
                "best_val_auc": self.best_val_auc,
                "best_checkpoint_path": checkpoint_callback.best_model_path if checkpoint_callback is not None else "",
                "last_checkpoint_path": checkpoint_callback.last_model_path if checkpoint_callback is not None else "",
                "checkpoints": checkpoint_summaries,
                "final_epoch": int(trainer.current_epoch),
                "final_metrics": {
                    str(name): self._to_float(value)
                    for name, value in trainer.callback_metrics.items()
                    if torch.is_tensor(value) or isinstance(value, (int, float))
                },
            }

            transformer_summary = {
                "d_model": model_cfg.get("d_model", None),
                "nhead": model_cfg.get("nhead", None),
                "num_layers": model_cfg.get("num_layers", None),
                "dim_feedforward": model_cfg.get("dim_feedforward", None),
                "dropout": model_cfg.get("dropout", None),
                "pooling": model_cfg.get("pooling", None),
                "classifier_hidden_dim": model_cfg.get("classifier_hidden_dim", None),
                "use_type_embedding": model_cfg.get("use_type_embedding", None),
            }

            summary = {
                **common_summary,
                **transformer_summary,
            }

            summary_path = log_dir / "metrics_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
            logger.info("Saved run metrics summary to %s", summary_path)

            run_manifest = {
                "log_dir": str(log_dir),
                "run_tag": str(self.cfg.get("outputs", {}).get("run_tag", "")),
                "experiment_name": str(self.cfg.logging.train.experiment_name),
                "input_h5_path": str(self.cfg.get("dataset", {}).get("input_h5_path", "")),
                "cache": self.cfg.get("dataset", {}).get("cache", {}),
                "best_checkpoint_path": summary.get("best_checkpoint_path", ""),
                "last_checkpoint_path": summary.get("last_checkpoint_path", ""),
                "checkpoints": summary.get("checkpoints", {}),
                "monitor_metric": summary.get("monitor_metric", ""),
                "monitor_mode": summary.get("monitor_mode", ""),
                "best_val_loss": summary.get("best_val_loss"),
                "best_val_auc": summary.get("best_val_auc"),
                "final_epoch": summary.get("final_epoch"),
                "seed": int(self.cfg.general.get("seed", -1)),
                "dataset_limit_events": self.cfg.dataset.get("limit_events", None),
                "split_test_size": self.cfg.dataset.get("test_size", None),
                "architecture": transformer_summary,
                "precision": self.cfg.trainer.get("precision", None),
                "output_directory": str(log_dir),
            }
            run_manifest["scheduler_monitor"] = summary.get("scheduler_monitor", "")
            run_manifest["best_monitor_value"] = summary.get("best_monitor_value")
            run_manifest["best_monitor_epoch"] = summary.get("best_monitor_epoch")
            run_manifest["validated_epochs"] = summary.get("validated_epochs", [])
            manifest_path = log_dir / "run_manifest.json"
            manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True, default=str))
            logger.info("Saved run manifest to %s", manifest_path)
        except Exception as exc:
            raise RuntimeError("Failed to save the Transformer metrics summary.") from exc


def _compile_model_if_requested(model, cfg):
    if not bool(cfg.train.get("compile", False)):
        return model

    if not hasattr(model, "model"):
        logger.warning("train.compile=true but model has no 'model' attribute; leaving %s uncompiled.", type(model).__name__)
        return model

    torch.set_float32_matmul_precision("high")
    try:
        model.model = torch.compile(model.model)
        logger.info("Compiled model.model with torch.compile.")
    except Exception:
        logger.exception("torch.compile failed for model.model; continuing without compile.")
    return model


def initialise_joint_from_reconstruction_checkpoint(model, checkpoint_path):
    """Load exactly shared and reconstruction parameters into a joint model."""
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Invalid reconstruction checkpoint state in {checkpoint_path}.")
    if not any(str(key).startswith("model.dynamic_candidate_attention.") for key in state):
        raise ValueError(
            f"Checkpoint {checkpoint_path} predates semantic candidate-pool reconstruction; "
            "old binary BCE reconstruction checkpoints are incompatible."
        )
    checkpoint_architecture = (payload.get("hyper_parameters", {}) or {}).get("reconstruction_architecture")
    current_architecture = getattr(model, "hparams", {}).get("reconstruction_architecture", None)
    if current_architecture is not None:
        if checkpoint_architecture is None:
            raise ValueError("Reconstruction checkpoint has no candidate-pool/class-order metadata.")
        if checkpoint_architecture != current_architecture:
            raise ValueError(
                "Reconstruction checkpoint candidate-pool/class ordering does not match the joint model: "
                f"checkpoint={checkpoint_architecture}, current={current_architecture}."
            )
    current = model.state_dict()
    allowed_missing = {key for key in current if key.startswith("model.classifier.")}
    unexpected = sorted(key for key in state if key not in current)
    missing = sorted(key for key in current if key not in state and key not in allowed_missing)
    shape_mismatch = sorted(
        key for key in state if key in current and tuple(state[key].shape) != tuple(current[key].shape)
    )
    if unexpected or missing or shape_mismatch:
        raise RuntimeError(
            "Reconstruction checkpoint is incompatible with the joint architecture: "
            f"missing_shared_or_reconstruction={missing}, unexpected={unexpected}, "
            f"shape_mismatch={shape_mismatch}."
        )
    loaded = {key: value for key, value in state.items() if key in current}
    result = model.load_state_dict(loaded, strict=False)
    if set(result.missing_keys) != allowed_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Narrow reconstruction initialisation failed: missing={sorted(result.missing_keys)}, "
            f"expected_missing={sorted(allowed_missing)}, unexpected={result.unexpected_keys}."
        )
    summary = {"loaded": sorted(loaded), "missing": sorted(allowed_missing), "unexpected": []}
    logger.info(
        "Initialised joint model from reconstruction checkpoint %s: loaded=%d missing_classification=%s unexpected=[]",
        checkpoint_path, len(loaded), summary["missing"],
    )
    return summary


def _trainer_kwargs(cfg, fallback_accelerator):
    trainer_cfg = cfg.get("trainer", None)
    if trainer_cfg is None:
        kwargs = {
            "accelerator": fallback_accelerator,
            "devices": "auto",
        }
    else:
        kwargs = {
            "accelerator": trainer_cfg.get("accelerator", fallback_accelerator),
            "devices": trainer_cfg.get("devices", "auto"),
            "precision": trainer_cfg.get("precision", "32-true"),
            "log_every_n_steps": trainer_cfg.get("log_every_n_steps", 50),
            "enable_progress_bar": trainer_cfg.get("enable_progress_bar", True),
        }
        for key in (
            "check_val_every_n_epoch", "limit_train_batches", "limit_val_batches",
            "limit_test_batches", "accumulate_grad_batches", "deterministic",
            "num_sanity_val_steps", "enable_model_summary",
        ):
            if key in trainer_cfg and trainer_cfg.get(key) is not None:
                kwargs[key] = trainer_cfg.get(key)

    train_cfg = cfg.get("train", {})
    gradient_clip_val = None
    if trainer_cfg is not None:
        gradient_clip_val = trainer_cfg.get("gradient_clip_val", train_cfg.get("gradient_clip", None))
    else:
        gradient_clip_val = train_cfg.get("gradient_clip", None)
    if gradient_clip_val is not None and float(gradient_clip_val) > 0:
        kwargs["gradient_clip_val"] = float(gradient_clip_val)
        kwargs["gradient_clip_algorithm"] = str(
            trainer_cfg.get("gradient_clip_algorithm", "norm") if trainer_cfg is not None else "norm"
        )

    return kwargs


def _monitor_settings(cfg):
    checkpoint_cfg = cfg.train.get("checkpoint", {}) or {}
    monitor = str(checkpoint_cfg.get("monitor", cfg.train.get("monitor_metric", "val_loss")))
    mode = str(checkpoint_cfg.get("mode", cfg.train.get("monitor_mode", "min"))).lower()
    if mode not in {"min", "max"}:
        raise ValueError(f"Checkpoint mode must be 'min' or 'max', found {mode!r}.")
    return monitor, mode


def _checkpoint_filename(checkpoint_cfg, monitor, *, tuning=False):
    if tuning:
        return "best"
    configured = checkpoint_cfg.get("filename", None)
    if configured:
        return str(configured)
    return "{epoch:03d}-{" + str(monitor) + ":.6f}"


def _make_checkpoint_callback(checkpoint_cfg, *, checkpoint_dir=None, tuning=False):
    monitor = str(checkpoint_cfg.get("monitor", ""))
    mode = str(checkpoint_cfg.get("mode", "min")).lower()
    if not monitor or mode not in {"min", "max"}:
        raise ValueError(
            "Every checkpoint specification requires a non-empty monitor and mode in {'min', 'max'}."
        )
    return ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=1 if tuning else int(checkpoint_cfg.get("save_top_k", 1)),
        save_last=False if tuning else bool(checkpoint_cfg.get("save_last", False)),
        verbose=True,
        dirpath=str(checkpoint_dir) if checkpoint_dir else None,
        filename=_checkpoint_filename(checkpoint_cfg, monitor, tuning=tuning),
        auto_insert_metric_name=False,
        save_on_train_epoch_end=False,
    )


def build_training_components(
    cfg: DictConfig,
    model,
    *,
    trial=None,
    output_directory=None,
    tuning=False,
):
    """Build the maintained Transformer Trainer, callbacks, and loggers."""
    if trial is not None or tuning:
        raise ValueError("M0 tuning machinery was deliberately removed; train an explicit configuration.")
    device = set_execution_device(cfg.general.device)
    model_name = str(cfg.get("model", {}).get("name", cfg.general.model)).lower()
    if model_name != "transformer":
        raise ValueError(f"TransformerTrain received unsupported model {model_name!r}.")
    checkpoint_cfg = cfg.train.get("checkpoint", {}) or {}
    monitor_metric, monitor_mode = _monitor_settings(cfg)
    checkpoint_dir = Path(output_directory).resolve() if output_directory else None
    checkpoint_callback = _make_checkpoint_callback(
        checkpoint_cfg, checkpoint_dir=checkpoint_dir, tuning=tuning
    )
    additional_checkpoint_callbacks = []
    if not tuning:
        additional_cfg = cfg.train.get("additional_checkpoints", {}) or {}
        for name, specification in additional_cfg.items():
            specification = dict(specification or {})
            specification.setdefault("filename", f"best_{name}")
            additional_checkpoint_callbacks.append(
                _make_checkpoint_callback(
                    specification, checkpoint_dir=checkpoint_dir, tuning=False
                )
            )
    early_cfg = cfg.train.get("early_stopping", {}) or {}
    early_stopping_patience = int(
        early_cfg.get("patience", cfg.train.get("early_stopping_patience", cfg.train.get("lr_patience", 30)))
    )
    early_monitor = str(early_cfg.get("monitor", monitor_metric))
    early_mode = str(early_cfg.get("mode", monitor_mode))
    logger.info("Early stopping patience: %d validation checks monitored on %s.", early_stopping_patience, early_monitor)

    callbacks = [
        checkpoint_callback,
        *additional_checkpoint_callbacks,
        BestCheckpointLogger(),
        MetricsSummaryWriter(cfg),
        PersistentEpochSummary(),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if bool(early_cfg.get("enabled", True)):
        callbacks.append(EarlyStopping(
            monitor=early_monitor,
            patience=early_stopping_patience,
            mode=early_mode,
            verbose=True,
            strict=True,
        ))

    initialise_path = cfg.train.get("initialise_from_reconstruction_checkpoint", None)
    if initialise_path:
        classification_enabled = bool((cfg.get("tasks", {}).get("classification", {}) or {}).get("enabled", True))
        reconstruction_enabled = bool((cfg.get("tasks", {}).get("reconstruction", {}) or {}).get("enabled", False))
        if not (classification_enabled and reconstruction_enabled):
            raise ValueError("train.initialise_from_reconstruction_checkpoint requires a joint model.")
        initialise_joint_from_reconstruction_checkpoint(model, initialise_path)
    model = _compile_model_if_requested(model, cfg)

    if output_directory:
        log_root = Path(output_directory).resolve() / "lightning_logs"
        experiment_name = "trial" if tuning else str(cfg.logging.train.experiment_name)
    else:
        log_root = Path(str(cfg.logging.train.output_dir))
        experiment_name = str(cfg.logging.train.experiment_name)
    train_loggers = [
        TensorBoardLogger(save_dir=str(log_root), name=experiment_name),
        CSVLogger(save_dir=str(log_root), name=experiment_name),
    ]

    trainer_kwargs = _trainer_kwargs(cfg, device)
    max_epochs = int(cfg.train.n_epochs)
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=train_loggers,
        **trainer_kwargs,
    )
    return {
        "model": model,
        "trainer": trainer,
        "checkpoint_callback": checkpoint_callback,
        "checkpoint_callbacks": [checkpoint_callback, *additional_checkpoint_callbacks],
        "callbacks": callbacks,
        "loggers": train_loggers,
        "monitor": monitor_metric,
        "mode": monitor_mode,
    }


def train(datamodule, model, cfg: DictConfig) -> None:
    """Train one configured Transformer model."""
    set_seed(cfg.general.seed)
    components = build_training_components(cfg, model)

    torch.set_float32_matmul_precision("high")
    components["trainer"].fit(model=components["model"], datamodule=datamodule)
