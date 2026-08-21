import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


class ColoredHeaderFormatter(logging.Formatter):
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    NAME = "\033[36m"
    LEVEL_STYLES = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;37;41m",
    }

    def format(self, record: logging.LogRecord) -> str:
        colored_record = logging.makeLogRecord(record.__dict__.copy())
        level_style = self.LEVEL_STYLES.get(record.levelno, self.BOLD)
        colored_record.levelname = f"{level_style}[{record.levelname:^8}]{self.RESET}"
        colored_record.name = f"{self.NAME}{record.name}{self.RESET}"

        rendered = super().format(colored_record)
        if self.usesTime():
            timestamp = self.formatTime(record, self.datefmt)
            rendered = rendered.replace(timestamp, f"{self.DIM}{timestamp}{self.RESET}", 1)

        return rendered


def should_use_colors(color_mode: object, stream: object) -> bool:
    if os.getenv("NO_COLOR"):
        return False

    if isinstance(color_mode, bool):
        return color_mode

    mode = str(color_mode).lower()
    if mode in {"false", "off", "no", "0"}:
        return False
    if mode in {"true", "on", "yes", "1", "always"}:
        return True

    return hasattr(stream, "isatty") and stream.isatty()


def setup_logging(cfg) -> logging.Logger:
    log_cfg = getattr(cfg, "logging", None)
    level_name = str(getattr(log_cfg, "level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    log_format = str(
        getattr(
            log_cfg,
            "format",
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    )
    date_format = str(getattr(log_cfg, "datefmt", "%Y-%m-%d %H:%M:%S"))
    file_path = getattr(log_cfg, "file", None)
    capture_warnings = bool(getattr(log_cfg, "capture_warnings", True))
    color_mode = getattr(log_cfg, "colors", "auto")

    console_handler = logging.StreamHandler(stream=sys.stderr)
    formatter_cls = ColoredHeaderFormatter if should_use_colors(color_mode, console_handler.stream) else logging.Formatter
    console_handler.setFormatter(formatter_cls(log_format, datefmt=date_format))

    handlers: list[logging.Handler] = [console_handler]

    if file_path:
        resolved_file_path = resolve_runtime_path(file_path)
        ensure_parent_dir(resolved_file_path)
        file_handler = logging.FileHandler(resolved_file_path)
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(file_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    for handler in handlers:
        root_logger.addHandler(handler)

    logging.captureWarnings(capture_warnings)

    logger = logging.getLogger("RecoSBInteractions")
    logger.setLevel(level)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_execution_device(cfg_device: str) -> str:
    if cfg_device is None:
        return "cpu"

    if cfg_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    if cfg_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")

    return cfg_device


def resolve_runtime_path(path_str: str) -> str:
    if path_str is None:
        return None
    return str(Path(path_str).expanduser().resolve())


def ensure_parent_dir(path_str: str) -> None:
    parent = Path(path_str).parent
    parent.mkdir(parents=True, exist_ok=True)


def safe_path_name(text: object, default: str = "run") -> str:
    """Convert arbitrary text into a filesystem-safe path component."""
    value = str(text) if text is not None else default
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)
    cleaned = cleaned.strip("_")
    return cleaned or default


def get_hydra_config_name(default: str = "config") -> str:
    """Retained M0 output-name hook; new callers set outputs.config_name explicitly."""
    return default


_RUN_TAG_CACHE: dict[int, tuple[str, bool]] = {}


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _outputs_cfg(cfg):
    return _cfg_get(cfg, "outputs", {})


def get_run_tag(cfg) -> str:
    """Return the sanitized tag shared by prediction and performance outputs."""
    cache_key = id(cfg)
    if cache_key in _RUN_TAG_CACHE:
        return _RUN_TAG_CACHE[cache_key][0]

    outputs_cfg = _outputs_cfg(cfg)
    explicit_tag = _cfg_get(outputs_cfg, "run_tag", None)
    if explicit_tag:
        run_tag = safe_path_name(explicit_tag)
        _RUN_TAG_CACHE[cache_key] = (run_tag, True)
        return run_tag

    config_name = safe_path_name(get_hydra_config_name(default="config"))
    use_timestamp = bool(_cfg_get(outputs_cfg, "timestamp", True))
    if use_timestamp:
        run_tag = f"{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        run_tag = config_name

    _RUN_TAG_CACHE[cache_key] = (run_tag, False)
    return run_tag


def _has_explicit_run_tag(cfg) -> bool:
    get_run_tag(cfg)
    return _RUN_TAG_CACHE[id(cfg)][1]


def _unique_run_dir(base_dir: Path, run_tag: str, *, create: bool, allow_existing: bool) -> Path:
    path = base_dir / run_tag
    if not allow_existing:
        counter = 1
        while path.exists():
            path = base_dir / f"{run_tag}_{counter:02d}"
            counter += 1
    if create:
        path.mkdir(parents=True, exist_ok=allow_existing)
    return path


def resolve_prediction_output_dir(cfg, *, create: bool = True) -> Path:
    """Resolve results/predictions/<run_tag>/ from predict.output_csv parent."""
    predict_cfg = _cfg_get(cfg, "predict", {})
    output_csv = Path(str(_cfg_get(predict_cfg, "output_csv", "results/predictions/predictions.csv")))
    base_dir = Path(resolve_runtime_path(str(output_csv.parent)))
    return _unique_run_dir(
        base_dir,
        get_run_tag(cfg),
        create=create,
        allow_existing=_has_explicit_run_tag(cfg) or not create,
    )


def resolve_performance_output_dir(cfg, *, create: bool = True) -> Path:
    """Resolve <performance.output_dir>/<run_tag>/."""
    performance_cfg = _cfg_get(cfg, "performance", {})
    base_dir = Path(resolve_runtime_path(str(_cfg_get(performance_cfg, "output_dir", "results/performance/Transformer"))))
    return _unique_run_dir(
        base_dir,
        get_run_tag(cfg),
        create=create,
        allow_existing=_has_explicit_run_tag(cfg) or not create,
    )


def get_latest_checkpoint_path(checkpoint_dir: str) -> str:
    checkpoint_dir = resolve_runtime_path(checkpoint_dir)

    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_dir}")

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if len(checkpoints) != 1:
        raise ValueError(f"Expected exactly one .ckpt file in {checkpoint_dir}, found {len(checkpoints)}")

    return os.path.join(checkpoint_dir, checkpoints[0])


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if hasattr(model, "validate_checkpoint_configuration"):
        model.validate_checkpoint_configuration(checkpoint)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        checkpoint_has_reco = any("reconstruction_heads" in str(key) for key in state_dict)
        model_has_reco = any("reconstruction_heads" in str(key) for key in model.state_dict())
        hint = (
            " Checkpoint/config reconstruction mismatch is likely: "
            f"checkpoint_has_reconstruction_heads={checkpoint_has_reco}, "
            f"model_has_reconstruction_heads={model_has_reco}. "
            "Load reconstruction-enabled checkpoints with reconstruction-enabled configs, "
            "and classification-only checkpoints with classification-only configs."
        )
        raise RuntimeError(f"Failed to load checkpoint {ckpt_path!r}.{hint}") from exc
    return model

## This section defines the factories for models and datasets

def selected_model_name(cfg) -> str:
    """Return the configured model name from either supported config layout."""
    model_cfg = getattr(cfg, "model", None)
    if model_cfg is not None:
        model_name = model_cfg.get("name", None)
        if model_name is not None:
            return str(model_name).lower()
    return str(cfg.general.model).lower()


def _selected_model_name(cfg) -> str:
    """Backward-compatible private alias for existing utility callers."""
    return selected_model_name(cfg)

def dataset_factory(cfg):
    dataset_name = selected_model_name(cfg)
    if dataset_name == "transformer":
        from joint_event_reconstruction.m0.data.module import TransformerDataModule
        return TransformerDataModule(cfg)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

def model_factory(cfg, datamodule=None):
    model_name = selected_model_name(cfg)
    if model_name == "transformer":
        from joint_event_reconstruction.m0.model import TransformerModel
        return TransformerModel(cfg)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
