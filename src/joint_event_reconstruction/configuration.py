"""Load and strictly validate the single joint experiment configuration."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_experiment(path: str | Path) -> DictConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration does not exist: {path}")
    config = OmegaConf.load(path)
    validate_experiment(config)
    return config


def validate_experiment(config: DictConfig) -> None:
    required = {"experiment", "topology_config", "visible_config", "data", "model", "inference", "training"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Experiment configuration is missing sections {sorted(missing)}")
    reasoning = config.model.reasoning
    reconstruction = config.model.reconstruction
    invisible = config.model.invisible
    coupling = config.model.coupling
    compatibility = config.model.compatibility
    if reconstruction.feedback.enabled and not reasoning.enabled:
        raise ValueError("model.reconstruction.feedback.enabled=true requires model.reasoning.enabled=true")
    if invisible.formulation not in {"standard", "edm"}:
        raise ValueError(f"model.invisible.formulation must be standard or edm, found {invisible.formulation!r}")
    assignment = invisible.conditioning.assignment
    if assignment not in {"none", "soft", "hard", "truth"}:
        raise ValueError(f"Invisible assignment conditioning must be none, soft, hard or truth, found {assignment!r}")
    if assignment == "truth" and not invisible.conditioning.allow_truth_oracle:
        raise ValueError("Truth assignment conditioning is an oracle and requires allow_truth_oracle=true")
    feedback = coupling.invisible_to_reconstruction
    if feedback.enabled and not invisible.enabled:
        raise ValueError("Invisible-to-reconstruction feedback requires model.invisible.enabled=true")
    if feedback.enabled and not (reasoning.enabled and reconstruction.enabled):
        raise ValueError("Invisible-to-reconstruction feedback requires recurrent structured reconstruction")
    if feedback.enabled and not feedback.detach:
        raise ValueError("This implementation requires detached invisible posterior feedback")
    if int(coupling.rounds) not in {1, 2}:
        raise ValueError("model.coupling.rounds must be 1 or 2")
    if compatibility.enabled and not (reconstruction.enabled and invisible.enabled):
        raise ValueError("Compatibility scoring requires reconstruction and invisible hypotheses")
    if int(config.inference.top_k_assignments) <= 0:
        raise ValueError("inference.top_k_assignments must be positive")
    if int(config.inference.posterior_samples) <= 0:
        raise ValueError("inference.posterior_samples must be positive")
    steps = tuple(int(step) for step in reasoning.train_steps)
    if not steps or any(step <= 0 for step in steps) or int(reasoning.eval_steps) <= 0:
        raise ValueError("Recurrent train_steps and eval_steps must be positive")
    for split in ("train", "val", "test"):
        limit = config.data.max_events[split]
        if limit is not None and int(limit) <= 0:
            raise ValueError(f"data.max_events.{split} must be positive or null")
    train_limit = config.data.max_events.train
    if train_limit is not None and int(train_limit) > 500_000:
        raise ValueError("This development configuration forbids more than 500000 training events")
