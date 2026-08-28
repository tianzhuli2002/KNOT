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

    mode = str(config.model.get("mode", "factorized"))
    if mode not in {"factorized", "joint_flow"}:
        raise ValueError(f"model.mode must be factorized or joint_flow, found {mode!r}")

    reasoning = config.model.reasoning
    reconstruction = config.model.reconstruction
    invisible = config.model.invisible
    coupling = config.model.coupling
    compatibility = config.model.compatibility
    reconstruction_formulation = str(reconstruction.get("formulation", "discriminative"))

    if reconstruction_formulation not in {"discriminative", "flow_matching"}:
        raise ValueError("model.reconstruction.formulation must be discriminative or flow_matching")
    if reconstruction.feedback.enabled and not reasoning.enabled:
        raise ValueError("model.reconstruction.feedback.enabled=true requires model.reasoning.enabled=true")
    if reconstruction_formulation == "flow_matching":
        flow = reconstruction.get("flow_matching")
        if flow is None or str(flow.source) != "mask":
            raise ValueError("Discrete assignment Flow Matching currently requires source=mask")
        if mode == "factorized":
            if bool(reasoning.enabled) or bool(reconstruction.feedback.enabled):
                raise ValueError(
                    "Factorized discrete Flow Matching has its own time-conditioned topology reasoner; "
                    "disable model.reasoning and reconstruction.feedback"
                )
            if bool(coupling.invisible_to_reconstruction.enabled) or int(coupling.rounds) != 1:
                raise ValueError("Factorized discrete Flow Matching cannot use C2 posterior feedback")
            if bool(compatibility.enabled):
                raise ValueError(
                    "Compatibility reranking is currently maintained for discriminative factorized reconstruction only"
                )

    if invisible.formulation not in {"standard", "edm", "flow_matching"}:
        raise ValueError(
            "model.invisible.formulation must be standard, edm or flow_matching, "
            f"found {invisible.formulation!r}"
        )
    if invisible.formulation == "flow_matching":
        flow = invisible.get("flow_matching")
        if flow is None:
            raise ValueError("Flow Matching invisible inference requires model.invisible.flow_matching")
        if str(flow.path) != "linear":
            raise ValueError("Only linear continuous Flow Matching is currently implemented")
        if str(flow.coupling) not in {"independent", "minibatch_ot"}:
            raise ValueError("Flow Matching coupling must be independent or minibatch_ot")
        if str(flow.sampler) not in {"euler", "heun"}:
            raise ValueError("Standalone Flow Matching sampler must be euler or heun")
        if int(flow.inference_steps) <= 0:
            raise ValueError("Flow Matching inference_steps must be positive")

    assignment = invisible.conditioning.assignment
    if assignment not in {"none", "soft", "hard", "truth"}:
        raise ValueError(
            f"Invisible assignment conditioning must be none, soft, hard or truth, found {assignment!r}"
        )
    if assignment == "truth" and not invisible.conditioning.allow_truth_oracle:
        raise ValueError("Truth assignment conditioning is an oracle and requires allow_truth_oracle=true")

    feedback = coupling.invisible_to_reconstruction
    if feedback.enabled and not invisible.enabled:
        raise ValueError("Invisible-to-reconstruction feedback requires model.invisible.enabled=true")
    if feedback.enabled and not (reasoning.enabled and reconstruction.get("enabled", True)):
        raise ValueError("Invisible-to-reconstruction feedback requires recurrent structured reconstruction")
    if feedback.enabled and not feedback.detach:
        raise ValueError("This implementation requires detached invisible posterior feedback")
    if int(coupling.rounds) not in {1, 2}:
        raise ValueError("model.coupling.rounds must be 1 or 2")
    if compatibility.enabled and not (reconstruction.get("enabled", True) and invisible.enabled):
        raise ValueError("Compatibility scoring requires reconstruction and invisible hypotheses")

    if mode == "joint_flow":
        if reconstruction_formulation != "flow_matching":
            raise ValueError("joint_flow requires model.reconstruction.formulation=flow_matching")
        if not invisible.enabled or str(invisible.formulation) != "flow_matching":
            raise ValueError("joint_flow requires enabled continuous Flow Matching invisible inference")
        if bool(reasoning.enabled):
            raise ValueError("joint_flow uses model.joint_flow.recurrent_state; disable factorized model.reasoning")
        if bool(reconstruction.feedback.enabled):
            raise ValueError("joint_flow cannot use factorized soft reconstruction feedback")
        if bool(feedback.enabled) or int(coupling.rounds) != 1:
            raise ValueError("joint_flow cannot use factorized invisible-to-reconstruction coupling")
        if bool(compatibility.enabled):
            raise ValueError("joint_flow compatibility reranking is deliberately disabled")
        if str(assignment) != "none":
            raise ValueError(
                "joint_flow assignment-neutrino coupling is internal; "
                "set model.invisible.conditioning.assignment=none"
            )
        joint = config.model.get("joint_flow")
        if joint is None:
            raise ValueError("joint_flow mode requires model.joint_flow configuration")
        if int(joint.block_layers) <= 0:
            raise ValueError("model.joint_flow.block_layers must be positive")
        train_steps = tuple(int(step) for step in joint.train_steps)
        if not train_steps or any(step <= 0 for step in train_steps):
            raise ValueError("model.joint_flow.train_steps must contain positive integers")

    if int(config.inference.top_k_assignments) <= 0:
        raise ValueError("inference.top_k_assignments must be positive")
    if int(config.inference.posterior_samples) <= 0:
        raise ValueError("inference.posterior_samples must be positive")
    if mode == "joint_flow":
        joint_inference = config.inference.get("joint_flow")
        if joint_inference is None:
            raise ValueError("joint_flow mode requires inference.joint_flow configuration")
        if int(joint_inference.steps) <= 0:
            raise ValueError("inference.joint_flow.steps must be positive")
        if str(joint_inference.solver) != "euler":
            raise ValueError("Joint discrete-continuous Flow Matching currently uses Euler integration")

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
