"""Load and strictly validate the single maintained experiment configuration."""

from __future__ import annotations

import math
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


FREEZE_KEYS = {"visible", "assignment", "invisible", "compatibility"}


def _positive(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive, found {value!r}")
    return result


def _nonnegative(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative, found {value!r}")
    return result


def _positive_int(value, name: str) -> int:
    result = int(value)
    if isinstance(value, bool) or result <= 0 or float(value) != result:
        raise ValueError(f"{name} must be a positive integer, found {value!r}")
    return result


def load_experiment(
    path: str | Path,
    overrides: list[str] | tuple[str, ...] | None = None,
) -> DictConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration does not exist: {path}")
    config = OmegaConf.load(path)
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides)))
    validate_experiment(config)
    return config


def validate_experiment(config: DictConfig) -> None:
    required = {
        "experiment", "topology_config", "visible_config", "data",
        "model", "inference", "training",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(
            f"Experiment configuration is missing sections {sorted(missing)}"
        )

    model = config.model
    mode = str(model.get("mode", "factorized"))
    if mode not in {"factorized", "joint_flow"}:
        raise ValueError(
            f"model.mode must be factorized or joint_flow, found {mode!r}"
        )

    visible = model.visible
    d_model = _positive_int(visible.d_model, "model.visible.d_model")
    n_heads = _positive_int(visible.n_heads, "model.visible.n_heads")
    if d_model % n_heads:
        raise ValueError(
            "model.visible.d_model must be divisible by model.visible.n_heads"
        )
    _positive_int(visible.layers, "model.visible.layers")
    _positive_int(visible.d_feedforward, "model.visible.d_feedforward")
    dropout = float(visible.dropout)
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("model.visible.dropout must lie in [0,1)")

    reasoning = model.reasoning
    reconstruction = model.reconstruction
    invisible = model.invisible
    coupling = model.coupling
    compatibility = model.compatibility
    reconstruction_formulation = str(
        reconstruction.get("formulation", "discriminative")
    )
    _positive_int(reasoning.block_layers, "model.reasoning.block_layers")

    if reconstruction_formulation not in {"discriminative", "flow_matching"}:
        raise ValueError(
            "model.reconstruction.formulation must be discriminative or flow_matching"
        )
    assignment_flow = reconstruction.get("flow_matching")
    if assignment_flow is None:
        raise ValueError(
            "model.reconstruction.flow_matching configuration is required"
        )
    _positive_int(
        assignment_flow.get("max_states", 0),
        "model.reconstruction.flow_matching.max_states",
    )
    if reconstruction.feedback.enabled and not reasoning.enabled:
        raise ValueError(
            "model.reconstruction.feedback.enabled=true requires "
            "model.reasoning.enabled=true"
        )
    if reconstruction.feedback.enabled:
        _positive_int(
            reconstruction.feedback.start_step,
            "model.reconstruction.feedback.start_step",
        )
        if not math.isfinite(float(reconstruction.feedback.gate_initial_bias)):
            raise ValueError(
                "model.reconstruction.feedback.gate_initial_bias must be finite"
            )

    if reconstruction_formulation == "flow_matching":
        if str(assignment_flow.source) != "mask":
            raise ValueError(
                "Discrete assignment Flow Matching currently requires source=mask"
            )
        if mode == "factorized":
            if bool(reasoning.enabled) or bool(reconstruction.feedback.enabled):
                raise ValueError(
                    "Factorized discrete Flow Matching has its own time-conditioned "
                    "topology reasoner; disable model.reasoning and reconstruction.feedback"
                )
            if bool(invisible.enabled):
                raise ValueError(
                    "Factorized discrete Flow Matching is currently assignment-only. "
                    "Use discriminative reconstruction for p(nu|X,A), or "
                    "model.mode=joint_flow for coupled discrete-continuous inference."
                )
            if (
                bool(coupling.invisible_to_reconstruction.enabled)
                or int(coupling.rounds) != 1
            ):
                raise ValueError(
                    "Factorized discrete Flow Matching cannot use C2 posterior feedback"
                )
            if bool(compatibility.enabled):
                raise ValueError(
                    "Compatibility reranking is maintained for discriminative "
                    "factorized reconstruction only"
                )

    if str(invisible.formulation) not in {"standard", "edm", "flow_matching"}:
        raise ValueError(
            "model.invisible.formulation must be standard, edm or flow_matching, "
            f"found {invisible.formulation!r}"
        )
    if bool(invisible.enabled):
        denoiser = invisible.denoiser
        invisible_width = _positive_int(
            denoiser.d_model, "model.invisible.denoiser.d_model"
        )
        invisible_heads = _positive_int(
            denoiser.n_heads, "model.invisible.denoiser.n_heads"
        )
        if invisible_width % invisible_heads:
            raise ValueError(
                "model.invisible.denoiser.d_model must be divisible by n_heads"
            )
        _positive_int(denoiser.layers, "model.invisible.denoiser.layers")
        _positive_int(
            denoiser.d_feedforward, "model.invisible.denoiser.d_feedforward"
        )
        invisible_dropout = float(denoiser.dropout)
        if not math.isfinite(invisible_dropout) or not 0 <= invisible_dropout < 1:
            raise ValueError("model.invisible.denoiser.dropout must lie in [0,1)")

        if str(invisible.formulation) == "standard":
            _positive_int(
                invisible.standard.training_steps,
                "model.invisible.standard.training_steps",
            )
            _positive_int(
                invisible.standard.inference_steps,
                "model.invisible.standard.inference_steps",
            )
        elif str(invisible.formulation) == "edm":
            edm = invisible.edm
            _positive(edm.sigma_data, "model.invisible.edm.sigma_data")
            _positive(edm.p_std, "model.invisible.edm.p_std")
            sigma_min = _positive(edm.sigma_min, "model.invisible.edm.sigma_min")
            sigma_max = _positive(edm.sigma_max, "model.invisible.edm.sigma_max")
            if sigma_min >= sigma_max:
                raise ValueError("EDM requires sigma_min < sigma_max")
            _positive(edm.rho, "model.invisible.edm.rho")
            _positive_int(edm.inference_nfe, "model.invisible.edm.inference_nfe")
            if not math.isfinite(float(edm.p_mean)):
                raise ValueError("model.invisible.edm.p_mean must be finite")
        else:
            flow = invisible.get("flow_matching")
            if flow is None:
                raise ValueError(
                    "Flow Matching invisible inference requires model.invisible.flow_matching"
                )
            if str(flow.path) != "linear":
                raise ValueError(
                    "Only linear continuous Flow Matching is currently implemented"
                )
            if str(flow.coupling) not in {"independent", "minibatch_ot"}:
                raise ValueError(
                    "Flow Matching coupling must be independent or minibatch_ot"
                )
            if str(flow.sampler) not in {"euler", "heun"}:
                raise ValueError(
                    "Standalone Flow Matching sampler must be euler or heun"
                )
            _positive_int(
                flow.inference_steps,
                "model.invisible.flow_matching.inference_steps",
            )

    assignment = str(invisible.conditioning.assignment)
    if assignment not in {"none", "soft", "hard", "hypothesis", "truth"}:
        raise ValueError(
            "Invisible assignment conditioning must be none, soft, hard, "
            f"hypothesis or truth, found {assignment!r}"
        )
    if assignment == "truth" and not invisible.conditioning.allow_truth_oracle:
        raise ValueError(
            "Truth assignment conditioning is an inference oracle and requires "
            "allow_truth_oracle=true"
        )
    if assignment == "hypothesis" and bool(
        invisible.conditioning.allow_truth_oracle
    ):
        raise ValueError(
            "hypothesis conditioning is not an inference oracle; set "
            "allow_truth_oracle=false"
        )

    feedback = coupling.invisible_to_reconstruction
    if feedback.enabled and not invisible.enabled:
        raise ValueError(
            "Invisible-to-reconstruction feedback requires model.invisible.enabled=true"
        )
    if feedback.enabled and not reasoning.enabled:
        raise ValueError(
            "Invisible-to-reconstruction feedback requires recurrent structured reconstruction"
        )
    if feedback.enabled and not feedback.detach:
        raise ValueError(
            "This implementation requires detached invisible posterior feedback"
        )
    if int(coupling.rounds) not in {1, 2}:
        raise ValueError("model.coupling.rounds must be 1 or 2")
    if feedback.enabled:
        _positive_int(
            feedback.posterior_samples,
            "model.coupling.invisible_to_reconstruction.posterior_samples",
        )
    if compatibility.enabled and not invisible.enabled:
        raise ValueError(
            "Compatibility scoring requires reconstruction and invisible hypotheses"
        )
    if not math.isfinite(float(compatibility.get("lambda", 1.0))):
        raise ValueError("model.compatibility.lambda must be finite")

    if mode == "joint_flow":
        if reconstruction_formulation != "flow_matching":
            raise ValueError(
                "joint_flow requires model.reconstruction.formulation=flow_matching"
            )
        if not invisible.enabled or str(invisible.formulation) != "flow_matching":
            raise ValueError(
                "joint_flow requires enabled continuous Flow Matching invisible inference"
            )
        if bool(reasoning.enabled):
            raise ValueError(
                "joint_flow uses model.joint_flow.recurrent_state; disable factorized "
                "model.reasoning"
            )
        if bool(reconstruction.feedback.enabled):
            raise ValueError(
                "joint_flow cannot use factorized soft reconstruction feedback"
            )
        if bool(feedback.enabled) or int(coupling.rounds) != 1:
            raise ValueError(
                "joint_flow cannot use factorized invisible-to-reconstruction coupling"
            )
        if bool(compatibility.enabled):
            raise ValueError(
                "joint_flow compatibility reranking is deliberately disabled"
            )
        if assignment != "none":
            raise ValueError(
                "joint_flow assignment-neutrino coupling is internal; set "
                "model.invisible.conditioning.assignment=none"
            )
        joint = model.get("joint_flow")
        if joint is None:
            raise ValueError("joint_flow mode requires model.joint_flow configuration")
        _positive_int(joint.block_layers, "model.joint_flow.block_layers")
        train_steps = tuple(int(step) for step in joint.train_steps)
        if not train_steps or any(step <= 0 for step in train_steps):
            raise ValueError(
                "model.joint_flow.train_steps must contain positive integers"
            )

    inference = config.inference
    _positive_int(inference.top_k_assignments, "inference.top_k_assignments")
    _positive_int(inference.posterior_samples, "inference.posterior_samples")
    assignment_inference = inference.get("assignment_flow")
    if assignment_inference is None:
        raise ValueError("inference.assignment_flow configuration is required")
    _positive_int(assignment_inference.steps, "inference.assignment_flow.steps")
    if mode == "joint_flow":
        joint_inference = inference.get("joint_flow")
        if joint_inference is None:
            raise ValueError(
                "joint_flow mode requires inference.joint_flow configuration"
            )
        _positive_int(joint_inference.steps, "inference.joint_flow.steps")
        _positive_int(
            joint_inference.get("sample_chunk_size", 1),
            "inference.joint_flow.sample_chunk_size",
        )
        if str(joint_inference.solver) != "euler":
            raise ValueError(
                "Joint discrete-continuous Flow Matching currently uses Euler integration"
            )

    steps = tuple(int(step) for step in reasoning.train_steps)
    if not steps or any(step <= 0 for step in steps):
        raise ValueError("model.reasoning.train_steps must contain positive integers")
    _positive_int(reasoning.eval_steps, "model.reasoning.eval_steps")

    for split in ("train", "val", "test"):
        limit = config.data.max_events[split]
        if limit is not None:
            _positive_int(limit, f"data.max_events.{split}")
    train_limit = config.data.max_events.train
    if train_limit is not None and int(train_limit) > 500_000:
        raise ValueError(
            "This development configuration forbids more than 500000 training events"
        )

    training = config.training
    _positive(training.learning_rate, "training.learning_rate")
    _nonnegative(training.weight_decay, "training.weight_decay")
    _positive_int(training.epochs, "training.epochs")
    _nonnegative(training.gradient_clip, "training.gradient_clip")
    _positive_int(
        training.accumulate_grad_batches, "training.accumulate_grad_batches"
    )
    for name in ("reconstruction", "invisible", "compatibility"):
        _nonnegative(
            training.loss_weights[name], f"training.loss_weights.{name}"
        )

    freeze = training.freeze
    freeze_keys = set(freeze)
    if freeze_keys != FREEZE_KEYS:
        missing = sorted(FREEZE_KEYS - freeze_keys)
        extra = sorted(freeze_keys - FREEZE_KEYS)
        raise ValueError(
            "training.freeze keys must be exactly "
            f"{sorted(FREEZE_KEYS)}; missing={missing}, obsolete_or_unknown={extra}"
        )
    if mode == "joint_flow" and (
        bool(freeze.assignment) or bool(freeze.invisible) or bool(freeze.compatibility)
    ):
        raise ValueError(
            "joint_flow has shared assignment/invisible dynamics; partial assignment, "
            "invisible or compatibility freezing is intentionally unsupported. "
            "Only training.freeze.visible may be used in joint_flow mode."
        )

    if bool(training.scheduler.enabled):
        factor = float(training.scheduler.factor)
        if not math.isfinite(factor) or not 0 < factor < 1:
            raise ValueError("training.scheduler.factor must lie in (0,1)")
        _nonnegative(training.scheduler.patience, "training.scheduler.patience")
        _nonnegative(training.scheduler.min_lr, "training.scheduler.min_lr")
    if bool(training.early_stopping.enabled):
        _positive_int(
            training.early_stopping.patience,
            "training.early_stopping.patience",
        )
