"""Load and strictly validate the maintained experiment configuration."""

from __future__ import annotations

import math
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


TOP_LEVEL_KEYS = {
    "experiment",
    "topology_config",
    "visible_config",
    "data",
    "model",
    "inference",
    "training",
}
MODEL_KEYS = {
    "mode",
    "visible",
    "assignment",
    "invisible",
    "joint_flow",
    "coupling",
    "compatibility",
}
FREEZE_KEYS = {"visible", "assignment", "invisible", "compatibility"}


def _exact_keys(section, expected: set[str], name: str) -> None:
    observed = set(section)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{name} keys must be exactly {sorted(expected)}; "
            f"missing={missing}, unknown_or_obsolete={extra}"
        )


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


def _finite(value, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, found {value!r}")
    return result


def _validate_network(section, name: str) -> None:
    _exact_keys(
        section,
        {"d_model", "n_heads", "layers", "d_feedforward", "dropout"},
        name,
    )
    width = _positive_int(section.d_model, f"{name}.d_model")
    heads = _positive_int(section.n_heads, f"{name}.n_heads")
    if width % heads:
        raise ValueError(f"{name}.d_model must be divisible by {name}.n_heads")
    _positive_int(section.layers, f"{name}.layers")
    _positive_int(section.d_feedforward, f"{name}.d_feedforward")
    dropout = float(section.dropout)
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError(f"{name}.dropout must lie in [0,1)")


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
    """Validate the public experiment contract.

    The YAML deliberately exposes only genuine scientific choices, inference budgets and
    safety limits. Fixed implementation details (linear CFM path, MASK-source DFM,
    detached C2 feedback and Euler joint integration) are not configurable.
    """
    _exact_keys(config, TOP_LEVEL_KEYS, "experiment config")
    _exact_keys(config.experiment, {"name", "seed"}, "experiment")
    if not str(config.experiment.name).strip():
        raise ValueError("experiment.name must be non-empty")
    int(config.experiment.seed)
    if not str(config.topology_config).strip() or not str(config.visible_config).strip():
        raise ValueError("topology_config and visible_config must be non-empty")

    data = config.data
    _exact_keys(
        data,
        {"tensor_cache", "max_events", "max_assignment_states", "invisible"},
        "data",
    )
    _exact_keys(data.max_events, {"train", "val", "test"}, "data.max_events")
    _exact_keys(data.invisible, {"path", "allow_synthetic"}, "data.invisible")
    _positive_int(data.max_assignment_states, "data.max_assignment_states")
    for split in ("train", "val", "test"):
        limit = data.max_events[split]
        if limit is not None:
            _positive_int(limit, f"data.max_events.{split}")
    if data.max_events.train is not None and int(data.max_events.train) > 500_000:
        raise ValueError(
            "This development configuration forbids more than 500000 training events"
        )

    model = config.model
    _exact_keys(model, MODEL_KEYS, "model")
    mode = str(model.mode)
    if mode not in {"factorized", "joint_flow"}:
        raise ValueError(f"model.mode must be factorized or joint_flow, found {mode!r}")
    _validate_network(model.visible, "model.visible")

    assignment = model.assignment
    _exact_keys(
        assignment,
        {"formulation", "reasoning", "feedback"},
        "model.assignment",
    )
    formulation = str(assignment.formulation)
    if formulation not in {"discriminative", "flow_matching"}:
        raise ValueError(
            "model.assignment.formulation must be discriminative or flow_matching"
        )

    reasoning = assignment.reasoning
    _exact_keys(
        reasoning,
        {
            "enabled",
            "block_layers",
            "train_steps",
            "eval_steps",
            "fixed_input_reinjection",
        },
        "model.assignment.reasoning",
    )
    _positive_int(reasoning.block_layers, "model.assignment.reasoning.block_layers")
    train_steps = tuple(int(step) for step in reasoning.train_steps)
    if not train_steps or any(step <= 0 for step in train_steps):
        raise ValueError(
            "model.assignment.reasoning.train_steps must contain positive integers"
        )
    _positive_int(reasoning.eval_steps, "model.assignment.reasoning.eval_steps")

    feedback = assignment.feedback
    _exact_keys(
        feedback,
        {"enabled", "start_step", "gate_initial_bias"},
        "model.assignment.feedback",
    )
    _positive_int(feedback.start_step, "model.assignment.feedback.start_step")
    _finite(feedback.gate_initial_bias, "model.assignment.feedback.gate_initial_bias")
    if bool(feedback.enabled) and not bool(reasoning.enabled):
        raise ValueError(
            "model.assignment.feedback.enabled=true requires "
            "model.assignment.reasoning.enabled=true"
        )

    invisible = model.invisible
    _exact_keys(
        invisible,
        {
            "enabled",
            "formulation",
            "conditioning",
            "denoiser",
            "standard",
            "edm",
            "flow_matching",
        },
        "model.invisible",
    )
    invisible_formulation = str(invisible.formulation)
    if invisible_formulation not in {"standard", "edm", "flow_matching"}:
        raise ValueError(
            "model.invisible.formulation must be standard, edm or flow_matching"
        )
    _exact_keys(
        invisible.conditioning,
        {"assignment", "allow_truth_oracle"},
        "model.invisible.conditioning",
    )
    conditioning = str(invisible.conditioning.assignment)
    if conditioning not in {"none", "soft", "hard", "hypothesis", "truth"}:
        raise ValueError(
            "model.invisible.conditioning.assignment must be none, soft, hard, "
            "hypothesis or truth"
        )
    if conditioning == "truth" and not bool(invisible.conditioning.allow_truth_oracle):
        raise ValueError(
            "truth assignment conditioning is an oracle and requires "
            "allow_truth_oracle=true"
        )
    if conditioning != "truth" and bool(invisible.conditioning.allow_truth_oracle):
        raise ValueError(
            "allow_truth_oracle=true is only meaningful with assignment=truth"
        )

    _exact_keys(
        invisible.denoiser,
        {
            "d_model",
            "n_heads",
            "layers",
            "d_feedforward",
            "dropout",
            "zero_init_output",
        },
        "model.invisible.denoiser",
    )
    if bool(invisible.enabled):
        denoiser = invisible.denoiser
        width = _positive_int(denoiser.d_model, "model.invisible.denoiser.d_model")
        heads = _positive_int(denoiser.n_heads, "model.invisible.denoiser.n_heads")
        if width % heads:
            raise ValueError(
                "model.invisible.denoiser.d_model must be divisible by n_heads"
            )
        _positive_int(denoiser.layers, "model.invisible.denoiser.layers")
        _positive_int(
            denoiser.d_feedforward, "model.invisible.denoiser.d_feedforward"
        )
        denoiser_dropout = float(denoiser.dropout)
        if not math.isfinite(denoiser_dropout) or not 0 <= denoiser_dropout < 1:
            raise ValueError("model.invisible.denoiser.dropout must lie in [0,1)")

    _exact_keys(invisible.standard, {"training_steps"}, "model.invisible.standard")
    _positive_int(
        invisible.standard.training_steps, "model.invisible.standard.training_steps"
    )
    _exact_keys(
        invisible.edm,
        {"sigma_data", "p_mean", "p_std"},
        "model.invisible.edm",
    )
    _positive(invisible.edm.sigma_data, "model.invisible.edm.sigma_data")
    _positive(invisible.edm.p_std, "model.invisible.edm.p_std")
    _finite(invisible.edm.p_mean, "model.invisible.edm.p_mean")
    _exact_keys(
        invisible.flow_matching,
        {"coupling"},
        "model.invisible.flow_matching",
    )
    if str(invisible.flow_matching.coupling) not in {"independent", "minibatch_ot"}:
        raise ValueError(
            "model.invisible.flow_matching.coupling must be independent or minibatch_ot"
        )

    joint = model.joint_flow
    _exact_keys(
        joint,
        {"recurrent_state", "block_layers", "train_steps"},
        "model.joint_flow",
    )
    _positive_int(joint.block_layers, "model.joint_flow.block_layers")
    joint_steps = tuple(int(step) for step in joint.train_steps)
    if not joint_steps or any(step <= 0 for step in joint_steps):
        raise ValueError("model.joint_flow.train_steps must contain positive integers")

    coupling = model.coupling
    _exact_keys(coupling, {"invisible_to_assignment"}, "model.coupling")
    c2 = coupling.invisible_to_assignment
    _exact_keys(
        c2,
        {"enabled", "posterior_samples", "gate_initial_bias"},
        "model.coupling.invisible_to_assignment",
    )
    _positive_int(c2.posterior_samples, "model.coupling.invisible_to_assignment.posterior_samples")
    _finite(c2.gate_initial_bias, "model.coupling.invisible_to_assignment.gate_initial_bias")

    compatibility = model.compatibility
    _exact_keys(
        compatibility,
        {"enabled", "include_invisible"},
        "model.compatibility",
    )

    if formulation == "flow_matching" and mode == "factorized":
        if bool(reasoning.enabled) or bool(feedback.enabled):
            raise ValueError(
                "factorized assignment Flow Matching has its own time-conditioned "
                "reasoner; disable model.assignment.reasoning and feedback"
            )
        if bool(invisible.enabled):
            raise ValueError(
                "factorized assignment Flow Matching is assignment-only; use "
                "discriminative assignment for p(nu|X,A), or joint_flow"
            )
        if bool(c2.enabled) or bool(compatibility.enabled):
            raise ValueError(
                "factorized assignment Flow Matching cannot use C2 or compatibility reranking"
            )

    if bool(c2.enabled):
        if mode != "factorized" or formulation != "discriminative":
            raise ValueError("C2 feedback is maintained for factorized discriminative assignment")
        if not bool(invisible.enabled) or not bool(reasoning.enabled):
            raise ValueError(
                "C2 invisible-to-assignment feedback requires invisible inference and "
                "recurrent assignment reasoning"
            )

    if bool(compatibility.enabled):
        if mode != "factorized" or formulation != "discriminative":
            raise ValueError(
                "compatibility reranking is maintained for factorized discriminative assignment"
            )
        if not bool(invisible.enabled):
            raise ValueError("compatibility reranking requires invisible hypotheses")

    if mode == "joint_flow":
        if formulation != "flow_matching":
            raise ValueError("joint_flow requires model.assignment.formulation=flow_matching")
        if not bool(invisible.enabled) or invisible_formulation != "flow_matching":
            raise ValueError("joint_flow requires enabled continuous Flow Matching")
        if bool(reasoning.enabled) or bool(feedback.enabled):
            raise ValueError(
                "joint_flow uses model.joint_flow.recurrent_state; disable factorized "
                "assignment reasoning and feedback"
            )
        if bool(c2.enabled) or bool(compatibility.enabled):
            raise ValueError("joint_flow cannot use factorized C2 or compatibility reranking")
        if conditioning != "none":
            raise ValueError(
                "joint_flow assignment-neutrino coupling is internal; set "
                "model.invisible.conditioning.assignment=none"
            )

    inference = config.inference
    _exact_keys(
        inference,
        {
            "top_k_assignments",
            "posterior_samples",
            "invisible",
            "assignment_flow",
            "joint_flow",
            "compatibility_weight",
        },
        "inference",
    )
    _positive_int(inference.top_k_assignments, "inference.top_k_assignments")
    _positive_int(inference.posterior_samples, "inference.posterior_samples")
    _nonnegative(inference.compatibility_weight, "inference.compatibility_weight")

    invisible_inference = inference.invisible
    _exact_keys(
        invisible_inference,
        {"standard", "edm", "flow_matching"},
        "inference.invisible",
    )
    _exact_keys(invisible_inference.standard, {"steps"}, "inference.invisible.standard")
    standard_steps = _positive_int(
        invisible_inference.standard.steps, "inference.invisible.standard.steps"
    )
    if standard_steps > int(invisible.standard.training_steps):
        raise ValueError(
            "inference.invisible.standard.steps cannot exceed model.invisible.standard.training_steps"
        )

    _exact_keys(
        invisible_inference.edm,
        {"sigma_min", "sigma_max", "rho", "nfe"},
        "inference.invisible.edm",
    )
    sigma_min = _positive(invisible_inference.edm.sigma_min, "inference.invisible.edm.sigma_min")
    sigma_max = _positive(invisible_inference.edm.sigma_max, "inference.invisible.edm.sigma_max")
    if sigma_min >= sigma_max:
        raise ValueError("EDM inference requires sigma_min < sigma_max")
    _positive(invisible_inference.edm.rho, "inference.invisible.edm.rho")
    edm_nfe = _positive_int(invisible_inference.edm.nfe, "inference.invisible.edm.nfe")
    if edm_nfe < 3 or edm_nfe % 2 == 0:
        raise ValueError("inference.invisible.edm.nfe must be an odd integer >= 3")

    _exact_keys(
        invisible_inference.flow_matching,
        {"steps", "solver"},
        "inference.invisible.flow_matching",
    )
    _positive_int(
        invisible_inference.flow_matching.steps,
        "inference.invisible.flow_matching.steps",
    )
    if str(invisible_inference.flow_matching.solver) not in {"euler", "heun"}:
        raise ValueError("inference.invisible.flow_matching.solver must be euler or heun")

    _exact_keys(inference.assignment_flow, {"steps"}, "inference.assignment_flow")
    _positive_int(inference.assignment_flow.steps, "inference.assignment_flow.steps")
    _exact_keys(
        inference.joint_flow,
        {"steps", "sample_chunk_size"},
        "inference.joint_flow",
    )
    _positive_int(inference.joint_flow.steps, "inference.joint_flow.steps")
    _positive_int(
        inference.joint_flow.sample_chunk_size,
        "inference.joint_flow.sample_chunk_size",
    )

    training = config.training
    _exact_keys(
        training,
        {
            "learning_rate",
            "weight_decay",
            "epochs",
            "precision",
            "gradient_clip",
            "accumulate_grad_batches",
            "loss_weights",
            "freeze",
            "scheduler",
            "early_stopping",
            "output",
        },
        "training",
    )
    _positive(training.learning_rate, "training.learning_rate")
    _nonnegative(training.weight_decay, "training.weight_decay")
    _positive_int(training.epochs, "training.epochs")
    _nonnegative(training.gradient_clip, "training.gradient_clip")
    _positive_int(
        training.accumulate_grad_batches, "training.accumulate_grad_batches"
    )
    if not str(training.precision).strip() or not str(training.output).strip():
        raise ValueError("training.precision and training.output must be non-empty")

    _exact_keys(
        training.loss_weights,
        {"assignment", "invisible", "compatibility"},
        "training.loss_weights",
    )
    for name in ("assignment", "invisible", "compatibility"):
        _nonnegative(training.loss_weights[name], f"training.loss_weights.{name}")

    _exact_keys(training.freeze, FREEZE_KEYS, "training.freeze")
    if mode == "joint_flow" and (
        bool(training.freeze.assignment)
        or bool(training.freeze.invisible)
        or bool(training.freeze.compatibility)
    ):
        raise ValueError(
            "joint_flow has shared assignment/invisible dynamics; only "
            "training.freeze.visible may be used"
        )

    _exact_keys(
        training.scheduler,
        {"enabled", "factor", "patience", "min_lr"},
        "training.scheduler",
    )
    if bool(training.scheduler.enabled):
        factor = float(training.scheduler.factor)
        if not math.isfinite(factor) or not 0 < factor < 1:
            raise ValueError("training.scheduler.factor must lie in (0,1)")
        _nonnegative(training.scheduler.patience, "training.scheduler.patience")
        _nonnegative(training.scheduler.min_lr, "training.scheduler.min_lr")

    _exact_keys(
        training.early_stopping,
        {"enabled", "patience"},
        "training.early_stopping",
    )
    if bool(training.early_stopping.enabled):
        _positive_int(
            training.early_stopping.patience,
            "training.early_stopping.patience",
        )
