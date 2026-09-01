"""Load compact JER experiment YAML and validate the resolved internal contract."""

from __future__ import annotations

import math
from pathlib import Path

from omegaconf import DictConfig

from joint_event_reconstruction.public_config import (
    build_visible_loader_config,
    load_public_experiment,
)


TOP_LEVEL_KEYS = {
    "experiment",
    "topology_config",
    "data_config",
    "data",
    "model",
    "inference",
    "training",
}
MODEL_KEYS = {
    "mode",
    "latent",
    "visible",
    "assignment",
    "invisible",
    "joint_flow",
    "coupling",
    "compatibility",
}
TRAINABLE_COMPONENTS = {"visible", "assignment", "invisible", "c2", "compatibility"}


def _exact_keys(section, expected: set[str], name: str) -> None:
    observed = set(section)
    if observed != expected:
        raise ValueError(
            f"{name} keys must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - observed)}, "
            f"unknown_or_obsolete={sorted(observed - expected)}"
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


def _network(section, name: str, *, zero_init: bool = False) -> None:
    keys = {"d_model", "n_heads", "layers", "d_feedforward", "dropout"}
    if zero_init:
        keys.add("zero_init_output")
    _exact_keys(section, keys, name)
    width = _positive_int(section.d_model, f"{name}.d_model")
    heads = _positive_int(section.n_heads, f"{name}.n_heads")
    if width % heads:
        raise ValueError(f"{name}.d_model must be divisible by n_heads")
    _positive_int(section.layers, f"{name}.layers")
    _positive_int(section.d_feedforward, f"{name}.d_feedforward")
    dropout = float(section.dropout)
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError(f"{name}.dropout must lie in [0,1)")


def _latent(section, name: str) -> None:
    _exact_keys(section, {"d_model", "n_heads", "d_feedforward", "dropout"}, name)
    width = _positive_int(section.d_model, f"{name}.d_model")
    heads = _positive_int(section.n_heads, f"{name}.n_heads")
    if width % heads:
        raise ValueError(f"{name}.d_model must be divisible by n_heads")
    _positive_int(section.d_feedforward, f"{name}.d_feedforward")
    dropout = float(section.dropout)
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError(f"{name}.dropout must lie in [0,1)")


def load_experiment(
    path: str | Path,
    overrides: list[str] | tuple[str, ...] | None = None,
) -> DictConfig:
    config = load_public_experiment(path, overrides)
    validate_experiment(config)
    return config


def load_visible_config(experiment: DictConfig) -> DictConfig:
    """Build the explicit fixed-token loader config from JER data+topology config."""
    return build_visible_loader_config(experiment)


def validate_experiment(config: DictConfig) -> None:
    """Validate the fully resolved internal experiment representation."""
    _exact_keys(config, TOP_LEVEL_KEYS, "resolved experiment")
    _exact_keys(config.experiment, {"name", "seed"}, "experiment")
    if not str(config.experiment.name).strip():
        raise ValueError("experiment.name must be non-empty")
    int(config.experiment.seed)
    if not str(config.topology_config).strip() or not str(config.data_config).strip():
        raise ValueError("resolved topology_config and data_config must be non-empty")

    data = config.data
    _exact_keys(
        data,
        {"tensor_cache", "subset_seed", "max_events", "max_assignment_states", "invisible"},
        "data",
    )
    int(data.subset_seed)
    _exact_keys(data.max_events, {"train", "val", "test"}, "data.max_events")
    _exact_keys(data.invisible, {"path", "allow_synthetic"}, "data.invisible")
    _positive_int(data.max_assignment_states, "data.max_assignment_states")
    for split in ("train", "val", "test"):
        limit = data.max_events[split]
        if limit is not None:
            _positive_int(limit, f"data.max_events.{split}")
    if data.max_events.train is not None and int(data.max_events.train) > 500_000:
        raise ValueError("Maintained studies cap training at 500000 events")

    model = config.model
    _exact_keys(model, MODEL_KEYS, "model")
    mode = str(model.mode)
    if mode not in {"factorized", "joint_flow"}:
        raise ValueError("model.mode must resolve to factorized or joint_flow")
    _latent(model.latent, "model.latent")
    visible_type = str(model.visible.type)
    visible_keys = {"type", "d_model", "n_heads", "layers", "d_feedforward", "dropout"}
    if visible_type == "physics_transformer":
        visible_keys |= {"momentum_scale_gev", "scalar_features", "pairwise"}
    elif visible_type == "lgatr":
        visible_keys = {
            "type", "d_model", "blocks", "heads", "hidden_mv_channels",
            "hidden_s_channels", "dropout", "momentum_scale_gev",
            "scalar_features", "spurions",
        }
    elif visible_type == "lgatr_slim":
        visible_keys = {
            "type", "d_model", "blocks", "heads", "hidden_vector_channels",
            "hidden_scalar_channels", "dropout", "momentum_scale_gev",
            "scalar_features", "spurions",
        }
    elif visible_type != "transformer":
        raise ValueError("resolved model.visible.type is invalid")
    _exact_keys(model.visible, visible_keys, "model.visible")
    width = _positive_int(model.visible.d_model, "model.visible.d_model")
    if visible_type in {"transformer", "physics_transformer"}:
        heads = _positive_int(model.visible.n_heads, "model.visible.n_heads")
        if width % heads:
            raise ValueError("model.visible.d_model must be divisible by n_heads")
        _positive_int(model.visible.layers, "model.visible.layers")
        _positive_int(model.visible.d_feedforward, "model.visible.d_feedforward")
    else:
        _positive_int(model.visible.blocks, "model.visible.blocks")
        _positive_int(model.visible.heads, "model.visible.heads")
        hidden_name = (
            "hidden_mv_channels" if visible_type == "lgatr" else "hidden_vector_channels"
        )
        scalar_name = (
            "hidden_s_channels" if visible_type == "lgatr" else "hidden_scalar_channels"
        )
        _positive_int(model.visible[hidden_name], f"model.visible.{hidden_name}")
        _positive_int(model.visible[scalar_name], f"model.visible.{scalar_name}")
    visible_dropout = float(model.visible.dropout)
    if not math.isfinite(visible_dropout) or not 0 <= visible_dropout < 1:
        raise ValueError("model.visible.dropout must lie in [0,1)")
    if int(model.visible.d_model) != int(model.latent.d_model):
        raise ValueError("visible encoder output width must equal model.latent.d_model")
    if visible_type == "physics_transformer":
        _positive(model.visible.momentum_scale_gev, "model.visible.momentum_scale_gev")
        if not set(model.visible.scalar_features) <= {"btag", "charge"}:
            raise ValueError("model.visible.scalar_features contains an invalid feature")
        _exact_keys(model.visible.pairwise, {"hidden_dim", "features"}, "model.visible.pairwise")
        _positive_int(model.visible.pairwise.hidden_dim, "model.visible.pairwise.hidden_dim")
    elif visible_type in {"lgatr", "lgatr_slim"}:
        _positive(model.visible.momentum_scale_gev, "model.visible.momentum_scale_gev")
        if not set(model.visible.scalar_features) <= {"btag", "charge"}:
            raise ValueError("model.visible.scalar_features contains an invalid feature")
        _exact_keys(model.visible.spurions, {"beam", "time"}, "model.visible.spurions")
        if str(model.visible.spurions.beam) not in {"xyplane", "none"}:
            raise ValueError("model.visible.spurions.beam is invalid")

    assignment = model.assignment
    _exact_keys(assignment, {"formulation", "reasoning", "feedback"}, "model.assignment")
    formulation = str(assignment.formulation)
    if formulation not in {"discriminative", "flow_matching"}:
        raise ValueError("resolved assignment formulation is invalid")
    reasoning = assignment.reasoning
    _exact_keys(
        reasoning,
        {"enabled", "block_layers", "train_steps", "eval_steps", "fixed_input_reinjection"},
        "model.assignment.reasoning",
    )
    _positive_int(reasoning.block_layers, "model.assignment.reasoning.block_layers")
    train_steps = tuple(int(value) for value in reasoning.train_steps)
    if not train_steps or any(value <= 0 for value in train_steps):
        raise ValueError("model.assignment.reasoning.train_steps must be positive")
    _positive_int(reasoning.eval_steps, "model.assignment.reasoning.eval_steps")
    feedback = assignment.feedback
    _exact_keys(feedback, {"enabled", "start_step", "gate_initial_bias"}, "model.assignment.feedback")
    _positive_int(feedback.start_step, "model.assignment.feedback.start_step")
    _finite(feedback.gate_initial_bias, "model.assignment.feedback.gate_initial_bias")
    if bool(feedback.enabled) and not bool(reasoning.enabled):
        raise ValueError("assignment feedback requires recurrent assignment")

    invisible = model.invisible
    _exact_keys(
        invisible,
        {"enabled", "formulation", "conditioning", "denoiser", "standard", "edm", "flow_matching"},
        "model.invisible",
    )
    if str(invisible.formulation) not in {"standard", "edm", "flow_matching"}:
        raise ValueError("resolved invisible formulation is invalid")
    _exact_keys(invisible.conditioning, {"assignment", "allow_truth_oracle"}, "model.invisible.conditioning")
    conditioning = str(invisible.conditioning.assignment)
    if conditioning not in {"none", "soft", "hard", "hypothesis", "truth"}:
        raise ValueError("resolved invisible conditioning is invalid")
    if (conditioning == "truth") != bool(invisible.conditioning.allow_truth_oracle):
        raise ValueError("truth conditioning and oracle permission must agree")
    _network(invisible.denoiser, "model.invisible.denoiser", zero_init=True)
    _exact_keys(invisible.standard, {"training_steps"}, "model.invisible.standard")
    _positive_int(invisible.standard.training_steps, "model.invisible.standard.training_steps")
    _exact_keys(invisible.edm, {"sigma_data", "p_mean", "p_std"}, "model.invisible.edm")
    _positive(invisible.edm.sigma_data, "model.invisible.edm.sigma_data")
    _finite(invisible.edm.p_mean, "model.invisible.edm.p_mean")
    _positive(invisible.edm.p_std, "model.invisible.edm.p_std")
    _exact_keys(invisible.flow_matching, {"coupling"}, "model.invisible.flow_matching")
    if str(invisible.flow_matching.coupling) not in {"independent", "minibatch_ot"}:
        raise ValueError("CFM coupling must be independent or minibatch_ot")

    joint = model.joint_flow
    _exact_keys(joint, {"recurrent_state", "block_layers", "train_steps"}, "model.joint_flow")
    _positive_int(joint.block_layers, "model.joint_flow.block_layers")
    if not joint.train_steps or any(int(value) <= 0 for value in joint.train_steps):
        raise ValueError("model.joint_flow.train_steps must be positive")

    c2 = model.coupling.invisible_to_assignment
    _exact_keys(model.coupling, {"invisible_to_assignment"}, "model.coupling")
    _exact_keys(
        c2,
        {"enabled", "posterior_samples", "gate_initial_bias", "training_sampler"},
        "model.coupling.invisible_to_assignment",
    )
    _positive_int(c2.posterior_samples, "model.coupling.invisible_to_assignment.posterior_samples")
    _finite(c2.gate_initial_bias, "model.coupling.invisible_to_assignment.gate_initial_bias")
    if c2.enabled:
        _exact_keys(
            c2.training_sampler,
            {"formulation", "budget", "solver"},
            "model.coupling.invisible_to_assignment.training_sampler",
        )
        if str(c2.training_sampler.formulation) != str(invisible.formulation):
            raise ValueError("C2 training sampler formulation must match invisible model")
        _positive_int(
            c2.training_sampler.budget,
            "model.coupling.invisible_to_assignment.training_sampler.budget",
        )
        if invisible.formulation == "flow_matching":
            if str(c2.training_sampler.solver) not in {"euler", "heun"}:
                raise ValueError("C2 flow-matching training solver must be euler or heun")
        elif invisible.formulation == "standard":
            if int(c2.training_sampler.budget) > int(invisible.standard.training_steps):
                raise ValueError(
                    "C2 DDIM training steps cannot exceed diffusion training steps"
                )
        elif int(c2.training_sampler.budget) < 3 or int(c2.training_sampler.budget) % 2 == 0:
            raise ValueError("C2 EDM nfe must be odd and >=3")
        if invisible.formulation != "flow_matching" and c2.training_sampler.solver is not None:
            raise ValueError("C2 training sampler solver applies only to Flow Matching")
    elif c2.training_sampler is not None:
        raise ValueError("Inactive C2 must not define a training sampler")
    _exact_keys(model.compatibility, {"enabled", "include_invisible"}, "model.compatibility")

    if mode == "factorized" and formulation == "flow_matching":
        if reasoning.enabled or feedback.enabled or invisible.enabled or c2.enabled or model.compatibility.enabled:
            raise ValueError("factorized DFM is assignment-only")
    if c2.enabled and (
        mode != "factorized"
        or formulation != "discriminative"
        or not reasoning.enabled
        or not invisible.enabled
    ):
        raise ValueError("C2 requires recurrent factorized discriminative assignment plus invisible inference")
    if model.compatibility.enabled and (
        mode != "factorized" or formulation != "discriminative" or not invisible.enabled
    ):
        raise ValueError("compatibility requires factorized discriminative assignment plus invisible inference")
    if mode == "joint_flow":
        if formulation != "flow_matching" or not invisible.enabled or invisible.formulation != "flow_matching":
            raise ValueError("joint_flow requires DFM assignment and CFM invisible inference")
        if reasoning.enabled or feedback.enabled or c2.enabled or model.compatibility.enabled or conditioning != "none":
            raise ValueError("joint_flow owns its coupling; factorized recurrence/C2/compatibility must be absent")

    inference = config.inference
    _exact_keys(
        inference,
        {"seed", "top_k_assignments", "posterior_samples", "invisible", "assignment_flow", "joint_flow", "compatibility_weight"},
        "inference",
    )
    int(inference.seed)
    _positive_int(inference.top_k_assignments, "inference.top_k_assignments")
    _positive_int(inference.posterior_samples, "inference.posterior_samples")
    _nonnegative(inference.compatibility_weight, "inference.compatibility_weight")
    _exact_keys(inference.invisible, {"standard", "edm", "flow_matching"}, "inference.invisible")
    _exact_keys(inference.invisible.standard, {"steps"}, "inference.invisible.standard")
    standard_steps = _positive_int(inference.invisible.standard.steps, "inference.invisible.standard.steps")
    if standard_steps > int(invisible.standard.training_steps):
        raise ValueError("DDIM prediction steps cannot exceed diffusion training steps")
    _exact_keys(inference.invisible.edm, {"sigma_min", "sigma_max", "rho", "nfe"}, "inference.invisible.edm")
    sigma_min = _positive(inference.invisible.edm.sigma_min, "inference.invisible.edm.sigma_min")
    sigma_max = _positive(inference.invisible.edm.sigma_max, "inference.invisible.edm.sigma_max")
    if sigma_min >= sigma_max:
        raise ValueError("EDM requires sigma_min < sigma_max")
    _positive(inference.invisible.edm.rho, "inference.invisible.edm.rho")
    nfe = _positive_int(inference.invisible.edm.nfe, "inference.invisible.edm.nfe")
    if nfe < 3 or nfe % 2 == 0:
        raise ValueError("EDM nfe must be odd and >=3")
    _exact_keys(inference.invisible.flow_matching, {"steps", "solver"}, "inference.invisible.flow_matching")
    _positive_int(inference.invisible.flow_matching.steps, "inference.invisible.flow_matching.steps")
    if str(inference.invisible.flow_matching.solver) not in {"euler", "heun"}:
        raise ValueError("CFM solver must be euler or heun")
    _exact_keys(inference.assignment_flow, {"steps"}, "inference.assignment_flow")
    _positive_int(inference.assignment_flow.steps, "inference.assignment_flow.steps")
    _exact_keys(inference.joint_flow, {"steps", "sample_chunk_size"}, "inference.joint_flow")
    _positive_int(inference.joint_flow.steps, "inference.joint_flow.steps")
    _positive_int(inference.joint_flow.sample_chunk_size, "inference.joint_flow.sample_chunk_size")

    training = config.training
    _exact_keys(
        training,
        {"learning_rate", "weight_decay", "epochs", "precision", "gradient_clip", "accumulate_grad_batches", "trainable", "loss_weights", "scheduler", "early_stopping", "runtime", "output"},
        "training",
    )
    _positive(training.learning_rate, "training.learning_rate")
    _nonnegative(training.weight_decay, "training.weight_decay")
    _positive_int(training.epochs, "training.epochs")
    _nonnegative(training.gradient_clip, "training.gradient_clip")
    _positive_int(training.accumulate_grad_batches, "training.accumulate_grad_batches")
    if not str(training.precision).strip() or not str(training.output).strip():
        raise ValueError("training.precision and output must be non-empty")
    _exact_keys(
        training.runtime,
        {"compile", "activation_checkpointing"},
        "training.runtime",
    )
    if not isinstance(training.runtime.compile, bool) or not isinstance(
        training.runtime.activation_checkpointing, bool
    ):
        raise ValueError("training.runtime settings must be boolean")
    trainable = [str(value) for value in training.trainable]
    if not trainable or len(set(trainable)) != len(trainable):
        raise ValueError("training.trainable must be non-empty and unique")
    unknown = sorted(set(trainable) - TRAINABLE_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown training.trainable components: {unknown}")
    if "invisible" in trainable and not invisible.enabled:
        raise ValueError("training.trainable includes invisible but no invisible model is active")
    if "c2" in trainable and not c2.enabled:
        raise ValueError("training.trainable includes c2 but C2 is inactive")
    if "compatibility" in trainable and not model.compatibility.enabled:
        raise ValueError("training.trainable includes compatibility but compatibility is inactive")
    if mode == "joint_flow" and (("assignment" in trainable) != ("invisible" in trainable)):
        raise ValueError("joint_flow assignment and invisible dynamics must be trained/frozen together")
    _exact_keys(training.loss_weights, {"assignment", "invisible", "compatibility"}, "training.loss_weights")
    for key in training.loss_weights:
        _nonnegative(training.loss_weights[key], f"training.loss_weights.{key}")
    _exact_keys(training.scheduler, {"enabled", "factor", "patience", "min_lr"}, "training.scheduler")
    if training.scheduler.enabled:
        factor = float(training.scheduler.factor)
        if not 0 < factor < 1:
            raise ValueError("training.scheduler.factor must lie in (0,1)")
        _nonnegative(training.scheduler.patience, "training.scheduler.patience")
        _nonnegative(training.scheduler.min_lr, "training.scheduler.min_lr")
    _exact_keys(training.early_stopping, {"enabled", "patience"}, "training.early_stopping")
    if training.early_stopping.enabled:
        _positive_int(training.early_stopping.patience, "training.early_stopping.patience")
