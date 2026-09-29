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
    "input_definition",
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
    "heads",
    "topology_queries",
    "parent_cfm",
}
TRAINABLE_COMPONENTS = {
    "visible", "assignment", "invisible", "point", "parent", "partonic",
    "queries", "c2", "compatibility", "partonic_cfm",
}


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


def _mass_shell(section, name: str) -> None:
    _exact_keys(section, {"enforcement", "mass", "roles"}, name)

    def validate_entry(entry, entry_name: str) -> None:
        _exact_keys(entry, {"enforcement", "mass", "roles"}, entry_name)
        enforcement = str(entry.enforcement)
        mass = entry.mass
        _exact_keys(mass, {"mode", "value", "parameterization"}, f"{entry_name}.mass")
        mode = str(mass.mode)
        if mode not in {"fixed", "generated", "unconstrained"}:
            raise ValueError(f"{entry_name}.mass.mode is invalid")
        if mode == "unconstrained":
            if enforcement not in {"unconstrained", "learned"}:
                raise ValueError(
                    f"{entry_name} unconstrained mode requires learned or unconstrained enforcement"
                )
            if mass.value is not None or mass.parameterization is not None:
                raise ValueError(f"{entry_name} unconstrained mode has mass settings")
        elif enforcement != "exact":
            raise ValueError(f"{entry_name} exact mass modes require exact enforcement")
        elif mode == "fixed":
            if mass.value is None or not math.isfinite(float(mass.value)) or float(mass.value) < 0:
                raise ValueError(f"{entry_name}.mass.value must be finite and non-negative")
            if mass.parameterization is not None:
                raise ValueError(f"{entry_name} fixed mass must not have parameterization")
        elif mass.value is not None or str(mass.parameterization) not in {"log_mass", "softplus"}:
            raise ValueError(
                f"{entry_name} generated mass requires log_mass or softplus and no fixed value"
            )
        roles = entry.roles
        if not isinstance(roles, (dict, DictConfig)):
            raise ValueError(f"{entry_name}.roles must be a mapping")
        for role_name, role_entry in roles.items():
            validate_entry(role_entry, f"{entry_name}.roles.{role_name}")

    validate_entry(section, name)


def _invisible_backbone(section, name: str) -> None:
    backend = str(section.type)
    if backend == "transformer":
        _exact_keys(section, {"type"}, name)
        return
    if backend != "lgatr":
        raise ValueError(f"{name}.type must be transformer or lgatr")
    _exact_keys(
        section,
        {
            "type",
            "layers",
            "n_heads",
            "hidden",
            "dropout",
            "momentum_scale_gev",
            "symmetry_breaking",
        },
        name,
    )
    _positive_int(section.layers, f"{name}.layers")
    _positive_int(section.n_heads, f"{name}.n_heads")
    _exact_keys(
        section.hidden,
        {"multivector_channels", "scalar_channels"},
        f"{name}.hidden",
    )
    _positive_int(section.hidden.multivector_channels, f"{name}.hidden.multivector_channels")
    _positive_int(section.hidden.scalar_channels, f"{name}.hidden.scalar_channels")
    dropout = float(section.dropout)
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError(f"{name}.dropout must lie in [0,1)")
    _positive(section.momentum_scale_gev, f"{name}.momentum_scale_gev")
    _exact_keys(section.symmetry_breaking, {"beam", "time"}, f"{name}.symmetry_breaking")
    if str(section.symmetry_breaking.beam) not in {"xyplane", "none"}:
        raise ValueError(f"{name}.symmetry_breaking.beam is invalid")


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
        {"input", "inference_input", "visible_truth", "tensor_cache", "force_rebuild_cache", "subset_seed", "test_size", "split", "truth_bindings", "max_events", "max_assignment_states", "loader", "invisible", "parent", "partonic"},
        "data",
    )
    if data.inference_input is not None and not Path(str(data.inference_input)).is_file():
        raise FileNotFoundError(
            f"Configured truth-free inference input does not exist: {data.inference_input}"
        )
    if data.visible_truth is not None and not Path(str(data.visible_truth)).is_file():
        raise FileNotFoundError(
            f"Configured visible truth sidecar does not exist: {data.visible_truth}"
        )
    _exact_keys(data.loader, {"pin_memory", "persistent_workers", "prefetch_factor"}, "data.loader")
    if not isinstance(data.loader.pin_memory, bool) or not isinstance(data.loader.persistent_workers, bool):
        raise ValueError("data.loader pin_memory and persistent_workers must be boolean")
    if data.loader.prefetch_factor is not None:
        _positive_int(data.loader.prefetch_factor, "data.loader.prefetch_factor")
    int(data.subset_seed)
    test_size = float(data.test_size)
    if not math.isfinite(test_size) or not 0.0 < test_size < 1.0:
        raise ValueError("data.test_size must lie strictly between zero and one")
    _exact_keys(data.split, {"indices_path", "require_existing"}, "data.split")
    if data.split.indices_path is None and bool(data.split.require_existing):
        raise ValueError("data.split.require_existing=true requires data.split.indices_path")
    if not isinstance(data.split.require_existing, bool):
        raise ValueError("data.split.require_existing must be boolean")
    if not data.truth_bindings:
        raise ValueError("data.truth_bindings must be non-empty")
    _exact_keys(data.max_events, {"train", "val", "test"}, "data.max_events")
    _exact_keys(data.invisible, {"path", "allow_synthetic"}, "data.invisible")
    if not isinstance(data.invisible.allow_synthetic, bool):
        raise ValueError("data.invisible.allow_synthetic must be boolean")
    _exact_keys(data.parent, {"path", "roles"}, "data.parent")
    if data.parent.path is not None:
        if not str(data.parent.path).strip() or not data.parent.roles:
            raise ValueError("data.parent requires both path and roles")
        if not Path(str(data.parent.path)).is_file():
            raise FileNotFoundError(f"Parent truth sidecar does not exist: {data.parent.path}")
    _exact_keys(data.partonic, {"path", "roles", "truth_bindings"}, "data.partonic")
    if data.partonic.path is not None:
        if not str(data.partonic.path).strip() or not data.partonic.roles:
            raise ValueError("data.partonic requires both path and roles")
        if not Path(str(data.partonic.path)).is_file():
            raise FileNotFoundError(f"Partonic truth sidecar does not exist: {data.partonic.path}")
    _positive_int(data.max_assignment_states, "data.max_assignment_states")
    for split in ("train", "val", "test"):
        limit = data.max_events[split]
        if limit is not None:
            _positive_int(limit, f"data.max_events.{split}")
    model = config.model
    _exact_keys(model, MODEL_KEYS, "model")
    mode = str(model.mode)
    if mode not in {"factorized", "joint_flow"}:
        raise ValueError("model.mode must resolve to factorized or joint_flow")
    _latent(model.latent, "model.latent")
    visible_type = str(model.visible.type)
    visible_keys = {"type", "d_model", "n_heads", "layers", "d_feedforward", "dropout", "inputs", "representation", "relations_type"}
    if visible_type == "relational_transformer":
        visible_keys |= {"momentum_scale_gev", "relations"}
    elif visible_type == "lgatr":
        visible_keys = {
            "type", "d_model", "layers", "n_heads", "hidden",
            "dropout", "momentum_scale_gev", "inputs", "symmetry_breaking", "representation", "relations_type",
        }
    elif visible_type == "lgatr_slim":
        visible_keys = {
            "type", "d_model", "layers", "n_heads", "hidden",
            "dropout", "momentum_scale_gev", "inputs", "symmetry_breaking", "representation", "relations_type",
        }
    elif visible_type != "transformer":
        raise ValueError("resolved model.visible.type is invalid")
    _exact_keys(model.visible, visible_keys, "model.visible")
    width = _positive_int(model.visible.d_model, "model.visible.d_model")
    if visible_type in {"transformer", "relational_transformer"}:
        heads = _positive_int(model.visible.n_heads, "model.visible.n_heads")
        if width % heads:
            raise ValueError("model.visible.d_model must be divisible by n_heads")
        _positive_int(model.visible.layers, "model.visible.layers")
        _positive_int(model.visible.d_feedforward, "model.visible.d_feedforward")
    else:
        _positive_int(model.visible.layers, "model.visible.layers")
        _positive_int(model.visible.n_heads, "model.visible.n_heads")
    visible_dropout = float(model.visible.dropout)
    if not math.isfinite(visible_dropout) or not 0 <= visible_dropout < 1:
        raise ValueError("model.visible.dropout must lie in [0,1)")
    if int(model.visible.d_model) != int(model.latent.d_model):
        raise ValueError("visible encoder output width must equal model.latent.d_model")
    _exact_keys(
        model.visible.inputs,
        {"kinematics", "object_features", "event_features"},
        "model.visible.inputs",
    )
    _exact_keys(
        model.topology_queries,
        {"enabled", "interaction", "layers", "n_heads", "dropout", "roles", "relation_edges"},
        "model.topology_queries",
    )
    if model.topology_queries.enabled:
        if not model.topology_queries.roles:
            raise ValueError("Enabled topology queries require roles")
        if str(model.topology_queries.interaction) not in {"full", "topology", "topology_plus_global"}:
            raise ValueError("model.topology_queries.interaction is invalid")
        _positive_int(model.topology_queries.layers, "model.topology_queries.layers")
        _positive_int(model.topology_queries.n_heads, "model.topology_queries.n_heads")
        if int(model.latent.d_model) % int(model.topology_queries.n_heads):
            raise ValueError("model.topology_queries.n_heads must divide model.latent.d_model")
    _exact_keys(
        model.parent_cfm,
        {
            "enabled", "conditioning", "use_topology_queries", "roles", "truth_bindings", "target_path",
            "n_heads", "layers", "d_feedforward", "dropout",
            "p2_neutrino_steps", "p2_neutrino_solver", "sampling_steps",
            "sampling_solver",
        },
        "model.parent_cfm",
    )
    if not isinstance(model.parent_cfm.enabled, bool):
        raise ValueError("model.parent_cfm.enabled must be boolean")
    if str(model.parent_cfm.conditioning) not in {"P0", "P1", "P2"}:
        raise ValueError("model.parent_cfm.conditioning must be P0, P1 or P2")
    if not isinstance(model.parent_cfm.use_topology_queries, bool):
        raise ValueError("model.parent_cfm.use_topology_queries must be boolean")
    if list(model.parent_cfm.roles) != ["top", "antitop"]:
        raise ValueError("Partonic CFM roles must be ordered [top, antitop]")
    if set(model.parent_cfm.truth_bindings) != {"top", "antitop"}:
        raise ValueError("Partonic CFM truth_bindings must define top and antitop")
    if model.parent_cfm.enabled:
        if model.parent_cfm.use_topology_queries and not model.topology_queries.enabled:
            raise ValueError("Partonic CFM use_topology_queries requires enabled topology queries")
        if not model.parent_cfm.use_topology_queries:
            if model.parent_cfm.conditioning != "P0":
                raise ValueError("No-query parent CFM currently supports P0 conditioning only")
        _positive_int(model.parent_cfm.n_heads, "model.parent_cfm.n_heads")
        if int(model.latent.d_model) % int(model.parent_cfm.n_heads):
            raise ValueError("model.parent_cfm.n_heads must divide model.latent.d_model")
        _positive_int(model.parent_cfm.layers, "model.parent_cfm.layers")
        _positive_int(model.parent_cfm.d_feedforward, "model.parent_cfm.d_feedforward")
        dropout = float(model.parent_cfm.dropout)
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("model.parent_cfm.dropout must lie in [0,1)")
        _positive_int(model.parent_cfm.p2_neutrino_steps, "model.parent_cfm.p2_neutrino_steps")
        _positive_int(model.parent_cfm.sampling_steps, "model.parent_cfm.sampling_steps")
        if str(model.parent_cfm.p2_neutrino_solver) not in {"euler", "heun"}:
            raise ValueError("model.parent_cfm.p2_neutrino_solver must be euler or heun")
        if str(model.parent_cfm.sampling_solver) not in {"euler", "heun"}:
            raise ValueError("model.parent_cfm.sampling_solver must be euler or heun")
    if len(set(model.visible.inputs.object_features)) != len(model.visible.inputs.object_features):
        raise ValueError("model.visible.inputs.object_features must be unique")
    if len(set(model.visible.inputs.event_features)) != len(model.visible.inputs.event_features):
        raise ValueError("model.visible.inputs.event_features must be unique")
    if visible_type == "relational_transformer":
        _positive(model.visible.momentum_scale_gev, "model.visible.momentum_scale_gev")
        _exact_keys(model.visible.relations, {"enabled", "hidden_dim", "features"}, "model.visible.relations")
        _positive_int(model.visible.relations.hidden_dim, "model.visible.relations.hidden_dim")
    elif visible_type in {"lgatr", "lgatr_slim"}:
        _positive(model.visible.momentum_scale_gev, "model.visible.momentum_scale_gev")
        _exact_keys(model.visible.symmetry_breaking, {"beam", "time"}, "model.visible.symmetry_breaking")
        if str(model.visible.symmetry_breaking.beam) not in {"xyplane", "none"}:
            raise ValueError("model.visible.symmetry_breaking.beam is invalid")
        expected_hidden = (
            {"multivector_channels", "scalar_channels"}
            if visible_type == "lgatr"
            else {"vector_channels", "scalar_channels"}
        )
        _exact_keys(model.visible.hidden, expected_hidden, "model.visible.hidden")
        for name in expected_hidden:
            _positive_int(model.visible.hidden[name], f"model.visible.hidden.{name}")

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
        {
            "enabled",
            "formulation",
            "conditioning",
            "coordinates",
            "mass_shell",
            "backbone",
            "denoiser",
            "standard",
            "edm",
            "flow_matching",
        },
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
    _mass_shell(invisible.mass_shell, "model.invisible.mass_shell")
    _invisible_backbone(invisible.backbone, "model.invisible.backbone")
    _exact_keys(
        invisible.coordinates,
        {"type", "version", "basis_convention"},
        "model.invisible.coordinates",
    )
    if str(invisible.coordinates.type) not in {"cartesian", "met_residual", "met_exact"}:
        raise ValueError("resolved invisible coordinate type is invalid")
    if str(invisible.coordinates.version) != "v1":
        raise ValueError("resolved invisible coordinate version is invalid")
    expected_basis = (
        "helmert_orthonormal_v1"
        if str(invisible.coordinates.type) in {"met_residual", "met_exact"}
        else None
    )
    if invisible.coordinates.basis_convention != expected_basis:
        raise ValueError("resolved invisible coordinate basis convention is invalid")
    _network(invisible.denoiser, "model.invisible.denoiser", zero_init=True)
    _exact_keys(invisible.standard, {"training_steps"}, "model.invisible.standard")
    _positive_int(invisible.standard.training_steps, "model.invisible.standard.training_steps")
    _exact_keys(invisible.edm, {"sigma_data", "p_mean", "p_std"}, "model.invisible.edm")
    _positive(invisible.edm.sigma_data, "model.invisible.edm.sigma_data")
    _finite(invisible.edm.p_mean, "model.invisible.edm.p_mean")
    _positive(invisible.edm.p_std, "model.invisible.edm.p_std")
    _exact_keys(invisible.flow_matching, {"coupling"}, "model.invisible.flow_matching")
    _exact_keys(
        invisible.flow_matching.coupling,
        {"method", "condition_weight"},
        "model.invisible.flow_matching.coupling",
    )
    coupling_method = str(invisible.flow_matching.coupling.method)
    if coupling_method not in {"independent", "minibatch_ot", "conditional_ot"}:
        raise ValueError("CFM coupling method is invalid")
    condition_weight = float(invisible.flow_matching.coupling.condition_weight)
    if not math.isfinite(condition_weight) or condition_weight < 0:
        raise ValueError("CFM coupling condition_weight must be finite and nonnegative")
    if coupling_method != "conditional_ot" and condition_weight != 0:
        raise ValueError("CFM condition_weight must be zero outside conditional_ot")

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
    _exact_keys(model.heads, {"point_invisible", "parent_four_vector", "partonic"}, "model.heads")
    _exact_keys(model.heads.point_invisible, {"enabled"}, "model.heads.point_invisible")
    _exact_keys(model.heads.parent_four_vector, {"enabled", "roles", "target_path"}, "model.heads.parent_four_vector")
    _exact_keys(model.heads.partonic, {"enabled", "roles", "target_path", "truth_bindings", "targets"}, "model.heads.partonic")
    if model.heads.parent_four_vector.enabled:
        if not model.heads.parent_four_vector.roles:
            raise ValueError("Enabled parent four-vector head requires non-empty roles")
        if not str(model.heads.parent_four_vector.target_path).strip():
            raise ValueError("Enabled parent four-vector head requires target_path")
    if model.heads.point_invisible.enabled and not invisible.enabled:
        raise ValueError("Enabled point head requires an active invisible model")
    if model.heads.parent_four_vector.enabled:
        if not invisible.enabled:
            raise ValueError("Enabled parent head requires an active invisible model")
        if data.parent.path is None or list(data.parent.roles) != list(model.heads.parent_four_vector.roles):
            raise ValueError("Parent head roles/path must match the resolved data.parent contract")
    elif data.parent.path is not None:
        raise ValueError("data.parent must be disabled when the parent head is disabled")
    if model.heads.partonic.enabled:
        if not model.topology_queries.enabled:
            raise ValueError("Enabled partonic head requires topology queries")
        if not model.heads.partonic.roles:
            raise ValueError("Enabled partonic head requires non-empty roles")
        if data.partonic.path is not None:
            if list(data.partonic.roles) != list(model.heads.partonic.roles):
                raise ValueError("Partonic head roles must match data.partonic roles")
            if dict(data.partonic.truth_bindings) != dict(model.heads.partonic.truth_bindings):
                raise ValueError("Partonic truth-binding metadata must match the head declaration")
    elif data.partonic.path is not None and not model.parent_cfm.enabled:
        raise ValueError("data.partonic must be disabled when the partonic head is disabled")
    if model.parent_cfm.enabled:
        if model.mode != "factorized":
            raise ValueError("Partonic CFM requires model.mode=factorized")
        if model.parent_cfm.conditioning in {"P1", "P2"}:
            if model.assignment.formulation != "discriminative":
                raise ValueError("P1/P2 require predicted discriminative assignment probabilities")
        if model.parent_cfm.conditioning == "P2":
            if not invisible.enabled or str(invisible.formulation) != "flow_matching":
                raise ValueError("P2 requires the existing invisible flow-matching generator")
        if data.partonic.roles and list(data.partonic.roles) != ["top", "antitop"]:
            raise ValueError("Partonic CFM data must be role ordered [top, antitop]")
        if dict(data.partonic.truth_bindings) != dict(model.parent_cfm.truth_bindings):
            raise ValueError("Partonic CFM data and model truth bindings differ")

    if mode == "factorized" and formulation == "flow_matching":
        if reasoning.enabled or feedback.enabled or invisible.enabled or c2.enabled or model.compatibility.enabled:
            raise ValueError("factorized DFM is assignment-only")
    if c2.enabled and (
        mode != "factorized"
        or formulation != "discriminative"
        or not invisible.enabled
    ):
        raise ValueError(
            "C2 requires factorized discriminative assignment plus invisible inference"
        )
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
    _exact_keys(inference.invisible, {"point_estimator", "standard", "edm", "flow_matching"}, "inference.invisible")
    if str(inference.invisible.point_estimator) not in {
        "none", "cfm_single_draw", "cfm_cartesian_mean", "cfm_joint_medoid", "direct_point_head"
    }:
        raise ValueError("inference.invisible.point_estimator is invalid")
    point_estimator = str(inference.invisible.point_estimator)
    if point_estimator.startswith("cfm_") and (
        not invisible.enabled or str(invisible.formulation) != "flow_matching"
    ):
        raise ValueError("CFM point estimators require an active flow-matching invisible model")
    if point_estimator == "direct_point_head" and not model.heads.point_invisible.enabled:
        raise ValueError("direct_point_head inference requires an enabled point_invisible head")
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
        {"batch_size", "workers", "learning_rate", "weight_decay", "fused_adamw", "epochs", "precision", "gradient_clip", "accumulate_grad_batches", "trainable", "loss_weights", "scheduler", "early_stopping", "runtime", "output", "shared_encoder_gradients", "partonic_shared_encoder_gradients", "parent_cfm_shared_encoder_gradients"},
        "training",
    )
    _positive_int(training.batch_size, "training.batch_size")
    if int(training.workers) < 0:
        raise ValueError("training.workers must be non-negative")
    _positive(training.learning_rate, "training.learning_rate")
    _nonnegative(training.weight_decay, "training.weight_decay")
    if not isinstance(training.fused_adamw, bool):
        raise ValueError("training.fused_adamw must be boolean")
    _positive_int(training.epochs, "training.epochs")
    _nonnegative(training.gradient_clip, "training.gradient_clip")
    _positive_int(training.accumulate_grad_batches, "training.accumulate_grad_batches")
    if not str(training.precision).strip() or not str(training.output).strip():
        raise ValueError("training.precision and output must be non-empty")
    if str(training.shared_encoder_gradients) not in {"both", "assignment_only"}:
        raise ValueError("training.shared_encoder_gradients must be both or assignment_only")
    if str(training.partonic_shared_encoder_gradients) not in {"both", "assignment_only"}:
        raise ValueError("training.partonic_shared_encoder_gradients must be both or assignment_only")
    if str(training.parent_cfm_shared_encoder_gradients) not in {"both", "assignment_only"}:
        raise ValueError("training.parent_cfm_shared_encoder_gradients must be both or assignment_only")
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
    if "point" in trainable and not model.heads.point_invisible.enabled:
        raise ValueError("training.trainable includes point but point_invisible is inactive")
    if "parent" in trainable and not model.heads.parent_four_vector.enabled:
        raise ValueError("training.trainable includes parent but parent_four_vector is inactive")
    if "queries" in trainable and not model.topology_queries.enabled:
        raise ValueError("training.trainable includes queries but topology_queries is inactive")
    if "partonic" in trainable and not model.heads.partonic.enabled:
        raise ValueError("training.trainable includes partonic but the partonic head is inactive")
    if "partonic_cfm" in trainable and not model.parent_cfm.enabled:
        raise ValueError("training.trainable includes partonic_cfm but the parent CFM is inactive")
    if model.parent_cfm.enabled and (
        "partonic_cfm" in trainable or float(training.loss_weights.partonic_cfm) > 0
    ) and model.parent_cfm.target_path is None:
        raise ValueError("Partonic CFM training requires an explicit truth sidecar")
    if mode == "joint_flow" and (("assignment" in trainable) != ("invisible" in trainable)):
        raise ValueError("joint_flow assignment and invisible dynamics must be trained/frozen together")
    _exact_keys(training.loss_weights, {"assignment", "invisible", "compatibility", "point", "parent", "partonic", "partonic_cfm"}, "training.loss_weights")
    for key in training.loss_weights:
        _nonnegative(training.loss_weights[key], f"training.loss_weights.{key}")
    if model.heads.partonic.enabled and (
        "partonic" in trainable or float(training.loss_weights.partonic) > 0.0
    ) and data.partonic.path is None:
        raise ValueError(
            "Partonic-head training requires data.partonic truth; "
            "truth-free inference may leave data.partonic unset."
        )
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
