"""Resolve compact human-facing JER configs into the explicit internal contract."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from joint_event_reconstruction.topology.compiler import load_topology


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = REPO_ROOT / "configs/defaults.yaml"
PUBLIC_TOP_KEYS = {"experiment", "topology", "data", "model", "prediction", "training"}
TRAINABLE_COMPONENTS = {"visible", "assignment", "invisible", "c2", "compatibility"}


def _path(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _mapping(value, name: str, *, allow_none: bool = False):
    if value is None and allow_none:
        return None
    if not OmegaConf.is_config(value) and not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping" + (" or null" if allow_none else ""))
    return value


def _unknown(section, allowed: set[str], name: str) -> None:
    extra = sorted(set(section) - allowed)
    if extra:
        raise ValueError(f"Unknown {name} setting(s): {extra}")


def _required(section, keys: set[str], name: str) -> None:
    missing = sorted(keys - set(section))
    if missing:
        raise ValueError(f"Missing required {name} setting(s): {missing}")


def _plain(value):
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value


def _merge(base, override):
    if override is None:
        return OmegaConf.create(_plain(base))
    return OmegaConf.merge(OmegaConf.create(_plain(base)), override)


def _load_data(path: Path) -> DictConfig:
    if not path.is_file():
        raise FileNotFoundError(f"JER data config does not exist: {path}")
    data = OmegaConf.load(path)
    _unknown(data, {"source", "invisible_truth", "split", "tensor_cache", "loader", "objects"}, "data config")
    _required(data, {"source", "invisible_truth", "split", "tensor_cache", "loader", "objects"}, "data config")
    _unknown(data.loader, {"batch_size", "workers", "pin_memory", "persistent_workers", "prefetch_factor"}, "data.loader")
    _required(data.loader, {"batch_size", "workers", "pin_memory", "persistent_workers", "prefetch_factor"}, "data.loader")
    if not data.objects:
        raise ValueError("data.objects must be non-empty")
    for source, spec in data.objects.items():
        _unknown(spec, {"count", "variables", "scalers"}, f"data.objects.{source}")
        _required(spec, {"variables", "scalers"}, f"data.objects.{source}")
        if len(spec.variables) != len(spec.scalers):
            raise ValueError(f"data.objects.{source}: variables and scalers must have equal length")
        if source != "GLOBAL" and spec.get("count") is None:
            raise ValueError(f"data.objects.{source}.count is required")
    return data


def _visible_targets(topology_path: Path) -> dict:
    """Minimal M0-loader targets used only to request raw LABELS for topology sources."""
    spec = load_topology(topology_path)
    targets = {}
    for node_name in spec.dag_order:
        members = []
        for role in spec.nodes[node_name].leaf_roles:
            if role.invisible:
                continue
            members.extend([[role.source, int(index)] for index in role.truth_indices])
        if members:
            targets[node_name] = members
    if not targets:
        raise ValueError(f"Topology {topology_path} has no visible truth roles")
    return targets


def build_visible_loader_config(experiment: DictConfig) -> DictConfig:
    """Adapt one JER-native data config to the mature fixed-token visible loader."""
    data_path = _path(experiment.visible_config)
    data = _load_data(data_path)
    topology_path = _path(experiment.topology_config)
    loader = data.loader
    observables = []
    for source, spec in data.objects.items():
        entry = {
            "INPUTS": str(source),
            "variables": list(spec.variables),
            "scalers": list(spec.scalers),
        }
        if source != "GLOBAL":
            entry["num_particles"] = int(spec.count)
        observables.append(entry)

    object_counts = {
        str(source): int(spec.count)
        for source, spec in data.objects.items()
        if source != "GLOBAL"
    }
    lepton_counts = [
        count
        for source, count in object_counts.items()
        if source in {"LEPTON", "ELECTRON", "MUON"}
    ]
    dataset = {
        "input_h5_path": str(_path(data.source)),
        "max_jets": int(object_counts.get("JET", 0)),
        "max_leptons": max(lepton_counts, default=0),
        "include_met": "MET" in object_counts,
        "test_size": 0.2,
        "limit_events": None,
        "split": {
            "indices_path": str(_path(data.split)),
            "require_existing": True,
        },
        "padding_value": 0.0,
        "train_batch_size": int(loader.batch_size),
        "val_batch_size": int(loader.batch_size),
        "test_batch_size": int(loader.batch_size),
        "train_num_workers": int(loader.workers),
        "val_num_workers": int(loader.workers),
        "test_num_workers": int(loader.workers),
        "pin_memory": bool(loader.pin_memory),
        "persistent_workers": bool(loader.persistent_workers),
        "prefetch_factor": int(loader.prefetch_factor),
        "cache": {"enabled": False},
        "observables": observables,
    }
    return OmegaConf.create(
        {
            "general": {
                "model": "transformer",
                "topology": str(experiment.experiment.name),
                "mode": "train",
                "device": "cuda",
                "seed": int(experiment.experiment.seed),
            },
            "tasks": {
                "classification": {"enabled": False, "weight": 0.0},
                "reconstruction": {
                    "enabled": True,
                    "weight": 1.0,
                    "label_source": "LABELS",
                    "ignore_index": -100,
                    "targets": _visible_targets(topology_path),
                },
            },
            "dataset": dataset,
            "train": {
                "compute_val_auc": False,
                "checkpoint": {
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_top_k": 1,
                    "save_last": False,
                },
            },
        }
    )


def _resolve_model(public_model, defaults) -> dict:
    _unknown(public_model, {"encoder", "assignment", "invisible", "joint_flow", "c2", "compatibility"}, "model")
    _required(public_model, {"encoder"}, "model")
    encoder = public_model.encoder
    _unknown(encoder, {"d_model", "n_heads", "layers", "d_feedforward", "dropout"}, "model.encoder")
    _required(encoder, {"d_model", "n_heads", "layers", "d_feedforward", "dropout"}, "model.encoder")

    joint_public = public_model.get("joint_flow")
    is_joint = joint_public is not None
    assignment_public = public_model.get("assignment")
    invisible_public = public_model.get("invisible")
    c2_public = public_model.get("c2")
    compatibility_public = public_model.get("compatibility")

    if is_joint:
        for name, value in (("assignment", assignment_public), ("invisible", invisible_public), ("c2", c2_public), ("compatibility", compatibility_public)):
            if value is not None:
                raise ValueError(f"model.{name} must be omitted when model.joint_flow is used")
        _mapping(joint_public, "model.joint_flow")
        _unknown(joint_public, {"recurrent", "block_layers", "train_steps"}, "model.joint_flow")
        joint = _merge(defaults.model.joint_flow, joint_public)
        recurrent_state = bool(joint.get("recurrent", False))
        return {
            "mode": "joint_flow",
            "visible": _plain(encoder),
            "assignment": {
                "formulation": "flow_matching",
                "reasoning": {
                    "enabled": False,
                    "block_layers": int(defaults.model.assignment.recurrence.block_layers),
                    "train_steps": list(defaults.model.assignment.recurrence.train_steps),
                    "eval_steps": int(defaults.model.assignment.recurrence.eval_steps),
                    "fixed_input_reinjection": bool(defaults.model.assignment.recurrence.reinject_visible),
                },
                "feedback": {
                    "enabled": False,
                    "start_step": int(defaults.model.assignment.recurrence.feedback.start_step),
                    "gate_initial_bias": float(defaults.model.assignment.recurrence.feedback.gate_initial_bias),
                },
            },
            "invisible": {
                "enabled": True,
                "formulation": "flow_matching",
                "conditioning": {"assignment": "none", "allow_truth_oracle": False},
                "denoiser": _plain(defaults.model.invisible.network),
                "standard": _plain(defaults.model.invisible.standard),
                "edm": _plain(defaults.model.invisible.edm),
                "flow_matching": _plain(defaults.model.invisible.flow_matching),
            },
            "joint_flow": {
                "recurrent_state": recurrent_state,
                "block_layers": int(joint.block_layers),
                "train_steps": list(joint.train_steps),
            },
            "coupling": {
                "invisible_to_assignment": {
                    "enabled": False,
                    "posterior_samples": int(defaults.model.c2.posterior_samples),
                    "gate_initial_bias": float(defaults.model.c2.gate_initial_bias),
                }
            },
            "compatibility": {
                "enabled": False,
                "include_invisible": bool(defaults.model.compatibility.include_invisible),
            },
        }

    if assignment_public is None:
        raise ValueError("Factorized models require model.assignment")
    _mapping(assignment_public, "model.assignment")
    _unknown(assignment_public, {"method", "recurrence"}, "model.assignment")
    _required(assignment_public, {"method"}, "model.assignment")
    assignment_method = str(assignment_public.method)
    if assignment_method not in {"discriminative", "dfm"}:
        raise ValueError("model.assignment.method must be discriminative or dfm")

    recurrence_public = assignment_public.get("recurrence")
    recurrence_defaults = defaults.model.assignment.recurrence
    if recurrence_public is None:
        recurrence = recurrence_defaults
        recurrence_enabled = False
        feedback_enabled = False
    else:
        _mapping(recurrence_public, "model.assignment.recurrence")
        _unknown(recurrence_public, {"block_layers", "train_steps", "eval_steps", "reinject_visible", "feedback"}, "model.assignment.recurrence")
        recurrence = _merge(recurrence_defaults, recurrence_public)
        recurrence_enabled = True
        feedback = recurrence_public.get("feedback")
        if feedback not in {None, "soft"}:
            raise ValueError("model.assignment.recurrence.feedback must be soft or omitted")
        feedback_enabled = feedback == "soft"
    if assignment_method == "dfm" and recurrence_public is not None:
        raise ValueError("Factorized DFM has its own time-conditioned reasoner; omit recurrence")

    invisible = {
        "enabled": False,
        "formulation": "edm",
        "conditioning": {"assignment": "none", "allow_truth_oracle": False},
        "denoiser": _plain(defaults.model.invisible.network),
        "standard": _plain(defaults.model.invisible.standard),
        "edm": _plain(defaults.model.invisible.edm),
        "flow_matching": _plain(defaults.model.invisible.flow_matching),
    }
    if invisible_public is not None:
        if assignment_method == "dfm":
            raise ValueError("Factorized DFM is assignment-only; omit model.invisible or use joint_flow")
        _mapping(invisible_public, "model.invisible")
        _unknown(invisible_public, {"method", "conditioning", "coupling", "network", "training"}, "model.invisible")
        _required(invisible_public, {"method"}, "model.invisible")
        method = str(invisible_public.method)
        method_map = {"ddim": "standard", "edm": "edm", "cfm": "flow_matching"}
        if method not in method_map:
            raise ValueError("model.invisible.method must be ddim, edm or cfm")
        conditioning = str(invisible_public.get("conditioning", "none"))
        if conditioning not in {"none", "soft", "hard", "hypothesis", "truth_oracle"}:
            raise ValueError("model.invisible.conditioning must be none, soft, hard, hypothesis or truth_oracle")
        invisible["enabled"] = True
        invisible["formulation"] = method_map[method]
        invisible["conditioning"] = {
            "assignment": "truth" if conditioning == "truth_oracle" else conditioning,
            "allow_truth_oracle": conditioning == "truth_oracle",
        }
        if invisible_public.get("network") is not None:
            invisible["denoiser"] = _plain(_merge(defaults.model.invisible.network, invisible_public.network))
        if method == "cfm":
            coupling = str(invisible_public.get("coupling", defaults.model.invisible.flow_matching.coupling))
            if coupling not in {"independent", "minibatch_ot"}:
                raise ValueError("model.invisible.coupling must be independent or minibatch_ot")
            invisible["flow_matching"] = {"coupling": coupling}
        elif invisible_public.get("coupling") is not None:
            raise ValueError("model.invisible.coupling is only meaningful for method=cfm")
        training_override = invisible_public.get("training")
        if training_override is not None:
            _mapping(training_override, "model.invisible.training")
            if method == "ddim":
                invisible["standard"] = _plain(_merge(defaults.model.invisible.standard, training_override))
            elif method == "edm":
                invisible["edm"] = _plain(_merge(defaults.model.invisible.edm, training_override))
            elif training_override:
                raise ValueError("CFM has no public method-specific training settings")

    c2 = _plain(defaults.model.c2)
    c2_enabled = c2_public is not None
    if c2_enabled:
        if not recurrence_enabled or not invisible["enabled"]:
            raise ValueError("model.c2 requires recurrent discriminative assignment and invisible inference")
        _mapping(c2_public, "model.c2")
        _unknown(c2_public, {"posterior_samples", "gate_initial_bias"}, "model.c2")
        c2 = _plain(_merge(defaults.model.c2, c2_public))

    compatibility = _plain(defaults.model.compatibility)
    compatibility_enabled = compatibility_public is not None
    if compatibility_enabled:
        if not invisible["enabled"]:
            raise ValueError("model.compatibility requires invisible inference")
        _mapping(compatibility_public, "model.compatibility")
        _unknown(compatibility_public, {"include_invisible"}, "model.compatibility")
        compatibility = _plain(_merge(defaults.model.compatibility, compatibility_public))

    return {
        "mode": "factorized",
        "visible": _plain(encoder),
        "assignment": {
            "formulation": "flow_matching" if assignment_method == "dfm" else "discriminative",
            "reasoning": {
                "enabled": recurrence_enabled,
                "block_layers": int(recurrence.block_layers),
                "train_steps": list(recurrence.train_steps),
                "eval_steps": int(recurrence.eval_steps),
                "fixed_input_reinjection": bool(recurrence.reinject_visible),
            },
            "feedback": {
                "enabled": feedback_enabled,
                "start_step": int(recurrence.feedback.start_step),
                "gate_initial_bias": float(recurrence.feedback.gate_initial_bias),
            },
        },
        "invisible": invisible,
        "joint_flow": {
            "recurrent_state": False,
            "block_layers": int(defaults.model.joint_flow.block_layers),
            "train_steps": list(defaults.model.joint_flow.train_steps),
        },
        "coupling": {
            "invisible_to_assignment": {
                "enabled": c2_enabled,
                "posterior_samples": int(c2["posterior_samples"]),
                "gate_initial_bias": float(c2["gate_initial_bias"]),
            }
        },
        "compatibility": {
            "enabled": compatibility_enabled,
            "include_invisible": bool(compatibility["include_invisible"]),
        },
    }


def resolve_public_experiment(public: DictConfig, *, source_path: str | Path | None = None) -> DictConfig:
    _mapping(public, "experiment config")
    _unknown(public, PUBLIC_TOP_KEYS, "experiment")
    _required(public, PUBLIC_TOP_KEYS, "experiment")
    _unknown(public.experiment, {"name", "seed"}, "experiment.experiment")
    _required(public.experiment, {"name", "seed"}, "experiment.experiment")

    topology_path = _path(public.topology)
    if not topology_path.is_file():
        raise FileNotFoundError(f"Topology config does not exist: {topology_path}")

    data_public = public.data
    _mapping(data_public, "data")
    _unknown(data_public, {"config", "events", "max_assignment_states"}, "data")
    _required(data_public, {"config", "events", "max_assignment_states"}, "data")
    _unknown(data_public.events, {"train", "validation", "test"}, "data.events")
    _required(data_public.events, {"train", "validation", "test"}, "data.events")
    data_path = _path(data_public.config)
    data_config = _load_data(data_path)

    defaults = OmegaConf.load(DEFAULTS_PATH)
    model = _resolve_model(public.model, defaults)

    prediction_public = public.prediction
    _mapping(prediction_public, "prediction")
    _unknown(prediction_public, {"seed", "top_k", "samples", "steps", "solver", "sample_chunk_size", "nfe", "sigma_min", "sigma_max", "rho", "compatibility_weight"}, "prediction")
    prediction = _merge(defaults.prediction, prediction_public)

    training_public = public.training
    _mapping(training_public, "training")
    _unknown(training_public, {"learning_rate", "weight_decay", "epochs", "precision", "gradient_clip", "accumulate_grad_batches", "trainable", "loss_weights", "scheduler", "early_stopping", "output"}, "training")
    training = _merge(defaults.training, training_public)

    if training_public.get("trainable") is None:
        active = ["visible", "assignment"]
        if model["invisible"]["enabled"]:
            active.append("invisible")
        if model["coupling"]["invisible_to_assignment"]["enabled"]:
            active.append("c2")
        if model["compatibility"]["enabled"]:
            active.append("compatibility")
        trainable = active
    else:
        trainable = [str(value) for value in training_public.trainable]
    if not trainable or len(set(trainable)) != len(trainable):
        raise ValueError("training.trainable must contain unique component names")
    unknown_trainable = sorted(set(trainable) - TRAINABLE_COMPONENTS)
    if unknown_trainable:
        raise ValueError(f"Unknown training.trainable components: {unknown_trainable}")
    if "c2" in trainable and "assignment" not in trainable:
        raise ValueError("training.trainable=c2 currently requires assignment to be trainable")
    if model["mode"] == "joint_flow" and (("assignment" in trainable) != ("invisible" in trainable)):
        raise ValueError("joint_flow assignment and invisible dynamics must be trainable/frozen together")

    default_weights = {
        "assignment": 1.0 if "assignment" in trainable else 0.0,
        "invisible": 1.0 if "invisible" in trainable else 0.0,
        "compatibility": 1.0 if "compatibility" in trainable else 0.0,
    }
    loss_weights = dict(default_weights)
    if training_public.get("loss_weights") is not None:
        for key, value in training_public.loss_weights.items():
            if key not in loss_weights:
                raise ValueError(f"Unknown training.loss_weights key {key!r}")
            loss_weights[key] = float(value)

    invisible_truth = data_config.invisible_truth
    internal = {
        "experiment": _plain(public.experiment),
        "topology_config": str(topology_path),
        "visible_config": str(data_path),
        "data": {
            "tensor_cache": None if data_config.tensor_cache is None else str(_path(data_config.tensor_cache)),
            "max_events": {
                "train": public.data.events.train,
                "val": public.data.events.validation,
                "test": public.data.events.test,
            },
            "max_assignment_states": int(public.data.max_assignment_states),
            "invisible": {
                "path": None if invisible_truth is None else str(_path(invisible_truth)),
                "allow_synthetic": False,
            },
        },
        "model": model,
        "inference": {
            "seed": int(prediction.seed),
            "top_k_assignments": int(prediction.top_k),
            "posterior_samples": int(prediction.samples),
            "invisible": {
                "standard": {"steps": int(prediction.get("steps", defaults.prediction.standard_steps))},
                "edm": {
                    "sigma_min": float(prediction.sigma_min),
                    "sigma_max": float(prediction.sigma_max),
                    "rho": float(prediction.rho),
                    "nfe": int(prediction.get("nfe", defaults.prediction.edm.nfe)),
                },
                "flow_matching": {
                    "steps": int(prediction.steps),
                    "solver": str(prediction.solver),
                },
            },
            "assignment_flow": {"steps": int(prediction.steps)},
            "joint_flow": {
                "steps": int(prediction.steps),
                "sample_chunk_size": int(prediction.sample_chunk_size),
            },
            "compatibility_weight": float(prediction.compatibility_weight),
        },
        "training": {
            "learning_rate": float(training.learning_rate),
            "weight_decay": float(training.weight_decay),
            "epochs": int(training.epochs),
            "precision": str(training.precision),
            "gradient_clip": float(training.gradient_clip),
            "accumulate_grad_batches": int(training.accumulate_grad_batches),
            "trainable": trainable,
            "loss_weights": loss_weights,
            "scheduler": _plain(training.scheduler),
            "early_stopping": _plain(training.early_stopping),
            "output": str(training_public.get("output", f"results/{public.experiment.name}")),
        },
    }
    return OmegaConf.create(internal)


def load_public_experiment(path: str | Path, overrides: list[str] | tuple[str, ...] | None = None) -> DictConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration does not exist: {path}")
    public = OmegaConf.load(path)
    if overrides:
        public = OmegaConf.merge(public, OmegaConf.from_dotlist(list(overrides)))
    return resolve_public_experiment(public, source_path=path)
