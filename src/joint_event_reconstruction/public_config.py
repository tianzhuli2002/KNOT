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


def _mapping(value, name: str):
    if not OmegaConf.is_config(value) and not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
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
    allowed = {"source", "invisible_truth", "split", "tensor_cache", "loader", "objects"}
    _unknown(data, allowed, "data config")
    _required(data, allowed, "data config")
    loader_keys = {"batch_size", "workers", "pin_memory", "persistent_workers", "prefetch_factor"}
    _unknown(data.loader, loader_keys, "data.loader")
    _required(data.loader, loader_keys, "data.loader")
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
    """Minimal loader targets used only to request raw LABELS for topology sources."""
    spec = load_topology(topology_path)
    targets = {}
    for node_name in spec.dag_order:
        members = []
        for role in spec.nodes[node_name].leaf_roles:
            if not role.invisible:
                members.extend([[role.source, int(index)] for index in role.truth_indices])
        if members:
            targets[node_name] = members
    if not targets:
        raise ValueError(f"Topology {topology_path} has no visible truth roles")
    return targets


def build_visible_loader_config(experiment: DictConfig) -> DictConfig:
    """Adapt the JER data config to the mature fixed-token visible loader."""
    data = _load_data(_path(experiment.data_config))
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

    counts = {
        str(source): int(spec.count)
        for source, spec in data.objects.items()
        if source != "GLOBAL"
    }
    lepton_counts = [
        count for source, count in counts.items() if source in {"LEPTON", "ELECTRON", "MUON"}
    ]
    dataset = {
        "input_h5_path": str(_path(data.source)),
        "max_jets": int(counts.get("JET", 0)),
        "max_leptons": max(lepton_counts, default=0),
        "include_met": "MET" in counts,
        "test_size": 0.2,
        "limit_events": None,
        "split": {"indices_path": str(_path(data.split)), "require_existing": True},
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
    visible = OmegaConf.create(
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
            "trainer": {
                "check_val_every_n_epoch": 1,
            },
            "train": {
                "compute_val_auc": False,
                "n_epochs": int(experiment.training.epochs),
                "checkpoint": {
                    "monitor": "val_loss",
                    "mode": "min",
                    "save_top_k": 1,
                    "save_last": False,
                },
            },
        }
    )
    expected_epochs = int(experiment.training.epochs)
    if int(visible.train.n_epochs) != expected_epochs:
        raise ValueError(
            "Visible-loader train.n_epochs must mirror JER training.epochs: "
            f"found {visible.train.n_epochs}, expected {expected_epochs}."
        )
    return visible


def _resolve_recurrence(public, defaults) -> tuple[dict, bool, bool]:
    if public is None:
        recurrence = _plain(defaults)
        feedback_defaults = recurrence.pop("feedback")
        recurrence["feedback"] = feedback_defaults
        return recurrence, False, False

    _mapping(public, "model.assignment.recurrence")
    allowed = {"block_layers", "train_steps", "eval_steps", "reinject_visible", "feedback"}
    _unknown(public, allowed, "model.assignment.recurrence")
    feedback = public.get("feedback")
    if feedback not in {None, "soft"}:
        raise ValueError("model.assignment.recurrence.feedback must be soft or omitted")
    numeric = {key: _plain(value) for key, value in public.items() if key != "feedback"}
    recurrence = _plain(_merge(defaults, numeric))
    return recurrence, True, feedback == "soft"


def _resolve_invisible(public, defaults) -> dict:
    invisible = {
        "enabled": False,
        "formulation": "edm",
        "conditioning": {"assignment": "none", "allow_truth_oracle": False},
        "denoiser": _plain(defaults.network),
        "standard": _plain(defaults.standard),
        "edm": _plain(defaults.edm),
        "flow_matching": _plain(defaults.flow_matching),
    }
    if public is None:
        return invisible

    _mapping(public, "model.invisible")
    _unknown(public, {"method", "conditioning", "coupling", "network", "training"}, "model.invisible")
    _required(public, {"method"}, "model.invisible")
    method = str(public.method)
    method_map = {"ddim": "standard", "edm": "edm", "cfm": "flow_matching"}
    if method not in method_map:
        raise ValueError("model.invisible.method must be ddim, edm or cfm")
    conditioning = str(public.get("conditioning", "none"))
    if conditioning not in {"none", "soft", "hard", "hypothesis", "truth_oracle"}:
        raise ValueError(
            "model.invisible.conditioning must be none, soft, hard, hypothesis or truth_oracle"
        )

    invisible["enabled"] = True
    invisible["formulation"] = method_map[method]
    invisible["conditioning"] = {
        "assignment": "truth" if conditioning == "truth_oracle" else conditioning,
        "allow_truth_oracle": conditioning == "truth_oracle",
    }
    if public.get("network") is not None:
        invisible["denoiser"] = _plain(_merge(defaults.network, public.network))

    coupling = public.get("coupling")
    if method == "cfm":
        coupling = str(coupling if coupling is not None else defaults.flow_matching.coupling)
        if coupling not in {"independent", "minibatch_ot"}:
            raise ValueError("model.invisible.coupling must be independent or minibatch_ot")
        invisible["flow_matching"] = {"coupling": coupling}
    elif coupling is not None:
        raise ValueError("model.invisible.coupling is only meaningful for method=cfm")

    training = public.get("training")
    if training is not None:
        _mapping(training, "model.invisible.training")
        if method == "ddim":
            _unknown(training, {"training_steps"}, "model.invisible.training")
            invisible["standard"] = _plain(_merge(defaults.standard, training))
        elif method == "edm":
            _unknown(training, {"sigma_data", "p_mean", "p_std"}, "model.invisible.training")
            invisible["edm"] = _plain(_merge(defaults.edm, training))
        elif len(training):
            raise ValueError("CFM has no public method-specific training settings")
    return invisible


def _resolve_model(public, defaults) -> dict:
    _mapping(public, "model")
    allowed = {"encoder", "assignment", "invisible", "joint_flow", "c2", "compatibility"}
    _unknown(public, allowed, "model")
    _required(public, {"encoder"}, "model")
    encoder = public.encoder
    encoder_keys = {"d_model", "n_heads", "layers", "d_feedforward", "dropout"}
    _unknown(encoder, encoder_keys, "model.encoder")
    _required(encoder, encoder_keys, "model.encoder")

    joint_public = public.get("joint_flow")
    if joint_public is not None:
        for name in ("assignment", "invisible", "c2", "compatibility"):
            if public.get(name) is not None:
                raise ValueError(f"model.{name} must be omitted when model.joint_flow is used")
        _mapping(joint_public, "model.joint_flow")
        _unknown(joint_public, {"recurrent", "block_layers", "train_steps"}, "model.joint_flow")
        joint = _plain(_merge(defaults.joint_flow, joint_public))
        recurrence = defaults.assignment.recurrence
        return {
            "mode": "joint_flow",
            "visible": _plain(encoder),
            "assignment": {
                "formulation": "flow_matching",
                "reasoning": {
                    "enabled": False,
                    "block_layers": int(recurrence.block_layers),
                    "train_steps": list(recurrence.train_steps),
                    "eval_steps": int(recurrence.eval_steps),
                    "fixed_input_reinjection": bool(recurrence.reinject_visible),
                },
                "feedback": {
                    "enabled": False,
                    "start_step": int(recurrence.feedback.start_step),
                    "gate_initial_bias": float(recurrence.feedback.gate_initial_bias),
                },
            },
            "invisible": {
                "enabled": True,
                "formulation": "flow_matching",
                "conditioning": {"assignment": "none", "allow_truth_oracle": False},
                "denoiser": _plain(defaults.invisible.network),
                "standard": _plain(defaults.invisible.standard),
                "edm": _plain(defaults.invisible.edm),
                "flow_matching": _plain(defaults.invisible.flow_matching),
            },
            "joint_flow": {
                "recurrent_state": bool(joint.get("recurrent", False)),
                "block_layers": int(joint["block_layers"]),
                "train_steps": list(joint["train_steps"]),
            },
            "coupling": {
                "invisible_to_assignment": {
                    "enabled": False,
                    "posterior_samples": int(defaults.c2.posterior_samples),
                    "gate_initial_bias": float(defaults.c2.gate_initial_bias),
                }
            },
            "compatibility": {
                "enabled": False,
                "include_invisible": bool(defaults.compatibility.include_invisible),
            },
        }

    assignment_public = public.get("assignment")
    if assignment_public is None:
        raise ValueError("Factorized models require model.assignment")
    _mapping(assignment_public, "model.assignment")
    _unknown(assignment_public, {"method", "recurrence"}, "model.assignment")
    _required(assignment_public, {"method"}, "model.assignment")
    method = str(assignment_public.method)
    if method not in {"discriminative", "dfm"}:
        raise ValueError("model.assignment.method must be discriminative or dfm")

    recurrence, recurrent, feedback = _resolve_recurrence(
        assignment_public.get("recurrence"), defaults.assignment.recurrence
    )
    if method == "dfm" and recurrent:
        raise ValueError("Factorized DFM has its own time-conditioned reasoner; omit recurrence")

    invisible = _resolve_invisible(public.get("invisible"), defaults.invisible)
    if method == "dfm" and invisible["enabled"]:
        raise ValueError("Factorized DFM is assignment-only; omit model.invisible or use joint_flow")

    c2_public = public.get("c2")
    c2 = _plain(defaults.c2)
    c2_enabled = c2_public is not None
    if c2_enabled:
        if method != "discriminative" or not recurrent or not invisible["enabled"]:
            raise ValueError("model.c2 requires recurrent discriminative assignment and invisible inference")
        _mapping(c2_public, "model.c2")
        _unknown(c2_public, {"posterior_samples", "gate_initial_bias"}, "model.c2")
        c2 = _plain(_merge(defaults.c2, c2_public))

    compatibility_public = public.get("compatibility")
    compatibility = _plain(defaults.compatibility)
    compatibility_enabled = compatibility_public is not None
    if compatibility_enabled:
        if method != "discriminative" or not invisible["enabled"]:
            raise ValueError("model.compatibility requires factorized invisible inference")
        _mapping(compatibility_public, "model.compatibility")
        _unknown(compatibility_public, {"include_invisible"}, "model.compatibility")
        compatibility = _plain(_merge(defaults.compatibility, compatibility_public))

    return {
        "mode": "factorized",
        "visible": _plain(encoder),
        "assignment": {
            "formulation": "flow_matching" if method == "dfm" else "discriminative",
            "reasoning": {
                "enabled": recurrent,
                "block_layers": int(recurrence["block_layers"]),
                "train_steps": list(recurrence["train_steps"]),
                "eval_steps": int(recurrence["eval_steps"]),
                "fixed_input_reinjection": bool(recurrence["reinject_visible"]),
            },
            "feedback": {
                "enabled": feedback,
                "start_step": int(recurrence["feedback"]["start_step"]),
                "gate_initial_bias": float(recurrence["feedback"]["gate_initial_bias"]),
            },
        },
        "invisible": invisible,
        "joint_flow": {
            "recurrent_state": False,
            "block_layers": int(defaults.joint_flow.block_layers),
            "train_steps": list(defaults.joint_flow.train_steps),
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
    """Resolve one compact experiment; ``source_path`` is retained for provenance callers."""
    del source_path
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
    model = _resolve_model(public.model, defaults.model)

    prediction_public = public.prediction
    _mapping(prediction_public, "prediction")
    prediction_keys = {
        "seed", "top_k", "samples", "steps", "solver", "sample_chunk_size",
        "nfe", "sigma_min", "sigma_max", "rho", "compatibility_weight",
    }
    _unknown(prediction_public, prediction_keys, "prediction")
    prediction = _merge(defaults.prediction, prediction_public)
    standard_steps = (
        int(prediction_public.steps)
        if "steps" in prediction_public
        else int(defaults.prediction.standard_steps)
    )

    training_public = public.training
    _mapping(training_public, "training")
    training_keys = {
        "learning_rate", "weight_decay", "epochs", "precision", "gradient_clip",
        "accumulate_grad_batches", "trainable", "loss_weights", "scheduler",
        "early_stopping", "output",
    }
    _unknown(training_public, training_keys, "training")
    training = _merge(defaults.training, training_public)

    if training_public.get("trainable") is None:
        trainable = ["visible", "assignment"]
        if model["invisible"]["enabled"]:
            trainable.append("invisible")
        if model["coupling"]["invisible_to_assignment"]["enabled"]:
            trainable.append("c2")
        if model["compatibility"]["enabled"]:
            trainable.append("compatibility")
    else:
        trainable = [str(value) for value in training_public.trainable]
    if not trainable or len(set(trainable)) != len(trainable):
        raise ValueError("training.trainable must contain unique component names")
    unknown_trainable = sorted(set(trainable) - TRAINABLE_COMPONENTS)
    if unknown_trainable:
        raise ValueError(f"Unknown training.trainable components: {unknown_trainable}")
    if "c2" in trainable and "assignment" not in trainable:
        raise ValueError("training.trainable=c2 requires assignment to be trainable")
    if model["mode"] == "joint_flow" and (("assignment" in trainable) != ("invisible" in trainable)):
        raise ValueError("joint_flow assignment and invisible dynamics must be trainable/frozen together")

    loss_weights = {
        "assignment": 1.0 if "assignment" in trainable else 0.0,
        "invisible": 1.0 if "invisible" in trainable else 0.0,
        "compatibility": 1.0 if "compatibility" in trainable else 0.0,
    }
    if training_public.get("loss_weights") is not None:
        _mapping(training_public.loss_weights, "training.loss_weights")
        _unknown(training_public.loss_weights, set(loss_weights), "training.loss_weights")
        loss_weights.update(
            {key: float(value) for key, value in training_public.loss_weights.items()}
        )

    invisible_truth = data_config.invisible_truth
    return OmegaConf.create(
        {
            "experiment": _plain(public.experiment),
            "topology_config": str(topology_path),
            "data_config": str(data_path),
            "data": {
                "tensor_cache": None if data_config.tensor_cache is None else str(_path(data_config.tensor_cache)),
                "max_events": {
                    "train": data_public.events.train,
                    "val": data_public.events.validation,
                    "test": data_public.events.test,
                },
                "max_assignment_states": int(data_public.max_assignment_states),
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
                    "standard": {"steps": standard_steps},
                    "edm": {
                        "sigma_min": float(prediction.sigma_min),
                        "sigma_max": float(prediction.sigma_max),
                        "rho": float(prediction.rho),
                        "nfe": int(prediction.nfe),
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
    )


def load_public_experiment(
    path: str | Path,
    overrides: list[str] | tuple[str, ...] | None = None,
) -> DictConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration does not exist: {path}")
    public = OmegaConf.load(path)
    if overrides:
        public = OmegaConf.merge(public, OmegaConf.from_dotlist(list(overrides)))
    return resolve_public_experiment(public, source_path=path)
