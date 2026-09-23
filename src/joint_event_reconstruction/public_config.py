"""Resolve compact human-facing JER configs into the explicit internal contract."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from joint_event_reconstruction.topology.compiler import load_topology
from joint_event_reconstruction.models.invisible.mass_shell import (
    mass_shell_config as canonical_mass_shell_config,
)
from joint_event_reconstruction.runtime_paths import runtime_paths


REPO_ROOT = runtime_paths().source_root
DEFAULTS_PATH = REPO_ROOT / "configs/defaults.yaml"
PUBLIC_TOP_KEYS = {
    "experiment", "topology", "data", "inputs", "model", "inference", "training"
}
TRAINABLE_COMPONENTS = {
    "visible", "assignment", "invisible", "point", "parent", "partonic",
    "queries", "c2", "compatibility",
}


def _path(value: object, *, data_path: bool = False) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    roots = runtime_paths()
    root = roots.project_root if data_path else roots.source_root
    return (root / path).resolve()


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


def _load_data(value) -> DictConfig:
    """Validate and normalise the compact public ``data`` + ``inputs`` contract."""
    data = OmegaConf.load(value) if isinstance(value, Path) else value
    _mapping(data, "input definition")
    obsolete = {
        "files", "loader", "objects", "event_features", "truth", "tensor_cache"
    }
    found = sorted(obsolete & set(data))
    if found:
        raise ValueError(
            f"Obsolete public data setting(s) {found}; use data.input/data.cache and inputs"
        )
    _unknown(data, {"input", "cache", "transforms", "collections"}, "input definition")
    _required(data, {"input", "cache", "transforms", "collections"}, "input definition")
    for name, transform in data.transforms.items():
        _transform_name(transform, f"inputs.transforms.{name}")
    if not data.collections:
        raise ValueError("inputs must contain at least one object collection")
    for source, spec in data.collections.items():
        _mapping(spec, f"inputs.{source}")
        if source == "GLOBAL":
            _unknown(spec, {"features"}, "inputs.GLOBAL")
            _required(spec, {"features"}, "inputs.GLOBAL")
        else:
            if "max_count" in spec:
                raise ValueError(f"inputs.{source}.max_count is obsolete; use count")
            _unknown(spec, {"count", "features"}, f"inputs.{source}")
            _required(spec, {"count", "features"}, f"inputs.{source}")
            if int(spec.count) <= 0:
                raise ValueError(f"inputs.{source}.count must be positive")
        for feature in spec.features:
            if feature not in data.transforms:
                raise ValueError(
                    f"input collection {source} requests unknown feature {feature!r}"
                )
    return data


def _transform_name(transform, path: str) -> str:
    if isinstance(transform, str):
        if transform not in {"none", "log", "sincos", "standard", "pid"}:
            raise ValueError(f"{path} must be none, log, sincos, standard, pid or [minmax, MIN, MAX]")
        return transform
    values = list(transform) if OmegaConf.is_list(transform) or isinstance(transform, (list, tuple)) else None
    if values is None or len(values) != 3 or str(values[0]) != "minmax":
        raise ValueError(f"{path} must be [minmax, MIN, MAX]")
    lower, upper = float(values[1]), float(values[2])
    if not lower < upper:
        raise ValueError(f"{path} requires min < max")
    return f"minmax:min={lower:g},max={upper:g}"


def _truth_bindings(topology_path: Path) -> dict[str, dict]:
    """Maintained canonical storage adapters; never part of experiment YAML."""
    adapters = {
        "ttbar2L": {
            "top_e_b": ("JET", 1), "top_e_lepton": ("ELECTRON", 1),
            "top_mu_b": ("JET", 2), "top_mu_lepton": ("MUON", 1),
            "top_e_neutrino": ("NEUTRINO", 0), "top_mu_neutrino": ("NEUTRINO", 1),
        },
        "ttbar1L": {
            "leptonic_b": ("JET", 1), "hadronic_b": ("JET", 2),
            "hadronic_q1": ("JET", 3), "hadronic_q2": ("JET", 4),
            "charged_lepton": ("LEPTON", 1), "leptonic_neutrino": ("NEUTRINO", 0),
        },
        "ttH1L": {
            "hadronic_b": ("JET", 1), "hadronic_q1": ("JET", 2),
            "hadronic_q2": ("JET", 3), "leptonic_b": ("JET", 4),
            "higgs_b1": ("JET", 5), "higgs_b2": ("JET", 6),
            "charged_lepton": ("LEPTON", 1), "leptonic_neutrino": ("NEUTRINO", 0),
        },
        # These adapters are used only by the file-free topology contracts in
        # smoke_synthetic.py.  They make the synthetic topology names obey the
        # same semantic source/index contract as the maintained real samples.
        "synthetic_contract": {
            "pair_jet_1": ("JET", 1), "pair_jet_2": ("JET", 2),
            "bachelor_jet": ("JET", 3), "other_jet": ("JET", 4),
            "charged_lepton": ("LEPTON", 1), "invisible_1": ("NEUTRINO", 0),
        },
        "synthetic_multinu": {
            "jet_1": ("JET", 1), "jet_2": ("JET", 2), "jet_3": ("JET", 3),
            "jet_4": ("JET", 4), "jet_5": ("JET", 5), "jet_6": ("JET", 6),
            "charged_lepton": ("LEPTON", 1),
            "invisible_1": ("NEUTRINO", 0), "invisible_2": ("NEUTRINO", 1),
            "invisible_3": ("NEUTRINO", 2),
        },
    }
    name = topology_path.stem
    if name not in adapters:
        raise ValueError(f"No maintained dataset truth adapter for topology {name!r}")
    return {key: {"source": source, "index": index} for key, (source, index) in adapters[name].items()}


def _visible_targets(topology_path: Path) -> dict:
    """Minimal loader targets used only to request raw LABELS for topology sources."""
    spec = load_topology(topology_path, _truth_bindings(topology_path))
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
    data = _load_data(experiment.input_definition)
    topology_path = _path(experiment.topology_config)
    observables = []
    for source, spec in data.collections.items():
        if source == "GLOBAL":
            continue
        features = [str(name) for name in spec.features]
        kinematics = _canonical_kinematics(source, features)
        entry = {
            "INPUTS": str(source),
            "variables": features,
            "scalers": [_transform_name(data.transforms[name], f"inputs.transforms.{name}") for name in features],
            "feature_names": features,
            "kinematics": kinematics,
        }
        entry["num_particles"] = int(spec.count)
        observables.append(entry)
    global_spec = data.collections.get("GLOBAL")
    if global_spec is not None:
        features = [str(name) for name in global_spec.features]
        observables.append(
            {
                "INPUTS": "GLOBAL",
                "variables": features,
                "scalers": [_transform_name(data.transforms[name], f"inputs.transforms.{name}") for name in features],
                "feature_names": features,
            }
        )

    counts = {
        str(source): int(spec.count)
        for source, spec in data.collections.items() if source != "GLOBAL"
    }
    lepton_counts = [
        count for source, count in counts.items() if source in {"LEPTON", "ELECTRON", "MUON"}
    ]
    dataset = {
        "input_h5_path": str(_path(data.input, data_path=True)),
        "max_jets": int(counts.get("JET", 0)),
        "max_leptons": max(lepton_counts, default=0),
        "include_met": "MET" in counts,
        "test_size": float(experiment.data.test_size),
        "limit_events": None,
        "split": {
            "indices_path": experiment.data.split.indices_path,
            "require_existing": bool(experiment.data.split.require_existing),
        },
        "padding_value": 0.0,
        "train_batch_size": int(experiment.training.batch_size),
        "val_batch_size": int(experiment.training.batch_size),
        "test_batch_size": int(experiment.training.batch_size),
        "train_num_workers": int(experiment.training.workers),
        "val_num_workers": int(experiment.training.workers),
        "test_num_workers": int(experiment.training.workers),
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
        "cache": {
            "enabled": data.cache is not None,
            "path": data.cache,
            "build_chunk_size": 131072,
            "build_workers": 1,
            "rebuild": bool(experiment.data.get("force_rebuild_cache", False)),
        },
        "observables": observables,
    }
    visible = OmegaConf.create(
        {
            "general": {
                "model": "transformer",
                "topology": str(experiment.experiment.name),
                "mode": "train",
                "device": "cuda",
                # The mature loader uses this only for split construction.  Keep
                # split identity independent of model/training randomness.
                "seed": int(experiment.data.subset_seed),
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


def _canonical_kinematics(source: str, features: list[str]) -> dict:
    """Return one of the explicit maintained physical-vector layouts."""
    names = set(features)
    if source == "MET":
        if {"pt", "phi"} <= names:
            return {"representation": "transverse_pseudovector", "pt": "pt", "phi": "phi"}
        if {"pt", "sin_phi", "cos_phi"} <= names:
            return {"representation": "transverse_pseudovector", "pt": "pt", "sin_phi": "sin_phi", "cos_phi": "cos_phi"}
    elif {"e", "pt", "eta", "phi"} <= names:
        return {"energy": "e", "pt": "pt", "eta": "eta", "phi": "phi"}
    elif {"e", "pt", "eta", "sin_phi", "cos_phi"} <= names:
        return {"energy": "e", "pt": "pt", "eta": "eta", "sin_phi": "sin_phi", "cos_phi": "cos_phi"}
    raise ValueError(
        f"input collection {source} does not provide a maintained canonical physical layout; "
        f"features={features}"
    )


def _resolve_recurrence(public, defaults) -> tuple[dict, bool, bool]:
    if public is None:
        recurrence = _plain(defaults)
        feedback_defaults = recurrence.pop("feedback")
        recurrence["feedback"] = feedback_defaults
        return recurrence, False, False

    _mapping(public, "model.assignment.refinement")
    allowed = {"layers", "train_iterations", "inference_iterations", "reinject_encoder_context", "feedback"}
    _unknown(public, allowed, "model.assignment.refinement")
    feedback = public.get("feedback")
    if feedback not in {None, "soft"}:
        raise ValueError("model.assignment.refinement.feedback must be soft or omitted")
    recurrence = _plain(defaults)
    if public.get("layers") is not None:
        recurrence["block_layers"] = int(public.layers)
    recurrence["train_steps"] = list(public.get("train_iterations", recurrence["train_steps"]))
    recurrence["eval_steps"] = int(public.get("inference_iterations", recurrence["eval_steps"]))
    if public.get("reinject_encoder_context") is not None:
        recurrence["reinject_visible"] = bool(public.reinject_encoder_context)
    return recurrence, True, feedback == "soft"


def _coordinate_config(coordinate_type: str) -> dict:
    return {
        "type": coordinate_type,
        "version": "v1",
        "basis_convention": (
            "helmert_orthonormal_v1"
            if coordinate_type in {"met_residual", "met_exact"}
            else None
        ),
    }


def _mass_shell_config(public=None) -> dict:
    """Validate and normalise one shared or per-role mass-shell declaration."""
    if public is None:
        return canonical_mass_shell_config()
    _mapping(public, "model.invisible.mass_shell")
    _unknown(public, {"enforcement", "mass", "roles"}, "model.invisible.mass_shell")
    mass = public.get("mass")
    if mass is not None:
        _mapping(mass, "model.invisible.mass_shell.mass")
        _unknown(
            mass,
            {"mode", "value", "parameterization"},
            "model.invisible.mass_shell.mass",
        )
    roles = public.get("roles", {})
    _mapping(roles, "model.invisible.mass_shell.roles")
    for role, role_config in roles.items():
        _mapping(role_config, f"model.invisible.mass_shell.roles.{role}")
        _unknown(
            role_config,
            {"enforcement", "mass"},
            f"model.invisible.mass_shell.roles.{role}",
        )
        role_mass = role_config.get("mass")
        if role_mass is not None:
            _mapping(role_mass, f"model.invisible.mass_shell.roles.{role}.mass")
            _unknown(
                role_mass,
                {"mode", "value", "parameterization"},
                f"model.invisible.mass_shell.roles.{role}.mass",
            )
    # The physics-level class performs the numerical checks (finite values,
    # positive generated mass transforms and exact-mode compatibility).
    return canonical_mass_shell_config(public)


def _invisible_backbone_config(public, defaults) -> dict:
    if public is None:
        return {"type": "transformer"}
    _mapping(public, "model.invisible.generator.backbone")
    _required(public, {"type"}, "model.invisible.generator.backbone")
    backend = str(public.type)
    if backend == "transformer":
        _unknown(public, {"type"}, "model.invisible.generator.backbone")
        return {"type": backend}
    if backend != "lgatr":
        raise ValueError("Invisible generator backbone must be transformer or lgatr")
    _unknown(
        public,
        {
            "type",
            "layers",
            "n_heads",
            "hidden",
            "dropout",
            "momentum_scale_gev",
            "symmetry_breaking",
        },
        "model.invisible.generator.backbone",
    )
    selected = _plain(_merge(defaults.encoder.lgatr, public))
    return {
        "type": backend,
        "layers": int(selected["layers"]),
        "n_heads": int(selected["n_heads"]),
        "hidden": selected["hidden"],
        "dropout": float(selected["dropout"]),
        "momentum_scale_gev": float(selected["momentum_scale_gev"]),
        "symmetry_breaking": selected["symmetry_breaking"],
    }


def _resolve_invisible(public, defaults, *, backbone_defaults=None) -> dict:
    invisible = {
        "enabled": False,
        "formulation": "edm",
        "conditioning": {"assignment": "none", "allow_truth_oracle": False},
        "coordinates": _coordinate_config(str(defaults.coordinates.type)),
        "mass_shell": _mass_shell_config(getattr(defaults, "mass_shell", None)),
        "backbone": {"type": "transformer"},
        "denoiser": _plain(defaults.network),
        "standard": _plain(defaults.standard),
        "edm": _plain(defaults.edm),
        "flow_matching": {
            "coupling": {
                "method": str(defaults.flow_matching.coupling.type),
                "condition_weight": float(defaults.flow_matching.coupling.condition_weight),
            }
        },
    }
    if public is None:
        return invisible

    _mapping(public, "model.invisible")
    _unknown(
        public,
        {"generator", "conditioning", "coordinates", "network", "training", "mass_shell"},
        "model.invisible",
    )
    if "method" in public:
        raise ValueError("model.invisible.method is obsolete; use model.invisible.generator")
    invisible["mass_shell"] = _mass_shell_config(public.get("mass_shell"))
    if "generator" not in public:
        if set(public) == {"mass_shell"}:
            return invisible
        _required(public, {"generator"}, "model.invisible")
    _mapping(public.generator, "model.invisible.generator")
    _unknown(
        public.generator,
        {"type", "coupling", "network", "backbone"},
        "model.invisible.generator",
    )
    _required(public.generator, {"type"}, "model.invisible.generator")
    method = str(public.generator.type)
    method_map = {"ddim": "standard", "edm": "edm", "cfm": "flow_matching"}
    if method not in method_map:
        raise ValueError("model.invisible.generator must be ddim, edm or cfm")
    conditioning_public = public.get("conditioning", {"assignment": "none"})
    _mapping(conditioning_public, "model.invisible.conditioning")
    _unknown(conditioning_public, {"assignment"}, "model.invisible.conditioning")
    conditioning = str(conditioning_public.get("assignment", "none"))
    if conditioning not in {"none", "soft", "hard", "hypothesis", "truth_oracle"}:
        raise ValueError(
            "model.invisible.conditioning.assignment must be none, soft, hard, hypothesis or truth_oracle"
        )

    invisible["enabled"] = True
    invisible["formulation"] = method_map[method]
    invisible["conditioning"] = {
        "assignment": "truth" if conditioning == "truth_oracle" else conditioning,
        "allow_truth_oracle": conditioning == "truth_oracle",
    }
    invisible["backbone"] = _invisible_backbone_config(
        public.generator.get("backbone"),
        defaults if backbone_defaults is None else backbone_defaults,
    )
    coordinates = public.get("coordinates", defaults.coordinates)
    _mapping(coordinates, "model.invisible.coordinates")
    _unknown(coordinates, {"type"}, "model.invisible.coordinates")
    coordinate_type = str(coordinates.get("type", "cartesian"))
    if coordinate_type not in {"cartesian", "met_residual", "met_exact"}:
        raise ValueError(
            "model.invisible.coordinates.type must be cartesian, met_residual or met_exact"
        )
    invisible["coordinates"] = _coordinate_config(coordinate_type)
    network = public.generator.get("network", public.get("network"))
    if network is not None:
        invisible["denoiser"] = _plain(_merge(defaults.network, network))

    coupling = public.generator.get("coupling")
    if method == "cfm":
        coupling = coupling if coupling is not None else defaults.flow_matching.coupling
        _mapping(coupling, "model.invisible.coupling")
        if "method" in coupling:
            raise ValueError("model.invisible.coupling.method is obsolete; use type")
        _unknown(coupling, {"type", "condition_weight"}, "model.invisible.coupling")
        method_name = str(coupling.get("type", "independent"))
        if method_name not in {"independent", "minibatch_ot", "conditional_ot"}:
            raise ValueError(
                "model.invisible.coupling.method must be independent, minibatch_ot or conditional_ot"
            )
        condition_weight = float(coupling.get("condition_weight", 0.0))
        if method_name != "conditional_ot" and condition_weight != 0.0:
            raise ValueError("condition_weight must be zero unless coupling.type=conditional_ot")
        invisible["flow_matching"] = {
            "coupling": {"method": method_name, "condition_weight": condition_weight}
        }
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


def _resolve_visible(public, latent: dict, defaults) -> dict:
    _mapping(public, "model.encoder")
    if "d_model" in public:
        raise ValueError(
            "model.encoder.d_model is obsolete; move shared latent settings to "
            "model.latent"
        )
    backend = str(public.get("type", ""))
    common = {"type", "representation", "relations", "layers", "n_heads", "d_feedforward", "dropout"}
    if "scalar_features" in public:
        raise ValueError(
            "model.encoder.scalar_features is obsolete; inputs are declared once in inputs"
        )
    if "spurions" in public:
        raise ValueError(
            "model.encoder.spurions is obsolete; use model.encoder.symmetry_breaking"
        )
    if "inputs" in public:
        raise ValueError("model.encoder.inputs is obsolete; declare inputs once at top level")
    _required(public, {"type"}, "model.encoder")
    if backend == "transformer":
        _unknown(public, common, "model.encoder (transformer)")
        selected = _plain(_merge(defaults.encoder.transformer, public))
        representation = str(public.get("representation", "features"))
        if representation not in {"features", "four_vector"}:
            raise ValueError("transformer representation must be features or four_vector")
        relations_public = public.get("relations", "none")
        relation_type = str(relations_public.get("type")) if OmegaConf.is_config(relations_public) or isinstance(relations_public, dict) else str(relations_public)
        if relation_type not in {"none", "physics"}:
            raise ValueError("transformer relations must be none or physics")
        if relation_type == "physics" and representation != "four_vector":
            raise ValueError("transformer relations=physics requires representation=four_vector")
        if relation_type == "physics":
            backend = "relational_transformer"
            selected = _plain(_merge(defaults.encoder.relational_transformer, public))
        relations = _plain(defaults.encoder.relational_transformer.relations)
        if relation_type == "physics" and (OmegaConf.is_config(relations_public) or isinstance(relations_public, dict)):
            relations = _plain(_merge(relations, relations_public))
            relations.pop("type", None)
        if relation_type == "physics":
            _unknown(relations, {"enabled", "hidden_dim", "features"}, "model.encoder.relations")
        allowed_features = {
            "delta_eta", "sin_delta_phi", "cos_delta_phi", "log_delta_r",
            "minkowski_dot", "log_pair_mass",
        }
        if relation_type == "physics" and set(relations["features"]) != allowed_features:
            raise ValueError(
                "relational_transformer relations.features must contain the six "
                "maintained physical relations exactly"
            )
    elif backend == "lgatr":
        allowed = common | {"hidden", "momentum_scale_gev", "symmetry_breaking"}
        _unknown(public, allowed, "model.encoder (lgatr)")
        selected = _plain(_merge(defaults.encoder.lgatr, public))
    elif backend == "lgatr_slim":
        allowed = common | {"hidden", "momentum_scale_gev", "symmetry_breaking"}
        _unknown(public, allowed, "model.encoder (lgatr_slim)")
        selected = _plain(_merge(defaults.encoder.lgatr_slim, public))
    else:
        raise ValueError(
            "model.encoder.type must be transformer, lgatr or lgatr_slim; "
            "use transformer relations=physics for the relational Transformer"
        )
    representation = "four_vector" if backend in {"relational_transformer", "lgatr", "lgatr_slim"} else str(public.get("representation", "features"))
    result = {
        "type": backend,
        "d_model": int(latent["d_model"]),
        "inputs": {"kinematics": "transformed_features" if representation == "features" else "four_vector"},
        "representation": representation,
        "relations_type": "physics" if backend == "relational_transformer" else "none",
    }
    if backend in {"transformer", "relational_transformer"}:
        result.update(
            {
                "n_heads": int(selected.get("n_heads", latent["n_heads"])),
                "layers": int(selected["layers"]),
                "d_feedforward": int(
                    selected.get("d_feedforward", latent["d_feedforward"])
                ),
                "dropout": float(selected.get("dropout", latent["dropout"])),
            }
        )
        if backend == "relational_transformer":
            result.update(
                {
                    "momentum_scale_gev": float(selected["momentum_scale_gev"]),
                    "relations": relations,
                }
            )
    else:
        _unknown(selected["symmetry_breaking"], {"beam", "time"}, "model.encoder.symmetry_breaking")
        if str(selected["symmetry_breaking"]["beam"]) not in {"xyplane", "none"}:
            raise ValueError("model.encoder.symmetry_breaking.beam must be xyplane or none")
        result.update(
            {
                "layers": int(selected["layers"]),
                "n_heads": int(selected["n_heads"]),
                "dropout": float(selected["dropout"]),
                "momentum_scale_gev": float(selected["momentum_scale_gev"]),
                "hidden": selected["hidden"],
                "symmetry_breaking": selected["symmetry_breaking"],
            }
        )
    return result


def _resolve_optional_heads(public) -> dict:
    """Resolve optional heads without creating topology-specific output names."""
    result = {
        "point_invisible": {"enabled": False},
        "parent_four_vector": {"enabled": False, "roles": [], "target_path": None},
        "partonic": {
            "enabled": False,
            "roles": [],
            "target_path": None,
            "truth_bindings": {},
            "targets": [],
        },
    }
    if public is None:
        return result
    _mapping(public, "model.heads")
    _unknown(public, {"point_invisible", "parent_four_vector", "partonic"}, "model.heads")
    point = public.get("point_invisible")
    if point is not None:
        _mapping(point, "model.heads.point_invisible")
        _unknown(point, {"enabled"}, "model.heads.point_invisible")
        result["point_invisible"]["enabled"] = bool(point.get("enabled", False))
    parent = public.get("parent_four_vector")
    if parent is not None:
        _mapping(parent, "model.heads.parent_four_vector")
        _unknown(parent, {"enabled", "roles", "target_path"}, "model.heads.parent_four_vector")
        enabled = bool(parent.get("enabled", False))
        roles = [str(role) for role in parent.get("roles", [])]
        target_path = parent.get("target_path")
        if enabled and (not roles or target_path is None):
            raise ValueError(
                "model.heads.parent_four_vector requires roles and target_path when enabled"
            )
        result["parent_four_vector"] = {
            "enabled": enabled,
            "roles": roles,
            "target_path": None if target_path is None else str(target_path),
        }
    partonic = public.get("partonic")
    if partonic is not None:
        _mapping(partonic, "model.heads.partonic")
        _unknown(
            partonic,
            {"enabled", "targets", "target_path"},
            "model.heads.partonic",
        )
        enabled = bool(partonic.get("enabled", False))
        targets = partonic.get("targets", [])
        if OmegaConf.is_config(targets):
            targets = OmegaConf.to_container(targets, resolve=True)
        if not isinstance(targets, (list, tuple)) or not targets:
            if enabled:
                raise ValueError("model.heads.partonic requires non-empty targets when enabled")
            targets = []
        normalised = []
        roles = []
        bindings = {}
        for target in targets:
            _mapping(target, "model.heads.partonic.targets[]")
            _unknown(target, {"role", "truth_binding"}, "model.heads.partonic.targets[]")
            role = str(target.get("role", "")).strip()
            binding = str(target.get("truth_binding", "")).strip()
            if not role or not binding:
                raise ValueError("Every partonic target requires role and truth_binding")
            if role in roles:
                raise ValueError(f"Duplicate partonic target role {role!r}")
            roles.append(role)
            bindings[role] = binding
            normalised.append({"role": role, "truth_binding": binding})
        target_path = partonic.get("target_path")
        if enabled and target_path is None:
            raise ValueError("Enabled model.heads.partonic requires target_path")
        result["partonic"] = {
            "enabled": enabled,
            "roles": roles,
            "target_path": None if target_path is None else str(target_path),
            "truth_bindings": bindings,
            "targets": normalised,
        }
    if result["point_invisible"]["enabled"] and result["parent_four_vector"]["enabled"]:
        # Both are allowed; their losses remain independently weighted.
        pass
    return result


def _resolve_topology_queries(public, latent: dict) -> dict:
    """Resolve role metadata without introducing a second topology grammar."""
    result = {
        "enabled": False,
        "interaction": "full",
        "layers": 1,
        "n_heads": int(latent["n_heads"]),
        "dropout": 0.0,
        "roles": [],
        "relation_edges": [],
    }
    if public is None:
        return result
    _mapping(public, "model.topology_queries")
    _unknown(
        public,
        {"enabled", "roles", "interaction", "layers", "n_heads", "dropout", "relations"},
        "model.topology_queries",
    )
    enabled = bool(public.get("enabled", False))
    raw_roles = public.get("roles", [])
    if OmegaConf.is_config(raw_roles):
        raw_roles = OmegaConf.to_container(raw_roles, resolve=True)
    if enabled and not raw_roles:
        raise ValueError("Enabled model.topology_queries requires non-empty roles")
    roles = []
    names = []
    for raw in raw_roles:
        _mapping(raw, "model.topology_queries.roles[]")
        _unknown(raw, {"name", "node", "kind", "related"}, "model.topology_queries.roles[]")
        name = str(raw.get("name", "")).strip()
        if not name or name in names:
            raise ValueError("Topology query names must be non-empty and unique")
        names.append(name)
        roles.append({
            "name": name,
            "node": None if raw.get("node") is None else str(raw.get("node")),
            "kind": str(raw.get("kind", "role")),
            "related": [str(value) for value in raw.get("related", [])],
        })
    interaction = str(public.get("interaction", "full"))
    if interaction not in {"full", "topology", "topology_plus_global"}:
        raise ValueError("model.topology_queries.interaction is invalid")
    relation_edges = []
    raw_relations = public.get("relations", [])
    if OmegaConf.is_config(raw_relations):
        raw_relations = OmegaConf.to_container(raw_relations, resolve=True)
    for relation in raw_relations:
        if not isinstance(relation, (list, tuple)) or len(relation) != 2:
            raise ValueError("model.topology_queries.relations entries must be [role_a, role_b]")
        left, right = str(relation[0]), str(relation[1])
        if left not in names or right not in names:
            raise ValueError(f"Topology-query relation refers to an undeclared role: {relation}")
        relation_edges.append([names.index(left), names.index(right)])
    width = int(public.get("n_heads", latent["n_heads"]))
    if int(latent["d_model"]) % width:
        raise ValueError("model.topology_queries.n_heads must divide model.latent.d_model")
    dropout = float(public.get("dropout", 0.0))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("model.topology_queries.dropout must lie in [0,1)")
    return {
        "enabled": enabled,
        "interaction": interaction,
        "layers": int(public.get("layers", 1)),
        "n_heads": width,
        "dropout": dropout,
        "roles": roles,
        "relation_edges": relation_edges,
    }


def _resolve_model(public, defaults) -> dict:
    _mapping(public, "model")
    allowed = {
        "mode", "latent", "encoder", "assignment", "invisible", "joint",
        "feedback", "compatibility", "heads", "topology_queries",
    }
    _unknown(public, allowed, "model")
    for obsolete, replacement in {
        "joint_flow": "joint",
        "c2": "feedback.invisible_to_assignment",
    }.items():
        if obsolete in public:
            raise ValueError(f"model.{obsolete} is obsolete; use model.{replacement}")
    _required(public, {"mode", "latent", "encoder"}, "model")
    _mapping(public.latent, "model.latent")
    latent_keys = {"d_model", "n_heads", "d_feedforward", "dropout"}
    _unknown(public.latent, latent_keys, "model.latent")
    _required(public.latent, latent_keys, "model.latent")
    latent = _plain(public.latent)
    encoder = _resolve_visible(public.encoder, latent, defaults)
    heads = _resolve_optional_heads(public.get("heads"))
    topology_queries = _resolve_topology_queries(public.get("topology_queries"), latent)

    mode = str(public.mode)
    if mode not in {"factorized", "joint"}:
        raise ValueError("model.mode must be factorized or joint")
    joint_public = public.get("joint")
    if mode == "joint":
        if any(value["enabled"] for value in heads.values()):
            raise ValueError("Optional point/parent heads currently require model.mode=factorized")
        if joint_public is None:
            raise ValueError("model.mode=joint requires model.joint")
        for name in ("assignment", "invisible", "feedback", "compatibility"):
            if public.get(name) is not None:
                raise ValueError(f"model.{name} must be omitted when model.mode=joint")
        _mapping(joint_public, "model.joint")
        _unknown(
            joint_public,
            {"recurrent", "block_layers", "train_steps", "coordinates", "coupling", "mass_shell"},
            "model.joint",
        )
        joint = _plain(_merge(defaults.joint_flow, joint_public))
        coordinates = joint["coordinates"]
        _unknown(coordinates, {"type"}, "model.joint.coordinates")
        coordinate_type = str(coordinates["type"])
        if coordinate_type not in {"cartesian", "met_residual", "met_exact"}:
            raise ValueError("model.joint.coordinates.type is invalid")
        coupling = joint["coupling"]
        if "method" in coupling:
            raise ValueError("model.joint.coupling.method is obsolete; use type")
        _unknown(coupling, {"type", "condition_weight"}, "model.joint.coupling")
        coupling_method = str(coupling["type"])
        coupling_weight = float(coupling.get("condition_weight", 0.0))
        if coupling_method not in {"independent", "minibatch_ot", "conditional_ot"}:
            raise ValueError("model.joint.coupling.type is invalid")
        if coupling_method != "conditional_ot" and coupling_weight != 0.0:
            raise ValueError("condition_weight must be zero unless coupling.type=conditional_ot")
        mass_shell = _mass_shell_config(
            joint_public.get("mass_shell", getattr(defaults.invisible, "mass_shell", None))
        )
        recurrence = defaults.assignment.recurrence
        return {
            "mode": "joint_flow",
            "latent": latent,
            "visible": encoder,
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
                "coordinates": _coordinate_config(coordinate_type),
                "mass_shell": mass_shell,
                "backbone": {"type": "transformer"},
                "denoiser": _plain(defaults.invisible.network),
                "standard": _plain(defaults.invisible.standard),
                "edm": _plain(defaults.invisible.edm),
                "flow_matching": {
                    "coupling": {
                        "method": coupling_method,
                        "condition_weight": coupling_weight,
                    }
                },
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
                    "training_sampler": None,
                }
            },
            "compatibility": {
                "enabled": False,
                "include_invisible": bool(defaults.compatibility.include_invisible),
            },
            "heads": heads,
            "topology_queries": topology_queries,
        }

    if joint_public is not None:
        raise ValueError("model.joint is only valid when model.mode=joint")
    assignment_public = public.get("assignment")
    if assignment_public is None:
        raise ValueError("Factorized models require model.assignment")
    _mapping(assignment_public, "model.assignment")
    if "method" in assignment_public:
        raise ValueError("model.assignment.method is obsolete; use model.assignment.formulation")
    if "recurrence" in assignment_public:
        raise ValueError("model.assignment.recurrence is obsolete; use model.assignment.refinement")
    _unknown(assignment_public, {"formulation", "refinement"}, "model.assignment")
    _required(assignment_public, {"formulation"}, "model.assignment")
    method = str(assignment_public.formulation)
    if method not in {"discriminative", "discrete_flow"}:
        raise ValueError("model.assignment.formulation must be discriminative or discrete_flow")

    recurrence, recurrent, feedback = _resolve_recurrence(
        assignment_public.get("refinement"), defaults.assignment.recurrence
    )
    if method == "discrete_flow" and recurrent:
        raise ValueError("Factorized DFM has its own time-conditioned reasoner; omit recurrence")

    invisible = _resolve_invisible(
        public.get("invisible"),
        defaults.invisible,
        backbone_defaults=defaults,
    )
    if method == "discrete_flow" and invisible["enabled"]:
        raise ValueError("Factorized DFM is assignment-only; omit model.invisible or use joint_flow")
    if heads["point_invisible"]["enabled"] and not invisible["enabled"]:
        raise ValueError("model.heads.point_invisible requires an active invisible generator")
    if heads["parent_four_vector"]["enabled"] and not invisible["enabled"]:
        raise ValueError("model.heads.parent_four_vector requires an active invisible generator")

    feedback_public = public.get("feedback")
    if feedback_public is not None:
        _mapping(feedback_public, "model.feedback")
        _unknown(feedback_public, {"invisible_to_assignment"}, "model.feedback")
    c2_public = None if feedback_public is None else feedback_public.get("invisible_to_assignment")
    c2 = _plain(defaults.c2)
    c2_enabled = c2_public is not None
    if c2_enabled:
        if method != "discriminative" or not invisible["enabled"]:
            raise ValueError(
                "model.feedback.invisible_to_assignment requires discriminative "
                "assignment and invisible inference"
            )
        _mapping(c2_public, "model.feedback.invisible_to_assignment")
        _required(c2_public, {"enabled"}, "model.feedback.invisible_to_assignment")
        c2_enabled = bool(c2_public.enabled)
        _unknown(
            c2_public,
            {"enabled", "posterior_samples", "gate_initial_bias", "training_sampler"},
            "model.feedback.invisible_to_assignment",
        )
        c2 = _plain(_merge(defaults.c2, c2_public))
        sampler_public = c2_public.get("training_sampler")
        if sampler_public is not None:
            _mapping(sampler_public, "model.feedback.invisible_to_assignment.training_sampler")
            formulation = invisible["formulation"]
            allowed = {
                "standard": {"steps"},
                "edm": {"nfe"},
                "flow_matching": {"steps", "solver"},
            }[formulation]
            _unknown(
                sampler_public,
                allowed,
                f"model.feedback.invisible_to_assignment.training_sampler ({formulation})",
            )

    c2_sampler = None
    if c2_enabled:
        formulation = invisible["formulation"]
        selected = _plain(
            _merge(
                defaults.c2.training_sampler[formulation],
                c2_public.get("training_sampler"),
            )
        )
        c2_sampler = {
            "formulation": formulation,
            "budget": int(selected.get("nfe", selected.get("steps"))),
            "solver": (
                str(selected["solver"]) if formulation == "flow_matching" else None
            ),
        }

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
        "latent": latent,
        "visible": encoder,
        "assignment": {
            "formulation": "flow_matching" if method == "discrete_flow" else "discriminative",
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
                "training_sampler": c2_sampler,
            }
        },
        "compatibility": {
            "enabled": compatibility_enabled,
            "include_invisible": bool(compatibility["include_invisible"]),
        },
        "heads": heads,
        "topology_queries": topology_queries,
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
    data_keys = {
        "input", "cache", "events", "subset_seed", "max_assignment_states",
        "test_size", "split", "allow_synthetic",
    }
    _unknown(data_public, data_keys, "data")
    _required(data_public, {"input", "cache"}, "data")
    test_size = float(data_public.get("test_size", 0.2))
    if not 0.0 < test_size < 1.0:
        raise ValueError("data.test_size must lie strictly between zero and one")
    split_public = data_public.get(
        "split", {"indices_path": None, "require_existing": False}
    )
    _mapping(split_public, "data.split")
    _unknown(split_public, {"indices_path", "require_existing"}, "data.split")
    split_path = split_public.get("indices_path")
    if split_path is not None:
        split_path = str(_path(split_path, data_path=True))
    if not isinstance(split_public.get("require_existing", False), bool):
        raise ValueError("data.split.require_existing must be boolean")
    if bool(split_public.get("require_existing", False)) and split_path is None:
        raise ValueError("data.split.require_existing=true requires data.split.indices_path")
    events = data_public.get("events", {"train": None, "validation": None, "test": None})
    _unknown(events, {"train", "validation", "test"}, "data.events")
    _required(events, {"train", "validation", "test"}, "data.events")
    input_path = _path(data_public.input, data_path=True)
    cache_value = data_public.cache
    cache_path = None
    if cache_value is not None:
        cache_path = _path(cache_value, data_path=True)
        if cache_path.suffix != ".pt":
            cache_path = cache_path / f"{input_path.stem}.pt"
    inputs_public = public.inputs
    _mapping(inputs_public, "inputs")
    _required(inputs_public, {"transforms"}, "inputs")
    collections = {key: value for key, value in inputs_public.items() if key != "transforms"}
    data_config = _load_data(OmegaConf.create({
        "input": str(input_path), "cache": None if cache_path is None else str(cache_path),
        "transforms": _plain(inputs_public.transforms), "collections": _plain(collections),
    }))

    defaults = OmegaConf.load(DEFAULTS_PATH)
    model = _resolve_model(public.model, defaults.model)
    parent_head = model["heads"]["parent_four_vector"]
    parent_path = None
    if parent_head["enabled"]:
        parent_path = _path(parent_head["target_path"], data_path=True)
        if not parent_path.is_file():
            raise FileNotFoundError(
                "Enabled parent four-vector head requires an existing explicit "
                f"truth sidecar: {parent_path}"
            )
        parent_head["target_path"] = str(parent_path)
    partonic_head = model["heads"]["partonic"]
    partonic_path = None
    if partonic_head["enabled"]:
        partonic_path = _path(partonic_head["target_path"], data_path=True)
        if not partonic_path.is_file():
            raise FileNotFoundError(
                "Enabled partonic head requires an explicit partonic-truth sidecar: "
                f"{partonic_path}"
            )
        partonic_head["target_path"] = str(partonic_path)
    kinematic_names = {"e", "pt", "eta", "phi", "sin_phi", "cos_phi"}
    object_features = []
    for source, spec in data_config.collections.items():
        if source == "GLOBAL":
            continue
        for name in spec.features:
            if name not in kinematic_names and name not in object_features:
                object_features.append(str(name))
    global_spec = data_config.collections.get("GLOBAL")
    model["visible"]["inputs"]["object_features"] = object_features
    model["visible"]["inputs"]["event_features"] = (
        [] if global_spec is None else [str(name) for name in global_spec.features]
    )

    inference_public = public.inference
    _mapping(inference_public, "inference")
    _unknown(
        inference_public,
        {"seed", "assignment", "invisible", "sample_chunk_size", "compatibility"},
        "inference",
    )
    assignment_inference = inference_public.get("assignment", {})
    invisible_inference = inference_public.get("invisible", {})
    compatibility_inference = inference_public.get("compatibility", {})
    _unknown(assignment_inference, {"top_k"}, "inference.assignment")
    _unknown(invisible_inference, {"samples_per_assignment", "flow", "edm", "ddim", "point_estimator"}, "inference.invisible")
    _unknown(compatibility_inference, {"weight"}, "inference.compatibility")
    flow_inference = invisible_inference.get("flow", {})
    edm_inference = invisible_inference.get("edm", {})
    ddim_inference = invisible_inference.get("ddim", {})
    _unknown(flow_inference, {"steps", "solver"}, "inference.invisible.flow")
    _unknown(edm_inference, {"nfe", "sigma_min", "sigma_max", "rho"}, "inference.invisible.edm")
    _unknown(ddim_inference, {"steps"}, "inference.invisible.ddim")
    defaults_inference = defaults.inference
    prediction = {
        "seed": inference_public.get("seed", defaults_inference.seed),
        "top_k": assignment_inference.get("top_k", defaults_inference.top_k),
        "samples": invisible_inference.get("samples_per_assignment", defaults_inference.samples),
        "steps": flow_inference.get("steps", defaults_inference.steps),
        "solver": flow_inference.get("solver", defaults_inference.solver),
        "sample_chunk_size": inference_public.get("sample_chunk_size", defaults_inference.sample_chunk_size),
        "nfe": edm_inference.get("nfe", defaults_inference.nfe),
        "sigma_min": edm_inference.get("sigma_min", defaults_inference.sigma_min),
        "sigma_max": edm_inference.get("sigma_max", defaults_inference.sigma_max),
        "rho": edm_inference.get("rho", defaults_inference.rho),
        "standard_steps": ddim_inference.get("steps", defaults_inference.standard_steps),
        "compatibility_weight": compatibility_inference.get("weight", defaults_inference.compatibility_weight),
        "point_estimator": str(invisible_inference.get("point_estimator", getattr(defaults_inference, "point_estimator", "none"))),
    }
    if prediction["point_estimator"] not in {
        "none", "cfm_single_draw", "cfm_cartesian_mean", "cfm_joint_medoid", "direct_point_head"
    }:
        raise ValueError("inference.invisible.point_estimator is invalid")
    standard_steps = int(prediction["standard_steps"])

    training_public = public.training
    _mapping(training_public, "training")
    training_keys = {
        "batch_size", "workers", "optimizer", "learning_rate", "weight_decay", "epochs", "precision", "gradient_clip",
        "accumulate_grad_batches", "trainable", "loss_weights", "scheduler",
        "early_stopping", "runtime", "output", "shared_encoder_gradients",
        "partonic_shared_encoder_gradients",
    }
    _unknown(training_public, training_keys, "training")
    training = _merge(defaults.training, training_public)
    optimizer = training_public.get("optimizer")
    if optimizer is not None:
        _mapping(optimizer, "training.optimizer")
        _unknown(optimizer, {"type", "learning_rate", "weight_decay"}, "training.optimizer")
        if str(optimizer.get("type", "adamw")) != "adamw":
            raise ValueError("training.optimizer.type must be adamw")
        if optimizer.get("learning_rate") is not None:
            training.learning_rate = optimizer.learning_rate
        if optimizer.get("weight_decay") is not None:
            training.weight_decay = optimizer.weight_decay

    if training_public.get("trainable") is None:
        trainable = ["visible", "assignment"]
        if model["invisible"]["enabled"]:
            trainable.append("invisible")
        if model["coupling"]["invisible_to_assignment"]["enabled"]:
            trainable.append("c2")
        if model["compatibility"]["enabled"]:
            trainable.append("compatibility")
        if model["heads"]["point_invisible"]["enabled"]:
            trainable.append("point")
        if model["heads"]["parent_four_vector"]["enabled"]:
            trainable.append("parent")
        if model["topology_queries"]["enabled"]:
            trainable.append("queries")
        if model["heads"]["partonic"]["enabled"]:
            trainable.append("partonic")
    else:
        trainable = [str(value) for value in training_public.trainable]
    if not trainable or len(set(trainable)) != len(trainable):
        raise ValueError("training.trainable must contain unique component names")
    unknown_trainable = sorted(set(trainable) - TRAINABLE_COMPONENTS)
    if unknown_trainable:
        raise ValueError(f"Unknown training.trainable components: {unknown_trainable}")
    if "c2" in trainable and "assignment" not in trainable:
        raise ValueError("training.trainable=c2 requires assignment to be trainable")
    if "point" in trainable and not model["heads"]["point_invisible"]["enabled"]:
        raise ValueError("training.trainable=point requires an enabled point_invisible head")
    if "parent" in trainable and not model["heads"]["parent_four_vector"]["enabled"]:
        raise ValueError("training.trainable=parent requires an enabled parent_four_vector head")
    if "queries" in trainable and not model["topology_queries"]["enabled"]:
        raise ValueError("training.trainable=queries requires enabled topology_queries")
    if "partonic" in trainable and not model["heads"]["partonic"]["enabled"]:
        raise ValueError("training.trainable=partonic requires an enabled partonic head")
    if model["heads"]["partonic"]["enabled"] and not model["topology_queries"]["enabled"]:
        raise ValueError("model.heads.partonic requires enabled model.topology_queries")
    partonic_gradient_policy = str(
        training_public.get("partonic_shared_encoder_gradients", "both")
    )
    if partonic_gradient_policy not in {"both", "assignment_only"}:
        raise ValueError(
            "training.partonic_shared_encoder_gradients must be both or assignment_only"
        )
    if model["mode"] == "joint_flow" and (("assignment" in trainable) != ("invisible" in trainable)):
        raise ValueError("joint_flow assignment and invisible dynamics must be trainable/frozen together")

    loss_weights = {
        "assignment": 1.0 if "assignment" in trainable else 0.0,
        "invisible": 1.0 if "invisible" in trainable else 0.0,
        "compatibility": 1.0 if "compatibility" in trainable else 0.0,
        "point": 1.0 if "point" in trainable else 0.0,
        "parent": 1.0 if "parent" in trainable else 0.0,
        "partonic": 1.0 if "partonic" in trainable else 0.0,
    }
    if training_public.get("loss_weights") is not None:
        _mapping(training_public.loss_weights, "training.loss_weights")
        _unknown(training_public.loss_weights, set(loss_weights), "training.loss_weights")
        loss_weights.update(
            {key: float(value) for key, value in training_public.loss_weights.items()}
        )

    output_value = training_public.get("output", f"results/{public.experiment.name}")
    output_path = runtime_paths().project(output_value)
    return OmegaConf.create(
        {
            "experiment": _plain(public.experiment),
            "topology_config": str(topology_path),
            "data_config": str(input_path),
            "input_definition": _plain(data_config),
            "data": {
                "input": str(input_path),
                "tensor_cache": None if cache_path is None else str(cache_path),
                "force_rebuild_cache": False,
                "subset_seed": int(data_public.get("subset_seed", public.experiment.seed)),
                "test_size": test_size,
                "split": {
                    "indices_path": split_path,
                    "require_existing": bool(split_public.get("require_existing", False)),
                },
                "truth_bindings": _truth_bindings(topology_path),
                "max_events": {
                    "train": events.train,
                    "val": events.validation,
                    "test": events.test,
                },
                "max_assignment_states": int(data_public.get("max_assignment_states", 100000)),
                "invisible": {
                    "path": str(input_path),
                    "allow_synthetic": bool(data_public.get("allow_synthetic", False)),
                },
                "parent": {
                    "path": None if parent_path is None else str(parent_path),
                    "roles": list(parent_head["roles"]),
                },
                "partonic": {
                    "path": None if partonic_path is None else str(partonic_path),
                    "roles": list(partonic_head["roles"]),
                    "truth_bindings": dict(partonic_head["truth_bindings"]),
                },
            },
            "model": model,
            "inference": {
                "seed": int(prediction["seed"]),
                "top_k_assignments": int(prediction["top_k"]),
                "posterior_samples": int(prediction["samples"]),
                "invisible": {
                    "point_estimator": str(prediction["point_estimator"]),
                    "standard": {"steps": standard_steps},
                    "edm": {
                        "sigma_min": float(prediction["sigma_min"]),
                        "sigma_max": float(prediction["sigma_max"]),
                        "rho": float(prediction["rho"]),
                        "nfe": int(prediction["nfe"]),
                    },
                    "flow_matching": {
                        "steps": int(prediction["steps"]),
                        "solver": str(prediction["solver"]),
                    },
                },
                "assignment_flow": {"steps": int(prediction["steps"])},
                "joint_flow": {
                    "steps": int(prediction["steps"]),
                    "sample_chunk_size": int(prediction["sample_chunk_size"]),
                },
                "compatibility_weight": float(prediction["compatibility_weight"]),
            },
            "training": {
                "batch_size": int(training_public.get("batch_size", 512)),
                "workers": int(training_public.get("workers", 4)),
                "learning_rate": float(training.learning_rate),
                "weight_decay": float(training.weight_decay),
                "epochs": int(training.epochs),
                "precision": str(training.precision),
                "gradient_clip": float(training.gradient_clip),
                "accumulate_grad_batches": int(training.accumulate_grad_batches),
                "runtime": _plain(training.runtime),
                "trainable": trainable,
                "loss_weights": loss_weights,
                "shared_encoder_gradients": str(
                    training_public.get("shared_encoder_gradients", "both")
                ),
                "partonic_shared_encoder_gradients": partonic_gradient_policy,
                "scheduler": _plain(training.scheduler),
                "early_stopping": _plain(training.early_stopping),
                "output": str(output_path),
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
