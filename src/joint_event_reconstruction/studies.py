"""Strict, lightweight study manifests and scientific gate resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re

from omegaconf import OmegaConf

from joint_event_reconstruction.runtime_paths import runtime_paths


REPO_ROOT = runtime_paths().source_root
DEFAULT_GATES_PATH = REPO_ROOT / "configs/studies/UPSTREAM_SELECTIONS.yaml"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPORTING = {"primary", "diagnostic", "oracle", "setup"}
_COMPONENTS = {"visible", "assignment", "invisible", "c2", "compatibility"}
_GATE_STATUSES = {"pending", "locked", "not_applicable"}
_GATE_KEYS = {
    "status",
    "source_study",
    "selected_variant",
    "validated_sha",
    "required_by",
    "selected_patch",
    "artifacts",
}
_GATE_OPTIONAL_KEYS = {"selected_decision"}
_STUDY_KEYS = {
    "name",
    "title",
    "description",
    "experiment_pattern",
    "topologies",
    "seeds",
    "split",
    "requires",
    "inherit",
    "training",
    "campaign",
}
_STUDY_REQUIRED_KEYS = _STUDY_KEYS - {"campaign"}
_CAMPAIGN_KEYS = {"enabled", "state", "reason"}
_VARIANT_KEYS = {
    "label",
    "reporting",
    "patch",
    "initialise_from",
    "initialise_from_gate",
    "initialise_components",
    "checkpoint_from",
    "checkpoint_from_gate",
    "predictions",
}
_VARIANT_REQUIRED_KEYS = _VARIANT_KEYS - {"checkpoint_from_gate"}
_PREDICTION_KEYS = {"label", "patch", "recurrent_steps"}
_CHECKPOINT_SOURCE_KEYS = {"study", "variant"}
_ARTIFACT_KEYS = {
    "source_study",
    "variant",
    "model_seed",
    "data_subset_seed",
    "git_sha",
    "architecture_identity",
    "checkpoint",
    "checkpoint_sha256",
    "identity_file",
    "identity_sha256",
}
_ARTIFACT_OPTIONAL_KEYS = {
    "selection_provenance",
    "training_provenance",
}


class ScientificGatePending(RuntimeError):
    """Raised when a requested study still needs a human scientific decision."""


class ScientificGateNotApplicable(RuntimeError):
    """Raised when a human gate explicitly makes a study irrelevant."""


@dataclass(frozen=True)
class PredictionSpec:
    name: str
    label: str
    patch: dict
    recurrent_steps: int | None


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    reporting: str
    patch: dict
    initialise_from: str | None
    initialise_from_gate: str | None
    initialise_components: tuple[str, ...]
    checkpoint_from: tuple[str, str] | None
    checkpoint_from_gate: str | None
    predictions: tuple[PredictionSpec, ...]


@dataclass(frozen=True)
class StudySpec:
    path: Path
    name: str
    title: str
    description: str
    experiment_pattern: str
    topologies: tuple[str, ...]
    seeds: tuple[int, ...]
    split: str
    requires: tuple[str, ...]
    inherit: tuple[str, ...]
    training: bool
    variants: tuple[VariantSpec, ...]
    campaign_enabled: bool = True
    campaign_state: str = "active"
    campaign_reason: str = ""
    resolved_gates: dict[str, dict] | None = None
    gates_path: Path | None = None
    gates_digest: str | None = None

    @property
    def variant_map(self) -> dict[str, VariantSpec]:
        return {variant.name: variant for variant in self.variants}

    def dependency_order(self) -> tuple[VariantSpec, ...]:
        """Return locally initialised variants in checkpoint order."""
        remaining = dict(self.variant_map)
        complete: set[str] = set()
        ordered: list[VariantSpec] = []
        while remaining:
            ready = [
                variant
                for variant in remaining.values()
                if variant.initialise_from is None
                or variant.initialise_from in complete
            ]
            if not ready:
                cycle = {
                    name: variant.initialise_from
                    for name, variant in remaining.items()
                }
                raise ValueError(f"Study initialisation dependency cycle: {cycle}")
            for variant in ready:
                ordered.append(variant)
                complete.add(variant.name)
                del remaining[variant.name]
        return tuple(ordered)

    def dependency_closure(self, selected: tuple[str, ...]) -> tuple[VariantSpec, ...]:
        """Include every local checkpoint prerequisite of selected variants."""
        unknown = sorted(set(selected) - set(self.variant_map))
        if unknown:
            raise ValueError(f"Unknown study variants: {unknown}")
        required = set(selected)
        frontier = list(selected)
        while frontier:
            name = frontier.pop()
            parent = self.variant_map[name].initialise_from
            if parent is not None and parent not in required:
                required.add(parent)
                frontier.append(parent)
        return tuple(
            variant for variant in self.dependency_order() if variant.name in required
        )

    def training_order(self, selected: tuple[str, ...] | None = None) -> tuple[VariantSpec, ...]:
        """Return local-training variants, excluding immutable checkpoint consumers."""
        ordered = (
            self.dependency_order()
            if selected is None
            else self.dependency_closure(selected)
        )
        return tuple(
            variant
            for variant in ordered
            if variant.checkpoint_from is None and variant.checkpoint_from_gate is None
        )

    def experiment_path(self, repo_root: Path, topology: str) -> Path:
        if topology not in self.topologies:
            raise ValueError(f"Topology {topology!r} is not in study {self.name!r}")
        path = Path(self.experiment_pattern.format(topology=topology))
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"Study experiment for topology {topology!r} does not exist: {path}"
            )
        return path

    def gate_artifact(self, gate_name: str, topology: str, seed: int) -> dict:
        """Return the exact selected checkpoint record for one topology and seed."""
        if self.resolved_gates is None or gate_name not in self.resolved_gates:
            raise RuntimeError(
                f"Study {self.name!r} has no resolved scientific gate {gate_name!r}"
            )
        gate = self.resolved_gates[gate_name]
        artifacts = gate.get("artifacts") or {}
        topology_artifacts = artifacts.get(topology)
        if not isinstance(topology_artifacts, dict):
            raise RuntimeError(
                f"Locked gate {gate_name!r} has no artifact registry for topology {topology!r}"
            )
        entry = topology_artifacts.get(f"seed_{int(seed)}")
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"Locked gate {gate_name!r} has no checkpoint for topology={topology!r}, "
                f"model_seed={int(seed)}"
            )
        if not _ARTIFACT_KEYS <= set(entry) <= _ARTIFACT_KEYS | _ARTIFACT_OPTIONAL_KEYS:
            raise ValueError(
                f"Gate artifact {gate_name}:{topology}:seed_{int(seed)} keys must contain "
                f"{sorted(_ARTIFACT_KEYS)} and only optional provenance keys"
            )
        return entry

    def required_gate(self, gate_name: str) -> dict:
        if self.resolved_gates is None or gate_name not in self.resolved_gates:
            raise RuntimeError(
                f"Study {self.name!r} has no resolved scientific gate {gate_name!r}"
            )
        return self.resolved_gates[gate_name]


def _exact_keys(mapping: dict, expected: set[str], name: str) -> None:
    observed = set(mapping)
    if observed != expected:
        raise ValueError(
            f"{name} keys must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - observed)}, "
            f"unknown={sorted(observed - expected)}"
        )


def _identifier(value: object, name: str) -> str:
    text = str(value)
    if not _IDENTIFIER.fullmatch(text):
        raise ValueError(f"{name} must match {_IDENTIFIER.pattern!r}; found {text!r}")
    return text


def _patch(value: object, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _string_list(
    value: object, name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    result = tuple(_identifier(item, f"{name}[]") for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _merge_patch(base: dict, override: dict) -> dict:
    """Deep-merge explicit YAML mappings; the right-hand patch owns conflicts."""
    result = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_patch(result[key], value)
        else:
            result[key] = json.loads(json.dumps(value))
    return result


def _study_paths() -> dict[str, Path]:
    paths = {}
    for path in sorted((REPO_ROOT / "configs/studies").glob("[0-9][0-9]_*.yaml")):
        study = load_study(path)
        if study.name in paths:
            raise ValueError(f"Duplicate maintained study name {study.name!r}")
        paths[study.name] = path
    return paths


def load_scientific_gates(path: str | Path = DEFAULT_GATES_PATH) -> dict[str, dict]:
    """Load and strictly validate the human scientific decision gates."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Scientific selection gates do not exist: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Scientific selection gates must be a non-empty mapping: {path}")
    study_paths = _study_paths()
    for name, gate in raw.items():
        _identifier(name, "scientific gate name")
        if not isinstance(gate, dict):
            raise ValueError(f"Scientific gate {name!r} must be a mapping")
        if not _GATE_KEYS <= set(gate) <= _GATE_KEYS | _GATE_OPTIONAL_KEYS:
            raise ValueError(
                f"scientific gate {name} keys must contain {sorted(_GATE_KEYS)} and only "
                f"optional keys {sorted(_GATE_OPTIONAL_KEYS)}"
            )
        status = str(gate["status"])
        if status not in _GATE_STATUSES:
            raise ValueError(f"Scientific gate {name!r} has invalid status {status!r}")
        source_study = str(gate["source_study"]).strip()
        if source_study not in study_paths:
            raise ValueError(
                f"Scientific gate {name!r} refers to unknown source study {source_study!r}"
            )
        required_by = gate["required_by"]
        if not isinstance(required_by, list) or any(
            not isinstance(item, str) or not item.strip() for item in required_by
        ):
            raise ValueError(f"Scientific gate {name!r} has invalid required_by")
        if len(set(required_by)) != len(required_by):
            raise ValueError(f"Scientific gate {name!r} has duplicate required_by entries")
        unknown_dependents = sorted(set(required_by) - set(study_paths))
        if unknown_dependents:
            raise ValueError(
                f"Scientific gate {name!r} refers to unknown dependent studies "
                f"{unknown_dependents}"
            )

        required_gate_users = {
            study.name
            for study in (load_study(study_path) for study_path in study_paths.values())
            if name in study.requires
        }
        if required_gate_users != set(required_by):
            raise ValueError(
                f"Scientific gate {name!r}.required_by must match manifest requirements: "
                f"declared={sorted(required_by)}, manifests={sorted(required_gate_users)}"
            )

        selected_patch = gate["selected_patch"]
        if not isinstance(selected_patch, dict):
            raise ValueError(f"Scientific gate {name!r}.selected_patch must be a mapping")
        artifacts = gate["artifacts"]
        if not isinstance(artifacts, dict):
            raise ValueError(f"Scientific gate {name!r}.artifacts must be a mapping")

        if status == "locked":
            selected_variant = str(gate.get("selected_variant") or "").strip()
            if not selected_variant:
                raise ValueError(f"Locked scientific gate {name!r} has no selected_variant")
            source = load_study(study_paths[source_study])
            if selected_variant not in source.variant_map:
                raise ValueError(
                    f"Locked scientific gate {name!r} selects unknown variant "
                    f"{selected_variant!r} from {source_study!r}"
                )
            validated_sha = gate.get("validated_sha")
            if not isinstance(validated_sha, str) or not _COMMIT_SHA.fullmatch(validated_sha):
                raise ValueError(
                    f"Locked scientific gate {name!r} must record a 40-character validated_sha"
                )
            selected_decision = gate.get("selected_decision")
            if selected_decision is not None and (
                not isinstance(selected_decision, str) or not selected_decision.strip()
            ):
                raise ValueError(
                    f"Locked scientific gate {name!r} has an invalid selected_decision"
                )
            if not selected_patch and selected_decision is None:
                raise ValueError(f"Locked scientific gate {name!r} has no selected_patch")
        else:
            if gate.get("selected_variant") not in (None, ""):
                raise ValueError(
                    f"Non-locked scientific gate {name!r} must not select a variant"
                )
            if gate.get("validated_sha") not in (None, ""):
                raise ValueError(
                    f"Non-locked scientific gate {name!r} must not record validated_sha"
                )
            if selected_patch or artifacts:
                raise ValueError(
                    f"Non-locked scientific gate {name!r} must not contain selected data"
                )
            if gate.get("selected_decision") not in (None, ""):
                raise ValueError(
                    f"Non-locked scientific gate {name!r} must not select a decision"
                )

        for topology, topology_artifacts in artifacts.items():
            if not isinstance(topology_artifacts, dict):
                raise ValueError(
                    f"Gate artifact registry {name!r}:{topology!r} must be a mapping"
                )
            for seed_name, entry in topology_artifacts.items():
                if not re.fullmatch(r"seed_[0-9]+", str(seed_name)):
                    raise ValueError(
                        f"Gate artifact registry {name!r}:{topology!r} has invalid seed key "
                        f"{seed_name!r}"
                    )
                if (
                    not isinstance(entry, dict)
                    or not _ARTIFACT_KEYS <= set(entry)
                    or not set(entry) <= _ARTIFACT_KEYS | _ARTIFACT_OPTIONAL_KEYS
                ):
                    raise ValueError(
                        f"Gate artifact {name}:{topology}:{seed_name} keys must be "
                        f"contain {sorted(_ARTIFACT_KEYS)} and only optional provenance keys"
                    )
                if entry["source_study"] not in study_paths:
                    raise ValueError(
                        f"Gate artifact {name}:{topology}:{seed_name} has unknown "
                        f"training source study {entry['source_study']!r}"
                    )
                if not _COMMIT_SHA.fullmatch(str(entry["git_sha"])):
                    raise ValueError(
                        f"Gate artifact {name}:{topology}:{seed_name} has invalid git_sha"
                    )
                for key in ("checkpoint", "identity_file"):
                    if not str(entry[key]).strip():
                        raise ValueError(
                            f"Gate artifact {name}:{topology}:{seed_name} has empty {key}"
                        )
                for key in ("checkpoint_sha256", "identity_sha256", "architecture_identity"):
                    if not re.fullmatch(r"[0-9a-f]{64}", str(entry[key])):
                        raise ValueError(
                            f"Gate artifact {name}:{topology}:{seed_name} has invalid {key}"
                        )
                if int(entry["model_seed"]) < 0 or int(entry["data_subset_seed"]) < 0:
                    raise ValueError(
                        f"Gate artifact {name}:{topology}:{seed_name} has invalid seed metadata"
                    )
                if int(entry["model_seed"]) != int(str(seed_name).split("_", 1)[1]):
                    raise ValueError(
                        f"Gate artifact {name}:{topology}:{seed_name} model_seed disagrees "
                        "with its registry key"
                    )
                selection = entry.get("selection_provenance")
                if selection is None:
                    if entry["source_study"] != source_study or entry["variant"] != gate.get(
                        "selected_variant"
                    ):
                        raise ValueError(
                            f"Gate artifact {name}:{topology}:{seed_name} training provenance "
                            "must agree with its locked gate unless explicit selection_provenance "
                            "is recorded"
                        )
                else:
                    if not isinstance(selection, dict):
                        raise ValueError(
                            f"Gate artifact {name}:{topology}:{seed_name} selection_provenance "
                            "must be a mapping"
                        )
                    if selection.get("source_study") != source_study or selection.get(
                        "variant"
                    ) != gate.get("selected_variant"):
                        raise ValueError(
                            f"Gate artifact {name}:{topology}:{seed_name} selection provenance "
                            "does not identify the selected gate variant"
                        )
    return raw


def _raw_study(path: str | Path) -> StudySpec:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Study manifest does not exist: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or set(raw) != {"study", "variants"}:
        raise ValueError("Study manifest must contain exactly 'study' and 'variants'")

    study = raw["study"]
    if not isinstance(study, dict):
        raise ValueError("study must be a mapping")
    observed_study_keys = set(study)
    if not _STUDY_REQUIRED_KEYS <= observed_study_keys <= _STUDY_KEYS:
        raise ValueError(
            "study keys must contain exactly the required keys "
            f"{sorted(_STUDY_REQUIRED_KEYS)} and only optional keys "
            f"{sorted(_STUDY_KEYS - _STUDY_REQUIRED_KEYS)}; "
            f"missing={sorted(_STUDY_REQUIRED_KEYS - observed_study_keys)}, "
            f"unknown={sorted(observed_study_keys - _STUDY_KEYS)}"
        )
    name = _identifier(study["name"], "study.name")
    title = str(study["title"]).strip()
    if not title:
        raise ValueError("study.title must be non-empty")
    description = str(study["description"]).strip()
    if not description:
        raise ValueError("study.description must be non-empty")
    pattern = str(study["experiment_pattern"]).strip()
    if "{topology}" not in pattern:
        raise ValueError("study.experiment_pattern must contain {topology}")
    topologies = _string_list(study["topologies"], "study.topologies")
    seeds = tuple(int(value) for value in study["seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("study.seeds must be non-empty and unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("study.seeds must be non-negative")
    split = str(study["split"])
    if split not in {"train", "validation", "test"}:
        raise ValueError("study.split must be train, validation or test")
    requires = _string_list(study["requires"], "study.requires", allow_empty=True)
    inherit = _string_list(study["inherit"], "study.inherit", allow_empty=True)
    if not set(inherit) <= set(requires):
        raise ValueError("study.inherit must be a subset of study.requires")
    if not isinstance(study["training"], bool):
        raise ValueError("study.training must be boolean")
    campaign = study.get("campaign", {})
    if not isinstance(campaign, dict) or (campaign and set(campaign) != _CAMPAIGN_KEYS):
        raise ValueError(
            "study.campaign must contain exactly enabled, state and reason when present"
        )
    campaign_enabled = True if not campaign else campaign["enabled"]
    if not isinstance(campaign_enabled, bool):
        raise ValueError("study.campaign.enabled must be boolean")
    campaign_state = "active" if not campaign else str(campaign["state"])
    if campaign_state not in {"active", "deferred", "superseded"}:
        raise ValueError("study.campaign.state must be active, deferred or superseded")
    campaign_reason = "" if not campaign else str(campaign["reason"]).strip()
    if campaign_state != "active" and (campaign_enabled or not campaign_reason):
        raise ValueError(
            "deferred/superseded studies must be disabled and carry a reason"
        )

    raw_variants = raw["variants"]
    if not isinstance(raw_variants, dict) or not raw_variants:
        raise ValueError("variants must be a non-empty mapping")
    variants: list[VariantSpec] = []
    for raw_name, values in raw_variants.items():
        variant_name = _identifier(raw_name, "variant name")
        if not isinstance(values, dict):
            raise ValueError(f"variants.{variant_name} must be a mapping")
        observed_keys = set(values)
        if not _VARIANT_REQUIRED_KEYS <= observed_keys or not observed_keys <= _VARIANT_KEYS:
            raise ValueError(
                f"variants.{variant_name} keys must contain exactly the required keys "
                f"{sorted(_VARIANT_REQUIRED_KEYS)} and only optional keys "
                f"{sorted(_VARIANT_KEYS - _VARIANT_REQUIRED_KEYS)}; "
                f"missing={sorted(_VARIANT_REQUIRED_KEYS - observed_keys)}, "
                f"unknown={sorted(observed_keys - _VARIANT_KEYS)}"
            )
        reporting = str(values["reporting"])
        if reporting not in _REPORTING:
            raise ValueError(
                f"variants.{variant_name}.reporting must be one of {sorted(_REPORTING)}"
            )
        initialise_from = values["initialise_from"]
        if initialise_from is not None:
            initialise_from = _identifier(
                initialise_from, f"variants.{variant_name}.initialise_from"
            )
            if initialise_from == variant_name:
                raise ValueError(f"Variant {variant_name!r} cannot initialise from itself")
        initialise_from_gate = values["initialise_from_gate"]
        if initialise_from_gate is not None:
            initialise_from_gate = _identifier(
                initialise_from_gate,
                f"variants.{variant_name}.initialise_from_gate",
            )
        components = tuple(str(value) for value in values["initialise_components"])
        if initialise_from is not None and initialise_from_gate is not None:
            raise ValueError(
                f"variants.{variant_name} cannot use both local and gate initialisation"
            )
        if (initialise_from is not None or initialise_from_gate is not None) != bool(components):
            raise ValueError(
                f"variants.{variant_name}: an initialisation source and initialise_components "
                "must be supplied together"
            )
        unknown_components = sorted(set(components) - _COMPONENTS)
        if unknown_components:
            raise ValueError(
                f"variants.{variant_name} has unknown initialisation components "
                f"{unknown_components}"
            )

        checkpoint_from = values["checkpoint_from"]
        parsed_checkpoint_from = None
        if checkpoint_from is not None:
            if not isinstance(checkpoint_from, dict) or set(checkpoint_from) != _CHECKPOINT_SOURCE_KEYS:
                raise ValueError(
                    f"variants.{variant_name}.checkpoint_from must contain exactly "
                    f"{sorted(_CHECKPOINT_SOURCE_KEYS)}"
                )
            parsed_checkpoint_from = (
                _identifier(checkpoint_from["study"], f"variants.{variant_name}.checkpoint_from.study"),
                _identifier(checkpoint_from["variant"], f"variants.{variant_name}.checkpoint_from.variant"),
            )
        checkpoint_from_gate = values.get("checkpoint_from_gate")
        if checkpoint_from_gate is not None:
            checkpoint_from_gate = _identifier(
                checkpoint_from_gate,
                f"variants.{variant_name}.checkpoint_from_gate",
            )
        if checkpoint_from is not None and checkpoint_from_gate is not None:
            raise ValueError(
                f"variants.{variant_name} cannot use both checkpoint_from and "
                "checkpoint_from_gate"
            )
        if checkpoint_from_gate is not None and (
            initialise_from is not None or initialise_from_gate is not None
        ):
            raise ValueError(
                f"variants.{variant_name} cannot initialise a variant that reads a "
                "gate checkpoint"
            )
        raw_predictions = values["predictions"]
        if not isinstance(raw_predictions, dict):
            raise ValueError(f"variants.{variant_name}.predictions must be a mapping")
        predictions = []
        for raw_prediction_name, prediction in raw_predictions.items():
            prediction_name = _identifier(raw_prediction_name, "prediction name")
            if not isinstance(prediction, dict):
                raise ValueError(
                    f"variants.{variant_name}.predictions.{prediction_name} must be a mapping"
                )
            _exact_keys(
                prediction,
                _PREDICTION_KEYS,
                f"variants.{variant_name}.predictions.{prediction_name}",
            )
            recurrent_steps = prediction["recurrent_steps"]
            if recurrent_steps is not None:
                recurrent_steps = int(recurrent_steps)
                if recurrent_steps <= 0:
                    raise ValueError("prediction recurrent_steps must be positive or null")
            predictions.append(
                PredictionSpec(
                    name=prediction_name,
                    label=str(prediction["label"]),
                    patch=_patch(
                        prediction["patch"],
                        f"variants.{variant_name}.predictions.{prediction_name}.patch",
                    ),
                    recurrent_steps=recurrent_steps,
                )
            )
        if reporting != "setup" and not predictions:
            raise ValueError(
                f"Non-setup variant {variant_name!r} must define at least one prediction"
            )
        variants.append(
            VariantSpec(
                name=variant_name,
                label=str(values["label"]),
                reporting=reporting,
                patch=_patch(values["patch"], f"variants.{variant_name}.patch"),
                initialise_from=initialise_from,
                initialise_from_gate=initialise_from_gate,
                initialise_components=components,
                checkpoint_from=parsed_checkpoint_from,
                checkpoint_from_gate=checkpoint_from_gate,
                predictions=tuple(predictions),
            )
        )

    names = {variant.name for variant in variants}
    for variant in variants:
        if variant.initialise_from is not None and variant.initialise_from not in names:
            raise ValueError(
                f"Variant {variant.name!r} initialises from unknown variant "
                f"{variant.initialise_from!r}"
            )
    result = StudySpec(
        path=path,
        name=name,
        title=title,
        description=description,
        experiment_pattern=pattern,
        topologies=topologies,
        seeds=seeds,
        split=split,
        requires=requires,
        inherit=inherit,
        training=bool(study["training"]),
        variants=tuple(variants),
        campaign_enabled=campaign_enabled,
        campaign_state=campaign_state,
        campaign_reason=campaign_reason,
    )
    result.dependency_order()
    return result


def load_study(path: str | Path) -> StudySpec:
    """Load a manifest without applying human gate patches."""
    return _raw_study(path)


def resolve_study(
    path: str | Path,
    gates_path: str | Path | None = None,
) -> StudySpec:
    """Resolve all required locked gate patches into a study manifest."""
    study = _raw_study(path)
    gates_path = Path(gates_path or DEFAULT_GATES_PATH).expanduser().resolve()
    gates = load_scientific_gates(gates_path)
    unknown = sorted(set(study.requires) - set(gates))
    if unknown:
        raise ValueError(
            f"Study {study.name} requires unknown scientific gate(s): {unknown}"
        )
    not_applicable = [
        gate_name
        for gate_name in study.requires
        if gates[gate_name]["status"] == "not_applicable"
    ]
    if not_applicable:
        raise ScientificGateNotApplicable(
            f"NOT APPLICABLE: Study {study.name} requires gate(s) marked "
            + ", ".join(not_applicable)
        )
    missing = [
        gate_name for gate_name in study.requires if gates[gate_name]["status"] == "pending"
    ]
    if missing:
        raise ScientificGatePending(
            f"BLOCKED: Study {study.name} requires human gate(s) "
            + ", ".join(missing)
        )
    selected = {gate_name: gates[gate_name] for gate_name in study.requires}
    if (
        study.name == "joint_recurrence"
        and selected["joint_model"].get("selected_variant") == "factorized"
    ):
        raise ScientificGateNotApplicable(
            "NOT APPLICABLE: joint recurrence requires a selected joint model; "
            "the joint-model gate selected the factorized hypothesis"
        )
    variants = study.variants
    if (
        study.name == "posterior_size"
        and selected["joint_model"].get("selected_variant") == "factorized"
    ):
        variants = tuple(variant for variant in variants if variant.name != "joint")
    merged_variants = []
    for variant in variants:
        patch = {}
        for gate_name in study.inherit:
            patch = _merge_patch(patch, selected[gate_name]["selected_patch"])
        patch = _merge_patch(patch, variant.patch)
        if variant.initialise_from_gate is not None:
            if variant.initialise_from_gate not in selected:
                raise ValueError(
                    f"Variant {variant.name!r} initialises from gate "
                    f"{variant.initialise_from_gate!r}, which is not in study.requires"
                )
            for topology in study.topologies:
                for seed in study.seeds:
                    study_gate = selected[variant.initialise_from_gate]
                    if not (study_gate.get("artifacts") or {}).get(topology, {}).get(
                        f"seed_{seed}"
                    ):
                        raise ValueError(
                            f"Locked gate {variant.initialise_from_gate!r} has no selected "
                            f"checkpoint for topology={topology}, model_seed={seed}; "
                            "checkpoint initialisation cannot use an unregistered artifact"
                        )
        if variant.checkpoint_from is not None:
            source_name, source_variant = variant.checkpoint_from
            source_paths = _study_paths()
            if source_name not in source_paths:
                raise ValueError(
                    f"Variant {variant.name!r} references unknown checkpoint study "
                    f"{source_name!r}"
                )
            source_study = load_study(source_paths[source_name])
            if source_variant not in source_study.variant_map:
                raise ValueError(
                    f"Variant {variant.name!r} references unknown checkpoint variant "
                    f"{source_name}:{source_variant}"
                )
        if variant.checkpoint_from_gate is not None:
            gate_name = variant.checkpoint_from_gate
            if gate_name not in selected:
                raise ValueError(
                    f"Variant {variant.name!r} references gate checkpoint {gate_name!r}, "
                    "which is not in study.requires"
                )
            for topology in study.topologies:
                for seed in study.seeds:
                    artifact = (
                        selected[gate_name].get("artifacts") or {}
                    ).get(topology, {}).get(f"seed_{seed}")
                    if not artifact:
                        raise ValueError(
                            f"Locked gate {gate_name!r} has no selected checkpoint for "
                            f"topology={topology}, model_seed={seed}; "
                            "gate checkpoint consumption requires a registered artifact"
                        )
        merged_variants.append(replace(variant, patch=patch))
    selected_payload = json.dumps(selected, sort_keys=True, separators=(",", ":")).encode()
    return replace(
        study,
        variants=tuple(merged_variants),
        resolved_gates=selected,
        gates_path=gates_path,
        gates_digest=hashlib.sha256(selected_payload).hexdigest(),
    )
