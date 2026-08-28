"""Strict, lightweight study manifests for reproducible JER campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from omegaconf import OmegaConf


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_REPORTING = {"primary", "diagnostic", "oracle", "setup"}
_STUDY_KEYS = {
    "name",
    "description",
    "experiment_pattern",
    "topologies",
    "seeds",
    "split",
}
_VARIANT_KEYS = {
    "label",
    "reporting",
    "train_overrides",
    "initialise_from",
    "initialise_components",
    "predictions",
}
_PREDICTION_KEYS = {"label", "overrides", "recurrent_steps"}
_COMPONENTS = {"visible", "assignment", "invisible", "coupling", "compatibility"}


@dataclass(frozen=True)
class PredictionSpec:
    name: str
    label: str
    overrides: tuple[str, ...]
    recurrent_steps: int | None


@dataclass(frozen=True)
class VariantSpec:
    name: str
    label: str
    reporting: str
    train_overrides: tuple[str, ...]
    initialise_from: str | None
    initialise_components: tuple[str, ...]
    predictions: tuple[PredictionSpec, ...]


@dataclass(frozen=True)
class StudySpec:
    path: Path
    name: str
    description: str
    experiment_pattern: str
    topologies: tuple[str, ...]
    seeds: tuple[int, ...]
    split: str
    variants: tuple[VariantSpec, ...]

    @property
    def variant_map(self) -> dict[str, VariantSpec]:
        return {variant.name: variant for variant in self.variants}

    def dependency_order(self) -> tuple[VariantSpec, ...]:
        """Topologically order variants by checkpoint initialisation dependencies."""
        remaining = dict(self.variant_map)
        ordered: list[VariantSpec] = []
        complete: set[str] = set()
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
        raise ValueError(
            f"{name} must match {_IDENTIFIER.pattern!r}; found {text!r}"
        )
    return text


def _override_list(values: object, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a YAML list of KEY=VALUE strings")
    result = tuple(str(value) for value in values)
    for value in result:
        if "=" not in value or value.startswith("="):
            raise ValueError(f"Invalid override in {name}: {value!r}")
    return result


def load_study(path: str | Path) -> StudySpec:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Study manifest does not exist: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or set(raw) != {"study", "variants"}:
        raise ValueError("Study manifest must contain exactly 'study' and 'variants'")

    study = raw["study"]
    if not isinstance(study, dict):
        raise ValueError("study must be a mapping")
    _exact_keys(study, _STUDY_KEYS, "study")
    name = _identifier(study["name"], "study.name")
    description = str(study["description"]).strip()
    if not description:
        raise ValueError("study.description must be non-empty")
    pattern = str(study["experiment_pattern"]).strip()
    if "{topology}" not in pattern:
        raise ValueError("study.experiment_pattern must contain {topology}")
    topologies = tuple(
        _identifier(value, "study.topologies[]") for value in study["topologies"]
    )
    if not topologies or len(set(topologies)) != len(topologies):
        raise ValueError("study.topologies must be non-empty and unique")
    seeds = tuple(int(value) for value in study["seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("study.seeds must be non-empty and unique")
    split = str(study["split"])
    if split not in {"train", "validation", "test"}:
        raise ValueError("study.split must be train, validation or test")

    raw_variants = raw["variants"]
    if not isinstance(raw_variants, dict) or not raw_variants:
        raise ValueError("variants must be a non-empty mapping")
    variants: list[VariantSpec] = []
    for raw_name, values in raw_variants.items():
        variant_name = _identifier(raw_name, "variant name")
        if not isinstance(values, dict):
            raise ValueError(f"variants.{variant_name} must be a mapping")
        _exact_keys(values, _VARIANT_KEYS, f"variants.{variant_name}")
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
        components = tuple(str(value) for value in values["initialise_components"])
        if bool(initialise_from) != bool(components):
            raise ValueError(
                f"variants.{variant_name}: initialise_from and initialise_components "
                "must be supplied together"
            )
        unknown_components = sorted(set(components) - _COMPONENTS)
        if unknown_components:
            raise ValueError(
                f"variants.{variant_name} has unknown initialisation components "
                f"{unknown_components}"
            )

        raw_predictions = values["predictions"]
        if not isinstance(raw_predictions, dict):
            raise ValueError(f"variants.{variant_name}.predictions must be a mapping")
        predictions: list[PredictionSpec] = []
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
                    overrides=_override_list(
                        prediction["overrides"],
                        f"variants.{variant_name}.predictions.{prediction_name}.overrides",
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
                train_overrides=_override_list(
                    values["train_overrides"],
                    f"variants.{variant_name}.train_overrides",
                ),
                initialise_from=initialise_from,
                initialise_components=components,
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
        description=description,
        experiment_pattern=pattern,
        topologies=topologies,
        seeds=seeds,
        split=split,
        variants=tuple(variants),
    )
    result.dependency_order()
    return result
