"""Human-readable encoder input contracts and Study-12 fairness checks."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.public_config import resolve_public_experiment
from joint_event_reconstruction.studies import resolve_study


REPO_ROOT = Path(__file__).resolve().parents[2]


def encoder_input_contract(experiment) -> dict:
    data = experiment.input_definition
    encoder = experiment.model.visible
    objects = {}
    for source, spec in data.collections.items():
        if source == "GLOBAL":
            continue
        objects[str(source)] = {
            "count": int(spec.count),
            "available_features": [str(name) for name in spec.features],
            "selected_object_features": [str(name) for name in spec.features],
            "object_type": str(source),
        }
    result = {
        "dataset": str(experiment.data.input),
        "cache": str(experiment.data.tensor_cache),
        "encoder": "transformer" if str(encoder.type) == "relational_transformer" else str(encoder.type),
        "representation": str(encoder.representation),
        "relations": str(encoder.relations_type),
        "physical_vector_representation": "E_px_py_pz_GeV_v1",
        "met_representation": "detector_transverse_pseudovector_Et_px_py_0_GeV_v1",
        "objects": objects,
        "event_features": [str(name) for name in encoder.inputs.event_features],
        "output": {"tokens": "[B,N,D]", "pooled": "[B,D]"},
    }
    if str(encoder.type) in {"lgatr", "lgatr_slim"}:
        result["symmetry_breaking"] = OmegaConf.to_container(
            encoder.symmetry_breaking, resolve=True
        )
    if str(encoder.relations_type) == "physics":
        result["relation_settings"] = OmegaConf.to_container(encoder.relations, resolve=True)
    return result


def load_input_contract(path: str | Path) -> dict:
    return encoder_input_contract(load_experiment(path))


def study_input_contracts(path: str | Path) -> dict[str, dict]:
    study = resolve_study(path)
    contracts = {}
    for topology in study.topologies:
        experiment_path = study.experiment_path(REPO_ROOT, topology)
        public = OmegaConf.load(experiment_path)
        for variant in study.variants:
            resolved = resolve_public_experiment(
                OmegaConf.merge(public, variant.patch), source_path=experiment_path
            )
            contracts[variant.name] = encoder_input_contract(resolved)
    return contracts


def assert_study12_fairness(contracts: dict[str, dict]) -> None:
    names = (
        "transformer_baseline",
        "fourvector_transformer",
        "relational_transformer",
        "lgatr",
        "lgatr_slim",
    )
    missing = sorted(set(names) - set(contracts))
    if missing:
        raise ValueError(f"Study-12 fairness variants are missing: {missing}")
    reference = contracts[names[0]]
    keys = ("met_representation", "event_features")
    for name in names[1:]:
        for key in keys:
            if contracts[name][key] != reference[key]:
                raise ValueError(f"Study-12 input mismatch for {name}: {key}")
        if set(contracts[name]["objects"]) != set(reference["objects"]):
            raise ValueError(f"Study-12 source-collection mismatch for {name}")
        for source in reference["objects"]:
            if contracts[name]["objects"][source]["available_features"] != reference["objects"][source]["available_features"]:
                raise ValueError(
                    f"Study-12 object-feature mismatch for {name}:{source}"
                )


def format_input_contract(contract: dict) -> str:
    lines = ["Dataset", "-------", contract["dataset"], "", "Cache", "-----", contract["cache"], "", "Encoder", "-------", contract["encoder"], f"representation: {contract['representation']}", f"relations: {contract['relations']}"]
    for source, values in contract["objects"].items():
        lines.extend(
            [
                "",
                source,
                "-" * len(source),
                f"count: {values['count']}",
                f"physical kinematics: {contract['physical_vector_representation']}",
                "object features: " + (", ".join(values["selected_object_features"]) or "none"),
                f"type: {values['object_type']}",
            ]
        )
    lines.extend(["", "Event features", "--------------", ", ".join(contract["event_features"]) or "none"])
    if "symmetry_breaking" in contract:
        symmetry = contract["symmetry_breaking"]
        lines.extend(["", "Symmetry breaking", "-----------------", f"beam = {symmetry['beam']}", f"time = {symmetry['time']}"])
    lines.extend(["", "Output", "------", "tokens: [B,N,D]", "pooled: [B,D]"])
    return "\n".join(lines)
