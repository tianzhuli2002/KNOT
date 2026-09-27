"""Create hash-recorded common initial states for the real-neutrino release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import lightning as L
import torch

from joint_event_reconstruction.cli.run import (
    _load_initial_components,
    build_data,
)
from joint_event_reconstruction.configuration import (
    load_experiment,
    load_visible_config,
)
from joint_event_reconstruction.training.joint import JointEventModule

CONFIGS = {
    "Q0": "q0.public.yaml",
    "Q1": "q1.public.yaml",
    "PCFM_Q": "pcfm_q.public.yaml",
    "J0": "j0.public.yaml",
    "J_A": "j_a.public.yaml",
    "J_AN": "j_an.public.yaml",
}
COMPONENTS = {
    "event_model.visible.": "visible",
    "event_model.reconstruction.": "assignment",
    "event_model.reasoner.": "assignment",
    "event_model.feedback.": "assignment",
    "event_model.assignment_flow.": "assignment",
    "event_model.invisible.": "invisible",
    "event_model.topology_queries.": "queries",
    "event_model.parent_cfm.": "partonic_cfm",
}


def _component_for(key: str) -> str | None:
    for prefix, component in COMPONENTS.items():
        if key.startswith(prefix):
            return component
    return None


def _hash_state(module, components: set[str]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        if _component_for(key) in components:
            tensor = value.detach().cpu().contiguous()
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameter_counts(module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, parameter in module.named_parameters():
        component = _component_for(name)
        if component is not None:
            counts[component] = counts.get(component, 0) + parameter.numel()
    return counts


def _save_initial(path: Path, module, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(
            {
                "state_dict": {key: value.detach().cpu() for key, value in module.state_dict().items()},
                "hyper_parameters": dict(module.hparams),
                "invisible_scaler": None if module.scaler_metadata is None else module.scaler_metadata,
                "partonic_scaler": module.partonic_scaler_metadata,
                "initialisation_metadata": metadata,
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _build(name: str, config_path: Path):
    # Initial weights depend on the resolved architecture and train-fitted
    # scalers, not on a materialised tensor cache. Avoid loading the full
    # multi-million-event cache once per model while creating these artifacts.
    config = load_experiment(config_path, ["data.cache=null"])
    L.seed_everything(int(config.experiment.seed), workers=True)
    visible = load_visible_config(config)
    visible.dataset.pin_memory = False
    visible.dataset.persistent_workers = False
    data = build_data(config, visible, device="cpu")
    data.setup("fit")
    module = JointEventModule(
        config,
        visible,
        invisible_scaler=data.invisible_scaler,
        point_scaler=data.point_scaler,
        partonic_scaler=data.partonic_scaler,
        parent_cfm_scaler=data.partonic_scaler,
        synthetic_data=False,
    )
    del data
    return config, module


def prepare(config_dir: Path, output_dir: Path) -> dict:
    config_dir = config_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any((output_dir / f"{name}_initial.ckpt").exists() for name in CONFIGS):
        raise FileExistsError(f"Refusing to overwrite initialisation artifacts in {output_dir}")

    built = {}
    for name, filename in CONFIGS.items():
        built[name] = _build(name, config_dir / filename)

    transfer = {
        "Q0": (),
        "Q1": ("visible", "assignment", "invisible"),
        "PCFM_Q": ("visible", "queries"),
        "J0": ("visible", "assignment", "invisible", "queries"),
        "J_A": ("visible", "assignment", "invisible", "queries"),
        "J_AN": ("visible", "assignment", "invisible", "queries"),
    }
    sources = {
        "Q1": "Q0",
        "PCFM_Q": "Q1",
        "J0": "Q1",
        "J_A": "Q1",
        "J_AN": "Q1",
    }
    artifacts = {}
    canonical = {
        "Q0": output_dir / "Q0_initial.ckpt",
    }
    _save_initial(
        canonical["Q0"],
        built["Q0"][1],
        {"mode": "Q0", "seed": int(built["Q0"][0].experiment.seed), "copied": []},
    )
    for name in ("Q1", "PCFM_Q", "J0", "J_A", "J_AN"):
        config, module = built[name]
        source_name = sources[name]
        source_path = canonical.get(source_name, output_dir / f"{source_name}_initial.ckpt")
        copied = transfer[name]
        if copied:
            _load_initial_components(module, source_path, copied)
        copied_sources = [source_name] if copied else []
        metadata = {
            "mode": name,
            "seed": int(config.experiment.seed),
            "copied_components": list(copied),
            "source_initialisations": copied_sources,
        }
        target = output_dir / f"{name}_initial.ckpt"
        _save_initial(target, module, metadata)
        canonical[name] = target

    for name, (_config, module) in built.items():
        components = {
            component for component in COMPONENTS.values()
            if any(_component_for(key) == component for key in module.state_dict())
        }
        artifacts[name] = {
            "checkpoint": str(canonical[name]),
            "initialisation": (
                {"mode": name, "seed": int(_config.experiment.seed),
                 "copied_components": list(transfer[name]),
                 "source_initialisations": ([] if not transfer[name] else [sources[name]]),
                 "parent_cfm_conditioning": (
                     str(_config.model.parent_cfm.conditioning)
                     if bool(_config.model.parent_cfm.enabled)
                     else None
                 )}
            ),
            "component_sha256": {
                component: _hash_state(module, {component}) for component in sorted(components)
            },
            "parameter_counts": _parameter_counts(module),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in module.parameters() if parameter.requires_grad
            ),
        }
    for name in ("J0", "J_A", "J_AN"):
        if (
            artifacts[name]["component_sha256"].get("invisible")
            != artifacts["Q1"]["component_sha256"].get("invisible")
        ):
            raise AssertionError(f"{name} invisible initialization differs from the shared Q1 predictor")
    for name in ("Q0", "Q1"):
        shared = ("visible", "assignment", "invisible")
        for component in shared:
            if artifacts["Q0"]["component_sha256"].get(component) != artifacts["Q1"]["component_sha256"].get(component):
                raise AssertionError(f"Q0/Q1 initial {component} states differ")
    report = {
        "source_contract": "real source measured/development inputs; reserved-test events absent",
        "seed": int(built["Q0"][0].experiment.seed),
        "modes": artifacts,
        "q0_q1_shared_initialisation_equal": True,
        "q0_q1_shared_components": ["visible", "assignment", "invisible"],
        "parent_cfm_initialisation": "mode-specific seeded weights for P0/P1/P2 conditioning",
    }
    (output_dir / "initialisation_manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=Path("configs/experiments/realnu_v1"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/partonic_ttbar2L_2M_realnu_v1/cpu_e2e/initialisations"),
    )
    args = parser.parse_args()
    result = prepare(args.config_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
