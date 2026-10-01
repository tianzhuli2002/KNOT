"""One maintained CLI for training, prediction, evaluation and plotting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from omegaconf import OmegaConf

from joint_event_reconstruction.configuration import (
    load_experiment,
    load_visible_config,
)
from joint_event_reconstruction.runtime_paths import runtime_paths

if TYPE_CHECKING:
    from joint_event_reconstruction.training.joint import JointEventModule


_COMPONENT_PREFIXES = {
    "visible": ("event_model.visible.",),
    "assignment": (
        "event_model.reconstruction.",
        "event_model.reasoner.",
        "event_model.feedback.",
        "event_model.assignment_flow.",
    ),
    "invisible": ("event_model.invisible.",),
    "c2": ("event_model.posterior_feedback.",),
    "compatibility": ("event_model.compatibility.",),
    "point": ("event_model.point_head.",),
    "parent": ("event_model.parent_head.",),
    "queries": ("event_model.topology_queries.",),
    "partonic": ("event_model.partonic_head.",),
    "partonic_cfm": ("event_model.parent_cfm.",),
}


def _sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_branch() -> str | None:
    if os.environ.get("JER_SOURCE_BRANCH"):
        return os.environ["JER_SOURCE_BRANCH"]
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=runtime_paths().project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None


def _invisible_truth_contract(data) -> dict | None:
    store = data.structured_collator.invisible_store
    if store is None:
        return None
    metadata = store.metadata

    def attr(name: str):
        value = metadata.get(name)
        return value.decode("utf-8") if isinstance(value, bytes) else value

    stat = store.path.stat()
    return {
        "invisible_role_convention": attr("invisible_role_convention"),
        "invisible_role_order": list(store.role_names),
        "truth_definition": attr("neutrino_truth_definition"),
        "units": attr("units"),
        "synthetic": bool(store.synthetic),
        "source_file_sha256": attr("source_file_sha256"),
        "source_manifest_sha256": attr("source_manifest_sha256"),
        "split_membership_sha256": attr("split_membership_sha256"),
        "split_version": attr("split_version"),
        "support_sha256": attr("neutrino_role_support_sha256"),
        "source_path": attr("source_path"),
        "truth_store": {
            "path": str(store.path),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": _sha256_path(store.path),
        },
    }


def _training_runtime_metadata(source_sha: str) -> dict:
    branch = _source_branch()
    import torch

    gpu_type = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "git_sha": source_sha,
        "branch": branch,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_type": gpu_type,
    }


def _load_repo_script(relative_path: str, module_name: str):
    """Load an executable repository script without depending on a package named scripts."""
    import importlib.util

    path = runtime_paths().source(relative_path if relative_path.startswith("scripts/") else f"scripts/{relative_path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load repository script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_data(config, visible, *, include_truth: bool = True, device: str | None = None, complete_assignments: bool = False):
    from joint_event_reconstruction.data.structured import StructuredDataModule

    visible_data = OmegaConf.create(OmegaConf.to_container(visible, resolve=True))
    assignment_supervision = (
        include_truth and float(config.training.loss_weights.assignment) > 0.0
    )
    visible_truth_path = (
        config.data.visible_truth
        if assignment_supervision and config.data.visible_truth is not None
        else None
    )
    if not include_truth or visible_truth_path is not None or not assignment_supervision:
        visible_data.general.mode = "predict"
        if not include_truth:
            inference_input = config.data.get("inference_input")
            if inference_input is not None:
                visible_data.dataset.input_h5_path = str(inference_input)
        visible_data.tasks.reconstruction.enabled = False
    if device == "cpu":
        visible_data.dataset.pin_memory = False

    invisible = config.data.invisible
    if include_truth and config.model.invisible.enabled and invisible.path is None:
        raise ValueError("Invisible inference is enabled but the data config has no invisible_truth")
    build_assignments = (
        complete_assignments
        or str(config.model.mode) == "joint_flow"
        or str(config.model.assignment.formulation) == "flow_matching"
        or (
            bool(config.model.parent_cfm.enabled)
            and str(config.model.parent_cfm.conditioning) in {"P1", "P2"}
        )
    )
    return StructuredDataModule(
        visible_data,
        config.topology_config,
        truth_bindings=OmegaConf.to_container(config.data.truth_bindings, resolve=True),
        tensor_cache=config.data.tensor_cache,
        invisible_path=(
            invisible.path if include_truth and config.model.invisible.enabled else None
        ),
        point_enabled=bool(config.model.heads.point_invisible.enabled),
        parent_path=(
            config.data.parent.path
            if include_truth and config.model.heads.parent_four_vector.enabled
            else None
        ),
        parent_roles=(
            tuple(str(role) for role in config.model.heads.parent_four_vector.roles)
            if include_truth and config.model.heads.parent_four_vector.enabled
            else ()
        ),
        partonic_path=(
            config.data.partonic.path
            if include_truth
            and (
                config.model.heads.partonic.enabled
                or config.model.parent_cfm.enabled
            )
            else None
        ),
        partonic_roles=(
            tuple(str(role) for role in config.data.partonic.roles)
            if config.model.heads.partonic.enabled or config.model.parent_cfm.enabled
            else ()
        ),
        partonic_truth_bindings=(
            OmegaConf.to_container(config.data.partonic.truth_bindings, resolve=True)
            if config.model.heads.partonic.enabled or config.model.parent_cfm.enabled
            else None
        ),
        visible_truth_path=visible_truth_path,
        allow_synthetic=bool(invisible.allow_synthetic),
        max_events=OmegaConf.to_container(config.data.max_events, resolve=True),
        subset_seed=int(config.data.subset_seed),
        build_assignments=build_assignments,
        max_assignment_states=(
            int(config.data.max_assignment_states) if build_assignments else None
        ),
        coordinates_config=config.model.invisible.coordinates,
        mass_shell_config=config.model.invisible.mass_shell,
        require_truth=assignment_supervision,
    )


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _selected_component_key(key: str, components: tuple[str, ...]) -> bool:
    return any(
        key.startswith(prefix)
        for component in components
        for prefix in _COMPONENT_PREFIXES[component]
    )


def _scaler_metadata_matches(saved: dict | None, current: dict | None, *, allow_fit_roundoff: bool) -> bool:
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return saved == current
    if set(saved) != set(current):
        return False
    for key in saved:
        if allow_fit_roundoff and key in {"mean", "std"}:
            import numpy as np

            saved_values = np.asarray(saved[key], dtype=np.float64)
            current_values = np.asarray(current[key], dtype=np.float64)
            if saved_values.shape != current_values.shape or not np.allclose(
                saved_values, current_values, rtol=2e-8, atol=1e-10, equal_nan=False
            ):
                return False
        elif saved[key] != current[key]:
            return False
    return True


def _require_matching_scaler_fit(saved: dict, module, *, name: str) -> bool:
    fit_key = f"{name}_scaler_fit"
    saved_fit = saved.get(fit_key)
    current_fit = module.hparams.get(fit_key)
    if current_fit is not None and saved_fit != current_fit:
        raise ValueError(
            f"{name.capitalize()} component initialisation uses a different training fit: "
            f"checkpoint={saved_fit!r}, current={current_fit!r}"
        )
    return (
        isinstance(current_fit, dict)
        and current_fit.get("population") == "frozen_train_split"
        and saved_fit == current_fit
    )


def _load_initial_components(
    module: JointEventModule,
    checkpoint_path: str | Path,
    components: tuple[str, ...],
) -> int:
    """Load complete semantic components without relaxing checkpoint shape checks."""
    import torch

    if not components:
        raise ValueError("Initial checkpoint requires at least one component")
    unknown = sorted(set(components) - set(_COMPONENT_PREFIXES))
    if unknown:
        raise ValueError(f"Unknown initialisation components: {unknown}")

    path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("state_dict")
    if not isinstance(source, dict):
        raise TypeError(f"Initial checkpoint has no state_dict: {path}")

    saved = checkpoint.get("hyper_parameters", {})
    for key in ("topology_signature", "input_features", "source_type_ids"):
        if saved.get(key) != module.hparams.get(key):
            raise ValueError(
                f"Initial checkpoint has incompatible {key}: "
                f"checkpoint={saved.get(key)!r}, current={module.hparams.get(key)!r}"
            )
    source_signatures = saved.get("architecture_component_signatures")
    target_signatures = module.hparams.get("architecture_component_signatures")
    if not isinstance(source_signatures, dict) or not isinstance(target_signatures, dict):
        raise TypeError(
            "Initial checkpoint is missing architecture component signatures"
        )
    incompatible_components = [
        component
        for component in components
        if source_signatures.get(component) != target_signatures.get(component)
    ]
    if incompatible_components:
        raise ValueError(
            "Initial checkpoint has incompatible architecture component signatures: "
            f"{incompatible_components}"
        )
    if "invisible" in components:
        allow_fit_roundoff = _require_matching_scaler_fit(saved, module, name="invisible")
        if not _scaler_metadata_matches(
            checkpoint.get("invisible_scaler"),
            module.scaler_metadata,
            allow_fit_roundoff=allow_fit_roundoff,
        ):
            raise ValueError(
                "Invisible component initialisation has incompatible scaler metadata; "
                "role definitions and training-fit identity must match exactly"
            )
    if "partonic_cfm" in components:
        allow_fit_roundoff = _require_matching_scaler_fit(saved, module, name="partonic")
        if not _scaler_metadata_matches(
            checkpoint.get("partonic_scaler"),
            module.partonic_scaler_metadata,
            allow_fit_roundoff=allow_fit_roundoff,
        ):
            raise ValueError(
                "Parent CFM initialisation has incompatible train-fitted scaler metadata; "
                "role definitions and training-fit identity must match exactly"
            )

    target = module.state_dict()
    expected = {key for key in target if _selected_component_key(key, components)}
    available = {key for key in source if _selected_component_key(key, components)}
    missing = sorted(expected - available)
    unexpected = sorted(available - expected)
    mismatched = sorted(
        key
        for key in expected & available
        if tuple(source[key].shape) != tuple(target[key].shape)
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            "Initial checkpoint component mismatch: "
            f"components={list(components)}, missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}, shape_mismatch={mismatched[:20]}"
        )
    if not expected:
        raise ValueError(
            f"Selected initialisation components have no parameters/buffers: {components}"
        )

    module.load_state_dict({key: source[key] for key in expected}, strict=False)
    module.hparams["initial_checkpoint"] = str(path)
    module.hparams["initial_components"] = list(components)
    return len(expected)


def train(
    config,
    *,
    device: str = "auto",
    initial_checkpoint: str | None = None,
    initial_components: tuple[str, ...] = (),
    public_config_path: str | None = None,
    resume_from: str | None = None,
) -> None:
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    from joint_event_reconstruction.training.joint import JointEventModule

    torch.set_float32_matmul_precision("high")
    L.seed_everything(int(config.experiment.seed), workers=True)
    visible = load_visible_config(config)
    data = build_data(config, visible, device=device)
    data.setup("fit")
    module = JointEventModule(
        config,
        visible,
        invisible_scaler=data.invisible_scaler,
        point_scaler=data.point_scaler,
        synthetic_data=(
            bool(data.structured_collator.invisible_store.synthetic)
            if data.structured_collator.invisible_store is not None
            else bool(
                data.structured_collator.partonic_store is not None
                and data.structured_collator.partonic_store.synthetic
            )
        ),
        partonic_scaler=data.partonic_scaler,
    )
    output_dir = Path(config.training.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    from joint_event_reconstruction.inputs import encoder_input_contract
    from joint_event_reconstruction.training.joint import active_architecture
    if public_config_path is not None:
        public_source = Path(public_config_path).expanduser().resolve()
        (output_dir / "config.public.yaml").write_text(public_source.read_text())
    resolved_config_text = OmegaConf.to_yaml(config, resolve=True)
    (output_dir / "config.resolved.yaml").write_text(resolved_config_text)
    (output_dir / "architecture.json").write_text(
        json.dumps(active_architecture(config.model), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "inputs.json").write_text(
        json.dumps(encoder_input_contract(config), indent=2, sort_keys=True) + "\n"
    )
    git_sha = runtime_paths().source_sha
    runtime_metadata = _training_runtime_metadata(git_sha)
    resolved_config_sha256 = hashlib.sha256(resolved_config_text.encode("utf-8")).hexdigest()
    public_config_sha256 = (
        _sha256_path(public_config_path) if public_config_path is not None else None
    )
    module.hparams["git_sha"] = git_sha
    module.hparams["branch"] = runtime_metadata["branch"]
    module.hparams["slurm_job_id"] = runtime_metadata["slurm_job_id"]
    module.hparams["gpu_type"] = runtime_metadata["gpu_type"]
    module.hparams["resolved_config_sha256"] = resolved_config_sha256
    module.hparams["public_config_sha256"] = public_config_sha256
    if config.data.tensor_cache is None:
        # Small synthetic and contract fixtures deliberately run without a
        # materialised tensor cache.  Preserve the same provenance fields as
        # cached runs, using the immutable input file and resolved split
        # metadata as the dataset identity instead of pretending a cache
        # exists.
        source_path = Path(config.data.input).expanduser().resolve()
        source_stat = source_path.stat()
        cache_metadata = {
            "cache_key": None,
            "source_identity": {
                "path": str(source_path),
                "size_bytes": int(source_stat.st_size),
                "mtime_ns": int(source_stat.st_mtime_ns),
            },
            "cache_kind": "uncached_hdf5",
        }
    else:
        cache_manifest_path = Path(config.data.tensor_cache).with_suffix(".json")
        cache_metadata = json.loads(cache_manifest_path.read_text())
    module.hparams["cache_metadata"] = cache_metadata
    module.hparams["dataset_identity"] = cache_metadata.get("source_identity")
    module.hparams["split_identity"] = data.split_metadata
    if data.structured_collator.partonic_store is not None:
        truth_metadata = data.structured_collator.partonic_store.metadata
        module.hparams["partonic_truth_contract"] = {
            "truth_definition": truth_metadata.get("partonic_truth_definition"),
            "units": truth_metadata.get("partonic_units"),
            "roles": list(data.structured_collator.partonic_store.role_names),
            "truth_bindings": dict(data.structured_collator.partonic_store.truth_bindings),
            "source_manifest_sha256": truth_metadata.get("source_manifest_sha256"),
            "source_file_sha256": truth_metadata.get("source_file_sha256"),
            "split_membership_sha256": truth_metadata.get("split_membership_sha256"),
            "split_version": truth_metadata.get("split_version"),
        }
        module.hparams["partonic_scaler_fit"] = data.partonic_scaler_fit_metadata
    invisible_contract = _invisible_truth_contract(data)
    if invisible_contract is not None:
        module.hparams["invisible_truth_contract"] = invisible_contract
        module.hparams["invisible_scaler_fit"] = data.invisible_scaler_fit_metadata
    if data.structured_collator.visible_store is not None:
        module.hparams["visible_truth_metadata"] = dict(
            data.structured_collator.visible_store.metadata
        )
    if resume_from is not None:
        resume_checkpoint = torch.load(
            Path(resume_from).expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        module.validate_checkpoint_metadata(resume_checkpoint)
        _validate_checkpoint_data_contract(resume_checkpoint, config, data)
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha,
                "branch": runtime_metadata["branch"],
                "slurm_job_id": runtime_metadata["slurm_job_id"],
                "gpu_type": runtime_metadata["gpu_type"],
                "resolved_config_sha256": resolved_config_sha256,
                "public_config_sha256": public_config_sha256,
                "experiment_seed": int(config.experiment.seed),
                "data_subset_seed": int(config.data.subset_seed),
                "precision": str(config.training.precision),
                "trainable_components": list(config.training.trainable),
                "cache_fingerprint": cache_metadata.get("cache_key"),
                "cache_metadata": cache_metadata,
                "dataset_identity": cache_metadata.get("source_identity"),
                "split_identity": data.split_metadata,
                "visible_truth_metadata": module.hparams.get("visible_truth_metadata"),
                "invisible_truth_contract": invisible_contract,
                "invisible_scaler": module.scaler_metadata,
                "invisible_scaler_fit": data.invisible_scaler_fit_metadata,
                "partonic_truth_contract": module.hparams.get("partonic_truth_contract"),
                "external_package_versions": module.hparams["external_package_versions"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if initial_checkpoint is not None:
        loaded = _load_initial_components(module, initial_checkpoint, initial_components)
        print(
            f"Initialised {loaded} state entries from {initial_checkpoint} "
            f"for components={list(initial_components)}"
        )
    elif initial_components:
        raise ValueError("--initial-components requires --initial-checkpoint")

    callbacks = [
        ModelCheckpoint(
            dirpath=Path(config.training.output) / "checkpoints",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            enable_version_counter=False,
            filename="best-loss",
        )
    ]
    if config.training.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=int(config.training.early_stopping.patience),
            )
        )
    accelerator = {"auto": "auto", "cpu": "cpu", "cuda": "gpu"}[device]
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=int(config.training.epochs),
        precision=str(config.training.precision),
        gradient_clip_val=float(config.training.gradient_clip),
        accumulate_grad_batches=int(config.training.accumulate_grad_batches),
        callbacks=callbacks,
        default_root_dir=config.training.output,
    )
    trainer.fit(module, datamodule=data, ckpt_path=resume_from)


def _validate_checkpoint_data_contract(
    checkpoint: dict, config, data, *, allow_reserved_test: bool = False
) -> None:
    saved_hparams = checkpoint.get("hyper_parameters", {})
    saved_split = saved_hparams.get("split_identity")
    current_split = data.split_metadata
    if saved_split is not None:
        if current_split is None:
            raise ValueError("Checkpoint contains a frozen split identity but input has none")
        for key in ("event_id_hashes", "event_id_counts", "source_contract"):
            expected = saved_split.get(key)
            observed = current_split.get(key)
            if expected != observed and not (
                allow_reserved_test
                and key in {"event_id_hashes", "event_id_counts"}
            ):
                raise ValueError(
                    "Input does not match checkpoint event/split identity: "
                    f"{key} checkpoint={expected!r}, current={observed!r}"
                )
        if saved_split.get("event_id_hashes") != current_split.get("event_id_hashes"):
            saved_counts = saved_split.get("event_id_counts", {})
            current_counts = current_split.get("event_id_counts", {})
            is_reserved_test_input = (
                allow_reserved_test
                and saved_split.get("policy") == "canonical_hdf5_SPLIT"
                and current_split.get("policy") == "canonical_hdf5_SPLIT"
                and int(saved_counts.get("train", 0)) > 0
                and int(saved_counts.get("validation", 0)) > 0
                and int(saved_counts.get("test", 0)) == 0
                and int(current_counts.get("train", 0)) == 0
                and int(current_counts.get("validation", 0)) == 0
                and int(current_counts.get("test", 0)) > 0
                and saved_split.get("source_contract") == current_split.get("source_contract")
            )
            if not is_reserved_test_input:
                raise ValueError(
                    "Prediction event IDs are outside the checkpoint train/validation "
                    "population and are not a separately identified reserved-test input"
                )
    if bool(config.model.invisible.enabled):
        invisible_contract = saved_hparams.get("invisible_truth_contract")
        if not isinstance(invisible_contract, dict):
            raise ValueError("Invisible checkpoint is missing its source role contract")
        expected_roles = [role.key for role in data.structured_collator.spec.invisible_roles]
        if invisible_contract.get("invisible_role_order") != expected_roles:
            raise ValueError(
                "Invisible checkpoint role order differs from the topology: "
                f"checkpoint={invisible_contract.get('invisible_role_order')!r}, "
                f"topology={expected_roles!r}"
            )
        if expected_roles == ["top.nu", "antitop.nu"] and not bool(
            invisible_contract.get("synthetic", False)
        ):
            if (
                invisible_contract.get("invisible_role_convention") != "top_antitop"
                or invisible_contract.get("truth_definition")
                != "generator top-side neutrino and generator antitop-side neutrino"
                or invisible_contract.get("units") != "GeV"
            ):
                raise ValueError("Invisible checkpoint does not declare the real top/antitop neutrino contract")
            source_contract = (current_split or {}).get("source_contract", {})
            for key in ("source_manifest_sha256", "split_membership_sha256", "split_version"):
                if invisible_contract.get(key) != source_contract.get(key):
                    raise ValueError(
                        "Invisible checkpoint source/split identity differs from measured input: "
                        f"{key} checkpoint={invisible_contract.get(key)!r}, "
                        f"input={source_contract.get(key)!r}"
                    )
    saved_visible = saved_hparams.get("visible_truth_metadata")
    current_visible = data.structured_collator.visible_store
    if current_visible is not None:
        if not isinstance(saved_visible, dict):
            raise ValueError("Checkpoint is missing the visible assignment truth-store metadata")
        current_metadata = current_visible.metadata
        for key in (
            "path",
            "size_bytes",
            "mtime_ns",
            "source_names",
            "source_manifest_sha256",
            "split_membership_sha256",
            "split_version",
            "max_jets",
        ):
            if saved_visible.get(key) != current_metadata.get(key):
                raise ValueError(
                    "Visible assignment truth sidecar differs from the checkpoint: "
                    f"{key} checkpoint={saved_visible.get(key)!r}, "
                    f"current={current_metadata.get(key)!r}"
                )
    if not bool(config.model.parent_cfm.enabled):
        return
    contract = saved_hparams.get("partonic_truth_contract")
    if not isinstance(contract, dict):
        raise TypeError("Partonic CFM checkpoint is missing its truth contract")
    expected_bindings = {
        str(key): str(value)
        for key, value in config.model.parent_cfm.truth_bindings.items()
    }
    if contract.get("truth_definition") != "partonic_after_fsr":
        raise ValueError(
            "Partonic CFM checkpoint does not declare genuine partonic_after_fsr truth: "
            f"{contract.get('truth_definition')!r}"
        )
    if contract.get("units") != "GeV":
        raise ValueError(
            f"Partonic CFM checkpoint target units must be GeV, found {contract.get('units')!r}"
        )
    if contract.get("roles") != ["top", "antitop"] or contract.get("truth_bindings") != expected_bindings:
        raise ValueError(
            "Partonic CFM checkpoint target roles/bindings differ from the current configuration: "
            f"checkpoint_roles={contract.get('roles')!r}, "
            f"checkpoint_bindings={contract.get('truth_bindings')!r}, "
            f"current_bindings={expected_bindings!r}"
        )
    source_contract = (current_split or {}).get("source_contract", {})
    for key in ("source_manifest_sha256", "split_membership_sha256", "split_version"):
        expected = contract.get(key)
        observed = source_contract.get(key)
        if expected is None or expected != observed:
            raise ValueError(
                "Partonic CFM checkpoint target/source identity differs from measured input: "
                f"{key} checkpoint={expected!r}, input={observed!r}"
            )


def load_trained(config, visible, data, checkpoint_path: str) -> JointEventModule:
    import torch

    from joint_event_reconstruction.data.invisible import InvisibleScaler
    from joint_event_reconstruction.training.joint import JointEventModule

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _validate_checkpoint_data_contract(
        checkpoint, config, data, allow_reserved_test=True
    )
    checkpoint_scaler = checkpoint.get("invisible_scaler")
    if checkpoint_scaler is not None:
        scaler = InvisibleScaler.from_dict(checkpoint_scaler)
        data.invisible_scaler = scaler
        data.structured_collator.invisible_scaler = scaler
    else:
        scaler = data.invisible_scaler
    checkpoint_point_scaler = checkpoint.get("point_scaler")
    if checkpoint_point_scaler is not None:
        point_scaler = InvisibleScaler.from_dict(checkpoint_point_scaler)
        data.point_scaler = point_scaler
        data.structured_collator.point_scaler = point_scaler
    else:
        point_scaler = data.point_scaler

    from joint_event_reconstruction.data.partonic import PartonicScaler
    checkpoint_partonic_scaler = checkpoint.get("partonic_scaler")
    if checkpoint_partonic_scaler is not None:
        partonic_scaler = PartonicScaler.from_dict(checkpoint_partonic_scaler)
        data.partonic_scaler = partonic_scaler
        data.structured_collator.partonic_scaler = partonic_scaler
    else:
        partonic_scaler = data.partonic_scaler

    module = JointEventModule(
        config,
        visible,
        invisible_scaler=scaler,
        point_scaler=point_scaler,
        partonic_scaler=partonic_scaler,
        parent_cfm_scaler=partonic_scaler,
        synthetic_data=(
            bool(data.structured_collator.invisible_store.synthetic)
            if data.structured_collator.invisible_store is not None
            else bool(
                data.structured_collator.partonic_store is not None
                and data.structured_collator.partonic_store.synthetic
            )
        ),
    )
    saved_hparams = checkpoint.get("hyper_parameters", {})
    for key in (
        "git_sha",
        "branch",
        "slurm_job_id",
        "gpu_type",
        "resolved_config_sha256",
        "public_config_sha256",
        "cache_metadata",
        "dataset_identity",
        "split_identity",
        "partonic_truth_contract",
        "partonic_scaler_fit",
        "invisible_truth_contract",
        "invisible_scaler_fit",
    ):
        if key in saved_hparams:
            module.hparams[key] = saved_hparams[key]
    module.validate_checkpoint_metadata(checkpoint)
    module.load_state_dict(checkpoint["state_dict"], strict=True)
    return module


def predict(config, args) -> None:
    import h5py
    import lightning as L
    import torch

    from joint_event_reconstruction.prediction import predict_hdf5
    from joint_event_reconstruction.prediction_discrete import (
        predict_discrete_flow_hdf5,
    )
    from joint_event_reconstruction.prediction_joint import predict_joint_flow_hdf5

    torch.set_float32_matmul_precision("high")
    prediction_seed = int(config.inference.seed)
    L.seed_everything(prediction_seed, workers=True)
    device = _resolve_device(args.device)
    visible = load_visible_config(config)
    data = build_data(config, visible, include_truth=False, device=device)
    data.setup("predict")
    module = load_trained(config, visible, data, args.checkpoint)

    mode = str(config.model.mode)
    assignment_formulation = str(config.model.assignment.formulation)
    if mode == "joint_flow":
        predict_joint_flow_hdf5(
            module,
            data,
            data.invisible_scaler,
            args.output,
            split=args.split,
            posterior_samples=int(config.inference.posterior_samples),
            steps=int(config.inference.joint_flow.steps),
            sample_chunk_size=int(config.inference.joint_flow.sample_chunk_size),
            device=device,
            prediction_seed=prediction_seed,
            checkpoint_path=args.checkpoint,
        )
    elif assignment_formulation == "flow_matching":
        predict_discrete_flow_hdf5(
            module,
            data,
            args.output,
            split=args.split,
            posterior_samples=int(config.inference.posterior_samples),
            steps=int(config.inference.assignment_flow.steps),
            device=device,
            prediction_seed=prediction_seed,
            checkpoint_path=args.checkpoint,
        )
    else:
        reasoning = config.model.assignment.reasoning
        predict_hdf5(
            module,
            data,
            data.invisible_scaler,
            args.output,
            split=args.split,
            top_k=int(config.inference.top_k_assignments),
            posterior_samples=int(config.inference.posterior_samples),
            recurrent_steps=(
                args.recurrent_steps
                if args.recurrent_steps is not None
                else int(reasoning.eval_steps)
                if reasoning.enabled
                else None
            ),
            device=device,
            prediction_seed=prediction_seed,
            checkpoint_path=args.checkpoint,
            compatibility_weight=float(config.inference.compatibility_weight),
            point_estimator=str(config.inference.invisible.point_estimator),
        )

    with h5py.File(Path(args.output).expanduser().resolve(), "r+") as output:
        output.attrs["prediction_seed"] = prediction_seed
        truth_contract = module.hparams.get("invisible_truth_contract")
        if isinstance(truth_contract, dict):
            output.attrs["invisible_role_convention"] = truth_contract["invisible_role_convention"]
            output.attrs["invisible_role_order"] = json.dumps(truth_contract["invisible_role_order"])
            output.attrs["neutrino_truth_definition"] = truth_contract["truth_definition"]
            output.attrs["invisible_units"] = truth_contract["units"]
            output.attrs["source_file_sha256"] = truth_contract["source_file_sha256"]
            output.attrs["source_manifest_sha256"] = truth_contract["source_manifest_sha256"]
            output.attrs["split_membership_sha256"] = truth_contract["split_membership_sha256"]
            output.attrs["truth_store_identity"] = json.dumps(
                truth_contract["truth_store"], sort_keys=True
            )
            output.attrs["measured_cache_identity"] = json.dumps(
                module.hparams.get("cache_metadata", {}), sort_keys=True
            )
            output.attrs["training_git_sha"] = str(module.hparams.get("git_sha", ""))
            output.attrs["training_branch"] = str(module.hparams.get("branch") or "")
            output.attrs["training_config_sha256"] = str(
                module.hparams.get("resolved_config_sha256", "")
            )
            output.attrs["invisible_scaler_identity"] = hashlib.sha256(
                json.dumps(module.scaler_metadata, sort_keys=True).encode("utf-8")
            ).hexdigest()
        output.attrs["prediction_git_sha"] = runtime_paths().source_sha
        output.attrs["prediction_branch"] = _source_branch() or ""
        output.attrs["prediction_slurm_job_id"] = os.environ.get("SLURM_JOB_ID", "")
        output.attrs["prediction_gpu_type"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )


def predict_partonic(config, args) -> None:
    """Reload a maintained checkpoint and write truth-free parent samples."""
    import lightning as L
    import torch

    from joint_event_reconstruction.prediction_partonic import predict_partonic_hdf5

    if not bool(config.model.parent_cfm.enabled):
        raise ValueError("predict-partonic requires model.parent_cfm.enabled=true")
    device = _resolve_device(args.device)
    seed = int(config.inference.seed if args.seed is None else args.seed)
    samples = int(
        config.inference.posterior_samples if args.samples is None else args.samples
    )
    steps = int(
        config.model.parent_cfm.sampling_steps if args.steps is None else args.steps
    )
    solver = str(
        config.model.parent_cfm.sampling_solver if args.solver is None else args.solver
    )
    torch.set_float32_matmul_precision("high")
    L.seed_everything(seed, workers=True)
    visible = load_visible_config(config)
    data = build_data(config, visible, include_truth=False, device=device)
    data.setup("predict")
    module = load_trained(config, visible, data, args.checkpoint)
    result = predict_partonic_hdf5(
        module,
        data,
        args.output,
        split=args.split,
        samples=samples,
        steps=steps,
        solver=solver,
        seed=seed,
        device=device,
        checkpoint_path=args.checkpoint,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def _plot_predictions(
    paths: list[str],
    output: str,
    labels: list[str] | None,
    formats: tuple[str, ...],
) -> None:
    from joint_event_reconstruction.plotting import plot_comparison, plot_prediction

    output_dir = Path(output).expanduser().resolve()
    if len(paths) == 1:
        written = plot_prediction(paths[0], output_dir, formats=formats)
    else:
        written = []
        for path in paths:
            written.extend(
                plot_prediction(path, output_dir / Path(path).stem, formats=formats)
            )
        written.extend(
            plot_comparison(
                paths,
                output_dir / "comparison",
                labels=labels,
                formats=formats,
            )
        )
    print(f"Wrote {len(written)} evaluation/plot products to {output_dir}")


def _experiment_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override one compact public setting using OmegaConf dot notation; repeat as needed.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Topology-aware event reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train one configured experiment")
    _experiment_options(train_parser)
    train_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_parser.add_argument("--force-rebuild-cache", action="store_true")
    train_parser.add_argument(
        "--initial-checkpoint",
        help="Initialise selected semantic components before training.",
    )
    train_parser.add_argument(
        "--initial-components",
        nargs="+",
        choices=tuple(_COMPONENT_PREFIXES),
        default=(),
        help="Components copied exactly from --initial-checkpoint.",
    )
    train_parser.add_argument(
        "--resume-from",
        help="Resume the same run, including optimiser and scheduler state, from a checkpoint.",
    )

    predict_parser = commands.add_parser(
        "predict", help="write a self-describing prediction HDF5"
    )
    _experiment_options(predict_parser)
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    predict_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    predict_parser.add_argument("--recurrent-steps", type=int)

    partonic_predict_parser = commands.add_parser(
        "predict-partonic", help="write truth-free joint top/antitop CFM samples"
    )
    _experiment_options(partonic_predict_parser)
    partonic_predict_parser.add_argument("--checkpoint", required=True)
    partonic_predict_parser.add_argument("--output", required=True)
    partonic_predict_parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="validation"
    )
    partonic_predict_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    partonic_predict_parser.add_argument("--samples", type=int)
    partonic_predict_parser.add_argument("--steps", type=int)
    partonic_predict_parser.add_argument("--solver", choices=("euler", "heun"))
    partonic_predict_parser.add_argument("--seed", type=int)

    evaluate_parser = commands.add_parser(
        "evaluate", help="evaluate one prediction HDF5; no experiment YAML required"
    )
    evaluate_parser.add_argument("--predictions", required=True)

    evaluate_partonic_parser = commands.add_parser(
        "evaluate-partonic", help="join parent samples to explicit after-FSR truth"
    )
    evaluate_partonic_parser.add_argument("--predictions", required=True)
    evaluate_partonic_parser.add_argument("--truth", required=True)
    evaluate_partonic_parser.add_argument("--output")

    plot_parser = commands.add_parser(
        "plot", help="render diagnostics for one or more prediction HDF5 files"
    )
    plot_parser.add_argument("--predictions", nargs="+", required=True)
    plot_parser.add_argument("--output", required=True)
    plot_parser.add_argument("--labels", nargs="+")
    plot_parser.add_argument(
        "--formats", nargs="+", choices=("png", "pdf"), default=("png", "pdf")
    )

    smoke_parser = commands.add_parser(
        "smoke", help="run the maintained software/mechanism smoke suite"
    )
    smoke_parser.add_argument("--experiment", required=True)
    smoke_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    smoke_parser.add_argument("--events", type=int, default=8)

    validate_parser = commands.add_parser(
        "validate-data", help="validate the configured canonical invisible HDF5"
    )
    _experiment_options(validate_parser)

    show_parser = commands.add_parser(
        "show-config", help="validate and print the fully resolved internal experiment"
    )
    _experiment_options(show_parser)

    config_parser = commands.add_parser("config", help="validate or show an experiment")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    for name in ("validate", "show"):
        command = config_commands.add_parser(name)
        command.add_argument("path")

    inputs_parser = commands.add_parser("inputs", help="inspect encoder information")
    inputs_commands = inputs_parser.add_subparsers(dest="inputs_command", required=True)
    inputs_show = inputs_commands.add_parser("show")
    inputs_show.add_argument("path")
    inputs_compare = inputs_commands.add_parser("compare")
    inputs_compare.add_argument("path")

    cache_parser = commands.add_parser("cache", help="inspect or build the tensor cache")
    cache_commands = cache_parser.add_subparsers(dest="cache_command", required=True)
    cache_show = cache_commands.add_parser("show")
    cache_show.add_argument("path")
    cache_build = cache_commands.add_parser("build")
    cache_build.add_argument("path")
    cache_build.add_argument("--force", action="store_true")

    topology_parser = commands.add_parser("topology", help="inspect semantic topology")
    topology_commands = topology_parser.add_subparsers(dest="topology_command", required=True)
    topology_show = topology_commands.add_parser("show")
    topology_show.add_argument("path")

    checkpoint_parser = commands.add_parser("checkpoint", help="inspect checkpoint metadata")
    checkpoint_commands = checkpoint_parser.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_show = checkpoint_commands.add_parser("show")
    checkpoint_show.add_argument("path")

    study_parser = commands.add_parser("study", help="inspect a study or campaign")
    study_commands = study_parser.add_subparsers(dest="study_command", required=True)
    study_plan = study_commands.add_parser("plan")
    study_plan.add_argument("path")
    study_status = study_commands.add_parser("status")
    study_status.add_argument("path")
    return parser


def _checkpoint_summary(path: str) -> dict:
    import torch

    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = checkpoint.get("hyper_parameters", {})
    architecture = saved.get("architecture", {})
    return {
        "path": str(checkpoint_path),
        "git_sha": saved.get("git_sha"),
        "topology_signature": saved.get("topology_signature"),
        "architecture_signature": saved.get("architecture_signature"),
        "visible_encoder": saved.get("visible_encoder"),
        "encoder_input_contract": saved.get("encoder_input_contract"),
        "architecture": architecture,
        "trainable_components": saved.get("trainable_components"),
        "training_event_fingerprint": saved.get("training_event_fingerprint"),
        "training_seed": saved.get("experiment_seed"),
        "data_subset_seed": saved.get("data_subset_seed"),
        "precision": saved.get("precision"),
        "external_package_versions": saved.get("external_package_versions"),
    }


def _study_status(path: str) -> dict:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        submission = candidate
    else:
        matches = sorted(candidate.glob("**/submission.json"))
        if not matches:
            raise FileNotFoundError(f"No submission.json found under {candidate}")
        submission = matches[-1]
    payload = json.loads(submission.read_text())
    return {"submission": str(submission), **payload}


def _main() -> None:
    args = build_parser().parse_args()

    if args.command == "config":
        config = load_experiment(args.path)
        if args.config_command == "show":
            print(OmegaConf.to_yaml(config, resolve=True))
        else:
            print(f"VALID: {Path(args.path).resolve()}")
        return
    if args.command == "inputs":
        from joint_event_reconstruction.inputs import (
            assert_study12_fairness,
            format_input_contract,
            load_input_contract,
            study_input_contracts,
        )

        if args.inputs_command == "show":
            print(format_input_contract(load_input_contract(args.path)))
        else:
            contracts = study_input_contracts(args.path)
            assert_study12_fairness(contracts)
            print(json.dumps(contracts, indent=2, sort_keys=True))
        return
    if args.command == "cache":
        config = load_experiment(args.path)
        cache_path = Path(config.data.tensor_cache)
        manifest_path = cache_path.with_suffix(".json")
        if args.cache_command == "show":
            state = "MISS"
            manifest = {}
            if cache_path.exists() and manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    identity = manifest.get("source_identity", {})
                    input_stat = Path(config.data.input).stat()
                    compatible = (
                        manifest.get("cache_version") == 9
                        and identity.get("path") == str(Path(config.data.input).resolve())
                        and identity.get("size_bytes") == input_stat.st_size
                    )
                    state = "HIT" if compatible else "MISMATCH"
                except (OSError, json.JSONDecodeError):
                    state = "MISMATCH"
            print(json.dumps({
                "input": str(config.data.input), "cache": str(cache_path),
                "state": state, "cache_version": manifest.get("cache_version"),
                "event_count": manifest.get("n_events"),
                "dataset_fingerprint": manifest.get("source_identity"),
                "input_fingerprint": manifest.get("cache_key"),
                "size_bytes": cache_path.stat().st_size if cache_path.is_file() else None,
            }, indent=2, sort_keys=True))
        else:
            config.data.force_rebuild_cache = bool(args.force)
            visible = load_visible_config(config)
            data = build_data(config, visible)
            data.setup("fit")
            print(f"Cache built: {cache_path}")
        return
    if args.command == "topology":
        from joint_event_reconstruction.topology.compiler import load_topology

        spec = load_topology(args.path)
        print(json.dumps({
            "signature": spec.signature,
            "roots": spec.roots,
            "dag_order": spec.dag_order,
            "roles": {
                key: {"source": role.source, "truth": role.truth_names}
                for key, role in spec.leaf_roles.items()
            },
        }, indent=2, sort_keys=True))
        return
    if args.command == "checkpoint":
        print(json.dumps(_checkpoint_summary(args.path), indent=2, sort_keys=True))
        return
    if args.command == "study":
        if args.study_command == "status":
            print(json.dumps(_study_status(args.path), indent=2, sort_keys=True))
        else:
            from joint_event_reconstruction.studies import load_study

            study = load_study(args.path)
            print(json.dumps({
                "name": study.name,
                "split": study.split,
                "topologies": study.topologies,
                "seeds": study.seeds,
                "dependency_order": [variant.name for variant in study.dependency_order()],
                "variants": {
                    variant.name: [prediction.name for prediction in variant.predictions]
                    for variant in study.variants
                },
            }, indent=2, sort_keys=True))
        return

    if args.command == "evaluate":
        from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions

        print(json.dumps(evaluate_any_predictions(args.predictions), indent=2, sort_keys=True))
        return
    if args.command == "evaluate-partonic":
        from joint_event_reconstruction.prediction_partonic import (
            evaluate_partonic_hdf5,
        )

        print(json.dumps(
            evaluate_partonic_hdf5(args.predictions, args.truth, args.output),
            indent=2,
            sort_keys=True,
        ))
        return
    if args.command == "plot":
        if args.labels is not None and len(args.labels) != len(args.predictions):
            raise ValueError("--labels must match the number of prediction files")
        _plot_predictions(args.predictions, args.output, args.labels, tuple(args.formats))
        return

    overrides = getattr(args, "overrides", [])
    config = load_experiment(args.experiment, overrides)
    if args.command == "show-config":
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    if args.command == "train":
        config.data.force_rebuild_cache = bool(args.force_rebuild_cache)
        if bool(args.initial_checkpoint) != bool(args.initial_components):
            raise ValueError(
                "--initial-checkpoint and --initial-components must be supplied together"
            )
        train(
            config,
            device=args.device,
            initial_checkpoint=args.initial_checkpoint,
            initial_components=tuple(args.initial_components),
            public_config_path=args.experiment,
            resume_from=args.resume_from,
        )
        return
    if args.command == "predict":
        predict(config, args)
        return
    if args.command == "predict-partonic":
        predict_partonic(config, args)
        return
    if args.command == "validate-data":
        validator = _load_repo_script("data/validate_joint_h5.py", "jer_validate_joint_h5")

        if config.data.invisible.path is None:
            raise ValueError("validate-data requires invisible_truth in the JER data config")
        print(json.dumps(validator.validate(Path(config.data.invisible.path)), indent=2, sort_keys=True))
        return
    if args.command == "smoke":
        smoke_all = _load_repo_script("smoke_all.py", "jer_smoke_all")
        smoke_flows = _load_repo_script("smoke_flows.py", "jer_smoke_flows")

        device = _resolve_device(args.device)
        import torch

        torch.set_float32_matmul_precision("high")
        real_data = config.data.tensor_cache is None
        smoke_all.run_experiment_smoke(
            Path(args.experiment), device=device, events=args.events, real_data=real_data
        )
        smoke_flows.run_flow_smoke(
            Path(args.experiment), device=device, events=args.events, real_data=real_data
        )
        return
    raise RuntimeError(f"Unhandled command {args.command!r}")


def main() -> None:
    try:
        _main()
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        raise SystemExit(f"jer: {error}") from None


if __name__ == "__main__":
    main()
