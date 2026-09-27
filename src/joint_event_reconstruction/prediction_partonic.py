"""Truth-free prediction and validation for joint partonic-parent CFM samples."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch

from joint_event_reconstruction.data.partonic import PartonicTruthStore

PARTONIC_PREDICTION_SCHEMA = "joint_event_reconstruction_partonic_cfm_v1"
PARENT_ROLES = ("top", "antitop")


def _open_validation_file(path: Path) -> h5py.File:
    """Read the job-private, closed temporary without shared-filesystem locking."""
    return h5py.File(path, "r", locking=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


def _extendible(group, name: str, tail: tuple[int, ...], dtype) -> h5py.Dataset:
    shape = tuple(int(value) for value in tail)
    return group.create_dataset(
        name,
        (0,) + shape,
        maxshape=(None,) + shape,
        dtype=dtype,
        chunks=True,
    )


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    values = np.asarray(values)
    if values.ndim != dataset.ndim or tuple(values.shape[1:]) != tuple(dataset.shape[1:]):
        raise ValueError(
            f"Partonic prediction append shape mismatch for {dataset.name}: "
            f"expected [N,{dataset.shape[1:]}], observed {values.shape}"
        )
    start = len(dataset)
    dataset.resize(start + len(values), axis=0)
    dataset[start:] = values


def _loader(data, split: str):
    choices = {
        "train": data.train_dataloader,
        "validation": data.val_dataloader,
        "test": data.test_dataloader,
    }
    try:
        result = choices[split]()
    except KeyError as error:
        raise ValueError(f"Unsupported partonic prediction split {split!r}") from error
    if len(result.dataset) == 0:
        raise ValueError(f"Partonic prediction split {split!r} is empty")
    return result


def _conditioning_metadata(conditioning: str) -> dict[str, str | None]:
    assignment = (
        "predicted_soft_complete_assignment_marginal_with_null_embeddings"
        if conditioning in {"P1", "P2"}
        else "none"
    )
    invisible = (
        "predicted_CFM_sample_top_antitop_role_order_one_sample_per_event"
        if conditioning == "P2"
        else "none"
    )
    return {
        "assignment_context": assignment,
        "invisible_context": invisible,
        "hypothesis_index_semantics": (
            "-1 denotes a soft marginal; no discrete assignment hypothesis was sampled"
        ),
        "invisible_sample_index_semantics": (
            "0 denotes the single predicted invisible sample reused across parent samples"
            if conditioning == "P2"
            else "-1 denotes no invisible sample conditioning"
        ),
    }


def _validate_prediction_file(path: Path, expected_events: int, samples: int) -> None:
    with _open_validation_file(path) as source:
        if source.attrs.get("prediction_schema") != PARTONIC_PREDICTION_SCHEMA:
            raise ValueError(f"Invalid partonic prediction schema in {path}")
        if json.loads(str(source.attrs.get("parent_roles", "[]"))) != list(PARENT_ROLES):
            raise ValueError(f"Invalid parent-role metadata in {path}")
        raw_order = source.attrs.get("parent_role_order")
        if isinstance(raw_order, bytes):
            raw_order = raw_order.decode("utf-8")
        if (
            source.attrs.get("parent_role_convention") != "top_antitop"
            or json.loads(str(raw_order)) != list(PARENT_ROLES)
        ):
            raise ValueError(f"Invalid top/antitop parent-role convention in {path}")
        required = (
            "EVENT_ID",
            "PARTONIC/parents",
            "PARTONIC/ttbar",
            "PARTONIC/observables/mttbar",
            "CONDITIONING/assignment_hypothesis_index",
            "CONDITIONING/invisible_sample_index",
            "CONDITIONING/parent_sample_index",
        )
        missing = [name for name in required if name not in source]
        if missing:
            raise KeyError(f"Partonic prediction {path} is missing datasets {missing}")
        event_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        parents = source["PARTONIC/parents"]
        pair = source["PARTONIC/ttbar"]
        if event_ids.shape != (expected_events,) or len(np.unique(event_ids)) != len(event_ids):
            raise ValueError(f"Partonic prediction {path} has invalid or duplicate EVENT_ID values")
        if parents.shape != (expected_events, samples, 2, 4):
            raise ValueError(
                f"Partonic parent samples have shape {parents.shape}; "
                f"expected {(expected_events, samples, 2, 4)}"
            )
        if pair.shape != (expected_events, samples, 4):
            raise ValueError(f"Partonic ttbar samples have invalid shape {pair.shape}")
        parent_values = np.asarray(parents[:], dtype=np.float32)
        pair_values = np.asarray(pair[:], dtype=np.float32)
        if not np.isfinite(parent_values).all() or not np.isfinite(pair_values).all():
            raise ValueError(f"Partonic prediction {path} contains non-finite four-vectors")
        exact_pair = parent_values[:, :, 0] + parent_values[:, :, 1]
        if not np.array_equal(pair_values, exact_pair):
            raise ValueError(f"Partonic ttbar samples in {path} are not exact parent sums")
        for label, vectors in (("parent", parent_values), ("pair", pair_values)):
            energy = vectors[..., 0]
            momentum_squared = np.square(vectors[..., 1:]).sum(axis=-1)
            scale = np.maximum(np.square(energy), momentum_squared)
            tolerance = 32.0 * np.finfo(np.float32).eps * np.maximum(scale, 1.0)
            if (energy <= 0).any() or ((np.square(energy) - momentum_squared) < -tolerance).any():
                raise ValueError(f"Partonic {label} samples in {path} violate the future-timelike domain")
        for name in (
            "PARTONIC/observables/mttbar",
            "PARTONIC/observables/ptttbar",
            "PARTONIC/observables/yttbar",
            "PARTONIC/observables/phittbar",
            "PARTONIC/observables/cos_theta_star",
        ):
            values = np.asarray(source[name][:])
            if values.shape != (expected_events, samples) or not np.isfinite(values).all():
                raise ValueError(f"Partonic derived observable {name} is malformed or non-finite")
        for name in (
            "assignment_hypothesis_index",
            "invisible_sample_index",
            "parent_sample_index",
        ):
            values = source[f"CONDITIONING/{name}"]
            if values.shape != (expected_events, samples) or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"Partonic conditioning index {values.name} has invalid shape or dtype")


@torch.no_grad()
def predict_partonic_hdf5(
    module,
    data,
    output_path: str | Path,
    *,
    split: str,
    samples: int,
    steps: int,
    solver: str,
    seed: int,
    device: str,
    checkpoint_path: str | Path,
) -> dict:
    """Write sampled parents and derived observables without opening truth stores."""
    if module.event_model.parent_cfm is None:
        raise ValueError("Partonic prediction requires an enabled joint parent CFM")
    if samples <= 0 or steps <= 0 or solver not in {"euler", "heun"}:
        raise ValueError("Partonic samples/steps must be positive and solver must be euler or heun")
    conditioning = module.event_model.parent_cfm.conditioning
    source_path = Path(data.h5_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    loader = _loader(data, split)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    checkpoint_sha = _sha256(checkpoint)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    count = 0
    try:
        with h5py.File(source_path, "r") as event_source, h5py.File(temporary, "w") as output:
            output.attrs["prediction_schema"] = PARTONIC_PREDICTION_SCHEMA
            output.attrs["parent_roles"] = json.dumps(PARENT_ROLES)
            output.attrs["parent_role_convention"] = "top_antitop"
            output.attrs["parent_role_order"] = json.dumps(PARENT_ROLES)
            output.attrs["truth_definition"] = "partonic_after_fsr_target_contract"
            output.attrs["units"] = "GeV"
            output.attrs["four_vector_order"] = "E,px,py,pz"
            output.attrs["state_order"] = "top_px,top_py,top_pz,top_log1p_mass,antitop_px,antitop_py,antitop_pz,antitop_log1p_mass"
            output.attrs["conditioning"] = conditioning
            output.attrs["conditioning_contract"] = json.dumps(
                _conditioning_metadata(conditioning), sort_keys=True
            )
            output.attrs["solver"] = solver
            output.attrs["solver_steps"] = int(steps)
            output.attrs["nfe"] = 2 * int(steps) if solver == "heun" else int(steps)
            output.attrs["samples_per_event"] = int(samples)
            output.attrs["prediction_seed"] = int(seed)
            output.attrs["split"] = split
            output.attrs["checkpoint_path"] = str(checkpoint)
            output.attrs["checkpoint_sha256"] = checkpoint_sha
            output.attrs["topology_signature"] = module.hparams["topology_signature"]
            output.attrs["input_path"] = str(source_path)
            output.attrs["truth_free_input"] = True
            output.attrs["partonic_scaler"] = json.dumps(
                module.partonic_scaler_metadata, sort_keys=True
            )
            partonic_contract = module.hparams.get("partonic_truth_contract", {})
            if partonic_contract.get("source_file_sha256"):
                output.attrs["source_file_sha256"] = str(
                    partonic_contract["source_file_sha256"]
                )
            if conditioning == "P2":
                invisible_contract = module.hparams.get("invisible_truth_contract")
                if not isinstance(invisible_contract, dict):
                    raise ValueError("P2 parent prediction requires real invisible-role provenance")
                if (
                    invisible_contract.get("invisible_role_convention") != "top_antitop"
                    or invisible_contract.get("invisible_role_order")
                    != ["top.nu", "antitop.nu"]
                ):
                    raise ValueError("P2 parent prediction requires [top.nu, antitop.nu] context")
                output.attrs["invisible_role_convention"] = "top_antitop"
                output.attrs["invisible_role_order"] = json.dumps(
                    invisible_contract["invisible_role_order"]
                )
                output.attrs["invisible_source_file_sha256"] = str(
                    invisible_contract.get("source_file_sha256", "")
                )
                output.attrs["invisible_source_manifest_sha256"] = str(
                    invisible_contract.get("source_manifest_sha256", "")
                )
                output.attrs["invisible_split_membership_sha256"] = str(
                    invisible_contract.get("split_membership_sha256", "")
                )
            output.attrs["split_identity"] = json.dumps(data.split_metadata, sort_keys=True, default=str)
            for name in ("source_manifest_sha256", "split_membership_sha256", "split_version"):
                if name in event_source.attrs:
                    output.attrs[name] = event_source.attrs[name]

            events = _extendible(output, "EVENT_ID", (), np.int64)
            partonic = output.create_group("PARTONIC")
            parent_dataset = _extendible(partonic, "parents", (samples, 2, 4), np.float32)
            pair_dataset = _extendible(partonic, "ttbar", (samples, 4), np.float32)
            observables_group = partonic.create_group("observables")
            observable_names = (
                "mttbar",
                "ptttbar",
                "yttbar",
                "phittbar",
                "delta_phi_top_antitop",
                "delta_rapidity_top_antitop",
                "cos_theta_star",
            )
            observable_datasets = {
                name: _extendible(observables_group, name, (samples,), np.float32)
                for name in observable_names
            }
            condition_group = output.create_group("CONDITIONING")
            index_datasets = {
                name: _extendible(condition_group, name, (samples,), np.int64)
                for name in (
                    "assignment_hypothesis_index",
                    "invisible_sample_index",
                    "parent_sample_index",
                )
            }

            module.to(device).eval()
            for batch in loader:
                moved = _to_device(batch, device)
                result = module.event_model.sample_partonic(
                    moved,
                    samples=int(samples),
                    steps=int(steps),
                    solver=solver,
                    generator=generator,
                )
                ids = result.get("event_id")
                if ids is None:
                    ids = moved.get("event_index")
                if ids is None:
                    raise KeyError("Partonic prediction batch has no stable event identity")
                parents = result["parents"]
                pair = result["pair"]["ttbar"]
                if parents.shape != (len(ids), samples, 2, 4):
                    raise ValueError(f"Model returned malformed parent sample shape {tuple(parents.shape)}")
                if pair.shape != (len(ids), samples, 4):
                    raise ValueError(f"Model returned malformed pair sample shape {tuple(pair.shape)}")
                _append(events, ids.detach().cpu().numpy().astype(np.int64, copy=False))
                _append(parent_dataset, parents.detach().cpu().numpy().astype(np.float32, copy=False))
                _append(pair_dataset, pair.detach().cpu().numpy().astype(np.float32, copy=False))
                for name in observable_names:
                    _append(
                        observable_datasets[name],
                        result["pair"][name].detach().cpu().numpy().astype(np.float32, copy=False),
                    )
                for name, dataset in index_datasets.items():
                    _append(dataset, result[name].detach().cpu().numpy().astype(np.int64, copy=False))
                count += len(ids)
            if count == 0:
                raise ValueError(f"Partonic prediction split {split!r} produced no events")
            output.attrs["event_count"] = int(count)
            output.flush()
        _validate_prediction_file(temporary, count, samples)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "output": str(target),
        "events": count,
        "samples_per_event": samples,
        "conditioning": conditioning,
        "solver": solver,
        "steps": steps,
        "nfe": 2 * steps if solver == "heun" else steps,
        "seed": seed,
        "checkpoint_sha256": checkpoint_sha,
        "truth_free_input": True,
    }


def evaluate_partonic_hdf5(
    predictions_path: str | Path,
    truth_path: str | Path,
    output_path: str | Path | None = None,
) -> dict:
    """Join parent samples to the explicit after-FSR sidecar for diagnostics."""
    predictions = Path(predictions_path).expanduser().resolve()
    truth = Path(truth_path).expanduser().resolve()
    target = None if output_path is None else Path(output_path).expanduser().resolve()
    if not predictions.is_file() or not truth.is_file():
        raise FileNotFoundError(f"Prediction or truth sidecar is missing: {predictions}, {truth}")
    with h5py.File(predictions, "r") as source:
        if source.attrs.get("prediction_schema") != PARTONIC_PREDICTION_SCHEMA:
            raise ValueError(f"Unsupported partonic prediction file: {predictions}")
        if bool(source.attrs.get("truth_free_input", False)) is not True:
            raise ValueError("Partonic predictions must declare truth_free_input=true")
        prediction_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        samples = np.asarray(source["PARTONIC/parents"][:], dtype=np.float64)
        if samples.ndim != 4 or samples.shape[0] != len(prediction_ids) or samples.shape[2:] != (2, 4):
            raise ValueError("Partonic posterior samples must have shape [N,S,2,4]")
        prediction_identity = {
            key: source.attrs.get(key)
            for key in (
                "source_file_sha256",
                "source_manifest_sha256",
                "split_membership_sha256",
                "split_version",
            )
        }
        raw_roles = source.attrs.get("parent_roles")
        if isinstance(raw_roles, bytes):
            raw_roles = raw_roles.decode("utf-8")
        try:
            parent_roles = json.loads(str(raw_roles))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{predictions}: invalid parent_roles metadata") from error
        if parent_roles != list(PARENT_ROLES):
            raise ValueError(f"{predictions}: parent roles must be ordered {PARENT_ROLES}")
        role_convention = source.attrs.get("parent_role_convention")
        role_order = source.attrs.get("parent_role_order")
        if role_convention is not None or role_order is not None:
            if isinstance(role_convention, bytes):
                role_convention = role_convention.decode("utf-8")
            if isinstance(role_order, bytes):
                role_order = role_order.decode("utf-8")
            try:
                parsed_order = json.loads(str(role_order))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{predictions}: invalid parent_role_order metadata") from error
            if role_convention != "top_antitop" or parsed_order != list(PARENT_ROLES):
                raise ValueError(f"{predictions}: incompatible top/antitop parent-role metadata")
        split = str(source.attrs.get("split", ""))
        sampling_metadata = {
            key: source.attrs.get(key)
            for key in (
                "solver",
                "solver_steps",
                "nfe",
                "samples_per_event",
                "prediction_seed",
                "conditioning",
            )
        }
        if sampling_metadata["conditioning"] == "P2":
            convention = source.attrs.get("invisible_role_convention")
            order = source.attrs.get("invisible_role_order")
            if isinstance(convention, bytes):
                convention = convention.decode("utf-8")
            if isinstance(order, bytes):
                order = order.decode("utf-8")
            try:
                parsed_order = json.loads(str(order))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{predictions}: invalid invisible_role_order metadata") from error
            if convention != "top_antitop" or parsed_order != ["top.nu", "antitop.nu"]:
                raise ValueError(f"{predictions}: P2 context must be ordered [top.nu, antitop.nu]")
            prediction_identity.update(
                {
                    "invisible_source_file_sha256": source.attrs.get(
                        "invisible_source_file_sha256"
                    ),
                    "invisible_source_manifest_sha256": source.attrs.get(
                        "invisible_source_manifest_sha256"
                    ),
                    "invisible_split_membership_sha256": source.attrs.get(
                        "invisible_split_membership_sha256"
                    ),
                }
            )
    store = PartonicTruthStore(truth, PARENT_ROLES)
    truth_values, support = store.lookup(prediction_ids)
    truth_values = truth_values.numpy().astype(np.float64, copy=False)
    support = support.numpy()
    if len(np.unique(prediction_ids)) != len(prediction_ids):
        raise ValueError("Partonic prediction EVENT_ID contains duplicates")
    with h5py.File(truth, "r") as source:
        for key in (
            "source_file_sha256",
            "source_manifest_sha256",
            "split_membership_sha256",
            "split_version",
        ):
            expected = prediction_identity.get(key)
            observed = source.attrs.get(key)
            if expected is not None and observed is not None and str(expected) != str(observed):
                raise ValueError(f"Prediction/truth {key} mismatch: {expected!r} vs {observed!r}")
        if sampling_metadata["conditioning"] == "P2":
            for prediction_key, truth_key in (
                ("invisible_source_file_sha256", "source_file_sha256"),
                ("invisible_source_manifest_sha256", "source_manifest_sha256"),
                ("invisible_split_membership_sha256", "split_membership_sha256"),
            ):
                expected = prediction_identity.get(prediction_key)
                observed = source.attrs.get(truth_key)
                if expected is None or observed is None or str(expected) != str(observed):
                    raise ValueError(
                        f"P2 prediction/truth {truth_key} mismatch: {expected!r} vs {observed!r}"
                    )
    if samples.shape[0] != len(truth_values):
        raise ValueError("Prediction/truth event join changed event count")

    metrics: dict[str, object] = {
        "prediction_path": str(predictions),
        "truth_path": str(truth),
        "split": split,
        "event_count": len(prediction_ids),
        "supported_event_count": int(support.all(axis=1).sum()),
        "supported_parent_counts": {
            role: int(support[:, index].sum())
            for index, role in enumerate(PARENT_ROLES)
        },
        "samples_per_event": int(samples.shape[1]),
        "sampling": {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in sampling_metadata.items()
        },
        "metric_contract": (
            "descriptive posterior-sample mean residuals and empirical central-interval coverage; "
            "not a calibration claim"
        ),
    }
    mean_samples = samples.mean(axis=1)
    residual_metrics = {}
    for index, role in enumerate(PARENT_ROLES):
        valid = support[:, index]
        residual = mean_samples[valid, index] - truth_values[valid, index]
        residual_metrics[role] = {
            "events": int(valid.sum()),
            "component_bias_GeV_E_px_py_pz": residual.mean(axis=0).tolist() if valid.any() else None,
            "component_rmse_GeV_E_px_py_pz": np.sqrt(np.square(residual).mean(axis=0)).tolist() if valid.any() else None,
        }
    metrics["parent_p4_residuals"] = residual_metrics

    component_names = ("E", "px", "py", "pz")
    component_coverage = {}
    for index, role in enumerate(PARENT_ROLES):
        valid = support[:, index]
        role_samples = samples[valid, :, index, :]
        role_truth = truth_values[valid, index]
        by_component = {}
        for component_index, component_name in enumerate(component_names):
            values = role_samples[:, :, component_index]
            target_values = role_truth[:, component_index]
            by_component[component_name] = {
                "central_68": float(
                    np.mean(
                        (target_values >= np.quantile(values, 0.16, axis=1))
                        & (target_values <= np.quantile(values, 0.84, axis=1))
                    )
                ) if len(values) else None,
                "central_95": float(
                    np.mean(
                        (target_values >= np.quantile(values, 0.025, axis=1))
                        & (target_values <= np.quantile(values, 0.975, axis=1))
                    )
                ) if len(values) else None,
                "events": len(values),
            }
        component_coverage[role] = by_component
    metrics["parent_p4_empirical_interval_coverage"] = component_coverage

    if support.all(axis=1).any():
        import torch

        from joint_event_reconstruction.models.partonic import parent_pair_observables

        selected = support.all(axis=1)
        selected_samples = samples[selected]
        selected_truth = truth_values[selected]
        sample_observables = parent_pair_observables(
            torch.as_tensor(selected_samples, dtype=torch.float64)
        )
        truth_observables = parent_pair_observables(
            torch.as_tensor(selected_truth, dtype=torch.float64)
        )
        observable_metrics = {}
        names = (
            "mttbar",
            "ptttbar",
            "yttbar",
            "phittbar",
            "delta_phi_top_antitop",
            "delta_rapidity_top_antitop",
            "cos_theta_star",
        )
        for name in names:
            values = sample_observables[name].detach().cpu().numpy()
            targets = truth_observables[name].detach().cpu().numpy()
            if name in {"phittbar", "delta_phi_top_antitop"}:
                centers = np.arctan2(np.sin(values).mean(axis=1), np.cos(values).mean(axis=1))
                unwrapped = np.arctan2(
                    np.sin(values - centers[:, None]), np.cos(values - centers[:, None])
                )
                central_values = unwrapped
                target_interval_values = np.arctan2(
                    np.sin(targets - centers), np.cos(targets - centers)
                )
                residual = -target_interval_values
            else:
                central_values = values
                target_interval_values = targets
                residual = values.mean(axis=1) - targets
            observable_metrics[name] = {
                "events": len(values),
                "bias": float(residual.mean()) if len(residual) else None,
                "rmse": float(np.sqrt(np.square(residual).mean())) if len(residual) else None,
                "central_68_coverage": float(
                    np.mean(
                        (target_interval_values >= np.quantile(central_values, 0.16, axis=1))
                        & (target_interval_values <= np.quantile(central_values, 0.84, axis=1))
                    )
                ) if len(values) else None,
                "central_95_coverage": float(
                    np.mean(
                        (target_interval_values >= np.quantile(central_values, 0.025, axis=1))
                        & (target_interval_values <= np.quantile(central_values, 0.975, axis=1))
                    )
                ) if len(values) else None,
            }
        metrics["derived_pair_observables"] = observable_metrics

        if samples.shape[1] > 1:
            energy = samples[..., 0]
            momentum = samples[..., 1:]
            mass = np.sqrt(np.maximum(np.square(energy) - np.square(momentum).sum(axis=-1), 0.0))
            state = np.concatenate(
                (samples[..., 0, 1:], np.log1p(mass[..., 0:1]),
                 samples[..., 1, 1:], np.log1p(mass[..., 1:2])),
                axis=-1,
            )[selected]
            centered = state - state.mean(axis=1, keepdims=True)
            covariance = np.einsum("nsi,nsj->ij", centered, centered) / (
                len(centered) * (state.shape[1] - 1)
            )
            scales = np.sqrt(np.maximum(np.diag(covariance), 0.0))
            denominator = scales[:, None] * scales[None, :]
            correlation = np.divide(
                covariance,
                denominator,
                out=np.full_like(covariance, np.nan),
                where=denominator > 0,
            )
            metrics["within_event_joint_parent_sample_correlation"] = {
                "state_order": [
                    "top_px", "top_py", "top_pz", "top_log1p_mass",
                    "antitop_px", "antitop_py", "antitop_pz", "antitop_log1p_mass",
                ],
                "events": len(state),
                "correlation": np.nan_to_num(correlation, nan=0.0).tolist(),
                "undefined_zero_variance_pairs": int(np.isnan(correlation).sum()),
            }

    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        try:
            temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, target)
        finally:
            if temporary.exists():
                temporary.unlink()
        metrics["output"] = str(target)
    return metrics


__all__ = [
    "PARTONIC_PREDICTION_SCHEMA",
    "evaluate_partonic_hdf5",
    "predict_partonic_hdf5",
]
