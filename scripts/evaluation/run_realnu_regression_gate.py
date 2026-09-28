#!/usr/bin/env python3
"""Reproduce the accepted one-draw REALNU outputs on a fixed validation prefix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

TRAINING_SHA = "cb2b60445499f78421503d4fccc4a43cae1444fd"
EVALUATION_SHA = "e867d2cea14c1ec84fcfa86c66f6fa2b09fb27a9"
REPORT_FINALIZER_SHA = "cbed6a380f89d197afd5a65c37aee6648235e4eb"
EVENT_COUNT = 100_342
PARENT_ROLES = ["top", "antitop"]
NEUTRINO_ROLES = ["top.nu", "antitop.nu"]
MODES = ("Q0", "Q1", "PCFM_Q", "J0", "J_A", "J_AN")
ATOL = 0.1
RTOL = 5e-5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _json_attr(handle: h5py.File, name: str):
    value = handle.attrs[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _close(name: str, observed: np.ndarray, expected: np.ndarray) -> dict:
    observed = np.asarray(observed)
    expected = np.asarray(expected)
    if observed.shape != expected.shape:
        raise RuntimeError(
            f"{name} shape differs: reproduced={observed.shape}, accepted={expected.shape}"
        )
    if not np.isfinite(observed).all() or not np.isfinite(expected).all():
        raise RuntimeError(f"{name} contains a non-finite value")
    difference = observed.astype(np.float64) - expected.astype(np.float64)
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0
    if not np.allclose(observed, expected, atol=ATOL, rtol=RTOL):
        raise RuntimeError(
            f"{name} exceeds regression tolerance: "
            f"max_abs={max_abs:.8g}, rms={rms:.8g}, atol={ATOL}, rtol={RTOL}"
        )
    return {"max_abs": max_abs, "rms": rms, "atol": ATOL, "rtol": RTOL}


def _configure_subset(data, event_count: int, device: str) -> None:
    from torch.utils.data import Subset

    if len(data.val_dataset) < event_count:
        raise RuntimeError(
            f"Validation dataset has {len(data.val_dataset)} events; requested {event_count}"
        )
    data.val_dataset = Subset(data.val_dataset, range(event_count))
    if int(data.config.val_batch_size) != 512:
        raise RuntimeError(f"Regression validation batch size changed: {data.config.val_batch_size}")
    if device == "cuda" and not bool(data.config.pin_memory):
        raise RuntimeError("GPU regression inference requires the accepted pinned-memory loader")


def _reference_prediction_path(mode: str, production_root: Path) -> Path:
    training_root = production_root / mode
    if mode in {"Q0", "Q1"}:
        return training_root / "validation/standard/validation.h5"
    if mode == "PCFM_Q":
        return training_root / "validation/parent/parent_validation.h5"
    return training_root / "validation_r2/parent/parent_validation.h5"


def _run_mode(
    mode: str,
    *,
    runner,
    run_manifest: dict,
    production_root: Path,
    r2_root: Path,
    output_root: Path,
    event_count: int,
    cpu_threads: int,
    device: str,
    reference_gpu_type: str,
) -> dict:
    import lightning as L
    import torch

    from joint_event_reconstruction.cli.run import (
        build_data,
        load_trained,
        load_visible_config,
    )

    config_path = runner.MODE_CONFIGS[mode]
    training_root = production_root / mode
    baseline_path = _reference_prediction_path(mode, production_root)
    config, _, resolved_output = runner.load_and_validate_config(
        mode, config_path, training_root
    )
    config_sha = runner.sha256(config_path)
    if config_sha != run_manifest["modes"][mode]["config_sha256"]:
        raise RuntimeError(f"{mode} config differs from the accepted production config")
    record_path = training_root / "production_training.json"
    record = json.loads(record_path.read_text())
    checkpoint = Path(record.get("checkpoint", "")).expanduser().resolve()
    if (
        record.get("status") != "passed"
        or record.get("source_commit") != TRAINING_SHA
        or not checkpoint.is_file()
        or _sha256(checkpoint) != record.get("checkpoint_sha256")
    ):
        raise RuntimeError(f"{mode} accepted production checkpoint identity failed: {record_path}")
    if resolved_output != training_root.resolve():
        raise RuntimeError(f"{mode} config resolves to unexpected output path: {resolved_output}")

    L.seed_everything(int(config.inference.seed), workers=True)
    torch.set_num_threads(cpu_threads if device == "cpu" else 1)
    torch.set_float32_matmul_precision("high")
    visible = load_visible_config(config)
    if device == "cpu":
        visible.general.mode = "predict"
        visible.dataset.val_batch_size = 512
        visible.dataset.val_num_workers = 2
        visible.dataset.pin_memory = False
        visible.dataset.persistent_workers = True
        visible.dataset.prefetch_factor = 2
    data = build_data(config, visible, include_truth=False, device=device)
    data.setup("predict")
    collator = data.structured_collator
    if any(
        store is not None
        for store in (collator.visible_store, collator.invisible_store, collator.partonic_store)
    ):
        raise RuntimeError(f"{mode} regression inference loaded truth")
    if len(data.val_dataset) != EVENT_COUNT:
        raise RuntimeError(
            f"{mode} validation population changed: observed={len(data.val_dataset)}, "
            f"expected={EVENT_COUNT}"
        )
    _configure_subset(data, event_count, device)
    module = load_trained(config, visible, data, str(checkpoint))

    output_path = output_root / f"{mode}_subset.h5"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite regression output: {output_path}")
    if mode in {"Q0", "Q1"}:
        from joint_event_reconstruction.prediction import predict_hdf5

        predict_hdf5(
            module,
            data,
            data.invisible_scaler,
            output_path,
            split="validation",
            top_k=int(config.inference.top_k_assignments),
            posterior_samples=int(config.inference.posterior_samples),
            device=device,
            prediction_seed=int(config.inference.seed),
            checkpoint_path=str(checkpoint),
            compatibility_weight=float(config.inference.compatibility_weight),
            point_estimator=str(config.inference.invisible.point_estimator),
        )
        with h5py.File(output_path, "r") as produced, h5py.File(baseline_path, "r") as baseline:
            if str(baseline.attrs.get("gpu_type", "")) != reference_gpu_type:
                raise RuntimeError(f"{mode} standard baseline execution device changed")
            produced_ids = np.asarray(produced["EVENT_ID"][:], dtype=np.int64)
            baseline_ids = np.asarray(baseline["EVENT_ID"][:event_count], dtype=np.int64)
            if not np.array_equal(produced_ids, baseline_ids):
                raise RuntimeError(f"{mode} reproduced EVENT_IDs differ from accepted order")
            if _json_attr(produced, "invisible_roles") != NEUTRINO_ROLES:
                raise RuntimeError(f"{mode} reproduced neutrino roles changed")
            if _json_attr(baseline, "invisible_roles") != NEUTRINO_ROLES:
                raise RuntimeError(f"{mode} accepted neutrino roles are unexpected")
            if Path(str(produced.attrs["checkpoint"])).resolve() != checkpoint:
                raise RuntimeError(f"{mode} reproduced prediction names a different checkpoint")
            if Path(str(baseline.attrs["checkpoint"])).resolve() != checkpoint:
                raise RuntimeError(f"{mode} accepted prediction names a different checkpoint")
            if int(produced.attrs["prediction_seed"]) != int(config.inference.seed):
                raise RuntimeError(f"{mode} reproduced sample seed changed")
            if int(baseline.attrs["prediction_seed"]) != int(config.inference.seed):
                raise RuntimeError(f"{mode} accepted sample seed differs from config")
            if int(produced.attrs["K"]) != int(baseline.attrs["K"]) or int(
                produced.attrs["M"]
            ) != int(baseline.attrs["M"]):
                raise RuntimeError(f"{mode} assignment/posterior sample count changed")
            if str(produced.attrs["point_estimator"]) != str(
                baseline.attrs["point_estimator"]
            ):
                raise RuntimeError(f"{mode} neutrino point estimator changed")
            if json.loads(str(produced.attrs["scaler"])) != json.loads(str(baseline.attrs["scaler"])):
                raise RuntimeError(f"{mode} invisible scaler differs from accepted prediction")
            root_order = _json_attr(produced, "root_order")
            baseline_root_order = _json_attr(baseline, "root_order")
            if root_order != baseline_root_order:
                raise RuntimeError(f"{mode} assignment root order differs from accepted prediction")
            predicted_roots = np.asarray(
                produced["VISIBLE/root_candidate_indices"][:, 0], dtype=np.int64
            )
            accepted_roots = np.asarray(
                baseline["VISIBLE/root_candidate_indices"][:event_count, 0], dtype=np.int64
            )
            if not np.array_equal(predicted_roots, accepted_roots):
                raise RuntimeError(f"{mode} assignment output differs from accepted prediction")
            assignment_by_role = np.stack(
                [
                    predicted_roots[:, root_order.index("top")],
                    predicted_roots[:, root_order.index("antitop")],
                ],
                axis=1,
            )
            with h5py.File(r2_root / "core/event_metrics.h5", "r") as metrics:
                if not np.array_equal(
                    assignment_by_role,
                    metrics[f"models/{mode}/assignment_root_indices"][:event_count],
                ):
                    raise RuntimeError(f"{mode} role-ordered assignment differs from R2 metrics")
                if not np.array_equal(
                    produced["INVISIBLE/samples"][:, 0, 0],
                    metrics[f"models/{mode}/neutrino_joint_medoid"][:event_count],
                ):
                    raise RuntimeError(f"{mode} neutrino draw differs from R2 metrics")
            numerical = _close(
                f"{mode}.neutrino_samples",
                produced["INVISIBLE/samples"][:],
                baseline["INVISIBLE/samples"][:event_count],
            )
            return {
                "mode": mode,
                "output": str(output_path),
                "output_sha256": _sha256(output_path),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": record["checkpoint_sha256"],
                "config_sha256": config_sha,
                "execution_device": device,
                "baseline_gpu_type": reference_gpu_type,
                "roles": NEUTRINO_ROLES,
                "solver": str(config.inference.invisible.flow.solver),
                "steps": int(config.inference.invisible.flow.steps),
                "samples": int(config.inference.posterior_samples),
                "assignment_exact": True,
                "numerical": {"neutrino_samples": numerical},
            }

    from joint_event_reconstruction.prediction_partonic import predict_partonic_hdf5

    solver = str(config.model.parent_cfm.sampling_solver)
    steps = int(config.model.parent_cfm.sampling_steps)
    samples = int(config.inference.posterior_samples)
    predict_partonic_hdf5(
        module,
        data,
        output_path,
        split="validation",
        samples=samples,
        steps=steps,
        solver=solver,
        seed=int(config.inference.seed),
        device=device,
        checkpoint_path=checkpoint,
    )
    with h5py.File(output_path, "r") as produced, h5py.File(baseline_path, "r") as baseline:
        if str(baseline.attrs.get("gpu_type", "")) != reference_gpu_type:
            raise RuntimeError(f"{mode} parent baseline execution device changed")
        produced_ids = np.asarray(produced["EVENT_ID"][:], dtype=np.int64)
        baseline_ids = np.asarray(baseline["EVENT_ID"][:event_count], dtype=np.int64)
        if not np.array_equal(produced_ids, baseline_ids):
            raise RuntimeError(f"{mode} reproduced EVENT_IDs differ from accepted order")
        if _json_attr(produced, "parent_roles") != PARENT_ROLES:
            raise RuntimeError(f"{mode} reproduced parent role order changed")
        if _json_attr(baseline, "parent_roles") != PARENT_ROLES:
            raise RuntimeError(f"{mode} accepted parent role order is unexpected")
        if str(produced.attrs["truth_definition"]) != "partonic_after_fsr_target_contract":
            raise RuntimeError(f"{mode} reproduced parent truth definition changed")
        if str(baseline.attrs["truth_definition"]) != "partonic_after_fsr_target_contract":
            raise RuntimeError(f"{mode} accepted parent truth definition is unexpected")
        if Path(str(produced.attrs["checkpoint_path"])).resolve() != checkpoint:
            raise RuntimeError(f"{mode} reproduced prediction names a different checkpoint")
        if Path(str(baseline.attrs["checkpoint_path"])).resolve() != checkpoint:
            raise RuntimeError(f"{mode} accepted prediction names a different checkpoint")
        if str(produced.attrs["conditioning"]) != str(baseline.attrs["conditioning"]):
            raise RuntimeError(f"{mode} parent conditioning changed")
        if (
            str(produced.attrs["solver"]) != solver
            or int(produced.attrs["solver_steps"]) != steps
            or int(produced.attrs["samples_per_event"]) != samples
            or int(produced.attrs["prediction_seed"]) != int(config.inference.seed)
        ):
            raise RuntimeError(f"{mode} reproduced solver/sample policy differs from config")
        if (
            str(baseline.attrs["solver"]) != solver
            or int(baseline.attrs["solver_steps"]) != steps
            or int(baseline.attrs["samples_per_event"]) != samples
            or int(baseline.attrs["prediction_seed"]) != int(config.inference.seed)
        ):
            raise RuntimeError(f"{mode} accepted solver/sample policy differs from config")
        if json.loads(str(produced.attrs["partonic_scaler"])) != json.loads(
            str(baseline.attrs["partonic_scaler"])
        ):
            raise RuntimeError(f"{mode} parent scaler differs from accepted prediction")
        if int(produced.attrs["nfe"]) != 2 * steps or int(baseline.attrs["nfe"]) != 2 * steps:
            raise RuntimeError(f"{mode} Heun NFE accounting differs from the accepted policy")
        for path in (
            "CONDITIONING/assignment_hypothesis_index",
            "CONDITIONING/invisible_sample_index",
            "CONDITIONING/parent_sample_index",
        ):
            if not np.array_equal(
                produced[path][:], baseline[path][:event_count]
            ):
                raise RuntimeError(f"{mode} conditioning indices changed: {path}")
        parent = np.asarray(produced["PARTONIC/parents"][:], dtype=np.float64)
        pair = np.asarray(produced["PARTONIC/ttbar"][:], dtype=np.float64)
        if not np.array_equal(pair, parent[:, :, 0] + parent[:, :, 1]):
            raise RuntimeError(f"{mode} reproduced ttbar is not the exact parent sum")
        numerical = {
            "parent_four_vectors": _close(
                f"{mode}.parent_four_vectors",
                parent,
                baseline["PARTONIC/parents"][:event_count],
            ),
            "ttbar_four_vector": _close(
                f"{mode}.ttbar_four_vector",
                pair,
                baseline["PARTONIC/ttbar"][:event_count],
            ),
            "mttbar": _close(
                f"{mode}.mttbar",
                produced["PARTONIC/observables/mttbar"][:],
                baseline["PARTONIC/observables/mttbar"][:event_count],
            ),
        }
        with h5py.File(r2_root / "core/event_metrics.h5", "r") as metrics:
            numerical["r2_parent_four_vectors"] = _close(
                f"{mode}.r2_parent_four_vectors",
                parent[:, 0],
                metrics[f"models/{mode}/parent_joint_medoid"][:event_count],
            )
            numerical["r2_mttbar"] = _close(
                f"{mode}.r2_mttbar",
                produced["PARTONIC/observables/mttbar"][:, 0],
                metrics[f"models/{mode}/parent_mttbar_joint_medoid"][:event_count],
            )
        return {
            "mode": mode,
            "output": str(output_path),
            "output_sha256": _sha256(output_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": record["checkpoint_sha256"],
            "config_sha256": config_sha,
            "execution_device": device,
            "baseline_gpu_type": reference_gpu_type,
            "roles": PARENT_ROLES,
            "conditioning": str(produced.attrs["conditioning"]),
            "solver": solver,
            "steps": steps,
            "samples": samples,
            "nfe_per_sample": int(produced.attrs["nfe"]),
            "numerical": numerical,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--r2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--events", type=int, default=2048)
    parser.add_argument("--cpu-threads", type=int, default=6)
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    project_root = args.project_root.expanduser().resolve()
    production_root = args.production_root.expanduser().resolve()
    r2_root = args.r2_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if args.events != 2048:
        raise ValueError("REALNU_V1_REGRESSION_GATE uses the frozen 2048-event subset")
    for label, path in (
        ("source root", source_root),
        ("project root", project_root),
        ("production root", production_root),
        ("R2 root", r2_root),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} is missing: {path}")
    diagnostics_root = production_root / "evaluation_v3_diagnostics"
    if not output_root.is_relative_to(diagnostics_root):
        raise ValueError(f"Regression outputs must remain under {diagnostics_root}")
    if output_root == r2_root or r2_root in output_root.parents:
        raise ValueError(f"Regression outputs may not be written under evaluation_r2: {r2_root}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite regression gate output: {output_root}")

    import torch

    baseline_gpu_types = {}
    for mode in MODES:
        baseline_path = _reference_prediction_path(mode, production_root)
        if not baseline_path.is_file():
            raise FileNotFoundError(f"{mode} accepted prediction is missing: {baseline_path}")
        with h5py.File(baseline_path, "r") as baseline:
            gpu_type = str(baseline.attrs.get("gpu_type", ""))
        if gpu_type not in {"NVIDIA L40S", "CPU multicore"}:
            raise RuntimeError(
                f"{mode} accepted prediction has unsupported execution device: {gpu_type!r}"
            )
        baseline_gpu_types[mode] = gpu_type

    device_by_mode = {
        mode: "cuda" if gpu_type == "NVIDIA L40S" else "cpu"
        for mode, gpu_type in baseline_gpu_types.items()
    }
    if any(device == "cuda" for device in device_by_mode.values()):
        if not torch.cuda.is_available():
            raise RuntimeError("REALNU-v1 regression reproduction requires the accepted L40S device")
        allocated_gpu = torch.cuda.get_device_name(0)
        if allocated_gpu != "NVIDIA L40S":
            raise RuntimeError(
                f"Regression device differs from accepted L40S baseline: {allocated_gpu}"
            )
    else:
        allocated_gpu = None
    torch.set_num_interop_threads(1)

    run_manifest_path = r2_root / "manifests/run_manifest_r2.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    if (
        run_manifest.get("g0_source_commit") != TRAINING_SHA
        or run_manifest.get("evaluation_recovery", {}).get("source_sha") != EVALUATION_SHA
        or int(run_manifest.get("inputs", {}).get("validation_count", -1)) != EVENT_COUNT
        or int(run_manifest.get("inputs", {}).get("train_count", -1)) != 1_799_753
        or int(run_manifest.get("inputs", {}).get("reserved_test_rows_materialised", -1)) != 0
    ):
        raise RuntimeError("Frozen production source/split identities do not match the REALNU-v1 contract")

    os.environ["JER_PROJECT_ROOT"] = str(project_root)
    os.environ["JER_SOURCE_ROOT"] = str(source_root)
    os.environ["JER_REALNU_PROJECT_ROOT"] = str(project_root)
    os.environ["JER_REALNU_SOURCE_ROOT"] = str(source_root)
    os.environ["JER_SOURCE_SHA"] = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    sys.path.insert(0, str(source_root / "src"))
    sys.path.insert(0, str(source_root / "scripts/production"))
    runner = importlib.import_module("run_realnu_production")

    event_metrics_path = r2_root / "core/event_metrics.h5"
    standard_root = production_root
    with h5py.File(event_metrics_path, "r") as metrics:
        if bool(metrics.attrs.get("sealed_test_used", True)):
            raise RuntimeError("Accepted regression reference violates sealed_test_used=false")
        if _json_attr(metrics, "parent_role_order") != PARENT_ROLES:
            raise RuntimeError("Accepted parent truth role order differs from [top, antitop]")
        if _json_attr(metrics, "neutrino_role_order") != NEUTRINO_ROLES:
            raise RuntimeError("Accepted neutrino truth role order differs from [top.nu, antitop.nu]")
        ids = np.asarray(metrics["EVENT_ID"][:], dtype=np.int64)
        truth_mttbar = np.asarray(metrics["truth/mttbar"][:], dtype=np.float64)
        channels = np.asarray(metrics["channel"][:])
    if (
        len(ids) != EVENT_COUNT
        or len(np.unique(ids)) != EVENT_COUNT
        or not np.isfinite(truth_mttbar).all()
    ):
        raise RuntimeError("Accepted validation reference has malformed events or truth")
    selected_ids = ids[: args.events]
    selected_truth = truth_mttbar[: args.events]
    selected_channels = channels[: args.events]
    if len(np.unique(selected_ids)) != args.events:
        raise RuntimeError("Fixed regression subset contains duplicate EVENT_ID values")

    output_root.mkdir(parents=True)
    id_path = output_root / "EVENT_IDS.csv"
    with id_path.open("w", encoding="utf-8") as stream:
        stream.write("index,EVENT_ID,truth_mttbar_GeV,channel\n")
        for index, (event_id, mass, channel) in enumerate(
            zip(selected_ids, selected_truth, selected_channels, strict=True)
        ):
            channel_text = channel.decode("utf-8") if isinstance(channel, bytes) else str(channel)
            stream.write(f"{index},{int(event_id)},{float(mass):.9g},{channel_text}\n")

    records = []
    for mode in MODES:
        record = _run_mode(
            mode,
            runner=runner,
            run_manifest=run_manifest,
            production_root=standard_root,
            r2_root=r2_root,
            output_root=output_root,
            event_count=args.events,
            cpu_threads=args.cpu_threads,
            device=device_by_mode[mode],
            reference_gpu_type=baseline_gpu_types[mode],
        )
        with h5py.File(record["output"], "r") as result:
            predicted_ids = np.asarray(result["EVENT_ID"][:], dtype=np.int64)
        if not np.array_equal(predicted_ids, selected_ids):
            raise RuntimeError(
                f"{mode} inference EVENT_IDs do not match frozen regression list"
            )
        records.append(record)
        del record
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "status": "passed",
        "gate": "REALNU_V1_REGRESSION_GATE",
        "training_source_sha": TRAINING_SHA,
        "evaluation_recovery_sha": EVALUATION_SHA,
        "report_finalizer_sha": REPORT_FINALIZER_SHA,
        "diagnostics_source_sha": os.environ["JER_SOURCE_SHA"],
        "new_diagnostics_sha": os.environ["JER_SOURCE_SHA"],
        "diagnostics_branch": "research/realnu-cfm-query-diagnostics-v2",
        "allocated_gpu": allocated_gpu,
        "execution_device_by_mode": device_by_mode,
        "baseline_gpu_type_by_mode": baseline_gpu_types,
        "event_count": args.events,
        "event_id_ordered_sha256": _array_hash(selected_ids),
        "event_id_sorted_sha256": _array_hash(np.sort(selected_ids)),
        "validation_event_id_sha256": run_manifest["inputs"]["validation_sorted_event_id_sha256"],
        "source_manifest_sha256": run_manifest["inputs"]["source_manifest_sha256"],
        "split_membership_sha256": run_manifest["inputs"]["split_membership_sha256"],
        "run_manifest_sha256": _sha256(run_manifest_path),
        "accepted_event_metrics_sha256": _sha256(event_metrics_path),
        "train_count": 1_799_753,
        "validation_count": EVENT_COUNT,
        "sealed_test_used": False,
        "parent_truth_roles": PARENT_ROLES,
        "neutrino_truth_roles": NEUTRINO_ROLES,
        "numerical_tolerance": {"absolute": ATOL, "relative": RTOL},
        "truth_mttbar_subset_gev": {
            "minimum": float(selected_truth.min()),
            "maximum": float(selected_truth.max()),
            "mean": float(selected_truth.mean()),
            "below_345": int(np.sum(selected_truth < 345.0)),
            "above_1000": int(np.sum(selected_truth > 1000.0)),
        },
        "modes": records,
    }
    result_path = output_root / "REGRESSION_GATE.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
