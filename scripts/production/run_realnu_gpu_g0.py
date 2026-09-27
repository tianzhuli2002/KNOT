"""Run the bounded real-top/antitop end-to-end GPU acceptance gate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.prediction_schema import validate_prediction
from joint_event_reconstruction.topology.compiler import load_topology

SOURCE_ROOT = Path(
    os.environ.get("JER_REALNU_SOURCE_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
ROOT = Path(
    os.environ.get("JER_REALNU_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
EXPERIMENT_ROOT = ROOT / "results/experiments/partonic_ttbar2L_2M_realnu_v1"
CONFIG_ROOT = SOURCE_ROOT / "configs/experiments/realnu_v1/gpu_smoke"
RAW_SOURCE_SHA256 = "75e895c0c9ce3984d4c925fa6f850748258e7b1d0e007a757f74cbeaa9cf9196"
SOURCE_MANIFEST_SHA256 = "5f4d44cad0b6f8857ebb2a4094d40e3901459c5648cb7464cb9af42279502601"
SPLIT_MEMBERSHIP_SHA256 = "260af7bda73383dc2d5d5adf95f628e71f081642964ea7b875746d7d08225374"
TRUTH_STORE_SHA256 = "5abc962cb79cb0856a0b3d51b6b093e6bfe4c0d37d465153e6a85d9fa2e72f65"
CACHE_FINGERPRINT = "272496afa7d4e4785daed18a26a2904e5b21a7157d3ec871a0835e55c94e424b"
ROLE_ORDER = ["top.nu", "antitop.nu"]
PARENT_ROLES = ["top", "antitop"]
MODES = {
    "Q0": "q0.public.yaml",
    "Q1": "q1.public.yaml",
    "PCFM_Q": "pcfm_q.public.yaml",
    "J0": "j0.public.yaml",
    "J_A": "j_a.public.yaml",
    "J_AN": "j_an.public.yaml",
}
PLAIN_PREDICTION_MODES = ("Q0", "Q1", "J0", "J_A", "J_AN")
PARENT_PREDICTION_MODES = ("PCFM_Q", "J0", "J_A", "J_AN")
BATCH_PROBE_MODES = ("Q0", "Q1", "J0", "J_AN")
EXPECTED_CONDITIONING = {"PCFM_Q": "P0", "J0": "P0", "J_A": "P1", "J_AN": "P2"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _assert_source(expected_sha: str) -> None:
    observed_sha = _git("rev-parse", "HEAD")
    observed_branch = _git("branch", "--show-current")
    dirty = _git("status", "--porcelain=v1", "--untracked-files=all")
    if observed_sha != expected_sha:
        raise RuntimeError(f"G0 source SHA mismatch: expected {expected_sha}, found {observed_sha}")
    if observed_branch != "research/partonic-query-cfm-realnu-v1":
        raise RuntimeError(f"G0 branch mismatch: found {observed_branch!r}")
    if dirty:
        raise RuntimeError(f"G0 requires a clean source tree; git status reports: {dirty}")
    source_metadata_path = SOURCE_ROOT / "SOURCE_SNAPSHOT.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"Immutable G0 source snapshot metadata is missing: {source_metadata_path}")
    source_metadata = json.loads(source_metadata_path.read_text())
    expected_tree_sha = _git("rev-parse", f"{expected_sha}^{{tree}}")
    if (
        source_metadata.get("schema") != "jer_source_snapshot_v1"
        or source_metadata.get("commit_sha") != expected_sha
        or source_metadata.get("tree_sha") != expected_tree_sha
        or (SOURCE_ROOT / ".git").exists()
    ):
        raise RuntimeError("Immutable G0 source snapshot does not match the expected source commit")


def _preflight() -> dict:
    manifest_path = EXPERIMENT_ROOT / "source_manifest_v1.json"
    split_path = EXPERIMENT_ROOT / "split_membership_v1.npz"
    measured_path = (
        EXPERIMENT_ROOT
        / "adapted/train_validation/partonic_ttbar2l_train_validation_maxjets8_v1_measured.h5"
    )
    truth_path = (
        EXPERIMENT_ROOT
        / "adapted/train_validation_authoritative_20260926/"
        "partonic_ttbar2l_train_validation_maxjets8_v1_supervision.h5"
    )
    cache_path = EXPERIMENT_ROOT / "cache/train_validation_measured_maxjets8_v1.pt"
    cache_manifest_path = cache_path.with_suffix(".json")
    for path in (manifest_path, split_path, measured_path, truth_path, cache_path, cache_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"G0 required realnu input is missing: {path}")
    manifest_sha = _sha256(manifest_path)
    split_sha = _sha256(split_path)
    if manifest_sha != SOURCE_MANIFEST_SHA256:
        raise RuntimeError(f"Source manifest SHA changed: {manifest_sha}")
    if split_sha != SPLIT_MEMBERSHIP_SHA256:
        raise RuntimeError(f"Frozen train/validation split SHA changed: {split_sha}")
    source_manifest = json.loads(manifest_path.read_text())
    if source_manifest.get("new_source", {}).get("sha256") != RAW_SOURCE_SHA256:
        raise RuntimeError("Source manifest no longer identifies the audited raw realnu file")
    cache_manifest = json.loads(cache_manifest_path.read_text())
    if cache_manifest.get("event_id_fingerprint") != CACHE_FINGERPRINT:
        raise RuntimeError("Measured tensor cache event identity differs from the audited cache")
    if int(cache_manifest.get("cache_version", -1)) != 9:
        raise RuntimeError("Measured tensor cache version differs from the audited cache")
    cache_identity = cache_manifest.get("source_identity", {})
    if (
        Path(str(cache_identity.get("path", ""))).resolve() != measured_path.resolve()
        or int(cache_identity.get("n_events", -1)) != 1_900_095
        or int(cache_identity.get("size_bytes", -1)) != measured_path.stat().st_size
    ):
        raise RuntimeError("Measured tensor cache manifest does not match the current measured HDF5")
    with h5py.File(measured_path, "r") as measured, h5py.File(truth_path, "r") as truth:
        if len(measured["EVENT_ID"]) != 1_900_095 or len(truth["EVENT_ID"]) != 1_900_095:
            raise RuntimeError("Realnu train/validation product has an unexpected event count")
        if any(name in measured for name in ("LABELS", "TARGETS")):
            raise RuntimeError("Configured truth-free measured input contains LABELS or TARGETS")
        if (
            str(truth.attrs.get("source_manifest_sha256", "")) != SOURCE_MANIFEST_SHA256
            or str(truth.attrs.get("split_membership_sha256", "")) != SPLIT_MEMBERSHIP_SHA256
            or str(truth.attrs.get("invisible_role_convention", "")) != "top_antitop"
            or json.loads(str(truth.attrs.get("invisible_role_order", "[]"))) != ROLE_ORDER
            or str(truth.attrs.get("units", "")) != "GeV"
            or bool(truth.attrs.get("synthetic", True))
        ):
            raise RuntimeError("Real neutrino sidecar metadata does not match the top/antitop contract")
        if truth["SUPPORT/INVISIBLE_ROLE"].shape != (1_900_095, 2):
            raise RuntimeError("Real neutrino role support has an unexpected shape")
        truth_sha = _sha256(truth_path)
        if truth_sha != TRUTH_STORE_SHA256:
            raise RuntimeError(f"Authoritative supervision sidecar SHA changed: {truth_sha}")

    configs: dict[str, dict] = {}
    for mode, filename in MODES.items():
        config_path = CONFIG_ROOT / filename
        config = load_experiment(config_path)
        spec = load_topology(config.topology_config, config.data.truth_bindings)
        if (
            Path(config.data.input).resolve() != measured_path.resolve()
            or Path(config.data.inference_input).resolve() != measured_path.resolve()
            or Path(config.data.tensor_cache).resolve() != cache_path.resolve()
            or int(config.data.max_events.train) != 512
            or int(config.data.max_events.val) != 128
            or config.data.max_events.test is not None
            or int(config.training.batch_size) != 512
            or int(config.training.workers) != 4
            or str(config.training.precision) != "16-mixed"
        ):
            raise RuntimeError(f"{mode} G0 config does not use the common bounded production setup")
        if config.data.visible_truth is not None and Path(config.data.visible_truth).resolve() != truth_path.resolve():
            raise RuntimeError(f"{mode} visible truth path is not the authoritative realnu sidecar")
        if mode in PLAIN_PREDICTION_MODES and Path(config.data.invisible.path).resolve() != truth_path.resolve():
            raise RuntimeError(f"{mode} invisible truth path is not the authoritative realnu sidecar")
        if mode in PARENT_PREDICTION_MODES:
            if not bool(config.model.parent_cfm.enabled):
                raise RuntimeError(f"{mode} G0 config has no parent CFM enabled")
            conditioning = str(config.model.parent_cfm.conditioning)
            if conditioning != EXPECTED_CONDITIONING[mode]:
                raise RuntimeError(f"{mode} conditioning changed: {conditioning}")
            if Path(config.data.partonic.path).resolve() != truth_path.resolve():
                raise RuntimeError(f"{mode} partonic truth path is not the authoritative realnu sidecar")
            if list(config.data.partonic.roles) != PARENT_ROLES:
                raise RuntimeError(f"{mode} parent target roles are not [top, antitop]")
        if mode in {"Q0", "Q1", "J0", "J_A", "J_AN"}:
            if [role.key for role in spec.invisible_roles] != ROLE_ORDER:
                raise RuntimeError(f"{mode} topology roles are not [top.nu, antitop.nu]")
            if not bool(config.model.invisible.enabled):
                raise RuntimeError(f"{mode} config must train the real top/antitop neutrino model")
        configs[mode] = {
            "path": str(config_path.resolve()),
            "sha256": _sha256(config_path),
            "name": str(config.experiment.name),
            "conditioning": (
                str(config.model.parent_cfm.conditioning)
                if bool(config.model.parent_cfm.enabled)
                else None
            ),
        }
    return {
        "status": "passed",
        "source_manifest_sha256": manifest_sha,
        "split_membership_sha256": split_sha,
        "raw_source_sha256": RAW_SOURCE_SHA256,
        "truth_store_sha256": truth_sha,
        "measured_input": str(measured_path.resolve()),
        "measured_input_sha256": _sha256(measured_path),
        "truth_store": str(truth_path.resolve()),
        "cache": str(cache_path.resolve()),
        "cache_fingerprint": CACHE_FINGERPRINT,
        "event_count": 1_900_095,
        "reserved_test_rows_materialised": 0,
        "invisible_roles": ROLE_ORDER,
        "parent_roles": PARENT_ROLES,
        "legacy_flavour_charge_mismatch_policy": "diagnostic only; no support masking",
        "configs": configs,
    }


def _run_command(
    label: str,
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    progress: dict,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    progress["stage"] = label
    progress["updated_utc"] = datetime.now(UTC).isoformat()
    _atomic_json(Path(progress["root"]) / "progress.json", progress)
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    record = {
        "command": command,
        "log": str(log_path.resolve()),
        "exit_code": completed.returncode,
        "elapsed_seconds": time.time() - started,
    }
    progress["commands"][label] = record
    progress["completed_stages"].append(label)
    progress["updated_utc"] = datetime.now(UTC).isoformat()
    _atomic_json(Path(progress["root"]) / "progress.json", progress)
    if completed.returncode != 0:
        raise RuntimeError(f"G0 stage {label!r} failed with exit code {completed.returncode}; log={log_path}")
    return record


def _checkpoint_summary(checkpoint_path: Path) -> dict:
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("global_step", 0)) < 1:
        raise RuntimeError(f"Checkpoint has no completed optimiser step: {checkpoint_path}")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Checkpoint has no model state_dict: {checkpoint_path}")
    tensor_count = 0
    for name, value in state.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Checkpoint contains non-finite model weights at {name}: {checkpoint_path}")
            tensor_count += 1
    callback_states = checkpoint.get("callbacks", {})
    val_scores = [
        float(value["best_model_score"])
        for key, value in callback_states.items()
        if "ModelCheckpoint" in key
        and isinstance(value, dict)
        and value.get("best_model_score") is not None
    ]
    if not val_scores or not all(np.isfinite(score) for score in val_scores):
        raise RuntimeError(f"Checkpoint is missing a finite validation score: {checkpoint_path}")
    hparams = checkpoint.get("hyper_parameters", {})
    invisible_contract = hparams.get("invisible_truth_contract")
    if invisible_contract is not None and invisible_contract.get("invisible_role_order") != ROLE_ORDER:
        raise RuntimeError(f"Checkpoint has an incompatible invisible role order: {checkpoint_path}")
    return {
        "path": str(checkpoint_path.resolve()),
        "global_step": int(checkpoint["global_step"]),
        "epoch": int(checkpoint.get("epoch", -1)),
        "floating_state_tensors": tensor_count,
        "validation_score": min(val_scores),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "invisible_role_order": (
            invisible_contract.get("invisible_role_order") if invisible_contract else None
        ),
    }


def _validate_plain_prediction(path: Path, expected_source: Path) -> dict:
    validate_prediction(path, deep=True)
    with h5py.File(path, "r") as prediction:
        roles = json.loads(str(prediction.attrs.get("invisible_roles", "[]")))
        if roles != ROLE_ORDER:
            raise RuntimeError(f"Prediction has the wrong neutrino role order: {roles}")
        if bool(prediction.attrs.get("invisible_truth_available", True)):
            raise RuntimeError(f"Prediction unexpectedly loaded invisible truth: {path}")
        if bool(prediction.attrs.get("visible_truth_available", True)):
            raise RuntimeError(f"Prediction unexpectedly loaded visible truth: {path}")
        source = Path(str(prediction.attrs.get("source_data", ""))).resolve()
        if source != expected_source.resolve():
            raise RuntimeError(f"Prediction source is not the truth-free measured HDF5: {source}")
        truth_group = prediction.get("TRUTH")
        if truth_group is None:
            raise RuntimeError(f"Prediction is missing its TRUTH schema group: {path}")
        if (
            bool(truth_group.attrs.get("visible_available", True))
            or bool(truth_group.attrs.get("invisible_available", True))
            or set(truth_group.keys()) != {"topk_assignment"}
        ):
            raise RuntimeError(f"Truth-free prediction contains actual truth data: {path}")
        event_ids = np.asarray(prediction["EVENT_ID"][:], dtype=np.int64)
        if len(event_ids) != 128 or len(np.unique(event_ids)) != 128:
            raise RuntimeError(f"Prediction does not contain 128 unique validation events: {path}")
        return {"events": len(event_ids), "invisible_roles": roles, "truth_free": True}


def _validate_parent_prediction(
    path: Path, expected_source: Path, expected_conditioning: str, mode: str
) -> dict:
    from joint_event_reconstruction.prediction_partonic import (
        PARTONIC_PREDICTION_SCHEMA,
        _validate_prediction_file,
    )

    _validate_prediction_file(path, expected_events=128, samples=2)
    with h5py.File(path, "r") as prediction:
        roles = json.loads(str(prediction.attrs.get("parent_roles", "[]")))
        if roles != PARENT_ROLES:
            raise RuntimeError(f"{mode} parent prediction has the wrong role order: {roles}")
        if prediction.attrs.get("prediction_schema") != PARTONIC_PREDICTION_SCHEMA:
            raise RuntimeError(f"{mode} parent prediction has the wrong schema")
        if str(prediction.attrs.get("conditioning", "")) != expected_conditioning:
            raise RuntimeError(f"{mode} parent prediction conditioning changed")
        if bool(prediction.attrs.get("truth_free_input", False)) is not True:
            raise RuntimeError(f"{mode} parent prediction did not use a truth-free input")
        if Path(str(prediction.attrs.get("input_path", ""))).resolve() != expected_source.resolve():
            raise RuntimeError(f"{mode} parent prediction input is not the measured-only HDF5")
        event_ids = np.asarray(prediction["EVENT_ID"][:], dtype=np.int64)
        parents = np.asarray(prediction["PARTONIC/parents"][:], dtype=np.float32)
        pairs = np.asarray(prediction["PARTONIC/ttbar"][:], dtype=np.float32)
        if event_ids.shape != (128,) or len(np.unique(event_ids)) != 128:
            raise RuntimeError(f"{mode} parent prediction has invalid validation event IDs")
        if not np.array_equal(pairs, parents[:, :, 0] + parents[:, :, 1]):
            raise RuntimeError(f"{mode} parent pair is not the exact top+antitop sum")
        result = {"events": len(event_ids), "samples_per_event": 2, "conditioning": expected_conditioning}
        if expected_conditioning == "P2":
            invisible_roles = json.loads(str(prediction.attrs.get("invisible_role_order", "[]")))
            invisible_indices = np.asarray(prediction["CONDITIONING/invisible_sample_index"][:])
            parent_indices = np.asarray(prediction["CONDITIONING/parent_sample_index"][:])
            if invisible_roles != ROLE_ORDER:
                raise RuntimeError("P2 parent prediction has the wrong neutrino context role order")
            if invisible_indices.shape != (128, 2) or not np.equal(invisible_indices, 0).all():
                raise RuntimeError("P2 parent samples are not associated with invisible sample index zero")
            expected_parent_indices = np.broadcast_to(np.arange(2, dtype=np.int64), (128, 2))
            if not np.array_equal(parent_indices, expected_parent_indices):
                raise RuntimeError("P2 parent sample indices do not map to parent samples 0 and 1")
            result["invisible_sample_index"] = "zero for both parent samples"
            result["parent_sample_index"] = [0, 1]
        return result


def run(root: Path, expected_sha: str, python_bin: Path, contract_path: Path) -> dict:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The realnu G0 must run inside its explicitly allocated Slurm GPU job")
    if root.exists():
        raise FileExistsError(f"Refusing to reuse realnu G0 output root: {root}")
    _assert_source(expected_sha)
    if not contract_path.is_file():
        raise FileNotFoundError(f"Resolved G0 execution contract is missing: {contract_path}")
    contract = json.loads(contract_path.read_text())
    if (
        contract.get("schema") != "realnu_g0_execution_contract_v1"
        or contract.get("source_sha") != expected_sha
        or Path(contract.get("source_root", "")).resolve() != SOURCE_ROOT
        or Path(contract.get("project_root", "")).resolve() != ROOT
        or Path(contract.get("g0_root", "")).resolve() != root
        or Path(contract.get("python", "")).resolve() != python_bin.resolve()
    ):
        raise RuntimeError("Resolved G0 execution contract does not match this Slurm job")
    root.mkdir(parents=True)
    progress = {
        "status": "running",
        "root": str(root.resolve()),
        "source_commit": expected_sha,
        "branch": "research/partonic-query-cfm-realnu-v1",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "started_utc": datetime.now(UTC).isoformat(),
        "stage": "preflight",
        "completed_stages": [],
        "commands": {},
    }
    _atomic_json(root / "progress.json", progress)
    try:
        preflight = _preflight()
        if preflight != contract.get("g0_preflight"):
            raise RuntimeError("G0 preflight changed after submission; refusing to train")
        _atomic_json(root / "preflight.json", preflight)
        progress["completed_stages"].append("preflight")
        progress["updated_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(root / "progress.json", progress)
        expected_source = Path(preflight["measured_input"])
        env = os.environ.copy()
        env["JER_PROJECT_ROOT"] = str(ROOT)
        env["JER_SOURCE_ROOT"] = str(SOURCE_ROOT)
        env["JER_SOURCE_SHA"] = expected_sha
        env["JER_SOURCE_BRANCH"] = "research/partonic-query-cfm-realnu-v1"
        env["PYTHONPATH"] = f"{SOURCE_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("OPENBLAS_NUM_THREADS", "1")

        checkpoints: dict[str, dict] = {}
        for mode, filename in MODES.items():
            config = CONFIG_ROOT / filename
            training_output = root / mode / "training"
            command = [
                str(python_bin), "-m", "joint_event_reconstruction.cli.run", "train",
                "--experiment", str(config), "--device", "cuda",
                "--set", f"training.output={training_output}",
            ]
            _run_command(
                f"train_{mode}", command, root / mode / "logs/train.log", env, progress
            )
            checkpoint_path = training_output / "checkpoints/best-loss.ckpt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"{mode} training did not produce best-loss checkpoint: {checkpoint_path}")
            checkpoints[mode] = _checkpoint_summary(checkpoint_path)

        predictions: dict[str, dict] = {}
        for mode in PLAIN_PREDICTION_MODES:
            filename = MODES[mode]
            config = CONFIG_ROOT / filename
            checkpoint = Path(checkpoints[mode]["path"])
            output = root / mode / "predictions/validation.h5"
            command = [
                str(python_bin), "-m", "joint_event_reconstruction.cli.run", "predict",
                "--experiment", str(config), "--checkpoint", str(checkpoint),
                "--output", str(output), "--split", "validation", "--device", "cuda",
                "--set", "inference.top_k_assignments=1",
                "--set", "inference.posterior_samples=1",
                "--set", "inference.invisible.flow.steps=2",
            ]
            _run_command(
                f"predict_{mode}", command, root / mode / "logs/predict.log", env, progress
            )
            predictions[mode] = _validate_plain_prediction(output, expected_source)
            evaluation_path = root / mode / "evaluation/validation.json"
            _run_command(
                f"evaluate_{mode}",
                [
                    str(python_bin), "-m", "joint_event_reconstruction.cli.run", "evaluate",
                    "--predictions", str(output),
                ],
                evaluation_path,
                env,
                progress,
            )
            predictions[mode]["evaluation"] = str(evaluation_path.resolve())

        parent_predictions: dict[str, dict] = {}
        truth_path = Path(preflight["truth_store"])
        for mode in PARENT_PREDICTION_MODES:
            config = CONFIG_ROOT / MODES[mode]
            checkpoint = Path(checkpoints[mode]["path"])
            output = root / mode / "parent_predictions/validation.h5"
            command = [
                str(python_bin), "-m", "joint_event_reconstruction.cli.run", "predict-partonic",
                "--experiment", str(config), "--checkpoint", str(checkpoint),
                "--output", str(output), "--split", "validation", "--device", "cuda",
                "--samples", "2", "--steps", "2", "--solver", "heun", "--seed", "42011",
            ]
            _run_command(
                f"predict_parent_{mode}", command,
                root / mode / "logs/predict_parent.log", env, progress,
            )
            parent_predictions[mode] = _validate_parent_prediction(
                output, expected_source, EXPECTED_CONDITIONING[mode], mode
            )
            evaluation_path = root / mode / "evaluation/parent_validation.json"
            _run_command(
                f"evaluate_parent_{mode}",
                [
                    str(python_bin), "-m", "joint_event_reconstruction.cli.run", "evaluate-partonic",
                    "--predictions", str(output), "--truth", str(truth_path),
                    "--output", str(evaluation_path),
                ],
                root / mode / "logs/evaluate_parent.log",
                env,
                progress,
            )
            parent_predictions[mode]["evaluation"] = str(evaluation_path.resolve())

        batch_probes: dict[str, dict] = {}
        for mode in BATCH_PROBE_MODES:
            output = root / mode / "batch_probe.json"
            _run_command(
                f"production_batch_{mode}",
                [
                    str(python_bin),
                    str(SOURCE_ROOT / "scripts/validation/validate_realnu_production_batch_gpu.py"),
                    "--config", str(CONFIG_ROOT / MODES[mode]),
                    "--output", str(output),
                ],
                root / mode / "logs/production_batch.log",
                env,
                progress,
            )
            batch_probes[mode] = json.loads(output.read_text())
        q0_exposure = batch_probes["Q0"]["train_event_ids_sha256_sorted"]
        q1_exposure = batch_probes["Q1"]["train_event_ids_sha256_sorted"]
        if q0_exposure != q1_exposure:
            raise RuntimeError("Q0 and Q1 production batch probes used different train event populations")

        import torch

        acceptance = {
            "status": "passed",
            "campaign": "realnu_top_antitop_v1",
            "source_commit": expected_sha,
            "branch": "research/partonic-query-cfm-realnu-v1",
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "gpu": torch.cuda.get_device_name(0),
            "preflight": preflight,
            "checkpoints": checkpoints,
            "validation_predictions": predictions,
            "parent_predictions": parent_predictions,
            "production_batch_probes": batch_probes,
            "q0_q1_train_event_ids_sha256_sorted": q0_exposure,
            "test_rows_read": 0,
            "roles": {"invisible": ROLE_ORDER, "parent": PARENT_ROLES},
            "completed_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(root / "acceptance.json", acceptance)
        progress["status"] = "passed"
        progress["stage"] = "complete"
        progress["completed_utc"] = acceptance["completed_utc"]
        _atomic_json(root / "progress.json", progress)
        return acceptance
    except Exception as error:
        progress["status"] = "failed"
        progress["error"] = f"{type(error).__name__}: {error}"
        progress["updated_utc"] = datetime.now(UTC).isoformat()
        _atomic_json(root / "failure.json", progress)
        _atomic_json(root / "progress.json", progress)
        raise


def main() -> None:
    expected_sha = os.environ.get("JER_REALNU_EXPECTED_SHA")
    output_root = os.environ.get("JER_REALNU_G0_ROOT")
    contract_path = os.environ.get("JER_REALNU_PREFLIGHT_CONTRACT")
    python_bin = Path(os.environ.get("JER_PYTHON", sys.executable)).expanduser().resolve()
    if not expected_sha or not output_root or not contract_path:
        raise SystemExit(
            "JER_REALNU_EXPECTED_SHA, JER_REALNU_G0_ROOT and "
            "JER_REALNU_PREFLIGHT_CONTRACT are required"
        )
    result = run(
        Path(output_root).expanduser().resolve(), expected_sha, python_bin,
        Path(contract_path).expanduser().resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
