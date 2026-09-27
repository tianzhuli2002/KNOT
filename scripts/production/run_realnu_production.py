"""Run one full real-neutrino production or validation stage."""

from __future__ import annotations

import argparse
import gc
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

SOURCE_ROOT = Path(
    os.environ.get("JER_REALNU_SOURCE_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
ROOT = Path(
    os.environ.get("JER_REALNU_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
EXPERIMENT_ROOT = ROOT / "results/experiments/partonic_ttbar2L_2M_realnu_v1"
PRODUCTION_ROOT = EXPERIMENT_ROOT / "production/realnu_top_antitop_20260926"
ATTEMPT_ROOT = EXPERIMENT_ROOT / "attempts/realnu_production_20260926"
CONFIG_ROOT = SOURCE_ROOT / "configs/experiments/realnu_v1/production_draft"
MEASURED_PATH = (
    EXPERIMENT_ROOT
    / "adapted/train_validation/partonic_ttbar2l_train_validation_maxjets8_v1_measured.h5"
)
TRUTH_PATH = (
    EXPERIMENT_ROOT
    / "adapted/train_validation_authoritative_20260926/"
    "partonic_ttbar2l_train_validation_maxjets8_v1_supervision.h5"
)
CACHE_PATH = EXPERIMENT_ROOT / "cache/train_validation_measured_maxjets8_v1.pt"
INITIALISATION_ROOT = EXPERIMENT_ROOT / "production/initialisations_objective_matched_v1"
INITIALISATION_MANIFEST_PATH = INITIALISATION_ROOT / "initialisation_manifest.json"
SOURCE_MANIFEST_PATH = EXPERIMENT_ROOT / "source_manifest_v1.json"
SPLIT_PATH = EXPERIMENT_ROOT / "split_membership_v1.npz"
ROLE_ORDER = ["top.nu", "antitop.nu"]
PARENT_ROLES = ["top", "antitop"]
MODE_CONFIGS = {
    "Q0": CONFIG_ROOT / "q0.public.yaml",
    "Q1": CONFIG_ROOT / "q1.public.yaml",
    "PCFM_Q": CONFIG_ROOT / "pcfm_q.public.yaml",
    "J0": CONFIG_ROOT / "j0.public.yaml",
    "J_A": CONFIG_ROOT / "j_a.public.yaml",
    "J_AN": CONFIG_ROOT / "j_an.public.yaml",
}
STANDARD_MODES = ("Q0", "Q1", "J0", "J_A", "J_AN")
PARENT_MODES = ("PCFM_Q", "J0", "J_A", "J_AN")
EXPECTED_CONDITIONING = {"PCFM_Q": "P0", "J0": "P0", "J_A": "P1", "J_AN": "P2"}
EXPECTED_EVENT_COUNT = 100_342
EXPECTED_POPULATION = 1_900_095
EXPECTED_TRAIN_COUNT = 1_799_753
SOURCE_BRANCH = "research/realnu-training-hotpath-v1"
EXPECTED_G0_SHA = os.environ.get("JER_REALNU_EXPECTED_SHA", "")
if not EXPECTED_G0_SHA:
    source_metadata = SOURCE_ROOT / "SOURCE_SNAPSHOT.json"
    if source_metadata.is_file():
        EXPECTED_G0_SHA = str(json.loads(source_metadata.read_text()).get("commit_sha", ""))
POST_G0_ORCHESTRATION_FILES: set[str] = set()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def artifact_provenance(
    mode: str,
    config,
    manifest: dict,
    *,
    gpu_type: str,
    training_record: dict | None = None,
) -> dict:
    entry = _mode_entry(manifest, mode)
    initialisation = entry["initialisation"]
    standard = mode in STANDARD_MODES
    parent = mode in PARENT_MODES
    result = {
        "campaign": manifest["campaign"],
        "mode": mode,
        "source_commit": manifest["source_commit"],
        "source_branch": SOURCE_BRANCH,
        "run_manifest_path": str(Path(manifest["run_manifest_path"]).resolve()),
        "config_path": entry["config_path"],
        "config_sha256": entry["config_sha256"],
        "source_data_identity": manifest["inputs"],
        "initial_checkpoint": initialisation["checkpoint"],
        "initial_checkpoint_sha256": initialisation["checkpoint_sha256"],
        "initial_components": initialisation["components"],
        "invisible_role_convention": "top_antitop" if standard else None,
        "invisible_role_order": ROLE_ORDER if standard else [],
        "parent_role_order": PARENT_ROLES if parent else [],
        "experiment_seed": int(config.experiment.seed),
        "data_subset_seed": int(config.data.subset_seed),
        "prediction_seed": int(config.inference.seed),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu_type": gpu_type,
    }
    if training_record is not None:
        for key in (
            "resolved_config_sha256",
            "invisible_scaler_fit",
            "partonic_scaler_fit",
            "invisible_scaler_sha256",
            "partonic_scaler_sha256",
            "scaler_identity_sha256",
        ):
            result[key] = training_record.get(key)
    return result


def stamp_prediction_provenance(path: Path, provenance: dict) -> None:
    with h5py.File(path, "r+") as output:
        for key, value in provenance.items():
            if value is None:
                continue
            output.attrs[key] = (
                json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (dict, list))
                else value
            )


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def assert_source(expected_sha: str) -> None:
    if not expected_sha or expected_sha != EXPECTED_G0_SHA:
        raise RuntimeError(
            f"Production source SHA differs from the immutable accepted release: "
            f"expected={EXPECTED_G0_SHA}, requested={expected_sha}"
        )
    dirty = git("status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise RuntimeError(f"Production requires a clean source tree: {dirty}")
    if SOURCE_ROOT != ROOT / "runtime/source" / expected_sha:
        raise RuntimeError("Production source root is not the expected immutable snapshot")
    tree = git("rev-parse", f"{expected_sha}^{{tree}}")
    metadata_path = SOURCE_ROOT / "SOURCE_SNAPSHOT.json"
    metadata = _read_json(metadata_path)
    if (
        metadata.get("schema") != "jer_source_snapshot_v1"
        or metadata.get("commit_sha") != expected_sha
        or metadata.get("tree_sha") != tree
        or (SOURCE_ROOT / ".git").exists()
    ):
        raise RuntimeError("Production immutable source snapshot metadata is invalid")


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required production JSON is missing: {path}")
    return json.loads(path.read_text())


def verify_g0(acceptance_path: Path, expected_sha: str, manifest: dict) -> dict:
    acceptance_path = acceptance_path.expanduser().resolve()
    acceptance = _read_json(acceptance_path)
    if acceptance.get("status") != "passed" or acceptance.get("campaign") != "realnu_top_antitop_v1":
        raise RuntimeError(f"Realnu GPU G0 is not accepted: {acceptance_path}")
    if acceptance.get("source_commit") != EXPECTED_G0_SHA:
        raise RuntimeError(
            "Realnu production requires the accepted G0 from "
            f"{EXPECTED_G0_SHA}; found {acceptance.get('source_commit')!r}"
        )
    if acceptance.get("branch") != SOURCE_BRANCH:
        raise RuntimeError("The accepted G0 is from a different realnu release branch")
    accepted_preflight = acceptance.get("preflight", {})
    for key in ("source_manifest_sha256", "split_membership_sha256", "truth_store_sha256"):
        if accepted_preflight.get(key) != manifest["inputs"].get(key):
            raise RuntimeError(f"G0 acceptance and production input differ at {key}")
    if accepted_preflight.get("invisible_roles") != ROLE_ORDER:
        raise RuntimeError("G0 acceptance has the wrong [top.nu, antitop.nu] role contract")
    if accepted_preflight.get("parent_roles") != PARENT_ROLES:
        raise RuntimeError("G0 acceptance has the wrong [top, antitop] parent contract")
    if int(acceptance.get("test_rows_read", -1)) != 0:
        raise RuntimeError("G0 acceptance does not prove that reserved test rows were untouched")
    expected_path = Path(manifest["g0_acceptance_path"]).resolve()
    if acceptance_path != expected_path:
        raise RuntimeError(f"G0 acceptance path differs from the submitted manifest: {acceptance_path}")
    if sha256(acceptance_path) != manifest["g0_acceptance_sha256"]:
        raise RuntimeError("G0 acceptance artifact changed after production submission")
    return acceptance


def verify_inputs(expected: dict | None = None, *, hash_files: bool = True) -> dict:
    for path in (SOURCE_MANIFEST_PATH, SPLIT_PATH, MEASURED_PATH, TRUTH_PATH, CACHE_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Realnu production input is missing: {path}")
    cache_manifest_path = CACHE_PATH.with_suffix(".json")
    cache_manifest = _read_json(cache_manifest_path)
    source_manifest_sha = sha256(SOURCE_MANIFEST_PATH)
    split_sha = sha256(SPLIT_PATH)
    if hash_files:
        measured_sha = sha256(MEASURED_PATH)
        truth_sha = sha256(TRUTH_PATH)
    elif expected is not None:
        measured_sha = str(expected["measured_input_sha256"])
        truth_sha = str(expected["truth_store_sha256"])
    else:
        raise ValueError("Input hashes are required when full file hashing is disabled")
    measured_stat = MEASURED_PATH.stat()
    truth_stat = TRUTH_PATH.stat()
    cache_stat = CACHE_PATH.stat()
    validation_id_hashes = None
    source_manifest = _read_json(SOURCE_MANIFEST_PATH)
    if source_manifest.get("new_source", {}).get("sha256") != (
        "75e895c0c9ce3984d4c925fa6f850748258e7b1d0e007a757f74cbeaa9cf9196"
    ):
        raise RuntimeError("Realnu source manifest no longer identifies the audited producer file")
    if int(cache_manifest.get("cache_version", -1)) != 9:
        raise RuntimeError("Measured tensor cache is not cache version 9")
    if cache_manifest.get("event_id_fingerprint") != (
        "272496afa7d4e4785daed18a26a2904e5b21a7157d3ec871a0835e55c94e424b"
    ):
        raise RuntimeError("Measured tensor cache event identity differs from the audited population")
    cache_source = cache_manifest.get("source_identity", {})
    if (
        Path(str(cache_source.get("path", ""))).resolve() != MEASURED_PATH.resolve()
        or int(cache_source.get("n_events", -1)) != EXPECTED_POPULATION
        or int(cache_source.get("size_bytes", -1)) != MEASURED_PATH.stat().st_size
    ):
        raise RuntimeError("Measured tensor cache manifest does not match the measured HDF5")
    with h5py.File(MEASURED_PATH, "r") as measured, h5py.File(TRUTH_PATH, "r") as truth:
        if len(measured["EVENT_ID"]) != EXPECTED_POPULATION:
            raise RuntimeError("Measured input does not contain exactly train plus validation")
        if len(truth["EVENT_ID"]) != EXPECTED_POPULATION:
            raise RuntimeError("Truth sidecar does not contain exactly train plus validation")
        if "LABELS" in measured or "TARGETS" in measured:
            raise RuntimeError("Measured production input contains truth datasets")
        if hash_files:
            split_array = np.asarray(measured["SPLIT"][:])
            split_values, split_counts = np.unique(split_array, return_counts=True)
            if np.any(~np.isin(split_values, (0, 1))):
                raise RuntimeError("Measured production input contains reserved-test rows")
            split_counts = {int(key): int(value) for key, value in zip(split_values, split_counts, strict=True)}
            if split_counts != {0: EXPECTED_TRAIN_COUNT, 1: EXPECTED_EVENT_COUNT}:
                raise RuntimeError(f"Frozen train/validation counts changed: {split_counts}")
            validation_ids = np.asarray(measured["EVENT_ID"][:], dtype=np.int64)[split_array == 1]
            if validation_ids.shape != (EXPECTED_EVENT_COUNT,) or len(np.unique(validation_ids)) != EXPECTED_EVENT_COUNT:
                raise RuntimeError("Frozen validation split has invalid or duplicate event IDs")
            validation_id_hashes = {
                "validation_event_id_sha256": _event_id_hash(validation_ids),
                "validation_sorted_event_id_sha256": _event_id_hash(validation_ids, sort=True),
            }
        if (
            str(truth.attrs.get("source_manifest_sha256", "")) != source_manifest_sha
            or str(truth.attrs.get("split_membership_sha256", "")) != split_sha
            or str(truth.attrs.get("invisible_role_convention", "")) != "top_antitop"
            or json.loads(str(truth.attrs.get("invisible_role_order", "[]"))) != ROLE_ORDER
            or str(truth.attrs.get("units", "")) != "GeV"
            or bool(truth.attrs.get("synthetic", True))
        ):
            raise RuntimeError("Real neutrino sidecar metadata differs from the accepted contract")
        if truth["SUPPORT/INVISIBLE_ROLE"].shape != (EXPECTED_POPULATION, 2):
            raise RuntimeError("Real neutrino role support has an unexpected shape")
    identity = {
        "source_manifest_sha256": source_manifest_sha,
        "split_membership_sha256": split_sha,
        "measured_input": str(MEASURED_PATH.resolve()),
        "measured_input_sha256": measured_sha,
        "measured_input_size_bytes": int(measured_stat.st_size),
        "measured_input_mtime_ns": int(measured_stat.st_mtime_ns),
        "truth_store": str(TRUTH_PATH.resolve()),
        "truth_store_sha256": truth_sha,
        "truth_store_size_bytes": int(truth_stat.st_size),
        "truth_store_mtime_ns": int(truth_stat.st_mtime_ns),
        "cache": str(CACHE_PATH.resolve()),
        "cache_size_bytes": int(cache_stat.st_size),
        "cache_mtime_ns": int(cache_stat.st_mtime_ns),
        "cache_fingerprint": cache_manifest["event_id_fingerprint"],
        "event_count": EXPECTED_POPULATION,
        "train_count": EXPECTED_TRAIN_COUNT,
        "validation_count": EXPECTED_EVENT_COUNT,
        "reserved_test_rows_materialised": 0,
        "invisible_roles": ROLE_ORDER,
        "parent_roles": PARENT_ROLES,
        **(
            validation_id_hashes
            if validation_id_hashes is not None
            else {
                "validation_event_id_sha256": expected["validation_event_id_sha256"],
                "validation_sorted_event_id_sha256": expected[
                    "validation_sorted_event_id_sha256"
                ],
            }
        ),
    }
    if expected is not None:
        for key, value in identity.items():
            if key in expected and expected[key] != value:
                raise RuntimeError(f"Input identity changed after G0 acceptance at {key}")
    return identity


def load_and_validate_config(mode: str, config_path: Path, expected_output: Path | None = None):
    from joint_event_reconstruction.configuration import load_experiment
    from joint_event_reconstruction.topology.compiler import load_topology

    config = load_experiment(config_path)
    spec = load_topology(config.topology_config, config.data.truth_bindings)
    if (
        Path(config.data.input).resolve() != MEASURED_PATH.resolve()
        or Path(config.data.inference_input).resolve() != MEASURED_PATH.resolve()
        or Path(config.data.tensor_cache).resolve() != CACHE_PATH.resolve()
        or int(config.training.batch_size) != 512
        or int(config.training.epochs) != 20
        or int(config.training.workers) != 8
        or str(config.training.precision) != "bf16-mixed"
        or not bool(config.training.fused_adamw)
        or any(value is not None for value in (config.data.max_events.train, config.data.max_events.val, config.data.max_events.test))
    ):
        raise RuntimeError(f"{mode} production config differs from the full-data 20-epoch release")
    if mode == "PCFM_Q":
        if config.data.visible_truth is not None:
            raise RuntimeError("PCFM_Q must not load visible truth")
    elif Path(config.data.visible_truth).resolve() != TRUTH_PATH.resolve():
        raise RuntimeError(f"{mode} does not use the authoritative visible-truth sidecar")
    if mode in STANDARD_MODES:
        if [role.key for role in spec.invisible_roles] != ROLE_ORDER:
            raise RuntimeError(f"{mode} topology does not use [top.nu, antitop.nu]")
        if not bool(config.model.invisible.enabled):
            raise RuntimeError(f"{mode} must train the real top/antitop neutrino model")
        if Path(config.data.invisible.path).resolve() != TRUTH_PATH.resolve():
            raise RuntimeError(f"{mode} does not use the authoritative neutrino sidecar")
    if mode in PARENT_MODES:
        if not bool(config.model.parent_cfm.enabled):
            raise RuntimeError(f"{mode} parent CFM is disabled")
        if list(config.model.parent_cfm.roles) != PARENT_ROLES:
            raise RuntimeError(f"{mode} parent roles are not ordered [top, antitop]")
        if str(config.model.parent_cfm.conditioning) != EXPECTED_CONDITIONING[mode]:
            raise RuntimeError(f"{mode} parent conditioning differs from its frozen mode")
        if Path(config.data.partonic.path).resolve() != TRUTH_PATH.resolve():
            raise RuntimeError(f"{mode} does not train on the authoritative parent sidecar")
    output = Path(config.training.output).expanduser().resolve()
    if expected_output is not None and output != expected_output.resolve():
        raise RuntimeError(f"{mode} training output changed: {output}")
    return config, spec, output


def load_run_manifest(path: Path, expected_sha: str) -> dict:
    manifest = _read_json(path.expanduser().resolve())
    if manifest.get("campaign") != "realnu_top_antitop_v1":
        raise RuntimeError("Production run manifest names the wrong campaign")
    if manifest.get("source_commit") != expected_sha:
        raise RuntimeError("Production run manifest source SHA differs from the allocated job")
    if (
        manifest.get("source_branch") != SOURCE_BRANCH
        or Path(manifest.get("source_root", "")).resolve() != SOURCE_ROOT
    ):
        raise RuntimeError("Production run manifest has the wrong immutable source identity")
    changed = set(git("diff", "--name-only", EXPECTED_G0_SHA, expected_sha).splitlines())
    if changed != POST_G0_ORCHESTRATION_FILES:
        raise RuntimeError(
            "Production must use the exact accepted G0 source; "
            f"found {sorted(changed)}"
        )
    if Path(manifest.get("attempt_root", "")).resolve() != ATTEMPT_ROOT.resolve():
        raise RuntimeError("Production attempt root does not match the fixed campaign path")
    initialisation_manifest = Path(manifest.get("initialisation_manifest", "")).resolve()
    if (
        initialisation_manifest != INITIALISATION_MANIFEST_PATH.resolve()
        or not initialisation_manifest.is_file()
        or sha256(initialisation_manifest) != manifest.get("initialisation_manifest_sha256")
    ):
        raise RuntimeError("Production initialisation manifest is missing or changed")
    initialisation_payload = _read_json(initialisation_manifest)
    if set(manifest.get("modes", {})) != set(MODE_CONFIGS):
        raise RuntimeError("Production run manifest does not contain exactly the six release modes")
    for mode, entry in manifest["modes"].items():
        initial = entry.get("initialisation", {})
        initial_record = initialisation_payload.get("modes", {}).get(mode, {})
        checkpoint = Path(initial.get("checkpoint", "")).resolve()
        if (
            not checkpoint.is_file()
            or sha256(checkpoint) != initial.get("checkpoint_sha256")
            or not initial.get("components")
            or checkpoint != Path(initial_record.get("checkpoint", "")).resolve()
            or initial.get("component_sha256") != initial_record.get("component_sha256")
            or sorted(initial.get("components", []))
            != sorted(initial_record.get("component_sha256", {}))
        ):
            raise RuntimeError(f"{mode} initial checkpoint identity is missing or changed")
        if not initial.get("component_sha256") or sorted(initial.get("components", [])) != sorted(
            initial["component_sha256"]
        ):
            raise RuntimeError(f"{mode} initial checkpoint component contract is incomplete")
    verify_g0(Path(manifest["g0_acceptance_path"]), expected_sha, manifest)
    verify_inputs(manifest["inputs"], hash_files=False)
    for mode, entry in manifest["modes"].items():
        config_path = Path(entry["config_path"]).resolve()
        if config_path != MODE_CONFIGS[mode].resolve() or sha256(config_path) != entry["config_sha256"]:
            raise RuntimeError(f"{mode} production config changed after submission")
        load_and_validate_config(mode, config_path, Path(entry["training_output"]))
    return manifest


def _mode_entry(manifest: dict, mode: str) -> dict:
    try:
        return manifest["modes"][mode]
    except KeyError as error:
        raise ValueError(f"Unknown realnu production mode {mode!r}") from error


def run_training(mode: str, expected_sha: str, manifest_path: Path, python_bin: Path) -> dict:
    assert_source(expected_sha)
    manifest = load_run_manifest(manifest_path, expected_sha)
    entry = _mode_entry(manifest, mode)
    config_path = Path(entry["config_path"])
    config, _, output = load_and_validate_config(
        mode, config_path, Path(entry["training_output"])
    )
    if output.exists():
        raise FileExistsError(f"Refusing to reuse {mode} production training output: {output}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{mode} production training has no allocated CUDA device")
    gpu_type = torch.cuda.get_device_name(0)
    provenance = artifact_provenance(mode, config, manifest, gpu_type=gpu_type)
    output.mkdir(parents=True)
    progress_path = ATTEMPT_ROOT / f"train_{mode}.json"
    progress = {
        "status": "running",
        "g0_acceptance_sha256": manifest["g0_acceptance_sha256"],
        "training_output": str(output),
        "started_utc": datetime.now(UTC).isoformat(),
        **provenance,
    }
    atomic_json(progress_path, progress)
    command = [
        str(python_bin), "-m", "joint_event_reconstruction.cli.run", "train",
        "--experiment", str(config_path), "--device", "cuda",
        "--initial-checkpoint", entry["initialisation"]["checkpoint"],
        "--initial-components", *entry["initialisation"]["components"],
    ]
    env = os.environ.copy()
    env["JER_PROJECT_ROOT"] = str(ROOT)
    env["JER_SOURCE_SHA"] = expected_sha
    env["JER_SOURCE_BRANCH"] = SOURCE_BRANCH
    env["JER_SOURCE_ROOT"] = str(SOURCE_ROOT)
    env["JER_REALNU_SOURCE_ROOT"] = str(SOURCE_ROOT)
    env["JER_REALNU_PROJECT_ROOT"] = str(ROOT)
    env["JER_REALNU_EXPECTED_SHA"] = expected_sha
    env["PYTHONPATH"] = f"{SOURCE_ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(name, "1")
    started = time.time()
    subprocess.run(command, cwd=ROOT, env=env, check=True)

    checkpoint_path = output / "checkpoints/best-loss.ckpt"
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(f"{mode} training did not create a best-loss checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    hyper = checkpoint.get("hyper_parameters", {})
    if int(checkpoint.get("global_step", 0)) <= 0:
        raise RuntimeError(f"{mode} best checkpoint has no completed optimiser step")
    if hyper.get("git_sha") != expected_sha:
        raise RuntimeError(f"{mode} checkpoint source SHA does not match the submitted commit")
    if hyper.get("branch") != SOURCE_BRANCH:
        raise RuntimeError(f"{mode} checkpoint branch is not the realnu release branch")
    if sorted(hyper.get("trainable_components", [])) != sorted(entry["trainable_components"]):
        raise RuntimeError(f"{mode} checkpoint trainable components differ from its config")
    if (
        Path(str(hyper.get("initial_checkpoint", ""))).resolve()
        != Path(entry["initialisation"]["checkpoint"]).resolve()
        or sorted(hyper.get("initial_components", []))
        != sorted(entry["initialisation"]["components"])
    ):
        raise RuntimeError(f"{mode} checkpoint does not record the approved initialisation")
    if mode in STANDARD_MODES:
        contract = hyper.get("invisible_truth_contract", {})
        scaler_fit = hyper.get("invisible_scaler_fit", {})
        if contract.get("invisible_role_order") != ROLE_ORDER:
            raise RuntimeError(f"{mode} checkpoint lacks the [top.nu, antitop.nu] truth contract")
        if contract.get("source_manifest_sha256") != manifest["inputs"]["source_manifest_sha256"]:
            raise RuntimeError(f"{mode} checkpoint neutrino source manifest changed")
        if scaler_fit.get("population") != "frozen_train_split" or int(
            scaler_fit.get("event_count", -1)
        ) != EXPECTED_TRAIN_COUNT:
            raise RuntimeError(f"{mode} neutrino scaler was not fit on the complete frozen train split")
    if mode in PARENT_MODES:
        contract = hyper.get("partonic_truth_contract", {})
        scaler_fit = hyper.get("partonic_scaler_fit", {})
        if contract.get("roles") != PARENT_ROLES:
            raise RuntimeError(f"{mode} checkpoint lacks the [top, antitop] parent contract")
        if contract.get("source_manifest_sha256") != manifest["inputs"]["source_manifest_sha256"]:
            raise RuntimeError(f"{mode} checkpoint parent source manifest changed")
        if scaler_fit.get("population") != "frozen_train_split" or int(
            scaler_fit.get("event_count", -1)
        ) != EXPECTED_TRAIN_COUNT:
            raise RuntimeError(f"{mode} parent scaler was not fit on the complete frozen train split")
    finite_count = 0
    for name, value in checkpoint.get("state_dict", {}).items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"{mode} checkpoint has non-finite model weights at {name}")
            finite_count += 1
    callbacks = checkpoint.get("callbacks", {})
    validation_scores = [
        float(state["best_model_score"])
        for name, state in callbacks.items()
        if "ModelCheckpoint" in name and isinstance(state, dict) and state.get("best_model_score") is not None
    ]
    if not validation_scores or not np.isfinite(validation_scores).all():
        raise RuntimeError(f"{mode} checkpoint has no finite validation score")
    scaler_fits = {
        "invisible": hyper.get("invisible_scaler_fit"),
        "partonic": hyper.get("partonic_scaler_fit"),
    }
    scaler_values = {
        "invisible": checkpoint.get("invisible_scaler"),
        "partonic": checkpoint.get("partonic_scaler"),
    }
    provenance.update(
        {
            "resolved_config_sha256": hyper.get("resolved_config_sha256"),
            "invisible_scaler_fit": scaler_fits["invisible"],
            "partonic_scaler_fit": scaler_fits["partonic"],
            "invisible_scaler_sha256": (
                json_sha256({"scaler": scaler_values["invisible"], "fit": scaler_fits["invisible"]})
                if scaler_values["invisible"] is not None
                else None
            ),
            "partonic_scaler_sha256": (
                json_sha256({"scaler": scaler_values["partonic"], "fit": scaler_fits["partonic"]})
                if scaler_values["partonic"] is not None
                else None
            ),
            "scaler_identity_sha256": json_sha256(
                {"scalers": scaler_values, "fit": scaler_fits}
            ),
        }
    )
    record = {
        "status": "passed",
        "g0_acceptance_path": manifest["g0_acceptance_path"],
        "g0_acceptance_sha256": manifest["g0_acceptance_sha256"],
        "training_output": str(output),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "global_step": int(checkpoint["global_step"]),
        "floating_state_tensor_count": finite_count,
        "validation_score": min(validation_scores),
        "train_count": EXPECTED_TRAIN_COUNT,
        "validation_count": EXPECTED_EVENT_COUNT,
        "reserved_test_rows_read": 0,
        "elapsed_seconds": time.time() - started,
        "completed_utc": datetime.now(UTC).isoformat(),
        **provenance,
    }
    atomic_json(output / "production_training.json", record)
    progress.update(record)
    atomic_json(progress_path, progress)
    return record


def _event_id_hash(values: np.ndarray, *, sort: bool = False) -> str:
    ids = np.asarray(values, dtype="<i8")
    if sort:
        ids = np.sort(ids)
    return hashlib.sha256(ids.tobytes()).hexdigest()


def _read_prediction_ids(path: Path, expected_count: int) -> tuple[np.ndarray, dict]:
    with h5py.File(path, "r") as prediction:
        ids = np.asarray(prediction["EVENT_ID"][:], dtype=np.int64)
    if ids.shape != (expected_count,) or len(np.unique(ids)) != expected_count:
        raise RuntimeError(f"Validation prediction has malformed or duplicate event IDs: {path}")
    return ids, {
        "event_count": len(ids),
        "event_id_sha256": _event_id_hash(ids),
        "sorted_event_id_sha256": _event_id_hash(ids, sort=True),
    }


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot summarise empty or non-finite validation values")
    return {
        "mean": float(values.mean()),
        "rmse": float(np.sqrt(np.square(values).mean())),
        "median": float(np.median(values)),
    }


def _posterior_scalar_metrics(samples: np.ndarray, truth: np.ndarray) -> dict:
    samples = np.asarray(samples, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if (
        samples.ndim != 2
        or truth.shape != (len(samples),)
        or samples.shape[1] == 0
        or not np.isfinite(samples).all()
        or not np.isfinite(truth).all()
    ):
        raise ValueError("Posterior metrics require finite samples [N,S] and truth [N]")
    residual = samples.mean(axis=1) - truth
    low68, high68 = np.quantile(samples, [0.16, 0.84], axis=1)
    low95, high95 = np.quantile(samples, [0.025, 0.975], axis=1)
    return {
        "events": len(truth),
        "samples_per_event": int(samples.shape[1]),
        "posterior_mean_bias": float(residual.mean()),
        "posterior_mean_rmse": float(np.sqrt(np.square(residual).mean())),
        "posterior_mean_resolution": float(residual.std()),
        "median_posterior_mean_residual": float(np.median(residual)),
        "central_68_width": float(np.median(high68 - low68)),
        "central_68_coverage": float(((truth >= low68) & (truth <= high68)).mean()),
        "central_95_coverage": float(((truth >= low95) & (truth <= high95)).mean()),
    }


def evaluate_standard_prediction(
    prediction_path: Path,
    config,
    provenance: dict,
    *,
    expected_event_count: int | None = EXPECTED_EVENT_COUNT,
) -> dict:
    """Join a completed truth-free prediction to validation truth for metrics only."""
    from joint_event_reconstruction.cli.run import build_data, load_visible_config

    measured_path = Path(config.data.input).expanduser().resolve()
    inference_path = Path(config.data.inference_input).expanduser().resolve()
    if measured_path != inference_path:
        raise RuntimeError("Validation metrics require the same measured input used for inference")
    visible = load_visible_config(config)
    evaluation_data = build_data(config, visible, include_truth=True, device="cpu")
    evaluation_data.setup("predict")
    collator = evaluation_data.structured_collator
    if collator.visible_store is None or collator.invisible_store is None:
        raise RuntimeError("Validation metrics require visible and real-neutrino truth stores")
    if [role.key for role in collator.spec.invisible_roles] != ROLE_ORDER:
        raise RuntimeError("Validation truth loader changed the [top.nu, antitop.nu] order")
    truth_root_order = list(collator.spec.roots)

    truth_ids_parts = []
    neutrino_parts = []
    neutrino_support_parts = []
    node_truth_parts: dict[str, list[np.ndarray]] = {
        node: [] for node in collator.spec.roots
    }
    node_support_parts: dict[str, list[np.ndarray]] = {
        node: [] for node in collator.spec.roots
    }
    for batch in evaluation_data.val_dataloader():
        truth_ids_parts.append(
            batch["event_index"].detach().cpu().numpy().astype(np.int64)
        )
        neutrino_parts.append(
            batch["invisible_truth"].detach().cpu().numpy().astype(np.float64)
        )
        neutrino_support_parts.append(
            batch["invisible_support"].detach().cpu().numpy().astype(bool)
        )
        for node in collator.spec.roots:
            pool = batch["structured"]["pools"][node]
            node_truth_parts[node].append(
                pool["truth_index"].detach().cpu().numpy().astype(np.int64)
            )
            node_support_parts[node].append(
                pool["reconstructable"].detach().cpu().numpy().astype(bool)
            )
    del evaluation_data, visible, collator
    gc.collect()

    truth_ids = np.concatenate(truth_ids_parts)
    truth_neutrino = np.concatenate(neutrino_parts)
    truth_neutrino_support = np.concatenate(neutrino_support_parts)
    node_truth = {node: np.concatenate(values) for node, values in node_truth_parts.items()}
    node_support = {node: np.concatenate(values) for node, values in node_support_parts.items()}

    with h5py.File(measured_path, "r") as measured:
        split = np.asarray(measured["SPLIT"][:], dtype=np.uint8)
        measured_ids = np.asarray(measured["EVENT_ID"][:], dtype=np.int64)
        validation_mask = split == 1
        validation_ids = measured_ids[validation_mask]
        validation_global_inputs = np.asarray(measured["INPUTS/GLOBAL"][validation_mask])
    with h5py.File(prediction_path, "r") as prediction:
        prediction_ids = np.asarray(prediction["EVENT_ID"][:], dtype=np.int64)
        prediction_source = Path(
            str(prediction.attrs.get("source_data", ""))
        ).expanduser().resolve()
        prediction_split = str(prediction.attrs.get("split", ""))
        root_order = json.loads(str(prediction.attrs["root_order"]))
        root_predictions = np.asarray(
            prediction["VISIBLE/root_candidate_indices"][:], dtype=np.int64
        )
        njet = np.asarray(prediction["OBSERVED/njet"][:], dtype=np.int64)
        samples = np.asarray(prediction["INVISIBLE/samples"][:], dtype=np.float64)
        four_vectors = np.asarray(
            prediction["INVISIBLE/four_vectors"][:], dtype=np.float64
        )
        met = np.asarray(prediction["OBSERVED/MET"][:, 1:3], dtype=np.float64)
    if prediction_source != measured_path or prediction_split != "validation":
        raise RuntimeError("Prediction source or split differs from the truth-side validation input")
    if (
        prediction_ids.ndim != 1
        or len(np.unique(prediction_ids)) != len(prediction_ids)
        or (expected_event_count is not None and len(prediction_ids) != expected_event_count)
    ):
        raise RuntimeError("Standard validation prediction has malformed or unexpected EVENT_IDs")
    if len(np.unique(truth_ids)) != len(truth_ids):
        raise RuntimeError("Validation truth loader returned duplicate EVENT_ID values")
    truth_order = np.argsort(truth_ids, kind="stable")
    truth_positions = np.searchsorted(truth_ids[truth_order], prediction_ids)
    if np.any(truth_positions >= len(truth_order)):
        raise RuntimeError("Prediction EVENT_ID values are absent from validation truth")
    truth_rows = truth_order[truth_positions]
    if not np.array_equal(truth_ids[truth_rows], prediction_ids):
        raise RuntimeError("Prediction EVENT_ID values are absent from validation truth")
    truth_ids = truth_ids[truth_rows]
    truth_neutrino = truth_neutrino[truth_rows]
    truth_neutrino_support = truth_neutrino_support[truth_rows]
    node_truth = {node: values[truth_rows] for node, values in node_truth.items()}
    node_support = {node: values[truth_rows] for node, values in node_support.items()}
    if len(np.unique(validation_ids)) != len(validation_ids):
        raise RuntimeError("Measured validation split contains duplicate EVENT_ID values")
    validation_order = np.argsort(validation_ids, kind="stable")
    validation_positions = np.searchsorted(validation_ids[validation_order], prediction_ids)
    if (
        np.any(validation_positions >= len(validation_order))
        or not np.array_equal(
            validation_ids[validation_order[validation_positions]], prediction_ids
        )
    ):
        raise RuntimeError("Prediction EVENT_ID values are absent from the measured validation split")
    global_inputs = validation_global_inputs[validation_order[validation_positions]]
    if root_order != truth_root_order or root_predictions.shape[2] != len(root_order):
        raise RuntimeError("Prediction root order differs from the validation truth topology")
    if samples.shape[0] != len(prediction_ids) or samples.shape[3:] != (2, 3):
        raise RuntimeError("Standard validation prediction has malformed neutrino samples")

    electrons = np.rint(global_inputs["nelectron"]).astype(np.int64)
    muons = np.rint(global_inputs["nmuon"]).astype(np.int64)
    channels = np.full(len(prediction_ids), "other", dtype=object)
    channels[(electrons == 2) & (muons == 0)] = "ee"
    channels[(electrons == 1) & (muons == 1)] = "emu"
    channels[(electrons == 0) & (muons == 2)] = "mumu"
    if not np.array_equal(njet, np.rint(global_inputs["njet"]).astype(np.int64)):
        raise RuntimeError("Prediction and measured input jet counts differ")

    complete_support = np.ones(len(prediction_ids), dtype=bool)
    root_truth = []
    role_assignment = {}
    for index, node in enumerate(root_order):
        support = node_support[node] & (node_truth[node] >= 0)
        complete_support &= support
        root_truth.append(node_truth[node])
        prediction_node = root_predictions[:, 0, index]
        role_assignment[node] = {
            "events": len(prediction_ids),
            "supported_events": int(support.sum()),
            "top1_accuracy": float((prediction_node[support] == node_truth[node][support]).mean())
            if support.any()
            else None,
            "null_fraction": float((prediction_node[support] < 0).mean())
            if support.any()
            else None,
        }
        for variable_name, variable in (("njet", njet), ("channel", channels)):
            bins = {}
            for value in np.unique(variable):
                selected_mask = support & (variable == value)
                count = int(selected_mask.sum())
                bins[str(value)] = {
                    "supported_events": count,
                    "top1_accuracy": (
                        float(
                            (
                                prediction_node[selected_mask]
                                == node_truth[node][selected_mask]
                            ).mean()
                        )
                        if count
                        else None
                    ),
                }
            role_assignment[node][f"by_{variable_name}"] = bins
    root_truth_array = np.stack(root_truth, axis=1)
    topk_correct = (root_predictions == root_truth_array[:, None, :]).all(axis=2)
    supported_top1 = complete_support & topk_correct[:, 0]
    supported_topk = complete_support & topk_correct.any(axis=1)
    assignment = {
        "support_definition": "all root roles reconstructable and have a non-null truth candidate",
        "supported_events": int(complete_support.sum()),
        "top1_complete_event_accuracy": (
            float(supported_top1.sum() / complete_support.sum())
            if complete_support.any()
            else None
        ),
        "topk_complete_event_coverage": (
            float(supported_topk.sum() / complete_support.sum())
            if complete_support.any()
            else None
        ),
        "per_role": role_assignment,
    }

    if not truth_neutrino_support.any(axis=0).all():
        raise RuntimeError("Validation truth has an unsupported real-neutrino role")
    posterior = samples[:, 0]
    conditional = {"njet": {}, "channel": {}}

    def posterior_summary(selection: np.ndarray, role_indices: list[int]) -> dict:
        selected_samples = posterior[selection][:, :, role_indices, :]
        selected_truth = truth_neutrino[selection][:, role_indices, :]
        selected_support = truth_neutrino_support[selection][:, role_indices]
        if not selected_support.all():
            raise RuntimeError("Selected posterior metric events lack real-neutrino support")
        result = {"events": len(selected_truth), "roles": {}}
        for local_index, role_index in enumerate(role_indices):
            role = ROLE_ORDER[role_index]
            role_samples = selected_samples[:, :, local_index]
            role_truth = selected_truth[:, local_index]
            component_metrics = {
                name: _posterior_scalar_metrics(role_samples[..., index], role_truth[:, index])
                for index, name in enumerate(("px", "py", "pz"))
            }
            transverse_samples = np.hypot(role_samples[..., 0], role_samples[..., 1])
            transverse_truth = np.hypot(role_truth[:, 0], role_truth[:, 1])
            component_metrics["pT"] = _posterior_scalar_metrics(
                transverse_samples, transverse_truth
            )
            sample_phi = np.arctan2(role_samples[..., 1], role_samples[..., 0])
            truth_phi = np.arctan2(role_truth[:, 1], role_truth[:, 0])
            wrapped_phi = np.arctan2(
                np.sin(sample_phi - truth_phi[:, None]),
                np.cos(sample_phi - truth_phi[:, None]),
            )
            component_metrics["wrapped_phi_residual"] = _posterior_scalar_metrics(
                wrapped_phi, np.zeros(len(truth_phi))
            )
            result["roles"][role] = component_metrics
        return result

    both_supported = truth_neutrino_support.all(axis=1)
    if not both_supported.any():
        raise RuntimeError("No validation events support both real-neutrino roles")
    joint_posterior = posterior_summary(both_supported, [0, 1])
    role_metrics = {}
    for role_index, role in enumerate(ROLE_ORDER):
        mask = truth_neutrino_support[:, role_index]
        if not mask.any():
            raise RuntimeError(f"No validation truth supports {role}")
        role_metrics[role] = posterior_summary(mask, [role_index])["roles"][role]
    for variable_name, variable in (("njet", njet), ("channel", channels)):
        for value in np.unique(variable):
            selected_mask = both_supported & (variable == value)
            count = int(selected_mask.sum())
            conditional[variable_name][str(value)] = {
                "supported_events": count,
                "posterior": posterior_summary(selected_mask, [0, 1])
                if count
                else None,
            }

    met_prediction = four_vectors[:, 0, :, :, 1:3].sum(axis=2)
    met_residual = met_prediction - met[:, None, :]
    met_closure = {
        "events": len(prediction_ids),
        "samples_per_event": int(met_prediction.shape[1]),
        "px_residual": _summary(met_residual[..., 0]),
        "py_residual": _summary(met_residual[..., 1]),
        "transverse_residual_magnitude": _summary(
            np.linalg.norm(met_residual, axis=-1)
        ),
    }
    return {
        "status": "passed",
        "events": len(prediction_ids),
        "provenance": provenance,
        "prediction_truth_free": True,
        "event_id_sha256": _event_id_hash(prediction_ids),
        "sorted_event_id_sha256": _event_id_hash(prediction_ids, sort=True),
        "assignment": assignment,
        "neutrino": {
            "role_order": ROLE_ORDER,
            "support_by_role": {
                role: int(truth_neutrino_support[:, index].sum())
                for index, role in enumerate(ROLE_ORDER)
            },
            "both_roles_supported": int(both_supported.sum()),
            "posterior_top1_assignment_conditioned": joint_posterior,
            "per_role": role_metrics,
            "conditional": conditional,
            "met_closure": met_closure,
        },
    }


def _make_standard_prediction(
    mode: str, config, checkpoint: Path, output: Path, provenance: dict
) -> dict:
    import lightning as L
    import torch

    from joint_event_reconstruction.cli.run import (
        build_data,
        load_trained,
        load_visible_config,
    )
    from joint_event_reconstruction.evaluation_joint import evaluate_any_predictions
    from joint_event_reconstruction.prediction import predict_hdf5
    from joint_event_reconstruction.prediction_schema import validate_prediction

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    seed = int(config.inference.seed)
    L.seed_everything(seed, workers=True)
    visible = load_visible_config(config)
    data = build_data(config, visible, include_truth=False, device=device)
    data.setup("predict")
    if any(
        store is not None
        for store in (
            data.structured_collator.visible_store,
            data.structured_collator.invisible_store,
            data.structured_collator.partonic_store,
        )
    ):
        raise RuntimeError(f"{mode} standard prediction loaded truth before inference")
    module = load_trained(config, visible, data, str(checkpoint))
    prediction_path = output / "validation.h5"
    predict_hdf5(
        module,
        data,
        data.invisible_scaler,
        prediction_path,
        split="validation",
        top_k=int(config.inference.top_k_assignments),
        posterior_samples=int(config.inference.posterior_samples),
        device=device,
        prediction_seed=seed,
        checkpoint_path=str(checkpoint),
        compatibility_weight=float(config.inference.compatibility_weight),
        point_estimator=str(config.inference.invisible.point_estimator),
    )
    validate_prediction(prediction_path, deep=True)
    with h5py.File(prediction_path, "r") as prediction:
        roles = json.loads(str(prediction.attrs.get("invisible_roles", "[]")))
        if roles != ROLE_ORDER:
            raise RuntimeError(f"{mode} validation prediction has the wrong neutrino role order")
        if (
            str(prediction.attrs.get("split", "")) != "validation"
            or Path(str(prediction.attrs.get("source_data", ""))).resolve() != MEASURED_PATH.resolve()
            or bool(prediction.attrs.get("visible_truth_available", False))
            or bool(prediction.attrs.get("invisible_truth_available", False))
        ):
            raise RuntimeError(f"{mode} standard inference was not truth-free")
    stamp_prediction_provenance(prediction_path, provenance)
    ids, id_record = _read_prediction_ids(prediction_path, EXPECTED_EVENT_COUNT)
    evaluation_path = output / "evaluation.json"
    prediction_contract = evaluate_any_predictions(prediction_path)
    if (
        int(prediction_contract.get("events", -1)) != EXPECTED_EVENT_COUNT
        or prediction_contract.get("schema_validated") is not True
        or prediction_contract.get("visible_truth_available") is not False
        or prediction_contract.get("invisible_truth_available") is not False
    ):
        raise RuntimeError(f"{mode} truth-free validation prediction contract failed")
    del module, data, visible
    gc.collect()
    torch.cuda.empty_cache()
    evaluation = evaluate_standard_prediction(prediction_path, config, provenance)
    evaluation["prediction_contract"] = prediction_contract
    if (
        int(evaluation.get("events", -1)) != EXPECTED_EVENT_COUNT
        or evaluation.get("prediction_truth_free") is not True
        or int(evaluation.get("assignment", {}).get("supported_events", 0)) <= 0
        or int(evaluation.get("neutrino", {}).get("both_roles_supported", 0)) <= 0
    ):
        raise RuntimeError(f"{mode} truth-side evaluation did not cover valid validation events")
    atomic_json(evaluation_path, evaluation)
    result = {
        "prediction": str(prediction_path.resolve()),
        "evaluation": str(evaluation_path.resolve()),
        "roles": ROLE_ORDER,
        "truth_free_input": True,
        "truth_evaluated_after_prediction": True,
        "inference": {
            "top_k_assignments": int(config.inference.top_k_assignments),
            "posterior_samples": int(config.inference.posterior_samples),
            "flow_steps": int(config.inference.invisible.flow.steps),
        },
        **id_record,
    }
    del ids
    gc.collect()
    return result


def mttbar_threshold_metrics(prediction_path: Path, truth_path: Path) -> dict:
    with h5py.File(prediction_path, "r") as prediction:
        prediction_ids = np.asarray(prediction["EVENT_ID"][:], dtype=np.int64)
        pair_samples = np.asarray(prediction["PARTONIC/ttbar"][:], dtype=np.float64)
        if not bool(prediction.attrs.get("truth_free_input", False)):
            raise RuntimeError("Parent validation must derive from truth-free measured input")
        if json.loads(str(prediction.attrs.get("parent_roles", "[]"))) != PARENT_ROLES:
            raise RuntimeError("Parent validation has the wrong [top, antitop] role order")
    with h5py.File(truth_path, "r") as truth:
        truth_ids = np.asarray(truth["EVENT_ID"][:], dtype=np.int64)
        order = np.argsort(truth_ids, kind="stable")
        positions = np.searchsorted(truth_ids[order], prediction_ids)
        if np.any(positions >= len(order)):
            raise RuntimeError("Parent predictions contain event IDs absent from truth sidecar")
        rows = order[positions]
        if not np.array_equal(truth_ids[rows], prediction_ids):
            raise RuntimeError("Parent prediction and truth sidecar EVENT_ID sets differ")
        support = np.asarray(truth["PARTONIC/SUPPORT"][rows], dtype=bool)
        top = np.asarray(truth["PARTONIC/top"][rows], dtype=np.float64)
        antitop = np.asarray(truth["PARTONIC/antitop"][rows], dtype=np.float64)
    if support.shape != (len(prediction_ids), 2) or not support.all():
        raise RuntimeError("Parent mttbar validation requires both supported truth parents")
    truth_pair = top + antitop
    truth_mass2 = np.square(truth_pair[:, 0]) - np.square(truth_pair[:, 1:]).sum(axis=1)
    if (truth_mass2 < -1e-5).any() or not np.isfinite(truth_mass2).all():
        raise RuntimeError("Truth top+antitop sum has non-physical or non-finite invariant mass")
    truth_mass = np.sqrt(np.maximum(truth_mass2, 0.0))
    if not np.isfinite(truth_mass).all() or np.any(truth_mass <= 0.0):
        raise RuntimeError("Truth mttbar has a non-finite or zero normalization denominator")
    pair_mass2 = np.square(pair_samples[..., 0]) - np.square(pair_samples[..., 1:]).sum(axis=-1)
    if (pair_mass2 < -1e-5).any() or not np.isfinite(pair_mass2).all():
        raise RuntimeError("Predicted ttbar samples have non-physical or non-finite invariant mass")
    predicted_mass = np.sqrt(np.maximum(pair_mass2, 0.0))
    predicted_mean = predicted_mass.mean(axis=1)
    residual = predicted_mean - truth_mass
    edges = [400.0, 500.0, 700.0, 1000.0]
    labels = ["<400", "400-500", "500-700", "700-1000", ">1000"]
    truth_bin = np.digitize(truth_mass, edges, right=False)
    predicted_bin = np.digitize(predicted_mean, edges, right=False)
    bins = {}
    response_counts = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for index, label in enumerate(labels):
        mask = truth_bin == index
        response_counts[index] = np.bincount(
            predicted_bin[mask], minlength=len(labels)
        )[: len(labels)]
        if not mask.any():
            bins[label] = {"events": 0}
            continue
        local_samples = predicted_mass[mask]
        low68, high68 = np.quantile(local_samples, [0.16, 0.84], axis=1)
        local_residual = residual[mask]
        relative = local_residual / truth_mass[mask]
        bins[label] = {
            "events": int(mask.sum()),
            "truth_mean_gev": float(truth_mass[mask].mean()),
            "predicted_posterior_mean_gev": float(predicted_mean[mask].mean()),
            "bias_gev": float(local_residual.mean()),
            "resolution_gev": float(local_residual.std()),
            "rmse_gev": float(np.sqrt(np.square(local_residual).mean())),
            "relative_bias": float(relative.mean()),
            "posterior_central_68_coverage": float(
                ((truth_mass[mask] >= low68) & (truth_mass[mask] <= high68)).mean()
            ),
        }
    threshold_mask = (truth_mass >= 345.0) & (truth_mass < 400.0)
    if threshold_mask.any():
        threshold_residual = residual[threshold_mask]
        threshold_samples = predicted_mass[threshold_mask]
        low68, high68 = np.quantile(threshold_samples, [0.16, 0.84], axis=1)
        threshold_window = {
            "range_gev": [345.0, 400.0],
            "interpretation": "nominal 2 x 172.5 GeV threshold through 55 GeV above threshold",
            "events": int(threshold_mask.sum()),
            "bias_gev": float(threshold_residual.mean()),
            "resolution_gev": float(threshold_residual.std()),
            "rmse_gev": float(np.sqrt(np.square(threshold_residual).mean())),
            "posterior_central_68_coverage": float(
                (
                    (truth_mass[threshold_mask] >= low68)
                    & (truth_mass[threshold_mask] <= high68)
                ).mean()
            ),
        }
    else:
        threshold_window = {
            "range_gev": [345.0, 400.0],
            "interpretation": "nominal 2 x 172.5 GeV threshold through 55 GeV above threshold",
            "events": 0,
            "status": "no supported validation events in this window",
        }
    correlation = (
        float(np.corrcoef(truth_mass, predicted_mean)[0, 1])
        if np.std(truth_mass) > 0.0 and np.std(predicted_mean) > 0.0
        else None
    )
    row_totals = response_counts.sum(axis=1, keepdims=True)
    response_fraction = np.divide(
        response_counts,
        row_totals,
        out=np.zeros_like(response_counts, dtype=np.float64),
        where=row_totals > 0,
    )
    return {
        "status": "passed",
        "events": len(prediction_ids),
        "truth_support": int(support.all(axis=1).sum()),
        "observable": "mttbar invariant mass derived from top + antitop four-vectors",
        "truth_bin_labels": labels,
        "predicted_bin_labels": labels,
        "bias_gev": float(residual.mean()),
        "resolution_gev": float(residual.std()),
        "rmse_gev": float(np.sqrt(np.square(residual).mean())),
        "relative_bias": float((residual / truth_mass).mean()),
        "pearson_correlation": correlation,
        "threshold_window": threshold_window,
        "truth_binned": bins,
        "response_matrix_counts_truth_rows_predicted_columns": response_counts.tolist(),
        "response_matrix_row_fractions": response_fraction.tolist(),
        "prediction_summary": {
            "truth_mean_gev": float(truth_mass.mean()),
            "predicted_posterior_mean_gev": float(predicted_mean.mean()),
            "posterior_sample_count": int(pair_samples.shape[1]),
        },
    }


def parent_role_kinematic_metrics(prediction_path: Path, truth_path: Path) -> dict:
    from joint_event_reconstruction.physics_observables import (
        four_vector_observables,
        wrapped_delta,
    )

    with h5py.File(prediction_path, "r") as prediction:
        prediction_ids = np.asarray(prediction["EVENT_ID"][:], dtype=np.int64)
        parent_samples = np.asarray(prediction["PARTONIC/parents"][:], dtype=np.float64)
    with h5py.File(truth_path, "r") as source:
        truth_ids = np.asarray(source["EVENT_ID"][:], dtype=np.int64)
        order = np.argsort(truth_ids, kind="stable")
        positions = np.searchsorted(truth_ids[order], prediction_ids)
        if np.any(positions >= len(order)):
            raise RuntimeError("Parent predictions contain event IDs absent from truth sidecar")
        rows = order[positions]
        if not np.array_equal(truth_ids[rows], prediction_ids):
            raise RuntimeError("Parent prediction and truth sidecar EVENT_ID sets differ")
        support = np.asarray(source["PARTONIC/SUPPORT"][rows], dtype=bool)
        truth = np.stack(
            [
                np.asarray(source[f"PARTONIC/{role}"][rows], dtype=np.float64)
                for role in PARENT_ROLES
            ],
            axis=1,
        )
    if parent_samples.ndim != 4 or parent_samples.shape[2:] != (2, 4):
        raise RuntimeError("Parent validation samples must have shape [N,S,2,4]")
    if support.shape != (len(prediction_ids), 2) or not np.isfinite(truth).all():
        raise RuntimeError("Parent validation truth support or values are malformed")

    result = {}
    for role_index, role in enumerate(PARENT_ROLES):
        selected = support[:, role_index]
        if not selected.any():
            raise RuntimeError(f"Parent validation has no supported truth for {role}")
        samples = parent_samples[selected, :, role_index, :]
        targets = truth[selected, role_index, :]
        predicted_observables = four_vector_observables(samples)
        truth_observables = four_vector_observables(targets)
        sample_values = {
            "E": samples[..., 0],
            "px": samples[..., 1],
            "py": samples[..., 2],
            "pz": samples[..., 3],
            "pT": predicted_observables["pt"],
            "eta": predicted_observables["eta"],
            "y": predicted_observables["rapidity"],
            "phi": predicted_observables["phi"],
            "mass": predicted_observables["mass"],
        }
        truth_values = {
            "E": targets[..., 0],
            "px": targets[..., 1],
            "py": targets[..., 2],
            "pz": targets[..., 3],
            "pT": truth_observables["pt"],
            "eta": truth_observables["eta"],
            "y": truth_observables["rapidity"],
            "phi": truth_observables["phi"],
            "mass": truth_observables["mass"],
        }
        role_result = {"events": len(prediction_ids), "supported_events": int(selected.sum())}
        for observable, values in sample_values.items():
            target = truth_values[observable]
            valid = np.isfinite(target) & np.isfinite(values).all(axis=1)
            if not valid.any():
                role_result[observable] = {
                    "metric_events": 0,
                    "undefined_supported_events": int(selected.sum()),
                }
                continue
            metric_values = (
                wrapped_delta(values[valid], target[valid, None])
                if observable == "phi"
                else values[valid]
            )
            metric_truth = (
                np.zeros(int(valid.sum()), dtype=np.float64)
                if observable == "phi"
                else target[valid]
            )
            role_result[observable] = {
                "metric_events": int(valid.sum()),
                "undefined_supported_events": int(selected.sum() - valid.sum()),
                "units": "GeV" if observable in {"E", "px", "py", "pz", "pT", "mass"} else "dimensionless",
                **_posterior_scalar_metrics(metric_values, metric_truth),
            }
        result[role] = role_result
    return {
        "status": "passed",
        "events": len(prediction_ids),
        "roles": PARENT_ROLES,
        "four_vector_order": "E,px,py,pz",
        "per_role": result,
    }


def _make_parent_prediction(
    mode: str, config, checkpoint: Path, output: Path, provenance: dict
) -> dict:
    import lightning as L
    import torch

    from joint_event_reconstruction.cli.run import (
        build_data,
        load_trained,
        load_visible_config,
    )
    from joint_event_reconstruction.prediction_partonic import (
        _validate_prediction_file,
        evaluate_partonic_hdf5,
        predict_partonic_hdf5,
    )

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    seed = int(config.inference.seed)
    L.seed_everything(seed, workers=True)
    visible = load_visible_config(config)
    data = build_data(config, visible, include_truth=False, device=device)
    data.setup("predict")
    if data.structured_collator.invisible_store is not None or data.structured_collator.partonic_store is not None:
        raise RuntimeError(f"{mode} parent prediction loaded truth into its prediction data module")
    module = load_trained(config, visible, data, str(checkpoint))
    prediction_path = output / "parent_validation.h5"
    samples = int(config.inference.posterior_samples)
    steps = int(config.model.parent_cfm.sampling_steps)
    solver = str(config.model.parent_cfm.sampling_solver)
    predict_partonic_hdf5(
        module,
        data,
        prediction_path,
        split="validation",
        samples=samples,
        steps=steps,
        solver=solver,
        seed=seed,
        device=device,
        checkpoint_path=checkpoint,
    )
    _validate_prediction_file(prediction_path, expected_events=EXPECTED_EVENT_COUNT, samples=samples)
    with h5py.File(prediction_path, "r") as prediction:
        if (
            json.loads(str(prediction.attrs.get("parent_roles", "[]"))) != PARENT_ROLES
            or str(prediction.attrs.get("conditioning", "")) != EXPECTED_CONDITIONING[mode]
            or not bool(prediction.attrs.get("truth_free_input", False))
            or Path(str(prediction.attrs.get("input_path", ""))).resolve() != MEASURED_PATH.resolve()
            or str(prediction.attrs.get("split", "")) != "validation"
        ):
            raise RuntimeError(f"{mode} parent prediction does not satisfy its truth-free mode contract")
        if mode == "J_AN":
            roles = json.loads(str(prediction.attrs.get("invisible_role_order", "[]")))
            if roles != ROLE_ORDER:
                raise RuntimeError("J_AN P2 context is not ordered [top.nu, antitop.nu]")
    stamp_prediction_provenance(prediction_path, provenance)
    ids, id_record = _read_prediction_ids(prediction_path, EXPECTED_EVENT_COUNT)
    evaluation_path = output / "parent_evaluation.json"
    evaluation = evaluate_partonic_hdf5(prediction_path, TRUTH_PATH, evaluation_path)
    if (
        int(evaluation.get("event_count", -1)) != EXPECTED_EVENT_COUNT
        or int(evaluation.get("supported_event_count", -1)) != EXPECTED_EVENT_COUNT
    ):
        raise RuntimeError(f"{mode} parent evaluation did not close over validation truth support")
    evaluation["mttbar_thresholds"] = mttbar_threshold_metrics(
        prediction_path, TRUTH_PATH
    )
    evaluation["parent_role_kinematics"] = parent_role_kinematic_metrics(
        prediction_path, TRUTH_PATH
    )
    evaluation["provenance"] = provenance
    atomic_json(evaluation_path, evaluation)
    result = {
        "prediction": str(prediction_path.resolve()),
        "evaluation": str(evaluation_path.resolve()),
        "roles": PARENT_ROLES,
        "conditioning": EXPECTED_CONDITIONING[mode],
        "truth_free_input": True,
        "samples_per_event": samples,
        "solver": solver,
        "steps": steps,
        "mttbar_thresholds": evaluation["mttbar_thresholds"],
        **id_record,
    }
    del module, data, visible, ids
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_validation(mode: str, expected_sha: str, manifest_path: Path) -> dict:
    assert_source(expected_sha)
    manifest = load_run_manifest(manifest_path, expected_sha)
    entry = _mode_entry(manifest, mode)
    training_path = Path(entry["training_output"])
    training_record = _read_json(training_path / "production_training.json")
    if training_record.get("status") != "passed" or training_record.get("source_commit") != expected_sha:
        raise RuntimeError(f"{mode} has no accepted full-production training checkpoint")
    checkpoint = Path(training_record["checkpoint"])
    if not checkpoint.is_file() or sha256(checkpoint) != training_record.get("checkpoint_sha256"):
        raise RuntimeError(f"{mode} production checkpoint is missing or changed")
    validation_root = Path(entry["validation_output"])
    if validation_root.exists():
        raise FileExistsError(f"Refusing to reuse {mode} validation output: {validation_root}")
    validation_root.mkdir(parents=True, exist_ok=True)
    config, _, _ = load_and_validate_config(mode, Path(entry["config_path"]), training_path)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"{mode} production validation has no allocated CUDA device")
    provenance = artifact_provenance(
        mode,
        config,
        manifest,
        gpu_type=torch.cuda.get_device_name(0),
        training_record=training_record,
    )
    result = {
        "status": "running",
        "g0_acceptance_sha256": manifest["g0_acceptance_sha256"],
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": training_record["checkpoint_sha256"],
        "validation_count": EXPECTED_EVENT_COUNT,
        "reserved_test_rows_read": 0,
        "started_utc": datetime.now(UTC).isoformat(),
        **provenance,
    }
    atomic_json(validation_root / "progress.json", result)
    try:
        standard = None
        parent = None
        if mode in STANDARD_MODES:
            standard_root = validation_root / "standard"
            standard_root.mkdir()
            standard = _make_standard_prediction(
                mode, config, checkpoint, standard_root, provenance
            )
        if mode in PARENT_MODES:
            parent_root = validation_root / "parent"
            parent_root.mkdir()
            parent = _make_parent_prediction(
                mode, config, checkpoint, parent_root, provenance
            )
        if (
            standard is not None
            and parent is not None
            and standard["event_id_sha256"] != parent["event_id_sha256"]
        ):
            raise RuntimeError(f"{mode} standard and parent predictions use different validation events")
        for label, prediction in (("standard", standard), ("parent", parent)):
            if prediction is not None and (
                prediction["sorted_event_id_sha256"]
                != manifest["inputs"]["validation_sorted_event_id_sha256"]
            ):
                raise RuntimeError(
                    f"{mode} {label} prediction does not match the frozen validation EVENT_ID set"
                )
        result.update(
            {
                "status": "passed",
                "standard_validation": standard,
                "parent_validation": parent,
                "completed_utc": datetime.now(UTC).isoformat(),
            }
        )
        atomic_json(validation_root / "validation.json", result)
        atomic_json(validation_root / "progress.json", result)
        return result
    except Exception as error:
        result.update(
            {
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "failed_utc": datetime.now(UTC).isoformat(),
            }
        )
        atomic_json(validation_root / "failure.json", result)
        atomic_json(validation_root / "progress.json", result)
        raise


def run_aggregate(expected_sha: str, manifest_path: Path) -> dict:
    assert_source(expected_sha)
    manifest = load_run_manifest(manifest_path, expected_sha)
    validations = {}
    validation_statuses = {}
    prediction_ids = {}
    failures = []
    for mode, entry in manifest["modes"].items():
        training_path = Path(entry["training_output"])
        validation_path = Path(entry["validation_output"]) / "validation.json"
        validation_failure_path = ATTEMPT_ROOT / f"failure_validate_{mode}.json"
        try:
            training = _read_json(training_path / "production_training.json")
            validation = _read_json(validation_path)
            if training.get("status") != "passed" or validation.get("status") != "passed":
                raise RuntimeError("training or validation status is not passed")
            if training.get("source_commit") != expected_sha or validation.get("source_commit") != expected_sha:
                raise RuntimeError("training/validation source SHA differs from the submitted release")
            if training.get("reserved_test_rows_read") != 0 or validation.get("reserved_test_rows_read") != 0:
                raise RuntimeError("training/validation reports reserved-test reads")
            if validation.get("validation_count") != EXPECTED_EVENT_COUNT:
                raise RuntimeError("validation event count differs from the frozen split")
            standard_metrics = None
            parent_metrics = None
            if mode in STANDARD_MODES:
                standard = validation.get("standard_validation") or {}
                if (
                    standard.get("roles") != ROLE_ORDER
                    or not standard.get("truth_free_input")
                    or not standard.get("truth_evaluated_after_prediction")
                ):
                    raise RuntimeError(
                        "standard prediction did not keep [top.nu, antitop.nu] inference truth-free"
                    )
                standard_metrics = _read_json(Path(standard["evaluation"]))
                if (
                    standard_metrics.get("status") != "passed"
                    or standard_metrics.get("prediction_truth_free") is not True
                    or standard_metrics.get("events") != EXPECTED_EVENT_COUNT
                    or standard_metrics.get("neutrino", {}).get("role_order") != ROLE_ORDER
                    or standard_metrics.get("assignment", {}).get("supported_events", 0) <= 0
                    or standard_metrics.get("neutrino", {}).get("both_roles_supported", 0) <= 0
                ):
                    raise RuntimeError("standard truth-side validation metrics failed their contract")
                prediction_ids[f"{mode}:standard"] = standard
            if mode in PARENT_MODES:
                parent = validation.get("parent_validation") or {}
                if parent.get("roles") != PARENT_ROLES or not parent.get("truth_free_input"):
                    raise RuntimeError("parent prediction has no truth-free [top, antitop] contract")
                parent_metrics = _read_json(Path(parent["evaluation"]))
                threshold_metrics = parent_metrics.get("mttbar_thresholds", {})
                role_metrics = parent_metrics.get("parent_role_kinematics", {})
                if (
                    parent_metrics.get("event_count") != EXPECTED_EVENT_COUNT
                    or parent_metrics.get("supported_event_count") != EXPECTED_EVENT_COUNT
                    or threshold_metrics.get("status") != "passed"
                    or threshold_metrics.get("events") != EXPECTED_EVENT_COUNT
                    or role_metrics.get("status") != "passed"
                    or role_metrics.get("roles") != PARENT_ROLES
                ):
                    raise RuntimeError("parent role or mttbar threshold validation did not pass")
                prediction_ids[f"{mode}:parent"] = parent
            validations[mode] = {
                "training": str((training_path / "production_training.json").resolve()),
                "training_checkpoint": training["checkpoint"],
                "training_checkpoint_sha256": training["checkpoint_sha256"],
                "validation": str(validation_path.resolve()),
                "standard_validation": validation.get("standard_validation"),
                "standard_metrics": standard_metrics,
                "parent_validation": validation.get("parent_validation"),
                "parent_metrics": parent_metrics,
            }
            validation_statuses[mode] = "PASS"
        except (OSError, json.JSONDecodeError, RuntimeError, KeyError, TypeError, ValueError, AttributeError) as error:
            failures.append(f"{mode}: {type(error).__name__}: {error}")
            if validation_failure_path.is_file() or validation_path.is_file():
                validation_statuses[mode] = "FAILED"
            else:
                validation_statuses[mode] = "MISSING"
    observed_ids = {
        record.get("event_id_sha256") for record in prediction_ids.values()
    }
    observed_sorted_ids = {
        record.get("sorted_event_id_sha256") for record in prediction_ids.values()
    }
    if len(prediction_ids) != len(STANDARD_MODES) + len(PARENT_MODES):
        failures.append("validation artifact count is not five standard plus four parent predictions")
    if any(
        record.get("event_count") != EXPECTED_EVENT_COUNT
        or not record.get("event_id_sha256")
        or not record.get("sorted_event_id_sha256")
        for record in prediction_ids.values()
    ):
        failures.append("one or more validation prediction event identity records are incomplete")
    if len(observed_ids) != 1 or len(observed_sorted_ids) != 1:
        failures.append("validation prediction event IDs do not match across all model modes")
    if (
        len(observed_sorted_ids) == 1
        and next(iter(observed_sorted_ids))
        != manifest["inputs"]["validation_sorted_event_id_sha256"]
    ):
        failures.append("validation prediction IDs differ from the frozen measured validation split")
    comparisons = {}
    def delta(second, first):
        if second is None or first is None:
            return None
        return float(second - first)

    if "Q0" in validations and "Q1" in validations:
        q0 = validations["Q0"].get("standard_metrics") or {}
        q1 = validations["Q1"].get("standard_metrics") or {}
        q0_assignment = q0.get("assignment", {})
        q1_assignment = q1.get("assignment", {})
        comparisons["Q0_vs_Q1"] = {
            "interpretation": "Q1 minus Q0; query effect is evaluated separately for assignment and neutrino observables",
            "complete_event_top1_accuracy_delta": delta(
                q1_assignment.get("top1_complete_event_accuracy"),
                q0_assignment.get("top1_complete_event_accuracy"),
            ),
            "per_role_assignment_top1_accuracy_delta": {
                role: delta(
                    q1_assignment.get("per_role", {}).get(role, {}).get("top1_accuracy"),
                    q0_assignment.get("per_role", {}).get(role, {}).get("top1_accuracy"),
                )
                for role in PARENT_ROLES
            },
            "neutrino_posterior_mean_rmse_delta": {
                role: {
                    observable: delta(
                        q1.get("neutrino", {}).get("per_role", {}).get(role, {}).get(observable, {}).get("posterior_mean_rmse"),
                        q0.get("neutrino", {}).get("per_role", {}).get(role, {}).get(observable, {}).get("posterior_mean_rmse"),
                    )
                    for observable in ("px", "py", "pz", "pT", "wrapped_phi_residual")
                }
                for role in ROLE_ORDER
            },
        }
    for baseline, alternative in (("PCFM_Q", "J0"), ("J0", "J_A"), ("J_A", "J_AN")):
        if baseline in validations and alternative in validations:
            first = (validations[baseline].get("parent_metrics") or {}).get("mttbar_thresholds", {})
            second = (validations[alternative].get("parent_metrics") or {}).get("mttbar_thresholds", {})
            first_roles = (validations[baseline].get("parent_metrics") or {}).get(
                "parent_role_kinematics", {}
            ).get("per_role", {})
            second_roles = (validations[alternative].get("parent_metrics") or {}).get(
                "parent_role_kinematics", {}
            ).get("per_role", {})
            comparisons[f"{baseline}_vs_{alternative}"] = {
                "interpretation": "separate parent-study comparison; no global model ranking",
                "mttbar_bias_delta_gev": delta(second.get("bias_gev"), first.get("bias_gev")),
                "mttbar_rmse_delta_gev": delta(second.get("rmse_gev"), first.get("rmse_gev")),
                "threshold_window_bias_delta_gev": delta(
                    second.get("threshold_window", {}).get("bias_gev"),
                    first.get("threshold_window", {}).get("bias_gev"),
                ),
                "parent_role_posterior_mean_rmse_delta": {
                    role: {
                        observable: delta(
                            second_roles.get(role, {}).get(observable, {}).get("posterior_mean_rmse"),
                            first_roles.get(role, {}).get(observable, {}).get("posterior_mean_rmse"),
                        )
                        for observable in ("E", "px", "py", "pz", "pT", "eta", "y", "phi", "mass")
                    }
                    for role in PARENT_ROLES
                },
            }
    result = {
        "status": "passed" if not failures else "failed",
        "campaign": manifest["campaign"],
        "requirement": "R_VALIDATION",
        "source_commit": expected_sha,
        "g0_source_commit": manifest["g0_source_commit"],
        "g0_acceptance_path": manifest["g0_acceptance_path"],
        "g0_acceptance_sha256": manifest["g0_acceptance_sha256"],
        "mode_count": len(validations),
        "standard_prediction_count": len(STANDARD_MODES),
        "parent_prediction_count": len(PARENT_MODES),
        "validation_event_count": EXPECTED_EVENT_COUNT,
        "validation_event_id_sha256": next(iter(observed_ids)) if len(observed_ids) == 1 else None,
        "validation_sorted_event_id_sha256": next(iter(observed_sorted_ids)) if len(observed_sorted_ids) == 1 else None,
        "invisible_roles": ROLE_ORDER,
        "parent_roles": PARENT_ROLES,
        "reserved_test_rows_read": 0,
        "validations": validations,
        "validation_jobs": {
            mode: {
                "job_id": manifest.get("job_ids", {}).get("validation", {}).get(mode),
                "status": validation_statuses.get(mode, "MISSING"),
                "artifact": str(
                    (Path(entry["validation_output"]) / "validation.json").resolve()
                ),
            }
            for mode, entry in manifest["modes"].items()
        },
        "comparisons": comparisons,
        "failures": failures,
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    output_path = PRODUCTION_ROOT / "R_VALIDATION.json"
    if output_path.exists() or (PRODUCTION_ROOT / "R_VALIDATION.md").exists():
        raise FileExistsError(f"Refusing to overwrite existing R_VALIDATION output under {PRODUCTION_ROOT}")
    atomic_json(output_path, result)
    lines = [
        "# R_VALIDATION: real top/antitop production",
        "",
        f"**Status:** {result['status']}",
        f"**Source commit:** `{expected_sha}`",
        f"**Validation events:** {EXPECTED_EVENT_COUNT}",
        f"**Neutrino roles:** `{ROLE_ORDER}`",
        f"**Parent roles:** `{PARENT_ROLES}`",
        "**Reserved test rows read:** 0",
        "",
        "| Mode | Training checkpoint SHA-256 | Standard validation | Parent validation |",
        "|---|---|---|---|",
    ]
    for mode in MODE_CONFIGS:
        item = validations.get(mode, {})
        standard_status = "passed" if item.get("standard_validation") else "not applicable"
        parent_status = "passed" if item.get("parent_validation") else "not applicable"
        lines.append(
            f"| {mode} ({validation_statuses.get(mode, 'MISSING')}) | `{item.get('training_checkpoint_sha256', 'missing')}` | "
            f"{standard_status} | {parent_status} |"
        )
    lines.extend(
        [
            "",
            "## Standard validation",
            "",
            "Neutrino predictions were written from measured inputs only. The evaluation pass loaded truth after prediction and joined by the frozen validation EVENT_ID order.",
            "",
            "| Mode | Assignment support | Complete-event top-1 | top.nu pz RMSE (GeV) | antitop.nu pz RMSE (GeV) | MET residual RMSE (GeV) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )

    def display(value):
        return "n/a" if value is None else f"{float(value):.5g}"

    for mode in STANDARD_MODES:
        metrics = validations.get(mode, {}).get("standard_metrics") or {}
        neutrino = metrics.get("neutrino", {})
        roles = neutrino.get("per_role", {})
        closure = neutrino.get("met_closure", {})
        lines.append(
            f"| {mode} | {metrics.get('assignment', {}).get('supported_events', 'n/a')} | "
            f"{display(metrics.get('assignment', {}).get('top1_complete_event_accuracy'))} | "
            f"{display(roles.get('top.nu', {}).get('pz', {}).get('posterior_mean_rmse'))} | "
            f"{display(roles.get('antitop.nu', {}).get('pz', {}).get('posterior_mean_rmse'))} | "
            f"{display(closure.get('transverse_residual_magnitude', {}).get('rmse'))} |"
        )
    lines.extend(
        [
            "",
            "Per-role assignment accuracy, null fraction, njet and channel bins, px/py/pz/pT/wrapped-phi residuals, posterior interval coverage, and MET closure are recorded under each mode's `standard_metrics` in the machine-readable report.",
            "",
            "### Q0 versus Q1",
            "",
            "| Quantity | Q1 minus Q0 |",
            "|---|---:|",
        ]
    )
    q_comparison = result.get("comparisons", {}).get("Q0_vs_Q1", {})
    lines.append(
        f"| Complete-event top-1 assignment accuracy | {display(q_comparison.get('complete_event_top1_accuracy_delta'))} |"
    )
    for role in PARENT_ROLES:
        lines.append(
            f"| {role} top-1 assignment accuracy | {display(q_comparison.get('per_role_assignment_top1_accuracy_delta', {}).get(role))} |"
        )
    for role in ROLE_ORDER:
        for observable in ("px", "py", "pz", "pT", "wrapped_phi_residual"):
            lines.append(
                f"| {role} {observable} posterior-mean RMSE | {display(q_comparison.get('neutrino_posterior_mean_rmse_delta', {}).get(role, {}).get(observable))} |"
            )
    lines.extend(
        [
            "",
            "## Parent and mttbar validation",
            "",
            "The response matrix and truth-binned residuals use mttbar derived from each sampled top + antitop pair. The dedicated 345–400 GeV window covers the nominal threshold through 55 GeV above it.",
            "",
            "| Mode | mttbar bias (GeV) | mttbar resolution (GeV) | mttbar RMSE (GeV) | 345–400 GeV events | threshold-window bias (GeV) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mode in PARENT_MODES:
        parent = validations.get(mode, {}).get("parent_metrics") or {}
        threshold = parent.get("mttbar_thresholds", {})
        window = threshold.get("threshold_window", {})
        lines.append(
            f"| {mode} | {display(threshold.get('bias_gev'))} | "
            f"{display(threshold.get('resolution_gev'))} | "
            f"{display(threshold.get('rmse_gev'))} | {window.get('events', 'n/a')} | "
            f"{display(window.get('bias_gev'))} |"
        )
    lines.extend(
        [
            "",
            "Full top/antitop four-vector metrics, pair observables, truth-binned mttbar bias and resolution, response matrices, and threshold-window details are in each mode's `parent_metrics`.",
            "",
            "### Parent comparisons",
            "",
            "| Comparison | mttbar RMSE change (GeV) | threshold-window bias change (GeV) |",
            "|---|---:|---:|",
        ]
    )
    for name in ("PCFM_Q_vs_J0", "J0_vs_J_A", "J_A_vs_J_AN"):
        comparison = result.get("comparisons", {}).get(name, {})
        lines.append(
            f"| {name.replace('_vs_', ' ↔ ')} | "
            f"{display(comparison.get('mttbar_rmse_delta_gev'))} | "
            f"{display(comparison.get('threshold_window_bias_delta_gev'))} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.append("")
    md_path = PRODUCTION_ROOT / "R_VALIDATION.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = md_path.with_suffix(md_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(md_path)
    if failures:
        raise RuntimeError(f"R_VALIDATION failed; see {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("train", "validate", "aggregate"), required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODE_CONFIGS))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    expected_sha = args.source_sha
    manifest_path = args.run_manifest.expanduser().resolve()
    python_bin = args.python.expanduser().resolve()
    if not python_bin.is_file():
        raise FileNotFoundError(f"Production Python interpreter is missing: {python_bin}")
    if args.stage in {"train", "validate"} and args.mode is None:
        parser.error("--mode is required for train and validate stages")
    try:
        if args.stage == "train":
            if not os.environ.get("SLURM_JOB_ID"):
                raise RuntimeError("Full realnu training must run in its allocated Slurm GPU job")
            result = run_training(args.mode, expected_sha, manifest_path, python_bin)
        elif args.stage == "validate":
            if not os.environ.get("SLURM_JOB_ID"):
                raise RuntimeError("Full realnu validation must run in its allocated Slurm GPU job")
            result = run_validation(args.mode, expected_sha, manifest_path)
        else:
            if not os.environ.get("SLURM_JOB_ID"):
                raise RuntimeError("R_VALIDATION must run in its dependent Slurm job")
            result = run_aggregate(expected_sha, manifest_path)
    except Exception as error:
        if ATTEMPT_ROOT.is_dir():
            progress_name = f"train_{args.mode}.json" if args.stage == "train" else None
            if progress_name is not None:
                progress_path = ATTEMPT_ROOT / progress_name
                if progress_path.is_file():
                    progress = _read_json(progress_path)
                    progress.update(
                        {
                            "status": "failed",
                            "error": f"{type(error).__name__}: {error}",
                            "failed_utc": datetime.now(UTC).isoformat(),
                        }
                    )
                    atomic_json(progress_path, progress)
            failure_name = f"failure_{args.stage}_{args.mode}.json" if args.mode else "failure_R_VALIDATION.json"
            failure_path = ATTEMPT_ROOT / failure_name
            if not failure_path.is_file():
                atomic_json(
                    failure_path,
                    {
                        "status": "failed",
                        "campaign": "realnu_top_antitop_v1",
                        "stage": args.stage,
                        "mode": args.mode,
                        "source_commit": expected_sha,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "error": f"{type(error).__name__}: {error}",
                        "failed_utc": datetime.now(UTC).isoformat(),
                    },
                )
        raise
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
