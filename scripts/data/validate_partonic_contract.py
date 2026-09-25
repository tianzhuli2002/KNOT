"""Validate measured/target contracts for the charge-side parent-CFM study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.configuration import load_experiment
from joint_event_reconstruction.runtime_paths import runtime_paths
from joint_event_reconstruction.topology.compiler import load_topology


ROLES = ("top", "antitop")
TRUTH_DEFINITION = "partonic_after_fsr"
TARGET_REPRESENTATION = "partonic_px_py_pz_log1p_mass_v1"
SPLIT_POLICY = "event_number_splitmix64_90_5_5_v1"
CONTRACT_KEYS = (
    "source_manifest_sha256",
    "split_membership_sha256",
    "split_version",
    "max_jets",
    "adapter_version",
    "source_path",
)
TARGET_CONTRACT_KEYS = (
    "source_manifest_sha256",
    "split_membership_sha256",
    "split_version",
    "max_jets",
    "source_path",
)


def _attr_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_attr(value, name, path):
    try:
        return json.loads(_attr_text(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid JSON attribute {name}") from error


def _splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = values.astype(np.uint64, copy=False) ^ np.uint64(seed)
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def validate_experiment(
    experiment_path: str | Path,
    checkpoint_path: str | None = None,
    *,
    allow_reserved_test: bool = False,
    overrides: list[str] | None = None,
) -> dict:
    config = load_experiment(experiment_path, overrides or [])
    if not bool(config.model.parent_cfm.enabled):
        raise ValueError("Partonic contract validation requires model.parent_cfm.enabled=true")
    expected_bindings = {
        str(key): str(value)
        for key, value in config.model.parent_cfm.truth_bindings.items()
    }
    if list(config.model.parent_cfm.roles) != list(ROLES):
        raise ValueError(f"Parent-CFM roles must be ordered {ROLES}")
    topology = load_topology(
        config.topology_config,
        {
            str(key): value
            for key, value in config.data.truth_bindings.items()
        },
    )
    if set(topology.roots) != set(ROLES):
        raise ValueError(
            f"Charge-side topology roots must be exactly {ROLES}, found {topology.roots}"
        )
    query_roles = {
        str(role.name): role
        for role in config.model.topology_queries.roles
    }
    for role_name in ROLES:
        role = query_roles.get(role_name)
        if role is None or str(role.node) != role_name or str(role.kind) != "parent":
            raise ValueError(
                f"Shared topology query for {role_name!r} must bind to its compiled parent node"
            )

    measured_path = Path(config.data.input).expanduser().resolve()
    target_path = Path(config.model.parent_cfm.target_path).expanduser().resolve()
    if not measured_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(
            f"Measured input or parent-truth sidecar is missing: {measured_path}, {target_path}"
        )
    experiment_root = runtime_paths().project_root / "results/experiments/partonic_ttbar2L_2M_v1"
    manifest_path = experiment_root / "source_manifest_v1.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen source manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    split_info = manifest.get("split", {})
    if split_info.get("version") != SPLIT_POLICY:
        raise ValueError(
            f"Source manifest split version is not {SPLIT_POLICY!r}: {split_info.get('version')!r}"
        )
    split_seed = int(split_info.get("seed", -1))
    if split_seed < 0:
        raise ValueError(f"Source manifest has an invalid split seed: {split_seed}")

    with h5py.File(measured_path, "r") as measured:
        for name in (
            "INPUTS/JET",
            "INPUTS/LEPTON",
            "INPUTS/MET",
            "INPUTS/GLOBAL",
            "EVENT_ID",
            "SPLIT",
            "META/eventNumber",
            "META/source_entry_index",
        ):
            if name not in measured:
                raise KeyError(f"Measured input {measured_path} is missing {name}")
        event_count = len(measured["EVENT_ID"])
        for name in (
            "INPUTS/JET",
            "INPUTS/LEPTON",
            "INPUTS/MET",
            "INPUTS/GLOBAL",
            "SPLIT",
            "META/eventNumber",
            "META/source_entry_index",
        ):
            if len(measured[name]) != event_count:
                raise ValueError(
                    f"Measured event-axis mismatch: {measured_path}:{name} "
                    f"has {len(measured[name])}, EVENT_ID has {event_count}"
                )
        jet_width = int(_attr_text(measured.attrs.get("max_jets", "0")))
        for name, shape in (
            ("INPUTS/JET", (event_count, jet_width)),
            ("INPUTS/LEPTON", (event_count, 2)),
            ("INPUTS/MET", (event_count, 1)),
            ("INPUTS/GLOBAL", (event_count,)),
        ):
            dataset = measured[name]
            if dataset.shape != shape or dataset.dtype.names is None:
                raise TypeError(
                    f"{measured_path}:{name} must be a structured array with shape {shape}; "
                    f"found shape={dataset.shape}, dtype={dataset.dtype}"
                )
        for name in ("JET", "LEPTON", "MET"):
            fields = set(measured[f"INPUTS/{name}"].dtype.names or ())
            required_fields = {"e", "eta", "phi", "pt"}
            if not required_fields <= fields:
                raise ValueError(
                    f"{measured_path}:INPUTS/{name} misses measured fields "
                    f"{sorted(required_fields - fields)}"
                )
        lepton_fields = set(measured["INPUTS/LEPTON"].dtype.names or ())
        if not {"charge", "id", "source_type", "source_index"} <= lepton_fields:
            raise ValueError(
                f"{measured_path}:INPUTS/LEPTON lacks charge/type/source-index identity fields"
            )
        if measured["EVENT_ID"].ndim != 1 or not np.issubdtype(
            measured["EVENT_ID"].dtype, np.integer
        ):
            raise TypeError(f"{measured_path}:EVENT_ID must be a one-dimensional integer array")
        if measured["SPLIT"].ndim != 1 or not np.issubdtype(
            measured["SPLIT"].dtype, np.integer
        ):
            raise TypeError(f"{measured_path}:SPLIT must be a one-dimensional integer array")
        if "LABELS" in measured or "TARGETS" in measured:
            raise ValueError(
                f"Inference-input file must contain measured data only: {measured_path}"
            )
        if not bool(measured.attrs.get("measured_inputs_only", False)):
            raise ValueError(f"{measured_path} does not declare measured_inputs_only=true")
        measured_contract = {
            key: _attr_text(measured.attrs.get(key, "")) for key in CONTRACT_KEYS
        }
        if any(not value for value in measured_contract.values()):
            raise ValueError(f"{measured_path} is missing source/split contract attributes")
        if measured_contract["source_manifest_sha256"] != manifest_sha:
            raise ValueError(
                f"{measured_path} source manifest checksum differs from {manifest_path}"
            )
        if measured_contract["split_version"] != SPLIT_POLICY:
            raise ValueError(
                f"{measured_path} has an incompatible split version: "
                f"{measured_contract['split_version']!r}"
            )
        split_values, split_counts = np.unique(measured["SPLIT"][:], return_counts=True)
        split_counts = {str(int(key)): int(value) for key, value in zip(split_values, split_counts, strict=True)}
        expected_splits = (2,) if allow_reserved_test else (0, 1)
        if any(int(value) not in expected_splits for value in split_values):
            raise ValueError(
                f"Input has split values outside the allowed population "
                f"{expected_splits}: {split_counts}"
            )
        if not np.array_equal(
            np.asarray(measured["EVENT_ID"][:], dtype=np.int64),
            np.asarray(measured["META/eventNumber"][:], dtype=np.int64),
        ):
            raise ValueError(f"{measured_path}: EVENT_ID and META/eventNumber differ")
        if len(np.unique(measured["EVENT_ID"][:])) != event_count:
            raise ValueError(f"{measured_path}: EVENT_ID contains duplicates")
        if len(np.unique(measured["META/source_entry_index"][:])) != event_count:
            raise ValueError(f"{measured_path}: source_entry_index contains duplicates")

    target_support_counts = np.zeros(2, dtype=np.int64)
    objective_support_counts = {
        "visible_assignment": 0,
        "neutrino_cfm_joint": 0,
        "partonic_cfm_joint": 0,
        "visible_and_neutrino": 0,
        "visible_and_partonic": 0,
        "neutrino_and_partonic": 0,
        "all_three": 0,
    }
    invisible_role_support_counts = np.zeros(2, dtype=np.int64)
    with h5py.File(target_path, "r") as target:
        if "EVENT_ID" not in target or "PARTONIC" not in target:
            raise KeyError(f"Partonic truth sidecar {target_path} needs EVENT_ID and PARTONIC")
        target_count = len(target["EVENT_ID"])
        if target_count != event_count:
            raise ValueError(
                f"Measured/partonic event counts differ: {event_count} vs {target_count}"
            )
        roles = _json_attr(target.attrs.get("partonic_roles"), "partonic_roles", target_path)
        bindings = _json_attr(
            target.attrs.get("partonic_truth_bindings"),
            "partonic_truth_bindings",
            target_path,
        )
        if roles != list(ROLES) or bindings != expected_bindings:
            raise ValueError(
                "Partonic target role/binding mismatch: "
                f"roles={roles!r}, bindings={bindings!r}, expected={expected_bindings!r}"
            )
        if _attr_text(target.attrs.get("partonic_truth_definition", "")) != TRUTH_DEFINITION:
            raise ValueError(f"{target_path} is not genuine {TRUTH_DEFINITION} truth")
        if _attr_text(target.attrs.get("partonic_units", "")) != "GeV":
            raise ValueError(f"{target_path} target units must be GeV")
        if _attr_text(target.attrs.get("partonic_target_representation", "")) != TARGET_REPRESENTATION:
            raise ValueError(f"{target_path} has an incompatible partonic target representation")
        for key in TARGET_CONTRACT_KEYS:
            expected = measured_contract[key]
            observed = _attr_text(target.attrs.get(key, ""))
            if expected != observed:
                raise ValueError(
                    f"Measured/partonic source contract mismatch for {key}: "
                    f"{expected!r} vs {observed!r}"
                )
        if "PARTONIC/SUPPORT" not in target:
            raise KeyError(f"{target_path}: missing PARTONIC/SUPPORT")
        support_dataset = target["PARTONIC/SUPPORT"]
        if support_dataset.shape != (event_count, 2) or support_dataset.dtype != np.bool_:
            raise TypeError(
                f"{target_path}:PARTONIC/SUPPORT must be boolean [{event_count},2]"
            )
        if "SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE" not in target:
            raise KeyError(f"{target_path}: missing visible-assignment support")
        visible_support_dataset = target["SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"]
        if (
            visible_support_dataset.shape != (event_count,)
            or visible_support_dataset.dtype != np.bool_
        ):
            raise TypeError(
                f"{target_path}:SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE must be boolean [{event_count}]"
            )
        invisible_support_path = "SUPPORT/INVISIBLE_ROLE"
        if invisible_support_path not in target:
            raise KeyError(
                f"{target_path}: missing explicit unsupported/supported neutrino-role mask"
            )
        invisible_support_dataset = target[invisible_support_path]
        if (
            invisible_support_dataset.shape != (event_count, 2)
            or invisible_support_dataset.dtype != np.bool_
        ):
            raise TypeError(
                f"{target_path}:{invisible_support_path} must be boolean [{event_count},2]"
            )
        invisible_roles = _json_attr(
            invisible_support_dataset.attrs.get("role_order"),
            "SUPPORT/INVISIBLE_ROLE.role_order",
            target_path,
        )
        if invisible_roles != ["top.nu", "antitop.nu"]:
            raise ValueError(
                f"{target_path}: invisible support role order is incompatible: {invisible_roles}"
            )
        if not bool(manifest.get("bounded_sample", {}).get("neutrino_side_mapping_verified", False)):
            if np.asarray(invisible_support_dataset[:], dtype=bool).any():
                raise ValueError(
                    "Neutrino role support is enabled even though source/top-side mapping "
                    "remains unverified"
                )
        for role in ROLES:
            dataset = target[f"PARTONIC/{role}"]
            if dataset.shape != (event_count, 4) or dataset.dtype != np.float32:
                raise TypeError(
                    f"{target_path}:PARTONIC/{role} must be float32 [{event_count},4]"
                )
        chunk_size = 65536
        with h5py.File(measured_path, "r") as measured_for_join:
            for start in range(0, event_count, chunk_size):
                stop = min(event_count, start + chunk_size)
                event_ids = np.asarray(
                    measured_for_join["EVENT_ID"][start:stop], dtype=np.int64
                )
                target_ids = np.asarray(target["EVENT_ID"][start:stop], dtype=np.int64)
                if not np.array_equal(event_ids, target_ids):
                    raise ValueError(
                        f"Measured/partonic EVENT_ID join mismatch at rows {start}:{stop}"
                    )
                event_numbers = np.asarray(
                    measured_for_join["META/eventNumber"][start:stop], dtype=np.int64
                )
                bucket = _splitmix64(event_numbers, split_seed) % np.uint64(100)
                expected_split = np.where(
                    bucket < 90, 0, np.where(bucket < 95, 1, 2)
                ).astype(np.uint8)
                observed_split = np.asarray(measured_for_join["SPLIT"][start:stop], dtype=np.uint8)
                if not np.array_equal(expected_split, observed_split):
                    raise ValueError(
                        f"{measured_path}: event-number hash split mismatch at rows {start}:{stop}"
                    )
                values = np.stack(
                    [target[f"PARTONIC/{role}"][start:stop] for role in ROLES], axis=1
                )
                support = np.asarray(support_dataset[start:stop], dtype=bool)
                visible_support = np.asarray(
                    visible_support_dataset[start:stop], dtype=bool
                )
                invisible_support = np.asarray(
                    invisible_support_dataset[start:stop], dtype=bool
                )
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"{target_path}: non-finite partonic targets at {start}:{stop}"
                    )
                energy = values[..., 0]
                momentum = np.linalg.norm(values[..., 1:], axis=-1)
                if (
                    ((energy + 1.0e-4 < momentum) & support).any()
                    or ((energy <= 0.0) & support).any()
                ):
                    raise ValueError(
                        f"{target_path}: invalid supported parent p4 at {start}:{stop}"
                    )
                target_support_counts += support.sum(axis=0)
                invisible_role_support_counts += invisible_support.sum(axis=0)
                invisible_joint = invisible_support.all(axis=1)
                partonic_joint = support.all(axis=1)
                objective_support_counts["visible_assignment"] += int(visible_support.sum())
                objective_support_counts["neutrino_cfm_joint"] += int(invisible_joint.sum())
                objective_support_counts["partonic_cfm_joint"] += int(partonic_joint.sum())
                objective_support_counts["visible_and_neutrino"] += int((visible_support & invisible_joint).sum())
                objective_support_counts["visible_and_partonic"] += int((visible_support & partonic_joint).sum())
                objective_support_counts["neutrino_and_partonic"] += int((invisible_joint & partonic_joint).sum())
                objective_support_counts["all_three"] += int(
                    (visible_support & invisible_joint & partonic_joint).sum()
                )

    visible_target_path = config.data.visible_truth
    if visible_target_path is not None:
        visible_path = Path(visible_target_path).expanduser().resolve()
        with h5py.File(visible_path, "r") as visible:
            if len(visible["EVENT_ID"]) != event_count:
                raise ValueError(f"Visible assignment sidecar event count differs: {visible_path}")
            for key in TARGET_CONTRACT_KEYS:
                expected = measured_contract[key]
                if _attr_text(visible.attrs.get(key, "")) != expected:
                    raise ValueError(
                        f"Measured/visible source contract mismatch for {key}: {visible_path}"
                    )
            if "LABELS/JET" not in visible or "LABELS/LEPTON" not in visible:
                raise KeyError(f"Visible assignment sidecar is missing JET/LEPTON labels: {visible_path}")
            if "SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE" not in visible:
                raise KeyError(f"Visible assignment support is missing: {visible_path}")
            support = visible["SUPPORT/VISIBLE_ASSIGNMENT_RECONSTRUCTABLE"]
            if support.shape != (event_count,) or support.dtype != np.bool_:
                raise TypeError(
                    f"{visible_path}: visible reconstructability must be boolean [{event_count}]"
                )
            with h5py.File(measured_path, "r") as measured:
                for start in range(0, event_count, 65536):
                    stop = min(event_count, start + 65536)
                    if not np.array_equal(
                        measured["EVENT_ID"][start:stop], visible["EVENT_ID"][start:stop]
                    ):
                        raise ValueError(
                            f"Measured/visible EVENT_ID join mismatch at rows {start}:{stop}"
                        )
                    for source_name, width in (
                        ("JET", int(measured_contract["max_jets"])),
                        ("LEPTON", 2),
                    ):
                        labels = visible[f"LABELS/{source_name}"]
                        if labels.shape != (event_count, width) or labels.dtype != np.float32:
                            raise TypeError(
                                f"{visible_path}:LABELS/{source_name} must be float32 "
                                f"[{event_count},{width}]"
                            )
                        label_values = np.asarray(labels[start:stop])
                        if np.isinf(label_values).any() or not np.isin(
                            label_values[np.isfinite(label_values)], (0.0, 1.0, 2.0)
                        ).all():
                            raise ValueError(
                                f"{visible_path}:LABELS/{source_name} has labels outside "
                                "0/1/2 or NaN padding"
                            )

    result = {
        "status": "valid",
        "experiment": str(Path(experiment_path).expanduser().resolve()),
        "measured_input": str(measured_path),
        "partonic_truth": str(target_path),
        "visible_truth": None if visible_target_path is None else str(visible_target_path),
        "events": event_count,
        "split_counts": split_counts,
        "source_contract": measured_contract,
        "source_manifest_sha256": manifest_sha,
        "parent_roles": list(ROLES),
        "topology_signature": topology.signature,
        "truth_definition": TRUTH_DEFINITION,
        "support_counts": {role: int(target_support_counts[i]) for i, role in enumerate(ROLES)},
        "objective_support": {
            name: {"count": int(count), "fraction": float(count / max(event_count, 1))}
            for name, count in objective_support_counts.items()
        },
        "neutrino_role_support_counts": {
            name: int(invisible_role_support_counts[index])
            for index, name in enumerate(("top.nu", "antitop.nu"))
        },
        "parent_conditioning": str(config.model.parent_cfm.conditioning),
    }

    if checkpoint_path is not None:
        import torch

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hparams = saved.get("hyper_parameters", {})
        truth_contract = hparams.get("partonic_truth_contract", {})
        split_identity = hparams.get("split_identity", {})
        scaler = saved.get("partonic_scaler")
        if truth_contract.get("truth_definition") != TRUTH_DEFINITION:
            raise ValueError(f"Checkpoint does not declare {TRUTH_DEFINITION} truth")
        if truth_contract.get("roles") != list(ROLES) or truth_contract.get("truth_bindings") != expected_bindings:
            raise ValueError("Checkpoint target role/binding metadata differs from the experiment")
        if hparams.get("topology_signature") != topology.signature:
            raise ValueError("Checkpoint compiled topology signature differs from the experiment")
        if any(
            truth_contract.get(key) != measured_contract[key]
            for key in ("source_manifest_sha256", "split_membership_sha256", "split_version")
        ):
            raise ValueError("Checkpoint source/split identity differs from the measured input")
        if split_identity.get("source_contract") != measured_contract:
            raise ValueError("Checkpoint frozen split/source identity differs from the measured input")
        if allow_reserved_test:
            counts = split_identity.get("event_id_counts", {})
            if (
                split_identity.get("policy") != "canonical_hdf5_SPLIT"
                or int(counts.get("train", 0)) <= 0
                or int(counts.get("validation", 0)) <= 0
                or int(counts.get("test", 0)) != 0
                or int(split_counts.get("2", 0)) != event_count
            ):
                raise ValueError(
                    "Reserved-test inference requires a checkpoint trained on the matching "
                    "train/validation population and a test-only input"
                )
        scaler_fit = hparams.get("partonic_scaler_fit")
        if not isinstance(scaler_fit, dict) or scaler_fit.get("population") != "frozen_train_split":
            raise ValueError("Checkpoint does not declare train-only partonic scaler fitting")
        if (
            scaler_fit.get("event_id_hash")
            != split_identity.get("event_id_hashes", {}).get("train")
            or int(scaler_fit.get("event_count", -1))
            != int(split_identity.get("event_id_counts", {}).get("train", -2))
        ):
            raise ValueError("Checkpoint partonic scaler-fit identity is not the frozen training split")
        if not isinstance(scaler, dict) or scaler.get("representation") != TARGET_REPRESENTATION:
            raise ValueError("Checkpoint is missing the compatible train-fitted partonic scaler")
        if scaler.get("role_names") != list(ROLES):
            raise ValueError("Checkpoint partonic scaler role order is incompatible")
        mean = np.asarray(scaler.get("mean"), dtype=np.float64)
        std = np.asarray(scaler.get("std"), dtype=np.float64)
        if mean.shape != (2, 4) or std.shape != (2, 4) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Checkpoint partonic scaler arrays are malformed")
        result["checkpoint"] = str(checkpoint)
        result["checkpoint_scaler"] = scaler
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-reserved-test", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(
        validate_experiment(
            args.experiment,
            args.checkpoint,
            allow_reserved_test=args.allow_reserved_test,
            overrides=args.overrides,
        ),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
