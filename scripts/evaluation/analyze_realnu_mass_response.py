#!/usr/bin/env python3
"""Diagnose event and population mttbar response from an accepted posterior."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.isotonic import IsotonicRegression

from reduce_realnu_parent_posterior import _observables


EXPECTED_EVENTS = 100342
EXPECTED_SAMPLES = 32
EXPECTED_MASS_RMSE = {
    "single_draw": 108.282,
    "observable_mean": 79.052,
    "observable_q50": 81.270,
}
COARSE_EDGES = np.asarray([-np.inf, 345.0, 400.0, 500.0, 700.0, 1000.0, np.inf])
COARSE_LABELS = ("<345", "345-400", "400-500", "500-700", "700-1000", ">1000")
PARENT_OBSERVABLES = (
    "top_pt", "antitop_pt", "top_rapidity", "antitop_rapidity", "top_phi", "antitop_phi"
)
ORACLE_PATHS = {
    "perfect_reco": "Perfect reco: measured visible constituents + truth assignment + truth neutrinos",
    "predicted_assignment_truth_neutrinos": "Predicted assignment + truth neutrinos",
    "truth_assignment_predicted_neutrinos": "Truth assignment + predicted neutrinos",
    "predicted_assignment_predicted_neutrinos": "Predicted assignment + predicted neutrinos",
}


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_ids(ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write a headerless empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})
    os.replace(temporary, path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _resolve_inputs(manifest_path: Path) -> dict:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Comparison manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("sealed_test_used") is not False:
        raise ValueError("Comparison manifest does not establish sealed_test_used=false")
    try:
        accepted = manifest["paired_full_validation"]["models"]["PCFM_Q"]
        reductions = accepted["reduction_outputs"]
        summary_path = Path(reductions["summary"]["path"]).expanduser().resolve()
        arrays_path = Path(reductions["arrays"]["path"]).expanduser().resolve()
        event_table_path = Path(accepted["optional_perfect_reco_reference"]["event_table"]).expanduser().resolve()
        source_bank_path = Path(manifest["source_bank"]["file"]).expanduser().resolve()
    except (KeyError, TypeError) as error:
        raise ValueError("Comparison manifest lacks the accepted PCFM_Q reduction chain") from error
    for path in (summary_path, arrays_path, event_table_path, source_bank_path):
        if not path.is_file():
            raise FileNotFoundError(f"Manifest-recorded input is missing: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "passed" or summary.get("sealed_test_used") is not False:
        raise ValueError("Accepted Q reduction summary is not a passed validation-only artifact")
    prediction_path = Path(summary["predictions"]).expanduser().resolve()
    if Path(summary["event_table"]).expanduser().resolve() != event_table_path:
        raise ValueError("Accepted Q reduction and comparison manifest name different event tables")
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Manifest-linked Q prediction is missing: {prediction_path}")
    expected_samples = int(summary.get("samples", -1))
    expected_events = int(summary.get("events", -1))
    if expected_events != EXPECTED_EVENTS or expected_samples != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected full M32 validation ({EXPECTED_EVENTS} x {EXPECTED_SAMPLES}), "
            f"found {expected_events} x {expected_samples}"
        )
    if summary.get("solver") != "heun" or int(summary.get("steps", -1)) != 16:
        raise ValueError("Accepted Q summary is not the Heun16 inference policy")
    if summary.get("parent_role_order") != ["top", "antitop"]:
        raise ValueError("Accepted Q summary has unexpected parent ordering")
    if summary.get("truth_free_conditioning") is not True:
        raise ValueError("Accepted Q prediction is not marked truth-free")
    source_manifest = manifest["source_bank"]
    if summary.get("source_bank_sha256") != source_manifest.get("file_sha256"):
        raise ValueError("Q summary and comparison manifest name different source banks")
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "summary_path": summary_path,
        "summary": summary,
        "arrays_path": arrays_path,
        "event_table_path": event_table_path,
        "source_bank_path": source_bank_path,
        "prediction_path": prediction_path,
    }


def _fine_edges() -> np.ndarray:
    """Fixed bins with adequate central/tail counts and a populated >1700 GeV overflow."""
    edges = [-math.inf, 325.0, 345.0]
    edges.extend(np.arange(370.0, 671.0, 25.0).tolist())
    edges.extend([700.0, 750.0, 800.0, 850.0, 900.0, 950.0, 1000.0])
    edges.extend(np.arange(1100.0, 1701.0, 100.0).tolist())
    edges.append(np.inf)
    result = np.asarray(edges, dtype=np.float64)
    if not np.all(np.diff(result) > 0.0):
        raise RuntimeError("Fine mttbar edges are not strictly increasing")
    return result


def _h5_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    """Read unique HDF5 rows in sorted order, then restore the requested order."""
    rows = np.asarray(rows, dtype=np.int64)
    sort_order = np.argsort(rows, kind="stable")
    sorted_rows = rows[sort_order]
    if len(sorted_rows) > 1 and np.any(np.diff(sorted_rows) <= 0):
        raise ValueError("EVENT_ID join resolved multiple predictions to the same event-table row")
    sorted_values = np.asarray(dataset[sorted_rows])
    return sorted_values[np.argsort(sort_order, kind="stable")]


def _bin_ids(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Mass binning received non-finite values")
    indices = np.searchsorted(edges, values, side="right") - 1
    if np.array_equal(edges, COARSE_EDGES):
        indices[values == 1000.0] = 4
    if np.any(indices < 0) or np.any(indices >= len(edges) - 1):
        raise ValueError("Mass value fell outside explicit underflow/overflow bins")
    return indices.astype(np.int64, copy=False)


def _bin_labels(edges: np.ndarray, *, coarse: bool = False) -> list[str]:
    if coarse:
        return list(COARSE_LABELS)
    labels = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        if np.isneginf(low):
            labels.append(f"<{high:g}")
        elif np.isinf(high):
            labels.append(f">{low:g}")
        elif low == 0.0:
            labels.append(f"0-{high:g}")
        else:
            labels.append(f"{low:g}-{high:g}")
    return labels


def _join_truth_and_oracles(event_table_path: Path, event_ids: np.ndarray, *, limit: int | None = None) -> dict:
    with h5py.File(event_table_path, "r") as table:
        if table.attrs.get("schema") != "realnu_event_metrics_v1":
            raise ValueError("Event metrics table has an unexpected schema")
        if bool(table.attrs.get("sealed_test_used", True)):
            raise ValueError("Event metrics table does not establish sealed_test_used=false")
        role_order = _decode(table.attrs.get("parent_role_order", ""))
        if role_order != '["top", "antitop"]':
            raise ValueError("Event metrics table parent ordering is not (top, antitop)")
        table_ids = np.asarray(table["EVENT_ID"][:], dtype=np.int64)
        if len(np.unique(table_ids)) != len(table_ids):
            raise ValueError("Accepted event table contains duplicate EVENT_IDs")
        order = np.argsort(table_ids, kind="stable")
        positions = np.searchsorted(table_ids[order], event_ids)
        if np.any(positions >= len(order)) or not np.array_equal(table_ids[order[positions]], event_ids):
            raise ValueError("Prediction EVENT_IDs do not exactly join to accepted validation data")
        rows = order[positions]
        support = np.asarray(_h5_rows(table["support/parent_roles"], rows), dtype=bool)
        if support.shape != (len(event_ids), 2) or not support.all():
            raise ValueError("Truth parent support is incomplete for the requested validation events")
        truth_parents = np.asarray(_h5_rows(table["truth/parents"], rows), dtype=np.float64)
        truth_stored = np.asarray(_h5_rows(table["truth/mttbar"], rows), dtype=np.float64)
        truth_derived = _observables(truth_parents[:, None])["mttbar"][:, 0]
        if not np.allclose(truth_derived, truth_stored, rtol=5.0e-6, atol=1.0e-3):
            raise ValueError("Truth mttbar does not close from the accepted truth parent four-vectors")
        if not np.isfinite(truth_derived).all():
            raise ValueError("Truth mttbar contains non-finite validation values")
        q1 = table["models/Q1"]
        oracles: dict[str, np.ndarray] = {}
        for name in ORACLE_PATHS:
            path = f"constituent_oracles/{name}/parents"
            if path not in q1:
                raise KeyError(f"Event table is missing Q1 oracle product {path}")
            parents = np.asarray(_h5_rows(q1[path], rows), dtype=np.float64)
            finite = np.isfinite(parents).all(axis=(-1, -2))
            mass = np.full(len(event_ids), np.nan, dtype=np.float64)
            if finite.any():
                derived = _observables(parents[finite, None])["mttbar"][:, 0]
                stored = np.asarray(_h5_rows(q1[f"constituent_oracles/{name}/mttbar"], rows)[finite], dtype=np.float64)
                if not np.allclose(derived, stored, rtol=5.0e-6, atol=1.0e-3):
                    raise ValueError(f"Q1 {name} mass does not close from stored parent four-vectors")
                mass[finite] = derived
            oracles[name] = mass
        truth = {
            "mttbar": truth_derived,
            "parents": truth_parents,
            "channel": _h5_rows(table["channel"], rows),
            "source_njet": np.asarray(_h5_rows(table["source_njet"], rows), dtype=np.int64),
            "model_visible_njet": np.asarray(_h5_rows(table["model_visible_njet"], rows), dtype=np.int64),
        }
        assignment = {
            "assignment_supported": np.asarray(_h5_rows(q1["assignment_supported"], rows), dtype=bool),
            "assignment_complete_correct": np.asarray(_h5_rows(q1["assignment_complete_correct"], rows), dtype=bool),
            "assignment_role_supported": np.asarray(_h5_rows(q1["assignment_role_supported"], rows), dtype=bool).all(axis=1),
            "assignment_role_correct": np.asarray(_h5_rows(q1["assignment_role_correct"], rows), dtype=bool).all(axis=1),
        }
    return {"truth": truth, "oracles": oracles, "assignment": assignment}


def _prediction_provenance(inputs: dict, ids: np.ndarray) -> dict:
    pred_path = inputs["prediction_path"]
    summary = inputs["summary"]
    manifest = inputs["manifest"]
    prediction_sha = _sha256(pred_path)
    if prediction_sha != summary.get("predictions_sha256"):
        raise ValueError("Prediction SHA-256 differs from the accepted Q reduction summary")
    id_sha = _sha256_ids(ids)
    if id_sha != summary.get("event_id_sha256"):
        raise ValueError("Prediction EVENT_ID SHA-256 differs from the accepted Q summary")
    with h5py.File(pred_path, "r") as prediction:
        attrs = {name: _decode(prediction.attrs[name]) for name in prediction.attrs}
        if attrs.get("prediction_schema") != "realnu_parent_solver_scan_v1":
            raise ValueError("Unsupported Q prediction schema")
        if attrs.get("split") != "validation" or attrs.get("truth_free_input") is not True:
            raise ValueError("Q prediction is not a truth-free validation product")
        if attrs.get("sealed_test_used") is not False:
            raise ValueError("Q prediction does not record sealed_test_used=false")
        if int(attrs.get("event_count", -1)) != EXPECTED_EVENTS:
            raise ValueError("Q prediction event-count metadata is inconsistent")
        if attrs.get("event_id_sha256") != id_sha:
            raise ValueError("Prediction ordered EVENT_ID hash differs from its HDF5 metadata")
        if attrs.get("parent_role_order") != '["top", "antitop"]':
            raise ValueError("Q prediction parent ordering is not (top, antitop)")
        if attrs.get("solver") != "heun" or int(attrs.get("source_sample_prefix", -1)) != EXPECTED_SAMPLES:
            raise ValueError("Q prediction solver or source sample count differs from accepted policy")
        if attrs.get("checkpoint_sha256") != summary.get("checkpoint_sha256"):
            raise ValueError("Prediction checkpoint differs from accepted reduction metadata")
        if attrs.get("checkpoint_sha256") != manifest.get("checkpoints", {}).get("PCFM_Q_best_sha256"):
            raise ValueError("Prediction checkpoint differs from comparison manifest")
        if attrs.get("source_bank_sha256") != manifest["source_bank"].get("file_sha256"):
            raise ValueError("Prediction source bank differs from comparison manifest")
        if attrs.get("scaler_identity_sha256") != manifest["source_bank"].get("target_scaler_identity_sha256"):
            raise ValueError("Prediction target scaler differs from comparison manifest")
        checkpoint_path = Path(str(attrs.get("checkpoint_path", ""))).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Prediction-recorded checkpoint is missing: {checkpoint_path}")
        checkpoint_sha = _sha256(checkpoint_path)
        if checkpoint_sha != attrs.get("checkpoint_sha256"):
            raise ValueError("Prediction-recorded checkpoint file hash differs from its metadata")
        if checkpoint_sha != manifest.get("checkpoints", {}).get("PCFM_Q_best_sha256"):
            raise ValueError("Checkpoint file hash differs from the comparison manifest")
        source_path = Path(str(attrs.get("source_bank_path", ""))).expanduser().resolve()
        if source_path != inputs["source_bank_path"]:
            raise ValueError("Prediction and comparison manifest name different source-bank paths")
    source_sha = _sha256(inputs["source_bank_path"])
    if source_sha != manifest["source_bank"].get("file_sha256"):
        raise ValueError("Source bank SHA-256 differs from comparison manifest")
    with np.load(inputs["source_bank_path"], allow_pickle=False) as source_bank:
        bank_ids = np.asarray(source_bank["EVENT_ID"], dtype=np.int64)
        if len(np.unique(bank_ids)) != len(bank_ids) or len(bank_ids) != len(ids):
            raise ValueError("Source-bank EVENT_ID population is not unique and complete")
        bank_id_sha = _sha256_ids(bank_ids)
        if bank_id_sha != manifest["source_bank"].get("event_id_sha256"):
            raise ValueError("Source-bank EVENT_ID hash differs from comparison manifest")
        if not np.array_equal(np.sort(bank_ids), np.sort(ids)):
            raise ValueError("Source bank and Q prediction do not contain the same EVENT_ID set")
    return {
        "prediction": str(pred_path),
        "prediction_schema": "realnu_parent_solver_scan_v1",
        "prediction_sha256": prediction_sha,
        "event_id_sha256": id_sha,
        "comparison_manifest_sha256": _sha256(inputs["manifest_path"]),
        "accepted_reduction_summary": str(inputs["summary_path"]),
        "accepted_reduction_summary_sha256": _sha256(inputs["summary_path"]),
        "accepted_reduction_arrays": str(inputs["arrays_path"]),
        "accepted_reduction_arrays_sha256": _sha256(inputs["arrays_path"]),
        "source_bank": str(inputs["source_bank_path"]),
        "source_bank_sha256": source_sha,
        "source_bank_event_id_sha256": bank_id_sha,
        "source_bank_and_prediction_id_sets_equal": True,
        "source_bank_and_prediction_order_equal": bool(np.array_equal(bank_ids, ids)),
        "event_table": str(inputs["event_table_path"]),
        "event_table_sha256": _sha256(inputs["event_table_path"]),
        "checkpoint_path": attrs.get("checkpoint_path"),
        "checkpoint_sha256": checkpoint_sha,
        "scaler_identity_sha256": manifest["source_bank"]["target_scaler_identity_sha256"],
        "solver": summary["solver"],
        "steps": int(summary["steps"]),
        "samples": int(summary["samples"]),
        "inference_policy": {"solver": summary["solver"], "steps": int(summary["steps"]),
                             "posterior_samples": int(summary["samples"]),
                             "nfe_per_sample": 32, "source_sample_prefix": 32},
        "parent_role_order": summary["parent_role_order"],
        "truth_free_conditioning": summary["truth_free_conditioning"],
        "sealed_test_used": False,
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _check_mass_reducer_regression(samples: np.ndarray, medoid_index: np.ndarray,
                                  reference: dict[str, np.ndarray]) -> dict[str, float]:
    reducers = {
        "sample0": samples[:, 0],
        "observable_mean": samples.mean(axis=1),
        "observable_q50": np.median(samples, axis=1),
        "joint_medoid": samples[np.arange(len(samples)), medoid_index],
    }
    differences = {}
    for name, value in reducers.items():
        difference = float(np.max(np.abs(value - reference[name])))
        if difference > 1.0e-3:
            raise RuntimeError(
                f"Derived {name} mttbar differs from accepted reducer by {difference:.6g} GeV"
            )
        differences[name] = difference
    return differences


def _load_or_build_cache(inputs: dict, output_dir: Path, *, smoke_events: int | None = None) -> tuple[dict, dict]:
    summary = inputs["summary"]
    with h5py.File(inputs["prediction_path"], "r") as prediction:
        ids_all = np.asarray(prediction["EVENT_ID"][:], dtype=np.int64)
        if len(ids_all) != EXPECTED_EVENTS or len(np.unique(ids_all)) != EXPECTED_EVENTS:
            raise ValueError("Prediction does not have 100342 unique ordered validation EVENT_IDs")
    n = len(ids_all) if smoke_events is None else min(int(smoke_events), len(ids_all))
    if smoke_events is not None and not 2000 <= n <= 10000:
        raise ValueError("A bounded smoke must select between 2000 and 10000 events")
    ids = ids_all[:n]
    full_provenance = _prediction_provenance(inputs, ids_all)
    joined = _join_truth_and_oracles(inputs["event_table_path"], ids)
    truth = joined["truth"]

    with np.load(inputs["arrays_path"], allow_pickle=False) as reduced:
        reduced_ids = np.asarray(reduced["EVENT_ID"][:n], dtype=np.int64)
        if not np.array_equal(reduced_ids, ids):
            raise ValueError("Accepted reducer EVENT_ID order differs from prediction order")
        reference = {
            name: np.asarray(reduced[f"pred__{estimator}__mttbar"][:n], dtype=np.float64)
            for estimator, name in (
                ("single_draw", "sample0"),
                ("observable_mean", "observable_mean"),
                ("observable_q50", "observable_q50"),
                ("joint_medoid", "joint_medoid"),
            )
        }
        medoid_index = np.asarray(reduced["joint_medoid_sample_index"][:n], dtype=np.int64)
        expected_medoid_parents = np.asarray(reduced["parent_joint_medoid"][:n], dtype=np.float64)

    cache_path = output_dir / "cache/PCFM_Q_mass_samples_m32.npz"
    if smoke_events is None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_meta = json.loads(str(cache["metadata_json"].item()))
            cached_provenance = cached_meta.get("provenance", {})
            if not cached_provenance or any(full_provenance.get(key) != value for key, value in cached_provenance.items()):
                raise ValueError("Existing mass cache provenance differs from the authoritative inputs")
            values = {key: np.asarray(cache[key]) for key in cache.files if key != "metadata_json"}
        if not np.array_equal(values["EVENT_ID"], ids_all):
            raise ValueError("Existing mass cache EVENT_ID order differs from prediction")
        if values["sample_mttbar"].shape != (EXPECTED_EVENTS, EXPECTED_SAMPLES) or not np.isfinite(values["sample_mttbar"]).all():
            raise ValueError("Existing mttbar cache samples have malformed shape or non-finite values")
        if values["parent_samples"].shape != (EXPECTED_EVENTS, EXPECTED_SAMPLES, 2, 4) or not np.isfinite(values["parent_samples"]).all():
            raise ValueError("Existing parent sample cache has malformed shape or non-finite values")
        if not np.allclose(_observables(values["parent_samples"])["mttbar"], values["sample_mttbar"], rtol=0.0, atol=1.0e-10):
            raise ValueError("Existing mttbar cache does not close against its cached parent samples")
        if not np.array_equal(values["perfect_reco_mask"], np.isfinite(values["perfect_reco_mttbar"])):
            raise ValueError("Existing Perfect-reco support mask does not match cached values")
        regression = _check_mass_reducer_regression(
            values["sample_mttbar"], values["joint_medoid_sample_index"], reference
        )
        return {**values, "truth": truth, "oracles": joined["oracles"], "assignment": joined["assignment"], "reference": reference}, {
            **full_provenance,
            "cache_path": str(cache_path),
            "cache_reused": True,
            "cache_created_utc": cached_meta.get("created_utc"),
            "cache_reducer_max_abs_diff_gev": regression,
            "events": n,
        }

    with h5py.File(inputs["prediction_path"], "r") as prediction:
        group = prediction["CASES/heun_steps_16"]
        parents = np.asarray(group["parents"][:n], dtype=np.float32)
        states = np.asarray(group["scaled_states"][:n], dtype=np.float32)
        case_steps = int(group.attrs["steps"])
        case_solver = _decode(group.attrs["solver"])
        case_nfe = int(group.attrs["nfe_per_sample"])
    if parents.shape != (n, EXPECTED_SAMPLES, 2, 4) or states.shape != (n, EXPECTED_SAMPLES, 8):
        raise ValueError(f"Q parent sample schema has unexpected shapes: {parents.shape}, {states.shape}")
    if case_solver != "heun" or case_steps != 16 or case_nfe != 32:
        raise ValueError("Prediction case metadata differs from Heun16 / 32 NFE")
    if not np.isfinite(parents).all() or not np.isfinite(states).all():
        raise ValueError("Q prediction contains non-finite parent samples or scaled states")
    mass_samples = np.asarray(_observables(parents)["mttbar"], dtype=np.float64)
    if mass_samples.shape != (n, EXPECTED_SAMPLES) or not np.isfinite(mass_samples).all():
        raise ValueError("Derived Q mttbar sample array is malformed or non-finite")
    if np.any(medoid_index < 0) or np.any(medoid_index >= EXPECTED_SAMPLES):
        raise ValueError("Accepted joint-medoid sample indices fall outside M32")
    selected_medoid = parents[np.arange(n), medoid_index]
    if not np.allclose(selected_medoid, expected_medoid_parents, rtol=0.0, atol=1.0e-6):
        raise ValueError("Accepted joint-medoid sample index does not select its stored parent pair")
    derived_medoid_mass = mass_samples[np.arange(n), medoid_index]
    if not np.allclose(derived_medoid_mass, reference["joint_medoid"], rtol=5.0e-6, atol=1.0e-3):
        raise ValueError("Joint-medoid mass does not reproduce accepted reduced values")
    if not np.allclose(truth["mttbar"], np.asarray(_observables(truth["parents"][:, None])["mttbar"][:, 0]), rtol=0.0, atol=1.0e-12):
        raise ValueError("Truth mttbar derivation changed after exact EVENT_ID joining")
    with np.load(inputs["arrays_path"], allow_pickle=False) as reduced:
        reduced_truth = np.asarray(reduced["truth__mttbar"][:n], dtype=np.float64)
    if not np.allclose(truth["mttbar"], reduced_truth, rtol=5.0e-6, atol=1.0e-3):
        raise ValueError("Truth mttbar differs from the accepted full-validation reducer")
    reducer_regression = _check_mass_reducer_regression(mass_samples, medoid_index, reference)
    perfect = joined["oracles"]["perfect_reco"]
    perfect_mask = np.isfinite(perfect)
    if not perfect_mask.any():
        raise ValueError("Perfect-reco Q1 reference has no finite parent-pair support")

    if smoke_events is None:
        cache_meta = {
            "schema": "realnu_mttbar_sample_cache_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "provenance": dict(full_provenance),
        }
        _atomic_npz(
            cache_path,
            metadata_json=np.asarray(json.dumps(cache_meta, sort_keys=True)),
            EVENT_ID=ids,
            truth_mttbar=truth["mttbar"],
            sample_mttbar=mass_samples,
            parent_samples=parents,
            joint_medoid_sample_index=medoid_index,
            perfect_reco_mask=perfect_mask,
            perfect_reco_mttbar=perfect,
        )
    values = {
        "EVENT_ID": ids,
        "truth_mttbar": truth["mttbar"],
        "sample_mttbar": mass_samples,
        "parent_samples": parents,
        "joint_medoid_sample_index": medoid_index,
        "perfect_reco_mask": perfect_mask,
        "perfect_reco_mttbar": perfect,
    }
    return {**values, "truth": truth, "oracles": joined["oracles"],
            "assignment": joined["assignment"], "reference": reference}, {
        **full_provenance,
        "cache_path": str(cache_path) if smoke_events is None else None,
        "cache_reused": False,
        "cache_created_utc": cache_meta["created_utc"] if smoke_events is None else None,
        "cache_reducer_max_abs_diff_gev": reducer_regression,
        "events": n,
    }


def _region_masks(truth_mass: np.ndarray) -> dict[str, np.ndarray]:
    result = {"inclusive": np.ones(len(truth_mass), dtype=bool)}
    labels = _bin_ids(truth_mass, COARSE_EDGES)
    result.update({label: labels == index for index, label in enumerate(COARSE_LABELS)})
    return result


def _sigma68(residual: np.ndarray) -> float:
    q16, q84 = np.quantile(residual, [0.16, 0.84])
    return float((q84 - q16) / 2.0)


def _point_metrics(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    y = np.asarray(truth[mask], dtype=np.float64)
    x = np.asarray(prediction[mask], dtype=np.float64)
    if len(x) == 0:
        return {"events": 0}
    residual = x - y
    slope = float(np.cov(y, x, ddof=0)[0, 1] / np.var(y)) if np.var(y) > 0.0 else float("nan")
    return {
        "events": int(len(y)),
        "truth_mean_gev": float(y.mean()),
        "reco_mean_gev": float(x.mean()),
        "bias_gev": float(residual.mean()),
        "std_residual_gev": float(residual.std()),
        "sigma68_gev": _sigma68(residual),
        "mae_gev": float(np.abs(residual).mean()),
        "rmse_gev": float(np.sqrt(np.mean(np.square(residual)))),
        "relative_bias": float(residual.mean() / y.mean()) if y.mean() > 0.0 else float("nan"),
        "relative_resolution": float(_sigma68(residual) / y.mean()) if y.mean() > 0.0 else float("nan"),
        "response_slope": slope,
        "truth_std_gev": float(y.std()),
    }


def _mass_posterior_mode(samples: np.ndarray, *, chunk_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Sample-grid Gaussian KDE mode with event-wise robust Silverman bandwidth."""
    samples = np.asarray(samples, dtype=np.float64)
    q25, q75 = np.quantile(samples, [0.25, 0.75], axis=1)
    robust_scale = np.minimum(samples.std(axis=1), (q75 - q25) / 1.349)
    bandwidth = 0.9 * robust_scale * samples.shape[1] ** (-0.2)
    positive = bandwidth[bandwidth > 0.0]
    floor = max(float(np.median(positive)) * 1.0e-3 if len(positive) else 0.0, 1.0e-4)
    bandwidth = np.maximum(bandwidth, floor)
    index = np.empty(len(samples), dtype=np.int64)
    for start in range(0, len(samples), chunk_size):
        stop = min(start + chunk_size, len(samples))
        delta = (samples[start:stop, :, None] - samples[start:stop, None, :]) / bandwidth[start:stop, None, None]
        density = np.exp(-0.5 * np.square(delta)).sum(axis=-1)
        index[start:stop] = density.argmax(axis=1)
    return samples[np.arange(len(samples)), index], bandwidth


def _build_estimates(data: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    samples = np.asarray(data["sample_mttbar"], dtype=np.float64)
    parents = np.asarray(data["parent_samples"], dtype=np.float32)
    medoid_index = np.asarray(data["joint_medoid_sample_index"], dtype=np.int64)
    mean = samples.mean(axis=1)
    q50 = np.median(samples, axis=1)
    nearest_q50 = np.abs(samples - q50[:, None]).argmin(axis=1)
    nearest_mean = np.abs(samples - mean[:, None]).argmin(axis=1)
    medoid_parent = parents[np.arange(len(parents)), medoid_index]
    q50_parent = parents[np.arange(len(parents)), nearest_q50]
    mean_parent = parents[np.arange(len(parents)), nearest_mean]
    parent_observables = {
        "joint_medoid": _observables(medoid_parent[:, None]),
        "coherent_nearest_q50": _observables(q50_parent[:, None]),
        "coherent_nearest_mean": _observables(mean_parent[:, None]),
    }
    mode, bandwidth = _mass_posterior_mode(samples)
    estimates = {
        "sample0": samples[:, 0],
        "observable_mean": mean,
        "observable_q50": q50,
        "joint_medoid": samples[np.arange(len(samples)), medoid_index],
        "coherent_nearest_q50": samples[np.arange(len(samples)), nearest_q50],
        "coherent_nearest_mean": samples[np.arange(len(samples)), nearest_mean],
        "mass_posterior_mode": mode,
    }
    extras = {
        "nearest_q50_index": nearest_q50,
        "nearest_mean_index": nearest_mean,
        "medoid_index": medoid_index,
        "mass_mode_bandwidth": bandwidth,
        "medoid_parent": medoid_parent,
        "nearest_q50_parent": q50_parent,
        "nearest_mean_parent": mean_parent,
    }
    for reducer, observable_map in parent_observables.items():
        for observable in PARENT_OBSERVABLES:
            extras[f"{reducer}_{observable}"] = np.asarray(observable_map[observable])[:, 0]
    return estimates, extras


def _fine_bin_summary(estimates: dict[str, np.ndarray], truth: np.ndarray, edges: np.ndarray) -> list[dict]:
    truth_bin = _bin_ids(truth, edges)
    labels = _bin_labels(edges)
    rows = []
    for bin_index, label in enumerate(labels):
        mask = truth_bin == bin_index
        if not mask.any():
            continue
        bounds = (float(edges[bin_index]), float(edges[bin_index + 1]))
        for estimator, value in estimates.items():
            rows.append({"estimator": estimator, "truth_bin": label,
                         "truth_low_gev": bounds[0], "truth_high_gev": bounds[1],
                         **_point_metrics(value, truth, mask)})
    return rows


def _migration_matrix(truth: np.ndarray, prediction: np.ndarray | None, edges: np.ndarray,
                      *, samples: np.ndarray | None = None) -> dict:
    n_bins = len(edges) - 1
    true_id = _bin_ids(truth, edges)
    counts = np.bincount(true_id, minlength=n_bins).astype(np.float64)
    matrix = np.zeros((n_bins, n_bins), dtype=np.float64)
    if samples is None:
        reco_id = _bin_ids(np.asarray(prediction), edges)
        np.add.at(matrix, (true_id, reco_id), 1.0)
        displacement = reco_id - true_id
        correct_probability = (reco_id == true_id).astype(np.float64)
        adjacent_probability = (np.abs(displacement) <= 1).astype(np.float64)
    else:
        sample_id = _bin_ids(np.asarray(samples).reshape(-1), edges).reshape(samples.shape)
        n_samples = samples.shape[1]
        row_index = np.repeat(true_id, n_samples)
        np.add.at(matrix, (row_index, sample_id.reshape(-1)), 1.0 / n_samples)
        event_displacement = sample_id - true_id[:, None]
        correct_probability = np.mean(event_displacement == 0, axis=1)
        adjacent_probability = np.mean(np.abs(event_displacement) <= 1, axis=1)
        displacement = event_displacement.reshape(-1)
    row = np.divide(matrix, counts[:, None], out=np.zeros_like(matrix), where=counts[:, None] > 0.0)
    column_total = matrix.sum(axis=0)
    column = np.divide(matrix, column_total[None, :], out=np.zeros_like(matrix), where=column_total[None, :] > 0.0)
    row_entropy = -np.sum(np.where(row > 0.0, row * np.log(np.maximum(row, 1.0e-300)), 0.0), axis=1)
    column_purity = np.diag(matrix) / np.maximum(column_total, 1.0e-300)
    return {
        "matrix": matrix,
        "row": row,
        "column": column,
        "counts": counts,
        "row_entropy": row_entropy,
        "column_purity": column_purity,
        "diagonal_fraction": float(np.trace(matrix) / max(matrix.sum(), 1.0)),
        "adjacent_fraction": float(np.mean(adjacent_probability)),
        "mean_abs_bin_displacement": float(np.mean(np.abs(displacement))),
        "rms_bin_displacement": float(np.sqrt(np.mean(np.square(displacement)))),
        "mean_correct_probability": float(np.mean(correct_probability)),
        "truth_bin_ids": true_id,
    }


def _matrix_csv_rows(name: str, binning: str, matrix: dict, edges: np.ndarray) -> list[dict]:
    labels = _bin_labels(edges, coarse=binning == "coarse")
    rows = []
    for i, truth_label in enumerate(labels):
        for j, reco_label in enumerate(labels):
            rows.append({
                "estimator": name, "binning": binning,
                "truth_bin": truth_label, "reco_bin": reco_label,
                "event_count_truth_bin": int(matrix["counts"][i]),
                "weighted_count": float(matrix["matrix"][i, j]),
                "P_reco_given_truth": float(matrix["row"][i, j]),
                "P_truth_given_reco": float(matrix["column"][i, j]),
                "truth_row_entropy_nats": float(matrix["row_entropy"][i]),
                "reco_column_purity": float(matrix["column_purity"][j]),
            })
    return rows


def _posterior_predictive(samples: np.ndarray, truth: np.ndarray, edges: np.ndarray) -> dict:
    result = _migration_matrix(truth, None, edges, samples=samples)
    p = result["row"]
    truth_id = result["truth_bin_ids"]
    sample_id = _bin_ids(samples.reshape(-1), edges).reshape(samples.shape)
    event_p = np.zeros((len(truth), p.shape[1]), dtype=np.float64)
    np.add.at(
        event_p,
        (np.repeat(np.arange(len(truth)), samples.shape[1]), sample_id.reshape(-1)),
        1.0 / samples.shape[1],
    )
    p_correct = event_p[np.arange(len(truth)), truth_id]
    adjacent_p = np.zeros(len(truth), dtype=np.float64)
    for shift in (-1, 0, 1):
        idx = truth_id + shift
        valid = (idx >= 0) & (idx < p.shape[1])
        adjacent_p[valid] += event_p[np.arange(len(truth))[valid], idx[valid]]
    target = np.zeros_like(event_p)
    target[np.arange(len(truth)), truth_id] = 1.0
    brier = np.sum(np.square(event_p - target), axis=1)
    entropy = -np.sum(np.where(event_p > 0.0, event_p * np.log(np.maximum(event_p, 1.0e-300)), 0.0), axis=1)
    result.update({
        "event_correct_probability": p_correct,
        "event_adjacent_probability": adjacent_p,
        "event_brier": brier,
        "event_entropy": entropy,
        "event_probabilities": event_p,
        "mean_correct_probability": float(p_correct.mean()),
        "mean_adjacent_probability": float(adjacent_p.mean()),
        "brier_score": float(brier.mean()),
        "mean_entropy": float(entropy.mean()),
    })
    if not np.allclose(result["matrix"].sum(axis=1), result["counts"], rtol=0.0, atol=1.0e-8):
        raise RuntimeError("Posterior-predictive migration rows do not preserve event-equivalent weight")
    return result


def _posterior_diagnostics(samples: np.ndarray, truth: np.ndarray, masks: dict) -> tuple[list[dict], list[dict], dict]:
    q05, q16, q50, q84, q95 = np.quantile(samples, [0.05, 0.16, 0.50, 0.84, 0.95], axis=1)
    width68, width90 = q84 - q16, q95 - q05
    true_bin = _bin_ids(truth, COARSE_EDGES)
    sample_bin = _bin_ids(samples.reshape(-1), COARSE_EDGES).reshape(samples.shape)
    exact = np.mean(sample_bin == true_bin[:, None], axis=1)
    adjacent = np.mean(np.abs(sample_bin - true_bin[:, None]) <= 1, axis=1)
    best_error = np.min(np.abs(samples - truth[:, None]), axis=1)
    best_index = np.abs(samples - truth[:, None]).argmin(axis=1)
    best_mass = samples[np.arange(len(samples)), best_index]
    rank = np.sum(samples < truth[:, None], axis=1) + 0.5 * np.sum(samples == truth[:, None], axis=1)
    rows, rank_rows = [], []
    rank_edges = np.arange(-0.5, samples.shape[1] + 1.5, 1.0)
    for region, mask in masks.items():
        bin_counts, _ = np.histogram(rank[mask], bins=rank_edges)
        for index, count in enumerate(bin_counts):
            rank_rows.append({"truth_region": region, "rank_mid_samples": index,
                              "count": int(count), "fraction": float(count / max(mask.sum(), 1)),
                              "sample_count": int(samples.shape[1]),
                              "convention": "midrank = count(sample < truth) + 0.5*count(sample == truth)"})
        residual_mean = samples.mean(axis=1) - truth
        width_corr = spearmanr(width68[mask], np.abs(residual_mean[mask])).statistic if mask.sum() > 2 else float("nan")
        best_residual = best_mass[mask] - truth[mask]
        rows.append({
            "truth_region": region, "events": int(mask.sum()),
            "coverage68": float(np.mean((truth[mask] >= q16[mask]) & (truth[mask] <= q84[mask]))),
            "coverage90": float(np.mean((truth[mask] >= q05[mask]) & (truth[mask] <= q95[mask]))),
            "width68_mean_gev": float(width68[mask].mean()),
            "width68_median_gev": float(np.median(width68[mask])),
            "width90_mean_gev": float(width90[mask].mean()),
            "width90_median_gev": float(np.median(width90[mask])),
            "spearman_width68_abs_mean_error": float(width_corr),
            "mean_true_bin_probability": float(exact[mask].mean()),
            "mean_true_or_adjacent_probability": float(adjacent[mask].mean()),
            "closest_sample_median_error_gev": float(np.median(best_error[mask])),
            "closest_sample_mae_gev": float(best_error[mask].mean()),
            "closest_sample_rms_gev": float(np.sqrt(np.mean(np.square(best_error[mask])))),
            "closest_sample_sigma68_gev": _sigma68(best_residual),
            "best_of_M32_diagonal_fraction_truth_using": float(np.mean(_bin_ids(best_mass[mask], COARSE_EDGES) == true_bin[mask])),
            "best_of_M32_RMSE_truth_using_gev": float(np.sqrt(np.mean(np.square(best_residual)))),
            "mean_rank_samples": float(rank[mask].mean()),
        })
    quartile_id = np.minimum(3, np.searchsorted(np.quantile(width68, [0.25, 0.5, 0.75]), width68, side="right"))
    quartile_rows = []
    mean = samples.mean(axis=1)
    for q in range(4):
        mask = quartile_id == q
        quartile_rows.append({
            "width_quartile": q + 1, "events": int(mask.sum()),
            "mean_width68_gev": float(width68[mask].mean()),
            "mean_abs_mean_error_gev": float(np.abs(mean[mask] - truth[mask]).mean()),
            "mean_rmse_gev": float(np.sqrt(np.mean(np.square(mean[mask] - truth[mask])))),
        })
    return rows, rank_rows, {
        "q05": q05, "q16": q16, "q50": q50, "q84": q84, "q95": q95,
        "width68": width68, "width90": width90, "true_bin_probability": exact,
        "adjacent_probability": adjacent, "best_error": best_error,
        "best_index": best_index, "best_mass": best_mass, "rank_mid": rank,
        "width_quartile": quartile_id, "quartile_rows": quartile_rows,
    }


def _mse_decomposition(samples: np.ndarray, truth: np.ndarray, masks: dict) -> list[dict]:
    mean = samples.mean(axis=1)
    draw_mse = np.mean(np.square(samples - truth[:, None]), axis=1)
    center_mse = np.square(mean - truth)
    posterior_var = np.var(samples, axis=1, ddof=0)
    if not np.allclose(draw_mse, center_mse + posterior_var, rtol=1.0e-12, atol=1.0e-9):
        residual = float(np.max(np.abs(draw_mse - center_mse - posterior_var)))
        raise RuntimeError(f"Finite-M draw MSE identity failed; max event residual={residual}")
    rows = []
    for region, mask in masks.items():
        all_mse = float(draw_mse[mask].mean())
        centre, variance = float(center_mse[mask].mean()), float(posterior_var[mask].mean())
        if not math.isclose(all_mse, centre + variance, rel_tol=1.0e-12, abs_tol=1.0e-9):
            raise RuntimeError(f"MSE identity failed in truth region {region}")
        rows.append({
            "truth_region": region, "events": int(mask.sum()),
            "all_draw_mse_gev2": all_mse,
            "posterior_centre_mse_gev2": centre,
            "within_event_variance_gev2": variance,
            "centre_fraction": centre / all_mse,
            "variance_fraction": variance / all_mse,
            "identity_abs_residual_gev2": all_mse - centre - variance,
            "all_draw_rmse_gev": math.sqrt(all_mse),
            "posterior_centre_rmse_gev": math.sqrt(centre),
            "posterior_rms_width_gev": math.sqrt(variance),
        })
    return rows


def _nested_m(samples: np.ndarray, truth: np.ndarray, masks: dict) -> list[dict]:
    rows = []
    true_id = _bin_ids(truth, COARSE_EDGES)
    for count in (1, 2, 4, 8, 16, 32):
        prefix = samples[:, :count]
        mean, q50 = prefix.mean(axis=1), np.median(prefix, axis=1)
        best = np.min(np.abs(prefix - truth[:, None]), axis=1)
        sample_id = _bin_ids(prefix.reshape(-1), COARSE_EDGES).reshape(prefix.shape)
        width = np.quantile(prefix, 0.84, axis=1) - np.quantile(prefix, 0.16, axis=1)
        best_index = np.abs(prefix - truth[:, None]).argmin(axis=1)
        best_mass = prefix[np.arange(len(prefix)), best_index]
        best_bin = _bin_ids(best_mass, COARSE_EDGES)
        for region, mask in masks.items():
            rows.append({
                "M": count, "truth_region": region, "events": int(mask.sum()),
                "mean_rmse_gev": float(np.sqrt(np.mean(np.square(mean[mask] - truth[mask])))),
                "q50_rmse_gev": float(np.sqrt(np.mean(np.square(q50[mask] - truth[mask])))),
                "correct_bin_probability": float(np.mean(np.mean(sample_id[mask] == true_id[mask, None], axis=1))),
                "correct_or_adjacent_probability": float(np.mean(np.mean(np.abs(sample_id[mask] - true_id[mask, None]) <= 1, axis=1))),
                "best_of_M_mae_gev_truth_using": float(best[mask].mean()),
                "best_of_M_rmse_gev_truth_using": float(np.sqrt(np.mean(np.square(best[mask])))),
                "best_of_M_diagonal_fraction_truth_using": float(np.mean(best_bin[mask] == true_id[mask])),
                "mean_width68_gev": float(width[mask].mean()),
            })
    return rows


def _fold_ids(event_ids: np.ndarray, folds: int = 5) -> np.ndarray:
    values = np.asarray(event_ids, dtype=np.uint64).copy()
    with np.errstate(over="ignore"):
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values ^= values >> np.uint64(31)
    return (values % np.uint64(folds)).astype(np.int8)


def _crossfit_calibration(event_ids: np.ndarray, truth: np.ndarray, estimates: dict[str, np.ndarray], masks: dict) -> tuple[dict[str, np.ndarray], list[dict], dict]:
    folds = _fold_ids(event_ids, 5)
    if set(np.unique(folds).tolist()) != set(range(5)):
        raise RuntimeError("Deterministic EVENT_ID folding did not produce all five folds")
    calibrated: dict[str, np.ndarray] = {}
    fold_records = []
    rows = []
    for estimator in ("observable_mean", "observable_q50"):
        raw = np.asarray(estimates[estimator], dtype=np.float64)
        for kind in ("affine", "isotonic"):
            result = np.full(len(raw), np.nan, dtype=np.float64)
            for fold in range(5):
                test = folds == fold
                train = ~test
                if np.any(train & test) or not train.any() or not test.any():
                    raise RuntimeError(f"Cross-fit calibration fold {fold} leaks or is empty")
                if kind == "affine":
                    design = np.column_stack((np.ones(int(train.sum())), raw[train]))
                    coefficients = np.linalg.lstsq(design, truth[train], rcond=None)[0]
                    result[test] = coefficients[0] + coefficients[1] * raw[test]
                    fit_metadata = {"intercept_gev": float(coefficients[0]), "slope": float(coefficients[1])}
                else:
                    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
                    model.fit(raw[train], truth[train])
                    result[test] = model.predict(raw[test])
                    fit_metadata = {"threshold_count": int(len(model.X_thresholds_))}
                fold_records.append({
                    "estimator": estimator, "calibration": kind, "held_out_fold": fold,
                    "train_events": int(train.sum()), "held_out_events": int(test.sum()),
                    "held_out_fold_excluded_from_fit": True, **fit_metadata,
                })
            if not np.isfinite(result).all():
                raise RuntimeError(f"Cross-fit {kind} {estimator} predictions are incomplete")
            calibrated[f"crossfit_{kind}_{estimator}"] = result
            for region, mask in masks.items():
                rows.append({
                    "estimator": estimator, "calibration": kind, "truth_region": region,
                    "fit_protocol": "deterministic 5-fold EVENT_ID hash; report fold prediction only",
                    **_point_metrics(result, truth, mask),
                })
    # Every held-out prediction above belongs to one fold and all fitting sets excluded it.
    for record in fold_records:
        if record["held_out_fold_excluded_from_fit"] is not True:
            raise RuntimeError("Cross-fit leakage audit failed")
    if sum(item["held_out_events"] for item in fold_records if item["estimator"] == "observable_mean" and item["calibration"] == "affine") != len(event_ids):
        raise RuntimeError("Cross-fit fold predictions do not cover every event exactly once")
    manifest = {
        "folds": 5,
        "fold_hash": "SplitMix64(EVENT_ID) modulo 5",
        "fold_counts": {str(index): int(np.sum(folds == index)) for index in range(5)},
        "fit_scope": "validation diagnostic only; each prediction uses a fit excluding its full EVENT_ID fold",
        "calibrators": {
            "affine": "ordinary least squares truth = intercept + slope * raw estimate",
            "isotonic": "sklearn IsotonicRegression(increasing=True, out_of_bounds=clip)",
        },
        "leakage_checked": True,
    }
    return calibrated, rows, {"manifest": manifest, "fold_records": fold_records}


def _histogram(values: np.ndarray, edges: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.histogram(values, bins=edges, weights=weights)[0].astype(np.float64)


def _spectrum_metrics(truth: np.ndarray, prediction: np.ndarray, edges: np.ndarray,
                      *, truth_weights: np.ndarray | None = None,
                      prediction_weights: np.ndarray | None = None) -> dict:
    h_truth = _histogram(truth, edges, truth_weights)
    h_prediction = _histogram(prediction, edges, prediction_weights)
    p = h_truth / max(h_truth.sum(), 1.0)
    q = h_prediction / max(h_prediction.sum(), 1.0)
    total_variation = 0.5 * float(np.abs(p - q).sum())
    shape_chi2 = float(np.sum(np.square(p - q) / np.maximum(p + q, 1.0e-15)))
    return {
        "truth_weight": float(h_truth.sum()),
        "reco_weight": float(h_prediction.sum()),
        "wasserstein1_gev": float(wasserstein_distance(truth, prediction, u_weights=truth_weights, v_weights=prediction_weights)),
        "total_variation": total_variation,
        "ks_statistic": float(ks_2samp(truth, prediction, method="auto").statistic),
        "shape_chi2_like": shape_chi2,
        "truth_histogram": h_truth,
        "reco_histogram": h_prediction,
        "truth_probability": p,
        "reco_probability": q,
    }


def _spectra(data: dict, estimates: dict[str, np.ndarray], matched: np.ndarray) -> tuple[list[dict], list[dict], dict]:
    truth = np.asarray(data["truth_mttbar"], dtype=np.float64)
    samples = np.asarray(data["sample_mttbar"], dtype=np.float64)
    all_values = [truth, samples.reshape(-1), *[np.asarray(v) for v in estimates.values()]]
    perfect = np.asarray(data["perfect_reco_mttbar"], dtype=np.float64)
    all_values.append(perfect[matched])
    finite_max = max(float(np.max(v)) for v in all_values if len(v))
    upper = max(1200.0, math.ceil(finite_max / 100.0) * 100.0)
    full_edges = np.concatenate((np.linspace(0.0, upper, 101), [np.inf]))
    threshold_edges = np.arange(300.0, 555.0, 5.0)
    tail_edges = np.concatenate((np.arange(700.0, max(upper + 1.0, 800.0), 100.0), [np.inf]))
    if tail_edges[0] != 700.0:
        tail_edges = np.asarray([700.0, np.inf])
    records, bin_records = [], []
    values_by_name = {
        "sample0": estimates["sample0"],
        "observable_mean": estimates["observable_mean"],
        "observable_q50": estimates["observable_q50"],
        "joint_medoid": estimates["joint_medoid"],
        "crossfit_affine_observable_mean": estimates["crossfit_affine_observable_mean"],
        "crossfit_isotonic_observable_mean": estimates["crossfit_isotonic_observable_mean"],
        "crossfit_affine_observable_q50": estimates["crossfit_affine_observable_q50"],
        "crossfit_isotonic_observable_q50": estimates["crossfit_isotonic_observable_q50"],
        "all_samples_weighted_1_over_M": samples.reshape(-1),
        "posterior_predictive_matched_support": samples[matched].reshape(-1),
        "sample0_matched_perfect_support": estimates["sample0"][matched],
        "posterior_mean_matched_perfect_support": estimates["observable_mean"][matched],
        "posterior_q50_matched_perfect_support": estimates["observable_q50"][matched],
        "joint_medoid_matched_perfect_support": estimates["joint_medoid"][matched],
        "perfect_reco_matched_support": perfect[matched],
    }
    for name, prediction in values_by_name.items():
        matched_scope = name.endswith("_matched_perfect_support") or name == "posterior_predictive_matched_support"
        target = truth[matched] if matched_scope else truth
        pred = prediction
        if matched_scope:
            scopes = (("full", target, pred, full_edges, None, None),)
        else:
            scopes = (
                ("full", target, pred, full_edges, None, None),
                ("threshold_300_550", target[(target >= 300.0) & (target < 550.0)],
                 pred[(pred >= 300.0) & (pred < 550.0)], threshold_edges, None, None),
                ("tail_ge_700", target[target >= 700.0], pred[pred >= 700.0], tail_edges, None, None),
            )
        for scope, scoped_truth, scoped_pred, bins, _, _ in scopes:
            if len(scoped_truth) == 0 or len(scoped_pred) == 0:
                continue
            # Each event contributes unit total mass to the all-draw spectrum.
            if name in ("all_samples_weighted_1_over_M", "posterior_predictive_matched_support"):
                tw = None
                pw = np.full(len(scoped_pred), 1.0 / EXPECTED_SAMPLES, dtype=np.float64)
            else:
                tw = pw = None
            metrics = _spectrum_metrics(scoped_truth, scoped_pred, bins,
                                        truth_weights=tw, prediction_weights=pw)
            truth_center = float(np.mean(scoped_truth))
            reco_center = float(np.average(scoped_pred, weights=pw)) if pw is not None else float(np.mean(scoped_pred))
            truth_width = float(np.sqrt(np.mean(np.square(scoped_truth - truth_center))))
            reco_width = float(np.sqrt(np.average(np.square(scoped_pred - reco_center), weights=pw))) if pw is not None else float(np.std(scoped_pred))
            records.append({
                "estimator": name, "scope": scope,
                "truth_events_in_scope": int(len(scoped_truth)),
                "reco_entries_in_scope": int(len(scoped_pred)),
                "truth_acceptance": float(len(scoped_truth) / max(len(target), 1)),
                "reco_acceptance": float(len(scoped_pred) / max(len(pred), 1)),
                "wasserstein1_gev": metrics["wasserstein1_gev"],
                "total_variation": metrics["total_variation"],
                "ks_statistic": metrics["ks_statistic"],
                "shape_chi2_like": metrics["shape_chi2_like"],
                "truth_population_std_gev": truth_width,
                "reco_population_std_gev": reco_width,
                "reco_to_truth_population_width_ratio": reco_width / truth_width if truth_width > 0 else float("nan"),
                "truth_event_equivalent_weight": metrics["truth_weight"],
                "reco_event_equivalent_weight": metrics["reco_weight"],
            })
            labels = _bin_labels(bins)
            ratio = np.divide(metrics["reco_histogram"], metrics["truth_histogram"],
                              out=np.full(len(labels), np.nan), where=metrics["truth_histogram"] > 0.0)
            for index, label in enumerate(labels):
                bin_records.append({
                    "estimator": name, "scope": scope, "mass_bin": label,
                    "truth_weight": float(metrics["truth_histogram"][index]),
                    "reco_weight": float(metrics["reco_histogram"][index]),
                    "reco_over_truth": float(ratio[index]),
                    "truth_probability": float(metrics["truth_probability"][index]),
                    "reco_probability": float(metrics["reco_probability"][index]),
                })
    # Predictive spectrum uses the same full truth population and fractional event weights.
    flat_samples = samples.reshape(-1)
    sample_weights = np.full(len(flat_samples), 1.0 / samples.shape[1], dtype=np.float64)
    pp = _spectrum_metrics(truth, flat_samples, full_edges, prediction_weights=sample_weights)
    ppc = {
        "edges": full_edges,
        "truth_histogram": pp["truth_histogram"],
        "reco_histogram": pp["reco_histogram"],
        "truth_probability": pp["truth_probability"],
        "reco_probability": pp["reco_probability"],
        "metrics": {
            key: pp[key] for key in ("wasserstein1_gev", "total_variation", "ks_statistic", "shape_chi2_like")
        },
    }
    ppc_bins = _bin_labels(full_edges)
    for index, label in enumerate(ppc_bins):
        bin_records.append({
            "estimator": "posterior_predictive_weighted_1_over_M", "scope": "full",
            "mass_bin": label, "truth_weight": float(pp["truth_histogram"][index]),
            "reco_weight": float(pp["reco_histogram"][index]),
            "reco_over_truth": float(pp["reco_histogram"][index] / pp["truth_histogram"][index])
            if pp["truth_histogram"][index] > 0 else float("nan"),
            "truth_probability": float(pp["truth_probability"][index]),
            "reco_probability": float(pp["reco_probability"][index]),
        })
    expected_weight = len(truth)
    if not math.isclose(float(sample_weights.sum()), expected_weight, rel_tol=0.0, abs_tol=1.0e-8):
        raise RuntimeError("All-posterior-sample spectrum does not have one event-equivalent weight per event")
    return records, bin_records, {"full_edges": full_edges, "threshold_edges": threshold_edges,
                                 "tail_edges": tail_edges, "posterior_predictive": ppc,
                                 "bin_records": bin_records}


def _oracle_metrics(data: dict, masks: dict) -> tuple[list[dict], list[dict]]:
    truth = np.asarray(data["truth_mttbar"], dtype=np.float64)
    metric_rows, matrix_rows = [], []
    for name, mass in data["oracles"].items():
        finite = np.isfinite(mass)
        for region, mask in masks.items():
            common = mask & finite
            metric_rows.append({
                "reference": name, "definition": ORACLE_PATHS[name], "truth_region": region,
                "population_events": int(mask.sum()), "finite_support_events": int(common.sum()),
                "finite_support_fraction": float(common.sum() / max(mask.sum(), 1)),
                **_point_metrics(mass, truth, common),
            })
        common = finite
        matrix = _migration_matrix(truth[common], mass[common], COARSE_EDGES)
        matrix_rows.extend(_matrix_csv_rows(name, "coarse", matrix, COARSE_EDGES))
    return metric_rows, matrix_rows


def _bootstrap_headlines(truth: np.ndarray, estimates: dict[str, np.ndarray], perfect: np.ndarray,
                         perfect_mask: np.ndarray, *, replicates: int = 500, seed: int = 20260929) -> list[dict]:
    rng = np.random.default_rng(seed)
    comparisons = [
        ("observable_mean", "sample0", np.arange(len(truth))),
        ("observable_q50", "sample0", np.arange(len(truth))),
        ("observable_q50", "observable_mean", np.arange(len(truth))),
        ("observable_mean", "perfect_reco", np.flatnonzero(perfect_mask)),
    ]
    differences = {f"{left}_minus_{right}": np.empty(replicates, dtype=np.float64)
                   for left, right, _ in comparisons}
    for rep in range(replicates):
        for left, right, population in comparisons:
            chosen = rng.integers(0, len(population), size=len(population))
            rows = population[chosen]
            left_values = estimates[left][rows]
            right_values = perfect[rows] if right == "perfect_reco" else estimates[right][rows]
            y = truth[rows]
            differences[f"{left}_minus_{right}"][rep] = (
                np.sqrt(np.mean(np.square(left_values - y)))
                - np.sqrt(np.mean(np.square(right_values - y)))
            )
    result = []
    for left, right, population in comparisons:
        key = f"{left}_minus_{right}"
        values = differences[key]
        rows = population
        left_values = estimates[left][rows]
        right_values = perfect[rows] if right == "perfect_reco" else estimates[right][rows]
        y = truth[rows]
        estimate = float(np.sqrt(np.mean(np.square(left_values - y))) - np.sqrt(np.mean(np.square(right_values - y))))
        result.append({
            "contrast": key, "events": int(len(rows)), "replicates": replicates,
            "paired_rmse_difference_gev": estimate,
            "bootstrap_2p5_gev": float(np.quantile(values, 0.025)),
            "bootstrap_97p5_gev": float(np.quantile(values, 0.975)),
            "seed": seed,
        })
    return result


def _posterior_probability_rows(predictive: dict, truth: np.ndarray, masks: dict,
                               binning: str, edges: np.ndarray) -> list[dict]:
    event_p = predictive["event_probabilities"]
    true_id = _bin_ids(truth, edges)
    rows = []
    for region, mask in masks.items():
        p_correct = event_p[np.arange(len(truth)), true_id]
        p_adjacent = np.zeros(len(truth), dtype=np.float64)
        for shift in (-1, 0, 1):
            index = true_id + shift
            valid = (index >= 0) & (index < event_p.shape[1])
            p_adjacent[valid] += event_p[np.arange(len(truth))[valid], index[valid]]
        target = np.zeros_like(event_p)
        target[np.arange(len(truth)), true_id] = 1.0
        brier = np.sum(np.square(event_p - target), axis=1)
        entropy = -np.sum(np.where(event_p > 0.0, event_p * np.log(np.maximum(event_p, 1.0e-300)), 0.0), axis=1)
        rows.append({
            "record_type": "summary", "binning": binning, "truth_region": region,
            "events": int(mask.sum()),
            "mean_correct_bin_probability": float(p_correct[mask].mean()),
            "mean_correct_or_adjacent_probability": float(p_adjacent[mask].mean()),
            "events_with_any_correct_bin_sample": int(np.sum(np.any(event_p[mask] > 0.0, axis=1))),
            "fraction_events_with_any_correct_bin_sample": float(np.mean(np.any(event_p[mask] > 0.0, axis=1))),
            "categorical_brier_score": float(brier[mask].mean()),
            "posterior_entropy_nats": float(entropy[mask].mean()),
            "mean_probability_max": float(event_p[mask].max(axis=1).mean()),
            "convention": "each event assigns 1/M posterior mass to its M sample bins",
        })
    return rows


def _migration_summary_rows(name: str, binning: str, migration: dict, edges: np.ndarray) -> list[dict]:
    labels = _bin_labels(edges, coarse=binning == "coarse")
    matrix = migration["row"]
    counts = migration["counts"]
    rows = [{
        "record_type": "global", "estimator": name, "binning": binning,
        "truth_bin": "all", "reco_bin": "all", "events": int(counts.sum()),
        "diagonal_fraction": migration["diagonal_fraction"],
        "adjacent_fraction": migration["adjacent_fraction"],
        "mean_abs_bin_displacement": migration["mean_abs_bin_displacement"],
        "rms_bin_displacement": migration["rms_bin_displacement"],
        "mean_correct_probability": migration["mean_correct_probability"],
    }]
    for index, label in enumerate(labels):
        adjacent = matrix[index, max(0, index - 1):min(len(labels), index + 2)].sum()
        rows.append({
            "record_type": "truth_row", "estimator": name, "binning": binning,
            "truth_bin": label, "reco_bin": "all", "events": int(counts[index]),
            "diagonal_fraction": float(matrix[index, index]),
            "adjacent_fraction": float(adjacent),
            "mean_abs_bin_displacement": float(np.sum(np.abs(np.arange(len(labels)) - index) * matrix[index])),
            "rms_bin_displacement": float(np.sqrt(np.sum(np.square(np.arange(len(labels)) - index) * matrix[index]))),
            "truth_row_entropy_nats": float(migration["row_entropy"][index]),
        })
    for index, label in enumerate(labels):
        rows.append({
            "record_type": "reco_column", "estimator": name, "binning": binning,
            "truth_bin": "all", "reco_bin": label, "events": int(migration["matrix"][:, index].sum()),
            "reco_column_purity": float(migration["column_purity"][index]),
        })
    return rows


def _coherent_reducer_metrics(data: dict, extras: dict, masks: dict) -> list[dict]:
    truth_observables = _observables(np.asarray(data["truth"]["parents"])[:, None])
    reducers = {
        "joint_medoid": extras["medoid_parent"],
        "coherent_nearest_q50": extras["nearest_q50_parent"],
        "coherent_nearest_mean": extras["nearest_mean_parent"],
    }
    rows = []
    for name, parents in reducers.items():
        observed = _observables(parents[:, None])
        for observable in PARENT_OBSERVABLES:
            truth_values = np.asarray(truth_observables[observable])[:, 0]
            predicted = np.asarray(observed[observable])[:, 0]
            for region, mask in masks.items():
                y, x = truth_values[mask], predicted[mask]
                residual = x - y
                if observable.endswith("_phi"):
                    residual = np.remainder(residual + np.pi, 2.0 * np.pi) - np.pi
                rows.append({
                    "reducer": name, "observable": observable, "truth_region": region,
                    "events": int(mask.sum()), "bias": float(residual.mean()),
                    "std_residual": float(residual.std()), "sigma68": _sigma68(residual),
                    "mae": float(np.abs(residual).mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                    "unit": "rad" if observable.endswith("_phi") else "GeV or dimensionless rapidity",
                })
    return rows


def _failure_stratification(data: dict, estimates: dict[str, np.ndarray], extras: dict,
                            posterior: dict, masks: dict) -> list[dict]:
    truth = np.asarray(data["truth_mttbar"], dtype=np.float64)
    q50 = estimates["observable_q50"]
    region_summaries = []
    for region in COARSE_LABELS:
        mask = masks[region]
        if mask.sum() >= 100:
            true_id = _bin_ids(truth[mask], COARSE_EDGES)
            reco_id = _bin_ids(q50[mask], COARSE_EDGES)
            region_summaries.append((float(np.mean(true_id != reco_id)), region, mask))
    selected = [item[1:] for item in sorted(region_summaries, reverse=True)[:2]]
    groups: dict[str, np.ndarray] = {
        "source_njet": np.asarray(data["truth"]["source_njet"]),
        "model_visible_njet": np.asarray(data["truth"]["model_visible_njet"]),
        "channel": np.asarray([_decode(value) for value in data["truth"]["channel"]], dtype=str),
        "posterior_width68_quartile": (posterior["width_quartile"] + 1).astype(str),
        "assignment_supported": data["assignment"]["assignment_supported"].astype(str),
        "assignment_complete_correct": data["assignment"]["assignment_complete_correct"].astype(str),
        "assignment_roles_supported": data["assignment"]["assignment_role_supported"].astype(str),
        "assignment_roles_correct": data["assignment"]["assignment_role_correct"].astype(str),
    }
    perfect = np.asarray(data["perfect_reco_mttbar"], dtype=np.float64)
    perfect_error = np.abs(perfect - truth)
    perfect_q = np.full(len(truth), "unsupported", dtype=object)
    finite = np.isfinite(perfect_error)
    if finite.any():
        cuts = np.quantile(perfect_error[finite], [0.25, 0.5, 0.75])
        perfect_q[finite] = (np.searchsorted(cuts, perfect_error[finite], side="right") + 1).astype(str)
    groups["perfect_reco_error_quartile"] = perfect_q
    neutrino = np.asarray(data["oracles"]["truth_assignment_predicted_neutrinos"], dtype=np.float64)
    neutrino_error = np.abs(neutrino - truth)
    neutrino_q = np.full(len(truth), "unsupported", dtype=object)
    finite_nu = np.isfinite(neutrino_error)
    if finite_nu.any():
        cuts = np.quantile(neutrino_error[finite_nu], [0.25, 0.5, 0.75])
        neutrino_q[finite_nu] = (np.searchsorted(cuts, neutrino_error[finite_nu], side="right") + 1).astype(str)
    groups["truth_assignment_predicted_neutrino_mass_error_quartile"] = neutrino_q
    rows = []
    for region, region_mask in selected:
        for category, values in groups.items():
            for value in np.unique(values[region_mask]):
                mask = region_mask & (values == value)
                if mask.sum() < 25:
                    continue
                metrics = _point_metrics(q50, truth, mask)
                rows.append({"truth_region": region, "stratifier": category,
                             "category": str(value), **metrics})
    return rows


def _regression_gate(data: dict, summary: dict, estimates: dict[str, np.ndarray]) -> dict:
    reference = data["reference"]
    array_names = {
        "sample0": "sample0",
        "observable_mean": "observable_mean",
        "observable_q50": "observable_q50",
        "joint_medoid": "joint_medoid",
    }
    summary_names = {"sample0": "single_draw", **{name: name for name in array_names if name != "sample0"}}
    comparisons = {}
    for estimate_name, array_name in array_names.items():
        difference = np.abs(estimates[estimate_name] - reference[array_name])
        maximum = float(np.max(difference))
        if maximum > 1.0e-3:
            raise RuntimeError(
                f"Mass cache regression failed for {estimate_name}: max difference {maximum:.6g} GeV"
            )
        comparisons[estimate_name] = {
            "max_abs_difference_from_accepted_reduction_gev": maximum,
            "passed_tolerance_gev": 1.0e-3,
        }
    expected_values = {
        "sample0": EXPECTED_MASS_RMSE["single_draw"],
        "observable_mean": EXPECTED_MASS_RMSE["observable_mean"],
        "observable_q50": EXPECTED_MASS_RMSE["observable_q50"],
    }
    observed = {}
    for estimator, expected in expected_values.items():
        reduced_name = summary_names[estimator]
        accepted_rmse = float(summary["metrics"][reduced_name]["mttbar"]["inclusive"]["rmse"])
        actual = float(np.sqrt(np.mean(np.square(estimates[estimator] - data["truth_mttbar"]))))
        if not math.isclose(actual, accepted_rmse, rel_tol=0.0, abs_tol=1.0e-6):
            raise RuntimeError(
                f"Derived {estimator} RMSE {actual:.9f} differs from accepted metric {accepted_rmse:.9f}"
            )
        if abs(actual - expected) > 0.5:
            raise RuntimeError(f"Derived {estimator} RMSE is outside the established accepted value")
        observed[estimator] = {
            "rmse_gev": actual,
            "accepted_reduction_rmse_gev": accepted_rmse,
            "established_approximation_gev": expected,
            "difference_from_accepted_gev": actual - accepted_rmse,
        }
    return {"status": "passed", "array_comparisons": comparisons, "rmse": observed}


def _run_smoke(data: dict, output_dir: Path, provenance: dict) -> dict:
    truth = np.asarray(data["truth_mttbar"], dtype=np.float64)
    samples = np.asarray(data["sample_mttbar"], dtype=np.float64)
    ids = np.asarray(data["EVENT_ID"], dtype=np.int64)
    if samples.shape != (len(ids), EXPECTED_SAMPLES) or not np.isfinite(samples).all():
        raise RuntimeError("Smoke failed M32 mass-cache shape or finite-value check")
    estimates, extras = _build_estimates(data)
    masks = {name: mask for name, mask in _region_masks(truth).items() if mask.any()}
    boundary_values = np.asarray([344.999, 345.0, 400.0, 500.0, 700.0, 1000.0, 1000.001])
    if not np.array_equal(_bin_ids(boundary_values, COARSE_EDGES), np.asarray([0, 1, 2, 3, 4, 4, 5])):
        raise RuntimeError("Smoke failed exact physics-region boundary semantics")
    mse = _mse_decomposition(samples, truth, masks)
    coarse = _posterior_predictive(samples, truth, COARSE_EDGES)
    fine_edges = _fine_edges()
    fine = _posterior_predictive(samples, truth, fine_edges)
    if not np.allclose(coarse["event_probabilities"].sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Smoke failed coarse posterior probability normalisation")
    if not np.allclose(fine["event_probabilities"].sum(axis=1), 1.0, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Smoke failed fine posterior probability normalisation")
    migration = _migration_matrix(truth, estimates["observable_mean"], COARSE_EDGES)
    if not np.allclose(migration["row"].sum(axis=1)[migration["counts"] > 0], 1.0):
        raise RuntimeError("Smoke failed hard migration row normalisation")
    if not np.allclose(coarse["row"].sum(axis=1)[coarse["counts"] > 0], 1.0):
        raise RuntimeError("Smoke failed predictive migration row normalisation")
    best = np.abs(samples - truth[:, None]).argmin(axis=1)
    best_values = samples[np.arange(len(samples)), best]
    if not np.allclose(np.abs(best_values - truth), np.min(np.abs(samples - truth[:, None]), axis=1), rtol=0.0, atol=1.0e-12):
        raise RuntimeError("Smoke failed best-of-M selection consistency")
    nested = _nested_m(samples, truth, masks)
    if {row["M"] for row in nested} != {1, 2, 4, 8, 16, 32}:
        raise RuntimeError("Smoke failed nested-M prefix coverage")
    perfect_mask = np.asarray(data["perfect_reco_mask"], dtype=bool)
    if not np.array_equal(perfect_mask, np.isfinite(data["perfect_reco_mttbar"])):
        raise RuntimeError("Smoke Perfect-reco finite support mask is inconsistent")
    folds = _fold_ids(ids, 5)
    if set(np.unique(folds).tolist()) != set(range(5)):
        raise RuntimeError("Smoke failed deterministic cross-fit fold assignment")
    calibration, _, details = _crossfit_calibration(ids, truth, estimates, masks)
    if any(not np.isfinite(value).all() for value in calibration.values()):
        raise RuntimeError("Smoke cross-fit predictions contain non-finite values")
    result = {
        "status": "passed",
        "events": int(len(ids)),
        "samples": int(samples.shape[1]),
        "event_id_order_checked": True,
        "mass_decoding": "maintained _observables() from parent four-vectors",
        "coarse_migration_rows_normalised": True,
        "posterior_probability_event_sums_max_abs_error": float(np.max(np.abs(coarse["event_probabilities"].sum(axis=1) - 1.0))),
        "fine_posterior_probability_event_sums_max_abs_error": float(np.max(np.abs(fine["event_probabilities"].sum(axis=1) - 1.0))),
        "all_sample_mass_weight": float(len(ids)),
        "best_of_m32_selection_checked": True,
        "nested_M_prefixes": [1, 2, 4, 8, 16, 32],
        "perfect_reco_finite_support": int(perfect_mask.sum()),
        "crossfit_folds": details["manifest"]["fold_counts"],
        "crossfit_leakage_checked": details["manifest"]["leakage_checked"],
        "mse_identity_max_abs_residual_gev2": max(abs(row["identity_abs_residual_gev2"]) for row in mse),
        "provenance": provenance,
    }
    _json_write(output_dir / "smoke/smoke.json", result)
    return result


def _save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_migration(matrix: dict, edges: np.ndarray, target: Path, title: str,
                    *, diverging: bool = False) -> None:
    labels = _bin_labels(edges, coarse=True)
    counts = matrix["counts"]
    tick_labels = [f"{label}\nN={int(count):,}" for label, count in zip(labels, counts, strict=True)]
    values = matrix["row"]
    fig, ax = plt.subplots(figsize=(8.4, 7.2), constrained_layout=True)
    if diverging:
        limit = max(float(np.max(np.abs(values))), 1.0e-3)
        image = ax.imshow(values, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit))
    else:
        image = ax.imshow(values, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(labels)), tick_labels)
    ax.set_xlabel("Reconstructed mttbar bin")
    ax.set_ylabel("Truth mttbar bin (truth count)")
    ax.set_title(title)
    if len(labels) <= 10:
        for i in range(len(labels)):
            for j in range(len(labels)):
                color = "white" if abs(values[i, j]) > 0.55 else "black"
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, label="P(reco bin | truth bin)" if not diverging else "Q mean − Perfect reco")
    _save(fig, target)


def _plot_response(differential: list[dict], output_dir: Path) -> None:
    primary = ("sample0", "observable_mean", "observable_q50", "joint_medoid")
    colors = {"sample0": "#777777", "observable_mean": "#0072B2", "observable_q50": "#D55E00", "joint_medoid": "#009E73"}
    names = {"sample0": "Sample 0", "observable_mean": "Posterior mean", "observable_q50": "Posterior q50", "joint_medoid": "Joint medoid"}
    fig, ax = plt.subplots(figsize=(8.2, 6.8), constrained_layout=True)
    low = min(row["truth_mean_gev"] for row in differential if row["estimator"] in primary)
    high = max(row["truth_mean_gev"] for row in differential if row["estimator"] in primary)
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=1.2, label="y = x")
    for estimator in primary:
        selected = [row for row in differential if row["estimator"] == estimator and row["events"] >= 40]
        x = [row["truth_mean_gev"] for row in selected]
        y = [row["reco_mean_gev"] for row in selected]
        ax.plot(x, y, marker="o", markersize=3.3, linewidth=1.2, color=colors[estimator], label=names[estimator])
    ax.set(xlabel="Truth mttbar [GeV]", ylabel="Mean reconstructed mttbar [GeV]", title="Mttbar response in fine truth bins")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "response/mttbar_response_vs_truth")

    for metric, target, ylabel, title in (
        ("bias_gev", "mttbar_bias_vs_truth", "Mean response bias [GeV]", "Mttbar bias by truth mass"),
        ("sigma68_gev", "mttbar_sigma68_vs_truth", "Residual sigma68 [GeV]", "Mttbar central residual width by truth mass"),
        ("rmse_gev", "mttbar_rmse_vs_truth", "RMSE [GeV]", "Mttbar RMSE by truth mass"),
    ):
        fig, ax = plt.subplots(figsize=(8.2, 5.6), constrained_layout=True)
        for estimator in primary:
            selected = [row for row in differential if row["estimator"] == estimator and row["events"] >= 40]
            ax.plot([row["truth_mean_gev"] for row in selected], [row[metric] for row in selected],
                    marker="o", markersize=3.2, linewidth=1.2, color=colors[estimator], label=names[estimator])
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--") if metric == "bias_gev" else None
        ax.set(xlabel="Truth mttbar [GeV]", ylabel=ylabel, title=title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        _save(fig, output_dir / f"response/{target}")


def _plot_spectra(data: dict, estimates: dict[str, np.ndarray], spectra: dict, output_dir: Path) -> None:
    truth = np.asarray(data["truth_mttbar"], dtype=np.float64)
    samples = np.asarray(data["sample_mttbar"], dtype=np.float64)
    perfect = np.asarray(data["perfect_reco_mttbar"], dtype=np.float64)
    perfect_mask = np.asarray(data["perfect_reco_mask"], dtype=bool)
    ppc = spectra["posterior_predictive"]
    full_edges = spectra["full_edges"]

    def draw_spectrum(edges: np.ndarray, path: Path, title: str, threshold: bool = False) -> None:
        fig, (axis, ratio) = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True,
                                          gridspec_kw={"height_ratios": (3.0, 1.0)}, constrained_layout=True)
        truth_hist = _histogram(truth, edges)
        axis.stairs(truth_hist, edges, color="black", linewidth=2.0, label="Parton truth")
        axis.stairs(_histogram(truth[perfect_mask], edges), edges, color="#666666", linestyle="--", label="Truth on Perfect-reco support")
        axis.stairs(_histogram(perfect[perfect_mask], edges), edges, color="#666666", linewidth=1.3, label="Perfect reco")
        weights = np.full(samples.size, 1.0 / samples.shape[1], dtype=np.float64)
        curves = [
            ("All posterior samples, weight 1/32", samples.reshape(-1), weights, "#0072B2"),
            ("Sample 0", estimates["sample0"], None, "#777777"),
            ("Posterior mean", estimates["observable_mean"], None, "#D55E00"),
            ("Posterior q50", estimates["observable_q50"], None, "#009E73"),
            ("Joint medoid", estimates["joint_medoid"], None, "#CC79A7"),
        ]
        for label, values, curve_weights, color in curves:
            axis.stairs(_histogram(values, edges, curve_weights), edges, color=color, linewidth=1.4, label=label)
        centers = 0.5 * (edges[:-1] + edges[1:])
        denominator = _histogram(truth, edges)
        ratio_curves = []
        for label, values, curve_weights, color in curves:
            numerator = _histogram(values, edges, curve_weights)
            value = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0.0)
            ratio_curves.append((label, value, color))
        perfect_denominator = _histogram(truth[perfect_mask], edges)
        perfect_ratio = np.divide(_histogram(perfect[perfect_mask], edges), perfect_denominator,
                                  out=np.full_like(perfect_denominator, np.nan), where=perfect_denominator > 0.0)
        finite_ratio_values = [values[np.isfinite(values)] for _, values, _ in ratio_curves]
        finite_ratio_values.append(perfect_ratio[np.isfinite(perfect_ratio)])
        joined_ratios = np.concatenate([values for values in finite_ratio_values if len(values)])
        ratio_upper = max(2.0, float(np.quantile(joined_ratios, 0.98)) * 1.15) if len(joined_ratios) else 2.0
        for label, value, color in ratio_curves:
            ratio.plot(centers, value, color=color, linewidth=1.2, label=label)
            clipped = np.isfinite(value) & (value > ratio_upper)
            if clipped.any():
                ratio.plot(centers[clipped], np.full(int(clipped.sum()), ratio_upper * 0.97),
                           linestyle="none", marker="^", color=color, clip_on=False,
                           label=f"{label} above plot range ({int(clipped.sum())} bins)")
        ratio.plot(centers, perfect_ratio, color="#666666", linestyle="--", label="Perfect reco / matched truth")
        clipped_perfect = np.isfinite(perfect_ratio) & (perfect_ratio > ratio_upper)
        if clipped_perfect.any():
            ratio.plot(centers[clipped_perfect], np.full(int(clipped_perfect.sum()), ratio_upper * 0.97),
                       linestyle="none", marker="^", color="#666666", clip_on=False,
                       label=f"Perfect reco above plot range ({int(clipped_perfect.sum())} bins)")
        ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        ratio.set_ylim(0.0, ratio_upper)
        axis.set_yscale("log")
        axis.set_ylabel("Events / bin")
        ratio.set_ylabel("Reco / truth")
        ratio.set_xlabel("Truth or reconstructed mttbar [GeV]")
        axis.set_title(title)
        axis.legend(frameon=False, ncol=2, fontsize=8)
        ratio.grid(axis="y", alpha=0.25)
        if threshold:
            axis.set_xlim(300.0, 550.0)
        else:
            axis.set_xlim(0.0, float(edges[-2]))
        _save(fig, path)

    finite_full_edges = full_edges[:-1]
    draw_spectrum(finite_full_edges, output_dir / "spectra/mttbar_spectrum_full", "Full validation mttbar spectrum; Perfect reco uses matched truth support")
    draw_spectrum(spectra["threshold_edges"], output_dir / "spectra/mttbar_spectrum_threshold", "Threshold mttbar spectrum: 300–550 GeV, 5 GeV bins", threshold=True)

    edges = finite_full_edges
    fig, axis = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    truth_hist = _histogram(truth, edges)
    ppc_hist = _histogram(samples.reshape(-1), edges, np.full(samples.size, 1.0 / samples.shape[1]))
    axis.stairs(truth_hist, edges, color="black", linewidth=2.0, label="Parton truth")
    axis.stairs(ppc_hist, edges, color="#0072B2", linewidth=1.7, label="Posterior predictive, 1/32 per draw")
    axis.set_yscale("log")
    axis.set(xlabel="mttbar [GeV]", ylabel="Events / bin", title="Posterior predictive mttbar spectrum")
    axis.legend(frameon=False)
    _save(fig, output_dir / "spectra/mttbar_spectrum_posterior_predictive")

    matched_truth = truth[perfect_mask]
    matched_curves = [
        ("Perfect reco", perfect[perfect_mask], None, "#666666"),
        ("Sample 0", estimates["sample0"][perfect_mask], None, "#777777"),
        ("Posterior mean", estimates["observable_mean"][perfect_mask], None, "#D55E00"),
        ("Posterior q50", estimates["observable_q50"][perfect_mask], None, "#009E73"),
        ("All posterior samples, weight 1/32", samples[perfect_mask].reshape(-1),
         np.full(int(perfect_mask.sum()) * samples.shape[1], 1.0 / samples.shape[1]), "#0072B2"),
    ]
    fig, (axis, ratio) = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True,
                                      gridspec_kw={"height_ratios": (3.0, 1.0)}, constrained_layout=True)
    matched_truth_hist = _histogram(matched_truth, finite_full_edges)
    axis.stairs(matched_truth_hist, finite_full_edges, color="black", linewidth=2.0,
                label=f"Truth on Perfect-reco support (N={int(perfect_mask.sum()):,})")
    for label, values, curve_weights, color in matched_curves:
        axis.stairs(_histogram(values, finite_full_edges, curve_weights), finite_full_edges,
                    color=color, linewidth=1.4, label=label)
    centers = 0.5 * (finite_full_edges[:-1] + finite_full_edges[1:])
    ratios = []
    for label, values, curve_weights, color in matched_curves:
        numerator = _histogram(values, finite_full_edges, curve_weights)
        value = np.divide(numerator, matched_truth_hist, out=np.full_like(numerator, np.nan),
                          where=matched_truth_hist > 0.0)
        ratios.append((label, value, color))
    finite_ratios = [value[np.isfinite(value)] for _, value, _ in ratios]
    joined = np.concatenate([value for value in finite_ratios if len(value)])
    ratio_upper = max(2.0, float(np.quantile(joined, 0.98)) * 1.15) if len(joined) else 2.0
    for label, value, color in ratios:
        ratio.plot(centers, value, color=color, linewidth=1.2, label=label)
        clipped = np.isfinite(value) & (value > ratio_upper)
        if clipped.any():
            ratio.plot(centers[clipped], np.full(int(clipped.sum()), ratio_upper * 0.97),
                       linestyle="none", marker="^", color=color, clip_on=False,
                       label=f"{label} above range ({int(clipped.sum())} bins)")
    ratio.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ratio.set_ylim(0.0, ratio_upper)
    axis.set_yscale("log")
    axis.set_ylabel("Events / bin")
    ratio.set_ylabel("Reco / matched truth")
    ratio.set_xlabel("mttbar [GeV]")
    axis.set_title("Mttbar spectra on identical finite Perfect-reco support")
    axis.legend(frameon=False, ncol=2, fontsize=8)
    ratio.grid(axis="y", alpha=0.25)
    axis.set_xlim(0.0, float(finite_full_edges[-1]))
    _save(fig, output_dir / "spectra/mttbar_spectrum_perfect_matched_support")

    tail_rows = [row for row in spectra["bin_records"] if row["scope"] == "tail_ge_700"]
    tail_rows = [row for row in tail_rows if row["estimator"] in ("observable_mean", "observable_q50", "all_samples_weighted_1_over_M", "perfect_reco_matched_support")]
    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    for estimator in dict.fromkeys(row["estimator"] for row in tail_rows):
        selected = [row for row in tail_rows if row["estimator"] == estimator and row["truth_weight"] > 0]
        x, y = [], []
        for row in selected:
            label = row["mass_bin"]
            if label.startswith(">"):
                low = float(label[1:])
                high = low + 100.0
            else:
                low, high = map(float, label.split("-"))
            x.append(0.5 * (low + high))
            y.append(row["reco_over_truth"])
        ax.plot(x, y, marker="o", markersize=3, linewidth=1.1, label=estimator)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.set(xlabel="mttbar bin centre [GeV]", ylabel="Reco / truth", title="Tail spectrum ratios (truth mttbar ≥ 700 GeV)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, output_dir / "spectra/mttbar_spectrum_tail_ratio")


def _plot_mse(mse_rows: list[dict], output_dir: Path) -> None:
    selected = [row for row in mse_rows if row["truth_region"] != "inclusive"]
    labels = [row["truth_region"] for row in selected]
    center = np.asarray([row["posterior_centre_mse_gev2"] for row in selected])
    variance = np.asarray([row["within_event_variance_gev2"] for row in selected])
    fig, ax = plt.subplots(figsize=(8.8, 5.8), constrained_layout=True)
    position = np.arange(len(labels))
    ax.bar(position, center, color="#D55E00", label="Posterior-centre MSE")
    ax.bar(position, variance, bottom=center, color="#56B4E9", label="Within-posterior variance")
    ax.set_xticks(position, labels)
    ax.set_ylabel("Single-draw MSE contribution [GeV²]")
    ax.set_title("Exact finite-M single-draw MSE decomposition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir / "posterior/mttbar_mse_decomposition")


def _plot_posterior(posterior_rows: list[dict], coverage_rows: list[dict], nested_rows: list[dict],
                    rank_rows: list[dict], differential: list[dict], pp_rows: list[dict],
                    calibration_differential: list[dict], output_dir: Path) -> None:
    regions = list(COARSE_LABELS)
    rows = {row["truth_region"]: row for row in pp_rows if row["record_type"] == "summary" and row["binning"] == "coarse"}
    fig, ax = plt.subplots(figsize=(8.6, 5.5), constrained_layout=True)
    x = np.arange(len(regions))
    ax.bar(x - 0.18, [rows[r]["mean_correct_bin_probability"] for r in regions], width=0.36, label="Correct-bin probability", color="#0072B2")
    ax.bar(x + 0.18, [rows[r]["mean_correct_or_adjacent_probability"] for r in regions], width=0.36, label="Correct or adjacent", color="#E69F00")
    ax.set_xticks(x, regions)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Mean posterior probability")
    ax.set_title("Posterior probability in truth mttbar region")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir / "posterior/mttbar_correct_bin_probability")

    coverage = {row["truth_region"]: row for row in coverage_rows if row.get("record_type") != "width_quartile"}
    fig, ax = plt.subplots(figsize=(8.6, 5.5), constrained_layout=True)
    ax.plot(regions, [coverage[r]["coverage68"] for r in regions], marker="o", label="68% interval")
    ax.plot(regions, [coverage[r]["coverage90"] for r in regions], marker="s", label="90% interval")
    ax.axhline(0.68, color="#1f77b4", linestyle="--", linewidth=0.9)
    ax.axhline(0.90, color="#ff7f0e", linestyle="--", linewidth=0.9)
    ax.set_ylim(0.0, 1.0)
    ax.set(xlabel="Truth mttbar bin [GeV]", ylabel="Empirical coverage", title="Posterior interval coverage by truth mass")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "posterior/mttbar_coverage_vs_truth")

    mean_rows = [row for row in differential if row["estimator"] == "observable_mean" and row["events"] >= 40]
    fig, ax = plt.subplots(figsize=(8.5, 5.4), constrained_layout=True)
    ax.plot([row["truth_mean_gev"] for row in mean_rows], [row["width68_mean_gev"] for row in mean_rows], marker="o", label="Mean 68% width")
    ax.plot([row["truth_mean_gev"] for row in mean_rows], [row["width90_mean_gev"] for row in mean_rows], marker="s", label="Mean 90% width")
    ax.set(xlabel="Truth mttbar [GeV]", ylabel="Posterior interval width [GeV]", title="Posterior width by truth mass")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "posterior/mttbar_width_vs_truth")

    global_rows = [row for row in nested_rows if row["truth_region"] == "inclusive"]
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.3), constrained_layout=True)
    fields = (
        ("mean_rmse_gev", "Posterior mean RMSE [GeV]"),
        ("correct_bin_probability", "Correct-bin posterior probability"),
        ("best_of_M_rmse_gev_truth_using", "Best-of-M error RMSE [GeV]"),
        ("mean_width68_gev", "Mean posterior width68 [GeV]"),
    )
    for axis, (field, label) in zip(axes.flat, fields, strict=True):
        axis.plot([row["M"] for row in global_rows], [row[field] for row in global_rows], marker="o")
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8, 16, 32], ["1", "2", "4", "8", "16", "32"])
        axis.set(xlabel="Posterior prefix M", ylabel=label)
        axis.grid(alpha=0.25)
    axes[1, 0].set_title("TRUTH-USING DIAGNOSTIC — not deployable")
    fig.suptitle("Nested posterior sample convergence")
    _save(fig, output_dir / "convergence/mttbar_nested_M")

    selected_rank = [row for row in rank_rows if row["truth_region"] in ("inclusive", "<345", "345-400", "400-500", "500-700", "700-1000", ">1000")]
    fig, ax = plt.subplots(figsize=(9.0, 5.6), constrained_layout=True)
    for region in ("inclusive", "<345", "345-400", "400-500", "500-700", "700-1000", ">1000"):
        local = [row for row in selected_rank if row["truth_region"] == region]
        if not local:
            continue
        counts = np.asarray([row["fraction"] for row in local])
        ax.plot(np.arange(len(counts)), counts, marker=".", markersize=3, linewidth=1.0, label=region)
    ax.axhline(1.0 / 33.0, color="black", linestyle="--", linewidth=0.9, label="Uniform rank guide")
    ax.set(xlabel="Midrank of truth among 32 posterior samples", ylabel="Fraction", title="Finite-sample rank diagnostic")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "posterior/mttbar_rank_histogram")

    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
    for estimator in ("observable_mean", "observable_q50", "crossfit_affine_observable_mean", "crossfit_isotonic_observable_mean"):
        selected = [row for row in calibration_differential if row["estimator"] == estimator and row["events"] >= 40]
        if not selected:
            continue
        ax.plot([row["truth_mean_gev"] for row in selected], [row["reco_mean_gev"] for row in selected],
                marker="o", markersize=3, linewidth=1.0, label=estimator)
    axis_limits = ax.get_xlim()
    low = min(axis_limits[0], 300.0)
    high = max(axis_limits[1], 1200.0)
    ax.plot([low, high], [low, high], color="black", linestyle="--", linewidth=0.9, label="y = x")
    ax.set(xlabel="Truth mttbar [GeV]", ylabel="Calibrated/reconstructed mttbar [GeV]",
           title="Diagnostic cross-fitted response calibration")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "calibration/mttbar_crossfit_calibration")


def _lookup(rows: list[dict], **conditions) -> dict:
    for row in rows:
        if all(row.get(key) == value for key, value in conditions.items()):
            return row
    raise KeyError(f"No metric row matches {conditions}")


def _diagnosis_for_region(region: str, estimator_rows: list[dict], support_rows: list[dict],
                          mse_rows: list[dict], migration_summary: list[dict],
                          perfect_rows: list[dict], calibration_rows: list[dict]) -> str:
    q50 = _lookup(estimator_rows, estimator="observable_q50", truth_region=region)
    mean = _lookup(estimator_rows, estimator="observable_mean", truth_region=region)
    support = _lookup(support_rows, truth_region=region)
    mse = _lookup(mse_rows, truth_region=region)
    ppc_diag = _lookup(migration_summary, record_type="truth_row", estimator="posterior_predictive",
                       binning="coarse", truth_bin=region)
    q50_diag = _lookup(migration_summary, record_type="truth_row", estimator="observable_q50",
                       binning="coarse", truth_bin=region)
    best = _lookup(migration_summary, record_type="truth_row", estimator="best_of_M32_truth_using",
                   binning="coarse", truth_bin=region)
    perfect = _lookup(perfect_rows, estimator="perfect_reco", truth_region=region)
    affine = _lookup(calibration_rows, estimator="observable_mean", calibration="affine", truth_region=region)
    true_bin_probability = support["mean_correct_bin_probability"]
    affine_rmse_ratio = affine["rmse_gev"] / mean["rmse_gev"]
    if region == "345-400":
        return (f"The posterior has {true_bin_probability:.1%} truth-bin mass, but its reducers differ: q50 RMSE/bias is {q50['rmse_gev']:.1f}/{q50['bias_gev']:+.1f} GeV "
                f"versus mean {mean['rmse_gev']:.1f}/{mean['bias_gev']:+.1f} GeV. The truth-using selector reaches {best['diagonal_fraction']:.3f} diagonal; "
                "a mass reducer can recover some of the asymmetric posterior, while cross-fitted response correction does not help materially.")
    if region == "400-500":
        mean_diag = _lookup(migration_summary, record_type="truth_row", estimator="observable_mean",
                            binning="coarse", truth_bin=region)
        return (f"This is a strong Q region: posterior mean RMSE {mean['rmse_gev']:.1f} GeV and diagonal {mean_diag['diagonal_fraction']:.3f}; "
                f"q50 diagonal is {q50_diag['diagonal_fraction']:.3f} and fractional draw diagonal is {ppc_diag['diagonal_fraction']:.3f}. "
                f"{mse['variance_fraction']:.1%} of single-draw MSE is posterior variance. "
                f"Perfect reco is similar at {perfect['rmse_gev']:.1f} GeV, so most draw-level loss here is removable by posterior averaging.")
    if region == "500-700":
        mean_diag = _lookup(migration_summary, record_type="truth_row", estimator="observable_mean",
                            binning="coarse", truth_bin=region)
        return (f"The posterior retains truth-bin mass ({true_bin_probability:.1%}) and a truth-using draw is usually close, while the mean has modest low bias "
                f"({mean['bias_gev']:.1f} GeV) and posterior variance contributes {mse['variance_fraction']:.1%} of draw MSE. "
                f"Q mean diagonal is {mean_diag['diagonal_fraction']:.3f} versus Perfect reco {perfect['diagonal_fraction']:.3f}; the reference response is similar, with lower Perfect-reco RMSE.")
    if true_bin_probability < 0.10:
        return (f"Truth-bin posterior mass is only {true_bin_probability:.1%} even though at least one draw lands in the broad bin for every event; "
                f"the mean is biased by {mean['bias_gev']:+.1f} GeV and interval coverage is poor, so this is underweighted threshold support and miscentring, not an M32 draw-count gap.")
    if (abs(mean["bias_gev"]) >= max(30.0, 0.05 * mean["truth_mean_gev"])
            and mse["centre_fraction"] >= 0.55 and affine_rmse_ratio > 0.95):
        return (f"The posterior centre is pulled toward the populated middle of the spectrum (mean bias {mean['bias_gev']:+.1f} GeV; "
                f"{mse['centre_fraction']:.1%} of draw MSE is centre error). Cross-fitted affine calibration changes RMSE by only "
                f"{(1.0 - affine_rmse_ratio):.1%}; tail response/training-prior balance is implicated, with truth-near samples still present.")
    if (true_bin_probability >= 0.40 and support["fraction_events_with_any_correct_bin_sample"] >= 0.7
            and best["diagonal_fraction"] - q50_diag["diagonal_fraction"] >= 0.15):
        return (f"M32 carries appreciable truth-bin mass ({true_bin_probability:.1%}); the truth-using selector raises diagonal response "
                f"from {q50_diag['diagonal_fraction']:.3f} to {best['diagonal_fraction']:.3f}. A deployable reducer can matter, although this oracle gain is not itself deployable.")
    if support["fraction_events_with_any_correct_bin_sample"] < 0.5 or best["diagonal_fraction"] < 0.25:
        return "Truth-compatible coarse-bin samples are absent for many events and even the truth-using best-of-M32 selector remains non-diagonal; posterior support is insufficient."
    if perfect["rmse_gev"] <= 1.1 * q50["rmse_gev"] and abs(perfect["diagonal_fraction"] - q50_diag["diagonal_fraction"]) <= 0.10:
        return "The available Perfect-reco reference has similar response and migration in this region; much of the observed limitation is shared with that visible-plus-oracle route."
    if mse["centre_fraction"] >= 0.6:
        return "Single-draw error is dominated by posterior-centre error; the posterior is mis-centred in this region."
    if mse["variance_fraction"] >= 0.6:
        return "Single-draw error is dominated by within-posterior variance; coherent posterior reduction can help, while uncertainty remains broad."
    return "The regional response combines posterior-centre error and within-posterior width; no single reducer or simple response correction dominates."


def _write_report(path: Path, *, provenance: dict, regression: dict, estimator_rows: list[dict],
                  mse_rows: list[dict], posterior_rows: list[dict], coverage_rows: list[dict],
                  nested_rows: list[dict], spectra_rows: list[dict], migration_summary: list[dict],
                  perfect_rows: list[dict], oracle_rows: list[dict], calibration_rows: list[dict],
                  coherent_rows: list[dict], failure_rows: list[dict], perfect_support: int, n_events: int,
                  recommendation: str | None = None) -> str:
    regions = list(COARSE_LABELS)
    global_draw = _lookup(estimator_rows, estimator="sample0", truth_region="inclusive")
    global_mean = _lookup(estimator_rows, estimator="observable_mean", truth_region="inclusive")
    global_q50 = _lookup(estimator_rows, estimator="observable_q50", truth_region="inclusive")
    draw_matrix = _lookup(migration_summary, record_type="global", estimator="sample0", binning="coarse")
    mean_matrix = _lookup(migration_summary, record_type="global", estimator="observable_mean", binning="coarse")
    q50_matrix = _lookup(migration_summary, record_type="global", estimator="observable_q50", binning="coarse")
    ppc_matrix = _lookup(migration_summary, record_type="global", estimator="posterior_predictive", binning="coarse")
    best_matrix = _lookup(migration_summary, record_type="global", estimator="best_of_M32_truth_using", binning="coarse")
    perfect_global = _lookup(perfect_rows, estimator="perfect_reco", truth_region="inclusive")
    matched_ppc_global = _lookup(perfect_rows, estimator="posterior_predictive", truth_region="inclusive")
    matched_ppc_spectrum = _lookup(spectra_rows, estimator="posterior_predictive_matched_support", scope="full")
    full_spectra = [row for row in spectra_rows if row["scope"] == "full"]
    best_spectrum = min((row for row in full_spectra if "_matched_" not in row["estimator"]),
                        key=lambda row: row["wasserstein1_gev"])
    best_event = min((row for row in estimator_rows if row["truth_region"] == "inclusive" and not row["estimator"].startswith("crossfit_")),
                     key=lambda row: row["rmse_gev"])
    mse_global = _lookup(mse_rows, truth_region="inclusive")
    ppc_global = _lookup(posterior_rows, record_type="summary", binning="coarse", truth_region="inclusive")
    coverage_global = _lookup(coverage_rows, truth_region="inclusive")
    crossfit_mean = _lookup(calibration_rows, estimator="observable_mean", calibration="affine", truth_region="inclusive")
    crossfit_q50 = _lookup(calibration_rows, estimator="observable_q50", calibration="isotonic", truth_region="inclusive")
    q50_ppc_spectrum = _lookup(spectra_rows, estimator="all_samples_weighted_1_over_M", scope="full")
    mean_spectrum = _lookup(spectra_rows, estimator="observable_mean", scope="full")
    weak_support_region = min(
        COARSE_LABELS,
        key=lambda region: _lookup(posterior_rows, record_type="summary", binning="coarse", truth_region=region)["mean_correct_bin_probability"],
    )
    coherent_q50 = _lookup(estimator_rows, estimator="coherent_nearest_q50", truth_region="inclusive")
    coherent_mean = _lookup(estimator_rows, estimator="coherent_nearest_mean", truth_region="inclusive")
    medoid = _lookup(estimator_rows, estimator="joint_medoid", truth_region="inclusive")
    q50_parent_pt = _lookup(coherent_rows, reducer="coherent_nearest_q50", observable="top_pt", truth_region="inclusive")
    medoid_parent_pt = _lookup(coherent_rows, reducer="joint_medoid", observable="top_pt", truth_region="inclusive")
    nested16 = _lookup(nested_rows, M=16, truth_region="inclusive")
    nested32 = _lookup(nested_rows, M=32, truth_region="inclusive")
    bad_nu_q1 = _lookup(failure_rows, truth_region=">1000", stratifier="truth_assignment_predicted_neutrino_mass_error_quartile", category="1")
    bad_nu_q4 = _lookup(failure_rows, truth_region=">1000", stratifier="truth_assignment_predicted_neutrino_mass_error_quartile", category="4")
    wrong_assignment = _lookup(failure_rows, truth_region=">1000", stratifier="assignment_complete_correct", category="False")
    correct_assignment = _lookup(failure_rows, truth_region=">1000", stratifier="assignment_complete_correct", category="True")
    lines = [
        "# REALNU PCFM_Q mttbar mass-response diagnosis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Scope and provenance",
        "",
        f"- Population: {n_events:,} validation events; 32 ordered Heun16 samples per event.",
        f"- Prediction: `{provenance['prediction']}` (SHA-256 `{provenance['prediction_sha256']}`).",
        f"- Ordered prediction EVENT_ID SHA-256: `{provenance['event_id_sha256']}`.",
        f"- Source bank: `{provenance['source_bank']}` (SHA-256 `{provenance['source_bank_sha256']}`).",
        f"- Source-bank and prediction ID sets are equal: `{provenance['source_bank_and_prediction_id_sets_equal']}`; their stored orders are equal: `{provenance['source_bank_and_prediction_order_equal']}`.",
        f"- Checkpoint SHA-256: `{provenance['checkpoint_sha256']}`; scaler identity SHA-256: `{provenance['scaler_identity_sha256']}`.",
        f"- Event metrics table: `{provenance['event_table']}` (SHA-256 `{provenance['event_table_sha256']}`).",
        f"- Parent order: `(top, antitop)`; truth-free conditioning: `{provenance['truth_free_conditioning']}`; sealed test used: `{provenance['sealed_test_used']}`.",
        f"- Perfect-reco direct comparisons use exactly {perfect_support:,}/{n_events:,} finite Q1 parent pairs and the same matched truth mask.",
        "- The available constituent references are Q1 Perfect reco, predicted assignment with truth neutrinos, truth assignment with predicted neutrinos, and predicted assignment with predicted neutrinos. Their finite support is written explicitly in `oracle/constituent_oracle_metrics.csv`.",
        "",
        "## Main finding",
        "",
        f"The accepted reduction gate passed: sample0 RMSE {global_draw['rmse_gev']:.3f} GeV, observable-mean RMSE {global_mean['rmse_gev']:.3f} GeV, and observable-q50 RMSE {global_q50['rmse_gev']:.3f} GeV. Their hard coarse-bin diagonal fractions are {draw_matrix['diagonal_fraction']:.3f}, {mean_matrix['diagonal_fraction']:.3f}, and {q50_matrix['diagonal_fraction']:.3f}; posterior-predictive diagonal probability is {ppc_matrix['diagonal_fraction']:.3f}.",
        f"The exact draw-MSE split is {mse_global['centre_fraction']:.1%} posterior-centre error and {mse_global['variance_fraction']:.1%} within-event posterior variance. The truth-using best-of-M32 selector reaches diagonal fraction {best_matrix['diagonal_fraction']:.3f}; this is an oracle diagnostic, not a deployable reducer.",
        f"The posterior predictive spectrum has W1 {q50_ppc_spectrum['wasserstein1_gev']:.1f} GeV and TV {q50_ppc_spectrum['total_variation']:.3f}; the posterior-mean spectrum has W1 {mean_spectrum['wasserstein1_gev']:.1f} GeV. The best event-level RMSE estimator is `{best_event['estimator']}` ({best_event['rmse_gev']:.2f} GeV), while the best full-spectrum W1 is `{best_spectrum['estimator']}` ({best_spectrum['wasserstein1_gev']:.1f} GeV).",
        f"On the Perfect-reco matched support, Perfect reco RMSE is {perfect_global['rmse_gev']:.2f} GeV with diagonal fraction {perfect_global['diagonal_fraction']:.3f}; Q's posterior predictive has draw RMSE {matched_ppc_global['rmse_gev']:.2f} GeV and diagonal probability {matched_ppc_global['diagonal_fraction']:.3f}. Its matched-support spectrum W1 is {matched_ppc_spectrum['wasserstein1_gev']:.1f} GeV.",
        "",
        "## Regional diagnosis",
        "",
        "| Truth mttbar [GeV] | N | Mean bias / RMSE | q50 bias / RMSE | Centre / variance MSE | Width68 median | P(correct bin) | Events with any correct-bin draw | Best-of-M32 RMSE / diagonal | PPC diagonal | Perfect-reco RMSE / diagonal | 68% / 90% coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in regions:
        mean = _lookup(estimator_rows, estimator="observable_mean", truth_region=region)
        q50 = _lookup(estimator_rows, estimator="observable_q50", truth_region=region)
        mse = _lookup(mse_rows, truth_region=region)
        support = _lookup(posterior_rows, record_type="summary", binning="coarse", truth_region=region)
        best = _lookup(migration_summary, record_type="truth_row", estimator="best_of_M32_truth_using", binning="coarse", truth_bin=region)
        ppc = _lookup(migration_summary, record_type="truth_row", estimator="posterior_predictive", binning="coarse", truth_bin=region)
        perf = _lookup(perfect_rows, estimator="perfect_reco", truth_region=region)
        coverage = _lookup(coverage_rows, truth_region=region)
        lines.append(
            f"| {region} | {mean['events']:,} | {mean['bias_gev']:.1f} / {mean['rmse_gev']:.1f} | {q50['bias_gev']:.1f} / {q50['rmse_gev']:.1f} | {mse['centre_fraction']:.1%} / {mse['variance_fraction']:.1%} | {coverage['width68_median_gev']:.1f} GeV | {support['mean_correct_bin_probability']:.3f} | {support['fraction_events_with_any_correct_bin_sample']:.1%} | {coverage['best_of_M32_RMSE_truth_using_gev']:.1f} GeV / {best['diagonal_fraction']:.3f} | {ppc['diagonal_fraction']:.3f} | {perf['rmse_gev']:.1f} / {perf['diagonal_fraction']:.3f} | {coverage['coverage68']:.3f} / {coverage['coverage90']:.3f} |"
        )
    lines.extend(["", "Regional mechanism from the paired posterior, migration, oracle, and cross-fit evidence:", ""])
    for region in regions:
        diagnosis = _diagnosis_for_region(region, estimator_rows, posterior_rows, mse_rows,
                                          migration_summary, perfect_rows, calibration_rows)
        lines.append(f"- **{region} GeV:** {diagnosis}")

    lines.extend([
        "",
        "## Answers to the diagnosis questions",
        "",
        f"1. **Hard point-estimator migration:** sample0 diagonal {draw_matrix['diagonal_fraction']:.3f}, mean {mean_matrix['diagonal_fraction']:.3f}, q50 {q50_matrix['diagonal_fraction']:.3f}; see coarse and fine row/column-normalised matrices in `metrics/migration_metrics.csv`.",
        f"2. **Probabilistic response:** posterior-predictive diagonal fraction {ppc_matrix['diagonal_fraction']:.3f}, correct-or-adjacent probability {ppc_global['mean_correct_or_adjacent_probability']:.3f}, Brier {ppc_global['categorical_brier_score']:.3f}, mean entropy {ppc_global['posterior_entropy_nats']:.3f} nats.",
        f"3. **Single-draw MSE:** {mse_global['centre_fraction']:.1%} comes from posterior-centre error and {mse_global['variance_fraction']:.1%} from finite-M posterior variance; identity residual is {mse_global['identity_abs_residual_gev2']:.3g} GeV².",
        f"4. **Truth-compatible samples:** coarse true-bin posterior mass averages {ppc_global['mean_correct_bin_probability']:.3f}; {ppc_global['fraction_events_with_any_correct_bin_sample']:.1%} of events have at least one M32 draw in their truth bin. Fine-bin support is in `metrics/posterior_predictive.csv`.",
        f"5. **Underrepresented truth-bin support:** the lowest mean true-bin probability is in {weak_support_region} GeV at {_lookup(posterior_rows, record_type='summary', binning='coarse', truth_region=weak_support_region)['mean_correct_bin_probability']:.1%}; every event has at least one draw in its broad coarse bin, but that alone does not imply useful posterior weight. The regional table and `metrics/posterior_support.csv` show all bins.",
        f"6. **Truth-using support test:** best-of-M32 reaches diagonal fraction {best_matrix['diagonal_fraction']:.3f}, best-of-M32 error RMSE {nested32['best_of_M_rmse_gev_truth_using']:.2f} GeV, and the regional best-of-M migration appears above. Selection uses truth and is not deployable.",
        f"7. **Support versus M:** mean RMSE changes {nested16['mean_rmse_gev']:.2f}→{nested32['mean_rmse_gev']:.2f} GeV, q50 RMSE {nested16['q50_rmse_gev']:.2f}→{nested32['q50_rmse_gev']:.2f} GeV, true-bin probability {nested16['correct_bin_probability']:.3f}→{nested32['correct_bin_probability']:.3f}, and best-of-M error {nested16['best_of_M_rmse_gev_truth_using']:.2f}→{nested32['best_of_M_rmse_gev_truth_using']:.2f} GeV. Region-specific convergence is in `metrics/nested_M.csv`; there is no M64 extrapolation.",
        f"8. **Tail calibration:** inclusive posterior coverage is {coverage_global['coverage68']:.3f} (68%) and {coverage_global['coverage90']:.3f} (90%); 700–1000 and >1000 GeV rows show tail coverage. Rank histograms use the stated M32 midrank convention.",
        f"9. **Posterior-mean spectrum width:** truth population width is {q50_ppc_spectrum['truth_population_std_gev']:.1f} GeV; posterior-mean width is {mean_spectrum['reco_population_std_gev']:.1f} GeV (ratio {mean_spectrum['reco_to_truth_population_width_ratio']:.3f}) with W1 {mean_spectrum['wasserstein1_gev']:.1f} GeV.",
        f"10. **Posterior-predictive spectrum:** all M32 samples weighted 1/32 have width {q50_ppc_spectrum['reco_population_std_gev']:.1f} GeV (ratio {q50_ppc_spectrum['reco_to_truth_population_width_ratio']:.3f}), W1 {q50_ppc_spectrum['wasserstein1_gev']:.1f} GeV, and TV {q50_ppc_spectrum['total_variation']:.3f}. Event-RMSE optimum is `{best_event['estimator']}`; spectrum-W1 optimum is `{best_spectrum['estimator']}`; they {'are the same' if best_event['estimator'] == best_spectrum['estimator'] else 'are different'}.",
        f"11. **Migration beyond Perfect reco:** matched-support Perfect-reco diagonal is {perfect_global['diagonal_fraction']:.3f}; Q mean diagonal is {_lookup(perfect_rows, estimator='observable_mean', truth_region='inclusive')['diagonal_fraction']:.3f}; Q posterior-predictive diagonal is {matched_ppc_global['diagonal_fraction']:.3f}. The file `oracle/excess_migration_vs_perfect_reco.csv` contains row-normalised Q-minus-Perfect matrices for sample0, mean, q50, and posterior predictive on exactly the same finite support; matched-support spectra are in `metrics/spectrum_closure.csv` and `plots/spectra/mttbar_spectrum_perfect_matched_support.pdf`.",
        f"12. **Assignment and neutrino limits:** inclusive predicted-assignment + truth-neutrino RMSE is {_lookup(oracle_rows, reference='predicted_assignment_truth_neutrinos', truth_region='inclusive')['rmse_gev']:.1f} GeV; truth-assignment + predicted-neutrino RMSE is {_lookup(oracle_rows, reference='truth_assignment_predicted_neutrinos', truth_region='inclusive')['rmse_gev']:.1f} GeV. Above 1000 GeV these are {_lookup(oracle_rows, reference='predicted_assignment_truth_neutrinos', truth_region='>1000')['rmse_gev']:.1f} and {_lookup(oracle_rows, reference='truth_assignment_predicted_neutrinos', truth_region='>1000')['rmse_gev']:.1f} GeV, respectively, compared with Q mean {_lookup(estimator_rows, estimator='observable_mean', truth_region='>1000')['rmse_gev']:.1f} GeV. These substitutions show a much larger neutrino-reconstruction limitation; they are diagnostic routes, not a strict additive error decomposition. Regional RMSE and finite supports are tabulated below.",
        f"13. **Cross-fitted calibration:** affine mean RMSE is {crossfit_mean['rmse_gev']:.2f} GeV versus raw {global_mean['rmse_gev']:.2f}; its response slope rises from {global_mean['response_slope']:.3f} to {crossfit_mean['response_slope']:.3f}, still below one. Isotonic q50 RMSE is {crossfit_q50['rmse_gev']:.2f} GeV versus raw {global_q50['rmse_gev']:.2f}. These small held-out changes do not remove shrinkage; migration, spectrum and regional response are in the metrics files. This is validation diagnostic only.",
        f"14. **Coherent reducers:** nearest-q50 mass RMSE {coherent_q50['rmse_gev']:.2f} GeV; nearest-mean {coherent_mean['rmse_gev']:.2f}; joint medoid {medoid['rmse_gev']:.2f}. Nearest-q50 top-pT RMSE is {q50_parent_pt['rmse']:.2f} GeV versus medoid {medoid_parent_pt['rmse']:.2f}; all six parent observables and regions are in `metrics/coherent_reducers.csv`.",
        f"15. **Failure stratification:** the worst hard-migration regions are <345 and >1000 GeV. In >1000 GeV, q50 RMSE is {bad_nu_q1['rmse_gev']:.1f} GeV in the lowest truth-assignment/predicted-neutrino error quartile and {bad_nu_q4['rmse_gev']:.1f} GeV in the highest; incorrect versus correct assignment events are {wrong_assignment['rmse_gev']:.1f} versus {correct_assignment['rmse_gev']:.1f} GeV (N={wrong_assignment['events']} versus {correct_assignment['events']}). The association with poor neutrino support is strong; channels and jet multiplicity show smaller separation in the checked strata. These are correlations, not a causal factorization.",
        f"16. **Next action:** {recommendation or 'Evaluate the trained PCFM_geom prediction with this identical M32 mass audit and the prepared rotation/H_geom diagnostics.'}",
        "",
        "## Interpretation and next intervention",
        "",
        "The diagnosis is based on event-level truth, M32 support, posterior-predictive migration, matched-support Perfect reco, and held-out one-dimensional calibration. The Perfect-reco route is an available measured-visible plus truth-assignment/truth-neutrino reference, not a Bayes or irreducible-information limit. The validation-derived calibrators are diagnostic only.",
        "",
        "## Artifacts",
        "",
        "Machine-readable metrics live under `metrics/`, constituent and matched-support comparisons under `oracle/`, cross-fit records under `calibration/`, and plots under `plots/`. The provenance manifest records the exact input hashes and inference identity.",
        "",
    ])
    lines.extend([
        "",
        "## Constituent-oracle regional response",
        "",
        "RMSE in GeV; each cell reports finite parent support.",
        "",
        "| Truth bin | Perfect reco | Predicted assignment + truth neutrinos | Truth assignment + predicted neutrinos | Both predicted |",
        "|---|---:|---:|---:|---:|",
    ])
    for region in regions:
        oracle_values = []
        for oracle in (
            "perfect_reco",
            "predicted_assignment_truth_neutrinos",
            "truth_assignment_predicted_neutrinos",
            "predicted_assignment_predicted_neutrinos",
        ):
            row = _lookup(oracle_rows, reference=oracle, truth_region=region)
            rmse = row.get("rmse_gev")
            value = f"{float(rmse):.1f}" if rmse is not None else "NA"
            oracle_values.append(f"{value} (N={row['finite_support_events']:,})")
        lines.append(f"| {region} | " + " | ".join(oracle_values) + " |")
    text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _analyze_full(data: dict, provenance: dict, output_dir: Path, *, bootstrap_replicates: int) -> dict:
    ids = np.asarray(data["EVENT_ID"], dtype=np.int64)
    truth = np.asarray(data["truth_mttbar"], dtype=np.float64)
    samples = np.asarray(data["sample_mttbar"], dtype=np.float64)
    perfect = np.asarray(data["perfect_reco_mttbar"], dtype=np.float64)
    perfect_mask = np.asarray(data["perfect_reco_mask"], dtype=bool)
    if len(ids) != EXPECTED_EVENTS or samples.shape != (EXPECTED_EVENTS, EXPECTED_SAMPLES):
        raise ValueError("Full mass-response audit requires the exact 100342 x 32 validation population")
    expected_support = int(provenance["manifest"]["perfect_reco_plot_support_correction"]["finite_parent_events"])
    if int(perfect_mask.sum()) != expected_support:
        raise ValueError(f"Perfect-reco finite support {int(perfect_mask.sum())} differs from manifest {expected_support}")

    estimates, extras = _build_estimates(data)
    regression = _regression_gate(data, provenance["summary"], estimates)
    masks = _region_masks(truth)
    accepted_region_counts = provenance["summary"].get("region_counts", {})
    for label in COARSE_LABELS:
        key = f"truth_mttbar/{label}"
        expected_count = int(accepted_region_counts.get(key, -1))
        actual_count = int(masks[label].sum())
        if expected_count < 0 or actual_count != expected_count:
            raise RuntimeError(
                f"Truth mass-bin count differs from accepted reduction for {label}: "
                f"{actual_count} != {expected_count}"
            )
    calibrated, calibration_rows, calibration_data = _crossfit_calibration(ids, truth, estimates, masks)
    estimates.update(calibrated)
    estimator_rows = []
    for estimator, prediction in estimates.items():
        for region, mask in masks.items():
            estimator_rows.append({"estimator": estimator, "truth_region": region, **_point_metrics(prediction, truth, mask)})
    fine_edges = _fine_edges()
    differential = _fine_bin_summary(estimates, truth, fine_edges)
    fine_truth_bin = _bin_ids(truth, fine_edges)
    width68 = np.quantile(samples, 0.84, axis=1) - np.quantile(samples, 0.16, axis=1)
    width90 = np.quantile(samples, 0.95, axis=1) - np.quantile(samples, 0.05, axis=1)
    for row in differential:
        if row["estimator"] != "observable_mean":
            continue
        bin_idx = _bin_labels(fine_edges).index(row["truth_bin"])
        select = fine_truth_bin == bin_idx
        row["width68_mean_gev"] = float(width68[select].mean())
        row["width90_mean_gev"] = float(width90[select].mean())

    mse_rows = _mse_decomposition(samples, truth, masks)
    posterior_coverage, rank_rows, posterior = _posterior_diagnostics(samples, truth, masks)
    coverage_rows = [{"record_type": "truth_region", **row} for row in posterior_coverage]
    coverage_rows.extend({"record_type": "width_quartile", **row} for row in posterior["quartile_rows"])
    nested_rows = _nested_m(samples, truth, masks)
    ppc_coarse = _posterior_predictive(samples, truth, COARSE_EDGES)
    ppc_fine = _posterior_predictive(samples, truth, fine_edges)
    posterior_rows = _posterior_probability_rows(ppc_coarse, truth, masks, "coarse", COARSE_EDGES)
    posterior_rows += _posterior_probability_rows(ppc_fine, truth, masks, "fine", fine_edges)
    posterior_support_rows = []
    coarse_prob = ppc_coarse["event_probabilities"]
    coarse_true_id = _bin_ids(truth, COARSE_EDGES)
    event_any = np.any(coarse_prob > 0.0, axis=1)
    for row in posterior_coverage:
        region, mask = row["truth_region"], masks[row["truth_region"]]
        best_matrix_row = next(item for item in posterior_coverage if item["truth_region"] == region)
        posterior_support_rows.append({
            **best_matrix_row,
            "events_with_any_true_bin_sample": int(event_any[mask].sum()),
            "fraction_events_with_any_true_bin_sample": float(event_any[mask].mean()),
            "mean_true_bin_probability": float(coarse_prob[np.arange(len(truth))[mask], coarse_true_id[mask]].mean()),
            "mean_true_or_adjacent_probability": float(posterior["adjacent_probability"][mask].mean()),
            "best_of_M32_mae_gev_truth_using": float(posterior["best_error"][mask].mean()),
            "best_of_M32_rmse_gev_truth_using": float(np.sqrt(np.mean(np.square(posterior["best_error"][mask])))),
        })

    migration_rows: list[dict] = []
    migration_summary: list[dict] = []
    plot_matrices: dict[str, dict] = {}
    hard_estimators = [
        "sample0", "observable_mean", "observable_q50", "joint_medoid",
        "coherent_nearest_q50", "coherent_nearest_mean", "mass_posterior_mode",
        "crossfit_affine_observable_mean", "crossfit_isotonic_observable_mean",
        "crossfit_affine_observable_q50", "crossfit_isotonic_observable_q50",
    ]
    for estimator in hard_estimators:
        for binning, edges in (("coarse", COARSE_EDGES), ("fine", fine_edges)):
            matrix = _migration_matrix(truth, estimates[estimator], edges)
            migration_rows.extend(_matrix_csv_rows(estimator, binning, matrix, edges))
            migration_summary.extend(_migration_summary_rows(estimator, binning, matrix, edges))
            if binning == "coarse":
                plot_matrices[estimator] = matrix
    for binning, edges, matrix in (("coarse", COARSE_EDGES, ppc_coarse), ("fine", fine_edges, ppc_fine)):
        migration_rows.extend(_matrix_csv_rows("posterior_predictive", binning, matrix, edges))
        migration_summary.extend(_migration_summary_rows("posterior_predictive", binning, matrix, edges))
    plot_matrices["posterior_predictive"] = ppc_coarse
    best_migration = _migration_matrix(truth, posterior["best_mass"], COARSE_EDGES)
    migration_rows.extend(_matrix_csv_rows("best_of_M32_truth_using", "coarse", best_migration, COARSE_EDGES))
    migration_summary.extend(_migration_summary_rows("best_of_M32_truth_using", "coarse", best_migration, COARSE_EDGES))
    plot_matrices["best_of_M32_truth_using"] = best_migration

    # All direct Perfect-reco comparisons use precisely its finite parent support.
    perfect_matched = _migration_matrix(truth[perfect_mask], perfect[perfect_mask], COARSE_EDGES)
    matched_ppc = _posterior_predictive(samples[perfect_mask], truth[perfect_mask], COARSE_EDGES)
    migration_rows.extend(_matrix_csv_rows("perfect_reco", "coarse", perfect_matched, COARSE_EDGES))
    migration_summary.extend(_migration_summary_rows("perfect_reco", "coarse", perfect_matched, COARSE_EDGES))
    migration_rows.extend(_matrix_csv_rows("posterior_predictive_matched_perfect_support", "coarse", matched_ppc, COARSE_EDGES))
    migration_summary.extend(_migration_summary_rows("posterior_predictive_matched_perfect_support", "coarse", matched_ppc, COARSE_EDGES))
    plot_matrices["perfect_reco"] = perfect_matched
    plot_matrices["posterior_predictive_matched_perfect_support"] = matched_ppc
    matched_estimators = ["sample0", "observable_mean", "observable_q50", "joint_medoid"]
    perfect_rows = []
    for estimator in [*matched_estimators, "perfect_reco"]:
        prediction = perfect if estimator == "perfect_reco" else estimates[estimator]
        for region, region_mask in masks.items():
            common = region_mask & perfect_mask
            metrics = _point_metrics(prediction, truth, common)
            reco_id = _bin_ids(prediction[common], COARSE_EDGES)
            truth_id = _bin_ids(truth[common], COARSE_EDGES)
            metrics["diagonal_fraction"] = float(np.mean(reco_id == truth_id))
            metrics["adjacent_fraction"] = float(np.mean(np.abs(reco_id - truth_id) <= 1))
            perfect_rows.append({"estimator": estimator, "truth_region": region,
                                 "support_events": int(common.sum()), **metrics})
        matched = _migration_matrix(truth[perfect_mask], prediction[perfect_mask], COARSE_EDGES)
        if estimator != "perfect_reco":
            name = f"{estimator}_matched_perfect_support"
            migration_rows.extend(_matrix_csv_rows(name, "coarse", matched, COARSE_EDGES))
            migration_summary.extend(_migration_summary_rows(name, "coarse", matched, COARSE_EDGES))
    for region, region_mask in masks.items():
        common = region_mask & perfect_mask
        sample_values = samples[common].reshape(-1)
        repeated_truth = np.repeat(truth[common], samples.shape[1])
        metrics = _point_metrics(sample_values, repeated_truth, np.ones(len(sample_values), dtype=bool))
        selected = masks[region][perfect_mask]
        metrics["events"] = int(common.sum())
        metrics["diagonal_fraction"] = float(matched_ppc["event_correct_probability"][selected].mean())
        metrics["adjacent_fraction"] = float(matched_ppc["event_adjacent_probability"][selected].mean())
        perfect_rows.append({"estimator": "posterior_predictive", "truth_region": region,
                             "support_events": int(common.sum()), "posterior_draws": int(sample_values.size), **metrics})

    q_matched_matrices = {
        "sample0": _migration_matrix(truth[perfect_mask], estimates["sample0"][perfect_mask], COARSE_EDGES),
        "observable_mean": _migration_matrix(truth[perfect_mask], estimates["observable_mean"][perfect_mask], COARSE_EDGES),
        "observable_q50": _migration_matrix(truth[perfect_mask], estimates["observable_q50"][perfect_mask], COARSE_EDGES),
        "posterior_predictive": matched_ppc,
    }
    excess_rows = []
    labels = _bin_labels(COARSE_EDGES, coarse=True)
    excess = q_matched_matrices["observable_mean"]["row"] - perfect_matched["row"]
    for estimator, matched_matrix in q_matched_matrices.items():
        delta = matched_matrix["row"] - perfect_matched["row"]
        for i, truth_label in enumerate(labels):
            for j, reco_label in enumerate(labels):
                excess_rows.append({"estimator": estimator, "truth_bin": truth_label, "reco_bin": reco_label,
                                    "truth_events_matched": int(perfect_matched["counts"][i]),
                                    "Q_probability": float(matched_matrix["row"][i, j]),
                                    "perfect_reco_probability": float(perfect_matched["row"][i, j]),
                                    "Q_minus_perfect_reco": float(delta[i, j])})

    oracle_rows, oracle_matrix_rows = _oracle_metrics(data, masks)
    migration_rows.extend(oracle_matrix_rows)
    coherent_rows = _coherent_reducer_metrics(data, extras, masks)
    spectra_rows, spectrum_bin_rows, spectra = _spectra(data, estimates, perfect_mask)
    if not math.isclose(float(ppc_coarse["event_probabilities"].sum(axis=1).mean()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("Posterior-predictive coarse event weights do not sum to one")
    if not math.isclose(float(ppc_fine["event_probabilities"].sum(axis=1).mean()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("Posterior-predictive fine event weights do not sum to one")
    if not math.isclose(float(spectra["posterior_predictive"]["reco_histogram"].sum()), len(truth), rel_tol=0.0, abs_tol=1.0e-8):
        raise RuntimeError("Posterior-predictive spectrum weight is not one per validation event")
    failure_rows = _failure_stratification(data, estimates, extras, posterior, masks)
    bootstrap_rows = _bootstrap_headlines(truth, estimates, perfect, perfect_mask,
                                          replicates=bootstrap_replicates)

    base = output_dir / "metrics"
    _csv_write(base / "estimator_metrics.csv", estimator_rows)
    _csv_write(base / "differential_metrics.csv", differential)
    _csv_write(base / "migration_metrics.csv", migration_rows)
    _csv_write(base / "migration_summary.csv", migration_summary)
    _csv_write(base / "posterior_predictive.csv", posterior_rows)
    _csv_write(base / "mse_decomposition.csv", mse_rows)
    _csv_write(base / "posterior_support.csv", posterior_support_rows)
    _csv_write(base / "coverage.csv", coverage_rows)
    _csv_write(base / "nested_M.csv", nested_rows)
    _csv_write(base / "spectrum_closure.csv", spectra_rows)
    _csv_write(base / "spectrum_bins.csv", spectrum_bin_rows)
    _csv_write(base / "coherent_reducers.csv", coherent_rows)
    _csv_write(base / "failure_stratification.csv", failure_rows or [{"truth_region": "", "stratifier": "", "category": "", "events": 0}])
    _csv_write(base / "paired_bootstrap.csv", bootstrap_rows)
    _csv_write(output_dir / "calibration/crossfit_metrics.csv", calibration_rows)
    _json_write(output_dir / "calibration/calibration_manifest.json", {
        **calibration_data["manifest"], "fold_records": calibration_data["fold_records"],
        "prediction_population": n_events if (n_events := len(ids)) else len(ids),
        "prediction_event_id_sha256": provenance["event_id_sha256"],
    })
    _csv_write(output_dir / "oracle/perfect_reco_metrics.csv", perfect_rows)
    _csv_write(output_dir / "oracle/constituent_oracle_metrics.csv", oracle_rows)
    _csv_write(output_dir / "oracle/excess_migration_vs_perfect_reco.csv", excess_rows)

    # Attach a concise dynamic calibration response summary to the plotting input.
    _plot_response(differential, output_dir / "plots")
    _plot_mse(mse_rows, output_dir / "plots")
    _plot_spectra(data, estimates, {**spectra, "bin_records": spectrum_bin_rows}, output_dir / "plots")
    _plot_migration(plot_matrices["sample0"], COARSE_EDGES, output_dir / "plots/migration/mttbar_migration_draw",
                    "Sample 0 hard mttbar migration")
    _plot_migration(plot_matrices["observable_mean"], COARSE_EDGES, output_dir / "plots/migration/mttbar_migration_mean",
                    "Posterior-mean hard mttbar migration")
    _plot_migration(plot_matrices["observable_q50"], COARSE_EDGES, output_dir / "plots/migration/mttbar_migration_q50",
                    "Posterior-q50 hard mttbar migration")
    _plot_migration(plot_matrices["posterior_predictive"], COARSE_EDGES,
                    output_dir / "plots/migration/mttbar_migration_posterior_predictive",
                    "Posterior-predictive fractional mttbar migration")
    _plot_migration(plot_matrices["posterior_predictive_matched_perfect_support"], COARSE_EDGES,
                    output_dir / "plots/migration/mttbar_migration_posterior_predictive_matched_perfect_support",
                    f"Posterior predictive on Perfect-reco matched support, N={int(perfect_mask.sum()):,}")
    _plot_migration(plot_matrices["best_of_M32_truth_using"], COARSE_EDGES,
                    output_dir / "plots/migration/mttbar_migration_best_of_M32",
                    "TRUTH-USING DIAGNOSTIC — best of M32; not deployable")
    _plot_migration(plot_matrices["perfect_reco"], COARSE_EDGES,
                    output_dir / "plots/migration/mttbar_migration_perfect_reco",
                    f"Perfect reco, finite matched support N={int(perfect_mask.sum()):,}")
    excess_matrix = {"row": excess, "counts": perfect_matched["counts"]}
    _plot_migration(excess_matrix, COARSE_EDGES,
                    output_dir / "plots/migration/mttbar_excess_migration_vs_perfect_reco",
                    "Q posterior mean excess migration relative to Perfect reco", diverging=True)

    crossfit_differential = _fine_bin_summary(estimates, truth, fine_edges)
    _plot_posterior(posterior_rows, coverage_rows, nested_rows, rank_rows,
                    differential, posterior_rows, crossfit_differential, output_dir / "plots")

    # The report is generated only after all numerical gates and artifacts pass.
    global_stats = {
        "accepted_metric_regression": regression,
        "mse_decomposition_inclusive": _lookup(mse_rows, truth_region="inclusive"),
        "posterior_predictive_inclusive": _lookup(posterior_rows, record_type="summary", binning="coarse", truth_region="inclusive"),
        "hard_migration_global": {
            estimator: _lookup(migration_summary, record_type="global", estimator=estimator, binning="coarse")
            for estimator in ("sample0", "observable_mean", "observable_q50", "posterior_predictive", "best_of_M32_truth_using", "perfect_reco")
        },
        "posterior_predictive_matched_perfect_support": _lookup(
            migration_summary, record_type="global", estimator="posterior_predictive_matched_perfect_support", binning="coarse"
        ),
        "perfect_reco_finite_events": int(perfect_mask.sum()),
        "nested_M16": _lookup(nested_rows, M=16, truth_region="inclusive"),
        "nested_M32": _lookup(nested_rows, M=32, truth_region="inclusive"),
        "spectrum_full": {row["estimator"]: row for row in spectra_rows if row["scope"] == "full"},
        "estimator_inclusive": {row["estimator"]: row for row in estimator_rows if row["truth_region"] == "inclusive"},
        "crossfit_inclusive": {f"{row['calibration']}_{row['estimator']}": row for row in calibration_rows if row["truth_region"] == "inclusive"},
        "oracle_inclusive": {row["reference"]: row for row in oracle_rows if row["truth_region"] == "inclusive"},
        "population_spectrum_best_wasserstein1": min(
            (row for row in spectra_rows if row["scope"] == "full" and "_matched_" not in row["estimator"]),
            key=lambda row: row["wasserstein1_gev"],
        ),
        "event_rmse_best_non_crossfit": min((row for row in estimator_rows if row["truth_region"] == "inclusive" and not row["estimator"].startswith("crossfit_")), key=lambda row: row["rmse_gev"]),
        "fine_bin_edges_gev": [float(x) if np.isfinite(x) else None for x in fine_edges],
        "fine_overflow_label": f">{fine_edges[-2]:g}",
    }
    _json_write(output_dir / "metrics/headline.json", global_stats)
    report_path = output_dir / "report/MASS_RESPONSE_DIAGNOSIS.md"
    _write_report(report_path, provenance=provenance, regression=regression,
                  estimator_rows=estimator_rows, mse_rows=mse_rows,
                  posterior_rows=posterior_rows, coverage_rows=coverage_rows,
                  nested_rows=nested_rows, spectra_rows=spectra_rows,
                  migration_summary=migration_summary, perfect_rows=perfect_rows,
                  oracle_rows=oracle_rows, calibration_rows=calibration_rows,
                  coherent_rows=coherent_rows, failure_rows=failure_rows,
                  perfect_support=int(perfect_mask.sum()),
                  n_events=len(ids))
    return {
        "regression": regression,
        "perfect_reco_support": int(perfect_mask.sum()),
        "events": len(ids),
        "samples": samples.shape[1],
        "report": str(report_path),
        "headline": global_stats,
    }


def analyse(manifest_path: Path, output_dir: Path, *, smoke_events: int | None = None,
            bootstrap_replicates: int = 500) -> dict:
    start_time = time.monotonic()
    inputs = _resolve_inputs(manifest_path)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    from realnu_parent_geometry_diagnostics import write_contracts

    geometry_cpu_contract = write_contracts(output_dir / "geometry")
    data, cache_status = _load_or_build_cache(inputs, output_dir, smoke_events=smoke_events)
    provenance = {**cache_status}
    provenance["comparison_manifest_path"] = str(inputs["manifest_path"])
    provenance["event_table_path"] = str(inputs["event_table_path"])
    provenance["full_validation_population"] = EXPECTED_EVENTS
    provenance["geometry_cpu_contract_status"] = geometry_cpu_contract["status"]
    if smoke_events is not None:
        result = _run_smoke(data, output_dir, provenance)
        return {**result, "runtime_seconds": time.monotonic() - start_time}

    summary = inputs["summary"]
    if summary.get("events") != EXPECTED_EVENTS or summary.get("samples") != EXPECTED_SAMPLES:
        raise ValueError("Full run refuses a population smaller than the accepted M32 validation")
    result = _analyze_full(data, {**inputs, **provenance}, output_dir,
                           bootstrap_replicates=bootstrap_replicates)
    manifest = {
        "schema": "realnu_mass_response_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "passed",
        "inputs": provenance,
        "population": {"events": EXPECTED_EVENTS, "posterior_samples": EXPECTED_SAMPLES,
                       "solver": "heun", "steps": 16, "parent_role_order": ["top", "antitop"]},
        "perfect_reco_finite_events": result["perfect_reco_support"],
        "sealed_test_used": False,
        "accepted_metric_regression": result["regression"],
        "cache_reused": cache_status["cache_reused"],
        "runtime_seconds": time.monotonic() - start_time,
        "outputs": {
            "report": result["report"],
            "cache": cache_status["cache_path"],
            "headline": str(output_dir / "metrics/headline.json"),
            "geometry_contracts": [
                str(output_dir / "geometry" / name)
                for name in ("rotation_contract.json", "gradient_contract.json", "caching_contract.json", "ablation_contract.json")
            ],
        },
    }
    _json_write(output_dir / "manifest.json", manifest)
    result["runtime_seconds"] = manifest["runtime_seconds"]
    result["manifest"] = str(output_dir / "manifest.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-events", type=int)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    if args.bootstrap_replicates < 500:
        raise ValueError("Use at least 500 paired event-bootstrap replicates for the full audit")
    result = analyse(args.comparison_manifest, args.output_dir,
                     smoke_events=args.smoke_events,
                     bootstrap_replicates=args.bootstrap_replicates)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
