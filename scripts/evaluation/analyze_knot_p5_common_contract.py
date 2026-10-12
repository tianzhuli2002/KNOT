#!/usr/bin/env python3
"""Compare stored KNOT and P5 posteriors on the exact shared validation events."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

BOOTSTRAP_SEED = 42011
BOOTSTRAP_RESAMPLES = 2000
EXPECTED_INVISIBLE_ROLES = ["top.nu", "antitop.nu"]
EXPECTED_PARENT_ROLES = ["top", "antitop"]
EMPTY_TEST_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def event_digest(event_ids: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(event_ids, dtype="<i8").tobytes()).hexdigest()


def _json_attr(source: h5py.File, name: str) -> object:
    value = source.attrs[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def _required(source: h5py.File, name: str, path: Path) -> np.ndarray:
    if name not in source:
        raise KeyError(f"Required dataset {name!r} is missing from {path}")
    return np.asarray(source[name][:])


def _required_rows(
    source: h5py.File, name: str, path: Path, rows: np.ndarray
) -> np.ndarray:
    if name not in source:
        raise KeyError(f"Required dataset {name!r} is missing from {path}")
    order = np.argsort(rows)
    sorted_values = np.asarray(source[name][rows[order]])
    return sorted_values[np.argsort(order)]


def _unique_ids(values: np.ndarray, label: str) -> np.ndarray:
    ids = np.asarray(values, dtype=np.int64)
    if ids.ndim != 1 or len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{label} EVENT_ID must be non-empty, one-dimensional, and unique")
    return ids


def _finite(values: np.ndarray, label: str, path: Path) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError(f"Non-finite values in {label} from {path}")
    return result


def invariant_mass(four_vector: np.ndarray, label: str, path: Path) -> np.ndarray:
    vector = _finite(four_vector, label, path)
    if vector.shape[-1] != 4:
        raise ValueError(f"{label} must end in (E,px,py,pz), got {vector.shape}")
    mass_squared = vector[..., 0] ** 2 - np.sum(vector[..., 1:] ** 2, axis=-1)
    scale = np.maximum(vector[..., 0] ** 2, 1.0)
    if np.any(mass_squared < -1.0e-7 * scale):
        minimum = float(np.min(mass_squared))
        raise ValueError(f"{label} contains materially spacelike vectors: min m^2={minimum}")
    return np.sqrt(np.maximum(mass_squared, 0.0))


def paired_metric(
    first_mse: np.ndarray,
    second_mse: np.ndarray,
    first_abs_error: np.ndarray,
    second_abs_error: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict:
    if first_mse.shape != second_mse.shape or first_mse.ndim != 1:
        raise ValueError("Paired event MSE arrays must be one-dimensional and equal-sized")
    rmse_difference = np.sqrt(first_mse[bootstrap_indices].mean(axis=1)) - np.sqrt(
        second_mse[bootstrap_indices].mean(axis=1)
    )
    mae_difference = first_abs_error[bootstrap_indices].mean(axis=1) - second_abs_error[
        bootstrap_indices
    ].mean(axis=1)
    return {
        "first_minus_second_rmse": float(
            np.sqrt(first_mse.mean()) - np.sqrt(second_mse.mean())
        ),
        "first_minus_second_rmse_event_bootstrap_95pct": [
            float(value) for value in np.quantile(rmse_difference, [0.025, 0.975])
        ],
        "first_minus_second_mean_absolute_error": float(
            first_abs_error.mean() - second_abs_error.mean()
        ),
        "first_minus_second_mae_event_bootstrap_95pct": [
            float(value) for value in np.quantile(mae_difference, [0.025, 0.975])
        ],
    }


def _paired_binary_difference(
    first_correct: np.ndarray,
    second_correct: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict:
    delta = first_correct.astype(np.float64) - second_correct.astype(np.float64)
    bootstrap_delta = delta[bootstrap_indices].mean(axis=1)
    return {
        "first_minus_second_percentage_points": float(100.0 * delta.mean()),
        "event_bootstrap_95pct_percentage_points": [
            float(100.0 * value) for value in np.quantile(bootstrap_delta, [0.025, 0.975])
        ],
    }


def _model_summary(
    mtt_samples: np.ndarray,
    parent_samples: np.ndarray,
    neutrino_samples: np.ndarray,
    truth_mtt: np.ndarray,
    truth_parents: np.ndarray,
    truth_neutrinos: np.ndarray,
) -> tuple[dict, dict]:
    mtt_mean = mtt_samples.mean(axis=1)
    mtt_residual = mtt_mean - truth_mtt
    parent_pt_samples = np.linalg.norm(parent_samples[..., 1:3], axis=-1)
    if np.any(parent_pt_samples <= 0):
        raise ValueError("Parent posterior contains zero transverse momentum; eta is undefined")
    parent_pt_mean = parent_pt_samples.mean(axis=1)
    truth_parent_pt = np.linalg.norm(truth_parents[..., 1:3], axis=-1)
    parent_eta_samples = np.arcsinh(parent_samples[..., 3] / parent_pt_samples)
    parent_eta_mean = parent_eta_samples.mean(axis=1)
    if np.any(truth_parent_pt <= 0):
        raise ValueError("Partonic parent truth contains zero transverse momentum; eta is undefined")
    truth_parent_eta = np.arcsinh(truth_parents[..., 3] / truth_parent_pt)
    neutrino_mean = neutrino_samples.mean(axis=1)
    neutrino_residual = neutrino_mean - truth_neutrinos
    neutrino_norm = np.linalg.norm(neutrino_residual, axis=-1).mean(axis=-1)
    high_mtt_mask = truth_mtt >= np.quantile(truth_mtt, 0.9)
    high_mtt_residual = mtt_residual[high_mtt_mask]

    coverage = {}
    for label, tail in (("50", 0.25), ("68", 0.16), ("90", 0.05)):
        lower = np.quantile(mtt_samples, tail, axis=1)
        upper = np.quantile(mtt_samples, 1.0 - tail, axis=1)
        coverage[f"mtt_{label}_coverage"] = float(
            np.mean((truth_mtt >= lower) & (truth_mtt <= upper))
        )

    metrics = {
        "events": len(truth_mtt),
        "posterior_samples_per_event": int(mtt_samples.shape[1]),
        "mtt_bias_GeV": float(mtt_residual.mean()),
        "mtt_rmse_GeV": float(np.sqrt(np.mean(mtt_residual**2))),
        "mtt_response_slope": float(
            np.cov(truth_mtt, mtt_mean, ddof=0)[0, 1] / np.var(truth_mtt)
        ),
        "mtt_truth_top_decile_threshold_GeV": float(np.quantile(truth_mtt, 0.9)),
        "mtt_truth_top_decile_events": int(high_mtt_mask.sum()),
        "mtt_truth_top_decile_bias_GeV": float(high_mtt_residual.mean()),
        "mtt_truth_top_decile_rmse_GeV": float(
            np.sqrt(np.mean(high_mtt_residual**2))
        ),
        "parent_top_pt_rmse_GeV": float(
            np.sqrt(np.mean((parent_pt_mean[:, 0] - truth_parent_pt[:, 0]) ** 2))
        ),
        "parent_antitop_pt_rmse_GeV": float(
            np.sqrt(np.mean((parent_pt_mean[:, 1] - truth_parent_pt[:, 1]) ** 2))
        ),
        "parent_top_eta_rmse": float(
            np.sqrt(np.mean((parent_eta_mean[:, 0] - truth_parent_eta[:, 0]) ** 2))
        ),
        "parent_antitop_eta_rmse": float(
            np.sqrt(np.mean((parent_eta_mean[:, 1] - truth_parent_eta[:, 1]) ** 2))
        ),
        "neutrino_3vector_rmse_GeV": float(
            np.sqrt(np.mean(np.sum(neutrino_residual**2, axis=-1)))
        ),
        "mean_event_neutrino_3vector_error_GeV": float(neutrino_norm.mean()),
        **coverage,
    }
    event_errors = {
        "mtt_mse": mtt_residual**2,
        "mtt_abs": np.abs(mtt_residual),
        "top_pt_mse": (parent_pt_mean[:, 0] - truth_parent_pt[:, 0]) ** 2,
        "top_pt_abs": np.abs(parent_pt_mean[:, 0] - truth_parent_pt[:, 0]),
        "antitop_pt_mse": (parent_pt_mean[:, 1] - truth_parent_pt[:, 1]) ** 2,
        "antitop_pt_abs": np.abs(parent_pt_mean[:, 1] - truth_parent_pt[:, 1]),
        "top_eta_mse": (parent_eta_mean[:, 0] - truth_parent_eta[:, 0]) ** 2,
        "top_eta_abs": np.abs(parent_eta_mean[:, 0] - truth_parent_eta[:, 0]),
        "antitop_eta_mse": (parent_eta_mean[:, 1] - truth_parent_eta[:, 1]) ** 2,
        "antitop_eta_abs": np.abs(parent_eta_mean[:, 1] - truth_parent_eta[:, 1]),
        "nu3v_mse": np.sum(neutrino_residual**2, axis=-1).sum(axis=-1),
        "nu3v_abs": neutrino_norm,
        "mtt_top_decile_mask": high_mtt_mask,
        "mtt_top_decile_mse": mtt_residual[high_mtt_mask] ** 2,
        "mtt_top_decile_abs": np.abs(mtt_residual[high_mtt_mask]),
    }
    return metrics, event_errors


def analyze(knot_path: Path, p5_path: Path, input_path: Path) -> dict:
    with h5py.File(knot_path, "r") as knot, h5py.File(input_path, "r") as data:
        knot_ids = _unique_ids(_required(knot, "EVENT_ID", knot_path), "KNOT")
        all_input_ids = _unique_ids(_required(data, "EVENT_ID", input_path), "v4 input")
        split = _required(data, "SPLIT", input_path)
        if split.shape != all_input_ids.shape or not np.array_equal(np.unique(split), [0, 1]):
            raise ValueError("v4 SPLIT must contain exactly the train and validation labels 0,1")
        if bool(data.attrs.get("sealed_test_used", True)) or int(
            data.attrs.get("test_count", -1)
        ) != 0:
            raise ValueError("The v4 input must retain the sealed, empty test split")
        validation_ids = all_input_ids[split == 1]
        ordered_intersection = validation_ids[np.isin(validation_ids, knot_ids)]
        if not np.array_equal(ordered_intersection, knot_ids):
            raise ValueError("KNOT EVENT_ID sequence is not exactly an ordered subset of v4 validation")
        if bool(knot.attrs.get("sealed_test_used", True)):
            raise ValueError("KNOT posterior records sealed-test use")
        if _json_attr(knot, "invisible_roles") != EXPECTED_INVISIBLE_ROLES:
            raise ValueError("KNOT invisible role order is not top.nu, antitop.nu")
        if _json_attr(knot, "parent_roles") != EXPECTED_PARENT_ROLES:
            raise ValueError("KNOT parent role order is not top, antitop")
        knot_neutrino_truth = _finite(
            _required(knot, "TRUTH/neutrinos", knot_path), "KNOT neutrino truth", knot_path
        )
        knot_parent_truth = _finite(
            _required(knot, "TRUTH/parents", knot_path), "KNOT parent truth", knot_path
        )
        knot_nu_samples = _finite(
            _required(knot, "invisible", knot_path), "KNOT neutrino posterior", knot_path
        )[..., 1:4]
        knot_parent_samples = _finite(
            _required(knot, "parents", knot_path), "KNOT parent posterior", knot_path
        )
        knot_truth_assignment = _required(knot, "TRUTH/assignment_candidates", knot_path)
        knot_map_assignment = _required(
            knot, "diagnostics/assignment_MAP_candidates", knot_path
        )
        knot_assignment_support = _required(
            knot, "TRUTH/assignment_support", knot_path
        ).astype(bool)
        if knot_truth_assignment.shape != (len(knot_ids), 2) or knot_map_assignment.shape != (
            len(knot_ids), 2
        ):
            raise ValueError("KNOT assignment candidates must have [event,2] shape")
        if _json_attr(knot, "assignment_nodes") != ["antitop", "top"]:
            raise ValueError("KNOT assignment candidate role order is not antitop,top")
        if knot_assignment_support.shape != (len(knot_ids),):
            raise ValueError("KNOT truth assignment support must have [event] shape")
        data_row = {int(event_id): index for index, event_id in enumerate(all_input_ids)}
        common_rows = np.asarray([data_row[int(event_id)] for event_id in knot_ids], dtype=np.int64)
        v4_nu_truth = _required_rows(data, "LABELS/NEUTRINO", input_path, common_rows)
        v4_neutrinos = np.stack(
            [v4_nu_truth["px"], v4_nu_truth["py"], v4_nu_truth["pz"]], axis=-1
        ).astype(np.float64)
        if knot_neutrino_truth.shape != (len(knot_ids), 2, 4):
            raise ValueError(f"Unexpected KNOT neutrino truth shape {knot_neutrino_truth.shape}")
        if not np.allclose(
            knot_neutrino_truth[..., 1:4], v4_neutrinos, rtol=1.0e-7, atol=1.0e-4
        ):
            raise ValueError("KNOT and v4 after-FSR neutrino truth differ on the common IDs")
        v4_leptons = _required_rows(data, "INPUTS/LEPTON", input_path, common_rows)
        if not np.all(v4_leptons["charge"] == np.asarray([1.0, -1.0])[None, :]):
            raise ValueError("v4 leptons are not stored in positive- then negative-charge order")
        v4_lepton_p4 = np.stack(
            (
                v4_leptons["e"],
                v4_leptons["pt"] * np.cos(v4_leptons["phi"]),
                v4_leptons["pt"] * np.sin(v4_leptons["phi"]),
                v4_leptons["pt"] * np.sinh(v4_leptons["eta"]),
            ),
            axis=-1,
        ).astype(np.float64)
        knot_lepton_p4 = _finite(
            _required(knot, "measured/leptons", knot_path),
            "KNOT measured charge-ordered leptons",
            knot_path,
        )
        if not np.allclose(knot_lepton_p4, v4_lepton_p4, rtol=1.0e-7, atol=1.0e-4):
            raise ValueError("KNOT and v4 measured charge-ordered lepton four-vectors differ")
        v4_jets = _required_rows(data, "INPUTS/JET", input_path, common_rows)
        v4_jet_p4 = np.stack(
            (
                v4_jets["e"],
                v4_jets["pt"] * np.cos(v4_jets["phi"]),
                v4_jets["pt"] * np.sin(v4_jets["phi"]),
                v4_jets["pt"] * np.sinh(v4_jets["eta"]),
            ),
            axis=-1,
        ).astype(np.float64)
        v4_jet_support = np.isfinite(v4_jets["pt"]) & np.isfinite(v4_jet_p4).all(axis=-1)
        if not v4_jet_support.any(axis=1).all():
            raise ValueError("A common event has no finite measured v4 jets")
        knot_b_samples = _finite(
            _required(knot, "measured_b/sample_four_vectors", knot_path),
            "KNOT sampled measured b-jet four-vectors",
            knot_path,
        )
        if knot_b_samples.shape != (len(knot_ids), 32, 2, 4):
            raise ValueError(f"Unexpected KNOT sampled b-jet shape {knot_b_samples.shape}")
        maximum_b_match_difference = 0.0
        for start in range(0, len(knot_ids), 128):
            stop = min(start + 128, len(knot_ids))
            difference = np.max(
                np.abs(
                    knot_b_samples[start:stop, :, :, None, :]
                    - v4_jet_p4[start:stop, None, None, :, :]
                ),
                axis=-1,
            )
            difference = np.where(
                v4_jet_support[start:stop, None, None, :], difference, np.inf
            )
            nearest_difference = difference.min(axis=-1)
            nearest_jet = difference.argmin(axis=-1)
            maximum_b_match_difference = max(
                maximum_b_match_difference, float(nearest_difference.max())
            )
            if np.any(nearest_difference > 2.0e-4):
                raise ValueError("A KNOT sampled b jet is absent from the matched v4 measured jets")
            if np.any(nearest_jet[:, :, 0] == nearest_jet[:, :, 1]):
                raise ValueError("KNOT sampled top and antitop b jets are not distinct")

    with h5py.File(p5_path, "r") as p5:
        p5_ids = _unique_ids(_required(p5, "EVENT_ID", p5_path), "P5")
        split_attr = p5.attrs.get("split", "")
        if isinstance(split_attr, bytes):
            split_attr = split_attr.decode("utf-8")
        if str(split_attr) != "validation":
            raise ValueError(f"P5 prediction split must be validation, got {split_attr!r}")
        if int(p5.attrs.get("K", -1)) != 1 or int(p5.attrs.get("M", -1)) != 32:
            raise ValueError("P5 comparison requires top-1 assignment and 32 posterior samples")
        source_data_attr = p5.attrs.get("source_data", "")
        if isinstance(source_data_attr, bytes):
            source_data_attr = source_data_attr.decode("utf-8")
        if Path(str(source_data_attr)).resolve() != input_path.resolve():
            raise ValueError(
                f"P5 prediction source does not match the v4 input: {source_data_attr} != {input_path}"
            )
        if _json_attr(p5, "invisible_roles") != EXPECTED_INVISIBLE_ROLES:
            raise ValueError("P5 invisible role order is not top.nu, antitop.nu")
        p5_intersection = p5_ids[np.isin(p5_ids, knot_ids)]
        if not np.array_equal(p5_intersection, knot_ids):
            raise ValueError("P5 EVENT_ID intersection differs from ordered KNOT validation IDs")
        p5_row = {int(event_id): index for index, event_id in enumerate(p5_ids)}
        rows = np.asarray([p5_row[int(event_id)] for event_id in knot_ids], dtype=np.int64)
        p5_nu_truth = _finite(
            _required_rows(p5, "TRUTH/NEUTRINO", p5_path, rows),
            "P5 neutrino truth",
            p5_path,
        )
        if p5_nu_truth.shape != (len(knot_ids), 2, 3) or not np.allclose(
            p5_nu_truth, v4_neutrinos, rtol=1.0e-7, atol=1.0e-4
        ):
            raise ValueError("P5 and v4 after-FSR neutrino truth differ on the common IDs")
        p5_nu_samples = _finite(
            _required_rows(p5, "INVISIBLE/four_vectors", p5_path, rows),
            "P5 neutrino posterior",
            p5_path,
        )
        p5_top_samples = _finite(
            _required_rows(p5, "COMPOSITES/top", p5_path, rows),
            "P5 top posterior",
            p5_path,
        )
        p5_antitop_samples = _finite(
            _required_rows(p5, "COMPOSITES/antitop", p5_path, rows),
            "P5 antitop posterior",
            p5_path,
        )
        p5_truth_assignment = _required_rows(p5, "TRUTH/topk_assignment", p5_path, rows)
        p5_top_support = _required_rows(
            p5, "VISIBLE/nodes/top/reconstructable", p5_path, rows
        ).astype(bool)
        p5_antitop_support = _required_rows(
            p5, "VISIBLE/nodes/antitop/reconstructable", p5_path, rows
        ).astype(bool)
        if p5_nu_samples.shape != (len(knot_ids), 1, 32, 2, 4):
            raise ValueError(f"Unexpected P5 neutrino posterior shape {p5_nu_samples.shape}")
        p5_top_samples = p5_top_samples[:, 0]
        p5_antitop_samples = p5_antitop_samples[:, 0]
        if p5_top_samples.shape != (len(knot_ids), 32, 4) or p5_antitop_samples.shape != (
            len(knot_ids), 32, 4
        ):
            raise ValueError("P5 COMPOSITES parent samples must have [event,32,4] shape")
        p5_nu_samples = p5_nu_samples[:, 0, :, :, 1:4]
        if p5_truth_assignment.shape != (len(knot_ids), 1):
            raise ValueError("P5 truth assignment mask must have [event,1] shape")
        if p5_truth_assignment.dtype != np.bool_:
            raise ValueError("P5 truth assignment mask must be boolean")
        if p5_top_support.shape != (len(knot_ids),) or p5_antitop_support.shape != (
            len(knot_ids),
        ):
            raise ValueError("P5 assignment reconstructability must have [event] shape")

    truth_mtt = invariant_mass(
        knot_parent_truth.sum(axis=1), "KNOT truth parent pair", knot_path
    )
    knot_mtt_samples = invariant_mass(
        knot_parent_samples.sum(axis=2), "KNOT parent samples", knot_path
    )
    p5_parent_samples = np.stack((p5_top_samples, p5_antitop_samples), axis=2)
    p5_mtt_samples = invariant_mass(
        p5_top_samples + p5_antitop_samples, "P5 parent samples", p5_path
    )
    if knot_parent_samples.shape != (len(knot_ids), 32, 2, 4):
        raise ValueError(f"Unexpected KNOT parent posterior shape {knot_parent_samples.shape}")
    if knot_nu_samples.shape != (len(knot_ids), 32, 2, 3):
        raise ValueError(f"Unexpected KNOT neutrino posterior shape {knot_nu_samples.shape}")
    if knot_parent_truth.shape != (len(knot_ids), 2, 4):
        raise ValueError(f"Unexpected KNOT parent truth shape {knot_parent_truth.shape}")
    truth_neutrinos = knot_neutrino_truth[..., 1:4]
    knot_metrics, knot_errors = _model_summary(
        knot_mtt_samples,
        knot_parent_samples,
        knot_nu_samples,
        truth_mtt,
        knot_parent_truth,
        truth_neutrinos,
    )
    p5_metrics, p5_errors = _model_summary(
        p5_mtt_samples,
        p5_parent_samples,
        p5_nu_samples,
        truth_mtt,
        knot_parent_truth,
        truth_neutrinos,
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_indices = rng.integers(
        0, len(knot_ids), size=(BOOTSTRAP_RESAMPLES, len(knot_ids)), dtype=np.int32
    )
    paired = {
        "mtt_GeV": paired_metric(
            knot_errors["mtt_mse"], p5_errors["mtt_mse"],
            knot_errors["mtt_abs"], p5_errors["mtt_abs"], bootstrap_indices,
        ),
        "top_parent_pt_GeV": paired_metric(
            knot_errors["top_pt_mse"], p5_errors["top_pt_mse"],
            knot_errors["top_pt_abs"], p5_errors["top_pt_abs"], bootstrap_indices,
        ),
        "antitop_parent_pt_GeV": paired_metric(
            knot_errors["antitop_pt_mse"], p5_errors["antitop_pt_mse"],
            knot_errors["antitop_pt_abs"], p5_errors["antitop_pt_abs"], bootstrap_indices,
        ),
        "top_parent_eta": paired_metric(
            knot_errors["top_eta_mse"], p5_errors["top_eta_mse"],
            knot_errors["top_eta_abs"], p5_errors["top_eta_abs"], bootstrap_indices,
        ),
        "antitop_parent_eta": paired_metric(
            knot_errors["antitop_eta_mse"], p5_errors["antitop_eta_mse"],
            knot_errors["antitop_eta_abs"], p5_errors["antitop_eta_abs"], bootstrap_indices,
        ),
        "neutrino_3vector_GeV": paired_metric(
            knot_errors["nu3v_mse"], p5_errors["nu3v_mse"],
            knot_errors["nu3v_abs"], p5_errors["nu3v_abs"], bootstrap_indices,
        ),
    }
    if not np.array_equal(
        knot_errors["mtt_top_decile_mask"], p5_errors["mtt_top_decile_mask"]
    ):
        raise ValueError("KNOT and P5 truth-defined high-mtt populations differ")
    high_count = int(knot_errors["mtt_top_decile_mask"].sum())
    high_bootstrap_indices = rng.integers(
        0,
        high_count,
        size=(BOOTSTRAP_RESAMPLES, high_count),
        dtype=np.int32,
    )
    paired["mtt_truth_top_decile_GeV"] = paired_metric(
        knot_errors["mtt_top_decile_mse"],
        p5_errors["mtt_top_decile_mse"],
        knot_errors["mtt_top_decile_abs"],
        p5_errors["mtt_top_decile_abs"],
        high_bootstrap_indices,
    )
    common_assignment_support = knot_assignment_support & p5_top_support & p5_antitop_support
    if not common_assignment_support.any():
        raise ValueError("No common explicit assignment-support events in KNOT/P5")
    knot_assignment_correct = np.all(
        knot_map_assignment == knot_truth_assignment, axis=1
    )[common_assignment_support]
    p5_assignment_correct = p5_truth_assignment[:, 0].astype(bool)[common_assignment_support]
    assignment_bootstrap = rng.integers(
        0,
        int(common_assignment_support.sum()),
        size=(BOOTSTRAP_RESAMPLES, int(common_assignment_support.sum())),
        dtype=np.int32,
    )
    assignment = {
        "common_supported_events": int(common_assignment_support.sum()),
        "knot_map_accuracy": float(knot_assignment_correct.mean()),
        "p5_top1_accuracy": float(p5_assignment_correct.mean()),
        "knot_minus_p5": _paired_binary_difference(
            knot_assignment_correct, p5_assignment_correct, assignment_bootstrap
        ),
    }
    return {
        "contract": {
            "knot_prediction": str(knot_path),
            "p5_prediction": str(p5_path),
            "v4_input": str(input_path),
            "event_count": len(knot_ids),
            "event_id_sha256_ordered": event_digest(knot_ids),
            "input_sha256": sha256_file(input_path),
            "knot_prediction_sha256": sha256_file(knot_path),
            "p5_prediction_sha256": sha256_file(p5_path),
            "truth": "KNOT stored partonic after-FSR parents and charge-side neutrinos; P5 and v4 neutrino truth checked against the same EVENT_IDs",
            "roles": {"parents": EXPECTED_PARENT_ROLES, "neutrinos": EXPECTED_INVISIBLE_ROLES},
            "measured_input_checks": {
                "charge_ordered_lepton_four_vectors_match": True,
                "all_knot_sampled_b_jets_found_in_v4_input": True,
                "max_b_jet_four_vector_component_difference_GeV": maximum_b_match_difference,
                "four_vector_absolute_tolerance_GeV": 2.0e-4,
            },
            "posterior_samples": {"KNOT": 32, "P5": 32},
            "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES},
            "sealed_test_used": False,
        },
        "metrics": {"KNOT-J": knot_metrics, "P5": p5_metrics},
        "paired_event_bootstrap": paired,
        "complete_assignment": assignment,
    }


def render_summary(result: dict) -> str:
    knot = result["metrics"]["KNOT-J"]
    p5 = result["metrics"]["P5"]
    paired = result["paired_event_bootstrap"]
    assignment = result["complete_assignment"]
    lines = [
        "# KNOT-J versus P5 on the common v4 validation events",
        "",
        f"Matched events: {result['contract']['event_count']:,}; posterior draws: 32 per event.",
        "Truth is the stored partonic after-FSR parent pair and charge-side neutrinos. The P5/v4 neutrino truth is checked against the KNOT truth by exact EVENT_ID join.",
        "",
        "| Metric | KNOT-J | P5 | KNOT-J minus P5, paired 95% event-bootstrap interval |",
        "|---|---:|---:|---:|",
        f"| mtt RMSE [GeV] | {knot['mtt_rmse_GeV']:.3f} | {p5['mtt_rmse_GeV']:.3f} | {paired['mtt_GeV']['first_minus_second_rmse']:.3f} [{paired['mtt_GeV']['first_minus_second_rmse_event_bootstrap_95pct'][0]:.3f}, {paired['mtt_GeV']['first_minus_second_rmse_event_bootstrap_95pct'][1]:.3f}] |",
        f"| top parent pT RMSE [GeV] | {knot['parent_top_pt_rmse_GeV']:.3f} | {p5['parent_top_pt_rmse_GeV']:.3f} | {paired['top_parent_pt_GeV']['first_minus_second_rmse']:.3f} [{paired['top_parent_pt_GeV']['first_minus_second_rmse_event_bootstrap_95pct'][0]:.3f}, {paired['top_parent_pt_GeV']['first_minus_second_rmse_event_bootstrap_95pct'][1]:.3f}] |",
        f"| antitop parent pT RMSE [GeV] | {knot['parent_antitop_pt_rmse_GeV']:.3f} | {p5['parent_antitop_pt_rmse_GeV']:.3f} | {paired['antitop_parent_pt_GeV']['first_minus_second_rmse']:.3f} [{paired['antitop_parent_pt_GeV']['first_minus_second_rmse_event_bootstrap_95pct'][0]:.3f}, {paired['antitop_parent_pt_GeV']['first_minus_second_rmse_event_bootstrap_95pct'][1]:.3f}] |",
        f"| top parent eta RMSE | {knot['parent_top_eta_rmse']:.4f} | {p5['parent_top_eta_rmse']:.4f} | {paired['top_parent_eta']['first_minus_second_rmse']:.4f} [{paired['top_parent_eta']['first_minus_second_rmse_event_bootstrap_95pct'][0]:.4f}, {paired['top_parent_eta']['first_minus_second_rmse_event_bootstrap_95pct'][1]:.4f}] |",
        f"| antitop parent eta RMSE | {knot['parent_antitop_eta_rmse']:.4f} | {p5['parent_antitop_eta_rmse']:.4f} | {paired['antitop_parent_eta']['first_minus_second_rmse']:.4f} [{paired['antitop_parent_eta']['first_minus_second_rmse_event_bootstrap_95pct'][0]:.4f}, {paired['antitop_parent_eta']['first_minus_second_rmse_event_bootstrap_95pct'][1]:.4f}] |",
        f"| neutrino 3-vector RMSE [GeV] | {knot['neutrino_3vector_rmse_GeV']:.3f} | {p5['neutrino_3vector_rmse_GeV']:.3f} | {paired['neutrino_3vector_GeV']['first_minus_second_rmse']:.3f} [{paired['neutrino_3vector_GeV']['first_minus_second_rmse_event_bootstrap_95pct'][0]:.3f}, {paired['neutrino_3vector_GeV']['first_minus_second_rmse_event_bootstrap_95pct'][1]:.3f}] |",
        f"| mtt bias [GeV] | {knot['mtt_bias_GeV']:.3f} | {p5['mtt_bias_GeV']:.3f} | — |",
        f"| mtt 68% coverage | {knot['mtt_68_coverage']:.4f} | {p5['mtt_68_coverage']:.4f} | — |",
        f"| mtt truth top-decile RMSE [GeV] | {knot['mtt_truth_top_decile_rmse_GeV']:.3f} | {p5['mtt_truth_top_decile_rmse_GeV']:.3f} | {paired['mtt_truth_top_decile_GeV']['first_minus_second_rmse']:.3f} [{paired['mtt_truth_top_decile_GeV']['first_minus_second_rmse_event_bootstrap_95pct'][0]:.3f}, {paired['mtt_truth_top_decile_GeV']['first_minus_second_rmse_event_bootstrap_95pct'][1]:.3f}] |",
        f"| mtt truth top-decile bias [GeV] | {knot['mtt_truth_top_decile_bias_GeV']:.3f} | {p5['mtt_truth_top_decile_bias_GeV']:.3f} | — |",
        f"| supported complete assignment accuracy | {assignment['knot_map_accuracy']:.4f} | {assignment['p5_top1_accuracy']:.4f} | {assignment['knot_minus_p5']['first_minus_second_percentage_points']:.3f} pp [{assignment['knot_minus_p5']['event_bootstrap_95pct_percentage_points'][0]:.3f}, {assignment['knot_minus_p5']['event_bootstrap_95pct_percentage_points'][1]:.3f}] |",
        "",
        f"Assignment accuracy uses the {assignment['common_supported_events']:,}-event intersection with exact explicit support from both outputs. Parent pT/eta, neutrino, and mtt point estimates are posterior sample means; interval coverage uses the per-event posterior samples. The truth-defined mtt top decile uses the 90th-percentile threshold of the common truth sample. Exact assignment NLL is omitted because P5 stores top-1 assignment scores without the true complete-assignment log probability when truth is not top-1.",
        "",
        "The paired intervals resample events, never posterior draws. KNOT-J minus P5 is negative when KNOT-J has the lower error.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--knot",
        type=Path,
        default=Path("results/experiments/knot/matched-production-v1/evaluation-v1/full-10k/J/posterior.h5"),
    )
    parser.add_argument(
        "--p5",
        type=Path,
        default=Path("results/experiments/knot/jer-afterfsr-v1/training/P5/predictions/validation/prediction.h5"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/experiments/knot/jer-afterfsr-v1/data/jer_charge_side_after_fsr_v4.h5"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiments/knot/jer-afterfsr-v1/p5-knot-common-v1"),
    )
    args = parser.parse_args()
    for path in (args.knot, args.p5, args.input):
        if not path.is_file():
            raise FileNotFoundError(f"Required common-contract input is missing: {path}")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing comparison output: {args.output}")
    result = analyze(args.knot, args.p5, args.input)
    args.output.mkdir(parents=True)
    json_path = args.output / "metrics.json"
    markdown_path = args.output / "summary.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_summary(result))
    print(json.dumps({"status": "complete", "metrics": str(json_path), "summary": str(markdown_path)}))


if __name__ == "__main__":
    main()
