"""Lightweight schemas and validators for self-describing prediction HDF5 files."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from joint_event_reconstruction.assignment_semantics import (
    NON_NULL_SUPPORT,
    NULL_INCLUDING_SUPPORT,
    read_hdf5_support,
)


PREDICTION_SCHEMA = "joint_event_reconstruction_predictions_v1"
DISCRETE_PREDICTION_SCHEMA = "joint_event_reconstruction_discrete_flow_predictions_v1"
JOINT_PREDICTION_SCHEMA = "joint_event_reconstruction_joint_flow_predictions_v1"


def _read_invisible_roles(source: h5py.File, expected: int, label: str) -> list[str]:
    raw = source.attrs.get("invisible_roles")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        roles = json.loads(raw)
    except (TypeError, ValueError, UnicodeDecodeError) as error:
        raise ValueError(f"{label} has invalid invisible_roles metadata") from error
    if (
        not isinstance(roles, list)
        or not all(isinstance(role, str) for role in roles)
        or len(roles) != expected
        or len(set(roles)) != len(roles)
    ):
        raise ValueError(f"{label} has inconsistent invisible_roles metadata")
    return roles


def _require_integer(values: np.ndarray, label: str) -> None:
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{label} must contain integer values")


def _require_finite(values: np.ndarray, label: str) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")


def _require_finite_or_missing(values: np.ndarray, label: str) -> None:
    """Allow an all-NaN four-vector for an undefined composite hypothesis."""
    if np.isinf(values).any() or (
        np.isnan(values).any(axis=-1) & ~np.isnan(values).all(axis=-1)
    ).any():
        raise ValueError(f"{label} contains invalid non-finite values")


def validate_prediction(path: str | Path) -> None:
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema_version") != PREDICTION_SCHEMA:
            raise ValueError(f"Invalid prediction schema in {path}")
        support = read_hdf5_support(source)
        if support["name"] != NON_NULL_SUPPORT:
            raise ValueError(
                "Discriminative complete prediction must use non-null support"
            )
        count = len(source["EVENT_ID"])
        if count == 0 or len(np.unique(source["EVENT_ID"][:])) != count:
            raise ValueError(f"Prediction output {path} is empty or has duplicate EVENT_ID")
        top_k = int(source.attrs["K"])
        if top_k <= 0:
            raise ValueError("Prediction output must have a positive K")
        try:
            posterior_samples = int(source.attrs["M"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Prediction output is missing a valid M") from error
        if posterior_samples <= 0:
            raise ValueError("Prediction output must have a positive M")
        log_scores = source["VISIBLE/assignment_log_scores"][:]
        probabilities = source["VISIBLE/constrained_probabilities"][:]
        if (
            log_scores.ndim != 2
            or probabilities.shape != log_scores.shape
            or probabilities.shape[0] != count
            or probabilities.shape[1] != top_k
            or not np.isfinite(log_scores).all()
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities.sum(axis=1) <= 0).any()
        ):
            raise ValueError(
                "Prediction output has invalid assignment scores/probabilities: "
                f"scores={log_scores.shape}, probabilities={probabilities.shape}"
            )
        exact_logz = source["VISIBLE/exact_logZ"][:]
        if (
            exact_logz.shape != (count,)
            or not np.isfinite(exact_logz).all()
        ):
            raise ValueError("Prediction output has invalid VISIBLE/exact_logZ")
        expected = np.exp(log_scores.astype(np.float64) - exact_logz[:, None])
        if (
            not np.isfinite(expected).all()
            or not np.allclose(probabilities, expected, rtol=1.0e-5, atol=1.0e-7)
            or (probabilities.sum(axis=1) > 1.0 + 1.0e-5).any()
        ):
            raise ValueError(
                "Prediction output probabilities are inconsistent with exact_logZ"
            )
        roots = source["VISIBLE/root_candidate_indices"][:]
        if (
            roots.ndim != 3
            or roots.shape[0:2] != (count, top_k)
            or roots.shape[2] <= 0
        ):
            raise ValueError("Prediction output has invalid root candidate indices")
        _require_integer(roots, "Prediction root candidate indices")
        if (roots < 0).any():
            raise ValueError("Prediction root candidate indices must be non-negative")
        njet = source["OBSERVED/njet"][:]
        if njet.shape != (count,):
            raise ValueError("Prediction output has invalid OBSERVED/njet")
        _require_finite(njet, "Prediction OBSERVED/njet")
        nodes = source["VISIBLE/nodes"]
        if len(nodes) == 0:
            raise ValueError("Prediction output has no visible node diagnostics")
        for node in nodes:
            for field in (
                "selected_candidate",
                "truth_candidate",
                "reconstructable",
                "truth_probability",
                "entropy",
                "margin",
            ):
                name = f"VISIBLE/nodes/{node}/{field}"
                values = source[name][:]
                if values.shape != (count,):
                    raise ValueError(f"Prediction output has invalid {name}")
                if field in {"selected_candidate", "truth_candidate"}:
                    _require_integer(values, name)
                elif field == "reconstructable":
                    if values.dtype != np.bool_:
                        raise ValueError(f"{name} must contain boolean values")
                else:
                    _require_finite(values, name)
        truth_topk = source["TRUTH/topk_assignment"][:]
        if truth_topk.shape != probabilities.shape or truth_topk.dtype != np.bool_:
            raise ValueError("Prediction output has invalid TRUTH/topk_assignment")
        if "INVISIBLE/samples" in source:
            samples = source["INVISIBLE/samples"]
            if (
                len(samples) != count
                or samples.ndim != 5
                or samples.shape[1] != top_k
                or samples.shape[2] != posterior_samples
                or samples.shape[4] != 3
                or not np.isfinite(samples[:]).all()
            ):
                raise ValueError("Prediction output has invalid invisible samples")
            roles = _read_invisible_roles(source, samples.shape[3], "Prediction output")
            for name in ("INVISIBLE/mean", "INVISIBLE/std"):
                values = source[name][:]
                if values.shape != (count, top_k, len(roles), 3):
                    raise ValueError(f"Prediction output has invalid {name}")
                _require_finite(values, name)
            if "TRUTH/NEUTRINO" not in source:
                raise ValueError("Prediction output has invalid invisible truth")
            truth = source["TRUTH/NEUTRINO"][:]
            if (
                truth.shape != (count, samples.shape[3], 3)
                or not np.isfinite(truth).all()
            ):
                raise ValueError("Prediction output has invalid invisible truth")
            branches = int(
                source["INVISIBLE"].attrs[
                    "unique_assignment_conditioning_branches"
                ]
            )
            if branches not in {1, int(source.attrs["K"])}:
                raise ValueError("Factorized invisible branch count must be 1 or K")
        if "OBSERVED/MET" in source:
            met = source["OBSERVED/MET"][:]
            if met.shape != (count, 4):
                raise ValueError("Prediction output has invalid OBSERVED/MET")
            _require_finite(met, "Prediction OBSERVED/MET")
        if "COMPOSITES" in source:
            if "TRUTH/COMPOSITES" not in source:
                raise ValueError("Prediction output is missing truth composites")
            truth_composites = source["TRUTH/COMPOSITES"]
            for node in source["COMPOSITES"]:
                values = source[f"COMPOSITES/{node}"][:]
                truth_values = truth_composites[node][:]
                if (
                    values.shape != (count, top_k, posterior_samples, 4)
                    or truth_values.shape != (count, 4)
                ):
                    raise ValueError(f"Prediction output has invalid composite {node}")
                _require_finite_or_missing(values, f"Prediction COMPOSITES/{node}")
                _require_finite_or_missing(
                    truth_values, f"Prediction TRUTH/COMPOSITES/{node}"
                )
        joint = source["JOINT"]
        if "compatibility_logit" in joint:
            for name in ("compatibility_logit", "final_score", "rank_indices"):
                values = joint[name][:]
                if values.shape != (count, top_k):
                    raise ValueError(f"Prediction output has invalid JOINT/{name}")
                if name == "rank_indices":
                    _require_integer(values, f"Prediction JOINT/{name}")
                    if not np.all(
                        np.sort(values, axis=1) == np.arange(top_k)[None, :]
                    ):
                        raise ValueError("Prediction JOINT/rank_indices is not a permutation")
                else:
                    _require_finite(values, f"Prediction JOINT/{name}")
            if "compatibility_weight" not in joint.attrs:
                raise ValueError("Prediction output is missing compatibility_weight")
            if not np.isfinite(float(joint.attrs["compatibility_weight"])):
                raise ValueError("Prediction compatibility_weight is not finite")
        elif any(name in joint for name in ("final_score", "rank_indices")):
            raise ValueError("Prediction output has incomplete JOINT compatibility data")


def validate_discrete_prediction(path: str | Path) -> None:
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema_version") != DISCRETE_PREDICTION_SCHEMA:
            raise ValueError(f"Invalid discrete-flow prediction schema in {path}")
        support = read_hdf5_support(source)
        if support["name"] != NULL_INCLUDING_SUPPORT:
            raise ValueError("DFM prediction must use NULL-including clean support")
        count = len(source["EVENT_ID"])
        if count == 0 or len(np.unique(source["EVENT_ID"][:])) != count:
            raise ValueError(f"Prediction output {path} is empty or has duplicate EVENT_ID")
        for name in (
            "HYPOTHESES/assignment_state",
            "HYPOTHESES/root_candidate_indices",
            "HYPOTHESES/node_candidate_indices",
            "TRUTH/assignment_state",
            "TRUTH/root_candidate_indices",
            "TRUTH/node_candidate_indices",
            "OBSERVED/njet",
        ):
            if len(source[name]) != count:
                raise ValueError(f"Prediction output has inconsistent {name} length")
        states = source["HYPOTHESES/assignment_state"][:]
        try:
            posterior_samples = int(source.attrs["posterior_samples"])
            state_count = int(source.attrs["assignment_state_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Discrete-flow prediction has invalid state metadata") from error
        truth_state = source["TRUTH/assignment_state"][:]
        if (
            posterior_samples <= 0
            or state_count <= 0
            or states.ndim != 2
            or states.shape[0] != count
            or states.shape[1] != posterior_samples
            or truth_state.shape != (count,)
        ):
            raise ValueError("Discrete-flow prediction has invalid assignment state metadata")
        _require_integer(states, "Discrete-flow assignment states")
        _require_integer(truth_state, "Discrete-flow truth states")
        for name in (
            "HYPOTHESES/root_candidate_indices",
            "HYPOTHESES/node_candidate_indices",
        ):
            values = source[name][:]
            if values.ndim != 3 or values.shape[:2] != states.shape:
                raise ValueError(f"Discrete-flow prediction has invalid {name}")
            _require_integer(values, name)
        for name in (
            "TRUTH/root_candidate_indices",
            "TRUTH/node_candidate_indices",
        ):
            values = source[name][:]
            if values.ndim != 2 or values.shape[0] != count:
                raise ValueError(f"Discrete-flow prediction has invalid {name}")
            _require_integer(values, name)
        truth_hit = source["TRUTH/hypothesis_assignment"][:]
        if truth_hit.shape != (count, posterior_samples) or truth_hit.dtype != np.bool_:
            raise ValueError("Discrete-flow prediction has invalid truth-hit metadata")
        njet = source["OBSERVED/njet"][:]
        if njet.shape != (count,):
            raise ValueError("Discrete-flow prediction has invalid OBSERVED/njet")
        _require_finite(njet, "Discrete-flow OBSERVED/njet")
        if ((states < 0) | (states >= state_count)).any() or (
            (truth_state < 0) | (truth_state >= state_count)
        ).any():
            raise ValueError("Discrete-flow prediction contains invalid clean state indices")


def validate_joint_prediction(path: str | Path) -> None:
    with h5py.File(path, "r") as source:
        if source.attrs.get("schema_version") != JOINT_PREDICTION_SCHEMA:
            raise ValueError(f"Invalid joint-flow prediction schema in {path}")
        support = read_hdf5_support(source)
        if support["name"] != NULL_INCLUDING_SUPPORT:
            raise ValueError("Joint-flow prediction must use NULL-including clean support")
        count = len(source["EVENT_ID"])
        if count == 0 or len(np.unique(source["EVENT_ID"][:])) != count:
            raise ValueError(f"Prediction output {path} is empty or has duplicate EVENT_ID")
        for name in (
            "HYPOTHESES/assignment_state",
            "HYPOTHESES/root_candidate_indices",
            "HYPOTHESES/node_candidate_indices",
            "HYPOTHESES/invisible",
            "TRUTH/assignment_state",
            "TRUTH/root_candidate_indices",
            "TRUTH/NEUTRINO",
        ):
            if len(source[name]) != count:
                raise ValueError(f"Prediction output has inconsistent {name} length")
        states = source["HYPOTHESES/assignment_state"][:]
        try:
            posterior_samples = int(source.attrs["posterior_samples"])
            state_count = int(source.attrs["assignment_state_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Joint-flow prediction has invalid state metadata") from error
        truth_state = source["TRUTH/assignment_state"][:]
        invisible = source["HYPOTHESES/invisible"][:]
        if (
            posterior_samples <= 0
            or state_count <= 0
            or states.ndim != 2
            or states.shape[0] != count
            or states.shape[1] != posterior_samples
            or invisible.ndim != 4
            or invisible.shape[:2] != states.shape
            or invisible.shape[3] != 3
            or truth_state.shape != (count,)
        ):
            raise ValueError("Joint-flow prediction has invalid assignment state metadata")
        if ((states < 0) | (states >= state_count)).any() or (
            (truth_state < 0) | (truth_state >= state_count)
        ).any():
            raise ValueError("Joint-flow prediction contains invalid clean state indices")
        truth = source["TRUTH/NEUTRINO"][:]
        if (
            truth.shape != (count, invisible.shape[2], 3)
            or not np.isfinite(truth).all()
        ):
            raise ValueError("Joint-flow prediction has invalid invisible truth")
        if not np.isfinite(invisible).all():
            raise ValueError("Joint-flow invisible hypotheses contain non-finite values")
        _read_invisible_roles(source, invisible.shape[2], "Joint-flow prediction")
        _require_integer(states, "Joint-flow assignment states")
        _require_integer(truth_state, "Joint-flow truth states")
        for name in (
            "HYPOTHESES/root_candidate_indices",
            "HYPOTHESES/node_candidate_indices",
        ):
            values = source[name][:]
            if values.ndim != 3 or values.shape[:2] != states.shape:
                raise ValueError(f"Joint-flow prediction has invalid {name}")
            _require_integer(values, name)
        truth_roots = source["TRUTH/root_candidate_indices"][:]
        if truth_roots.ndim != 2 or truth_roots.shape[0] != count:
            raise ValueError("Joint-flow prediction has invalid truth root indices")
        _require_integer(truth_roots, "Joint-flow truth root indices")
        truth_hit = source["TRUTH/hypothesis_assignment"][:]
        if truth_hit.shape != (count, posterior_samples) or truth_hit.dtype != np.bool_:
            raise ValueError("Joint-flow prediction has invalid truth-hit metadata")
        njet = source["OBSERVED/njet"][:]
        if njet.shape != (count,):
            raise ValueError("Joint-flow prediction has invalid OBSERVED/njet")
        _require_finite(njet, "Joint-flow OBSERVED/njet")
        if "COMPOSITES" in source:
            if "TRUTH/COMPOSITES" not in source:
                raise ValueError("Joint-flow prediction is missing truth composites")
            truth_composites = source["TRUTH/COMPOSITES"]
            for node in source["COMPOSITES"]:
                values = source[f"COMPOSITES/{node}"][:]
                truth_values = truth_composites[node][:]
                if values.shape != (count, posterior_samples, 4) or truth_values.shape != (
                    count,
                    4,
                ):
                    raise ValueError(
                        f"Prediction output has invalid COMPOSITES/{node}"
                    )
                _require_finite_or_missing(values, f"Joint-flow COMPOSITES/{node}")
                _require_finite_or_missing(
                    truth_values, f"Joint-flow TRUTH/COMPOSITES/{node}"
                )
        if "OBSERVED/MET" in source:
            met = source["OBSERVED/MET"][:]
            if met.shape != (count, 4):
                raise ValueError("Joint-flow prediction has invalid OBSERVED/MET")
            _require_finite(met, "Joint-flow OBSERVED/MET")
