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


def _optional_framework_metadata(source: h5py.File, label: str) -> None:
    """Validate v2 provenance fields when present without rewriting old products."""
    representation = source.attrs.get("physics_vector_representation")
    if representation is not None:
        if isinstance(representation, bytes):
            representation = representation.decode("utf-8")
        if str(representation) != "E_px_py_pz_GeV_v1":
            raise ValueError(f"{label} has invalid physics_vector_representation")

    coordinates = source.attrs.get("invisible_coordinates")
    if coordinates is not None:
        if isinstance(coordinates, bytes):
            coordinates = coordinates.decode("utf-8")
        if str(coordinates) not in {"disabled", "cartesian", "met_residual", "met_exact"}:
            raise ValueError(f"{label} has invalid invisible_coordinates metadata")
        metadata = source.attrs.get("invisible_coordinate_metadata")
        if metadata is not None:
            if isinstance(metadata, bytes):
                metadata = metadata.decode("utf-8")
            try:
                parsed = json.loads(str(metadata))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{label} has invalid invisible_coordinate_metadata"
                ) from error
            if not isinstance(parsed, (dict, type(None))):
                raise ValueError(f"{label} has invalid invisible_coordinate_metadata")

    coupling = source.attrs.get("flow_coupling")
    if coupling is not None:
        if isinstance(coupling, bytes):
            coupling = coupling.decode("utf-8")
        try:
            parsed = json.loads(str(coupling))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{label} has invalid flow_coupling metadata") from error
        if parsed is not None and not isinstance(parsed, dict):
            raise ValueError(f"{label} has invalid flow_coupling metadata")

    if "visible_encoder_config" in source.attrs:
        raw = source.attrs["visible_encoder_config"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            parsed = json.loads(str(raw))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"{label} has invalid visible_encoder_config") from error
        if not isinstance(parsed, dict):
            raise ValueError(f"{label} has invalid visible_encoder_config")


def _validate_met_exact_projection(source: h5py.File, label: str, count: int) -> None:
    path = "TRUTH/MET_EXACT_PROJECTION"
    if path not in source:
        return
    coordinates = source.attrs.get("invisible_coordinates")
    if isinstance(coordinates, bytes):
        coordinates = coordinates.decode("utf-8")
    if coordinates != "met_exact":
        raise ValueError(f"{label} has MET_EXACT_PROJECTION without met_exact coordinates")
    group = source[path]
    required = {
        "NEUTRINO_PROJECTED",
        "px_shift",
        "py_shift",
        "per_role_transverse_magnitude",
    }
    if set(group) != required:
        raise ValueError(f"{label} has incomplete MET_EXACT_PROJECTION datasets")
    projected = group["NEUTRINO_PROJECTED"][:]
    truth = source["TRUTH/NEUTRINO"][:]
    if projected.shape != truth.shape or projected.shape[0] != count:
        raise ValueError(f"{label} has invalid projected invisible truth shape")
    shifts = group["px_shift"][:], group["py_shift"][:]
    magnitudes = group["per_role_transverse_magnitude"][:]
    if any(values.shape != (count,) for values in shifts):
        raise ValueError(f"{label} has invalid MET-exact projection shifts")
    if magnitudes.shape != truth.shape[:2]:
        raise ValueError(f"{label} has invalid MET-exact projection magnitudes")
    for name, values in (
        ("NEUTRINO_PROJECTED", projected),
        ("px_shift", shifts[0]),
        ("py_shift", shifts[1]),
        ("per_role_transverse_magnitude", magnitudes),
    ):
        _require_finite(values, f"{label} {path}/{name}")
    if "OBSERVED/MET" in source:
        met = source["OBSERVED/MET"][:, 1:3]
        if not np.allclose(projected[..., :2].sum(axis=1), met, rtol=2e-5, atol=2e-5):
            raise ValueError(f"{label} projected truth does not satisfy observed MET")


def validate_hypothesis_group(
    source: h5py.File | h5py.Group,
    *,
    count: int,
    top_k: int,
    label: str = "Prediction HYPOTHESES",
) -> None:
    """Validate the optional physics-aware hypothesis extension.

    Factorized prediction files written before hypothesis evaluation do not contain
    this group and remain valid under the original schema contract.
    """
    if "HYPOTHESES" not in source:
        return
    hypotheses = source["HYPOTHESES"]
    required = ("state", "visible_log_score", "visible_probability", "truth")
    missing = [name for name in required if name not in hypotheses]
    if missing:
        raise ValueError(f"{label} is missing datasets {missing}")
    state = hypotheses["state"][:]
    log_score = hypotheses["visible_log_score"][:]
    probability = hypotheses["visible_probability"][:]
    truth = hypotheses["truth"][:]
    if (
        state.ndim != 3
        or state.shape[:2] != (count, top_k)
        or log_score.shape != (count, top_k)
        or probability.shape != (count, top_k)
        or truth.shape != (count, top_k)
        or truth.dtype != np.bool_
    ):
        raise ValueError(f"{label} has invalid core dataset shapes")
    _require_integer(state, f"{label}/state")
    _require_finite(log_score, f"{label}/visible_log_score")
    _require_finite(probability, f"{label}/visible_probability")
    if (probability < 0).any() or (probability.sum(axis=1) <= 0).any():
        raise ValueError(f"{label}/visible_probability is invalid")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=1.0e-5, atol=1.0e-6):
        raise ValueError(f"{label}/visible_probability is not normalised")

    physics = hypotheses.get("PHYSICS")
    if physics is not None:
        required_physics = ("feature_mean", "feature_std", "feature_rms", "analytic_score")
        missing = [name for name in required_physics if name not in physics]
        if missing:
            raise ValueError(f"{label}/PHYSICS is missing datasets {missing}")
        means = physics["feature_mean"][:]
        std = physics["feature_std"][:]
        rms = physics["feature_rms"][:]
        analytic = physics["analytic_score"][:]
        if (
            means.ndim != 3
            or means.shape[:2] != (count, top_k)
            or std.shape != means.shape
            or rms.shape != means.shape
            or analytic.shape != (count, top_k)
        ):
            raise ValueError(f"{label}/PHYSICS has invalid dataset shapes")
        for name, values in (
            ("feature_mean", means),
            ("feature_std", std),
            ("feature_rms", rms),
            ("analytic_score", analytic),
        ):
            _require_finite(values, f"{label}/PHYSICS/{name}")
        raw_names = physics.attrs.get("feature_names")
        if raw_names is not None:
            if isinstance(raw_names, bytes):
                raw_names = raw_names.decode("utf-8")
            try:
                feature_names = json.loads(str(raw_names))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"{label}/PHYSICS has invalid feature_names") from error
            if (
                not isinstance(feature_names, list)
                or len(feature_names) != means.shape[-1]
                or not all(isinstance(name, str) and name for name in feature_names)
            ):
                raise ValueError(f"{label}/PHYSICS has invalid feature_names")
    for name in ("learned_delta", "final_log_score", "final_probability", "final_rank"):
        if name not in hypotheses:
            continue
        values = hypotheses[name][:]
        if values.shape != (count, top_k):
            raise ValueError(f"{label}/{name} has invalid shape")
        if name == "final_rank":
            _require_integer(values, f"{label}/{name}")
            if not np.all(np.sort(values, axis=1) == np.arange(top_k)[None, :]):
                raise ValueError(f"{label}/final_rank is not a permutation")
        else:
            _require_finite(values, f"{label}/{name}")


def validate_prediction(path: str | Path) -> None:
    with h5py.File(path, "r") as source:
        _optional_framework_metadata(source, "Prediction output")
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
        valid_hypotheses = source["VISIBLE/valid_hypotheses"][:]
        if (
            log_scores.ndim != 2
            or probabilities.shape != log_scores.shape
            or valid_hypotheses.shape != log_scores.shape
            or valid_hypotheses.dtype != np.bool_
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
        if not valid_hypotheses[:, 0].all() or (
            np.diff(valid_hypotheses.astype(np.int8), axis=1) > 0
        ).any():
            raise ValueError("Prediction hypothesis support must be a non-empty prefix")
        if (probabilities[~valid_hypotheses] != 0).any():
            raise ValueError("Unsupported prediction hypotheses must have zero probability")
        exact_logz = source["VISIBLE/exact_logZ"][:]
        if (
            exact_logz.shape != (count,)
            or not np.isfinite(exact_logz).all()
        ):
            raise ValueError("Prediction output has invalid VISIBLE/exact_logZ")
        expected = np.where(
            valid_hypotheses,
            np.exp(log_scores.astype(np.float64) - exact_logz[:, None]),
            0.0,
        )
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
            physical = source.get("INVISIBLE/four_vectors")
            if physical is not None:
                values = physical[:]
                expected = (count, top_k, posterior_samples, len(roles), 4)
                if values.shape != expected:
                    raise ValueError("Prediction output has invalid invisible four-vectors")
                _require_finite(values, "INVISIBLE/four_vectors")
                for name in ("INVISIBLE/four_vector_mean", "INVISIBLE/four_vector_std"):
                    values = source[name][:]
                    if values.shape != (count, top_k, len(roles), 4):
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
            truth_physical = source.get("TRUTH/INVISIBLE_FOUR_VECTOR")
            if truth_physical is not None:
                values = truth_physical[:]
                if values.shape != (count, samples.shape[3], 4):
                    raise ValueError("Prediction output has invalid invisible truth four-vectors")
                _require_finite(values, "TRUTH/INVISIBLE_FOUR_VECTOR")
            branches = int(
                source["INVISIBLE"].attrs[
                    "unique_assignment_conditioning_branches"
                ]
            )
            if branches not in {1, int(source.attrs["K"])}:
                raise ValueError("Factorized invisible branch count must be 1 or K")
            _validate_met_exact_projection(source, "Prediction output", count)
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
        validate_hypothesis_group(source, count=count, top_k=top_k)


def validate_discrete_prediction(path: str | Path) -> None:
    with h5py.File(path, "r") as source:
        _optional_framework_metadata(source, "Discrete-flow prediction")
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
        _optional_framework_metadata(source, "Joint-flow prediction")
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
        physical = source.get("HYPOTHESES/invisible_four_vectors")
        if physical is not None:
            values = physical[:]
            if values.shape != (count, posterior_samples, invisible.shape[2], 4):
                raise ValueError("Joint-flow prediction has invalid invisible four-vectors")
            _require_finite(values, "HYPOTHESES/invisible_four_vectors")
        truth_physical = source.get("TRUTH/INVISIBLE_FOUR_VECTOR")
        if truth_physical is not None:
            values = truth_physical[:]
            if values.shape != (count, invisible.shape[2], 4):
                raise ValueError("Joint-flow prediction has invalid invisible truth four-vectors")
            _require_finite(values, "TRUTH/INVISIBLE_FOUR_VECTOR")
        _read_invisible_roles(source, invisible.shape[2], "Joint-flow prediction")
        _validate_met_exact_projection(source, "Joint-flow prediction", count)
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
