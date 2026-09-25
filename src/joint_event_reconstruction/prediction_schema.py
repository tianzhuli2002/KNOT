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


def _iter_hdf5_chunks(
    dataset: h5py.Dataset, *, target_bytes: int = 32 * 1024 * 1024
):
    """Yield bounded first-axis chunks without materialising a large dataset."""
    if dataset.ndim == 0:
        yield dataset[()]
        return
    if dataset.shape[0] == 0:
        return
    trailing = int(np.prod(dataset.shape[1:], dtype=np.int64))
    row_bytes = max(1, trailing * dataset.dtype.itemsize)
    rows = max(1, int(target_bytes) // row_bytes)
    for start in range(0, dataset.shape[0], rows):
        yield dataset[start : min(dataset.shape[0], start + rows)]


def _require_finite_dataset(
    dataset: h5py.Dataset, label: str, *, deep: bool = True
) -> None:
    if not deep:
        return
    for chunk in _iter_hdf5_chunks(dataset):
        if not np.isfinite(chunk).all():
            raise ValueError(f"{label} contains non-finite values")


def validate_physical_posterior_shape(
    physical_samples,
    *,
    event_count: int,
    top_k: int,
    posterior_samples: int,
    n_invisible_particles: int,
    label: str = "physical posterior",
    deep: bool = True,
) -> None:
    """Validate the complete physical posterior tensor contract.

    The physical posterior is deliberately rank-five.  Its axes are
    ``(event, assignment hypothesis, posterior sample, invisible role,
    four-vector component)`` and the component convention is ``(E, px, py,
    pz)``.  Keeping this check as a public helper prevents callers from
    accidentally comparing a three-axis slice with a two-axis expectation.
    """
    expected = (
        int(event_count),
        int(top_k),
        int(posterior_samples),
        int(n_invisible_particles),
        4,
    )
    observed = tuple(getattr(physical_samples, "shape", ()))
    if observed != expected:
        raise ValueError(
            f"Unexpected {label} shape {observed}; expected {expected} "
            "for (events, hypotheses, samples, invisible_roles, E_px_py_pz)"
        )
    if deep and not np.isfinite(physical_samples).all():
        raise ValueError(f"{label} contains non-finite values")


def validate_physical_mass_shell(
    physical_samples,
    *,
    mass: float = 0.0,
    label: str = "physical posterior",
    ulp_tolerance: float = 8.0,
) -> dict[str, float | str]:
    """Validate an exact mass shell after the product's storage precision.

    Prediction HDF5 files store physical four-vectors as float32.  Comparing
    ``E**2 - |p|**2`` with a fixed absolute tolerance is therefore invalid for
    high-energy vectors: ordinary float32 round-off grows with the vector
    scale.  Compare the stored energy with the shell energy using a bounded
    number of storage-precision ulps instead.
    """
    values = np.asarray(physical_samples)
    if values.shape[-1:] != (4,):
        raise ValueError(f"{label} must end in (E, px, py, pz), observed {values.shape}")
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f"{label} must use a floating dtype, observed {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.isfinite(mass) or mass < 0:
        raise ValueError(f"{label} mass must be finite and non-negative")
    if (values[..., 0] < 0).any():
        raise ValueError(f"{label} contains negative energy")

    work = values.astype(np.float64, copy=False)
    spatial_squared = np.sum(work[..., 1:] ** 2, axis=-1)
    expected_energy = np.sqrt(spatial_squared + float(mass) ** 2)
    residual = np.abs(work[..., 0] - expected_energy)
    storage_epsilon = np.finfo(values.dtype).eps
    scale = np.maximum(1.0, np.maximum(np.abs(work[..., 0]), expected_energy))
    tolerance = float(ulp_tolerance) * storage_epsilon * scale
    if np.any(residual > tolerance):
        ratio = residual / np.maximum(tolerance, np.finfo(np.float64).tiny)
        raise ValueError(
            f"{label} is outside its configured mass shell after storage rounding: "
            f"max_energy_residual={float(residual.max()):.6g}, "
            f"max_allowed={float(tolerance.max()):.6g}, "
            f"max_ulp_ratio={float(ratio.max()):.6g}"
        )
    return {
        "max_energy_residual": float(residual.max(initial=0.0)),
        "max_relative_energy_residual": float(
            (residual / np.maximum(scale, np.finfo(np.float64).tiny)).max(initial=0.0)
        ),
        "storage_dtype": str(values.dtype),
        "ulp_tolerance": float(ulp_tolerance),
    }


def _require_score_dataset(
    dataset: h5py.Dataset,
    valid: np.ndarray,
    label: str,
    *,
    deep: bool = True,
) -> None:
    """Allow ``-inf`` only on padded score entries."""
    if not deep:
        return
    values = dataset[:]
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError(f"{label} contains NaN or +inf")
    if not np.isfinite(values[valid]).all():
        raise ValueError(f"{label} contains a non-finite valid value")


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
    deep: bool = True,
) -> None:
    """Validate the optional physics-aware hypothesis extension.

    Factorized prediction files written before hypothesis evaluation do not contain
    this group and remain valid under the original schema contract.
    """
    if "HYPOTHESES" not in source:
        return
    hypotheses = source["HYPOTHESES"]
    if hypotheses.attrs.get("schema_version") != "physics_hypotheses_v1":
        raise ValueError(f"{label} has an unsupported schema_version")
    required = (
        "state",
        "valid_hypotheses",
        "visible_log_score",
        "visible_probability",
        "truth",
    )
    missing = [name for name in required if name not in hypotheses]
    if missing:
        raise ValueError(f"{label} is missing datasets {missing}")
    state = hypotheses["state"][:]
    valid = hypotheses["valid_hypotheses"][:]
    log_score = hypotheses["visible_log_score"][:]
    probability = hypotheses["visible_probability"][:]
    truth = hypotheses["truth"][:]
    if (
        state.ndim != 3
        or state.shape[:2] != (count, top_k)
        or valid.shape != (count, top_k)
        or valid.dtype != np.bool_
        or log_score.shape != (count, top_k)
        or probability.shape != (count, top_k)
        or truth.shape != (count, top_k)
        or truth.dtype != np.bool_
    ):
        raise ValueError(f"{label} has invalid core dataset shapes")
    _require_integer(state, f"{label}/state")
    if not valid.any(axis=1).all():
        raise ValueError(f"{label}/valid_hypotheses has an empty event")
    if not valid[:, 0].all() or (
        np.diff(valid.astype(np.int8), axis=1) > 0
    ).any():
        raise ValueError(f"{label}/valid_hypotheses must be a non-empty prefix")
    _require_score_dataset(
        hypotheses["visible_log_score"], valid, f"{label}/visible_log_score", deep=deep
    )
    _require_finite_dataset(
        hypotheses["visible_probability"],
        f"{label}/visible_probability",
        deep=deep,
    )
    if deep and ((probability < 0).any() or (probability.sum(axis=1) <= 0).any()):
        raise ValueError(f"{label}/visible_probability is invalid")
    if (probability[~valid] != 0).any():
        raise ValueError(f"{label}/visible_probability is non-zero outside support")
    if deep and not np.allclose(
        probability.sum(axis=1), 1.0, rtol=1.0e-5, atol=1.0e-6
    ):
        raise ValueError(f"{label}/visible_probability is not normalised")
    if truth[~valid].any():
        raise ValueError(f"{label}/truth is true outside valid support")

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
        ):
            _require_finite_dataset(physics[name], f"{label}/PHYSICS/{name}", deep=deep)
            if (values[~valid] != 0).any():
                raise ValueError(f"{label}/PHYSICS/{name} is non-zero outside support")
        _require_score_dataset(
            physics["analytic_score"], valid, f"{label}/PHYSICS/analytic_score", deep=deep
        )
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
            if (values < 0).any() or (values >= top_k).any():
                raise ValueError(f"{label}/{name} contains an out-of-range index")
            if not np.all(np.sort(values, axis=1) == np.arange(top_k)[None, :]):
                raise ValueError(f"{label}/final_rank is not a permutation")
            ordered_valid = np.take_along_axis(valid, values, axis=1)
            if not ordered_valid[:, 0].all() or (
                np.diff(ordered_valid.astype(np.int8), axis=1) > 0
            ).any():
                raise ValueError(f"{label}/final_rank puts invalid hypotheses before valid ones")
        elif name in {"final_log_score"}:
            _require_score_dataset(
                hypotheses[name], valid, f"{label}/{name}", deep=deep
            )
        elif name == "learned_delta":
            _require_finite_dataset(hypotheses[name], f"{label}/{name}", deep=deep)
            if (values[~valid] != 0).any():
                raise ValueError(f"{label}/{name} is non-zero outside support")
        else:
            _require_finite_dataset(hypotheses[name], f"{label}/{name}", deep=deep)
            if (values[~valid] != 0).any():
                raise ValueError(f"{label}/{name} is non-zero outside support")
            if deep and not np.allclose(
                values.sum(axis=1), 1.0, rtol=1.0e-5, atol=1.0e-6
            ):
                raise ValueError(f"{label}/{name} is not normalised")


def validate_prediction(path: str | Path, *, deep: bool = True) -> None:
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
            ):
                raise ValueError("Prediction output has invalid invisible samples")
            _require_finite_dataset(samples, "INVISIBLE/samples", deep=deep)
            roles = _read_invisible_roles(source, samples.shape[3], "Prediction output")
            for name in ("INVISIBLE/mean", "INVISIBLE/std"):
                values = source[name]
                if values.shape != (count, top_k, len(roles), 3):
                    raise ValueError(f"Prediction output has invalid {name}")
                _require_finite_dataset(values, name, deep=deep)
            physical = source.get("INVISIBLE/four_vectors")
            if physical is not None:
                validate_physical_posterior_shape(
                    physical[:],
                    event_count=count,
                    top_k=top_k,
                    posterior_samples=posterior_samples,
                    n_invisible_particles=len(roles),
                    label="INVISIBLE/four_vectors",
                    deep=deep,
                )
                for name in ("INVISIBLE/four_vector_mean", "INVISIBLE/four_vector_std"):
                    values = source[name]
                    if values.shape != (count, top_k, len(roles), 4):
                        raise ValueError(f"Prediction output has invalid {name}")
                    _require_finite_dataset(values, name, deep=deep)
            invisible_truth_available = bool(
                source.attrs.get("invisible_truth_available", True)
            )
            if invisible_truth_available:
                if "TRUTH/NEUTRINO" not in source:
                    raise ValueError("Prediction output has invalid invisible truth")
                truth = source["TRUTH/NEUTRINO"][:]
                if (
                    truth.shape != (count, samples.shape[3], 3)
                    or not np.isfinite(truth).all()
                ):
                    raise ValueError("Prediction output has invalid invisible truth")
            truth_physical = source.get("TRUTH/INVISIBLE_FOUR_VECTOR")
            if truth_physical is not None and invisible_truth_available:
                values = truth_physical[:]
                if values.shape != (count, samples.shape[3], 4):
                    raise ValueError("Prediction output has invalid invisible truth four-vectors")
                _require_finite_dataset(
                    truth_physical, "TRUTH/INVISIBLE_FOUR_VECTOR", deep=deep
                )
            branches = int(
                source["INVISIBLE"].attrs[
                    "unique_assignment_conditioning_branches"
                ]
            )
            if branches not in {1, int(source.attrs["K"])}:
                raise ValueError("Factorized invisible branch count must be 1 or K")
            _validate_met_exact_projection(source, "Prediction output", count)
        if "INVISIBLE/point_momenta" in source:
            point = source["INVISIBLE/point_momenta"]
            if (
                point.shape[0] != count
                or point.ndim != 4
                or point.shape[1] != top_k
                or point.shape[3] != 3
            ):
                raise ValueError("Prediction output has invalid invisible point momenta")
            _require_finite_dataset(point, "INVISIBLE/point_momenta", deep=deep)
            point_four = source.get("INVISIBLE/point_four_vectors")
            if point_four is None or point_four.shape != (count, top_k, point.shape[2], 4):
                raise ValueError("Prediction output has invalid invisible point four-vectors")
            _require_finite_dataset(point_four, "INVISIBLE/point_four_vectors", deep=deep)
            estimator = source["INVISIBLE"].attrs.get("point_estimator")
            if isinstance(estimator, bytes):
                estimator = estimator.decode("utf-8")
            if str(estimator) not in {
                "cfm_single_draw", "cfm_cartesian_mean", "cfm_joint_medoid", "direct_point_head"
            }:
                raise ValueError("Prediction output has invalid point_estimator metadata")
        if "PARENTS" in source:
            parent = source["PARENTS"].get("direct_four_vectors")
            if parent is None or parent.shape[0] != count or parent.ndim != 3 or parent.shape[2] != 4:
                raise ValueError("Prediction output has invalid direct parent four-vectors")
            _require_finite_dataset(parent, "PARENTS/direct_four_vectors", deep=deep)
            if deep and (parent[:, :, 0] < 0).any():
                raise ValueError("PARENTS/direct_four_vectors contains negative energy")
        if "PARTONIC" in source:
            partonic = source["PARTONIC"].get("direct_four_vectors")
            if partonic is None or partonic.shape[0] != count or partonic.ndim != 3 or partonic.shape[2] != 4:
                raise ValueError("Prediction output has invalid direct partonic four-vectors")
            _require_finite_dataset(partonic, "PARTONIC/direct_four_vectors", deep=deep)
            if deep and (partonic[:, :, 0] < 0).any():
                raise ValueError("PARTONIC/direct_four_vectors contains negative energy")
            raw_roles = source["PARTONIC"].attrs.get("roles")
            if isinstance(raw_roles, bytes):
                raw_roles = raw_roles.decode("utf-8")
            try:
                roles = json.loads(str(raw_roles))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError("PARTONIC has invalid roles metadata") from error
            if not isinstance(roles, list) or len(roles) != partonic.shape[1] or not all(
                isinstance(role, str) and role for role in roles
            ):
                raise ValueError("PARTONIC roles metadata does not match direct outputs")
            support = source.get("TRUTH/PARTONIC_SUPPORT")
            truth = source.get("TRUTH/PARTONIC_FOUR_VECTORS")
            if (support is None) != (truth is None):
                raise ValueError("Partonic truth output requires both vectors and support")
            if support is not None:
                support_values = support[:]
                if support_values.shape != (count, len(roles)) or support_values.dtype != np.bool_:
                    raise ValueError("TRUTH/PARTONIC_SUPPORT has invalid shape or dtype")
                if truth.shape != (count, len(roles), 4):
                    raise ValueError("TRUTH/PARTONIC_FOUR_VECTORS has invalid shape")
                _require_finite_dataset(truth, "TRUTH/PARTONIC_FOUR_VECTORS", deep=deep)
                if deep and (truth[:, :, 0] < 0).any():
                    raise ValueError("TRUTH/PARTONIC_FOUR_VECTORS contains negative energy")
        if "OBSERVED/MET" in source:
            met = source["OBSERVED/MET"][:]
            if met.shape != (count, 4):
                raise ValueError("Prediction output has invalid OBSERVED/MET")
            _require_finite(met, "Prediction OBSERVED/MET")
        if "COMPOSITES" in source:
            truth_composites_available = bool(
                source.attrs.get("truth_composites_available", True)
            )
            if truth_composites_available and "TRUTH/COMPOSITES" not in source:
                raise ValueError("Prediction output is missing truth composites")
            truth_composites = source.get("TRUTH/COMPOSITES")
            for node in source["COMPOSITES"]:
                values = source[f"COMPOSITES/{node}"]
                if values.shape != (count, top_k, posterior_samples, 4):
                    raise ValueError(f"Prediction output has invalid composite {node}")
                if truth_composites is not None and node in truth_composites:
                    truth_values = truth_composites[node]
                    if truth_values.shape != (count, 4):
                        raise ValueError(f"Prediction output has invalid truth composite {node}")
                elif truth_composites_available:
                    raise ValueError(f"Prediction output is missing truth composite {node}")
                else:
                    truth_values = None
                if deep:
                    for chunk in _iter_hdf5_chunks(values):
                        _require_finite_or_missing(chunk, f"Prediction COMPOSITES/{node}")
                    if truth_values is not None:
                        for chunk in _iter_hdf5_chunks(truth_values):
                            _require_finite_or_missing(
                                chunk, f"Prediction TRUTH/COMPOSITES/{node}"
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
        validate_hypothesis_group(source, count=count, top_k=top_k, deep=deep)


def validate_discrete_prediction(path: str | Path, *, deep: bool = True) -> None:
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


def validate_joint_prediction(path: str | Path, *, deep: bool = True) -> None:
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
