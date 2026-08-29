"""Lightweight schemas and validators for self-describing prediction HDF5 files."""

from __future__ import annotations

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
        for name in (
            "VISIBLE/assignment_log_scores",
            "VISIBLE/exact_logZ",
            "VISIBLE/constrained_probabilities",
        ):
            if len(source[name]) != count or not np.isfinite(source[name][:]).all():
                raise ValueError(f"Prediction output has invalid {name}")
        if "INVISIBLE/samples" in source:
            samples = source["INVISIBLE/samples"]
            if len(samples) != count or not np.isfinite(samples[:]).all():
                raise ValueError("Prediction output has invalid invisible samples")
            branches = int(
                source["INVISIBLE"].attrs[
                    "unique_assignment_conditioning_branches"
                ]
            )
            if branches not in {1, int(source.attrs["K"])}:
                raise ValueError("Factorized invisible branch count must be 1 or K")


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
        state_count = int(source.attrs["assignment_state_count"])
        if ((states < 0) | (states >= state_count)).any():
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
        if not np.isfinite(source["HYPOTHESES/invisible"][:]).all():
            raise ValueError("Joint-flow invisible hypotheses contain non-finite values")
        if "COMPOSITES" in source:
            for node in source["COMPOSITES"]:
                if len(source[f"COMPOSITES/{node}"]) != count:
                    raise ValueError(
                        f"Prediction output has inconsistent COMPOSITES/{node} length"
                    )
