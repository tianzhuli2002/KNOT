"""Shared assignment-support semantics for prediction, evaluation and plotting."""

from __future__ import annotations


NON_NULL_SUPPORT = "complete_non_null"
NULL_INCLUDING_SUPPORT = "complete_null_including"

_SUPPORT = {
    NON_NULL_SUPPORT: {
        "name": NON_NULL_SUPPORT,
        "includes_null": False,
        "null_semantics": "excluded_from_complete_prediction_support",
        "mask_semantics": "not_used",
        "description": (
            "Complete globally compatible assignments in which every root selects a "
            "physical candidate. Per-node NULL probabilities exist in the local model "
            "but NULL is excluded from the decoded complete support."
        ),
    },
    NULL_INCLUDING_SUPPORT: {
        "name": NULL_INCLUDING_SUPPORT,
        "includes_null": True,
        "null_semantics": "physical_clean_assignment_state",
        "mask_semantics": "transient_generation_state_not_physical_support",
        "description": (
            "Complete globally feasible clean states including physical NULL choices. "
            "MASK is a transient Flow-Matching state and is never part of the final "
            "physical assignment support."
        ),
    },
}


def support_metadata(name: str) -> dict:
    """Return a copy of the canonical metadata for one assignment support."""
    if name not in _SUPPORT:
        raise ValueError(f"Unknown assignment support {name!r}")
    return dict(_SUPPORT[name])


def hdf5_support_attributes(name: str) -> dict:
    """Flatten support semantics into stable HDF5 attributes."""
    values = support_metadata(name)
    return {
        "assignment_support": values["name"],
        "assignment_support_includes_null": values["includes_null"],
        "assignment_null_semantics": values["null_semantics"],
        "assignment_mask_semantics": values["mask_semantics"],
        "assignment_support_description": values["description"],
    }


def read_hdf5_support(source) -> dict:
    """Read and validate assignment-support metadata from an open HDF5 file."""
    if "assignment_support" not in source.attrs:
        raise ValueError(
            "Prediction file predates explicit assignment-support metadata; regenerate "
            "predictions before comparing assignment performance."
        )
    name = str(source.attrs["assignment_support"])
    expected = support_metadata(name)
    observed = {
        "name": name,
        "includes_null": bool(source.attrs["assignment_support_includes_null"]),
        "null_semantics": str(source.attrs["assignment_null_semantics"]),
        "mask_semantics": str(source.attrs["assignment_mask_semantics"]),
        "description": str(source.attrs["assignment_support_description"]),
    }
    if observed != expected:
        raise ValueError(
            "Prediction assignment-support metadata is internally inconsistent: "
            f"expected={expected}, observed={observed}"
        )
    return observed
