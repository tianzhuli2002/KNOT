"""Small prediction-HDF5 adapter for optional physics-aware hypotheses."""

from __future__ import annotations

import json

import h5py
import numpy as np
import torch

from joint_event_reconstruction.topology.physics import HypothesisEvaluation


def _numpy(value: torch.Tensor, dtype=None) -> np.ndarray:
    result = value.detach().cpu().numpy()
    return result.astype(dtype, copy=False) if dtype is not None else result


def write_hypothesis_group(
    destination: h5py.File | h5py.Group,
    *,
    states: torch.Tensor,
    truth: torch.Tensor,
    evaluation: HypothesisEvaluation,
    scaler_identity: str = "none",
    scorer_type: str = "none",
    alpha: float = 1.0,
) -> h5py.Group:
    """Write the optional ``HYPOTHESES`` group without changing old groups."""
    if "HYPOTHESES" in destination:
        raise ValueError("Prediction output already contains HYPOTHESES")
    if states.ndim != 3 or truth.shape != states.shape[:2]:
        raise ValueError("Hypothesis states/truth must have shapes [B,K,N] and [B,K]")
    if evaluation.visible_log_score.shape != states.shape[:2]:
        raise ValueError("Hypothesis states disagree with evaluation score shape")
    if not torch.isfinite(truth.to(torch.float32)).all():
        raise ValueError("Hypothesis truth indicators must be finite")
    group = destination.create_group("HYPOTHESES")
    valid_hypotheses = torch.isfinite(evaluation.visible_log_score)
    group.attrs["schema_version"] = "physics_hypotheses_v1"
    group.create_dataset("state", data=_numpy(states, np.int32))
    group.create_dataset(
        "valid_hypotheses", data=_numpy(valid_hypotheses, np.bool_)
    )
    group.create_dataset(
        "visible_log_score", data=_numpy(evaluation.visible_log_score, np.float32)
    )
    group.create_dataset(
        "visible_probability", data=_numpy(evaluation.visible_probability, np.float32)
    )
    group.create_dataset("truth", data=_numpy(truth.to(torch.bool), np.bool_))
    group.create_dataset("learned_delta", data=_numpy(evaluation.learned_delta, np.float32))
    group.create_dataset(
        "final_log_score", data=_numpy(evaluation.final_log_score, np.float32)
    )
    group.create_dataset(
        "final_probability", data=_numpy(evaluation.final_probability, np.float32)
    )
    group.create_dataset("final_rank", data=_numpy(evaluation.ranking, np.int32))
    group.attrs["scorer_type"] = str(scorer_type)
    group.attrs["alpha"] = float(alpha)
    group.attrs["scaler_identity"] = str(scaler_identity)
    group.attrs["normalised_likelihood"] = False
    if evaluation.feature_result is not None and evaluation.reduction is not None:
        physics = group.create_group("PHYSICS")
        physics.create_dataset(
            "feature_mean", data=_numpy(evaluation.reduction.mean, np.float32)
        )
        physics.create_dataset(
            "feature_std", data=_numpy(evaluation.reduction.std, np.float32)
        )
        physics.create_dataset(
            "feature_rms", data=_numpy(evaluation.reduction.rms, np.float32)
        )
        physics.create_dataset(
            "analytic_score",
            data=_numpy(evaluation.reduction.analytic_log_evidence, np.float32),
        )
        physics.attrs["feature_names"] = json.dumps(
            list(evaluation.feature_result.feature_names)
        )
        physics.attrs["physics_signature"] = evaluation.feature_result.physics_signature
        group.attrs["physics_signature"] = evaluation.feature_result.physics_signature
    else:
        group.attrs["physics_signature"] = "none"
    return group


__all__ = ["write_hypothesis_group"]
