"""Offline selective-risk curves from fixed, non-trained reliability scores."""

from __future__ import annotations

import numpy as np


def selective_risk_curve(
    score: np.ndarray,
    *,
    fractions: list[float] | tuple[float, ...],
    assignment_correct: np.ndarray | None = None,
    physics_residual: np.ndarray | None = None,
    covered: np.ndarray | None = None,
) -> list[dict]:
    score = np.asarray(score, dtype=np.float64)
    if score.ndim != 1 or not len(score) or not np.isfinite(score).all():
        raise ValueError("Reliability score must be a non-empty finite vector")
    order = np.argsort(-score, kind="stable")
    result = []
    for fraction in fractions:
        if not 0 < fraction <= 1:
            raise ValueError("Reliability acceptance fractions must lie in (0,1]")
        count = max(1, int(np.ceil(fraction * len(score))))
        selected = order[:count]
        row = {"retained_fraction": float(fraction), "events": count}
        if assignment_correct is not None:
            row["complete_assignment_accuracy"] = float(
                np.asarray(assignment_correct)[selected].mean()
            )
        if physics_residual is not None:
            local = np.asarray(physics_residual, dtype=np.float64)[selected]
            local = local[np.isfinite(local)]
            row["physics_rmse"] = (
                float(np.sqrt(np.mean(local**2))) if len(local) else None
            )
        if covered is not None:
            row["posterior_coverage"] = float(np.asarray(covered)[selected].mean())
        result.append(row)
    return result


def reliability_scores(
    *, entropy=None, margin=None, retained_mass=None, posterior_width=None,
    recurrent_stability=None,
) -> dict[str, np.ndarray]:
    candidates = {
        "negative_assignment_entropy": None if entropy is None else -np.asarray(entropy),
        "assignment_margin": margin,
        "retained_assignment_mass": retained_mass,
        "negative_posterior_width": (
            None if posterior_width is None else -np.asarray(posterior_width)
        ),
        "recurrent_stability": recurrent_stability,
    }
    return {name: np.asarray(value) for name, value in candidates.items() if value is not None}
