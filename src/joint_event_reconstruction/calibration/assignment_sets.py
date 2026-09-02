"""Deterministic adaptive prediction sets for ranked complete assignments."""

from __future__ import annotations

import math

import numpy as np


def _validate(probabilities: np.ndarray, truth: np.ndarray) -> None:
    if probabilities.ndim != 2 or truth.shape != probabilities.shape:
        raise ValueError("APS requires matching [N,K] probabilities and truth indicators")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("APS probabilities must be finite and nonnegative")
    if (np.diff(probabilities, axis=1) > 1.0e-7).any():
        raise ValueError("APS candidates must be ordered by non-increasing probability")
    if truth.dtype != np.bool_:
        raise ValueError("APS truth indicators must be boolean")


def calibrate_aps(
    probabilities: np.ndarray,
    truth: np.ndarray,
    *,
    coverages: tuple[float, ...] = (0.68, 0.90, 0.95),
) -> dict:
    """Calibrate inclusive cumulative-mass APS scores on supported truth events.

    For truth rank ``r``, the nonconformity score is ``sum(p[:r+1])``.  The
    finite-sample quantile is order statistic
    ``ceil((n+1)*(1-alpha))``, capped at ``n``.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth)
    _validate(probabilities, truth)
    supported = truth.any(axis=1)
    cumulative = np.cumsum(probabilities, axis=1)
    ranks = np.argmax(truth[supported], axis=1)
    scores = cumulative[supported, ranks]
    if not len(scores):
        raise ValueError("APS calibration has no truth assignments inside candidate support")
    ordered = np.sort(scores)
    quantiles = {}
    for coverage in coverages:
        if not 0 < coverage < 1:
            raise ValueError("APS coverage targets must lie in (0,1)")
        order = min(int(math.ceil((len(scores) + 1) * coverage)), len(scores))
        quantiles[f"{coverage:.2f}"] = float(ordered[order - 1])
    return {
        "method": "deterministic APS; inclusive cumulative probability through truth rank",
        "finite_sample_quantile": "ceil((n+1)*(1-alpha)) order statistic, capped at n",
        "events": int(len(probabilities)),
        "supported_events": int(supported.sum()),
        "truth_outside_candidate_support": int((~supported).sum()),
        "candidate_kmax": int(probabilities.shape[1]),
        "quantiles": quantiles,
    }


def analyze_aps(
    probabilities: np.ndarray,
    truth: np.ndarray,
    calibrator: dict,
    *,
    njet: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(truth)
    _validate(probabilities, truth)
    cumulative = np.cumsum(probabilities, axis=1)
    supported = truth.any(axis=1)
    result = {
        "events": int(len(probabilities)),
        "truth_outside_candidate_support": int((~supported).sum()),
        "truth_outside_candidate_support_fraction": float((~supported).mean()),
        "targets": {},
    }
    for coverage, quantile in calibrator["quantiles"].items():
        sizes = np.asarray(
            [
                np.searchsorted(row, float(quantile), side="left") + 1
                for row in cumulative
            ]
        )
        sizes = np.minimum(sizes, probabilities.shape[1])
        ranks = np.argmax(truth, axis=1) + 1
        covered = supported & (ranks <= sizes)
        row = {
            "empirical_coverage_all_events": float(covered.mean()),
            "empirical_coverage_within_candidate_support": float(
                covered[supported].mean()
            ) if supported.any() else None,
            "mean_set_size": float(sizes.mean()),
            "median_set_size": float(np.median(sizes)),
            "p90_set_size": float(np.quantile(sizes, 0.90)),
            "p95_set_size": float(np.quantile(sizes, 0.95)),
        }
        if njet is not None:
            njet = np.asarray(njet)
            row["set_size_vs_njet"] = {
                str(int(value)): {
                    "events": int((njet == value).sum()),
                    "mean_set_size": float(sizes[njet == value].mean()),
                }
                for value in np.unique(njet)
            }
        if entropy is not None:
            entropy = np.asarray(entropy, dtype=np.float64)
            edges = np.unique(np.quantile(entropy, np.linspace(0, 1, 6)))
            row["set_size_vs_entropy"] = {
                "bin_edges": edges.tolist(),
                "bins": [
                    {
                        "events": int(mask.sum()),
                        "mean_set_size": float(sizes[mask].mean()) if mask.any() else None,
                    }
                    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:]))
                    for mask in [
                        (entropy >= low)
                        & (entropy <= high if index == len(edges) - 2 else entropy < high)
                    ]
                ],
            }
        result["targets"][coverage] = row
    return result
