"""Safe physical observables derived from Cartesian four-vectors and MET."""

from __future__ import annotations

import numpy as np


def four_vector_observables(vectors: np.ndarray) -> dict[str, np.ndarray]:
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape[-1] != 4:
        raise ValueError("Four-vectors must end in (E,px,py,pz)")
    energy, px, py, pz = np.moveaxis(vectors, -1, 0)
    pt = np.hypot(px, py)
    momentum2 = px**2 + py**2 + pz**2
    mass2 = energy**2 - momentum2
    tolerance = 1.0e-10 * np.maximum(energy**2 + momentum2, 1.0)
    mass = np.where(mass2 >= -tolerance, np.sqrt(np.maximum(mass2, 0.0)), np.nan)
    eta = np.full_like(pt, np.nan)
    np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pz), where=pt > 0), out=eta, where=pt > 0)
    numerator = energy + pz
    denominator = energy - pz
    rapidity = np.full_like(energy, np.nan)
    valid_y = (numerator > 0) & (denominator > 0)
    rapidity[valid_y] = 0.5 * np.log(numerator[valid_y] / denominator[valid_y])
    phi = np.arctan2(py, px)
    finite = np.isfinite(vectors).all(axis=-1)
    return {
        name: np.where(finite, value, np.nan)
        for name, value in {
            "mass": mass,
            "pt": pt,
            "eta": eta,
            "rapidity": rapidity,
            "phi": phi,
        }.items()
    }


def wrapped_delta(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(first - second), np.cos(first - second))


def root_system_observables(
    roots: dict[str, np.ndarray], root_order: list[str] | tuple[str, ...]
) -> dict[str, np.ndarray]:
    if not root_order:
        raise ValueError("Root-system observables require at least one topology root")
    selected = [np.asarray(roots[name]) for name in root_order]
    system = np.stack(selected, axis=0).sum(axis=0)
    result = {f"system_{key}": value for key, value in four_vector_observables(system).items()}
    if len(selected) == 2:
        first = four_vector_observables(selected[0])
        second = four_vector_observables(selected[1])
        delta_phi = wrapped_delta(first["phi"], second["phi"])
        result.update(
            {
                "delta_eta": first["eta"] - second["eta"],
                "delta_y": first["rapidity"] - second["rapidity"],
                "delta_phi": delta_phi,
                "delta_r": np.sqrt((first["eta"] - second["eta"]) ** 2 + delta_phi**2),
            }
        )
    return result


def met_closure(
    samples: np.ndarray, met_xy: np.ndarray, weights: np.ndarray | None = None
) -> dict:
    samples = np.asarray(samples, dtype=np.float64)
    met_xy = np.asarray(met_xy, dtype=np.float64)
    if samples.ndim != 4 or samples.shape[-1] != 3 or met_xy.shape != (len(samples), 2):
        raise ValueError("MET closure requires samples [N,S,R,3] and met_xy [N,2]")
    residual = samples[..., :2].sum(axis=2) - met_xy[:, None]
    magnitude = np.linalg.norm(residual, axis=-1)
    if weights is None:
        weights = np.full(samples.shape[:2], 1.0 / samples.shape[1])
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != samples.shape[:2] or not np.isfinite(weights).all() or (
        weights < 0
    ).any():
        raise ValueError("MET closure posterior weights are invalid")
    totals = weights.sum(axis=1, keepdims=True)
    if not np.isfinite(totals).all() or (totals <= 0).any():
        raise ValueError("MET closure posterior weights need positive finite event sums")
    weights = weights / totals
    mean = np.sum(residual * weights[..., None], axis=1)
    mean_magnitude = np.sum(magnitude * weights, axis=1)
    return {
        "px_bias": float(mean[:, 0].mean()),
        "py_bias": float(mean[:, 1].mean()),
        "px_rmse": float(np.sqrt(np.mean(mean[:, 0] ** 2))),
        "py_rmse": float(np.sqrt(np.mean(mean[:, 1] ** 2))),
        "transverse_rmse": float(np.sqrt(np.mean(np.sum(mean**2, axis=1)))),
        "magnitude_median": float(np.median(mean_magnitude)),
        "magnitude_central_68_width": float(
            np.quantile(mean_magnitude, 0.84) - np.quantile(mean_magnitude, 0.16)
        ),
        "posterior_magnitude_quantiles": {
            str(q): float(np.quantile(magnitude, q)) for q in (0.16, 0.5, 0.84, 0.95)
        },
    }
