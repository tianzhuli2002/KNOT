"""Compact posterior physics in canonical E,px,py,pz GeV coordinates.

Spin convention: lab -> reconstructed ttbar rest frame by a rotation-free boost;
then boost each charged lepton from that frame to its own parent rest frame.
Helicity axes are the corresponding parent's direction in the ttbar frame.
The opening angle compares both lepton directions in the common ttbar rest
frame, before either individual parent boost.
Slot zero is top/l+; slot one is antitop/l-. No antitop-axis sign reversal.
"""

import numpy as np

SPIN_DEFINITION = {
    "truth_benchmark": "hybrid: authoritative truth parents with measured charge-ordered leptons; charged-lepton truth is absent",
    "four_vectors": "E,px,py,pz GeV",
    "charge_order": ["positive lepton / top", "negative lepton / antitop"],
    "boost_sequence": "lab -> ttbar rest -> corresponding parent rest; canonical rotation-free boosts",
    "helicity_axis": "own parent direction in ttbar rest, including own antitop direction without extra sign",
    "opening_angle": "dot of both lepton unit directions in the common ttbar rest frame, before individual parent boosts",
    "delta_phi_ll": "absolute wrapped lab azimuth difference, range [0,pi]",
}


def boost(vector, beta):
    """Passive Lorentz boost to a frame moving with velocity beta."""
    vector, beta = (
        np.asarray(vector, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
    )
    beta2 = np.sum(beta**2, axis=-1)
    if (
        not np.isfinite(vector).all()
        or not np.isfinite(beta).all()
        or (beta2 >= 1).any()
    ):
        raise ValueError(
            "Lorentz boost requires finite four-vectors and subluminal beta"
        )
    gamma = 1 / np.sqrt(1 - beta2)
    dot = np.sum(vector[..., 1:] * beta, axis=-1)
    coefficient = np.divide(gamma - 1, beta2, out=np.zeros_like(gamma), where=beta2 > 0)
    momentum = (
        vector[..., 1:] + (coefficient * dot - gamma * vector[..., 0])[..., None] * beta
    )
    return np.concatenate(
        ((gamma * (vector[..., 0] - dot))[..., None], momentum), axis=-1
    )


def unit(vector):
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    if (norm <= 0).any():
        raise ValueError("Undefined angle for a zero spatial vector")
    return vector / norm


def mass(vector):
    return np.sqrt(
        np.maximum(vector[..., 0] ** 2 - np.sum(vector[..., 1:] ** 2, axis=-1), 0)
    )


def pt(vector):
    return np.linalg.norm(vector[..., 1:3], axis=-1)


def spin_observables(parents, leptons):
    pair = parents.sum(axis=-2)
    beta = pair[..., 1:] / pair[..., :1]
    parent_cm = boost(parents, beta[..., None, :])
    leptons = np.broadcast_to(leptons, parents.shape)
    lepton_cm = boost(leptons, beta[..., None, :])
    rest = boost(lepton_cm, parent_cm[..., 1:] / parent_cm[..., :1])
    direction = unit(rest[..., 1:])
    axis = unit(parent_cm[..., 1:])
    cosine = np.sum(direction * axis, axis=-1).clip(-1, 1)
    phi = np.arctan2(leptons[..., 2], leptons[..., 1])
    delta = np.abs((phi[..., 0] - phi[..., 1] + np.pi) % (2 * np.pi) - np.pi)
    return {
        "delta_phi_ll": delta,
        "cos_theta_plus": cosine[..., 0],
        "cos_theta_minus": cosine[..., 1],
        "helicity_product": cosine.prod(axis=-1),
        "cos_opening": np.sum(
            unit(lepton_cm[..., 0, 1:]) * unit(lepton_cm[..., 1, 1:]), axis=-1
        ).clip(-1, 1),
    }


def resolution(prediction, truth):
    difference = prediction - truth
    return {
        "bias_GeV": float(difference.mean()),
        "RMSE_GeV": float(np.sqrt(np.mean(difference**2))),
        "residual_std_GeV": float(difference.std()),
    }


def reconstruction_metrics(neutrinos, parents, truth_n, truth_p):
    result = {}
    for index, side in enumerate(["top", "antitop"]):
        n, p = neutrinos[:, :, index], parents[:, :, index]
        tn, tp = truth_n[:, index], truth_p[:, index]
        lower, upper = np.quantile(n[..., 3], [0.16, 0.84], axis=1)
        # A rank occupies one PIT bin. Integrate its overlap with the central
        # interval; unlike empirical quantiles this has 68% expectation even
        # with only 16 posterior draws under exchangeability.
        ranks = (n[..., 3] <= tn[:, None, 3]).sum(axis=1)
        bins = n.shape[1] + 1
        rank_coverage = np.maximum(
            np.minimum((ranks + 1) / bins, 0.84)
            - np.maximum(ranks / bins, 0.16),
            0,
        ) * bins
        phi = np.arctan2(n[..., 2], n[..., 1])
        true_phi = np.arctan2(tn[..., 2], tn[..., 1])
        dphi = np.angle(np.exp(1j * (phi - true_phi[:, None])))
        eta = np.arcsinh(n[..., 3] / np.maximum(pt(n), 1e-12))
        true_eta = np.arcsinh(tn[..., 3] / np.maximum(pt(tn), 1e-12))
        result[side] = {
            "neutrino_pT": resolution(pt(n).mean(axis=1), pt(tn)),
            "neutrino_pz": resolution(n[..., 3].mean(axis=1), tn[..., 3]),
            "neutrino_mean_deltaR": float(
                np.sqrt(dphi**2 + (eta - true_eta[:, None]) ** 2).mean()
            ),
            "neutrino_pz_68_coverage": float(
                ((tn[..., 3] >= lower) & (tn[..., 3] <= upper)).mean()
            ),
            "neutrino_pz_rank_68_coverage": float(rank_coverage.mean()),
            "neutrino_pz_rank_68_coverage_SE": float(
                rank_coverage.std(ddof=1) / np.sqrt(len(rank_coverage))
            ),
            "parent_pT": resolution(pt(p).mean(axis=1), pt(tp)),
            "parent_mass": resolution(mass(p).mean(axis=1), mass(tp)),
        }
    pair, truth_pair = parents.sum(axis=2), truth_p.sum(axis=1)
    result["mttbar"] = resolution(mass(pair).mean(axis=1), mass(truth_pair))
    result["pT_ttbar"] = resolution(pt(pair).mean(axis=1), pt(truth_pair))
    return result


def joint_metric(neutrinos, parents, seed=42012):
    n, p = neutrinos[:, :, 0, 3], parents[:, :, 0, 3]
    permutation = np.argsort(np.random.default_rng(seed).random(p.shape), axis=1)
    shuffled = np.take_along_axis(p, permutation, axis=1)
    nc = n - n.mean(axis=1, keepdims=True)

    def correlation(right):
        pc = right - right.mean(axis=1, keepdims=True)
        return np.mean(nc * pc, axis=1) / np.sqrt(
            np.maximum(np.mean(nc**2, axis=1) * np.mean(pc**2, axis=1), 1e-20)
        )

    coherent, broken = correlation(p), correlation(shuffled)
    return {
        "coherent_mean_conditional_correlation": float(coherent.mean()),
        "shuffled_mean_conditional_correlation": float(broken.mean()),
        "coherent_mean_absolute_correlation": float(np.abs(coherent).mean()),
        "shuffled_mean_absolute_correlation": float(np.abs(broken).mean()),
        "paired_difference_standard_error": float(
            (coherent - broken).std() / np.sqrt(len(coherent))
        ),
        "marginal_max_change": float(
            max(abs(p.mean() - shuffled.mean()), abs(p.var() - shuffled.var()))
        ),
    }


def joint_energy_score(neutrinos, parents, truth_n, truth_p, seed=42012):
    """Unbiased sample energy score for (nu_top pz, top pz), both in GeV.

    The pair-distance term excludes the diagonal; lower is better. Marginals
    are exactly preserved in the paired shuffle comparison.
    """
    values = np.stack((neutrinos[:, :, 0, 3], parents[:, :, 0, 3]), axis=-1)
    truth = np.stack((truth_n[:, 0, 3], truth_p[:, 0, 3]), axis=-1)
    permutation = np.argsort(
        np.random.default_rng(seed).random(values.shape[:2]), axis=1
    )
    shuffled = values.copy()
    shuffled[..., 1] = np.take_along_axis(values[..., 1], permutation, axis=1)

    def score(samples):
        count = samples.shape[1]
        distance_truth = np.linalg.norm(samples - truth[:, None], axis=-1).mean(axis=1)
        pair = np.linalg.norm(samples[:, :, None] - samples[:, None, :], axis=-1)
        return distance_truth - 0.5 * pair.sum(axis=(1, 2)) / (count * (count - 1))

    coherent, broken = score(values), score(shuffled)
    difference = broken - coherent
    return {
        "coherent_GeV": float(coherent.mean()),
        "shuffled_GeV": float(broken.mean()),
        "shuffle_minus_coherent_GeV": float(difference.mean()),
        "paired_difference_SE_GeV": float(difference.std() / np.sqrt(len(difference))),
    }
