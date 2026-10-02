"""Targeted ordered-role, assignment-support gradient and Lorentz-boost checks."""

from types import SimpleNamespace

import numpy as np
import torch

from joint_event_reconstruction.knot_physics import (
    boost,
    joint_energy_score,
    mass,
    spin_observables,
)
from joint_event_reconstruction.models.reconstruction.structured import (
    StructuredReconstruction,
)


def main():
    # The masked row must contribute exactly zero gradient, not a supervised NULL.
    owner = SimpleNamespace(spec=SimpleNamespace(dag_order=("top",)))
    logits = torch.tensor([[0.0, 1.0], [5.0, -5.0]], requires_grad=True)
    structured = {
        "pools": {
            "top": {
                "reconstructable": torch.tensor([True, False]),
                "truth_index": torch.tensor([0, -1]),
            }
        },
        "objective_support": torch.tensor([True, False]),
    }
    result = StructuredReconstruction.assignment_loss(
        owner, {"top": logits}, structured
    )
    result["assignment_loss"].backward()
    assert logits.grad[1].abs().sum() == 0 and logits.grad[0].abs().sum() > 0
    vector = np.array([[10.0, 2.0, 3.0, 4.0]])
    beta = np.array([[0.2, -0.1, 0.3]])
    moved = boost(vector, beta)
    np.testing.assert_allclose(mass(vector), mass(moved), atol=1e-12)
    np.testing.assert_allclose(boost(moved, -beta), vector, atol=1e-12)
    parents = np.array([[[200.0, 0.0, 0.0, 100.0], [200.0, 0.0, 0.0, -100.0]]])
    leptons = np.array([[[50.0, 30.0, 0.0, 40.0], [50.0, -30.0, 0.0, -40.0]]])
    obs = spin_observables(parents, leptons)
    assert all(np.isfinite(value).all() for value in obs.values())
    assert abs(obs["delta_phi_ll"][0] - np.pi) < 1e-12
    np.testing.assert_allclose(obs["cos_theta_plus"], 0.5, atol=1e-12)
    np.testing.assert_allclose(obs["cos_theta_minus"], 0.5, atol=1e-12)
    np.testing.assert_allclose(obs["helicity_product"], 0.25, atol=1e-12)
    np.testing.assert_allclose(obs["cos_opening"], -1, atol=1e-12)
    ranks = np.arange(17)
    overlap = np.maximum(
        np.minimum((ranks + 1) / 17, 0.84)
        - np.maximum(ranks / 17, 0.16), 0,
    ) * 17
    np.testing.assert_allclose(overlap.mean(), 0.68, atol=1e-14)
    # Validate only the new evaluator score, without retraining the closed toy.
    rng = np.random.default_rng(194)
    draws = rng.normal(size=(512, 32))
    exact = rng.normal(size=512)
    samples = np.zeros((512, 32, 2, 4))
    truth = np.zeros((512, 2, 4))
    samples[:, :, 0, 3] = draws
    truth[:, 0, 3] = exact
    score = joint_energy_score(samples, samples, truth, truth)
    assert score["shuffle_minus_coherent_GeV"] > 3 * score["paired_difference_SE_GeV"]
    print(
        "assignment mask gradient, boost invariance/inverse, spin finite/control: PASS"
    )


if __name__ == "__main__":
    main()
