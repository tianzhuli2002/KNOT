import numpy as np
import pytest
from matplotlib import pyplot as plt

from joint_event_reconstruction.plotting_style import controlled_ratio_ylim
from scripts.evaluation.plot_knot_posterior_usage import (
    _coverage_rows,
    _crps,
    _draw_choice_hash,
    _flatten_population,
    _parent_observables,
    _representation_values,
    event_keyed_indices,
    ordinary_event_residuals,
)


def test_one_draw_is_event_keyed_and_row_order_independent():
    event_ids = np.asarray([107, 204, 991, 1503], dtype=np.int64)
    indices = event_keyed_indices(event_ids, 32, seed=42011)[0]
    reversed_indices = event_keyed_indices(event_ids[::-1], 32, seed=42011)[0]
    assert np.array_equal(indices[::-1], reversed_indices)
    assert _draw_choice_hash(event_ids, indices) == _draw_choice_hash(event_ids[::-1], reversed_indices)


def test_full_posterior_is_rejected_by_point_residuals():
    with pytest.raises(ValueError, match="posterior_distribution is forbidden"):
        ordinary_event_residuals(
            np.zeros((3, 32)),
            np.zeros(3),
            representation="posterior_distribution",
            label="mttbar",
        )


def test_flattened_posterior_keeps_one_total_weight_per_event():
    truth = np.zeros((4, 2))
    samples = np.zeros((4, 7, 2))
    values, weights = _flatten_population(samples, truth, representation="posterior_distribution")
    assert values.shape == weights.shape == (56,)
    assert np.allclose(weights.reshape(4, 7, 2).sum(axis=(1, 2)), 1.0, atol=1e-14)


def test_nonlinear_observable_mean_is_mean_of_per_sample_observables():
    # The top's transverse direction rotates between two coherent hypotheses.
    e1 = 3.0
    e2 = np.sqrt(41.0)
    parents = np.asarray(
        [[
            [[e1, 3.0, 0.0, 0.0], [e1, -3.0, 0.0, 0.0]],
            [[e2, 0.0, 5.0, 0.0], [e2, 0.0, -5.0, 0.0]],
        ]],
        dtype=np.float64,
    )
    per_draw = _parent_observables(parents, "synthetic J")
    observable_mean = float(per_draw["top_pt"].mean(axis=1)[0, 0])
    mean_four_vector = parents.mean(axis=1)
    vector_mean_pt = float(np.linalg.norm(mean_four_vector[0, 0, 1:3]))
    assert observable_mean == pytest.approx(4.0)
    assert vector_mean_pt == pytest.approx(np.sqrt(1.5**2 + 2.5**2))
    assert observable_mean != pytest.approx(vector_mean_pt)


def test_coverage_preserves_sample_and_particle_axes():
    samples = np.asarray([[[0.0, 100.0], [2.0, 102.0]]])
    truth = np.asarray([[1.0, 101.0]])
    rows = _coverage_rows(samples, truth, observable="two_role")
    row = next(item for item in rows if item["nominal_credibility"] == 0.68)
    assert row["objects"] == 2
    assert row["empirical_coverage"] == 1.0


def test_empirical_crps_has_expected_two_sample_value():
    assert _crps(np.asarray([[0.0, 2.0]]), np.asarray([1.0])) == pytest.approx(0.5)


def test_physical_density_ratio_axis_never_extends_below_zero():
    figure, axis = plt.subplots()
    controlled_ratio_ylim(axis, [np.asarray([0.0, 1.0, 3.0])], np.arange(3), center=1.0)
    assert axis.get_ylim()[0] == 0.0
    plt.close(figure)


def test_spin_posterior_mean_averages_per_draw_observables():
    samples = np.asarray([[0.4, -0.2, 0.1], [0.7, 0.2, -0.1]])
    values = _representation_values(samples, np.asarray([2, 1]))
    assert values["one_draw"].tolist() == [0.1, 0.2]
    assert values["posterior_mean"].tolist() == pytest.approx([0.1, 0.26666666666666666])
