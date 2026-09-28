"""Numerical regressions for the realnu validation metrics."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.evaluation.plot_realnu_campaign import (
    STANDARD,
    _assignment_products,
    _paired_metric_bootstrap,
    _population_reweighting,
    _scalar_energy_score,
)


class RealnuEvaluationHelperTests(unittest.TestCase):
    def test_scalar_energy_score_uses_empirical_pair_distance(self) -> None:
        observed = _scalar_energy_score(np.asarray([[0.0, 1.0]]), np.asarray([0.0]))
        self.assertAlmostEqual(observed, 0.25)

    def test_paired_bootstrap_zero_delta_for_identical_residuals(self) -> None:
        residual = np.asarray([-3.0, -1.0, 0.5, 2.0])
        result = _paired_metric_bootstrap(residual, residual, seed=730241, draws=2000)
        self.assertEqual(result["events"], len(residual))
        self.assertEqual(result["bootstrap_draws"], 2000)
        for metric in result["metrics"].values():
            self.assertEqual(metric["delta_second_minus_first"], 0.0)
            self.assertEqual(metric["ci95"], [0.0, 0.0])

    def test_assignment_efficiency_uses_each_models_candidate_support(self) -> None:
        truth = {
            "assignment_support": np.asarray([True, True]),
            "source_njet": np.asarray([2, 9]),
            "model_visible_njet": np.asarray([2, 8]),
        }
        predictions = {
            mode: {
                "roots": np.asarray([[[0, 0]], [[1, 0]]]),
                "candidate_truth_index": {
                    "top": np.asarray([0, 1]),
                    "antitop": np.asarray([0, 0]),
                },
                "candidate_support": {
                    "top": np.asarray([True, True]),
                    "antitop": np.asarray([True, True]),
                },
            }
            for mode in STANDARD
        }
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "tables").mkdir()
            _correct, summary = _assignment_products(
                Path(directory), predictions, truth
            )
        self.assertEqual(summary["support_denominator"], 2)
        for mode in STANDARD:
            self.assertEqual(
                summary["complete_event"][mode]["supported_events"], 2
            )
            self.assertEqual(summary["complete_event"][mode]["efficiency"], 1.0)

    def test_spectrum_weights_include_subthreshold_tail_and_preserve_model_metrics(self) -> None:
        pair_energy = np.asarray([320.0, 350.0, 500.0, 800.0])
        parents = np.zeros((len(pair_energy), 2, 4), dtype=np.float64)
        parents[:, :, 0] = pair_energy[:, None] / 2
        truth = {
            "parents": parents,
            "neutrino": np.zeros((len(pair_energy), 2, 3)),
            "neutrino_support": np.ones((len(pair_energy), 2), dtype=bool),
        }
        standard = {
            mode: {
                "assignment_supported": np.ones(len(pair_energy), dtype=bool),
                "nu_joint_medoid": np.zeros((len(pair_energy), 2, 3)),
            }
            for mode in STANDARD
        }
        parent = {
            mode: {"parent_joint_medoid": parents.copy()}
            for mode in ("PCFM_Q", "J0", "J_A", "J_AN")
        }
        assignment_correct = {
            mode: np.ones(len(pair_energy), dtype=bool) for mode in STANDARD
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "tables").mkdir()
            (output / "evaluation_contract.json").write_text(
                '{"spectrum_reweighting":{"expected_events_below_345_gev":1}}'
            )
            result = _population_reweighting(
                output, standard, parent, truth, assignment_correct
            )
        self.assertEqual(result["truth_mttbar_below_345_gev_events"], 1)
        self.assertEqual(result["truth_mttbar_region_counts"]["threshold_lt_400"], 2)
        self.assertIn(
            "complete_assignment_efficiency",
            result["metrics"]["nominal_validation"]["models"]["J0"],
        )
        self.assertEqual(
            result["metrics"]["nominal_validation"]["models"]["J0"][
                "mttbar_rmse_gev"
            ],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
