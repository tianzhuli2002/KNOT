"""Regression checks for source and model-visible realnu jet multiplicities."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import h5py
import numpy as np

from scripts.production.run_realnu_production import (
    validate_maxjets8_adapter,
    validate_prediction_jet_counts,
)

PROJECT_ROOT = Path(
    os.environ.get(
        "JER_REALNU_PROJECT_ROOT", Path(__file__).resolve().parents[2]
    )
).expanduser().resolve()
EXPERIMENT_ROOT = PROJECT_ROOT / "results/experiments/partonic_ttbar2L_2M_realnu_v1"
PRODUCTION_ROOT = (
    EXPERIMENT_ROOT / "production/realnu_top_antitop_20260927_release2"
)
MEASURED_PATH = (
    EXPERIMENT_ROOT
    / "adapted/train_validation/partonic_ttbar2l_train_validation_maxjets8_v1_measured.h5"
)


class RealnuJetCountSemanticsTest(unittest.TestCase):
    def test_truncated_and_untruncated_events_pass(self) -> None:
        validate_prediction_jet_counts(
            np.array([8]), np.array([8]), np.array([10]), np.array([1001])
        )
        validate_prediction_jet_counts(
            np.array([4]), np.array([4]), np.array([4]), np.array([1002])
        )
        validate_maxjets8_adapter(
            np.array([10, 4]), np.array([8, 4]), np.array([1001, 1002]),
            model_max_jets=8,
        )

    def test_genuine_prediction_mismatch_fails_with_event_diagnostic(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            r"mismatching_events=1.*EVENT_ID.*1003.*prediction_njet.*5.*"
            r"model_visible_njet.*6.*source_njet.*6",
        ):
            validate_prediction_jet_counts(
                np.array([5]), np.array([6]), np.array([6]), np.array([1003])
            )

    def test_frozen_validation_population_closes_and_reports_truncation(self) -> None:
        with h5py.File(MEASURED_PATH, "r") as measured:
            split = np.asarray(measured["SPLIT"][:], dtype=np.uint8)
            validation_mask = split == 1
            measured_ids = np.asarray(measured["EVENT_ID"][validation_mask])
            global_inputs = np.asarray(measured["INPUTS/GLOBAL"][validation_mask])
            jets = np.asarray(measured["INPUTS/JET"][validation_mask])

        prediction_path = PRODUCTION_ROOT / "Q0/validation/standard/validation.h5"
        with h5py.File(prediction_path, "r") as prediction:
            event_ids = np.asarray(prediction["EVENT_ID"][:], dtype=np.int64)
            prediction_njet = np.asarray(prediction["OBSERVED/njet"][:], dtype=np.int64)

        order = np.argsort(measured_ids, kind="stable")
        positions = np.searchsorted(measured_ids[order], event_ids)
        self.assertTrue(np.all(positions < len(order)))
        self.assertTrue(np.array_equal(measured_ids[order[positions]], event_ids))
        global_inputs = global_inputs[order[positions]]
        jets = jets[order[positions]]
        source_njet = np.rint(global_inputs["njet"]).astype(np.int64)
        model_visible_njet = (
            np.isfinite(jets["pt"]) & (jets["pt"] > 0)
        ).sum(axis=1).astype(np.int64)

        validate_prediction_jet_counts(
            prediction_njet, model_visible_njet, source_njet, event_ids
        )
        validate_maxjets8_adapter(
            source_njet, model_visible_njet, event_ids, model_max_jets=jets.shape[1]
        )
        self.assertEqual(len(event_ids), 100_342)
        self.assertEqual(int((source_njet != model_visible_njet).sum()), 164)
        self.assertEqual(int((source_njet > 8).sum()), 164)
        self.assertEqual(int((source_njet == 9).sum()), 121)
        self.assertEqual(int((source_njet == 10).sum()), 28)
        self.assertEqual(int((source_njet == 11).sum()), 11)
        self.assertEqual(int((source_njet == 12).sum()), 4)


if __name__ == "__main__":
    unittest.main()
