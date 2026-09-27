"""Regression check for train-fitted scaler metadata in component initialisation."""

from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from joint_event_reconstruction.cli.run import _load_initial_components


class ScalerInitialisationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fit = {
            "population": "frozen_train_split",
            "event_count": 1_799_753,
            "event_id_hash": "7d63f66f5edb08296388e05c4ba90c3a293608493b9a81c24331b9c9aeb6f5b2",
        }
        self.scaler = {
            "representation": "partonic_px_py_pz_log1p_mass_v1",
            "role_names": ["top", "antitop"],
            "mean": [[0.0, 0.0, 0.6, 5.15], [0.0, 0.0, 0.4, 5.15]],
            "std": [[120.0, 121.0, 353.0, 0.0378], [121.0, 120.0, 351.0, 0.0379]],
        }
        self.state = {"event_model.parent_cfm.weight": torch.ones(2)}

    def _load(
        self,
        *,
        checkpoint_fit: dict | None = None,
        target_fit: dict | None = None,
        scaler: dict | None = None,
    ) -> None:
        metadata = {
            "topology_signature": "topology",
            "input_features": ["jet"],
            "source_type_ids": {"jet": 0},
            "architecture_component_signatures": {"partonic_cfm": "parent-cfm"},
            "partonic_scaler_fit": copy.deepcopy(
                self.fit if target_fit is None else target_fit
            ),
        }
        module = SimpleNamespace(
            hparams=copy.deepcopy(metadata),
            partonic_scaler_metadata=copy.deepcopy(self.scaler if scaler is None else scaler),
            state_dict=lambda: {key: value.clone() for key, value in self.state.items()},
            load_state_dict=lambda values, strict=False: self.state.update(values),
        )
        checkpoint = {
            "state_dict": self.state,
            "hyper_parameters": {
                **metadata,
                "partonic_scaler_fit": copy.deepcopy(
                    self.fit if checkpoint_fit is None else checkpoint_fit
                ),
            },
            "partonic_scaler": copy.deepcopy(self.scaler),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "initial.ckpt"
            torch.save(checkpoint, path)
            _load_initial_components(module, path, ("partonic_cfm",))

    def test_accepts_only_roundoff_for_identical_training_fit(self) -> None:
        near = copy.deepcopy(self.scaler)
        near["std"][0][3] += 2.8e-10
        self._load(scaler=near)

    def test_rejects_different_training_membership(self) -> None:
        other_fit = {**self.fit, "event_id_hash": "different"}
        with self.assertRaisesRegex(ValueError, "different training fit"):
            self._load(target_fit=other_fit)

    def test_rejects_different_role_order(self) -> None:
        other_scaler = copy.deepcopy(self.scaler)
        other_scaler["role_names"] = ["antitop", "top"]
        with self.assertRaisesRegex(ValueError, "incompatible train-fitted scaler metadata"):
            self._load(scaler=other_scaler)


if __name__ == "__main__":
    unittest.main()
