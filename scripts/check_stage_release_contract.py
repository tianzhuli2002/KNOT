#!/usr/bin/env python3
"""Deterministic tests for continuation and downstream release guards."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from joint_event_reconstruction import campaign_controller as controller
from joint_event_reconstruction.stage_release import StageReleaseError, validate_release_payload


PROJECT = Path(__file__).resolve().parents[1]
PARENT = PROJECT / "results/campaigns/paper1-f2084c8b2308-a1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _must_fail(function, text: str) -> None:
    try:
        function()
    except (StageReleaseError, controller.CampaignError):
        return
    raise AssertionError(f"expected failure: {text}")


def check_release_payload() -> None:
    with tempfile.TemporaryDirectory(prefix="jer-stage-release-") as directory:
        root = Path(directory)
        checkpoint = root / "selected.ckpt"
        checkpoint.write_bytes(b"selected checkpoint")
        conditional_checkpoint = root / "conditional.ckpt"
        conditional_checkpoint.write_bytes(b"conditional checkpoint")
        identity = root / "artifact_identity.json"
        identity.write_text(json.dumps({"schema": "identity_v1", "source_sha": "a" * 40}))
        prediction = {
            "physical_four_vectors": True,
            "event_id_unique": True,
            "events": 16,
            "four_vector_convention": "E_px_py_pz_GeV_v1",
            "mass_shell": "fixed massless exact",
            "mass_shell_evidence": {
                "max_energy_residual": 0.0001,
                "max_relative_energy_residual": 1.0e-7,
                "storage_dtype": "float32",
                "ulp_tolerance": 8.0,
            },
            "top_k": 10,
            "posterior_samples": 32,
            "event_id_sha256": "e" * 64,
        }
        evaluation = {
            "primary_metric_path": "invisible.top1_assignment_conditioned.energy_score_joint",
            "primary_metric": 1.0,
            "conditional_calibration": True,
        }
        conventional = root / "conventional.json"
        conditional = root / "conditional.json"
        conventional.write_text(json.dumps({"schema": "study15_branch_v1", "status": "PASS", "source_sha": "a" * 40}))
        conditional.write_text(json.dumps({"schema": "study15_branch_v1", "status": "PASS", "source_sha": "a" * 40}))
        integration = root / "integration.json"
        selected_record = {"path": str(checkpoint), "sha256": _sha(checkpoint)}
        conditional_record = {
            "path": str(conditional_checkpoint),
            "sha256": _sha(conditional_checkpoint),
            "identity_file": str(identity),
            "identity_sha256": _sha(identity),
            "training_sha": "a" * 40,
        }
        integration_payload = {
            "schema": "jer_study15_integration_v1",
            "status": "PASS",
            "source_sha": "a" * 40,
            "test_accessed": False,
            "selected_checkpoint_records": {
                gate: selected_record
                for gate in ("visible_assignment", "invisible_generator", "invisible_coordinates", "assignment_formulation")
            },
            "branches": {
                "conventional": {
                    "training_performed": False,
                    "checkpoint_reused": True,
                    "checkpoint": selected_record,
                    "prediction": prediction,
                    "evaluation": evaluation,
                },
                "conditional_lgatr": {
                    "training": {
                        "finite_initial_loss": True,
                        "finite_gradients": True,
                        "nonzero_invisible_update": True,
                        "finite_validation_loss": True,
                        "coordinate_inverse_check": True,
                        "data_subset": {
                            "train_validation_batch_disjoint": True,
                            "test_accessed": False,
                        },
                    },
                    "checkpoint_reload": {
                        "strict_reload": True,
                        "frozen_visible_equal": {"equal": True},
                        "frozen_assignment_equal": {"equal": True},
                        "backbone": "lgatr",
                    },
                    "checkpoint": conditional_record,
                    "prediction": prediction,
                    "evaluation": evaluation,
                },
            },
            "config_diff": {"status": "PASS", "unexpected_differences": []},
            "conditional_lgatr_backbone": {"type": "lgatr"},
            "subresults": {
                "conventional": {"path": str(conventional), "schema": "study15_branch_v1"},
                "conditional_lgatr": {"path": str(conditional), "schema": "study15_branch_v1"},
            },
        }
        integration.write_text(json.dumps(integration_payload))
        artifact = root / "release.json"
        payload = {
            "schema": "study15_release_v1",
            "status": "PASS",
            "software_integration_evidence_only": True,
            "source_sha": "a" * 40,
            "campaign_id": "paper1-test",
            "slurm_job_id": "123",
            "slurm_terminal_state": "COMPLETED",
            "slurm_exit_code": "0:0",
            "upstream_gate_digest": "b" * 64,
            "manifest_sha256": "c" * 64,
            "required_components": {"mini": True},
            "checkpoint_hashes": {str(checkpoint): _sha(checkpoint)},
            "subresults": {"integration": {"path": str(integration), "schema": "jer_study15_integration_v1"}},
        }
        validate_release_payload(
            payload,
            artifact_path=artifact,
            expected_schema="study15_release_v1",
            expected_source_sha="a" * 40,
            expected_campaign_id="paper1-test",
            expected_gate_digest="b" * 64,
            expected_manifest_sha256="c" * 64,
            expected_checkpoint_hashes={str(checkpoint): _sha(checkpoint)},
            validation_root=root,
        )
        bad_evidence = copy.deepcopy(integration_payload)
        bad_evidence["branches"]["conditional_lgatr"]["training"]["finite_gradients"] = False
        integration.write_text(json.dumps(bad_evidence))
        _must_fail(
            lambda: validate_release_payload(
                payload,
                artifact_path=artifact,
                expected_schema="study15_release_v1",
                expected_source_sha="a" * 40,
                expected_campaign_id="paper1-test",
                expected_gate_digest="b" * 64,
                expected_manifest_sha256="c" * 64,
                expected_checkpoint_hashes={str(checkpoint): _sha(checkpoint)},
                validation_root=root,
            ),
            "unsupported Study-15 evidence boolean",
        )
        integration.write_text(json.dumps(integration_payload))
        bad_split = copy.deepcopy(integration_payload)
        bad_split["branches"]["conditional_lgatr"]["training"]["data_subset"]["train_validation_batch_disjoint"] = False
        integration.write_text(json.dumps(bad_split))
        _must_fail(
            lambda: validate_release_payload(
                payload,
                artifact_path=artifact,
                expected_schema="study15_release_v1",
                expected_source_sha="a" * 40,
                expected_campaign_id="paper1-test",
                expected_gate_digest="b" * 64,
                expected_manifest_sha256="c" * 64,
                expected_checkpoint_hashes={str(checkpoint): _sha(checkpoint)},
                validation_root=root,
            ),
            "Study-15 train/validation split evidence",
        )
        integration.write_text(json.dumps(integration_payload))
        for key, value in (
            ("status", "EDITED"),
            ("source_sha", "d" * 40),
            ("campaign_id", "paper1-other"),
            ("manifest_sha256", "d" * 64),
            ("upstream_gate_digest", "d" * 64),
        ):
            bad = copy.deepcopy(payload)
            bad[key] = value
            _must_fail(
                lambda bad=bad: validate_release_payload(
                    bad,
                    artifact_path=artifact,
                    expected_schema="study15_release_v1",
                    expected_source_sha="a" * 40,
                    expected_campaign_id="paper1-test",
                    expected_gate_digest="b" * 64,
                    expected_manifest_sha256="c" * 64,
                    expected_checkpoint_hashes={str(checkpoint): _sha(checkpoint)},
                    validation_root=root,
                ),
                key,
            )
        missing = copy.deepcopy(payload)
        missing["subresults"] = {"missing": {"path": str(root / "missing.json"), "schema": "mini_v1"}}
        _must_fail(
            lambda: validate_release_payload(
                missing,
                artifact_path=artifact,
                expected_schema="study15_release_v1",
                expected_source_sha="a" * 40,
                expected_campaign_id="paper1-test",
                expected_gate_digest="b" * 64,
                expected_manifest_sha256="c" * 64,
                expected_checkpoint_hashes={str(checkpoint): _sha(checkpoint)},
                validation_root=root,
            ),
            "missing subresult",
        )
        production = copy.deepcopy(payload)
        production["subresults"] = {"production": {"path": str(PROJECT / "results/studies/forbidden.json"), "schema": "mini_v1"}}
        _must_fail(
            lambda: validate_release_payload(
                production,
                artifact_path=artifact,
                expected_schema="study15_release_v1",
                expected_source_sha="a" * 40,
                expected_campaign_id="paper1-test",
                expected_gate_digest="b" * 64,
                expected_manifest_sha256="c" * 64,
                expected_checkpoint_hashes={str(checkpoint): _sha(checkpoint)},
                validation_root=root,
            ),
            "production-path subresult",
        )


def check_terminal_preflight_reporting() -> None:
    """A terminal failure replaces, rather than preserves, an old pending reason."""
    release = {
        "status": "PENDING",
        "reason": "stage preflight 123 is PENDING: QOSGrpGRES",
    }
    controller._record_stage_preflight_observation(
        release,
        "123",
        {
            "state": "FAILED",
            "state_raw": "FAILED",
            "exit_code": "1:0",
            "reason": "state=FAILED; exit_code=1:0",
        },
    )
    if release.get("status") != "FAILED" or release.get("scheduler_state") != "FAILED":
        raise AssertionError("Terminal stage preflight failure was not recorded")
    if release.get("scheduler_exit_code") != "1:0" or "QOSGrpGRES" in release.get("reason", ""):
        raise AssertionError("Terminal stage preflight retained the stale pending reason")
    paths = controller.CampaignPaths(Path("/tmp"), Path("/tmp"), Path("/tmp/jer-stage-release-test"))
    original_job_state = controller._job_state
    original_run = controller._run
    try:
        controller._job_state = lambda _paths, _job: ("FAILED", "accounting terminal failure")
        controller._run = lambda _paths, _command, check=False: subprocess.CompletedProcess(
            _command, 0, stdout="123|FAILED|1:0\n", stderr=""
        )
        observation = controller._scheduler_observation(paths, "123")
    finally:
        controller._job_state = original_job_state
        controller._run = original_run
    if observation["state"] != "FAILED" or observation["exit_code"] != "1:0":
        raise AssertionError(f"Scheduler terminal evidence was not read exactly: {observation}")


def check_continuation() -> None:
    if not PARENT.is_dir():
        raise AssertionError(f"completed parent campaign is missing: {PARENT}")
    parent_files = {
        path.relative_to(PARENT): _sha(path)
        for path in PARENT.rglob("*")
        if path.is_file() and path.name != "campaign.lock"
    }
    with tempfile.TemporaryDirectory(prefix="jer-continuation-") as directory:
        target = Path(directory) / "paper1-test-continuation-a1"
        original_git = controller._git

        def clean_fixture_git(paths, *arguments):
            if arguments == ("status", "--porcelain"):
                return ""
            return original_git(paths, *arguments)

        controller._git = clean_fixture_git
        try:
            first = controller.continue_campaign(PARENT, target)
            second = controller.continue_campaign(PARENT, target)
        finally:
            controller._git = original_git
        if first != second:
            raise AssertionError("repeated continuation is not idempotent")
        if first["parent_campaign"]["campaign_id"] != "paper1-f2084c8b2308-a1":
            raise AssertionError("parent campaign provenance was not retained")
        if first["parent_campaign"]["source_sha"] != "f2084c8b2308a7567cec8a269520c32dff43638f":
            raise AssertionError("parent source provenance was changed")
        if first["stages"]["invisible_coordinates"].get("submission_record"):
            raise AssertionError("continuation inherited a Study-13 submission")
        if first["smoke"]["status"] != "INHERITED":
            raise AssertionError("continuation did not mark the old release as inherited")
        inherited = first["gates_path"]
        gates = json.loads(Path(inherited).read_text())
        if gates["invisible_coordinates"]["validated_sha"] != "f2084c8b2308a7567cec8a269520c32dff43638f":
            raise AssertionError("imported gate training/validation SHA was rewritten")
        after_files = {
            path.relative_to(PARENT): _sha(path)
            for path in PARENT.rglob("*")
            if path.is_file() and path.name != "campaign.lock"
        }
        if parent_files != after_files:
            raise AssertionError("parent campaign changed during continuation")
        copied_gate = copy.deepcopy(gates["invisible_coordinates"])
        copied_gate["artifacts"]["ttbar2L"]["seed_42"]["checkpoint_sha256"] = "0" * 64
        broken_root = Path(directory) / "broken-hash"
        shutil.copytree(PARENT, broken_root)
        broken_gates = json.loads((broken_root / "gates.json").read_text())
        broken_gates["invisible_coordinates"] = copied_gate
        (broken_root / "gates.json").write_text(json.dumps(broken_gates))
        _must_fail(
            lambda: controller._validate_parent_coordinate_gate(
                controller.campaign_paths(broken_root), json.loads((broken_root / "campaign.json").read_text())
            ),
            "mismatched parent checkpoint hash",
        )
        missing_root = Path(directory) / "missing-identity"
        shutil.copytree(PARENT, missing_root)
        missing_gates = json.loads((missing_root / "gates.json").read_text())
        missing_gates["invisible_coordinates"]["artifacts"]["ttbar2L"]["seed_42"]["identity_file"] = str(missing_root / "does-not-exist.json")
        (missing_root / "gates.json").write_text(json.dumps(missing_gates))
        _must_fail(
            lambda: controller._validate_parent_coordinate_gate(
                controller.campaign_paths(missing_root), json.loads((missing_root / "campaign.json").read_text())
            ),
            "missing parent identity",
        )


def main() -> None:
    check_release_payload()
    check_terminal_preflight_reporting()
    check_continuation()
    print("PASS: deterministic continuation and stage-release contract checks")


if __name__ == "__main__":
    main()
