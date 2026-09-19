#!/usr/bin/env python3
"""Deterministic tests for continuation and downstream release guards."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
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
        subresult = root / "subresult.json"
        subresult.write_text(json.dumps({"schema": "mini_v1", "status": "PASS", "source_sha": "a" * 40}))
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
            "subresults": {"mini": {"path": str(subresult), "schema": "mini_v1"}},
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
        first = controller.continue_campaign(PARENT, target)
        second = controller.continue_campaign(PARENT, target)
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
    check_continuation()
    print("PASS: deterministic continuation and stage-release contract checks")


if __name__ == "__main__":
    main()
