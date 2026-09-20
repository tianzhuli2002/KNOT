#!/usr/bin/env python3
"""Deterministic, Slurm-free checks for the Paper-1 campaign transitions."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.campaign_controller import (  # noqa: E402
    _controller_dependency,
    _job_group,
    _validate_parent_invisible_backbone_gate,
    campaign_paths,
    continue_campaign,
    transition_for_observation,
)


def _initial() -> dict:
    return {"state": "WAITING_SMOKE", "stage": None}


def check_happy_path() -> None:
    state = transition_for_observation(_initial(), {"smoke": "PASS"})
    if state != {"state": "STUDY13_RUNNING", "stage": "invisible_coordinates"}:
        raise AssertionError(f"smoke promotion failed: {state}")
    state = transition_for_observation(
        state, {"study13": "COMPLETE", "decision_status": "AUTO_SELECTED", "decision": "met_residual"}
    )
    if state["state"] != "STUDY15_PREFLIGHT_RUNNING":
        raise AssertionError("Study 13 did not arm the Study-15 preflight")
    state = transition_for_observation(
        state, {"study15": "COMPLETE", "decision_status": "AUTO_SELECTED", "decision": "conventional"}
    )
    if state["state"] != "STUDY10_PREFLIGHT_RUNNING":
        raise AssertionError("Study 15 did not arm the Study-10 preflight")
    state = transition_for_observation(
        state, {"study10": "COMPLETE", "decision_status": "AUTO_SELECTED", "decision": "none"}
    )
    if state["state"] != "READY_FOR_PRODUCTION_REVIEW":
        raise AssertionError("Study 10 no-coupling path did not stop before production review")


def check_review_and_failure_paths() -> None:
    state = transition_for_observation(_initial(), {"smoke": "FAILED"})
    if state["state"] != "SMOKE_FAILED":
        raise AssertionError("failed smoke was not blocked")
    state = transition_for_observation(
        transition_for_observation(_initial(), {"smoke": "PASS"}),
        {"study13": "COMPLETE", "decision_status": "REVIEW_REQUIRED"},
    )
    if state["state"] != "STUDY13_REVIEW":
        raise AssertionError("ambiguous Study 13 was not stopped for review")
    failed = transition_for_observation(
        {"state": "STUDY15_RUNNING", "stage": "invisible_backbone"},
        {"study15": "FAILED"},
    )
    if failed["state"] != "BLOCKED":
        raise AssertionError("training failure was incorrectly treated as repairable")


def check_duplicate_dependency_identity() -> None:
    if _controller_dependency(["12", "12", "13", "12"]) != "12:13":
        raise AssertionError("controller dependency identity is not deterministic/deduplicated")
    original = _initial()
    repeated = transition_for_observation(copy.deepcopy(original), {})
    if repeated != original:
        raise AssertionError("no-op controller observation changed state")


def check_scheduler_classification() -> None:
    import joint_event_reconstruction.campaign_controller as controller

    with tempfile.TemporaryDirectory(prefix="jer-controller-scheduler-") as directory:
        paths = controller.CampaignPaths(Path(directory), Path(directory), Path(directory) / "campaign")
        original = controller._job_state
        try:
            states = {"1": ("PENDING", "QOSGrpGRES"), "2": ("COMPLETED", ""), "3": ("FAILED", "exit 1")}
            controller._job_state = lambda _paths, job: states[str(job)]
            if _job_group(paths, ["1", "2"])[0] != "PENDING":
                raise AssertionError("pending Slurm state was not preserved")
            if _job_group(paths, ["2", "3"])[0] != "FAILED":
                raise AssertionError("failed Slurm state was not propagated")
            states["1"] = ("COMPLETED", "")
            if _job_group(paths, ["1", "2"])[0] != "COMPLETE":
                raise AssertionError("completed Slurm jobs were not recognised")
        finally:
            controller._job_state = original


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_study15_aware_continuation() -> None:
    """A blocked downstream parent can inherit only its completed Study-15 result."""
    import joint_event_reconstruction.campaign_controller as controller

    parent_root = ROOT / "results/campaigns/paper1-fe33fc854009-a1"
    parent_files = [
        parent_root / "campaign.json",
        parent_root / "gates.json",
        parent_root / "decisions/invisible_coordinates.json",
        parent_root / "decisions/invisible_backbone.json",
    ]
    before = {path: _file_digest(path) for path in parent_files}
    original_git = controller._git
    current_sha = original_git(controller.campaign_paths(), "rev-parse", "HEAD")

    def clean_checkout_git(paths, *arguments):
        if arguments == ("rev-parse", "HEAD"):
            return current_sha
        if arguments == ("status", "--porcelain"):
            return ""
        return original_git(paths, *arguments)

    with tempfile.TemporaryDirectory(prefix="jer-study15-continuation-") as directory:
        target = Path(directory) / "paper1-test-child"
        with patch.object(controller, "_git", side_effect=clean_checkout_git), patch.object(
            controller, "_stage_job_status", return_value=("COMPLETE", "fixture")
        ), patch.object(
            controller, "_stage_output_complete", return_value=(True, "fixture")
        ), patch.object(controller, "_stage_release_ready", return_value=(True, "fixture")):
            child = continue_campaign(
                parent_root,
                target,
                inherit_through="invisible_backbone",
            )
            repeated = continue_campaign(
                parent_root,
                target,
                inherit_through="invisible_backbone",
            )
        if repeated != child:
            raise AssertionError("Study-15-aware continuation is not idempotent")
        if child["state"] != "STUDY10_PREFLIGHT_REQUIRED" or child["stage"] != "assignment_coupling":
            raise AssertionError(f"Continuation did not start at Study-10 preflight: {child['state']}")
        if child["parent_campaign"]["inherit_through"] != "invisible_backbone":
            raise AssertionError("Continuation boundary was not recorded")
        if child["parent_campaign"]["imported_gates"] != ["invisible_coordinates", "invisible_backbone"]:
            raise AssertionError("Continuation did not import both completed scientific gates")
        if child["stages"]["invisible_coordinates"]["job_ids"] or child["stages"]["invisible_backbone"]["job_ids"]:
            raise AssertionError("Inherited Study-13/15 stages acquired new job IDs")
        if any(
            child["stages"][stage]["submission_attempts"]
            for stage in ("invisible_coordinates", "invisible_backbone")
        ):
            raise AssertionError("Continuation recorded a new upstream scientific submission")
        if child["stages"]["assignment_coupling"]["submission_attempts"]:
            raise AssertionError("Continuation submitted Study 10 unexpectedly")
        gates = json.loads((target / "gates.json").read_text())
        if gates["invisible_coordinates"]["selected_variant"] != "met_residual":
            raise AssertionError("Inherited Study-13 selection changed")
        if gates["invisible_backbone"]["selected_variant"] != "conventional":
            raise AssertionError("Inherited Study-15 selection changed")
        if gates["invisible_backbone"]["validated_sha"] != "fe33fc854009f66e1e318ad8ce9fb7ed01354147":
            raise AssertionError("Inherited Study-15 gate provenance changed")
        for path, digest in before.items():
            if _file_digest(path) != digest:
                raise AssertionError(f"Continuation modified completed parent artifact: {path}")

    with tempfile.TemporaryDirectory(prefix="jer-study13-continuation-") as directory:
        target = Path(directory) / "paper1-study13-child"
        with patch.object(controller, "_git", side_effect=clean_checkout_git), patch.object(
            controller, "_stage_release_ready", return_value=(True, "fixture")
        ):
            child = continue_campaign(parent_root.parent / "paper1-f2084c8b2308-a1", target)
        if child["state"] != "STUDY15_PREFLIGHT_REQUIRED":
            raise AssertionError("Study-13-only continuation no longer starts at Study-15 preflight")

    with tempfile.TemporaryDirectory(prefix="jer-invalid-parent-") as directory:
        fixture = Path(directory) / "parent"
        fixture.mkdir()
        source = parent_root
        for relative in ("campaign.json", "gates.json", "decisions/invisible_coordinates.json", "decisions/invisible_backbone.json"):
            destination = fixture / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((source / relative).read_bytes())
        state_path = fixture / "campaign.json"
        invalid = json.loads(state_path.read_text())
        invalid["stages"]["invisible_backbone"]["selected_variant"] = "conditional_lgatr"
        state_path.write_text(json.dumps(invalid))
        try:
            _validate_parent_invisible_backbone_gate(
                campaign_paths(fixture),
                invalid,
                json.loads((fixture / "gates.json").read_text())["invisible_coordinates"],
            )
        except controller.CampaignError:
            pass
        else:
            raise AssertionError("Invalid Study-15 parent was accepted")


def main() -> None:
    check_happy_path()
    check_review_and_failure_paths()
    check_duplicate_dependency_identity()
    check_scheduler_classification()
    check_study15_aware_continuation()
    print("PASS: deterministic Paper-1 campaign controller transitions")


if __name__ == "__main__":
    main()
