#!/usr/bin/env python3
"""Deterministic, Slurm-free checks for the Paper-1 campaign transitions."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from joint_event_reconstruction.campaign_controller import (  # noqa: E402
    _controller_dependency,
    _job_group,
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


def main() -> None:
    check_happy_path()
    check_review_and_failure_paths()
    check_duplicate_dependency_identity()
    check_scheduler_classification()
    print("PASS: deterministic Paper-1 campaign controller transitions")


if __name__ == "__main__":
    main()
