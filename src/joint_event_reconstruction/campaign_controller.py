"""Event-driven Paper-1 campaign controller.

The controller is deliberately a short-lived transition function.  Slurm runs
one controller job after each smoke/study DAG with ``afterany``; the controller
checks the immutable outputs, makes a policy-backed decision, submits the next
stage, and exits.  No process is expected to remain alive between stages.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import shutil
import subprocess
import sys
import time
from typing import Iterator

from omegaconf import OmegaConf

from joint_event_reconstruction.runtime_paths import (
    ensure_source_snapshot,
    runtime_paths,
    verify_source_snapshot,
)
from joint_event_reconstruction.studies import (
    ScientificGateNotApplicable,
    ScientificGatePending,
    load_scientific_gates,
    load_study,
    resolve_study,
)
from joint_event_reconstruction.stage_release import (
    StageReleaseError,
    gate_checkpoint_hashes,
    sha256_file,
    validate_release_payload,
)


PAPER1 = "paper1"
STAGES = (
    "invisible_coordinates",
    "invisible_backbone",
    "assignment_coupling",
    "assignment_conditioning",
    "final_production",
)
STAGE_STUDIES = {
    "invisible_coordinates": "invisible_coordinates",
    "invisible_backbone": "lgatr_invisible_generator",
    "assignment_coupling": "compatibility_reranking",
    "assignment_conditioning": "assignment_conditioning",
}
STAGE_GATES = {
    "invisible_coordinates": "invisible_coordinates",
    "invisible_backbone": "invisible_backbone",
    "assignment_coupling": "assignment_coupling",
    "assignment_conditioning": "assignment_conditioning",
}
STAGE_RELEASE_CONTRACTS = {
    "invisible_backbone": "study15_release_v1",
    "assignment_coupling": "study10_release_v1",
    "assignment_conditioning": "study03_release_v1",
}
STATES = {
    "NEW",
    "WAITING_SMOKE",
    "SMOKE_RUNNING",
    "SMOKE_FAILED",
    "STUDY13_RUNNING",
    "STUDY13_REPAIR",
    "STUDY13_REVIEW",
    "STUDY15_RUNNING",
    "STUDY15_PREFLIGHT_REQUIRED",
    "STUDY15_PREFLIGHT_RUNNING",
    "STUDY15_REPAIR",
    "STUDY15_REVIEW",
    "STUDY10_RUNNING",
    "STUDY10_PREFLIGHT_REQUIRED",
    "STUDY10_PREFLIGHT_RUNNING",
    "STUDY10_REPAIR",
    "STUDY10_REVIEW",
    "STUDY03_RUNNING",
    "STUDY03_PREFLIGHT_REQUIRED",
    "STUDY03_PREFLIGHT_RUNNING",
    "STUDY03_REPAIR",
    "STUDY03_REVIEW",
    "PRODUCTION_RUNNING",
    "READY_FOR_PRODUCTION_REVIEW",
    "READY_FOR_FINAL_TEST",
    "BLOCKED",
    "FAILED",
}
ACTIVE_SCHEDULER_STATES = {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING"}
FAILURE_STATES = {
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
}
TERMINAL_STATES = FAILURE_STATES | {"COMPLETED"}


class CampaignError(RuntimeError):
    """A campaign transition cannot safely proceed."""


@dataclass(frozen=True)
class CampaignPaths:
    project_root: Path
    source_root: Path
    root: Path

    @property
    def state(self) -> Path:
        return self.root / "campaign.json"

    @property
    def lock(self) -> Path:
        return self.root / "campaign.lock"

    @property
    def policy_source(self) -> Path:
        return self.source_root / "configs/campaigns/paper1.yaml"

    @property
    def policy_snapshot(self) -> Path:
        return self.root / "policy.snapshot.yaml"

    @property
    def source_json(self) -> Path:
        return self.root / "source.json"

    @property
    def gates(self) -> Path:
        return self.root / "gates.json"

    @property
    def decisions(self) -> Path:
        return self.root / "decisions"

    @property
    def controllers(self) -> Path:
        return self.root / "controllers"

    @property
    def submissions(self) -> Path:
        return self.root / "submissions"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


def campaign_paths(root: str | Path | None = None) -> CampaignPaths:
    paths = runtime_paths()
    project_root = paths.project_root
    source_root = paths.source_root
    campaign_root = (
        Path(root).expanduser().resolve()
        if root is not None
        else paths.results_root / "campaigns" / PAPER1
    )
    return CampaignPaths(project_root, source_root, campaign_root)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    _write_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"Cannot read campaign JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CampaignError(f"Campaign JSON must be a mapping: {path}")
    return value


@contextmanager
def _campaign_lock(paths: CampaignPaths) -> Iterator[None]:
    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _save_state(paths: CampaignPaths, state: dict) -> None:
    state["updated_at"] = _now()
    if state.get("state") not in STATES:
        raise CampaignError(f"Unknown campaign state {state.get('state')!r}")
    _write_json(paths.state, state)


def _set_state(state: dict, value: str, *, reason: str | None = None) -> None:
    if value not in STATES:
        raise CampaignError(f"Unknown campaign state {value!r}")
    state["state"] = value
    if reason is not None:
        state["state_reason"] = reason


def _run(paths: CampaignPaths, command: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "JER_PROJECT_ROOT": str(paths.project_root),
            "JER_SOURCE_ROOT": str(paths.source_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    if len(paths.source_root.name) == 40:
        environment["JER_SOURCE_SHA"] = paths.source_root.name
    else:
        environment.pop("JER_SOURCE_SHA", None)
    pythonpath = f"{paths.source_root / 'src'}:{paths.source_root}"
    if environment.get("PYTHONPATH"):
        pythonpath += os.pathsep + environment["PYTHONPATH"]
    environment["PYTHONPATH"] = pythonpath
    return subprocess.run(
        command,
        cwd=paths.project_root,
        env=environment,
        text=True,
        capture_output=True,
        check=check,
    )


def _git(paths: CampaignPaths, *arguments: str) -> str:
    result = _run(paths, ["git", *arguments], check=False)
    if result.returncode:
        raise CampaignError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _normalise_state(value: str) -> str:
    return value.strip().split("+", 1)[0].split(" ", 1)[0].upper()


def _job_state(paths: CampaignPaths, job_id: object) -> tuple[str, str]:
    if not job_id:
        return "NONE", ""
    job = str(job_id)
    queue = _run(paths, ["squeue", "-h", "-j", job, "-o", "%T|%R"], check=False)
    if queue.returncode == 0 and queue.stdout.strip():
        state, _, reason = queue.stdout.strip().split("\n", 1)[0].partition("|")
        return _normalise_state(state), reason
    account = _run(
        paths,
        ["sacct", "-n", "-P", "-j", job, "--format=JobIDRaw,State,ExitCode"],
        check=False,
    )
    rows = [line.strip().split("|") for line in account.stdout.splitlines() if line.strip()]
    if not rows:
        return "UNKNOWN", "no scheduler record"
    states = [_normalise_state(row[1]) for row in rows if len(row) > 1]
    if any(state in FAILURE_STATES for state in states):
        state = next(state for state in states if state in FAILURE_STATES)
        return state, "accounting terminal failure"
    if any(state in ACTIVE_SCHEDULER_STATES for state in states):
        state = next(state for state in states if state in ACTIVE_SCHEDULER_STATES)
        return state, "scheduler active"
    if states and all(state == "COMPLETED" for state in states):
        return "COMPLETED", "accounting complete"
    return states[-1] if states else "UNKNOWN", "accounting incomplete"


def _scheduler_terminal(paths: CampaignPaths, job_id: object) -> tuple[str, str]:
    """Return the actual terminal state and exit code for one Slurm job."""
    state, reason = _job_state(paths, job_id)
    if state != "COMPLETED":
        return state, reason
    account = _run(
        paths,
        ["sacct", "-n", "-P", "-j", str(job_id), "--format=JobIDRaw,State,ExitCode"],
        check=False,
    )
    rows = [line.strip().split("|") for line in account.stdout.splitlines() if line.strip()]
    exact = next((row for row in rows if row and row[0] == str(job_id)), None)
    if exact is None:
        exact = next((row for row in rows if len(row) > 2 and "." not in row[0]), None)
    if exact is None or len(exact) < 3:
        return "UNKNOWN", "completed job has no exact sacct exit-code record"
    exit_code = exact[2].strip()
    if exit_code != "0:0":
        return "FAILED", f"Slurm exit code {exit_code}"
    return "COMPLETED", exit_code


def _job_group(paths: CampaignPaths, job_ids: list[str]) -> tuple[str, str]:
    unique = list(dict.fromkeys(str(job) for job in job_ids if job))
    if not unique:
        return "NONE", "no jobs"
    observations = [(job, *_job_state(paths, job)) for job in unique]
    failures = [(job, state, reason) for job, state, reason in observations if state in FAILURE_STATES]
    if failures:
        return "FAILED", "; ".join(f"{job}={state}" for job, state, _ in failures)
    active = [(job, state, reason) for job, state, reason in observations if state in ACTIVE_SCHEDULER_STATES]
    if active:
        if any(state in {"RUNNING", "COMPLETING"} for _, state, _ in active):
            return "RUNNING", "; ".join(f"{job}={state} {reason}" for job, state, reason in active)
        return "PENDING", "; ".join(f"{job}={state} {reason}" for job, state, reason in active)
    if all(state == "COMPLETED" for _, state, _ in observations):
        return "COMPLETE", f"{len(observations)} jobs completed"
    unknown = "; ".join(f"{job}={state}" for job, state, _ in observations)
    return "UNKNOWN", unknown


def _study_job_ids(record: dict) -> list[str]:
    ids: list[str] = []
    for level in record.get("train_levels", []):
        if level.get("job_id"):
            ids.append(str(level["job_id"]))
    for key in (
        "prediction_job",
        "evaluation_job",
        "plot_job",
        "aggregate_job",
        "postprocess_job",
    ):
        if record.get(key):
            ids.append(str(record[key]))
    return list(dict.fromkeys(ids))


def _study_job_groups(record: dict) -> dict[str, list[str]]:
    training = [
        str(level["job_id"])
        for level in record.get("train_levels", [])
        if level.get("job_id")
    ]
    prediction = [str(record["prediction_job"])] if record.get("prediction_job") else []
    postprocess = [
        str(record[key])
        for key in ("evaluation_job", "plot_job", "aggregate_job", "postprocess_job")
        if record.get(key)
    ]
    return {
        "training": list(dict.fromkeys(training)),
        "prediction": list(dict.fromkeys(prediction)),
        "postprocess": list(dict.fromkeys(postprocess)),
    }


def _load_policy(paths: CampaignPaths) -> dict:
    if not paths.policy_snapshot.is_file():
        raise CampaignError(f"Campaign policy snapshot is missing: {paths.policy_snapshot}")
    raw = OmegaConf.to_container(OmegaConf.load(paths.policy_snapshot), resolve=True)
    if not isinstance(raw, dict) or set(raw) != {"campaign", "stages", "studies", "decisions", "production"}:
        raise CampaignError("Paper-1 policy must contain campaign, stages, studies, decisions and production")
    if raw["campaign"].get("name") != PAPER1 or raw["campaign"].get("stop_before_test") is not True:
        raise CampaignError("Paper-1 policy must be named paper1 and stop before the real test split")
    expected_studies = {
        "invisible_coordinates": "invisible_coordinates",
        "lgatr_invisible_generator": "lgatr_invisible_generator",
        "compatibility_reranking": "compatibility_reranking",
        "assignment_conditioning_if_needed": "assignment_conditioning",
    }
    if raw["studies"] != expected_studies:
        raise CampaignError(f"Paper-1 policy study allowlist changed: {raw['studies']}")
    if any(name not in expected_studies and name != "final_production" for name in raw["stages"]):
        raise CampaignError("Paper-1 policy contains a study outside the active allowlist")
    return raw


def _study_paths(paths: CampaignPaths) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((paths.source_root / "configs/studies").glob("[0-9][0-9]_*.yaml")):
        study = load_study(path)
        if study.name in result:
            raise CampaignError(f"Duplicate study manifest {study.name!r}")
        result[study.name] = path
    return result


def _source_metadata(paths: CampaignPaths, sha: str) -> dict:
    snapshot = ensure_source_snapshot(sha, project_root=paths.project_root)
    metadata_path = snapshot / "SOURCE_SNAPSHOT.json"
    metadata = _read_json(metadata_path)
    verify_source_snapshot(
        sha,
        project_root=paths.project_root,
        source_root=snapshot,
        expected_tree_sha=metadata.get("tree_sha"),
    )
    return {
        "source_sha": sha,
        "tree_sha": metadata["tree_sha"],
        "source_root": str(snapshot),
        "snapshot_metadata": str(metadata_path),
        "snapshot_metadata_sha256": _sha256_file(metadata_path),
        "canonical_repository": str(paths.project_root),
    }


def _base_gate_digest(gates: dict) -> str:
    return _json_digest(gates)


def _initial_state(paths: CampaignPaths, sha: str, source: dict, policy: dict, gates: dict) -> dict:
    source_root = Path(source["source_root"])
    manifests = {
        name: {
            "path": str(path),
            "sha256": _sha256_file(path),
        }
        for name, path in {
            load_study(path).name: path
            for path in sorted((source_root / "configs/studies").glob("[0-9][0-9]_*.yaml"))
        }.items()
    }
    policy_sha = _sha256_file(paths.policy_source)
    pending = sorted(name for name, gate in gates.items() if gate["status"] == "pending")
    state = {
        "schema": "jer_paper1_campaign_v1",
        "campaign_id": (
            paths.root.name if paths.root.name != PAPER1 else f"paper1-{sha[:12]}"
        ),
        "name": PAPER1,
        "mode": "AUTO",
        "source_sha": sha,
        "source_tree_sha": source["tree_sha"],
        "source_root": source["source_root"],
        "project_root": str(paths.project_root),
        "policy_sha256": policy_sha,
        "policy_snapshot": str(paths.policy_snapshot),
        "study_manifests": manifests,
        "base_gate_digest": _base_gate_digest(gates),
        "base_gates_sha256": _sha256_file(source_root / "configs/studies/UPSTREAM_SELECTIONS.yaml"),
        "gates_path": str(paths.gates),
        "allowed_pending_gates": pending,
        "created_at": _now(),
        "updated_at": _now(),
        "state": "NEW",
        "state_reason": "campaign created; waiting for the composite Study-13 release gate",
        "stage": None,
        "smoke": {
            "status": "UNKNOWN",
            "job_id": None,
            "attempts": 0,
            "release_contract": "jer_study13_release_v1",
        },
        "stages": {
            stage: {
                "state": "NOT_STARTED",
                "study": STAGE_STUDIES.get(stage),
                "submission_attempts": 0,
                "repair_attempts": 0,
                "controller_attempts": 0,
                "job_ids": [],
            }
            for stage in STAGES
        },
        "stage_releases": {
            stage: {
                "contract": contract,
                "status": "NOT_IMPLEMENTED",
                "artifact": str(
                    paths.project_root
                    / "results/validation/stage-release"
                    / sha
                    / {
                        "invisible_backbone": "invisible_backbone.json",
                        "assignment_coupling": "assignment_coupling.json",
                        "assignment_conditioning": "assignment_conditioning.json",
                    }[stage]
                ),
                "reason": "Required downstream production-equivalent preflight is not yet materialised",
            }
            for stage, contract in STAGE_RELEASE_CONTRACTS.items()
        },
        "decisions": {},
        "pending_review": None,
    }
    return state


def _ensure_campaign_files(paths: CampaignPaths, state: dict | None = None) -> None:
    for directory in (paths.decisions, paths.controllers, paths.submissions, paths.logs):
        directory.mkdir(parents=True, exist_ok=True)
    if state is not None:
        _save_state(paths, state)


def _smoke_registry(paths: CampaignPaths) -> tuple[Path, list[dict]]:
    registry = paths.project_root / "results/validation/gpu-smoke/registry.json"
    if not registry.is_file():
        return registry, []
    raw = _read_json(registry)
    records = raw.get("smokes", [])
    if not isinstance(records, list):
        raise CampaignError(f"GPU smoke registry has invalid records: {registry}")
    return registry, records


def _write_smoke_registry(path: Path, records: list[dict]) -> None:
    _write_json(path, {"schema": "jer_gpu_smoke_v1", "smokes": records})


def _release_registry(paths: CampaignPaths) -> tuple[Path, list[dict]]:
    registry = paths.project_root / "results/validation/study13-release/registry.json"
    if not registry.is_file():
        return registry, []
    raw = _read_json(registry)
    records = raw.get("releases", [])
    if not isinstance(records, list):
        raise CampaignError(f"Study-13 release registry has invalid records: {registry}")
    return registry, records


def _write_release_registry(path: Path, records: list[dict]) -> None:
    _write_json(path, {"schema": "jer_study13_release_registry_v1", "releases": records})


def _validate_release_artifact(
    paths: CampaignPaths,
    record: dict,
    sha: str,
    scheduler_state: str,
) -> tuple[bool, str]:
    """Validate the executed composite release, not just a status string."""
    del paths
    if scheduler_state != "COMPLETED":
        return False, f"release scheduler state is {scheduler_state}"
    artifact_path = Path(str(record.get("artifact_path", "")))
    result_path = artifact_path / "result.json"
    if not result_path.is_file():
        return False, f"missing composite release result: {result_path}"
    try:
        result = _read_json(result_path)
    except CampaignError as error:
        return False, str(error)
    if result.get("schema") != "jer_study13_release_v1":
        return False, f"unexpected release schema: {result.get('schema')!r}"
    if result.get("status") != "PASS":
        return False, "composite release result is not PASS"
    if result.get("source_sha") != sha:
        return False, "composite release source SHA does not match campaign"
    if result.get("slurm_terminal_state") not in {"RUNNING", "COMPLETED"}:
        return False, "composite release has no valid Slurm terminal-state marker"
    if record.get("job_id") and str(result.get("slurm_job_id")) != str(record["job_id"]):
        return False, "composite release job ID does not match registry"
    required = result.get("required_components")
    if not isinstance(required, dict) or not required or not all(required.values()):
        return False, "composite release required_components is incomplete"
    smoke = result.get("smoke", {})
    integration = result.get("study13_integration", {})
    smoke_result = Path(str(smoke.get("result", "")))
    integration_result = Path(str(integration.get("result", "")))
    if not smoke_result.is_file() or not integration_result.is_file():
        return False, "composite release sub-results are missing"
    try:
        smoke_payload = _read_json(smoke_result)
        integration_payload = _read_json(integration_result)
    except CampaignError as error:
        return False, str(error)
    if smoke_payload.get("status") != "PASS":
        return False, "selected-model hypothesis smoke is not PASS"
    if smoke_payload.get("model_composition_status") != "PASS":
        return False, "selected-model hypothesis smoke did not pass composition"
    if smoke_payload.get("source_sha") != sha:
        return False, "selected-model hypothesis smoke source SHA does not match campaign"
    if (
        smoke_payload.get("K") != 10
        or smoke_payload.get("M") != 8
        or smoke_payload.get("posterior_branches") != 1
    ):
        return False, "selected-model hypothesis smoke has the wrong posterior contract"
    if not Path(str(smoke_payload.get("prediction", ""))).is_file():
        return False, "selected-model hypothesis smoke prediction is missing"
    composition_path = Path(str(result.get("model_composition", "")))
    if not composition_path.is_file():
        return False, "selected-model composition report is missing"
    try:
        composition = _read_json(composition_path)
    except CampaignError as error:
        return False, str(error)
    if composition.get("schema") != "jer_study13_model_composition_v1":
        return False, "selected-model composition report has the wrong schema"
    if composition.get("status") != "PASS":
        return False, "selected-model composition report is not PASS"
    if composition.get("source_sha") != sha:
        return False, "selected-model composition source SHA does not match campaign"
    standalone = composition.get("standalone_vs_combined")
    if (
        not isinstance(standalone, dict)
        or standalone.get("candidate_hypotheses_equal") is not True
        or standalone.get("candidate_scores_equal") is not True
    ):
        return False, "selected-model standalone/combined output comparison is incomplete"
    if integration_payload.get("schema") != "jer_study13_integration_v1":
        return False, "Study-13 integration result has the wrong schema"
    if integration_payload.get("status") != "PASS":
        return False, "Study-13 integration result is not PASS"
    if integration_payload.get("source_sha") != sha:
        return False, "Study-13 integration source SHA does not match campaign"
    if integration_payload.get("software_integration_evidence_only") is not True:
        return False, "Study-13 integration is not labelled software evidence only"
    subset = integration_payload.get("data_subset", {})
    if subset != {
        "train_events": 32,
        "validation_events": 16,
        "batch_size": 8,
        "workers": 0,
        "subset_seed": 42,
        "test_accessed": False,
    }:
        return False, "Study-13 integration data subset contract is incomplete"
    level_a = integration_payload.get("level_a", {})
    if level_a.get("status") != "PASS" or level_a.get("model_composition", {}).get("status") != "PASS":
        return False, "Study-13 integration Level-A evidence is incomplete"
    population = integration_payload.get("matched_validation_population", {})
    if population.get("all_branches_equal") is not True or population.get("events") != 16:
        return False, "Study-13 integration validation population is not matched"
    branches = integration_payload.get("branches", {})
    for branch in ("cartesian", "met_residual", "met_exact"):
        if not isinstance(branches.get(branch), dict):
            return False, f"Study-13 integration branch is missing: {branch}"
        branch_payload = branches[branch]
        if not isinstance(branch_payload.get("evaluation"), dict):
            return False, f"Study-13 integration evaluation is missing: {branch}"
        prediction = branch_payload.get("prediction", {})
        evaluation = branch_payload["evaluation"]
        if (
            prediction.get("physical_four_vectors") is not True
            or prediction.get("event_id_unique") is not True
            or prediction.get("events") != 16
            or prediction.get("four_vector_convention") != "E_px_py_pz_GeV_v1"
            or not prediction.get("event_id_sha256")
            or not isinstance(evaluation.get("primary_metric"), (int, float))
            or evaluation.get("primary_metric_path")
            != "invisible.top1_assignment_conditioned.energy_score_joint"
        ):
            return False, f"Study-13 integration prediction/evaluation evidence is incomplete: {branch}"
        if branch != "cartesian":
            training = branch_payload.get("training", {})
            reload = branch_payload.get("checkpoint_reload", {})
            if (
                training.get("finite_initial_loss") is not True
                or training.get("finite_gradients") is not True
                or training.get("nonzero_invisible_update") is not True
                or training.get("finite_validation_loss") is not True
                or reload.get("strict_reload") is not True
                or reload.get("frozen_visible_equal", {}).get("equal") is not True
                or reload.get("frozen_assignment_equal", {}).get("equal") is not True
            ):
                return False, f"Study-13 training/checkpoint evidence is incomplete: {branch}"
            identity = branch_payload.get("artifact_identity", {})
            if (
                identity.get("source_sha") != sha
                or identity.get("resolved_gate_digest") != result.get("resolved_gate_digest")
                or not isinstance(identity.get("checkpoint"), dict)
                or not identity["checkpoint"].get("sha256")
                or not identity.get("prediction_sha256")
            ):
                return False, f"Study-13 artifact identity evidence is incomplete: {branch}"
    if result.get("resolved_gate_digest") != integration_payload.get("resolved_gate_digest"):
        return False, "release and integration gate digests differ"
    if not result.get("study_manifest_sha256") or not result.get("resolved_gate_digest"):
        return False, "release is missing immutable manifest or gate identity"
    return True, "composite Study-13 release checks passed"


def _reconcile_smoke(paths: CampaignPaths, sha: str) -> dict | None:
    registry, records = _release_registry(paths)
    matches = [record for record in records if record.get("source_sha") == sha]
    if not matches:
        return None
    record = matches[-1]
    status = str(record.get("status", "UNKNOWN"))
    if status in {"SUBMITTED", "PENDING", "RUNNING", "PASS"} and record.get("job_id"):
        scheduler, reason = _job_state(paths, record["job_id"])
        record["scheduler_state"] = scheduler
        record["scheduler_reason"] = reason
        if scheduler == "COMPLETED":
            passed, detail = _validate_release_artifact(paths, record, sha, scheduler)
            record["status"] = "PASS" if passed else "FAILED"
            record["release_validation"] = detail
            if passed:
                result_path = Path(record["artifact_path"]) / "result.json"
                result = _read_json(result_path)
                result["slurm_terminal_state"] = "COMPLETED"
                result["slurm_exit_code"] = "0:0"
                _write_json(result_path, result)
        elif scheduler in FAILURE_STATES:
            record["status"] = "FAILED"
        elif scheduler in {"PENDING", "CONFIGURING"}:
            record["status"] = "PENDING"
        elif scheduler in {"RUNNING", "COMPLETING"}:
            record["status"] = "RUNNING"
        elif status == "PASS":
            record["status"] = "FAILED"
            record["release_validation"] = f"scheduler state changed to {scheduler}"
        _write_release_registry(registry, records)
    return record


def _latest_preflight(paths: CampaignPaths, sha: str) -> bool:
    root = paths.project_root / "logs/preflight" / sha
    for result in sorted(root.glob("*/result.json"), reverse=True):
        try:
            payload = _read_json(result)
            if (
                payload.get("status") == "PASS"
                and payload.get("study13_level_a", {}).get("status") == "PASS"
            ):
                return True
        except CampaignError:
            continue
    for output in sorted(root.glob("*/jer-preflight-*.out"), reverse=True):
        try:
            payload = _read_json(output)
            if (
                payload.get("status") == "PASS"
                and payload.get("study13_level_a", {}).get("status") == "PASS"
            ):
                return True
        except CampaignError:
            continue
    return False


def _site_slurm(paths: CampaignPaths) -> dict:
    raw = OmegaConf.to_container(
        OmegaConf.load(paths.project_root / "configs/site/csf3.yaml"), resolve=True
    )
    return raw["slurm"]


def _release_source(paths: CampaignPaths, sha: str) -> Path:
    """Return the immutable tree used by a queued composite release worker."""
    source = ensure_source_snapshot(sha, project_root=paths.project_root)
    verify_source_snapshot(sha, project_root=paths.project_root, source_root=source)
    return source


def _submit_smoke(paths: CampaignPaths, sha: str) -> dict:
    if not _latest_preflight(paths, sha):
        raise CampaignError(
            f"No successful CPU preflight exists for source SHA {sha}; run "
            "jer doctor --compute-node before arming the campaign"
        )
    registry, records = _release_registry(paths)
    active = [
        record
        for record in records
        if record.get("source_sha") == sha
        and record.get("status") in {"SUBMITTED", "PENDING", "RUNNING"}
    ]
    if active:
        return active[-1]
    attempt = len(
        list((paths.project_root / "results/validation/study13-release" / sha).glob("attempt-*"))
    ) + 1
    output = paths.project_root / "results/validation/study13-release" / sha / f"attempt-{attempt:03d}"
    output.mkdir(parents=True, exist_ok=False)
    # The live CLI is normally imported from the canonical checkout.  The
    # release worker must nevertheless execute the immutable tree recorded in
    # the campaign state, not that checkout.
    source = _release_source(paths, sha)
    worker = output / "worker.sh"
    worker.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"export JER_PROJECT_ROOT={shlex.quote(str(paths.project_root))}\n"
        f"export JER_SOURCE_ROOT={shlex.quote(str(source))}\n"
        f"export JER_SOURCE_SHA={shlex.quote(sha)}\n"
        f"export PYTHONPATH={shlex.quote(str(source / 'src'))}:{shlex.quote(str(source))}\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        f"cd {shlex.quote(str(paths.project_root))}\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(source / 'scripts/release_study13_gpu.py'))} "
        f"--output-dir {shlex.quote(str(output))} --expected-source-sha {shlex.quote(sha)} "
        f"--gates-path {shlex.quote(str(paths.gates))}\n"
    )
    worker.chmod(0o755)
    slurm = _site_slurm(paths)
    submit = _run(
        paths,
        [
            "sbatch", "--parsable", "--job-name=jer-study13-release",
            f"--partition={slurm['gpu_partition']}", f"--gres={slurm['gpu_gres']}",
            f"--cpus-per-task={slurm['gpu_cpus']}", f"--mem={slurm['gpu_memory']}",
            f"--time={slurm['gpu_time']}", f"--output={output}/%x-%j.out",
            f"--error={output}/%x-%j.err", str(worker),
        ],
        check=False,
    )
    if submit.returncode:
        raise CampaignError(submit.stderr.strip() or "GPU smoke submission failed")
    job_id = submit.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise CampaignError(f"Could not parse GPU smoke job ID: {submit.stdout!r}")
    record = {
        "source_sha": sha,
        "job_id": job_id,
        "status": "SUBMITTED",
        "submitted_at": _now(),
        "artifact_path": str(output),
        "contract": "jer_study13_release_v1",
        "controller_armed": True,
    }
    records.append(record)
    _write_release_registry(registry, records)
    return record


def _submit_stage_preflight(paths: CampaignPaths, state: dict, stage: str) -> dict:
    """Submit exactly one bounded GPU preflight for a downstream stage."""
    if stage not in STAGE_RELEASE_CONTRACTS:
        raise CampaignError(f"No downstream release preflight is defined for {stage!r}")
    release = state.setdefault("stage_releases", {}).setdefault(stage, {})
    existing_job = release.get("preflight_job_id")
    if existing_job:
        scheduler, _ = _job_state(paths, existing_job)
        if scheduler in ACTIVE_SCHEDULER_STATES:
            return {"job_id": str(existing_job), "status": release.get("status", "SUBMITTED")}
        if scheduler == "COMPLETED" and release.get("status") == "PASS":
            return {"job_id": str(existing_job), "status": "PASS"}
        if scheduler in FAILURE_STATES or release.get("status") == "FAILED":
            raise CampaignError(
                f"{stage} preflight already failed ({existing_job}); create a new source-SHA continuation"
            )
    if not _latest_preflight(paths, state["source_sha"]):
        raise CampaignError(
            f"No successful CPU preflight exists for source SHA {state['source_sha']}; "
            "run jer doctor --compute-node before arming the stage preflight"
        )
    source = _release_source(paths, state["source_sha"])
    validation_root = paths.project_root / "results/validation/stage-release" / state["source_sha"]
    attempt = len(list((validation_root / stage).glob("attempt-*"))) + 1
    output = validation_root / stage / f"attempt-{attempt:03d}"
    output.mkdir(parents=True, exist_ok=False)
    artifact_name = {
        "invisible_backbone": "invisible_backbone.json",
        "assignment_coupling": "assignment_coupling.json",
        "assignment_conditioning": "assignment_conditioning.json",
    }[stage]
    artifact = validation_root / artifact_name
    worker = output / "worker.sh"
    worker.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"export JER_PROJECT_ROOT={shlex.quote(str(paths.project_root))}\n"
        f"export JER_SOURCE_ROOT={shlex.quote(str(source))}\n"
        f"export JER_SOURCE_SHA={shlex.quote(state['source_sha'])}\n"
        f"export PYTHONPATH={shlex.quote(str(source / 'src'))}:{shlex.quote(str(source))}\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        f"cd {shlex.quote(str(paths.project_root))}\n"
        f"exec {shlex.quote(sys.executable)} {shlex.quote(str(source / 'scripts/release_stage_gpu.py'))} "
        f"--stage {shlex.quote(stage)} --output-dir {shlex.quote(str(output))} "
        f"--release-artifact {shlex.quote(str(artifact))} "
        f"--campaign-root {shlex.quote(str(paths.root))} "
        f"--expected-source-sha {shlex.quote(state['source_sha'])} "
        f"--gates-path {shlex.quote(str(paths.gates))} "
        f"--campaign-id {shlex.quote(state['campaign_id'])}\n"
    )
    worker.chmod(0o755)
    slurm = _site_slurm(paths)
    submitted = _run(
        paths,
        [
            "sbatch", "--parsable", f"--job-name=jer-{stage.replace('_', '-')}"
            , f"--partition={slurm['gpu_partition']}", f"--gres={slurm['gpu_gres']}"
            , f"--cpus-per-task={slurm['gpu_cpus']}", f"--mem={slurm['gpu_memory']}"
            , f"--time={slurm['gpu_time']}", f"--output={output}/%x-%j.out"
            , f"--error={output}/%x-%j.err", str(worker),
        ],
        check=False,
    )
    if submitted.returncode:
        raise CampaignError(submitted.stderr.strip() or f"{stage} preflight submission failed")
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise CampaignError(f"Could not parse {stage} preflight job ID: {submitted.stdout!r}")
    release.update({
        "status": "SUBMITTED",
        "preflight_job_id": job_id,
        "preflight_output_dir": str(output),
        "artifact": str(artifact),
        "preflight_attempt": attempt,
        "preflight_submitted_at": _now(),
        "reason": "bounded real-data stage preflight submitted; controller is armed",
    })
    state["stages"][stage]["preflight_job_id"] = job_id
    state["stages"][stage]["preflight_output_dir"] = str(output)
    state["stage"] = stage
    _set_state(state, f"STUDY{_stage_number(stage)}_PREFLIGHT_RUNNING", reason=f"{stage} preflight submitted")
    _submit_controller(paths, state, stage, [job_id])
    return {"job_id": job_id, "status": "SUBMITTED", "artifact": str(artifact)}


def _controller_dependency(job_ids: list[str]) -> str | None:
    unique = list(dict.fromkeys(str(job) for job in job_ids if job))
    return ":".join(unique) if unique else None


def _submit_controller(paths: CampaignPaths, state: dict, event: str, parent_ids: list[str]) -> str:
    stage_record = state["stages"].get(event, {}) if event in state.get("stages", {}) else state["smoke"]
    attempts = int(stage_record.get("controller_attempts", 0)) + 1
    if attempts > 3:
        raise CampaignError(f"Controller retry limit exceeded for {event}")
    if event in state.get("stages", {}):
        stage_record["controller_attempts"] = attempts
    else:
        state["smoke"]["controller_attempts"] = attempts
    name = f"{event.replace('_', '-')}-a{attempts}"
    script = paths.controllers / f"{name}.sh"
    log_dir = paths.logs / "controllers"
    log_dir.mkdir(parents=True, exist_ok=True)
    source_sha = state["source_sha"]
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"export JER_PROJECT_ROOT={shlex.quote(str(paths.project_root))}\n"
        f"export JER_SOURCE_ROOT={shlex.quote(state['source_root'])}\n"
        f"export JER_SOURCE_SHA={shlex.quote(source_sha)}\n"
        f"export PYTHONPATH={shlex.quote(str(Path(state['source_root']) / 'src'))}:{shlex.quote(state['source_root'])}\n"
        "export PYTHONDONTWRITEBYTECODE=1\n"
        f"cd {shlex.quote(str(paths.project_root))}\n"
        f"exec {shlex.quote(sys.executable)} -m joint_event_reconstruction.campaign_controller "
        f"--campaign-root {shlex.quote(str(paths.root))} --event {shlex.quote(event)}\n"
    )
    script.chmod(0o755)
    slurm = _site_slurm(paths)
    command = [
        "sbatch", "--parsable", "--job-name=jer-p1-ctl",
        f"--partition={slurm['cpu_partition']}", f"--cpus-per-task={slurm['cpu_cpus']}",
        f"--mem={slurm['cpu_memory']}", f"--time={slurm['cpu_time']}",
        f"--output={log_dir}/%x-%j.out", f"--error={log_dir}/%x-%j.err",
    ]
    dependency = _controller_dependency(parent_ids)
    if dependency:
        command.append(f"--dependency=afterany:{dependency}")
    command.append(str(script))
    submitted = _run(paths, command, check=False)
    if submitted.returncode:
        raise CampaignError(submitted.stderr.strip() or f"Controller submission failed for {event}")
    job_id = submitted.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise CampaignError(f"Could not parse controller job ID: {submitted.stdout!r}")
    if event in state.get("stages", {}):
        stage_record["controller_job"] = job_id
        stage_record["controller_dependency"] = list(dict.fromkeys(parent_ids))
        stage_record["controller_script"] = str(script)
    else:
        state["smoke"]["controller_job"] = job_id
        state["smoke"]["controller_dependency"] = list(dict.fromkeys(parent_ids))
        state["smoke"]["controller_script"] = str(script)
    _write_json(
        paths.controllers / f"{name}.json",
        {
            "event": event,
            "job_id": job_id,
            "dependency": list(dict.fromkeys(parent_ids)),
            "submitted_at": _now(),
            "source_sha": source_sha,
        },
    )
    return job_id


def _ensure_controller(paths: CampaignPaths, state: dict, event: str, parent_ids: list[str]) -> None:
    record = state["stages"].get(event) if event in state.get("stages", {}) else state["smoke"]
    controller_job = record.get("controller_job") if isinstance(record, dict) else None
    if controller_job:
        scheduler_state, _ = _job_state(paths, controller_job)
        if scheduler_state in ACTIVE_SCHEDULER_STATES:
            return
        if scheduler_state == "COMPLETED":
            if event == "smoke" and state.get("stage") is not None:
                return
            if event in state.get("stages", {}) and state["stages"][event].get("state") not in {
                "NOT_STARTED", "REPAIR", "RUNNING"
            }:
                return
        # A cancelled/failed controller, or a completed controller that did not
        # make its expected transition, is safe to re-arm within the limit.
        record.pop("controller_job", None)
    if parent_ids:
        _submit_controller(paths, state, event, parent_ids)


def _existing_submission(paths: CampaignPaths, study_name: str, identity: str) -> tuple[Path, dict] | None:
    root = paths.project_root / "results/studies" / study_name / "_slurm"
    for path in sorted(root.glob("*/submission.json")):
        try:
            record = _read_json(path)
        except CampaignError:
            continue
        if record.get("submission_identity") == identity:
            return path, record
    return None


def _submit_study(paths: CampaignPaths, state: dict, stage: str, *, postprocess_only: bool = False) -> tuple[Path, dict]:
    if stage not in STAGE_STUDIES:
        raise CampaignError(f"No Slurm study is defined for stage {stage!r}")
    stage_record = state["stages"][stage]
    study_name = STAGE_STUDIES[stage]
    source_root = Path(state["source_root"]).resolve()
    maintained = {
        load_study(path).name: path
        for path in sorted((source_root / "configs/studies").glob("[0-9][0-9]_*.yaml"))
    }
    if study_name not in maintained:
        raise CampaignError(f"Missing immutable study manifest for {study_name}: {source_root}")
    manifest = maintained[study_name]
    gates_path = paths.gates
    manifest_sha = _sha256_file(manifest)
    identity_payload = {
        "campaign": state["campaign_id"],
        "source_sha": state["source_sha"],
        "stage": stage,
        "study": study_name,
        "manifest_sha256": manifest_sha,
        "gates_sha256": _sha256_file(gates_path),
        "mode": "postprocess_only" if postprocess_only else "normal",
    }
    identity = _json_digest(identity_payload)
    existing = _existing_submission(paths, study_name, identity)
    if existing is not None:
        return existing
    if postprocess_only:
        attempt_name = f"paper1-{stage}-repair{stage_record['repair_attempts']}"
    else:
        attempt_name = f"paper1-{stage}-a{stage_record['submission_attempts'] + 1}"
    command = [
        sys.executable,
        str(paths.source_root / "scripts/submit_study_slurm.py"),
        str(manifest),
        "--output-root", str(paths.project_root / "results/studies"),
        "--gates-path", str(gates_path),
        "--campaign-id", attempt_name,
        "--submission-identity", identity,
    ]
    if postprocess_only:
        command.append("--postprocess-only")
    result = _run(paths, command, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise CampaignError(f"Study {study_name} submission failed: {detail}")
    record_path = paths.project_root / "results/studies" / study_name / "_slurm" / attempt_name / "submission.json"
    if not record_path.is_file():
        found = _existing_submission(paths, study_name, identity)
        if found is None:
            raise CampaignError(f"Study submission produced no record: {record_path}")
        return found
    return record_path, _read_json(record_path)


def _stage_output_complete(paths: CampaignPaths, stage_record: dict, study_name: str) -> tuple[bool, str]:
    record_path = stage_record.get("submission_record")
    if not record_path:
        return False, "no study submission record"
    try:
        record = _read_json(Path(record_path))
        study = resolve_study(record["study_manifest_snapshot"], record.get("gates_path") or None)
        from scripts.submit_campaign import _outputs_complete

        return _outputs_complete(study, record)
    except (CampaignError, FileNotFoundError, ScientificGatePending, ScientificGateNotApplicable, ValueError) as error:
        return False, str(error)


def _stage_job_status(paths: CampaignPaths, stage_record: dict) -> tuple[str, str]:
    record_path = stage_record.get("repair_submission_record") or stage_record.get("submission_record")
    if not record_path:
        return "NONE", "no submission"
    try:
        record = _read_json(Path(record_path))
    except CampaignError as error:
        return "FAILED", str(error)
    groups = _study_job_groups(record)
    training_state, training_reason = _job_group(paths, groups["training"])
    prediction_state, prediction_reason = _job_group(paths, groups["prediction"])
    post_state, post_reason = _job_group(paths, groups["postprocess"])
    if training_state == "FAILED":
        return "FAILED", f"training: {training_reason}"
    if prediction_state == "FAILED":
        return "FAILED", f"prediction: {prediction_reason}"
    if post_state == "FAILED":
        return "REPAIRABLE", f"postprocessing: {post_reason}"
    if "RUNNING" in {training_state, prediction_state, post_state}:
        return "RUNNING", f"training={training_reason}; prediction={prediction_reason}; post={post_reason}"
    if "PENDING" in {training_state, prediction_state, post_state}:
        return "PENDING", f"training={training_reason}; prediction={prediction_reason}; post={post_reason}"
    if all(state in {"COMPLETE", "NONE"} for state in (training_state, prediction_state, post_state)):
        return "COMPLETE", "all submitted jobs completed"
    return "UNKNOWN", f"training={training_state}; prediction={prediction_state}; post={post_state}"


def _metric(payload: dict, dotted: str) -> float | None:
    value: object = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _bootstrap_interval(values: list[float], seed: int, samples: int) -> tuple[float, float] | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return means[int(0.025 * (len(means) - 1))], means[int(0.975 * (len(means) - 1))]


def _paired_evidence(payloads: dict[str, list[dict]], variant_a: str, variant_b: str, metric_path: str, seed: int, samples: int) -> dict:
    """Use event-level pairs when a maintained evaluation product exposes them."""
    arrays_a: list[float] = []
    arrays_b: list[float] = []
    for payload_a, payload_b in zip(payloads.get(variant_a, []), payloads.get(variant_b, [])):
        events_a = payload_a.get("event_metrics")
        events_b = payload_b.get("event_metrics")
        if not isinstance(events_a, list) or not isinstance(events_b, list) or len(events_a) != len(events_b):
            continue
        for item_a, item_b in zip(events_a, events_b):
            if isinstance(item_a, dict) and isinstance(item_b, dict):
                value_a = _metric(item_a, metric_path)
                value_b = _metric(item_b, metric_path)
                if value_a is not None and value_b is not None:
                    arrays_a.append(value_a)
                    arrays_b.append(value_b)
    if not arrays_a:
        return {"available": False, "reason": "evaluation JSONs contain no event-level paired metric arrays"}
    differences = [a - b for a, b in zip(arrays_a, arrays_b)]
    return {
        "available": True,
        "events": len(differences),
        "difference_mean": sum(differences) / len(differences),
        "bootstrap_95": _bootstrap_interval(differences, seed, samples),
    }


def _evaluation_payloads(paths: CampaignPaths, stage_record: dict, study_name: str, variants: list[str]) -> dict[str, list[dict]]:
    record = _read_json(Path(stage_record["submission_record"]))
    root = Path(record["output_root"]) / study_name
    study = resolve_study(record["study_manifest_snapshot"], record.get("gates_path") or None)
    payloads: dict[str, list[dict]] = {variant: [] for variant in variants}
    for topology in study.topologies:
        for variant in variants:
            spec = study.variant_map[variant]
            for seed in study.seeds:
                for prediction in spec.predictions:
                    path = root / topology / variant / f"seed_{seed}" / "evaluation" / f"{prediction.name}.json"
                    if not path.is_file():
                        raise CampaignError(f"Missing evaluation product for decision: {path}")
                    payloads[variant].append(_read_json(path))
    return payloads


def _pair_decision(
    *,
    paths: CampaignPaths,
    state: dict,
    stage: str,
    gate: str,
    baseline: str,
    alternative: str,
    policy: dict,
    diagnostic: list[str] | None = None,
) -> dict:
    stage_record = state["stages"][stage]
    variants = [baseline, alternative, *(diagnostic or [])]
    payloads = _evaluation_payloads(paths, stage_record, STAGE_STUDIES[stage], variants)
    metric_path = policy["primary_metric"]["path"]
    values: dict[str, list[float]] = {}
    missing: list[str] = []
    study = resolve_study(
        _read_json(Path(stage_record["submission_record"]))["study_manifest_snapshot"],
        _read_json(Path(stage_record["submission_record"])).get("gates_path") or None,
    )
    for variant in (baseline, alternative):
        values[variant] = []
        for payload in payloads[variant]:
            value = _metric(payload, metric_path)
            if value is None:
                missing.append(f"{variant}:{metric_path}")
            else:
                values[variant].append(value)
    if missing or any(len(values[name]) != len(study.topologies) * len(study.seeds) for name in (baseline, alternative)):
        return _decision(
            state, gate, stage, "FAILED_VALIDATION", None,
            {"missing": sorted(set(missing)), "metric_path": metric_path},
            "Primary decision metric is missing from one or more validated evaluations.",
        )
    seed_count = len(study.seeds)
    baseline_by_seed = [values[baseline][index] for index in range(0, len(values[baseline]), len(study.topologies))]
    alternative_by_seed = [values[alternative][index] for index in range(0, len(values[alternative]), len(study.topologies))]
    differences = [base - alt for base, alt in zip(baseline_by_seed, alternative_by_seed)]
    base_mean = sum(baseline_by_seed) / len(baseline_by_seed)
    alt_mean = sum(alternative_by_seed) / len(alternative_by_seed)
    relative_gain = (base_mean - alt_mean) / max(abs(base_mean), 1.0)
    margin = float(policy.get("equivalence_relative", 0.0))
    all_direction = all(value > 0 for value in differences) if policy.get("require_all_seed_direction", True) else True
    any_direction = all(value < 0 for value in differences)
    preferred_simple = str(policy.get("preferred_simple", baseline))
    guardrail = _coverage_guardrail(payloads, baseline, alternative, policy.get("coverage_guardrail", {}))
    cost_guardrail = _cost_guardrail(payloads, baseline, alternative, policy)
    if (not guardrail["pass"] or not cost_guardrail["pass"]) and relative_gain > margin:
        selected = None
        status = "REVIEW_REQUIRED"
        reason = "Alternative improves the primary metric but violates a configured calibration or cost guardrail."
    elif relative_gain > margin and all_direction:
        selected = alternative
        status = "AUTO_SELECTED"
        reason = "Alternative has a clear primary-metric gain with consistent seed direction and passing guardrails."
    elif relative_gain < -margin and any_direction:
        selected = baseline
        status = "AUTO_SELECTED"
        reason = "Preferred simple baseline has a clear primary-metric advantage."
    elif abs(relative_gain) <= margin:
        selected = preferred_simple
        status = "AUTO_SELECTED"
        reason = "Variants are within the predeclared equivalence margin; simplicity preference applies."
    else:
        selected = None
        status = "REVIEW_REQUIRED"
        reason = "Seed directions and primary metric do not satisfy an unambiguous promotion rule."
    paired = _paired_evidence(
        payloads, baseline, alternative, metric_path,
        int(_load_policy(paths)["campaign"]["bootstrap_seed"]),
        int(_load_policy(paths)["campaign"]["bootstrap_samples"]),
    )
    evidence = {
        "metric_path": metric_path,
        "values_by_seed": {baseline: baseline_by_seed, alternative: alternative_by_seed},
        "mean": {baseline: base_mean, alternative: alt_mean},
        "relative_gain_for_alternative": relative_gain,
        "seed_differences_baseline_minus_alternative": differences,
        "all_seed_direction": all_direction,
        "equivalence_relative": margin,
        "coverage_guardrail": guardrail,
        "cost_guardrail": cost_guardrail,
        "paired_comparison": paired,
        "diagnostic_variants": diagnostic or [],
    }
    return _decision(state, gate, stage, status, selected, evidence, reason)


def _cost_guardrail(payloads: dict[str, list[dict]], baseline: str, alternative: str, policy: dict) -> dict:
    maximum = policy.get("maximum_runtime_ratio")
    if maximum is None:
        return {"pass": True, "available": False, "reason": "no runtime guardrail configured"}
    def runtime(payload: dict) -> float | None:
        for key in ("runtime_seconds", "inference_runtime_seconds", "elapsed_seconds"):
            value = _metric(payload, key)
            if value is not None:
                return value
        return None
    base = [runtime(item) for item in payloads.get(baseline, [])]
    alt = [runtime(item) for item in payloads.get(alternative, [])]
    if not base or not alt or any(value is None for value in (*base, *alt)):
        return {"pass": True, "available": False, "reason": "evaluation products contain no runtime field"}
    ratios = [float(right) / max(float(left), 1.0e-12) for left, right in zip(base, alt)]
    return {"pass": all(ratio <= float(maximum) for ratio in ratios), "available": True, "maximum_ratio": float(maximum), "ratios": ratios}


def _coverage_guardrail(payloads: dict[str, list[dict]], baseline: str, alternative: str, config: dict) -> dict:
    paths = list(config.get("paths", [])) if isinstance(config, dict) else []
    if not paths:
        return {"pass": True, "available": False, "reason": "no coverage guardrail configured"}
    tolerance = float(config.get("maximum_absolute_regression", 0.0))
    value_key = str(config.get("value_key", "coverage_68"))

    def values_at(payload: dict, path: str) -> list[float]:
        value: object = payload
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return []
            value = value[part]
        found: list[float] = []

        def visit(node: object) -> None:
            if isinstance(node, dict):
                if value_key in node:
                    try:
                        number = float(node[value_key])
                    except (TypeError, ValueError):
                        number = float("nan")
                    if number == number and abs(number) != float("inf"):
                        found.append(number)
                for child in node.values():
                    visit(child)
            elif isinstance(node, list):
                for child in node:
                    visit(child)

        visit(value)
        return found

    regressions = []
    for path in paths:
        base = [sum(values) / len(values) for item in payloads.get(baseline, []) if (values := values_at(item, path))]
        alt = [sum(values) / len(values) for item in payloads.get(alternative, []) if (values := values_at(item, path))]
        if not base or not alt:
            continue
        for index, (base_value, alt_value) in enumerate(zip(base, alt)):
            if float(alt_value) < float(base_value) - tolerance:
                regressions.append({"path": path, "index": index, "baseline": base_value, "alternative": alt_value})
    return {"pass": not regressions, "available": True, "maximum_absolute_regression": tolerance, "regressions": regressions}


def _decision(state: dict, gate: str, stage: str, status: str, selected: str | None, evidence: dict, reason: str) -> dict:
    artifact = {
        "schema": "jer_paper1_decision_v1",
        "gate": gate,
        "source_study": STAGE_STUDIES.get(stage, stage),
        "source_sha": state["source_sha"],
        "decision": selected or "review_required",
        "status": status,
        "policy_sha256": state["policy_sha256"],
        "metrics": evidence.get("mean", evidence),
        "evidence": evidence,
        "reason": reason,
        "artifacts": {},
        "created_at": _now(),
    }
    return artifact


def _read_identity_artifact(paths: CampaignPaths, checkpoint: Path, identity_file: Path, *, expected_study: str, expected_variant: str, seed: int) -> dict:
    if not checkpoint.is_file() or not identity_file.is_file():
        raise CampaignError(f"Missing selected checkpoint identity: {checkpoint} / {identity_file}")
    identity_payload = _read_json(identity_file)
    identity = identity_payload.get("identity", {})
    artifact = identity_payload.get("artifact", {})
    if identity.get("study") not in {expected_study, None}:
        raise CampaignError(f"Checkpoint identity study mismatch: expected {expected_study}, found {identity.get('study')}")
    if identity.get("variant") not in {expected_variant, None}:
        raise CampaignError(f"Checkpoint identity variant mismatch: expected {expected_variant}, found {identity.get('variant')}")
    if int(identity.get("model_seed", seed)) != int(seed):
        raise CampaignError(f"Checkpoint identity seed mismatch: {identity_file}")
    if not identity.get("git_commit") or not identity.get("architecture_signature"):
        raise CampaignError(f"Checkpoint identity lacks training provenance: {identity_file}")
    checkpoint_sha = _sha256_file(checkpoint)
    identity_sha = _sha256_file(identity_file)
    if artifact.get("sha256") and artifact["sha256"] != checkpoint_sha:
        raise CampaignError(f"Checkpoint identity checksum disagrees with file: {checkpoint}")
    return {
        "source_study": str(identity.get("study", expected_study)),
        "variant": str(identity.get("variant", expected_variant)),
        "model_seed": int(identity.get("model_seed", seed)),
        "data_subset_seed": int(identity.get("data_subset", {}).get("subset_seed", 42)),
        "git_sha": str(identity.get("git_commit", "")),
        "architecture_identity": str(identity.get("architecture_signature", "")),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "identity_file": str(identity_file),
        "identity_sha256": identity_sha,
    }


def _artifact_from_variant(paths: CampaignPaths, state: dict, stage: str, selected: str, topology: str, seed: int) -> dict:
    study_name = STAGE_STUDIES[stage]
    record = _read_json(Path(state["stages"][stage]["submission_record"]))
    study = resolve_study(record["study_manifest_snapshot"], record.get("gates_path") or None)
    variant = study.variant_map[selected]
    if variant.checkpoint_from_gate:
        gate = load_scientific_gates(paths.gates)[variant.checkpoint_from_gate]
        base = deepcopy(gate["artifacts"][topology][f"seed_{seed}"])
        training = {
            "source_study": base["source_study"],
            "variant": base["variant"],
            "git_sha": base["git_sha"],
            "architecture_identity": base["architecture_identity"],
            "identity_sha256": base["identity_sha256"],
        }
        base["training_provenance"] = training
        return base
    root = Path(record["output_root"]) / study_name / topology / selected / f"seed_{seed}" / "train"
    checkpoint = root / "checkpoints/best-loss.ckpt"
    identity_file = root / "artifact_identity.json"
    entry = _read_identity_artifact(
        paths, checkpoint, identity_file, expected_study=study_name, expected_variant=selected, seed=seed
    )
    entry["training_provenance"] = {
        "source_study": entry["source_study"],
        "variant": entry["variant"],
        "git_sha": entry["git_sha"],
        "architecture_identity": entry["architecture_identity"],
        "identity_sha256": entry["identity_sha256"],
    }
    return entry


def _materialise_gate(paths: CampaignPaths, state: dict, stage: str, selected: str, decision: dict) -> None:
    gate_name = STAGE_GATES[stage]
    gates = load_scientific_gates(paths.gates)
    gate = deepcopy(gates[gate_name])
    if gate["status"] == "locked":
        if stage == "assignment_coupling" and gate.get("selected_decision") == selected:
            return
        if gate.get("selected_variant") != selected:
            raise CampaignError(f"Refusing to overwrite locked gate {gate_name}")
        return
    if gate_name not in state["allowed_pending_gates"]:
        raise CampaignError(f"Campaign may not materialise gate that was not pending at launch: {gate_name}")
    source_study = STAGE_STUDIES[stage]
    source_path = next(path for path in _study_paths(paths).values() if load_study(path).name == source_study)
    source = load_study(source_path)
    if stage != "assignment_coupling" and selected not in source.variant_map:
        raise CampaignError(f"Decision selected unknown {source_study} variant {selected}")
    gate["status"] = "locked"
    gate["selected_variant"] = "compatibility" if stage == "assignment_coupling" else selected
    if stage == "assignment_coupling":
        gate["selected_decision"] = selected
    gate["validated_sha"] = state["source_sha"]
    gate["selected_patch"] = (
        {}
        if stage == "assignment_coupling"
        else deepcopy(source.variant_map[selected].patch)
    )
    gate["artifacts"] = {}
    for topology in (() if stage == "assignment_coupling" else source.topologies):
        gate["artifacts"][topology] = {}
        for seed in source.seeds:
            artifact = _artifact_from_variant(paths, state, stage, selected, topology, seed)
            artifact["selection_provenance"] = {
                "source_study": source_study,
                "variant": selected,
                "decision_artifact": str(paths.decisions / f"{gate_name}.json"),
                "decision_sha256": _sha256_file(paths.decisions / f"{gate_name}.json"),
            }
            gate["artifacts"][topology][f"seed_{seed}"] = artifact
    gates[gate_name] = gate
    _write_json(paths.gates, gates)
    # Validate the complete overlay immediately; submission never consumes an unvalidated gate.
    load_scientific_gates(paths.gates)
    state["decisions"][gate_name] = {
        "decision_artifact": str(paths.decisions / f"{gate_name}.json"),
        "selected_variant": selected,
        "status": decision["status"],
        "materialised": True,
    }


def _write_decision(paths: CampaignPaths, state: dict, gate: str, decision: dict) -> None:
    path = paths.decisions / f"{gate}.json"
    _write_json(path, decision)
    decision["artifacts"] = {"decision": str(path), "decision_sha256": _sha256_file(path)}
    _write_json(path, decision)


def _policy_decision(paths: CampaignPaths, state: dict, stage: str) -> dict:
    policy = _load_policy(paths)["decisions"]
    if stage == "invisible_coordinates":
        return _pair_decision(
            paths=paths, state=state, stage=stage, gate="invisible_coordinates",
            baseline="cartesian", alternative="met_residual",
            policy=policy["invisible_coordinates"], diagnostic=["met_exact"],
        )
    if stage == "invisible_backbone":
        return _pair_decision(
            paths=paths, state=state, stage=stage, gate="invisible_backbone",
            baseline="conventional", alternative="conditional_lgatr",
            policy=policy["invisible_backbone"],
        )
    if stage == "assignment_conditioning":
        return _pair_decision(
            paths=paths, state=state, stage=stage, gate="assignment_conditioning",
            baseline="none", alternative="refined",
            policy=policy["assignment_conditioning"], diagnostic=["truth_oracle"],
        )
    if stage != "assignment_coupling":
        raise CampaignError(f"No decision policy for stage {stage}")
    stage_record = state["stages"][stage]
    record = _read_json(Path(stage_record["submission_record"]))
    review = Path(record["output_root"]) / "compatibility_reranking/study10_human_review.json"
    if not review.is_file():
        raise CampaignError(f"Study-10 review product is missing: {review}")
    payload = _read_json(review)
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise CampaignError("Study-10 review contains no per-seed records")
    config = policy["assignment_coupling"]
    methods = {}
    for method, key in {
        "visible": "visible_baseline",
        "visible_only": "matched_visible_only_control",
        "analytic": "analytic_physics",
        "learned": "learned_physics",
        "truth_nu_oracle": "truth_neutrino_oracle",
    }.items():
        methods[method] = [float(row["comparisons"][key]["accuracy"]) for row in rows]
    means = {name: sum(values) / len(values) for name, values in methods.items()}
    gains = {name: means[name] - means["visible"] for name in methods if name != "visible"}
    transitions = {}
    for method in ("analytic", "learned"):
        method_key = {"analytic": "analytic_physics", "learned": "learned_physics"}[method]
        per_seed = []
        for row in rows:
            counts = row["comparisons"][method_key]["transitions"]
            total = sum(counts.values())
            per_seed.append(
                {
                    "counts": counts,
                    "correct_to_wrong_fraction": counts["baseline_correct_to_broken"] / max(total, 1),
                }
            )
        transitions[method] = {"per_seed": per_seed}
    oracle_gain = gains["truth_nu_oracle"]
    minimum_gain = float(config.get("minimum_absolute_gain", 0.0))
    max_break = float(config.get("maximum_correct_to_wrong_fraction", 1.0))
    def method_Rcoupling(method: str, index: int) -> float | None:
        if method == "learned":
            return rows[index]["comparisons"].get("coupling_efficiency", {}).get("Rcoupling")
        delta_oracle = methods["truth_nu_oracle"][index] - methods["visible"][index]
        if delta_oracle <= 0:
            return None
        return (methods[method][index] - methods["visible"][index]) / delta_oracle

    minimum_r = float(config.get("minimum_Rcoupling", 0.0))
    method_ok = {
        method: all(
            gain >= minimum_gain
            and transitions[method]["per_seed"][index]["correct_to_wrong_fraction"] <= max_break
            and (method_Rcoupling(method, index) is not None)
            and float(method_Rcoupling(method, index)) >= minimum_r
            for index, gain in enumerate(
                [methods[method][item] - methods["visible"][item] for item in range(len(rows))]
            )
        )
        for method in ("analytic", "learned")
    }
    selected = "none"
    status = "AUTO_SELECTED"
    reason = "Neither deployable coupling method clears the predeclared gain and transition criteria."
    if oracle_gain > 0 and method_ok["analytic"]:
        selected = "analytic"
        reason = "Analytic compatibility gives a seed-consistent gain with acceptable correct-to-wrong transitions."
    if oracle_gain > 0 and method_ok["learned"] and gains["learned"] >= gains["analytic"] + minimum_gain:
        selected = "learned"
        reason = "Learned physics gives a robust gain beyond both the visible baseline and analytic coupling."
    if oracle_gain <= 0:
        reason = "Truth-neutrino oracle supplies no positive assignment headroom; coupling is not released."
    # A visible-only gain of the same size is not evidence for invisible physics.
    if selected != "none" and gains["visible_only"] >= gains[selected] - minimum_gain / 2:
        selected = None
        status = "REVIEW_REQUIRED"
        reason = "The matched visible-only control explains the apparent coupling gain."
    elif selected != "none" and not all(
        methods[selected][index] >= methods["visible"][index]
        for index in range(len(rows))
    ):
        selected = None
        status = "REVIEW_REQUIRED"
        reason = "Deployable coupling is not directionally consistent across seeds."
    r_values = {method: [method_Rcoupling(method, index) for index in range(len(rows))] for method in ("analytic", "learned")}
    if selected in {"analytic", "learned"}:
        if not all(value is not None and float(value) >= minimum_r for value in r_values[selected]):
            selected = None
            status = "REVIEW_REQUIRED"
            reason = "Coupling efficiency does not satisfy the per-seed Rcoupling threshold."
    convergence = _study10_convergence(rows, float(config.get("convergence_tolerance", 0.005)))
    if payload.get("test_split_used") is True or convergence.get("status") == "FAILED_VALIDATION":
        status = "FAILED_VALIDATION"
        selected = None
        reason = "Study-10 validation failed its sealed-test or nested K/M convergence contract."
    evidence = {
        "methods_by_seed": methods,
        "mean_accuracy": means,
        "gain_vs_visible": gains,
        "oracle_headroom": oracle_gain,
        "transitions": transitions,
        "method_pass": method_ok,
        "visible_only_control": {"gain": gains["visible_only"], "physics_blind": True},
        "Rcoupling": r_values,
        "minimum_Rcoupling": minimum_r,
        "nested_K_M_convergence": convergence,
        "top_k_coverage": [row["comparisons"].get("top_k_ceiling", {}) for row in rows],
        "test_split_used": payload.get("test_split_used", False),
        "review_artifact": str(review),
    }
    return _decision(state, "assignment_coupling", stage, status, selected, evidence, reason)


def _study10_convergence(rows: list[dict], tolerance: float) -> dict:
    available = set()
    for row in rows:
        available.update(row["comparisons"].get("nested_K_M_convergence", {}))
    if not available:
        return {"status": "FAILED_VALIDATION", "reason": "nested_K_M_convergence is absent"}
    m_selected = 64
    for m in (4, 8, 16, 32, 64):
        compare = []
        for row in rows:
            values = row["comparisons"].get("nested_K_M_convergence", {})
            a = values.get(f"K10_M{m}", {}).get("assignment_accuracy")
            b = values.get("K10_M64", {}).get("assignment_accuracy")
            if a is None or b is None:
                compare = []
                break
            compare.append(abs(float(a) - float(b)))
        if compare and max(compare) <= tolerance:
            m_selected = m
            break
    k_selected = 10
    for k in (1, 2, 3, 5, 10):
        compare = []
        for row in rows:
            values = row["comparisons"].get("nested_K_M_convergence", {})
            a = values.get(f"K{k}_M{m_selected}", {}).get("assignment_accuracy")
            b = values.get(f"K10_M{m_selected}", {}).get("assignment_accuracy")
            if a is None or b is None:
                compare = []
                break
            compare.append(abs(float(a) - float(b)))
        if compare and max(compare) <= tolerance:
            k_selected = k
            break
    return {"selected_K": k_selected, "selected_M": m_selected, "tolerance": tolerance, "Kmax": 10, "Mmax": 64}


def _validate_overlay_immutability(paths: CampaignPaths, state: dict) -> None:
    base_path = paths.source_root / "configs/studies/UPSTREAM_SELECTIONS.yaml"
    base = load_scientific_gates(base_path)
    overlay = load_scientific_gates(paths.gates)
    for name, gate in base.items():
        if gate["status"] == "locked" and overlay.get(name) != gate:
            raise CampaignError(f"Campaign overlay changed locked base gate {name!r}")
    for name, gate in overlay.items():
        if name not in base:
            raise CampaignError(f"Campaign overlay introduced unknown gate {name!r}")
        if base[name]["status"] != "pending" and gate != base[name]:
            raise CampaignError(f"Campaign overlay changed non-pending base gate {name!r}")


def _stage_release_ready(paths: CampaignPaths, state: dict, stage: str) -> tuple[bool, str]:
    """Check an execution-backed downstream release before full submission."""
    release = state.get("stage_releases", {}).get(stage)
    if not isinstance(release, dict):
        return False, f"no release contract is registered for {stage}"
    artifact = Path(str(release.get("artifact", "")))
    if not artifact.is_file():
        if release.get("preflight_job_id"):
            scheduler, detail = _scheduler_terminal(paths, release["preflight_job_id"])
            if scheduler in ACTIVE_SCHEDULER_STATES:
                release["reason"] = f"stage preflight {release['preflight_job_id']} is {scheduler}: {detail}"
                return False, release["reason"]
            if scheduler != "COMPLETED":
                release["status"] = "FAILED"
                release["reason"] = f"stage preflight {release['preflight_job_id']} failed: {detail}"
                return False, release["reason"]
        release["reason"] = f"{release.get('contract')} is not materialised; run the named production-equivalent preflight"
        return False, release["reason"]
    try:
        payload = _read_json(artifact)
    except CampaignError as error:
        release["status"] = "FAILED"
        release["reason"] = str(error)
        return False, release["reason"]
    if release.get("preflight_job_id") and str(payload.get("slurm_job_id")) != str(release["preflight_job_id"]):
        release["status"] = "FAILED"
        release["reason"] = "release artifact Slurm job does not match the submitted preflight"
        return False, release["reason"]
    preflight_job = release.get("preflight_job_id")
    if preflight_job is not None:
        scheduler, detail = _scheduler_terminal(paths, preflight_job)
        release["scheduler_state"] = scheduler
        release["scheduler_exit_code"] = detail if scheduler == "COMPLETED" else None
        if scheduler != "COMPLETED":
            if scheduler in ACTIVE_SCHEDULER_STATES:
                release["reason"] = f"stage preflight {preflight_job} is {scheduler}: {detail}"
            else:
                release["status"] = "FAILED"
                release["reason"] = f"stage preflight {preflight_job} did not complete successfully: {detail}"
            return False, release["reason"]
    try:
        source_root = Path(state["source_root"])
        manifest = next(
            path
            for path in sorted((source_root / "configs/studies").glob("[0-9][0-9]_*.yaml"))
            if load_study(path).name == STAGE_STUDIES[stage]
        )
        study = resolve_study(manifest, paths.gates)
        gates = load_scientific_gates(paths.gates)
        expected_checkpoints = gate_checkpoint_hashes(gates, study.requires, seeds=(42,))
        validation_root = paths.project_root / "results/validation/stage-release" / state["source_sha"]
        validate_release_payload(
            payload,
            artifact_path=artifact,
            expected_schema=str(release.get("contract")),
            expected_source_sha=str(state["source_sha"]),
            expected_campaign_id=str(state["campaign_id"]),
            expected_gate_digest=str(study.gates_digest),
            expected_manifest_sha256=sha256_file(manifest),
            expected_checkpoint_hashes=expected_checkpoints,
            validation_root=validation_root,
        )
        if preflight_job is None:
            scheduler, detail = _scheduler_terminal(paths, payload.get("slurm_job_id"))
            if scheduler != "COMPLETED":
                raise StageReleaseError(f"release Slurm job is not complete: {scheduler} {detail}")
    except (CampaignError, StageReleaseError, FileNotFoundError, ScientificGatePending, ScientificGateNotApplicable, ValueError, StopIteration) as error:
        release["status"] = "FAILED"
        release["reason"] = str(error)
        return False, release["reason"]
    release["status"] = "PASS"
    release["reason"] = "execution-backed production-equivalent preflight passed"
    release["validated_at"] = _now()
    return True, release["reason"]


def _set_stage_preflight_required(
    paths: CampaignPaths, state: dict, stage: str, reason: str
) -> None:
    state["stage"] = stage
    state.setdefault("stage_releases", {}).setdefault(stage, {})["reason"] = reason
    _set_state(
        state,
        f"STUDY{_stage_number(stage)}_PREFLIGHT_REQUIRED",
        reason=reason,
    )


def _advance_after_gate(paths: CampaignPaths, state: dict, stage: str, decision: dict) -> None:
    if decision["status"] == "FAILED_VALIDATION":
        _set_state(state, "BLOCKED", reason=decision["reason"])
        state["pending_review"] = {"gate": STAGE_GATES.get(stage), "reason": decision["reason"]}
        return
    if decision["status"] == "REVIEW_REQUIRED" or not decision.get("decision") or decision["decision"] == "review_required":
        state["pending_review"] = {
            "gate": STAGE_GATES.get(stage),
            "stage": stage,
            "decision_artifact": str(paths.decisions / f"{STAGE_GATES.get(stage)}.json"),
            "reason": decision["reason"],
        }
        _set_state(state, f"STUDY{_stage_number(stage)}_REVIEW", reason=decision["reason"])
        return
    selected = decision["decision"]
    _materialise_gate(paths, state, stage, selected, decision)
    state["stages"][stage]["state"] = "COMPLETE"
    state["stages"][stage]["selected_variant"] = selected
    state["pending_review"] = None
    if stage == "assignment_coupling" and selected == "none":
        state["stages"]["assignment_conditioning"]["state"] = "NOT_RELEASED"
        _set_state(
            state,
            "READY_FOR_PRODUCTION_REVIEW",
            reason="Study 10 found no deployable invisible-to-assignment coupling; Study 03 was not released.",
        )
        state["stage"] = "final_production"
        return
    next_stage = {
        "invisible_coordinates": "invisible_backbone",
        "invisible_backbone": "assignment_coupling",
        "assignment_coupling": "assignment_conditioning",
        "assignment_conditioning": "final_production",
    }.get(stage)
    if next_stage == "final_production":
        _set_state(
            state,
            "READY_FOR_PRODUCTION_REVIEW",
            reason="Architecture gates are resolved; production manifest is not enabled for automatic submission.",
        )
        state["stage"] = next_stage
        return
    if next_stage:
        ready, reason = _stage_release_ready(paths, state, next_stage)
        if not ready:
            _set_stage_preflight_required(paths, state, next_stage, reason)
            if state.get("mode") == "AUTO":
                try:
                    _submit_stage_preflight(paths, state, next_stage)
                except CampaignError as error:
                    state["stage_releases"][next_stage]["reason"] = str(error)
            return
        _submit_stage_and_controller(paths, state, next_stage)


def _stage_number(stage: str) -> str:
    return {"invisible_coordinates": "13", "invisible_backbone": "15", "assignment_coupling": "10", "assignment_conditioning": "03"}.get(stage, "")


def _submit_stage_and_controller(paths: CampaignPaths, state: dict, stage: str) -> None:
    if stage == "final_production":
        _set_state(state, "READY_FOR_PRODUCTION_REVIEW", reason="Production stage requires explicit manifest review")
        state["stage"] = stage
        return
    if stage != "invisible_coordinates":
        ready, reason = _stage_release_ready(paths, state, stage)
        if not ready:
            _set_stage_preflight_required(paths, state, stage, reason)
            return
    stage_record = state["stages"][stage]
    if stage_record.get("submission_record"):
        return
    stage_record["submission_attempts"] = int(stage_record.get("submission_attempts", 0)) + 1
    path, record = _submit_study(paths, state, stage)
    stage_record["submission_record"] = str(path)
    stage_record["state"] = "RUNNING"
    stage_record["job_ids"] = _study_job_ids(record)
    stage_record["study_manifest_sha256"] = record.get("study_manifest_sha256")
    stage_record["resolved_gate_digest"] = record.get("resolved_gate_digest")
    state["stage"] = stage
    _set_state(state, f"STUDY{_stage_number(stage)}_RUNNING", reason=f"{STAGE_STUDIES[stage]} submitted")
    if stage_record["job_ids"]:
        _submit_controller(paths, state, stage, stage_record["job_ids"])
    else:
        valid, reason = _stage_output_complete(paths, stage_record, STAGE_STUDIES[stage])
        if not valid:
            _set_state(state, "BLOCKED", reason=f"Study {STAGE_STUDIES[stage]} produced no jobs and is incomplete: {reason}")


def _reconcile_stage(paths: CampaignPaths, state: dict, stage: str) -> None:
    stage_record = state["stages"][stage]
    if stage_record.get("state") in {"COMPLETE", "NOT_RELEASED"}:
        return
    valid, valid_reason = _stage_output_complete(paths, stage_record, STAGE_STUDIES[stage])
    job_state, job_reason = _stage_job_status(paths, stage_record)
    if valid:
        stage_record["state"] = "COMPLETE"
        stage_record["completion_reason"] = valid_reason
        return
    if job_state in {"PENDING", "RUNNING", "UNKNOWN"}:
        _set_state(state, f"STUDY{_stage_number(stage)}_RUNNING", reason=job_reason)
        return
    if job_state == "REPAIRABLE":
        if int(stage_record.get("repair_attempts", 0)) < int(_load_policy(paths)["campaign"].get("repair_limit", 1)):
            stage_record["repair_attempts"] = int(stage_record.get("repair_attempts", 0)) + 1
            path, record = _submit_study(paths, state, stage, postprocess_only=True)
            stage_record["repair_submission_record"] = str(path)
            stage_record["repair_job_ids"] = _study_job_ids(record)
            stage_record["state"] = "REPAIR"
            _set_state(state, f"STUDY{_stage_number(stage)}_REPAIR", reason=f"repairing postprocessing: {job_reason}")
            if stage_record["repair_job_ids"]:
                _submit_controller(paths, state, stage, stage_record["repair_job_ids"])
            return
        _set_state(state, "BLOCKED", reason=f"Study {stage} exceeded automatic postprocessing repair limit: {valid_reason}")
        return
    if job_state == "FAILED":
        _set_state(state, "BLOCKED", reason=f"Study {STAGE_STUDIES[stage]} failed; training/prediction failures are not auto-retried: {job_reason}")
        return
    _set_state(state, "BLOCKED", reason=f"Study {STAGE_STUDIES[stage]} cannot be classified safely: {valid_reason}; {job_reason}")


def _advance_locked(paths: CampaignPaths, state: dict, *, event: str | None = None) -> None:
    _validate_overlay_immutability(paths, state)
    smoke = _reconcile_smoke(paths, state["source_sha"])
    if smoke is not None:
        state["smoke"].update({key: smoke.get(key) for key in ("job_id", "status", "artifact_path", "scheduler_state", "scheduler_reason") if key in smoke})
    if state["smoke"].get("status") in {"PASS", "INHERITED"}:
        if state.get("stage") is None:
            _submit_stage_and_controller(paths, state, "invisible_coordinates")
        elif state.get("stage") in STAGE_STUDIES:
            stage = state["stage"]
            if not state["stages"][stage].get("submission_record"):
                if stage == "invisible_coordinates":
                    _submit_stage_and_controller(paths, state, stage)
                    return
                release = state.get("stage_releases", {}).get(stage, {})
                if release.get("preflight_job_id"):
                    scheduler, detail = _scheduler_terminal(paths, release["preflight_job_id"])
                    if scheduler in ACTIVE_SCHEDULER_STATES:
                        _set_state(
                            state,
                            f"STUDY{_stage_number(stage)}_PREFLIGHT_RUNNING",
                            reason=f"{stage} preflight pending: {detail}",
                        )
                        return
                    if scheduler != "COMPLETED":
                        release["status"] = "FAILED"
                        _set_state(
                            state,
                            "BLOCKED",
                            reason=f"{stage} preflight failed: {detail}",
                        )
                        return
                ready, reason = _stage_release_ready(paths, state, stage)
                if not ready:
                    if release.get("preflight_job_id"):
                        _set_state(state, "BLOCKED", reason=reason)
                    else:
                        _set_stage_preflight_required(paths, state, stage, reason)
                    return
                _submit_stage_and_controller(paths, state, stage)
                return
            _reconcile_stage(paths, state, stage)
            if state["stages"][stage].get("state") == "COMPLETE" and not state["decisions"].get(STAGE_GATES[stage], {}).get("materialised"):
                decision = _policy_decision(paths, state, stage)
                _write_decision(paths, state, STAGE_GATES[stage], decision)
                if decision["status"] == "AUTO_SELECTED" and decision.get("decision"):
                    _materialise_gate(paths, state, stage, decision["decision"], decision)
                _advance_after_gate(paths, state, stage, decision)
        return
    if state["smoke"].get("status") in {"PENDING", "RUNNING", "SUBMITTED"}:
        _set_state(state, "WAITING_SMOKE", reason="exact source-SHA GPU smoke is pending/running")
        return
    if state["smoke"].get("status") == "FAILED":
        _set_state(state, "SMOKE_FAILED", reason="exact source-SHA GPU smoke failed; inspect its artifact before retrying")
        return
    _set_state(state, "WAITING_SMOKE", reason="exact source-SHA GPU smoke has not been submitted")


def _validate_parent_coordinate_gate(parent: CampaignPaths, parent_state: dict) -> tuple[dict, Path, str]:
    if parent_state.get("state") not in {
        "STUDY15_PREFLIGHT_REQUIRED",
        "STUDY15_PREFLIGHT_RUNNING",
        "STUDY15_RUNNING",
        "STUDY15_REVIEW",
        "STUDY10_PREFLIGHT_REQUIRED",
        "STUDY10_PREFLIGHT_RUNNING",
        "STUDY10_RUNNING",
        "READY_FOR_PRODUCTION_REVIEW",
    }:
        raise CampaignError(f"Parent campaign is not a completed Study-13 continuation point: {parent_state.get('state')}")
    stage = parent_state.get("stages", {}).get("invisible_coordinates", {})
    if stage.get("state") != "COMPLETE" or stage.get("selected_variant") != "met_residual":
        raise CampaignError("Parent campaign does not contain a completed met_residual Study-13 gate")
    decision_path = parent.decisions / "invisible_coordinates.json"
    decision = _read_json(decision_path)
    if decision.get("status") != "AUTO_SELECTED" or decision.get("decision") != "met_residual":
        raise CampaignError("Parent Study-13 coordinate decision is not the locked automatic selection")
    gates = load_scientific_gates(parent.gates)
    gate = gates.get("invisible_coordinates")
    if not isinstance(gate, dict) or gate.get("status") != "locked" or gate.get("selected_variant") != "met_residual":
        raise CampaignError("Parent Study-13 coordinate gate is not locked to met_residual")
    for topology, by_seed in (gate.get("artifacts") or {}).items():
        for seed, artifact in by_seed.items():
            checkpoint = Path(str(artifact["checkpoint"])).expanduser().resolve()
            identity = Path(str(artifact["identity_file"])).expanduser().resolve()
            if not checkpoint.is_file() or sha256_file(checkpoint) != artifact.get("checkpoint_sha256"):
                raise CampaignError(f"Parent selected checkpoint is missing or mismatched: {checkpoint}")
            if not identity.is_file() or sha256_file(identity) != artifact.get("identity_sha256"):
                raise CampaignError(f"Parent selected checkpoint identity is missing or mismatched: {identity}")
            payload = _read_json(identity)
            if payload.get("artifact", {}).get("sha256") != artifact.get("checkpoint_sha256"):
                raise CampaignError(f"Parent identity artifact does not match selected checkpoint: {identity}")
            observed = payload.get("identity", {})
            if observed.get("git_commit") != artifact.get("git_sha") or int(observed.get("model_seed", -1)) != int(str(seed).removeprefix("seed_")):
                raise CampaignError(f"Parent selected checkpoint training provenance is inconsistent: {identity}")
            if observed.get("topology") != topology:
                raise CampaignError(f"Parent selected checkpoint topology is inconsistent: {identity}")
    return gate, decision_path, _sha256_file(decision_path)


def continue_campaign(
    parent_root: str | Path,
    campaign_root: str | Path,
) -> dict:
    """Create a new source-pinned continuation while leaving the parent untouched."""
    parent = campaign_paths(parent_root)
    target = campaign_paths(campaign_root)
    if not parent.state.is_file():
        raise CampaignError(f"Parent campaign does not exist: {parent.root}")
    parent_state = _read_json(parent.state)
    gate, decision_path, decision_sha = _validate_parent_coordinate_gate(parent, parent_state)
    with _campaign_lock(target):
        current_sha = _git(target, "rev-parse", "HEAD")
        if _git(target, "status", "--porcelain"):
            raise CampaignError("Campaign continuation requires a clean committed checkout")
        source = _source_metadata(target, current_sha)
        if target.state.is_file():
            existing = _read_json(target.state)
            parent_record = existing.get("parent_campaign", {})
            if (
                existing.get("source_sha") == current_sha
                and parent_record.get("campaign_root") == str(parent.root)
                and parent_record.get("parent_decision_sha256") == decision_sha
            ):
                return existing
            raise CampaignError(f"Campaign root already exists with a different identity: {target.root}")
        existing_entries = [item for item in target.root.iterdir() if item.name != "campaign.lock"]
        if existing_entries:
            raise CampaignError(f"Refusing to initialise non-empty campaign root: {target.root}")
        target.root.mkdir(parents=True, exist_ok=True)
        policy = OmegaConf.to_container(OmegaConf.load(target.policy_source), resolve=True)
        base = load_scientific_gates(target.source_root / "configs/studies/UPSTREAM_SELECTIONS.yaml")
        state = _initial_state(target, current_sha, source, policy, base)
        shutil.copyfile(target.policy_source, target.policy_snapshot)
        _write_json(target.source_json, source)
        base["invisible_coordinates"] = deepcopy(gate)
        _write_json(target.gates, base)
        target.decisions.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(decision_path, target.decisions / decision_path.name)
        state["parent_campaign"] = {
            "campaign_id": parent_state.get("campaign_id"),
            "source_sha": parent_state.get("source_sha"),
            "campaign_root": str(parent.root),
            "imported_gates": ["invisible_coordinates"],
            "parent_decision_sha256": decision_sha,
            "parent_decision_artifact": str(decision_path),
            "continuation_source_sha": current_sha,
        }
        state["smoke"] = {
            "status": "INHERITED",
            "job_id": parent_state.get("smoke", {}).get("job_id"),
            "artifact_path": parent_state.get("smoke", {}).get("artifact_path"),
            "attempts": 0,
            "release_contract": "jer_study13_release_v1",
            "inherited_from": parent_state.get("campaign_id"),
            "validated_source_sha": parent_state.get("source_sha"),
        }
        state["stages"]["invisible_coordinates"].update({
            "state": "COMPLETE",
            "selected_variant": "met_residual",
            "completion_reason": "Inherited locked Study-13 scientific gate; no Study-13 resubmission permitted",
            "inherited_from": parent_state.get("campaign_id"),
            "inherited_gate_digest": _json_digest(gate),
        })
        state["decisions"]["invisible_coordinates"] = {
            "materialised": True,
            "path": str(target.decisions / decision_path.name),
            "sha256": decision_sha,
            "inherited_from": parent_state.get("campaign_id"),
        }
        state["stage"] = "invisible_backbone"
        _set_state(state, "STUDY15_PREFLIGHT_REQUIRED", reason="Study-13 gate inherited; run the execution-backed Study-15 preflight")
        _ensure_campaign_files(target)
        _validate_overlay_immutability(target, state)
        _save_state(target, state)
        return state


def preflight(root: str | Path, stage: str) -> dict:
    """Arm one stage preflight, idempotently, behind its current campaign state."""
    paths = campaign_paths(root)
    with _campaign_lock(paths):
        if not paths.state.is_file():
            raise CampaignError(f"No Paper-1 campaign exists: {paths.root}")
        state = _read_json(paths.state)
        if state.get("stage") != stage:
            raise CampaignError(f"Campaign current stage is {state.get('stage')!r}, not {stage!r}")
        if stage not in STAGE_RELEASE_CONTRACTS:
            raise CampaignError(f"Stage has no downstream preflight: {stage}")
        ready, reason = _stage_release_ready(paths, state, stage)
        if ready:
            _save_state(paths, state)
            return state
        if state.get("stage_releases", {}).get(stage, {}).get("status") == "FAILED":
            raise CampaignError(reason)
        _submit_stage_preflight(paths, state, stage)
        _save_state(paths, state)
        return state


def launch(root: str | Path | None = None, *, auto: bool = True) -> dict:
    paths = campaign_paths(root)
    with _campaign_lock(paths):
        current_sha = _git(paths, "rev-parse", "HEAD")
        if _git(paths, "status", "--porcelain"):
            raise CampaignError("Campaign launch requires a clean committed checkout")
        source = _source_metadata(paths, current_sha)
        if not paths.policy_source.is_file():
            raise CampaignError(f"Paper-1 policy is missing: {paths.policy_source}")
        policy = OmegaConf.to_container(OmegaConf.load(paths.policy_source), resolve=True)
        base = load_scientific_gates(paths.source_root / "configs/studies/UPSTREAM_SELECTIONS.yaml")
        if paths.state.is_file():
            state = _read_json(paths.state)
            if state.get("source_sha") != current_sha:
                raise CampaignError(
                    f"Campaign already belongs to source SHA {state.get('source_sha')}; "
                    f"current checkout is {current_sha}. Start a new campaign root for a new SHA."
                )
        else:
            shutil.copyfile(paths.policy_source, paths.policy_snapshot)
            _write_json(paths.source_json, source)
            _write_json(paths.gates, base)
            state = _initial_state(paths, current_sha, source, policy, base)
        if not paths.policy_snapshot.is_file():
            shutil.copyfile(paths.policy_source, paths.policy_snapshot)
        _load_policy(paths)
        if state.get("policy_sha256") != _sha256_file(paths.policy_snapshot):
            raise CampaignError("Campaign policy snapshot checksum does not match campaign state")
        _ensure_campaign_files(paths)
        _validate_overlay_immutability(paths, state)
        smoke = _reconcile_smoke(paths, current_sha)
        if smoke is None:
            smoke = _submit_smoke(paths, current_sha)
        state["smoke"].update({key: smoke.get(key) for key in ("job_id", "status", "artifact_path", "scheduler_state", "scheduler_reason") if key in smoke})
        state["smoke"]["attempts"] = int(state["smoke"].get("attempts", 0)) + (1 if state["smoke"].get("job_id") == smoke.get("job_id") and state["smoke"].get("attempts", 0) == 0 else 0)
        _advance_locked(paths, state, event="launch")
        if state["smoke"].get("job_id"):
            _ensure_controller(paths, state, "smoke", [str(state["smoke"]["job_id"])])
        if state.get("stage") in STAGE_STUDIES:
            current = state["stages"][state["stage"]]
            if current.get("job_ids"):
                _ensure_controller(paths, state, state["stage"], current["job_ids"])
        _save_state(paths, state)
        return state


def resume(root: str | Path | None = None) -> dict:
    paths = campaign_paths(root)
    with _campaign_lock(paths):
        if not paths.state.is_file():
            raise CampaignError("No Paper-1 campaign exists; run jer campaign launch paper1 --auto")
        state = _read_json(paths.state)
        _validate_overlay_immutability(paths, state)
        if state.get("state", "").endswith("_REVIEW") or state.get("pending_review"):
            raise CampaignError("Campaign is waiting for explicit review approval; use jer campaign review/approve")
        if state["smoke"].get("status") == "FAILED":
            if int(state["smoke"].get("attempts", 0)) >= 2:
                _set_state(state, "BLOCKED", reason="GPU smoke retry limit reached")
            else:
                smoke = _submit_smoke(paths, state["source_sha"])
                state["smoke"].update(smoke)
        _advance_locked(paths, state, event="resume")
        if state["smoke"].get("job_id"):
            _ensure_controller(paths, state, "smoke", [str(state["smoke"]["job_id"])])
        if state.get("stage") in STAGE_STUDIES:
            current = state["stages"][state["stage"]]
            if current.get("job_ids"):
                _ensure_controller(paths, state, state["stage"], current["job_ids"])
        _save_state(paths, state)
        return state


def controller_step(root: str | Path, event: str) -> dict:
    paths = campaign_paths(root)
    with _campaign_lock(paths):
        state = _read_json(paths.state)
        if state.get("source_sha") != os.environ.get("JER_SOURCE_SHA", state.get("source_sha")):
            raise CampaignError("Controller source SHA does not match campaign source SHA")
        _advance_locked(paths, state, event=event)
        _save_state(paths, state)
        return state


def review(root: str | Path | None = None) -> dict:
    paths = campaign_paths(root)
    with _campaign_lock(paths):
        state = _read_json(paths.state)
        pending = state.get("pending_review")
        if not pending:
            print("No unresolved Paper-1 scientific review gate.")
            return state
        print(f"User action required: {pending.get('gate')} ({pending.get('stage')})")
        print(f"Reason: {pending.get('reason')}")
        artifact = pending.get("decision_artifact")
        if artifact and Path(artifact).is_file():
            print(Path(artifact).read_text().rstrip())
        return state


def approve(root: str | Path | None, gate: str, variant: str) -> dict:
    paths = campaign_paths(root)
    with _campaign_lock(paths):
        state = _read_json(paths.state)
        pending = state.get("pending_review") or {}
        if pending.get("gate") != gate:
            raise CampaignError(f"Campaign is not waiting for approval of gate {gate!r}")
        stage = pending["stage"]
        decision_path = paths.decisions / f"{gate}.json"
        decision = _read_json(decision_path)
        decision["decision"] = variant
        decision["status"] = "USER_APPROVED"
        decision["reason"] = f"Explicit user approval for {gate}={variant}. Original evidence retained."
        decision["approved_at"] = _now()
        _write_json(decision_path, decision)
        _materialise_gate(paths, state, stage, variant, decision)
        state["stages"][stage]["state"] = "COMPLETE"
        state["stages"][stage]["selected_variant"] = variant
        state["pending_review"] = None
        _advance_after_gate(paths, state, stage, decision)
        _save_state(paths, state)
        return state


def status(root: str | Path | None = None) -> dict:
    paths = campaign_paths(root)
    if not paths.state.is_file():
        print(f"Paper-1 campaign: not launched ({paths.root})")
        return {}
    with _campaign_lock(paths):
        state = _read_json(paths.state)
        if state.get("smoke", {}).get("job_id"):
            smoke = _reconcile_smoke(paths, state["source_sha"])
            if smoke:
                state["smoke"].update({key: smoke.get(key) for key in ("status", "scheduler_state", "scheduler_reason") if key in smoke})
        print("Paper-1 campaign")
        print(f"  id:       {state.get('campaign_id')}")
        print(f"  source:   {state.get('source_sha')}")
        print(f"  mode:     {state.get('mode', 'AUTO')}")
        print(f"  state:    {state.get('state')} — {state.get('state_reason', '')}")
        smoke = state.get("smoke", {})
        print(f"\nSmoke\n  {smoke.get('status', 'UNKNOWN')} job={smoke.get('job_id', '-')}")
        for stage, release in state.get("stage_releases", {}).items():
            print(
                f"  release {stage}: {release.get('status', 'UNKNOWN')}"
                f" — {release.get('reason', '')}"
            )
        for stage in STAGES:
            item = state.get("stages", {}).get(stage, {})
            label = STAGE_STUDIES.get(stage, "production")
            selected = item.get("selected_variant") or "-"
            print(f"\n{label}")
            print(f"  {item.get('state', 'NOT_STARTED')}  decision={selected}")
            if item.get("preflight_job_id"):
                release = state.get("stage_releases", {}).get(stage, {})
                print(
                    f"  preflight: {item['preflight_job_id']} "
                    f"{release.get('status', 'UNKNOWN')}"
                )
            if item.get("submission_record"):
                job_status, job_reason = _stage_job_status(paths, item)
                print(f"  jobs: {job_status} {' '.join(item.get('job_ids', [])) or 'none'}")
                if job_reason:
                    print(f"  detail: {job_reason}")
            if item.get("controller_job"):
                print(f"  controller: {item['controller_job']}")
        if state.get("pending_review"):
            print(f"\nUser action required: approve {state['pending_review'].get('gate')}")
        elif state.get("state") == "READY_FOR_PRODUCTION_REVIEW":
            print("\nUser action required: validate and explicitly release the production manifest")
        else:
            print("\nNext automatic action: controller continuation after the current dependency")
        _save_state(paths, state)
        return state


def watch(root: str | Path | None = None, *, interval: float = 15.0) -> None:
    previous = None
    while True:
        state = status(root)
        marker = (
            state.get("state"),
            state.get("stage"),
            state.get("smoke", {}).get("status"),
            tuple((stage, state.get("stages", {}).get(stage, {}).get("state")) for stage in STAGES),
        ) if state else None
        if marker != previous:
            print(f"{datetime.now().strftime('%H:%M')} campaign transition: {marker}", flush=True)
            previous = marker
        if (
            state.get("state") in {
                "READY_FOR_FINAL_TEST",
                "READY_FOR_PRODUCTION_REVIEW",
                "BLOCKED",
                "FAILED",
                "STUDY15_PREFLIGHT_REQUIRED",
                "STUDY10_PREFLIGHT_REQUIRED",
                "STUDY03_PREFLIGHT_REQUIRED",
            }
            or state.get("pending_review")
        ):
            return
        time.sleep(interval)


def transition_for_observation(state: dict, observation: dict) -> dict:
    """Small pure state transition used by deterministic controller simulations."""
    result = deepcopy(state)
    if result.get("state") == "WAITING_SMOKE" and observation.get("smoke") == "PASS":
        result["state"] = "STUDY13_RUNNING"
        result["stage"] = "invisible_coordinates"
    elif result.get("state") == "WAITING_SMOKE" and observation.get("smoke") == "FAILED":
        result["state"] = "SMOKE_FAILED"
    elif result.get("stage") == "invisible_coordinates" and observation.get("study13") == "FAILED":
        result["state"] = "BLOCKED"
        result["state_reason"] = "Study 13 failed; no training retry"
    elif result.get("stage") == "invisible_coordinates" and observation.get("study13") == "COMPLETE":
        if observation.get("decision_status") == "REVIEW_REQUIRED":
            result["state"] = "STUDY13_REVIEW"
        elif observation.get("decision"):
            result["state"] = "STUDY15_RUNNING"
            result["stage"] = "invisible_backbone"
    elif result.get("stage") == "invisible_backbone" and observation.get("study15") == "FAILED":
        result["state"] = "BLOCKED"
        result["state_reason"] = "Study 15 failed; no training retry"
    elif result.get("stage") == "invisible_backbone" and observation.get("study15") == "COMPLETE":
        if observation.get("decision_status") == "REVIEW_REQUIRED":
            result["state"] = "STUDY15_REVIEW"
        elif observation.get("decision"):
            result["state"] = "STUDY10_RUNNING"
            result["stage"] = "assignment_coupling"
    elif result.get("stage") == "assignment_coupling" and observation.get("study10") == "FAILED":
        result["state"] = "BLOCKED"
    elif result.get("stage") == "assignment_coupling" and observation.get("study10") == "COMPLETE":
        if observation.get("decision") == "none":
            result["state"] = "READY_FOR_PRODUCTION_REVIEW"
        elif observation.get("decision") in {"analytic", "learned"}:
            result["state"] = "STUDY03_RUNNING"
            result["stage"] = "assignment_conditioning"
    elif result.get("stage") == "assignment_conditioning" and observation.get("study03") == "COMPLETE":
        result["state"] = "READY_FOR_PRODUCTION_REVIEW"
        result["stage"] = "final_production"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--event", choices=("smoke", *STAGE_STUDIES), help="Run one controller transition")
    args = parser.parse_args(argv)
    if not args.event:
        parser.error("--event is required for a Slurm controller job")
    controller_step(args.campaign_root, args.event)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
