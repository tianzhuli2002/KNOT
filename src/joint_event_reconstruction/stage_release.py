"""Execution-backed validation shared by the Paper-1 downstream gates.

The campaign controller owns Slurm observations.  This module validates the
scientific contents of a release artifact independently of the scheduler so a
hand-written ``PASS`` file cannot release a study.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


class StageReleaseError(RuntimeError):
    """A stage release artifact does not satisfy its declared contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise StageReleaseError(f"Cannot read release JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageReleaseError(f"Release JSON must be an object: {path}")
    return value


def gate_checkpoint_hashes(
    gates: dict[str, dict],
    required_gates: Iterable[str],
    *,
    seeds: Iterable[int] | None = None,
) -> dict[str, str]:
    """Return the exact selected checkpoint map for the gates used by a stage."""
    result: dict[str, str] = {}
    for gate_name in required_gates:
        gate = gates.get(gate_name)
        if not isinstance(gate, dict) or gate.get("status") != "locked":
            raise StageReleaseError(f"Required upstream gate is not locked: {gate_name}")
        artifacts = gate.get("artifacts")
        if artifacts == {}:
            continue
        if not isinstance(artifacts, dict):
            raise StageReleaseError(f"Locked gate has no artifact registry: {gate_name}")
        for topology, by_seed in artifacts.items():
            if not isinstance(by_seed, dict):
                raise StageReleaseError(f"Invalid artifact registry for {gate_name}:{topology}")
            for seed, artifact in by_seed.items():
                if seeds is not None and int(str(seed).removeprefix("seed_")) not in set(seeds):
                    continue
                if not isinstance(artifact, dict) or not artifact.get("checkpoint"):
                    raise StageReleaseError(f"Missing checkpoint for {gate_name}:{topology}:{seed}")
                path = str(Path(str(artifact["checkpoint"])).expanduser().resolve())
                result[path] = str(artifact.get("checkpoint_sha256", ""))
    return result


def _under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _subresult_paths(payload: dict) -> list[tuple[Path, str | None]]:
    subresults = payload.get("subresults")
    if not isinstance(subresults, dict) or not subresults:
        raise StageReleaseError("Release artifact has no subresults")
    result: list[tuple[Path, str | None]] = []
    for name, value in subresults.items():
        if isinstance(value, str):
            result.append((Path(value), None))
            continue
        if not isinstance(value, dict) or not isinstance(value.get("path"), str):
            raise StageReleaseError(f"Invalid subresult declaration: {name}")
        result.append((Path(value["path"]), value.get("schema")))
    return result


def validate_release_payload(
    payload: dict,
    *,
    artifact_path: Path,
    expected_schema: str,
    expected_source_sha: str,
    expected_campaign_id: str,
    expected_gate_digest: str,
    expected_manifest_sha256: str,
    expected_checkpoint_hashes: dict[str, str],
    validation_root: Path,
) -> dict:
    """Validate an executed stage artifact and all declared sub-results."""
    if payload.get("schema") != expected_schema:
        raise StageReleaseError(
            f"unexpected release schema: {payload.get('schema')!r}; expected {expected_schema!r}"
        )
    if payload.get("status") != "PASS":
        raise StageReleaseError("release artifact is not PASS")
    if payload.get("software_integration_evidence_only") is not True:
        raise StageReleaseError("release artifact is not marked software-integration evidence")
    for key, expected in (
        ("source_sha", expected_source_sha),
        ("campaign_id", expected_campaign_id),
        ("upstream_gate_digest", expected_gate_digest),
        ("manifest_sha256", expected_manifest_sha256),
    ):
        if payload.get(key) != expected:
            raise StageReleaseError(
                f"release {key} mismatch: expected={expected!r}, found={payload.get(key)!r}"
            )
    job_id = str(payload.get("slurm_job_id", ""))
    if not job_id.isdigit():
        raise StageReleaseError("release artifact has no numeric Slurm job ID")
    if payload.get("slurm_terminal_state") != "COMPLETED":
        raise StageReleaseError("release artifact does not record COMPLETED terminal state")
    if payload.get("slurm_exit_code") != "0:0":
        raise StageReleaseError("release artifact does not record exit code 0:0")

    required = payload.get("required_components")
    if not isinstance(required, dict) or not required:
        raise StageReleaseError("release required_components is missing")
    failed = sorted(name for name, value in required.items() if value is not True)
    if failed:
        raise StageReleaseError(f"release required components failed: {failed}")

    declared = payload.get("checkpoint_hashes")
    if not isinstance(declared, dict):
        raise StageReleaseError("release checkpoint_hashes is missing")
    for path_text, expected_hash in expected_checkpoint_hashes.items():
        if declared.get(path_text) != expected_hash:
            raise StageReleaseError(f"selected checkpoint is absent or mismatched: {path_text}")
    for path_text, expected_hash in declared.items():
        path = Path(str(path_text)).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(expected_hash):
            raise StageReleaseError(f"release checkpoint hash does not match file: {path}")

    if not _under(artifact_path, validation_root):
        raise StageReleaseError(f"release artifact is outside the validation root: {artifact_path}")
    for path, schema in _subresult_paths(payload):
        path = path.expanduser().resolve()
        if not _under(path, validation_root):
            raise StageReleaseError(f"subresult is outside the validation root: {path}")
        if not path.is_file():
            raise StageReleaseError(f"declared subresult is missing: {path}")
        subresult = _read_json(path)
        if subresult.get("status") != "PASS":
            raise StageReleaseError(f"subresult is not PASS: {path}")
        if schema is not None and subresult.get("schema") != schema:
            raise StageReleaseError(
                f"subresult schema mismatch for {path}: {subresult.get('schema')!r}"
            )
        if subresult.get("source_sha") != expected_source_sha:
            raise StageReleaseError(f"subresult source SHA mismatch: {path}")
        if subresult.get("test_accessed") is True:
            raise StageReleaseError(f"subresult accessed the sealed test split: {path}")

    return {
        "schema": expected_schema,
        "status": "PASS",
        "slurm_job_id": job_id,
        "validated_checkpoint_count": len(declared),
        "validated_subresult_count": len(payload["subresults"]),
    }
