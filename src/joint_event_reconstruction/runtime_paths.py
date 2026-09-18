"""Canonical project/source roots and immutable source snapshots.

The live checkout and a queued worker intentionally have different source roots:
workers execute an archived committed tree, while scientific data and results
remain under the one project root.  Keep this module small so every maintained
execution path uses the same rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

from omegaconf import OmegaConf


_SHA = re.compile(r"^[0-9a-f]{40}$")
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOT_SCHEMA = "jer_source_snapshot_v1"


@dataclass(frozen=True)
class RuntimePaths:
    project_root: Path
    source_root: Path
    data_root: Path
    cache_root: Path
    results_root: Path
    logs_root: Path
    runtime_root: Path
    source_sha: str

    def project(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def source(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.source_root / path).resolve()


def _absolute(root: Path, value: object, field: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _git_sha(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    value = result.stdout.strip()
    if result.returncode == 0 and _SHA.fullmatch(value):
        return value
    metadata = project_root / "SOURCE_SNAPSHOT.json"
    if metadata.is_file():
        payload = json.loads(metadata.read_text())
        value = str(payload.get("source_sha", ""))
        if _SHA.fullmatch(value):
            return value
    raise RuntimeError(
        "JER_SOURCE_SHA is unset and no committed Git checkout/source snapshot "
        f"could be resolved from {project_root}"
    )


def _site_values(project_root: Path) -> dict:
    configured = Path(os.environ.get("JER_SITE_CONFIG", "configs/site/csf3.yaml"))
    if not configured.is_absolute():
        configured = project_root / configured
    configured = configured.resolve()
    if not configured.is_file():
        raise FileNotFoundError(f"CSF3 site configuration does not exist: {configured}")
    raw = OmegaConf.to_container(OmegaConf.load(configured), resolve=True)
    if not isinstance(raw, dict) or set(raw) != {"site", "paths", "slurm"}:
        raise ValueError(
            f"Site configuration must contain exactly site, paths and slurm: {configured}"
        )
    if raw["site"].get("name") != "csf3":
        raise ValueError(f"Site configuration is not for csf3: {configured}")
    required_slurm = {
        "gpu_partition", "gpu_gres", "gpu_cpus", "gpu_memory", "gpu_time",
        "cpu_partition", "cpu_cpus", "cpu_memory", "cpu_time",
    }
    if set(raw["slurm"]) != required_slurm:
        raise ValueError(
            f"CSF3 Slurm configuration must contain exactly {sorted(required_slurm)}: {configured}"
        )
    paths = raw["paths"]
    required = {"project_root", "data_root", "cache_root", "results_root", "logs_root", "runtime_root"}
    if set(paths) != required:
        raise ValueError(
            f"Site paths must be exactly {sorted(required)}; found {sorted(paths)}"
        )
    return raw


def runtime_paths() -> RuntimePaths:
    """Resolve the one project root and the source/data split for this process."""
    configured_project = os.environ.get("JER_PROJECT_ROOT")
    project_root = (
        Path(configured_project).expanduser().resolve()
        if configured_project
        else _PACKAGE_ROOT
    )
    site = _site_values(project_root)
    site_project = _absolute(project_root, site["paths"]["project_root"], "project_root")
    if configured_project and site_project != project_root:
        raise ValueError(
            "JER_PROJECT_ROOT disagrees with configs/site/csf3.yaml: "
            f"environment={project_root}, config={site_project}"
        )
    project_root = site_project
    source_root = Path(os.environ.get("JER_SOURCE_ROOT", str(_PACKAGE_ROOT))).expanduser().resolve()

    def selected(name: str, environment: str) -> Path:
        value = os.environ.get(environment, str(site["paths"][name]))
        return _absolute(project_root, value, name)

    source_sha = os.environ.get("JER_SOURCE_SHA") or _git_sha(project_root)
    if not _SHA.fullmatch(source_sha):
        raise ValueError(f"JER_SOURCE_SHA must be a full 40-character SHA, found {source_sha!r}")
    return RuntimePaths(
        project_root=project_root,
        source_root=source_root,
        data_root=selected("data_root", "JER_DATA_ROOT"),
        cache_root=selected("cache_root", "JER_CACHE_ROOT"),
        results_root=selected("results_root", "JER_RESULTS_ROOT"),
        logs_root=selected("logs_root", "JER_LOGS_ROOT"),
        runtime_root=selected("runtime_root", "JER_RUNTIME_ROOT"),
        source_sha=source_sha,
    )


def snapshot_path(commit_sha: str, *, project_root: Path | None = None) -> Path:
    if not _SHA.fullmatch(commit_sha):
        raise ValueError(f"Source snapshot commit must be a full SHA, found {commit_sha!r}")
    root = (project_root or runtime_paths().project_root).resolve()
    return root / "runtime" / "source" / commit_sha


def _snapshot_metadata(path: Path) -> dict:
    metadata_path = path / "SOURCE_SNAPSHOT.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Source snapshot metadata is missing: {metadata_path}")
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Source snapshot metadata is not valid JSON: {metadata_path}") from error
    return payload


def verify_source_snapshot(
    commit_sha: str,
    *,
    project_root: Path | None = None,
    source_root: Path | None = None,
    expected_tree_sha: str | None = None,
) -> Path:
    """Verify metadata and required files before a snapshot is reused."""
    target = (source_root or snapshot_path(commit_sha, project_root=project_root)).resolve()
    if not target.is_dir():
        raise FileNotFoundError(f"Immutable JER source snapshot does not exist: {target}")
    payload = _snapshot_metadata(target)
    if payload.get("schema") != _SNAPSHOT_SCHEMA:
        raise RuntimeError(f"Unsupported source snapshot schema in {target}")
    if payload.get("commit_sha") != commit_sha:
        raise RuntimeError(
            f"Source snapshot SHA mismatch: expected={commit_sha}, found={payload.get('commit_sha')}"
        )
    if expected_tree_sha is None:
        observed_tree = subprocess.run(
            ["git", "-C", str((project_root or target).resolve()), "rev-parse", f"{commit_sha}^{{tree}}"],
            text=True,
            capture_output=True,
            check=False,
        )
        if observed_tree.returncode:
            raise RuntimeError(
                "Cannot verify source snapshot tree against the canonical repository: "
                f"{observed_tree.stderr.strip()}"
            )
        expected_tree_sha = observed_tree.stdout.strip()
    if payload.get("tree_sha") != expected_tree_sha:
        raise RuntimeError(
            f"Source snapshot tree mismatch: expected={expected_tree_sha}, found={payload.get('tree_sha')}"
        )
    required = ("src", "scripts", "configs", "pyproject.toml")
    missing = [name for name in required if not (target / name).exists()]
    if missing:
        raise RuntimeError(f"Source snapshot is incomplete: {target}; missing={missing}")
    if (target / ".git").exists():
        raise RuntimeError(f"Source snapshot must not contain Git metadata: {target / '.git'}")
    return target


def ensure_source_snapshot(commit_sha: str, *, project_root: Path | None = None) -> Path:
    """Create and verify a race-safe immutable archive of one committed tree."""
    root = (project_root or runtime_paths().project_root).resolve()
    target = snapshot_path(commit_sha, project_root=root)
    git_tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", f"{commit_sha}^{{tree}}"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return verify_source_snapshot(
            commit_sha,
            project_root=root,
            source_root=target,
            expected_tree_sha=git_tree,
        )

    temporary = Path(tempfile.mkdtemp(prefix=f".{commit_sha}.", dir=parent))
    archive_path = temporary.with_name(f"{temporary.name}.tar")
    try:
        with archive_path.open("wb") as archive:
            subprocess.run(
                ["git", "-C", str(root), "archive", "--format=tar", commit_sha],
                stdout=archive,
                check=True,
            )
        with tarfile.open(archive_path, mode="r") as archive:
            archive.extractall(temporary)
        archive_path.unlink()
        metadata = {
            "schema": _SNAPSHOT_SCHEMA,
            "commit_sha": commit_sha,
            "tree_sha": git_tree,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "canonical_repository": str(root),
        }
        (temporary / "SOURCE_SNAPSHOT.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        verify_source_snapshot(
            commit_sha,
            project_root=root,
            source_root=temporary,
            expected_tree_sha=git_tree,
        )
        try:
            temporary.rename(target)
        except FileExistsError:
            shutil.rmtree(temporary)
        return verify_source_snapshot(
            commit_sha,
            project_root=root,
            source_root=target,
            expected_tree_sha=git_tree,
        )
    except Exception:
        if archive_path.exists():
            archive_path.unlink()
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
