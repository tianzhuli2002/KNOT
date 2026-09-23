#!/usr/bin/env python3
"""Run the pinned upstream VyPER production training with identity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *arguments], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vyper-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--upstream-sha", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.vyper_root.expanduser().resolve()
    config = args.config.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    require_file(config, "VyPER configuration")
    require_file(manifest, "matched event manifest")
    require_file(args.python.expanduser().resolve(), "VyPER Python executable")
    observed_sha = git(root, "rev-parse", "HEAD")
    if observed_sha != args.upstream_sha:
        raise RuntimeError(
            f"VyPER source changed after submission: expected={args.upstream_sha}, observed={observed_sha}"
        )
    if sha256(config) != args.config_sha256:
        raise RuntimeError("VyPER configuration changed after submission")
    if sha256(manifest) != args.manifest_sha256:
        raise RuntimeError("Matched manifest changed after submission")
    if output.exists() and any(item.is_file() for item in output.rglob("*")):
        raise FileExistsError(f"Refusing to reuse non-empty VyPER output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [str(args.python.expanduser().resolve()), "-m", "VyPER.train", f"--config-name={args.config_name}"]
    print(json.dumps({"command": command, "upstream_sha": observed_sha}, sort_keys=True), flush=True)
    subprocess.run(command, cwd=root, check=True)
    checkpoints = sorted(output.rglob("*.ckpt"))
    if not checkpoints:
        raise RuntimeError(f"VyPER training produced no checkpoint under {output}")
    nonempty = [path for path in checkpoints if path.stat().st_size > 0]
    if len(nonempty) != len(checkpoints):
        raise RuntimeError(f"VyPER produced empty checkpoint files: {checkpoints}")
    record = {
        "upstream_sha": observed_sha,
        "vyper_root": str(root),
        "python": str(args.python.expanduser().resolve()),
        "config": str(config),
        "config_sha256": sha256(config),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "output": str(output),
        "checkpoints": [
            {"path": str(path.resolve()), "sha256": sha256(path)} for path in checkpoints
        ],
        "command": command,
        "noise_distribution": "uniform torch.rand (pinned upstream semantics)",
    }
    (output / "stage_provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
