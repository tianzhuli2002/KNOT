#!/usr/bin/env python3
"""Run one immutable, provenance-checked JER production training stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from joint_event_reconstruction.configuration import load_experiment


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True
    ).strip()


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")


def refuse_nonempty_output(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True)
        return
    existing = [item for item in path.rglob("*") if item.is_file()]
    if existing:
        raise FileExistsError(
            f"Refusing to reuse non-empty production output {path}; "
            f"existing files include {existing[:5]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--expected-output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--initial-checkpoint-sha256")
    parser.add_argument("--initial-components", nargs="+", default=())
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    config = args.config.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    expected_output = args.expected_output.expanduser().resolve()
    require_file(config, "JER public configuration")
    require_file(manifest, "matched event manifest")
    actual_source = git(repo_root, "rev-parse", "HEAD")
    if actual_source != args.source_sha:
        raise RuntimeError(
            f"JER source changed after submission: expected={args.source_sha}, "
            f"observed={actual_source}"
        )
    if git(repo_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("JER checkout has tracked modifications after submission")
    actual_config_sha = sha256(config)
    if actual_config_sha != args.config_sha256:
        raise RuntimeError(
            f"JER configuration changed after submission: expected={args.config_sha256}, "
            f"observed={actual_config_sha}"
        )
    actual_manifest_sha = sha256(manifest)
    if actual_manifest_sha != args.manifest_sha256:
        raise RuntimeError(
            f"Matched manifest changed after submission: expected={args.manifest_sha256}, "
            f"observed={actual_manifest_sha}"
        )
    if bool(args.initial_checkpoint) != bool(args.initial_components):
        raise ValueError(
            "--initial-checkpoint and --initial-components must be supplied together"
        )
    initial = None
    if args.initial_checkpoint is not None:
        initial_path = args.initial_checkpoint.expanduser().resolve()
        require_file(initial_path, "JER initialisation checkpoint")
        if args.initial_checkpoint_sha256 is None:
            raise ValueError("Initial checkpoints require --initial-checkpoint-sha256")
        actual_initial_sha = sha256(initial_path)
        if actual_initial_sha != args.initial_checkpoint_sha256:
            raise RuntimeError(
                "JER initialisation checkpoint changed after submission: "
                f"expected={args.initial_checkpoint_sha256}, observed={actual_initial_sha}"
            )
        initial = {
            "path": str(initial_path),
            "sha256": actual_initial_sha,
            "components": list(args.initial_components),
        }

    resolved = load_experiment(config)
    if Path(str(resolved.training.output)).expanduser().resolve() != expected_output:
        raise RuntimeError(
            "Configuration/output mismatch: "
            f"resolved={resolved.training.output}, expected={expected_output}"
        )
    refuse_nonempty_output(expected_output)

    command = [
        sys.executable,
        "-m",
        "joint_event_reconstruction.cli.run",
        "train",
        "--experiment",
        str(config),
        "--device",
        args.device,
    ]
    if initial is not None:
        command.extend(["--initial-checkpoint", initial["path"], "--initial-components", *initial["components"]])
    print(json.dumps({"command": command, "source_sha": actual_source}, sort_keys=True), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)

    checkpoint_candidates = sorted(expected_output.glob("checkpoints/best-loss*.ckpt"))
    if len(checkpoint_candidates) != 1:
        raise RuntimeError(
            "Expected exactly one best-loss checkpoint after JER training, "
            f"found={checkpoint_candidates}"
        )
    checkpoint = checkpoint_candidates[0].resolve()
    require_file(checkpoint, "JER best checkpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict")
    hyperparameters = payload.get("hyper_parameters", {})
    if not isinstance(state_dict, dict) or not state_dict:
        raise RuntimeError(f"JER checkpoint has no non-empty state_dict: {checkpoint}")
    if hyperparameters.get("git_sha") != actual_source:
        raise RuntimeError(
            "JER checkpoint source identity mismatch: "
            f"checkpoint={hyperparameters.get('git_sha')}, source={actual_source}"
        )
    expected_trainable = sorted(str(value) for value in resolved.training.trainable)
    if sorted(hyperparameters.get("trainable_components", [])) != expected_trainable:
        raise RuntimeError(
            "JER checkpoint trainable-component metadata mismatch: "
            f"checkpoint={hyperparameters.get('trainable_components')}, "
            f"config={expected_trainable}"
        )
    checkpoint_sha = sha256(checkpoint)
    record = {
        "stage": str(resolved.experiment.name),
        "source_sha": actual_source,
        "config": str(config),
        "config_sha256": actual_config_sha,
        "manifest": str(manifest),
        "manifest_sha256": actual_manifest_sha,
        "resolved_output": str(expected_output),
        "device": args.device,
        "trainable_components": expected_trainable,
        "shared_encoder_gradients": str(resolved.training.shared_encoder_gradients),
        "initialisation": initial,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_hyperparameters": {
            "architecture_signature": hyperparameters.get("architecture_signature"),
            "architecture_component_signatures": hyperparameters.get("architecture_component_signatures"),
            "topology_signature": hyperparameters.get("topology_signature"),
            "dataset_identity": hyperparameters.get("dataset_identity"),
            "split_identity": hyperparameters.get("split_identity"),
        },
        "command": command,
    }
    (expected_output / "stage_provenance.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )
    (expected_output / "checkpoint_validation.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "state_dict_entries": len(state_dict),
                "git_sha": hyperparameters.get("git_sha"),
                "trainable_components": hyperparameters.get("trainable_components"),
                "architecture_signature": hyperparameters.get("architecture_signature"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
