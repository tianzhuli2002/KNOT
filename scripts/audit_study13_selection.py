#!/usr/bin/env python3
"""Audit the locked Study-13 coordinate decision without changing it."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from joint_event_reconstruction.studies import load_scientific_gates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(checkpoint: Path, expected: dict) -> dict:
    identity_path = checkpoint.parent.parent / "artifact_identity.json"
    if not checkpoint.is_file() or not identity_path.is_file():
        raise RuntimeError(f"Missing Study-13 checkpoint or identity: {checkpoint}")
    checkpoint_sha = _sha256(checkpoint)
    identity_sha = _sha256(identity_path)
    if checkpoint_sha != expected["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint checksum mismatch: {checkpoint}")
    if identity_sha != expected["identity_sha256"]:
        raise RuntimeError(f"Checkpoint identity checksum mismatch: {identity_path}")
    payload = json.loads(identity_path.read_text())
    if payload.get("artifact", {}).get("sha256") != checkpoint_sha:
        raise RuntimeError(f"Identity artifact does not describe checkpoint: {identity_path}")
    identity = payload.get("identity", {})
    for key in ("study", "variant", "topology", "model_seed", "git_commit", "architecture_signature"):
        observed = identity.get(key)
        required = {
            "study": expected["source_study"],
            "variant": expected["variant"],
            "topology": "ttbar2L",
            "model_seed": int(expected["model_seed"]),
            "git_commit": expected["git_sha"],
            "architecture_signature": expected["architecture_identity"],
        }[key]
        if observed != required:
            raise RuntimeError(f"Identity mismatch for {identity_path}: {key}={observed!r}")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "identity_file": str(identity_path),
        "identity_sha256": identity_sha,
        "training_provenance": {
            "source_study": identity["study"],
            "variant": identity["variant"],
            "git_sha": identity["git_commit"],
            "model_seed": int(identity["model_seed"]),
            "architecture_identity": identity["architecture_signature"],
        },
        "data_subset": identity.get("data_subset", {}),
        "resolved_config": identity.get("resolved_config", {}),
    }


def run(parent: Path, output: Path) -> dict:
    state = json.loads((parent / "campaign.json").read_text())
    decision_path = parent / "decisions/invisible_coordinates.json"
    decision = json.loads(decision_path.read_text())
    gates = load_scientific_gates(parent / "gates.json")
    gate = gates["invisible_coordinates"]
    if decision.get("status") != "AUTO_SELECTED":
        raise RuntimeError(f"Study-13 decision is not AUTO_SELECTED: {decision.get('status')}")
    if decision.get("decision") != "met_residual":
        raise RuntimeError(f"Study-13 selected variant is not met_residual: {decision.get('decision')}")
    if gate.get("status") != "locked" or gate.get("selected_variant") != "met_residual":
        raise RuntimeError("Study-13 coordinate gate is not the locked met_residual gate")
    if state.get("source_sha") != gate.get("validated_sha"):
        raise RuntimeError("Study-13 gate validation SHA disagrees with campaign source SHA")

    checkpoints = {}
    architectures = set()
    coordinates = set()
    selected = gate["artifacts"]["ttbar2L"]
    for seed in (42, 43, 44):
        entry = selected[f"seed_{seed}"]
        record = _identity(Path(entry["checkpoint"]).expanduser().resolve(), entry)
        payload = torch.load(record["checkpoint"], map_location="cpu", weights_only=False)
        hparams = payload.get("hyper_parameters", {})
        architectures.add(str(entry["architecture_identity"]))
        coordinates.add(str(hparams.get("invisible_coordinates", entry.get("selected_patch", {}).get("coordinate"))))
        checkpoints[str(seed)] = record
    if len(architectures) != 1 or coordinates != {"met_residual"}:
        raise RuntimeError(f"Study-13 selected checkpoints disagree: architectures={architectures}, coordinates={coordinates}")

    baseline = gates["invisible_generator"]["artifacts"]["ttbar2L"]
    baseline_metadata = {}
    for seed in (42, 43, 44):
        entry = baseline[f"seed_{seed}"]
        record = _identity(Path(entry["checkpoint"]).expanduser().resolve(), entry)
        baseline_metadata[str(seed)] = {
            "checkpoint_sha256": record["checkpoint_sha256"],
            "training_sha": record["training_provenance"]["git_sha"],
            "data_subset": record["data_subset"],
            "training": record["resolved_config"].get("training", {}),
        }
    residual_comparability = {
        seed: {
            "training_sha": value["training_provenance"]["git_sha"],
            "data_subset": value["data_subset"],
            "training": value["resolved_config"].get("training", {}),
        }
        for seed, value in checkpoints.items()
    }
    result = {
        "schema": "jer_study13_selection_audit_v1",
        "status": "PASS",
        "campaign_id": state.get("campaign_id"),
        "campaign_source_sha": state.get("source_sha"),
        "decision_sha256": _sha256(decision_path),
        "decision_status": decision["status"],
        "selected_variant": "met_residual",
        "seed_checkpoint_hashes": {
            seed: record["checkpoint_sha256"] for seed, record in checkpoints.items()
        },
        "seed_identity_hashes": {
            seed: record["identity_sha256"] for seed, record in checkpoints.items()
        },
        "architecture_identities": sorted(architectures),
        "coordinate_representations": sorted(coordinates),
        "training_provenance": {
            seed: record["training_provenance"] for seed, record in checkpoints.items()
        },
        "training_comparability": {
            "cartesian_selected_generator": baseline_metadata,
            "met_residual_selected_generator": residual_comparability,
            "same_declared_train_validation_counts": all(
                value["data_subset"].get("splits") == next(iter(residual_comparability.values()))["data_subset"].get("splits")
                for value in baseline_metadata.values()
            ),
            "comparison_is_documentary": True,
        },
        "decision_limitations": {
            "event_level_paired_metric_available": False,
            "formal_paired_significance_claim": False,
        },
        "evaluation_contract": {
            "primary_metric_path": "invisible.top1_assignment_conditioned.energy_score_joint",
            "predicted_samples": "physical neutrino four-vector posterior samples for the top-1 visible assignment",
            "truth": "canonical validation neutrino four-vectors in the declared neutrino-role order",
            "assignment_conditioning": "top-1 assignment selects the evaluation population; it does not replace the posterior with truth-conditioned samples",
            "population": "the same validation event IDs and denominator are required across coordinate variants",
            "test_split_used": False,
            "automatic_policy": "predeclared relative equivalence margin, seed-direction requirement and coverage guardrail; ambiguous evidence requires review",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.campaign.expanduser().resolve(), args.output.expanduser().resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
