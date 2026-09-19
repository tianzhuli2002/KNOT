#!/usr/bin/env python3
"""Execute one downstream stage preflight and materialise its release record."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from joint_event_reconstruction.runtime_paths import runtime_paths, verify_source_snapshot
from joint_event_reconstruction.stage_release import gate_checkpoint_hashes
from joint_event_reconstruction.studies import resolve_study
from joint_event_reconstruction.studies import load_scientific_gates


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STAGE_INFO = {
    "invisible_backbone": (
        "study15_release_v1",
        "invisible_backbone.json",
        "configs/studies/15_lgatr_invisible_generator.yaml",
        "preflight_study15",
        "jer_study15_integration_v1",
    ),
    "assignment_coupling": (
        "study10_release_v1",
        "assignment_coupling.json",
        "configs/studies/10_compatibility_reranking.yaml",
        "preflight_study10",
        "jer_study10_preflight_v1",
    ),
    "assignment_conditioning": (
        "study03_release_v1",
        "assignment_conditioning.json",
        "configs/studies/03_assignment_conditioning.yaml",
        "preflight_study03",
        "jer_study03_integration_v1",
    ),
}


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run(*, stage: str, output_dir: Path, release_artifact: Path, campaign_root: Path, expected_source_sha: str, gates_path: Path, campaign_id: str) -> dict:
    if stage not in STAGE_INFO:
        raise ValueError(f"Unsupported downstream stage: {stage}")
    paths = runtime_paths()
    if paths.source_sha != expected_source_sha:
        raise RuntimeError(f"SOURCE SHA MISMATCH: expected {expected_source_sha}, found {paths.source_sha}")
    verify_source_snapshot(expected_source_sha, project_root=paths.project_root, source_root=paths.source_root)
    state = json.loads((campaign_root / "campaign.json").read_text())
    if state.get("campaign_id") != campaign_id or state.get("source_sha") != expected_source_sha:
        raise RuntimeError("Stage release campaign identity does not match the immutable campaign state")
    contract, _, manifest_relative, module_name, sub_schema = STAGE_INFO[stage]
    manifest = paths.source(manifest_relative)
    study = resolve_study(manifest, gates_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    module = __import__(f"scripts.{module_name}", fromlist=["run"])
    result = module.run(
        output_dir=output_dir,
        gates_path=gates_path,
        expected_source_sha=expected_source_sha,
        campaign_id=campaign_id,
    )
    if result.get("status") != "PASS" or result.get("schema") != sub_schema:
        raise RuntimeError(f"{stage} preflight did not produce a PASS {sub_schema} result")
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise RuntimeError("Stage release must run as a Slurm job with numeric SLURM_JOB_ID")
    selected = gate_checkpoint_hashes(
        load_scientific_gates(gates_path), study.requires, seeds=(42,)
    )
    payload = {
        "schema": contract,
        "status": "PASS",
        "software_integration_evidence_only": True,
        "source_sha": expected_source_sha,
        "campaign_id": campaign_id,
        "stage": stage,
        "slurm_job_id": job_id,
        "slurm_terminal_state": "COMPLETED",
        "slurm_exit_code": "0:0",
        "upstream_gate_digest": study.gates_digest,
        "manifest_sha256": __import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
        "required_components": result.get("required_components", {}),
        "checkpoint_hashes": selected,
        "subresults": {
            "integration": {"path": str(output_dir / "result.json"), "schema": sub_schema},
        },
        "integration_result": result,
    }
    _write(release_artifact, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_INFO))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-artifact", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--gates-path", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run(stage=args.stage, output_dir=args.output_dir.expanduser().resolve(), release_artifact=args.release_artifact.expanduser().resolve(), campaign_root=args.campaign_root.expanduser().resolve(), expected_source_sha=args.expected_source_sha, gates_path=args.gates_path.expanduser().resolve(), campaign_id=args.campaign_id), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
