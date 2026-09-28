"""Freeze the realnu R2 manifest and comparison contract before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

TRAINING_SOURCE_SHA = "cb2b60445499f78421503d4fccc4a43cae1444fd"
BASE_EVALUATION_SHA = "e1aea5cce1d8a6dce5545f7a0fe5b25ff53367c8"
MODES = ("Q0", "Q1", "PCFM_Q", "J0", "J_A", "J_AN")
PROJECT_ROOT = Path(
    os.environ.get("JER_REALNU_PROJECT_ROOT", Path(__file__).resolve().parents[2])
).expanduser().resolve()
PRODUCTION_ROOT = (
    PROJECT_ROOT
    / "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2"
)
EVALUATION_ROOT = PRODUCTION_ROOT / "evaluation_r2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def evaluation_contract(manifest: dict, evaluation_sha: str) -> dict:
    return {
        "schema": "realnu_evaluation_contract_v1",
        "training_source_sha": TRAINING_SOURCE_SHA,
        "evaluation_recovery_sha": evaluation_sha,
        "campaign": manifest["campaign"],
        "validation": {
            "split": "validation",
            "event_count": 100342,
            "sorted_event_id_sha256": manifest["inputs"]["validation_sorted_event_id_sha256"],
            "reserved_test_rows_read": 0,
            "support_masks": {
                "visible_assignment": "all reconstructed root roles have a non-null truth candidate",
                "expected_supported_events": 100323,
                "neutrino_truth": "both roles supported; expected full 100342-event population",
                "parent_truth": "both ordered parent roles supported; expected full 100342-event population",
            },
        },
        "jet_multiplicity": {
            "physics_definition": "source_njet = round(INPUTS/GLOBAL.njet) on EVENT_ID-aligned measured rows",
            "model_visible_definition": "count finite INPUTS/JET.pt > 0 slots on the same rows",
            "model_max_jets": 8,
            "adapter_check": "model_visible_njet == min(source_njet, 8)",
            "physics_bins": [2, 3, 4, 5, 6, 7, 8, ">=9"],
            "truncation_comparison": ["source_njet<=8", "source_njet>8"],
            "expected_truncated_events": 164,
        },
        "references": {
            "parton_truth": "generator-level after-FSR ordered top and antitop from authoritative supervision sidecar",
            "perfect_reco": "measured visible objects + truth assignment + truth neutrino momenta",
            "model_reco": "measured visible objects + model-predicted assignment and/or neutrino/parent state",
        },
        "primary_observables": [
            "ordered top and antitop pT, rapidity, mass, energy",
            "mttbar derived only from top + antitop four-vectors",
            "ttbar pT, rapidity, phi, energy",
            "delta rapidity and delta phi of top/antitop",
            "assignment complete-event efficiency and neutrino pT/pz/DeltaR",
        ],
        "primary_comparisons": [
            {"question": "topology queries", "baseline": "Q0", "alternative": "Q1"},
            {"question": "auxiliary reconstruction supervision", "baseline": "PCFM_Q", "alternative": "J0"},
            {"question": "predicted assignment context", "baseline": "J0", "alternative": "J_A"},
            {"question": "predicted neutrino context", "baseline": "J_A", "alternative": "J_AN"},
        ],
        "mttbar_binning_gev": {
            "primary_edges": [345, 400, 500, 700, 1000, "open-ended"],
            "fine_threshold_edges": [345, 360, 375, 390, 400, 415, 430, 450],
        },
        "posterior": {
            "sample_count": 64,
            "nested_prefixes": [1, 4, 8, 16, 32, 64],
            "nominal_coherent_event_estimator": "joint posterior medoid; choose one complete correlated top/antitop sample",
            "observable_summaries": ["mean", "q16", "q50", "q84"],
            "physicality": {
                "raw_samples_retained_for_diagnostics": True,
                "finite_components_required": True,
                "parent_energy_positive": True,
                "mass_squared_tolerance": "m2 >= -1e-10 * (E^2 + |p|^2); clamp accepted roundoff to zero",
                "nominal_sample_filter": "none; report raw physical fraction and any accepted fraction separately",
            },
        },
        "paired_statistics": {
            "resamples": 2000,
            "seed": 730241,
            "resampling": "paired EVENT_ID bootstrap on the common support mask for each observable",
            "interval": "central 95% percentile interval",
            "resample_size": "the full common-support event count",
        },
        "final_architecture_selection": {
            "candidates": ["PCFM_Q", "J0", "J_A", "J_AN"],
            "metrics": [
                "inclusive mttbar RMSE",
                "345-400 GeV mttbar RMSE",
                "top pT RMSE",
                "top rapidity RMSE",
                "antitop pT RMSE",
                "antitop rapidity RMSE",
            ],
            "rule": "Remove Pareto-dominated candidates; if more than one candidate remains, select the least-conditioned architecture in order PCFM_Q, J0, J_A, J_AN.",
            "no_aggregate_score": True,
        },
        "spectrum_reweighting": {
            "truth_regions_gev": {"threshold": [0, 400], "mid": [400, 700], "high": [700, "open-ended"]},
            "nominal_pair_threshold_window_gev": [345, 400],
            "expected_events_below_345_gev": 411,
            "populations": [
                "nominal",
                "approximately equal region weight",
                "threshold-enhanced 50/30/20",
                "high-mass-enhanced 15/25/60",
            ],
            "weight_cap": 5.0,
            "effective_sample_size": "report for each weight set",
            "interpretation": "population-reweighting sensitivity only; not OOD validation",
        },
        "reserved_test_used": False,
    }


def render_contract(contract: dict) -> str:
    return "\n".join(
        [
            "# Realnu evaluation contract",
            "",
            f"- Training source SHA: `{contract['training_source_sha']}`",
            f"- Evaluation recovery SHA: `{contract['evaluation_recovery_sha']}`",
            f"- Frozen validation population: {contract['validation']['event_count']:,} events",
            f"- Sorted EVENT_ID SHA-256: `{contract['validation']['sorted_event_id_sha256']}`",
            "- Sealed test rows read: 0",
            "",
            "## Frozen definitions",
            "",
            f"- Physics multiplicity: {contract['jet_multiplicity']['physics_definition']}",
            f"- Model-visible multiplicity: {contract['jet_multiplicity']['model_visible_definition']}",
            f"- Source-vs-visible truncation expectation: {contract['jet_multiplicity']['expected_truncated_events']} events",
            "- References: after-FSR parton truth, perfect reconstruction, and model reconstruction are kept separate.",
            "- Parent estimators use ordered top/antitop roles; mttbar is derived from their four-vector sum.",
            "- The coherent posterior event estimator is the joint medoid, selecting one correlated whole-event sample.",
            f"- Paired bootstrap: {contract['paired_statistics']['resamples']:,} resamples with seed {contract['paired_statistics']['seed']}.",
            "- Spectrum weights use truth mttbar below 400 GeV (including the recorded sub-345 GeV generator tail), 400–700 GeV, and >=700 GeV; the 345–400 GeV response is reported separately.",
            "- The spectrum study is a population-reweighting sensitivity check and will not be described as OOD validation.",
            "",
            "## Predeclared comparisons",
            "",
            "| Question | Baseline | Alternative |",
            "|---|---|---|",
            *[
                f"| {item['question']} | {item['baseline']} | {item['alternative']} |"
                for item in contract["primary_comparisons"]
            ],
            "",
            "The machine-readable values, support masks, bin edges, posterior prefixes, physicality checks, and robustness weights are in `evaluation_contract.json`.",
            "",
        ]
    )


def prepare(manifest_path: Path, evaluation_sha: str) -> dict:
    source_path = manifest_path.expanduser().resolve()
    manifest = json.loads(source_path.read_text())
    if manifest.get("source_commit") != TRAINING_SOURCE_SHA:
        raise RuntimeError("Campaign run manifest has a different training source SHA")
    if manifest.get("test_rows_read") != 0 or manifest["inputs"].get(
        "reserved_test_rows_materialised"
    ) != 0:
        raise RuntimeError("Campaign manifest does not prove sealed-test exclusion")
    if set(manifest.get("modes", {})) != set(MODES):
        raise RuntimeError("Campaign manifest does not contain the exact six release modes")
    snapshot = PROJECT_ROOT / "runtime/source" / evaluation_sha
    source_metadata_path = snapshot / "SOURCE_SNAPSHOT.json"
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"Evaluation recovery snapshot is missing: {snapshot}")
    source_metadata = json.loads(source_metadata_path.read_text())
    if source_metadata.get("commit_sha") != evaluation_sha:
        raise RuntimeError("Evaluation recovery snapshot SHA does not match requested source")
    if EVALUATION_ROOT.exists():
        raise FileExistsError(f"Refusing to replace existing evaluation R2 root: {EVALUATION_ROOT}")
    EVALUATION_ROOT.mkdir(parents=True)
    contract = evaluation_contract(manifest, evaluation_sha)
    recovery_manifest = json.loads(json.dumps(manifest))
    recovery_manifest["evaluation_recovery"] = {
        "branch": "research/realnu-evaluation-recovery-v2",
        "source_sha": evaluation_sha,
        "base_sha": BASE_EVALUATION_SHA,
        "source_root": str(snapshot.resolve()),
        "original_run_manifest": str(source_path),
        "original_run_manifest_sha256": _sha256(source_path),
        "r2_root": str(EVALUATION_ROOT.resolve()),
        "sealed_test_used": False,
    }
    recovery_manifest["job_ids"]["validation_r2"] = {}
    for mode in MODES:
        if mode == "PCFM_Q":
            continue
        training_output = Path(recovery_manifest["modes"][mode]["training_output"])
        original_validation_output = Path(
            recovery_manifest["modes"][mode]["validation_output"]
        )
        if mode in ("Q0", "Q1", "J0", "J_A", "J_AN"):
            recovery_manifest["modes"][mode]["reused_standard_prediction"] = str(
                (original_validation_output / "standard/validation.h5").resolve()
            )
        recovery_manifest["modes"][mode]["validation_output"] = str(
            (training_output / "validation_r2").resolve()
        )
    _atomic_json(EVALUATION_ROOT / "manifests/run_manifest_r2.json", recovery_manifest)
    _atomic_json(EVALUATION_ROOT / "evaluation_contract.json", contract)
    (EVALUATION_ROOT / "EVALUATION_CONTRACT.md").write_text(render_contract(contract))
    (EVALUATION_ROOT / "classical_baseline_status.md").write_text(
        "# Classical dilepton baseline status\n\n"
        "No locally maintained and validated ellipse, analytic dilepton, neutrino-weighting, or kinematic dilepton reconstruction was found during the R2 recovery audit.\n\n"
        "## Locations searched\n\n"
        "- `/net/scratch/w00238tl/JointEventReconstruction`\n"
        "- `/net/scratch/w00238tl/JointEventReconstruction_topology_queries`\n"
        "- `/net/scratch/w00238tl/VyPER`\n"
        "- `/net/scratch/w00238tl/HyPER`\n"
        "- `/net/scratch/w00238tl/TopTransformer`\n"
        "- `/net/scratch/w00238tl/SB_subscripts`\n"
        "- `/net/scratch/w00238tl/vyper_subscripts`\n"
        "- `/net/scratch/w00238tl/h5scripts`\n"
        "- `/net/scratch/w00238tl/phenoH5`\n\n"
        "The search covered Python/C++ source and analysis documentation for `ellipse`, `neutrino weight`, `analytic dilepton`, and `kinematic dilepton`. No implementation was available to verify for event inputs, solver assumptions, failure handling, and exact EVENT_ID joins.\n\n"
        "## Follow-up\n\n"
        "Integrate a documented collaboration implementation after its assumptions and conventions are reviewed, then run it on the identical validation EVENT_ID set with no-solution events retained explicitly. Do not compare a rushed local approximation to the trained models. This limitation does not block the six-model validation.\n"
    )
    return {
        "evaluation_root": str(EVALUATION_ROOT.resolve()),
        "manifest": str((EVALUATION_ROOT / "manifests/run_manifest_r2.json").resolve()),
        "contract": str((EVALUATION_ROOT / "evaluation_contract.json").resolve()),
        "evaluation_recovery_sha": evaluation_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--evaluation-recovery-sha", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.evaluation_recovery_sha), indent=2))


if __name__ == "__main__":
    main()
