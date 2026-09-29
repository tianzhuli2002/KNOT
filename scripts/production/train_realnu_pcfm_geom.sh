#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "PCFM_geom training must run inside an allocated Slurm job." >&2
  exit 2
fi

source_root=$(git rev-parse --show-toplevel)
project_root=${JER_REALNU_PROJECT_ROOT:?JER_REALNU_PROJECT_ROOT is required}
python_bin=${JER_PYTHON:-/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python}
experiment="$source_root/configs/experiments/realnu_v1/production_draft/pcfm_geom.public.yaml"
initial_checkpoint="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/initialisations_objective_matched_v5/PCFM_Q_initial.ckpt"
output="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v4_parent_comparison/training/PCFM_geom"
audit="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v4_parent_comparison/audit"

if [[ ! -x "$python_bin" || ! -f "$experiment" || ! -s "$initial_checkpoint" ]]; then
  echo "PCFM_geom Python, config, or shared initial checkpoint is missing." >&2
  exit 2
fi
if [[ -d "$output" ]] && find "$output" -type f -print -quit | rg -q .; then
  echo "Refusing to reuse non-empty PCFM_geom training output: $output" >&2
  exit 2
fi
mkdir -p "$output"
export JER_PROJECT_ROOT="$project_root"
export JER_REALNU_PROJECT_ROOT="$project_root"
source_sha=$(git -C "$source_root" rev-parse HEAD)
export JER_SOURCE_ROOT="$source_root"
export JER_SOURCE_SHA="$source_sha"
export JER_REALNU_SOURCE_ROOT="$source_root"
export JER_REALNU_EXPECTED_SHA="$source_sha"
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Allocated PCFM_geom job cannot access CUDA")
print(f"PCFM_geom CUDA device: {torch.cuda.get_device_name(0)}")
PY

git -C "$source_root" status --short --branch > "$output/source_status.txt"
git -C "$source_root" diff --binary HEAD > "$output/source_worktree.patch"
cp "$experiment" "$output/pcfm_geom.public.yaml"
cp "$source_root/scripts/production/train_realnu_pcfm_geom.sh" "$output/train_realnu_pcfm_geom.sh"
cp "$source_root/scripts/validation/validate_realnu_pcfm_geom_cpu.py" "$output/validate_realnu_pcfm_geom.py"
cp "$audit/PCFM_geom_cpu_validation.json" "$output/pcfm_geom_cpu_validation.json"
cp "$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v4_parent_comparison/source_bank/manifest.json" "$output/parent_source_bank_manifest.json"

"$python_bin" - "$source_root" "$experiment" "$initial_checkpoint" "$output" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

source, config, initial, output = map(Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

changed = subprocess.check_output(
    ["git", "-C", str(source), "ls-files", "--modified", "--others", "--exclude-standard"],
    text=True,
).splitlines()
files = {name: sha256(source / name) for name in changed}
record = {
    "source_root": str(source.resolve()),
    "source_commit": subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_branch": subprocess.check_output(
        ["git", "-C", str(source), "branch", "--show-current"], text=True
    ).strip(),
    "tracked_diff_sha256": sha256(output / "source_worktree.patch"),
    "changed_file_sha256": files,
    "config": str(config.resolve()),
    "config_sha256": sha256(config),
    "initial_checkpoint": str(initial.resolve()),
    "initial_checkpoint_sha256": sha256(initial),
    "initial_components": ["visible", "queries"],
}
(output / "source_manifest.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

# One bounded real-data GPU pass, including truth-free inference and the output writer.
"$python_bin" "$source_root/scripts/validation/validate_realnu_pcfm_geom_cpu.py" \
  --measured "$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/adapted/train_validation/partonic_ttbar2l_train_validation_maxjets8_v1_measured.h5" \
  --truth "$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/adapted/train_validation_authoritative_20260926/partonic_ttbar2l_train_validation_maxjets8_v1_supervision.h5" \
  --device cuda \
  --output "$output/pcfm_geom_gpu_smoke.json"

"$python_bin" -m joint_event_reconstruction.cli.run show-config \
  --experiment "$experiment" > "$output/config.resolved.yaml"
"$python_bin" -m joint_event_reconstruction.cli.run train \
  --experiment "$experiment" \
  --device cuda \
  --initial-checkpoint "$initial_checkpoint" \
  --initial-components visible queries

checkpoint="$output/checkpoints/best-loss.ckpt"
if [[ ! -s "$checkpoint" ]]; then
  echo "PCFM_geom training did not produce best-loss.ckpt: $checkpoint" >&2
  exit 2
fi

"$python_bin" - "$source_root" "$experiment" "$initial_checkpoint" "$output" "$checkpoint" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

source, config, initial, output, checkpoint = map(Path, sys.argv[1:])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
hparams = payload.get("hyper_parameters", {})
for name, value in payload.get("state_dict", {}).items():
    if torch.is_tensor(value) and value.is_floating_point() and not torch.isfinite(value).all():
        raise RuntimeError(f"Non-finite checkpoint tensor: {name}")
if hparams.get("git_sha") != subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
).strip():
    raise RuntimeError("PCFM_geom checkpoint source commit differs from its source snapshot")
if hparams.get("initial_components") != ["visible", "queries"]:
    raise RuntimeError(
        f"PCFM_geom did not load exactly visible and query components: {hparams.get('initial_components')}"
    )
if hparams.get("partonic_scaler_fit", {}).get("population") != "frozen_train_split":
    raise RuntimeError("PCFM_geom parent scaler was not fit on the frozen train split")

scalers = {
    "invisible": payload.get("invisible_scaler"),
    "partonic": payload.get("partonic_scaler"),
}
fits = {
    "invisible": hparams.get("invisible_scaler_fit"),
    "partonic": hparams.get("partonic_scaler_fit"),
}
def json_sha256(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()

source_manifest = json.loads(
    (Path("/net/scratch/w00238tl/JointEventReconstruction_topology_queries")
     / "results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v4_parent_comparison/source_bank/manifest.json").read_text()
)
partonic_scaler_sha = json_sha256({"scaler": scalers["partonic"], "fit": fits["partonic"]})
scaler_identity_sha = json_sha256({"scalers": scalers, "fit": fits})
if partonic_scaler_sha != source_manifest["target_scaler_identity"]["partonic_scaler_sha256"]:
    raise RuntimeError("PCFM_geom training fit a different parent scaler from PCFM_Q")
if scaler_identity_sha != source_manifest["target_scaler_identity"]["scaler_identity_sha256"]:
    raise RuntimeError("PCFM_geom scaler identity differs from the common source bank")

source_manifest_record = json.loads((output / "source_manifest.json").read_text())
record = {
    "status": "passed",
    "mode": "PCFM_geom",
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_sha256": sha256(checkpoint),
    "config_path": str(config.resolve()),
    "config_sha256": sha256(config),
    "resolved_config": str((output / "config.resolved.yaml").resolve()),
    "source_root": str(source.resolve()),
    "source_commit": source_manifest_record["source_commit"],
    "source_branch": source_manifest_record["source_branch"],
    "source_worktree_patch_sha256": source_manifest_record["tracked_diff_sha256"],
    "source_changed_file_sha256": source_manifest_record["changed_file_sha256"],
    "initial_checkpoint": str(initial.resolve()),
    "initial_checkpoint_sha256": sha256(initial),
    "initial_components": ["visible", "queries"],
    "parent_roles": ["top", "antitop"],
    "parent_conditioning": "P0 topology-query role states plus measured-event L-GATr geometry",
    "velocity_backbone": "geometric",
    "partonic_scaler_sha256": partonic_scaler_sha,
    "scaler_identity_sha256": scaler_identity_sha,
    "partonic_scaler_fit": fits["partonic"],
    "trainable_components": hparams.get("trainable_components"),
    "train_count": hparams.get("split_identity", {}).get("event_id_counts", {}).get("train"),
    "validation_count": hparams.get("split_identity", {}).get("event_id_counts", {}).get("validation"),
    "reserved_test_rows_read": 0,
    "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
    "slurm_partition": __import__("os").environ.get("SLURM_JOB_PARTITION"),
    "gpu": torch.cuda.get_device_name(0),
}
(output / "production_training.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
