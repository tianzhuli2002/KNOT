#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "PCFM_noQ training must run inside an allocated Slurm job." >&2
  exit 2
fi

source_root=$(git rev-parse --show-toplevel)
project_root=${JER_REALNU_PROJECT_ROOT:?JER_REALNU_PROJECT_ROOT is required}
python_bin=${JER_PYTHON:-/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python}
experiment="$source_root/configs/experiments/realnu_v1/production_draft/pcfm_noq.public.yaml"
output="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v3_diagnostics/parent_noquery/PCFM_noQ"

if [[ ! -x "$python_bin" || ! -f "$experiment" ]]; then
  echo "PCFM_noQ Python or config is missing: python=$python_bin config=$experiment" >&2
  exit 2
fi
mkdir -p "$output"
export JER_PROJECT_ROOT="$project_root"
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("Allocated PCFM_noQ job cannot access CUDA")
print(f"PCFM_noQ CUDA device: {torch.cuda.get_device_name(0)}")
PY

git -C "$source_root" status --short --branch > "$output/source_status.txt"
git -C "$source_root" diff --binary > "$output/source_worktree.patch"
cp "$experiment" "$output/pcfm_noq.public.yaml"
cp "$source_root/scripts/production/train_realnu_pcfm_noq.sh" "$output/train_realnu_pcfm_noq.sh"
cp "$source_root/scripts/validation/validate_realnu_pcfm_noq_cpu.py" "$output/validate_realnu_pcfm_noq.py"
cp "$source_root/scripts/production/smoke_realnu_pcfm_noq_gpu.sh" "$output/smoke_realnu_pcfm_noq_gpu.sh"

# Keep the full training behind one bounded, real-data GPU smoke in the same
# allocation. A smoke failure exits this job before any training starts.
bash "$source_root/scripts/production/smoke_realnu_pcfm_noq_gpu.sh"

"$python_bin" -m joint_event_reconstruction.cli.run show-config \
  --experiment "$experiment" > "$output/config.resolved.yaml"
"$python_bin" -m joint_event_reconstruction.cli.run train \
  --experiment "$experiment" \
  --device cuda

checkpoint="$output/checkpoints/best-loss.ckpt"
if [[ ! -s "$checkpoint" ]]; then
  echo "PCFM_noQ training did not produce best-loss.ckpt: $checkpoint" >&2
  exit 2
fi

"$python_bin" - "$source_root" "$experiment" "$output" "$checkpoint" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

source_root, experiment, output, checkpoint = map(Path, sys.argv[1:])
def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

diff_path = output / "source_worktree.patch"
record = {
    "status": "passed",
    "model": "PCFM_noQ",
    "split": "frozen train and validation only; sealed test unopened",
    "sealed_test_used": False,
    "parent_roles": ["top", "antitop"],
    "parent_conditioning": "P0 raw visible encoder event token plus measured object tokens; no topology query input",
    "source_root": str(source_root.resolve()),
    "source_commit": subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip(),
    "source_branch": subprocess.check_output(["git", "-C", str(source_root), "branch", "--show-current"], text=True).strip(),
    "source_worktree_patch_sha256": sha256(diff_path),
    "config": str(experiment.resolve()),
    "config_sha256": sha256(experiment),
    "resolved_config": str((output / "config.resolved.yaml").resolve()),
    "checkpoint": str(checkpoint.resolve()),
    "checkpoint_sha256": sha256(checkpoint),
    "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
    "slurm_partition": __import__("os").environ.get("SLURM_JOB_PARTITION"),
    "gpu": __import__("torch").cuda.get_device_name(0),
}
(output / "training_record.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY
