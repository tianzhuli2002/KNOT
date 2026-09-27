#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Partonic production training must run inside a deliberately allocated Slurm job." >&2
  exit 2
fi
if [[ $# -ne 1 ]]; then
  echo "Usage: run_partonic_phase.sh EXPERIMENT_CONFIG" >&2
  exit 2
fi

repo_root=$(realpath -e "${JER_PARTONIC_SOURCE_ROOT:?JER_PARTONIC_SOURCE_ROOT is required}")
expected_sha=${JER_PARTONIC_EXPECTED_SHA:?JER_PARTONIC_EXPECTED_SHA is required}
actual_sha=$(git -C "$repo_root" rev-parse HEAD)
if [[ "$actual_sha" != "$expected_sha" ]]; then
  echo "Partonic source SHA mismatch: expected=$expected_sha actual=$actual_sha" >&2
  exit 2
fi
if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=no)" ]]; then
  echo "Partonic Slurm source checkout is dirty: $repo_root" >&2
  exit 2
fi
cd "$repo_root"
experiment=$1
python_bin=${JER_PYTHON:-python}
export JER_PROJECT_ROOT="$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("The allocated partonic phase job cannot access CUDA")
print(f"Partonic phase CUDA device: {torch.cuda.get_device_name(0)}")
PY

"$python_bin" -m joint_event_reconstruction.cli.run show-config \
  --experiment "$experiment" > "${SLURM_SUBMIT_DIR:-$PWD}/partonic_resolved_config_${SLURM_JOB_ID}.yaml"
"$python_bin" "$repo_root/scripts/data/validate_partonic_contract.py" \
  --experiment "$experiment" \
  > "${SLURM_SUBMIT_DIR:-$PWD}/partonic_preflight_${SLURM_JOB_ID}.json"
"$python_bin" -m joint_event_reconstruction.cli.run train \
  --experiment "$experiment" \
  --device cuda

training_output=$("$python_bin" - "$experiment" <<'PY'
from joint_event_reconstruction.configuration import load_experiment
import sys

print(load_experiment(sys.argv[1]).training.output)
PY
)
checkpoint="$training_output/checkpoints/best-loss.ckpt"
test -s "$checkpoint"
"$python_bin" "$repo_root/scripts/data/validate_partonic_contract.py" \
  --experiment "$experiment" \
  --checkpoint "$checkpoint" \
  > "${SLURM_SUBMIT_DIR:-$PWD}/partonic_checkpoint_contract_${SLURM_JOB_ID}.json"
