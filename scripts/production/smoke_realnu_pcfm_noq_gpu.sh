#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "PCFM_noQ GPU smoke must run inside an allocated Slurm job." >&2
  exit 2
fi

source_root=$(git rev-parse --show-toplevel)
project_root=${JER_REALNU_PROJECT_ROOT:?JER_REALNU_PROJECT_ROOT is required}
python_bin=${JER_PYTHON:-/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python}
measured="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/adapted/train_validation/partonic_ttbar2l_train_validation_maxjets8_v1_measured.h5"
truth="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/adapted/train_validation_authoritative_20260926/partonic_ttbar2l_train_validation_maxjets8_v1_supervision.h5"
output="$project_root/results/experiments/partonic_ttbar2L_2M_realnu_v1/production/realnu_top_antitop_20260927_release2/evaluation_v3_diagnostics/parent_noquery/PCFM_noQ_gpu_smoke_${SLURM_JOB_ID}.json"

export JER_PROJECT_ROOT="$project_root"
export PYTHONPATH="$source_root/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$source_root"
"$python_bin" scripts/validation/validate_realnu_pcfm_noq_cpu.py \
  --measured "$measured" --truth "$truth" \
  --output "$output" --device cuda
