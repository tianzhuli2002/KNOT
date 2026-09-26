#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Realnu G0 requires its allocated Slurm GPU job." >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"
expected_sha=${JER_REALNU_EXPECTED_SHA:?JER_REALNU_EXPECTED_SHA is required}
g0_root=${JER_REALNU_G0_ROOT:?JER_REALNU_G0_ROOT is required}
python_bin=${JER_PYTHON:-/mnt/iusers01/fatpou01/phy01/w00238tl/miniforge3/bin/python}

export JER_PROJECT_ROOT="$repo_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

"$python_bin" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("The allocated realnu G0 job cannot access CUDA")
print(f"Realnu G0 CUDA device: {torch.cuda.get_device_name(0)}")
PY

exec "$python_bin" scripts/production/run_realnu_gpu_g0.py
